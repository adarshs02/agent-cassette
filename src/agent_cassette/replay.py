"""Deterministic sequential replay of recorded calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.json_codec import StrictJSONError, copy_json_value
from agent_cassette.matching import (
    DEFAULT_FUZZY_THRESHOLD,
    InputMatcher,
    MatchMode,
    inputs_match,
    normalize_input,
    validate_fuzzy_threshold,
)
from agent_cassette.redaction import redact
from agent_cassette.storage import load_events
from agent_cassette.tools import _ToolSessionMixin


class ReplayMismatchError(AssertionError):
    """Raised when execution no longer matches a cassette.

    Carries optional, code-owned structured fields so machine reports never parse the
    prose message. Every field defaults to ``None``/empty, so a legacy caller constructing
    the error with only a message keeps working. Payload values are never stored here.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        event_index: int | None = None,
        expected: dict[str, str | None] | None = None,
        actual: dict[str, str | None] | None = None,
        changed_paths: tuple[str, ...] = (),
        changed_paths_truncated: bool = False,
        match: str | None = None,
        remaining: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.event_index = event_index
        self.expected = expected
        self.actual = actual
        self.changed_paths = changed_paths
        self.changed_paths_truncated = changed_paths_truncated
        self.match = match
        self.remaining = remaining


class RateLimitError(ConnectionError):
    """Rate-limit-shaped error for deterministic resilience testing.

    Subclasses ``ConnectionError`` so retry logic that handles transient
    connection failures also handles injected rate limits.
    """

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RecordedCallError(RuntimeError):
    """Safe fallback for a replayed exception type that is not allowlisted."""

    def __init__(self, recorded_type: str, message: str) -> None:
        self.recorded_type = recorded_type
        self.recorded_message = message
        super().__init__(f"{recorded_type}: {message}")


class Replayer(_ToolSessionMixin):
    """Return recorded outputs for incoming calls in event order."""

    def __init__(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        match: MatchMode = "exact",
        ignore_paths: tuple[str, ...] = (),
        matcher: InputMatcher | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        validate_fuzzy_threshold(fuzzy_threshold)
        self.path = Path(path)
        self.strict = strict
        self.match = match
        self.ignore_paths = ignore_paths
        self.matcher = matcher
        self.fuzzy_threshold = fuzzy_threshold
        self.events = [event for event in load_events(path) if _is_replayable(event)]
        self.position = 0
        self._consumed: set[int] = set()

    def __enter__(self) -> Replayer:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is None and self.strict and self.remaining:
            first = next(i for i in range(len(self.events)) if i not in self._consumed)
            event = self.events[first]
            raise ReplayMismatchError(
                f"Replay finished with {self.remaining} unconsumed event(s)",
                kind="unconsumed",
                event_index=first + 1,  # one-based first unconsumed cassette index
                expected={"type": _call_type(event).value, "name": _safe_name(event.name)},
                match=self.match,
                remaining=self.remaining,
            )

    async def __aenter__(self) -> Replayer:
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        self.__exit__(exc_type, exc, traceback)

    @property
    def remaining(self) -> int:
        """Number of recorded calls not yet consumed."""
        return len(self.events) - len(self._consumed)

    @property
    def consumed_events(self) -> tuple[Event, ...]:
        """Detached copies of consumed events, in original cassette order.

        Returns copies round-tripped through the code-owned ``Event.to_dict``/
        ``Event.from_dict`` path (never references into ``self.events``), so mutating a
        returned event or its payload cannot alter Replayer state. Ordered by cassette
        index regardless of strict/non-strict or concurrent consumption order.
        """
        return tuple(
            Event.from_dict(self.events[index].to_dict()) for index in sorted(self._consumed)
        )

    def call(
        self,
        event_type: EventType | str,
        name: str,
        input: Any = None,
        function: Callable[[], Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Any], Any] | None = None,
        error_serializer: Callable[[BaseException], Any] | None = None,
    ) -> Any:
        """Match the next call and return its recorded output without calling live code."""
        event = self.consume(event_type, name, input)
        if event.type == EventType.ERROR:
            raise _restore_error(event)
        return event.output

    def consume(self, event_type: EventType | str, name: str, input: Any = None) -> Event:
        """Consume and return one matching event without interpreting its outcome."""
        expected_type = EventType(event_type)
        if not self.remaining:
            raise ReplayMismatchError(
                f"Unexpected {expected_type.value} {_safe_name(name)!r} at "
                f"step {len(self.events) + 1}; the cassette is exhausted",
                kind="exhausted",
                event_index=len(self.events) + 1,  # one-based incoming step past the end
                actual={"type": expected_type.value, "name": _safe_name(name)},
                match=self.match,
                remaining=self.remaining,
            )
        index = self.position if self.strict else self._find_match(expected_type, name, input)
        event = self.events[index]
        divergence = self._mismatch(event, expected_type, name, input)
        if divergence is not None:
            raise ReplayMismatchError(
                _mismatch_message(self.position + 1, divergence),
                kind=divergence["kind"],
                event_index=self.position + 1,  # one-based matched cassette index
                expected=divergence["expected"],
                actual=divergence["actual"],
                changed_paths=divergence["changed_paths"],
                changed_paths_truncated=divergence["changed_paths_truncated"],
                match=self.match,
                remaining=self.remaining,
            )
        self._consumed.add(index)
        self._advance_position()
        return event

    def _advance_position(self) -> None:
        while self.position < len(self.events) and self.position in self._consumed:
            self.position += 1

    def _find_match(self, event_type: EventType, name: str, input: Any) -> int:
        for index, event in enumerate(self.events):
            if index in self._consumed:
                continue
            if self._mismatch(event, event_type, name, input) is None:
                return index
        raise ReplayMismatchError(
            f"No remaining event matches {event_type.value} {_safe_name(name)!r}",
            kind="no-match",
            actual={"type": event_type.value, "name": _safe_name(name)},
            match=self.match,
            remaining=self.remaining,
        )

    async def acall(
        self,
        event_type: EventType | str,
        name: str,
        input: Any = None,
        function: Callable[[], Awaitable[Any]] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Any], Any] | None = None,
        error_serializer: Callable[[BaseException], Any] | None = None,
    ) -> Any:
        """Match an async call and return its recorded output without awaiting live code."""
        return self.call(
            event_type,
            name,
            input,
            metadata=metadata,
            cost=cost,
            serializer=serializer,
        )

    def _mismatch(
        self, event: Event, event_type: EventType, name: str, input: Any
    ) -> dict[str, Any] | None:
        """Return structured, payload-free divergence details, or ``None`` when matched."""
        recorded_type = _call_type(event)
        expected = {"type": recorded_type.value, "name": _safe_name(event.name)}
        actual = {"type": event_type.value, "name": _safe_name(name)}
        if recorded_type != event_type:
            return _divergence("type", expected, actual)
        if event.name != name:
            return _divergence("name", expected, actual)
        expected_input = normalize_input(event.input, self.ignore_paths)
        actual_input = normalize_input(input, self.ignore_paths)
        if not inputs_match(
            expected_input,
            actual_input,
            mode=self.match,
            matcher=self.matcher,
            fuzzy_threshold=self.fuzzy_threshold,
        ):
            paths, truncated = _diff_paths(expected_input, actual_input)
            return _divergence("input", expected, actual, paths, truncated)
        return None


_SAFE_EXCEPTIONS: dict[str, type[Exception]] = {
    "ConnectionError": ConnectionError,
    "RateLimitError": RateLimitError,
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}


_MAX_CHANGED_PATHS = 50


def _safe_name(name: Any) -> str | None:
    """Redact secrets, then strip control characters and bound, for a machine report.

    Runs the exact string through the shared redaction first, so a Bearer token, URI
    userinfo password, or secret query value embedded in a hostile cassette/call name
    cannot survive in an exception message, envelope, report, or next action.
    """
    if type(name) is not str:
        return None
    try:
        redacted = redact(name)
    except Exception:
        # A hostile name (e.g. deeply nested URIs) can make redaction fail closed; never
        # let that abort the mismatch it is describing, and never surface the raw name.
        return "[unrenderable-name]"
    cleaned = "".join(char for char in redacted if char.isprintable())
    return cleaned[:200]


def _divergence(
    kind: str,
    expected: dict[str, str | None],
    actual: dict[str, str | None],
    changed_paths: tuple[str, ...] = (),
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "expected": expected,
        "actual": actual,
        "changed_paths": changed_paths,
        "changed_paths_truncated": truncated,
    }


def _mismatch_message(step: int, divergence: dict[str, Any]) -> str:
    # Payload-safe: names/types are safe to render; input values never are.
    kind = divergence["kind"]
    expected = divergence["expected"]
    actual = divergence["actual"]
    if kind == "type":
        detail = f" (expected {expected['type']!r}, received {actual['type']!r})"
    elif kind == "name":
        detail = f" (expected {expected['name']!r}, received {actual['name']!r})"
    else:
        detail = ""
    return f"Replay diverged at step {step}: {kind} mismatch{detail}"


_UNSAFE_SEGMENT_CHARS = frozenset(".:/@%?#&= \t\"'\\")


def _sanitize_segment(segment: str) -> str:
    # Keys are structural, but a key can itself be payload (an email, token, or a URI used
    # as a map key). Strip path/URI/query punctuation and non-printables and bound length so
    # a sensitive key cannot survive intact in a changed-path segment.
    cleaned = "".join(
        char for char in segment if char.isprintable() and char not in _UNSAFE_SEGMENT_CHARS
    )
    return cleaned[:64] if cleaned else "?"


def _changed_paths(expected: Any, actual: Any) -> tuple[tuple[str, ...], bool]:
    """Dotted paths where two already-redacted/normalized JSON values differ.

    Contains no values, caps at ``_MAX_CHANGED_PATHS``, sanitizes/control-bounds each path
    segment, and reports truncation.
    """
    paths: list[str] = []
    truncated = False

    def add(prefix: list[str]) -> None:
        nonlocal truncated
        if len(paths) >= _MAX_CHANGED_PATHS:
            truncated = True
            return
        paths.append(".".join(prefix) if prefix else ".")

    def walk(exp: Any, act: Any, prefix: list[str]) -> None:
        nonlocal truncated
        if len(paths) >= _MAX_CHANGED_PATHS:
            truncated = True
            return
        if isinstance(exp, dict) and isinstance(act, dict):
            for key in dict.fromkeys([*exp, *act]):
                if len(paths) >= _MAX_CHANGED_PATHS:
                    truncated = True
                    return
                segment = prefix + [_sanitize_segment(str(key))]
                if key not in exp or key not in act:
                    add(segment)
                elif exp[key] != act[key]:
                    walk(exp[key], act[key], segment)
        elif isinstance(exp, list) and isinstance(act, list):
            if len(exp) != len(act):
                add(prefix)
            else:
                for index, (item_exp, item_act) in enumerate(zip(exp, act, strict=True)):
                    if len(paths) >= _MAX_CHANGED_PATHS:
                        truncated = True
                        return
                    if item_exp != item_act:
                        walk(item_exp, item_act, prefix + [str(index)])
        else:
            add(prefix)

    walk(expected, actual, [])
    return tuple(paths), truncated


def _diff_paths(expected: Any, actual: Any) -> tuple[tuple[str, ...], bool]:
    """Detach both normalized inputs through the exact bounded JSON copier before diffing.

    Guarantees the diff walk only ever sees strict JSON with exact ``str`` keys — no
    ``str``/``repr``/iteration/conversion of a hostile key or container. If an input is not
    strict JSON, return a value-free root difference rather than replacing the replay
    mismatch with a serialization exception.
    """
    try:
        safe_expected = copy_json_value(expected)
        safe_actual = copy_json_value(actual)
    except StrictJSONError:
        return (".",), False
    return _changed_paths(safe_expected, safe_actual)


def _call_type(event: Event) -> EventType:
    internal = event.metadata.get("_agent_cassette", {})
    if isinstance(internal, dict) and "call_type" in internal:
        try:
            return EventType(internal["call_type"])
        except ValueError:
            pass
    return event.type


def _is_replayable(event: Event) -> bool:
    internal = event.metadata.get("_agent_cassette", {})
    if isinstance(internal, dict) and internal.get("observational") is True:
        return False
    return event.type != EventType.ERROR or _call_type(event) != EventType.ERROR


def _restore_error(event: Event) -> Exception:
    payload = event.output if isinstance(event.output, dict) else {}
    error_type = str(payload.get("type", "RecordedCallError"))
    message = str(payload.get("message", "recorded call failed"))
    exception_type = _SAFE_EXCEPTIONS.get(error_type)
    return exception_type(message) if exception_type else RecordedCallError(error_type, message)
