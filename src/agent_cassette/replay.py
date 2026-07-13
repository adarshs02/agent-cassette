"""Deterministic sequential replay of recorded calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.matching import InputMatcher, MatchMode, inputs_match, normalize_input
from agent_cassette.storage import load_events


class ReplayMismatchError(AssertionError):
    """Raised when execution no longer matches a cassette."""


class RecordedCallError(RuntimeError):
    """Safe fallback for a replayed exception type that is not allowlisted."""

    def __init__(self, recorded_type: str, message: str) -> None:
        self.recorded_type = recorded_type
        self.recorded_message = message
        super().__init__(f"{recorded_type}: {message}")


class Replayer:
    """Return recorded outputs for incoming calls in event order."""

    def __init__(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        match: MatchMode = "exact",
        ignore_paths: tuple[str, ...] = (),
        matcher: InputMatcher | None = None,
    ) -> None:
        self.path = Path(path)
        self.strict = strict
        self.match = match
        self.ignore_paths = ignore_paths
        self.matcher = matcher
        self.events = [event for event in load_events(path) if _is_replayable(event)]
        self.position = 0
        self._consumed: set[int] = set()

    def __enter__(self) -> Replayer:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is None and self.strict and self.remaining:
            raise ReplayMismatchError(
                f"Replay finished with {self.remaining} unconsumed event(s) in {self.path}"
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
                f"Unexpected {expected_type.value} '{name}' at step {len(self.events) + 1}; "
                "the cassette is exhausted"
            )
        index = self.position if self.strict else self._find_match(expected_type, name, input)
        event = self.events[index]
        mismatch = self._mismatch(event, expected_type, name, input)
        if mismatch:
            raise ReplayMismatchError(f"Replay diverged at step {self.position + 1}: {mismatch}")
        self._consumed.add(index)
        self._advance_position()
        return event

    def _advance_position(self) -> None:
        while self.position < len(self.events) and self.position in self._consumed:
            self.position += 1

    def _find_match(self, event_type: EventType, name: str, input: Any) -> int:
        for index, event in enumerate(self.events):
            if index not in self._consumed and not self._mismatch(event, event_type, name, input):
                return index
        raise ReplayMismatchError(
            f"No remaining event matches {event_type.value} '{name}' with input {input!r}"
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

    def _mismatch(self, event: Event, event_type: EventType, name: str, input: Any) -> str | None:
        recorded_type = _call_type(event)
        if recorded_type != event_type:
            return f"expected type {recorded_type.value!r}, received {event_type.value!r}"
        if event.name != name:
            return f"expected name {event.name!r}, received {name!r}"
        expected_input = normalize_input(event.input, self.ignore_paths)
        actual_input = normalize_input(input, self.ignore_paths)
        if not inputs_match(
            expected_input,
            actual_input,
            mode=self.match,
            matcher=self.matcher,
        ):
            return f"input changed from {expected_input!r} to {actual_input!r}"
        return None


_SAFE_EXCEPTIONS: dict[str, type[Exception]] = {
    "ConnectionError": ConnectionError,
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}


def _call_type(event: Event) -> EventType:
    internal = event.metadata.get("_agent_cassette", {})
    if isinstance(internal, dict) and "call_type" in internal:
        try:
            return EventType(internal["call_type"])
        except ValueError:
            pass
    return event.type


def _is_replayable(event: Event) -> bool:
    return event.type != EventType.ERROR or _call_type(event) != EventType.ERROR


def _restore_error(event: Event) -> Exception:
    payload = event.output if isinstance(event.output, dict) else {}
    error_type = str(payload.get("type", "RecordedCallError"))
    message = str(payload.get("message", "recorded call failed"))
    exception_type = _SAFE_EXCEPTIONS.get(error_type)
    return exception_type(message) if exception_type else RecordedCallError(error_type, message)
