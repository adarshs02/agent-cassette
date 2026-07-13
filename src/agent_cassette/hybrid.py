"""Hybrid cassette replay, live forking, and deterministic fault injection."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from agent_cassette.events import Event, EventType
from agent_cassette.matching import DEFAULT_FUZZY_THRESHOLD, InputMatcher, MatchMode
from agent_cassette.recorder import Recorder
from agent_cassette.replay import RateLimitError, Replayer, ReplayMismatchError

__all__ = [
    "Delay",
    "Hybrid",
    "HybridConfigurationError",
    "HybridLiveCallError",
    "InjectionRule",
    "Raise",
    "RateLimitError",
    "Return",
]

Result = TypeVar("Result")
MismatchPolicy = Literal["raise", "live"]
LINEAGE_METADATA_KEY = "_agent_cassette"


class HybridConfigurationError(ValueError):
    """Raised when a hybrid session cannot be configured safely."""


class HybridLiveCallError(RuntimeError):
    """Raised when hybrid execution needs a live callable but none was supplied."""


@dataclass(frozen=True, slots=True)
class Return:
    """Return a deterministic value instead of executing a selected call."""

    value: Any


@dataclass(frozen=True, slots=True)
class Raise:
    """Raise a deterministic, recordable exception for a selected call."""

    error: Exception

    def __post_init__(self) -> None:
        if not isinstance(self.error, Exception):
            raise HybridConfigurationError("Raise requires an Exception instance")


@dataclass(frozen=True, slots=True)
class Delay:
    """Sleep for a deterministic duration, then apply ``then`` or continue live."""

    seconds: float
    then: Return | Raise | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or self.seconds < 0
        ):
            raise HybridConfigurationError("Delay seconds must be a non-negative number")
        if self.then is not None and not isinstance(self.then, (Return, Raise)):
            raise HybridConfigurationError("Delay then-action must be Return, Raise, or None")


InjectionAction = Return | Raise | Delay


@dataclass(frozen=True, slots=True, init=False)
class InjectionRule:
    """Select one call occurrence and apply a deterministic action."""

    action: InjectionAction
    event_type: EventType | None
    name: str | None
    occurrence: int

    def __init__(
        self,
        action: InjectionAction,
        *,
        event_type: EventType | str | None = None,
        type: EventType | str | None = None,
        name: str | None = None,
        occurrence: int = 1,
    ) -> None:
        if not isinstance(action, (Return, Raise, Delay)):
            raise HybridConfigurationError("InjectionRule action must be Return, Raise, or Delay")
        if event_type is not None and type is not None:
            raise HybridConfigurationError("Use either event_type or type, not both")
        selected_type = event_type if event_type is not None else type
        try:
            normalized_type = EventType(selected_type) if selected_type is not None else None
        except ValueError as error:
            raise HybridConfigurationError(
                f"Unknown injection event type: {selected_type!r}"
            ) from error
        if name is not None and not isinstance(name, str):
            raise HybridConfigurationError("InjectionRule name must be a string or None")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
            raise HybridConfigurationError("InjectionRule occurrence must be a positive integer")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "event_type", normalized_type)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "occurrence", occurrence)


class Hybrid:
    """Replay a source prefix, then permanently record live execution to a new cassette."""

    def __init__(
        self,
        source: str | Path,
        output: str | Path,
        *,
        prefix: int | None = None,
        mismatch: MismatchPolicy = "raise",
        injections: Sequence[InjectionRule] = (),
        match: MatchMode = "exact",
        ignore_paths: tuple[str, ...] = (),
        matcher: InputMatcher | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        redact_secrets: bool = True,
    ) -> None:
        self.source = Path(source)
        self.output = Path(output)
        same_path = self.source.resolve() == self.output.resolve()
        same_file = (
            self.source.exists() and self.output.exists() and self.source.samefile(self.output)
        )
        if same_path or same_file:
            raise HybridConfigurationError("Hybrid source and output cassette paths must differ")
        if isinstance(prefix, bool) or (
            prefix is not None and (not isinstance(prefix, int) or prefix < 0)
        ):
            raise HybridConfigurationError("Hybrid prefix must be a non-negative integer or None")
        if mismatch not in ("raise", "live"):
            raise HybridConfigurationError("Hybrid mismatch policy must be 'raise' or 'live'")
        if not all(isinstance(rule, InjectionRule) for rule in injections):
            raise HybridConfigurationError(
                "Hybrid injections must contain only InjectionRule values"
            )

        self.prefix = prefix
        self.mismatch = mismatch
        self.injections = tuple(injections)
        self.replayer = Replayer(
            self.source,
            strict=True,
            match=match,
            ignore_paths=ignore_paths,
            matcher=matcher,
            fuzzy_threshold=fuzzy_threshold,
        )
        self.recorder = Recorder(self.output, redact_secrets=redact_secrets)
        self._source_label = self.source.name
        self._replayed = 0
        self._live = prefix == 0
        self._rule_counts = [0] * len(self.injections)
        self._fired_rules: set[int] = set()

    def __enter__(self) -> Hybrid:
        self.recorder.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        self.save()

    async def __aenter__(self) -> Hybrid:
        await self.recorder.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        self.save()

    @property
    def is_live(self) -> bool:
        """Whether this session has permanently switched to live execution."""
        return self._live

    def add(
        self,
        event_type: EventType | str,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        cost: float | None = None,
        parent_id: str | None = None,
        span_id: str | None = None,
    ) -> Event:
        """Record a delayed live event such as a completed stream."""
        return self.recorder.add(
            event_type,
            name,
            input=input,
            output=output,
            metadata=self._metadata(metadata, mode="live"),
            duration_ms=duration_ms,
            cost=cost,
            parent_id=parent_id,
            span_id=span_id,
        )

    def prepare_stream(
        self,
        event_type: EventType | str,
        name: str,
        input: Any,
        *,
        metadata: dict[str, Any] | None = None,
        serializer: Callable[[Any], Any] | None = None,
        error_serializer: Callable[[Exception], Any] | None = None,
    ) -> tuple[bool, Any]:
        """Replay or inject a stream payload, or signal that a live stream is required."""
        normalized_type = self._event_type(event_type)
        injection = self._select_injection(normalized_type, name)
        if injection is not None:
            rule_index, action = injection
            self._live = True
            delay_seconds = None
            if isinstance(action, Delay):
                delay_seconds = action.seconds
                time.sleep(delay_seconds)
                action = action.then
            injection_metadata = self._metadata(
                metadata,
                mode="injected",
                **self._injection_details(rule_index, action, delay_seconds),
            )
            if action is None:
                return False, None
            if isinstance(action, Return):
                stored_output = serializer(action.value) if serializer else action.value
                self.recorder.add(
                    normalized_type,
                    name,
                    input=input,
                    output=stored_output,
                    metadata=injection_metadata,
                    duration_ms=0.0,
                )
                return True, stored_output
            error = action.error
            if error_serializer is not None:
                serialized_error = error_serializer(error)
                error_metadata = dict(injection_metadata)
                internal = dict(error_metadata.get(LINEAGE_METADATA_KEY, {}))
                internal["call_type"] = normalized_type.value
                error_metadata[LINEAGE_METADATA_KEY] = internal
                self.recorder.add(
                    EventType.ERROR,
                    name,
                    input=input,
                    output=serialized_error,
                    metadata=error_metadata,
                    duration_ms=0.0,
                )
                raise error
            self.recorder.call(
                normalized_type,
                name,
                input,
                lambda: _raise(error),
                metadata=injection_metadata,
            )
            raise AssertionError("injected failure unexpectedly returned")

        if self._live or (self.prefix is not None and self._replayed >= self.prefix):
            self._live = True
            return False, None
        if self.replayer.remaining == 0:
            self._live = True
            return False, None
        try:
            source_event = self.replayer.consume(normalized_type, name, input)
        except ReplayMismatchError:
            if self.mismatch == "raise":
                raise
            self._live = True
            return False, None
        self._record_replayed(source_event)
        self._replayed += 1
        if self.prefix is not None and self._replayed >= self.prefix:
            self._live = True
        return True, source_event.output

    def call(
        self,
        event_type: EventType | str,
        name: str,
        input: Any = None,
        function: Callable[[], Result] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Result], Any] | None = None,
    ) -> Result:
        """Replay, inject, or execute one synchronous call and record the outcome."""
        normalized_type = self._event_type(event_type)
        injection = self._select_injection(normalized_type, name)
        if injection is not None:
            rule_index, action = injection
            self._live = True
            delay_seconds = None
            if isinstance(action, Delay):
                delay_seconds = action.seconds
                time.sleep(delay_seconds)
                action = action.then
            injection_metadata = self._metadata(
                metadata,
                mode="injected",
                **self._injection_details(rule_index, action, delay_seconds),
            )
            if action is None:
                effect = self._require_live_function(normalized_type, name, function)
            elif isinstance(action, Return):
                value = action.value

                def effect() -> Result:
                    return value
            else:
                error = action.error

                def effect() -> Result:
                    return _raise(error)

            return self.recorder.call(
                normalized_type,
                name,
                input,
                effect,
                metadata=injection_metadata,
                cost=cost,
                serializer=serializer,
            )

        replayed = self._try_replay(normalized_type, name, input)
        if replayed is not _LIVE:
            return replayed
        live_function = self._require_live_function(normalized_type, name, function)
        return self.recorder.call(
            normalized_type,
            name,
            input,
            live_function,
            metadata=self._metadata(metadata, mode="live"),
            cost=cost,
            serializer=serializer,
        )

    async def acall(
        self,
        event_type: EventType | str,
        name: str,
        input: Any = None,
        function: Callable[[], Awaitable[Result]] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Result], Any] | None = None,
    ) -> Result:
        """Replay, inject, or execute one asynchronous call and record the outcome."""
        normalized_type = self._event_type(event_type)
        injection = self._select_injection(normalized_type, name)
        if injection is not None:
            rule_index, action = injection
            self._live = True
            delay_seconds = action.seconds if isinstance(action, Delay) else None
            effective = action.then if isinstance(action, Delay) else action
            injection_metadata = self._metadata(
                metadata,
                mode="injected",
                **self._injection_details(rule_index, effective, delay_seconds),
            )
            live_function = (
                self._require_live_function(normalized_type, name, function)
                if effective is None
                else None
            )

            async def inject() -> Result:
                if delay_seconds:
                    await asyncio.sleep(delay_seconds)
                if live_function is not None:
                    return await live_function()
                if isinstance(effective, Raise):
                    raise effective.error
                assert isinstance(effective, Return)
                return effective.value

            return await self.recorder.acall(
                normalized_type,
                name,
                input,
                inject,
                metadata=injection_metadata,
                cost=cost,
                serializer=serializer,
            )

        replayed = self._try_replay(normalized_type, name, input)
        if replayed is not _LIVE:
            return replayed
        live_function = self._require_live_function(normalized_type, name, function)
        return await self.recorder.acall(
            normalized_type,
            name,
            input,
            live_function,
            metadata=self._metadata(metadata, mode="live"),
            cost=cost,
            serializer=serializer,
        )

    def save(self) -> None:
        """Persist the replayed prefix and recorded suffix to the output cassette."""
        self.recorder.save()

    def _try_replay(self, event_type: EventType, name: str, input: Any) -> Any:
        if self._live:
            return _LIVE
        if self.prefix is not None and self._replayed >= self.prefix:
            self._live = True
            return _LIVE
        if self.replayer.remaining == 0:
            self._live = True
            return _LIVE

        event_index = self.replayer.position
        try:
            result = self.replayer.call(event_type, name, input)
        except ReplayMismatchError:
            if self.mismatch == "raise":
                raise
            self._live = True
            return _LIVE

        source_event = self.replayer.events[event_index]
        self._record_replayed(source_event)
        self._replayed += 1
        if self.prefix is not None and self._replayed >= self.prefix:
            self._live = True
        return result

    def _record_replayed(self, event: Event) -> None:
        self.recorder.add(
            event.type,
            event.name,
            input=event.input,
            output=event.output,
            metadata=self._metadata(
                event.metadata,
                mode="replayed",
                source_event_id=event.id,
            ),
            duration_ms=event.duration_ms,
            cost=event.cost,
            parent_id=event.parent_id,
            span_id=event.span_id,
        )

    def _select_injection(
        self, event_type: EventType, name: str
    ) -> tuple[int, InjectionAction] | None:
        matching_indexes: list[int] = []
        for index, rule in enumerate(self.injections):
            if (rule.event_type is None or rule.event_type == event_type) and (
                rule.name is None or rule.name == name
            ):
                self._rule_counts[index] += 1
                matching_indexes.append(index)
        for index in matching_indexes:
            rule = self.injections[index]
            if index not in self._fired_rules and self._rule_counts[index] == rule.occurrence:
                self._fired_rules.add(index)
                return index, rule.action
        return None

    def _metadata(
        self, metadata: dict[str, Any] | None, *, mode: str, **details: Any
    ) -> dict[str, Any]:
        combined = dict(metadata or {})
        existing = combined.get(LINEAGE_METADATA_KEY, {})
        lineage = dict(existing) if isinstance(existing, dict) else {}
        lineage.update({"mode": mode, "source": self._source_label, **details})
        combined[LINEAGE_METADATA_KEY] = lineage
        return combined

    @staticmethod
    def _injection_details(
        rule_index: int, action: Return | Raise | None, delay_seconds: float | None
    ) -> dict[str, Any]:
        if action is None:
            label = "delay"
        elif isinstance(action, Return):
            label = "return"
        else:
            label = "raise"
        details: dict[str, Any] = {"rule": rule_index + 1, "action": label}
        if delay_seconds is not None:
            details["delay_seconds"] = delay_seconds
        return details

    @staticmethod
    def _event_type(event_type: EventType | str) -> EventType:
        try:
            return EventType(event_type)
        except ValueError as error:
            raise HybridConfigurationError(f"Unknown event type: {event_type!r}") from error

    @staticmethod
    def _require_live_function(
        event_type: EventType,
        name: str,
        function: Callable[[], Result] | None,
    ) -> Callable[[], Result]:
        if function is None:
            raise HybridLiveCallError(
                f"Hybrid execution switched to live at {event_type.value} {name!r}, "
                "but no live function was supplied"
            )
        if not callable(function):
            raise HybridLiveCallError(
                f"Live function for {event_type.value} {name!r} must be callable"
            )
        return function


class _LiveSentinel:
    pass


_LIVE = _LiveSentinel()


def _raise(error: Exception) -> Any:
    raise error
