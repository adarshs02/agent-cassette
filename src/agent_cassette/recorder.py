"""Recording support for framework-independent agent events."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from agent_cassette.events import Event, EventType
from agent_cassette.redaction import redact
from agent_cassette.storage import append_event, save_events

Result = TypeVar("Result")


class _SpanContext:
    def __init__(self, recorder: Recorder, span_id: str) -> None:
        self._recorder = recorder
        self.span_id = span_id
        self.parent_id: str | None = None
        self._token: Token[tuple[str, ...]] | None = None

    def __enter__(self) -> str:
        spans = self._recorder._spans.get()
        self.parent_id = spans[-1] if spans else None
        self._token = self._recorder._spans.set((*spans, self.span_id))
        return self.span_id

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if self._token is not None:
            self._recorder._spans.reset(self._token)
            self._token = None

    async def __aenter__(self) -> str:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        self.__exit__(exc_type, exc, traceback)


class Recorder:
    """Collect and persist events from one agent execution."""

    def __init__(self, path: str | Path, *, redact_secrets: bool = True) -> None:
        self.path = Path(path)
        self.redact_secrets = redact_secrets
        self.events: list[Event] = []
        self._spans: ContextVar[tuple[str, ...]] = ContextVar(
            f"agent_cassette_spans_{id(self)}", default=()
        )

    def __enter__(self) -> Recorder:
        save_events(self.path, self.events)
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is not None and not self._already_recorded(exc):
            self.add(EventType.ERROR, "uncaught_exception", output={"message": str(exc)})
        self.save()

    async def __aenter__(self) -> Recorder:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        self.__exit__(exc_type, exc, traceback)

    def span(self, span_id: str | None = None) -> _SpanContext:
        """Create a sync or async context that relates events in a nested span."""
        return _SpanContext(self, span_id or str(uuid4()))

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
        """Append an event and return it."""
        clean = redact if self.redact_secrets else lambda value: value
        spans = self._spans.get()
        if span_id is None and spans:
            span_id = spans[-1]
        if parent_id is None and len(spans) > 1:
            parent_id = spans[-2]
        event = Event(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=EventType(event_type),
            name=name,
            input=clean(input),
            output=clean(output),
            metadata=clean(metadata or {}),
            duration_ms=duration_ms,
            cost=cost,
            parent_id=parent_id,
            span_id=span_id,
        )
        append_event(self.path, event)
        self.events.append(event)
        return event

    def call(
        self,
        event_type: EventType | str,
        name: str,
        input: Any,
        function: Callable[[], Result],
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Result], Any] | None = None,
    ) -> Result:
        """Execute a callable and record its result, duration, or error."""
        started = perf_counter()
        try:
            result = function()
        except Exception as error:
            self.add(
                EventType.ERROR,
                name,
                input=input,
                output={"type": type(error).__name__, "message": str(error)},
                metadata=_error_metadata(metadata, event_type),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise
        self.add(
            event_type,
            name,
            input=input,
            output=serializer(result) if serializer else result,
            metadata=metadata,
            duration_ms=(perf_counter() - started) * 1000,
            cost=cost,
        )
        return result

    async def acall(
        self,
        event_type: EventType | str,
        name: str,
        input: Any,
        function: Callable[[], Awaitable[Result]],
        *,
        metadata: dict[str, Any] | None = None,
        cost: float | None = None,
        serializer: Callable[[Result], Any] | None = None,
    ) -> Result:
        """Await a callable and record its result, duration, or error."""
        started = perf_counter()
        try:
            result = await function()
        except Exception as error:
            self.add(
                EventType.ERROR,
                name,
                input=input,
                output={"type": type(error).__name__, "message": str(error)},
                metadata=_error_metadata(metadata, event_type),
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise
        self.add(
            event_type,
            name,
            input=input,
            output=serializer(result) if serializer else result,
            metadata=metadata,
            duration_ms=(perf_counter() - started) * 1000,
            cost=cost,
        )
        return result

    def _already_recorded(self, error: BaseException) -> bool:
        if not self.events or self.events[-1].type != EventType.ERROR:
            return False
        output = self.events[-1].output
        return (
            isinstance(output, dict)
            and output.get("type") == type(error).__name__
            and output.get("message") == str(error)
        )

    def save(self) -> None:
        """Persist all currently recorded events."""
        save_events(self.path, self.events)


def _error_metadata(metadata: dict[str, Any] | None, event_type: EventType | str) -> dict[str, Any]:
    combined = dict(metadata or {})
    internal = dict(combined.get("_agent_cassette", {}))
    internal["call_type"] = EventType(event_type).value
    combined["_agent_cassette"] = internal
    return combined
