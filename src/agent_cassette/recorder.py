"""Recording support for framework-independent agent events."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar
from uuid import uuid4

from agent_cassette.events import Event, EventType
from agent_cassette.redaction import redact
from agent_cassette.storage import save_events

Result = TypeVar("Result")


class Recorder:
    """Collect and persist events from one agent execution."""

    def __init__(self, path: str | Path, *, redact_secrets: bool = True) -> None:
        self.path = Path(path)
        self.redact_secrets = redact_secrets
        self.events: list[Event] = []

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is not None:
            self.add(EventType.ERROR, "uncaught_exception", output={"message": str(exc)})
        self.save()

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
                metadata=metadata,
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise
        self.add(
            event_type,
            name,
            input=input,
            output=result,
            metadata=metadata,
            duration_ms=(perf_counter() - started) * 1000,
            cost=cost,
        )
        return result

    def save(self) -> None:
        """Persist all currently recorded events."""
        save_events(self.path, self.events)
