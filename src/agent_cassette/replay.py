"""Deterministic sequential replay of recorded calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.redaction import redact
from agent_cassette.storage import load_events


class ReplayMismatchError(AssertionError):
    """Raised when execution no longer matches a cassette."""


class Replayer:
    """Return recorded outputs for incoming calls in event order."""

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        self.path = Path(path)
        self.strict = strict
        self.events = [event for event in load_events(path) if event.type != EventType.ERROR]
        self.position = 0

    def __enter__(self) -> Replayer:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is None and self.strict and self.remaining:
            raise ReplayMismatchError(
                f"Replay finished with {self.remaining} unconsumed event(s) in {self.path}"
            )

    @property
    def remaining(self) -> int:
        """Number of recorded calls not yet consumed."""
        return len(self.events) - self.position

    def call(self, event_type: EventType | str, name: str, input: Any = None) -> Any:
        """Match the next call and return its recorded output."""
        expected_type = EventType(event_type)
        if self.position >= len(self.events):
            raise ReplayMismatchError(
                f"Unexpected {expected_type.value} '{name}' at step {self.position + 1}; "
                "the cassette is exhausted"
            )

        if self.strict:
            index = self.position
        else:
            index = self._find_match(expected_type, name, input)
        event = self.events[index]
        mismatch = self._mismatch(event, expected_type, name, input)
        if mismatch:
            raise ReplayMismatchError(f"Replay diverged at step {self.position + 1}: {mismatch}")
        self.position = index + 1
        return event.output

    def _find_match(self, event_type: EventType, name: str, input: Any) -> int:
        for index in range(self.position, len(self.events)):
            if not self._mismatch(self.events[index], event_type, name, input):
                return index
        raise ReplayMismatchError(
            f"No remaining event matches {event_type.value} '{name}' with input {input!r}"
        )

    @staticmethod
    def _mismatch(event: Event, event_type: EventType, name: str, input: Any) -> str | None:
        if event.type != event_type:
            return f"expected type {event.type.value!r}, received {event_type.value!r}"
        if event.name != name:
            return f"expected name {event.name!r}, received {name!r}"
        clean_input = redact(input)
        if event.input != clean_input:
            return f"input changed from {event.input!r} to {clean_input!r}"
        return None
