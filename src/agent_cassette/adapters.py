"""Explicit adapter protocol and local registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from agent_cassette.events import Event


@runtime_checkable
class Adapter(Protocol):
    """Convert one integration-specific value into cassette events."""

    def to_events(self, value: Any) -> Iterable[Event]: ...


class AdapterRegistry:
    """Caller-owned adapter registry with no import-time discovery or globals."""

    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(self, name: str, adapter: Adapter, *, replace: bool = False) -> None:
        if not name or not name.strip():
            raise ValueError("adapter name cannot be empty")
        if not isinstance(adapter, Adapter):
            raise TypeError("adapter must implement to_events(value)")
        if name in self._adapters and not replace:
            raise ValueError(f"adapter already registered: {name}")
        self._adapters[name] = adapter

    def unregister(self, name: str) -> Adapter:
        try:
            return self._adapters.pop(name)
        except KeyError as error:
            raise KeyError(f"unknown adapter: {name}") from error

    def get(self, name: str) -> Adapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise KeyError(f"unknown adapter: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def adapt(self, name: str, value: Any) -> list[Event]:
        events = list(self.get(name).to_events(value))
        if not all(isinstance(event, Event) for event in events):
            raise TypeError(f"adapter {name!r} returned a non-Event value")
        return events
