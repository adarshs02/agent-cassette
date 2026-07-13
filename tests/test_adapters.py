from __future__ import annotations

import pytest

from agent_cassette.adapters import Adapter, AdapterRegistry
from agent_cassette.events import Event, EventType


class TextAdapter:
    def to_events(self, value):
        yield Event(
            id="adapted",
            timestamp="2026-01-01T00:00:00Z",
            type=EventType.CUSTOM,
            name="text",
            input=value,
        )


class BrokenAdapter:
    def to_events(self, value):
        return [value]


def test_registry_is_explicit_local_and_name_sorted():
    first = AdapterRegistry()
    second = AdapterRegistry()
    adapter = TextAdapter()

    first.register("zeta", adapter)
    first.register("alpha", adapter)

    assert isinstance(adapter, Adapter)
    assert first.names() == ("alpha", "zeta")
    assert second.names() == ()
    assert first.adapt("alpha", "hello")[0].input == "hello"


def test_registry_rejects_duplicates_unless_replace_is_explicit():
    registry = AdapterRegistry()
    original = TextAdapter()
    replacement = TextAdapter()
    registry.register("text", original)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("text", replacement)
    registry.register("text", replacement, replace=True)

    assert registry.get("text") is replacement
    assert registry.unregister("text") is replacement
    with pytest.raises(KeyError, match="unknown"):
        registry.get("text")


def test_registry_validates_names_protocol_and_results():
    registry = AdapterRegistry()

    with pytest.raises(ValueError, match="empty"):
        registry.register(" ", TextAdapter())
    with pytest.raises(TypeError, match="to_events"):
        registry.register("invalid", object())
    registry.register("broken", BrokenAdapter())
    with pytest.raises(TypeError, match="non-Event"):
        registry.adapt("broken", "not an event")
