from __future__ import annotations

import math
from typing import Any, cast

import pytest

from agent_cassette.events import Event, EventType
from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError


def _event(**changes: Any) -> Event:
    values: dict[str, Any] = {
        "id": "event-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "type": EventType.CUSTOM,
        "name": "step",
    }
    values.update(changes)
    return Event(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": 1}, "id must be a nonempty string"),
        ({"id": ""}, "id must be a nonempty string"),
        ({"timestamp": "2026-01-01T00:00:00"}, "timezone offset"),
        ({"timestamp": "not-a-time"}, "valid ISO 8601"),
        ({"type": "custom"}, "type must be an EventType"),
        ({"name": None}, "name must be a nonempty string"),
        ({"name": ""}, "name must be a nonempty string"),
        ({"metadata": []}, "metadata must be a dictionary"),
        ({"parent_id": 1}, "parent_id must be a string or None"),
        ({"span_id": 1}, "span_id must be a string or None"),
        ({"schema_version": True}, "schema_version must equal"),
        ({"schema_version": 2}, "schema_version must equal"),
    ],
)
def test_event_rejects_invalid_fields(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _event(**changes)


@pytest.mark.parametrize("field_name", ["duration_ms", "cost"])
@pytest.mark.parametrize("value", [True, -1, math.inf, math.nan, "1"])
def test_event_rejects_invalid_numeric_fields(field_name, value):
    with pytest.raises((TypeError, ValueError), match=field_name):
        _event(**{field_name: value})


@pytest.mark.parametrize(
    "payload",
    [
        object(),
        ("tuple",),
        {1: "non-string key"},
        math.inf,
    ],
)
def test_event_rejects_non_json_payloads(payload):
    with pytest.raises(StrictJSONError):
        _event(input=payload)


def test_event_rejects_cyclic_payload():
    payload: list[Any] = []
    payload.append(payload)

    with pytest.raises(StrictJSONError, match="cyclic"):
        _event(output=payload)


def test_event_rejects_excessive_payload_depth():
    payload: object = None
    for _ in range(MAX_JSON_DEPTH + 1):
        payload = [payload]

    with pytest.raises(StrictJSONError, match="maximum cassette JSON depth"):
        _event(output=payload)


def test_to_dict_revalidates_mutable_payload():
    event = _event(output=[])
    event.output.append(event.output)

    with pytest.raises(StrictJSONError, match="cyclic"):
        event.to_dict()


def test_to_dict_detaches_mutable_payloads():
    event = _event(input={"nested": []}, metadata={"tags": []})

    data = event.to_dict()
    data["input"]["nested"].append("changed")
    data["metadata"]["tags"].append("changed")

    assert event.input == {"nested": []}
    assert event.metadata == {"tags": []}


def test_from_dict_requires_object_and_supported_event_type():
    with pytest.raises(TypeError, match="JSON object"):
        Event.from_dict(cast(Any, []))
    with pytest.raises(ValueError, match="type is not supported"):
        Event.from_dict(
            {
                "id": "1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "type": "plugin_named_type",
                "name": "bad",
            }
        )


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        Event.from_dict(
            {
                "id": "1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "type": "custom",
                "name": "bad",
                "unknown": True,
            }
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"secret-unknown-field": "value"},
        {"schema_version": "secret-schema-value"},
    ],
)
def test_from_dict_diagnostics_do_not_echo_attacker_controlled_keys_or_versions(changes):
    data = {
        "id": "1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "type": "custom",
        "name": "step",
        **changes,
    }

    with pytest.raises(ValueError) as caught:
        Event.from_dict(data)
    message = str(caught.value)
    assert "secret-unknown-field" not in message
    assert "secret-schema-value" not in message
