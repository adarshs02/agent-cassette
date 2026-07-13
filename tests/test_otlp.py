from __future__ import annotations

import json

import pytest

from agent_cassette.events import Event, EventType
from agent_cassette.interop.otlp import export_otlp, import_otlp


def sample_events() -> list[Event]:
    return [
        Event(
            id="root",
            timestamp="2026-01-01T00:00:00.000000Z",
            type=EventType.MODEL_CALL,
            name="plan",
            input={"prompt": "hello"},
            output={"answer": "use tool"},
            metadata={"model": "example"},
            duration_ms=12.5,
            cost=0.02,
        ),
        Event(
            id="child",
            timestamp="2026-01-01T00:00:00.020000Z",
            type=EventType.TOOL_CALL,
            name="search",
            input={"q": "hello"},
            output=["result"],
            metadata={"cached": False},
            duration_ms=3,
            parent_id="root",
        ),
    ]


def test_otlp_export_is_deterministic_and_maps_openinference_kinds():
    first = export_otlp(sample_events())
    second = export_otlp(sample_events())
    spans = first["resourceSpans"][0]["scopeSpans"][0]["spans"]

    assert first == second
    assert len(spans[0]["traceId"]) == 32
    assert len(spans[0]["spanId"]) == 16
    assert spans[1]["parentSpanId"] == spans[0]["spanId"]
    attributes = {attribute["key"]: attribute["value"] for attribute in spans[0]["attributes"]}
    assert attributes["openinference.span.kind"]["stringValue"] == "LLM"


def test_otlp_round_trip_preserves_event_data_and_relationships(tmp_path):
    destination = tmp_path / "trace.json"
    exported = export_otlp(sample_events(), destination)

    imported = import_otlp(destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == exported
    assert [event.id for event in imported] == ["root", "child"]
    assert imported[0].type == EventType.MODEL_CALL
    assert imported[0].input == {"prompt": "hello"}
    assert imported[0].output == {"answer": "use tool"}
    assert imported[0].metadata == {"model": "example"}
    assert imported[0].cost == 0.02
    assert imported[0].duration_ms == 12.5
    assert imported[1].type == EventType.TOOL_CALL
    assert imported[1].parent_id == "root"


def test_import_accepts_json_text_and_maps_external_openinference_span():
    document = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "0" * 32,
                                "spanId": "1" * 16,
                                "name": "external",
                                "startTimeUnixNano": "1767225600000000000",
                                "endTimeUnixNano": "1767225600001000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "RETRIEVER"},
                                    },
                                    {
                                        "key": "input.value",
                                        "value": {"stringValue": '{"query":"x"}'},
                                    },
                                    {
                                        "key": "input.mime_type",
                                        "value": {"stringValue": "application/json"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    imported = import_otlp(json.dumps(document))

    assert imported[0].id == f"otlp-{'1' * 16}"
    assert imported[0].type == EventType.CUSTOM
    assert imported[0].input == {"query": "x"}


def test_otlp_uses_unique_ids_and_preserves_recorder_span_relationships():
    events = sample_events()
    events[0].span_id = "shared-span"
    events[1].span_id = "child-span"
    events[1].parent_id = "shared-span"
    sibling = Event(
        id="sibling",
        timestamp="2026-01-01T00:00:00.030000Z",
        type=EventType.CUSTOM,
        name="same-span",
        span_id="shared-span",
    )
    events.insert(1, sibling)

    exported = export_otlp(events)
    spans = exported["resourceSpans"][0]["scopeSpans"][0]["spans"]
    imported = import_otlp(exported)

    assert len({span["spanId"] for span in spans}) == 3
    assert spans[2]["parentSpanId"] == spans[0]["spanId"]
    assert [event.span_id for event in imported] == ["shared-span", "shared-span", "child-span"]
    assert imported[2].parent_id == "shared-span"


def test_import_strict_rejects_and_permissive_skips_malformed_spans():
    document = {"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "missing identity"}]}]}]}

    with pytest.raises(ValueError, match="identity"):
        import_otlp(document)
    assert import_otlp(document, strict=False) == []


def test_import_unknown_kind_is_strict_or_custom_when_permissive():
    document = export_otlp(sample_events()[:1])
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["attributes"] = [
        {
            "key": "openinference.span.kind",
            "value": {"stringValue": "FUTURE_KIND"},
        }
    ]

    with pytest.raises(ValueError, match="unsupported"):
        import_otlp(document)
    assert import_otlp(document, strict=False)[0].type == EventType.CUSTOM
