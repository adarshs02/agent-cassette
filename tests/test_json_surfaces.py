from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent_cassette.events import Event, EventType
from agent_cassette.interop import otlp
from agent_cassette.interop.otlp import export_otlp, import_otlp
from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError
from agent_cassette.reports import CIReport
from agent_cassette.viewer import render_viewer, write_viewer


class _Hostile:
    def __init__(self) -> None:
        self.str_called = False
        self.repr_called = False

    def __str__(self) -> str:
        self.str_called = True
        return "SECRET-HOSTILE-PAYLOAD"

    def __repr__(self) -> str:
        self.repr_called = True
        return "SECRET-HOSTILE-PAYLOAD"


def _event() -> Event:
    return Event(
        id="event-1",
        timestamp="2026-01-01T00:00:00Z",
        type=EventType.CUSTOM,
        name="safe",
    )


def _deep_value() -> list[object]:
    root: list[object] = []
    current = root
    for _ in range(MAX_JSON_DEPTH + 1):
        child: list[object] = []
        current.append(child)
        current = child
    return root


@pytest.mark.parametrize("value", [math.nan, math.inf, _deep_value()])
def test_report_rejects_unsafe_json_without_stringification(value: object) -> None:
    report = CIReport(metadata={"value": value})

    with pytest.raises(StrictJSONError) as raised:
        report.to_json()

    assert "SECRET-HOSTILE-PAYLOAD" not in str(raised.value)


def test_report_rejects_unsupported_json_without_stringification() -> None:
    hostile = _Hostile()
    report = CIReport(metadata={"value": hostile})

    with pytest.raises(StrictJSONError) as raised:
        report.to_json()

    assert "SECRET-HOSTILE-PAYLOAD" not in str(raised.value)
    assert not hostile.str_called
    assert not hostile.repr_called


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_report_nonfinite_failure_preserves_destination(tmp_path: Path, value: float) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("preserve", encoding="utf-8")
    report = CIReport(metadata={"value": value})

    with pytest.raises(StrictJSONError):
        report.write_json(destination)

    assert destination.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))


def test_report_unsupported_failure_preserves_destination(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("preserve", encoding="utf-8")
    hostile = _Hostile()
    report = CIReport(metadata={"value": hostile})

    with pytest.raises(StrictJSONError):
        report.write_json(destination)

    assert destination.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))
    assert not hostile.str_called
    assert not hostile.repr_called


def test_viewer_rejects_mutated_hostile_event_without_stringification() -> None:
    event = _event()
    hostile = _Hostile()
    event.input = hostile

    with pytest.raises(StrictJSONError) as raised:
        render_viewer([event], redact_secrets=False)

    assert "SECRET-HOSTILE-PAYLOAD" not in str(raised.value)
    assert not hostile.str_called
    assert not hostile.repr_called


def test_viewer_failed_serialization_preserves_destination(tmp_path: Path) -> None:
    destination = tmp_path / "viewer.html"
    destination.write_text("preserve", encoding="utf-8")
    event = _event()
    event.output = math.nan

    with pytest.raises(StrictJSONError):
        write_viewer(destination, [event], redact_secrets=False)

    assert destination.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))


def test_viewer_failed_serialization_creates_no_parent_directory(tmp_path: Path) -> None:
    destination = tmp_path / "missing" / "viewer.html"
    event = _event()
    event.output = math.nan

    with pytest.raises(StrictJSONError):
        write_viewer(destination, [event], redact_secrets=False)

    assert not destination.parent.exists()


@pytest.mark.parametrize(
    "document, message",
    [
        ('{"resourceSpans":[],"resourceSpans":[]}', "duplicate"),
        ('{"resourceSpans":[],"value":NaN}', "non-finite"),
    ],
)
def test_otlp_text_rejects_non_strict_json(document: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        import_otlp(document)


def test_otlp_text_rejects_excessive_depth() -> None:
    nested = "[" * (MAX_JSON_DEPTH + 2) + "]" * (MAX_JSON_DEPTH + 2)
    document = '{"resourceSpans":[],"nested":' + nested + "}"

    with pytest.raises(ValueError, match="depth"):
        import_otlp(document)


def test_otlp_embedded_json_rejects_duplicate_keys() -> None:
    document = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "spanId": "1" * 16,
                                "name": "unsafe",
                                "startTimeUnixNano": "1",
                                "attributes": [
                                    {
                                        "key": "input.value",
                                        "value": {"stringValue": '{"x":1,"x":2}'},
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

    with pytest.raises(ValueError, match="embedded"):
        import_otlp(document)


def test_otlp_mapping_rejects_hostile_value_without_payload_repr() -> None:
    hostile = _Hostile()
    with pytest.raises(StrictJSONError) as raised:
        import_otlp({"resourceSpans": [], "hostile": hostile})

    assert "SECRET-HOSTILE-PAYLOAD" not in str(raised.value)
    assert not hostile.str_called
    assert not hostile.repr_called


def test_otlp_rejects_float_integer_fields() -> None:
    document = export_otlp([_event()])
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["startTimeUnixNano"] = 1.5

    with pytest.raises(ValueError, match="integer"):
        import_otlp(document)


def test_otlp_rejects_non_string_parent_span_id() -> None:
    document = export_otlp([_event()])
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["parentSpanId"] = 123

    with pytest.raises(ValueError, match="parent span identity"):
        import_otlp(document)


def test_otlp_failed_serialization_preserves_destination(tmp_path: Path) -> None:
    destination = tmp_path / "trace.json"
    destination.write_text("preserve", encoding="utf-8")
    event = _event()
    event.metadata["hostile"] = _Hostile()

    with pytest.raises(StrictJSONError):
        export_otlp([event], destination)

    assert destination.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))


def test_otlp_atomic_replace_failure_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "trace.json"
    destination.write_text("preserve", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(otlp.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        export_otlp([_event()], destination)

    assert destination.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))
