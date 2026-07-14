from __future__ import annotations

import json
import math
from typing import Any

import pytest

from agent_cassette.events import Event, EventType
from agent_cassette.storage import (
    CassetteCorruptionError,
    append_event,
    load_events,
    recover_cassette,
    save_events,
)


def _event(identifier: str = "1", **changes: Any) -> Event:
    values: dict[str, Any] = {
        "id": identifier,
        "timestamp": "2026-01-01T00:00:00Z",
        "type": EventType.CUSTOM,
        "name": "step",
    }
    values.update(changes)
    return Event(**values)


def _line(identifier: str = "1", **changes) -> bytes:
    data = _event(identifier, **changes).to_dict()
    return (json.dumps(data) + "\n").encode()


def test_load_accepts_blank_lines_and_valid_final_line_without_lf(tmp_path):
    path = tmp_path / "cassette.jsonl"
    path.write_bytes(b"\n  \r\n" + _line("1") + _line("2").rstrip(b"\n"))

    assert [event.id for event in load_events(path)] == ["1", "2"]


@pytest.mark.parametrize(
    "invalid",
    [
        b'{"a":1,"a":2}\n',
        b'{"id":"1","value":NaN}\n',
        b'{"id":"1","value":Infinity}\n',
    ],
)
def test_load_rejects_non_strict_json(tmp_path, invalid):
    path = tmp_path / "cassette.jsonl"
    path.write_bytes(invalid)

    with pytest.raises(CassetteCorruptionError) as caught:
        load_events(path)
    assert caught.value.path == path
    assert caught.value.line_number == 1
    assert caught.value.byte_offset == 0


def test_load_reports_precise_invalid_utf8_offset(tmp_path):
    path = tmp_path / "cassette.jsonl"
    prefix = _line()
    path.write_bytes(prefix + b'{"bad":"\xff"}\n')

    with pytest.raises(CassetteCorruptionError) as caught:
        load_events(path)
    assert caught.value.line_number == 2
    assert caught.value.byte_offset == len(prefix) + len(b'{"bad":"')
    assert caught.value.reason == "line is not valid UTF-8"


def test_corruption_diagnostic_escapes_path_control_characters(tmp_path):
    path = tmp_path / "bad\nname.jsonl"
    path.write_bytes(b"{")

    with pytest.raises(CassetteCorruptionError) as caught:
        load_events(path)
    message = str(caught.value)
    assert "bad\\nname.jsonl" in message
    assert "bad\nname.jsonl" not in message


def test_save_serializes_all_events_before_replacing_destination(tmp_path):
    path = tmp_path / "cassette.jsonl"
    path.write_text("original\n")
    valid = _event("1")
    invalid = _event("2", output=[])
    invalid.output.append(math.inf)

    with pytest.raises(ValueError, match="non-finite"):
        save_events(path, [valid, invalid])
    assert path.read_text() == "original\n"


def test_append_serializes_before_opening_destination(tmp_path):
    path = tmp_path / "cassette.jsonl"
    event = _event(output=[])
    event.output.append(object())

    with pytest.raises(ValueError, match="unsupported cassette JSON"):
        append_event(path, event)
    assert not path.exists()


def test_recover_discards_only_malformed_unterminated_tail(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    prefix = b"\n" + _line("1")
    tail = b'{"id":"truncated"'
    source.write_bytes(prefix + tail)

    report = recover_cassette(source, destination)

    assert source.read_bytes() == prefix + tail
    assert [event.id for event in load_events(destination)] == ["1"]
    assert report.source == source
    assert report.destination == destination
    assert report.events == 1
    assert report.discarded_offset == len(prefix)
    assert report.discarded_bytes == len(tail)
    assert report.recovered is True
    assert report.to_dict() == {
        "schema_version": 1,
        "command": "recover",
        "source": str(source),
        "destination": str(destination),
        "status": "recovered",
        "events": 1,
        "discarded": {"offset": len(prefix), "bytes": len(tail)},
    }


@pytest.mark.parametrize(
    "tail",
    [
        b"t",
        b"tr",
        b"tru",
        b"f",
        b"fa",
        b"fal",
        b"fals",
        b"n",
        b"nu",
        b"nul",
        b'{"value":tru',
        b"1e",
        b"1e+",
        b"1e-",
        b"1.",
        b"-",
        b"-1.25e+",
        b'{"value":1e-',
        b'"value\\u',
        b'"value\\u1',
        b'"value\\u12',
        b'"value\\u123',
        b'{"value":"text\\u12',
    ],
)
def test_recover_discards_clear_incomplete_json_token_prefixes(tmp_path, tail):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    prefix = _line()
    source.write_bytes(prefix + tail)

    report = recover_cassette(source, destination)

    assert report.discarded_offset == len(prefix)
    assert report.discarded_bytes == len(tail)
    assert [event.id for event in load_events(destination)] == ["1"]


@pytest.mark.parametrize(
    "tail",
    [
        b"{bad",
        b"tX",
        b"trueX",
        b"undefined",
        b"1.e",
        b"01.",
        b"+1",
        b'"value\\u12X',
        b'"value\\u123X',
        b'{"value":1,tru',
    ],
)
def test_recover_rejects_invalid_token_lookalikes(tmp_path, tail):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    source.write_bytes(_line() + tail)

    with pytest.raises(CassetteCorruptionError):
        recover_cassette(source, destination)
    assert not destination.exists()


def test_recover_normalizes_valid_final_line_without_lf(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"
    source.write_bytes(_line().rstrip(b"\n"))

    report = recover_cassette(source, destination)

    assert report.recovered is False
    assert report.discarded_offset is None
    assert report.discarded_bytes == 0
    assert report.to_dict()["status"] == "unchanged"
    assert destination.read_bytes().endswith(b"\n")
    assert len(load_events(destination)) == 1


def test_recover_never_discards_semantically_invalid_event(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    event = _event().to_dict()
    event["duration_ms"] = -1
    source.write_text(json.dumps(event))

    with pytest.raises(CassetteCorruptionError, match="duration_ms"):
        recover_cassette(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "tail",
    [
        b'{"id":1,"id":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b"\xef\xbb\xbf{}",
    ],
)
def test_recover_never_discards_strict_json_violations(tmp_path, tail):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    source.write_bytes(_line() + tail)

    with pytest.raises(CassetteCorruptionError):
        recover_cassette(source, destination)
    assert not destination.exists()


def test_raw_excessive_nesting_is_reported_as_corruption(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    source.write_bytes(b"[" * 2_000 + b"]" * 2_000)

    with pytest.raises(CassetteCorruptionError, match="maximum cassette JSON depth"):
        load_events(source)
    with pytest.raises(CassetteCorruptionError, match="maximum cassette JSON depth"):
        recover_cassette(source, destination)
    assert not destination.exists()


def test_recover_rejects_malformed_terminated_or_internal_line(tmp_path):
    destination = tmp_path / "recovered.jsonl"
    for name, content in [
        ("terminated", _line() + b"{bad}\n"),
        ("internal", b"{bad}\n" + _line()),
    ]:
        source = tmp_path / f"{name}.jsonl"
        source.write_bytes(content)
        with pytest.raises(CassetteCorruptionError):
            recover_cassette(source, destination)
        assert not destination.exists()


def test_recover_preserves_existing_destination_on_failure(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    source.write_bytes(b"{bad}\n")
    destination.write_text("original\n")

    with pytest.raises(CassetteCorruptionError):
        recover_cassette(source, destination)
    assert destination.read_text() == "original\n"


def test_recover_requires_distinct_destination(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(_line())

    with pytest.raises(ValueError, match="must differ"):
        recover_cassette(source, source)
