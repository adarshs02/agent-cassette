from __future__ import annotations

import json

import pytest

from agent_cassette import Cassette, EventType, ReplayMismatchError


def test_record_and_replay_without_calling_live_function(tmp_path):
    path = tmp_path / "run.jsonl"
    live_calls = 0

    def live_model():
        nonlocal live_calls
        live_calls += 1
        return {"answer": "recorded"}

    with Cassette.record(path) as cassette:
        assert cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "Hi"}, live_model) == {
            "answer": "recorded"
        }

    with Cassette.replay(path) as cassette:
        assert cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "Hi"}) == {
            "answer": "recorded"
        }

    assert live_calls == 1
    event = json.loads(path.read_text().splitlines()[0])
    assert event["schema_version"] == 1
    assert event["duration_ms"] is not None


def test_replay_reports_first_mismatch(tmp_path):
    path = tmp_path / "run.jsonl"
    with Cassette.record(path) as cassette:
        cassette.call(EventType.TOOL_CALL, "search", {"query": "agents"}, lambda: ["result"])

    with pytest.raises(ReplayMismatchError, match="input changed"):
        with Cassette.replay(path) as cassette:
            cassette.call(EventType.TOOL_CALL, "search", {"query": "different"})


def test_replay_requires_all_events_in_strict_mode(tmp_path):
    path = tmp_path / "run.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.CUSTOM, "one", output=1)
        cassette.add(EventType.CUSTOM, "two", output=2)

    with pytest.raises(ReplayMismatchError, match="1 unconsumed"):
        with Cassette.replay(path) as cassette:
            assert cassette.call(EventType.CUSTOM, "one") == 1
