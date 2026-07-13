from __future__ import annotations

import pytest

from agent_cassette import Cassette, EventType
from agent_cassette.replay import RecordedCallError


def test_recorded_timeout_replays_without_live_execution(tmp_path):
    path = tmp_path / "timeout.jsonl"
    live_calls = 0

    def fail():
        nonlocal live_calls
        live_calls += 1
        raise TimeoutError("provider timed out")

    with Cassette.record(path) as cassette:
        with pytest.raises(TimeoutError, match="provider timed out"):
            cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "hi"}, fail)

    with Cassette.replay(path) as cassette:
        with pytest.raises(TimeoutError, match="provider timed out"):
            cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "hi"}, fail)

    assert live_calls == 1


def test_unknown_recorded_exception_uses_safe_fallback(tmp_path):
    path = tmp_path / "custom-error.jsonl"

    class ProviderSpecificError(Exception):
        pass

    with Cassette.record(path) as cassette:
        with pytest.raises(ProviderSpecificError):
            cassette.call(
                EventType.TOOL_CALL,
                "search",
                {"query": "agents"},
                lambda: (_ for _ in ()).throw(ProviderSpecificError("unavailable")),
            )

    with Cassette.replay(path) as cassette:
        with pytest.raises(RecordedCallError, match="ProviderSpecificError: unavailable"):
            cassette.call(EventType.TOOL_CALL, "search", {"query": "agents"})


def test_propagated_recorded_error_is_not_duplicated(tmp_path):
    path = tmp_path / "single-error.jsonl"

    with pytest.raises(ValueError, match="bad call"):
        with Cassette.record(path) as cassette:
            cassette.call(
                EventType.TOOL_CALL,
                "tool",
                {},
                lambda: (_ for _ in ()).throw(ValueError("bad call")),
            )

    assert len(path.read_text().splitlines()) == 1


def test_permissive_replay_matches_concurrent_completion_order(tmp_path):
    path = tmp_path / "parallel.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.TOOL_CALL, "first", input={"id": 1}, output="one")
        cassette.add(EventType.TOOL_CALL, "second", input={"id": 2}, output="two")

    with Cassette.replay(path, strict=False) as cassette:
        assert cassette.call(EventType.TOOL_CALL, "second", {"id": 2}) == "two"
        assert cassette.call(EventType.TOOL_CALL, "first", {"id": 1}) == "one"
