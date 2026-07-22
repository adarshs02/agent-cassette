"""Core tests for the additive `error_serializer` keyword on call/acall."""

from __future__ import annotations

import asyncio

import pytest

from agent_cassette import Cassette, EventType, InjectionRule, Raise
from agent_cassette.storage import load_events


def _raise(exc):
    def function():
        raise exc

    return function


def _mark(error):
    return {"type": "MARK", "message": str(error), "original_type": type(error).__name__}


def test_default_error_output_unchanged(tmp_path):
    path = tmp_path / "c.jsonl"
    with Cassette.record(path) as cassette:
        with pytest.raises(ValueError, match="boom"):
            cassette.call(EventType.TOOL_CALL, "t", {"x": 1}, _raise(ValueError("boom")))
    events = load_events(path)
    assert events[0].type == EventType.ERROR
    assert events[0].output == {"type": "ValueError", "message": "boom"}


def test_custom_error_serializer_and_original_reraised(tmp_path):
    path = tmp_path / "c.jsonl"
    err = ValueError("boom")
    with Cassette.record(path) as cassette:
        with pytest.raises(ValueError) as info:
            cassette.call(EventType.TOOL_CALL, "t", {"x": 1}, _raise(err), error_serializer=_mark)
        assert info.value is err  # original live exception re-raised unchanged
    assert load_events(path)[0].output == {
        "type": "MARK",
        "message": "boom",
        "original_type": "ValueError",
    }


def test_async_custom_error_serializer(tmp_path):
    path = tmp_path / "c.jsonl"

    async def scenario():
        async def boom():
            raise RuntimeError("async boom")

        with Cassette.record(path) as cassette:
            with pytest.raises(RuntimeError, match="async boom"):
                await cassette.acall(
                    EventType.TOOL_CALL, "t", {"x": 1}, boom, error_serializer=_mark
                )

    asyncio.run(scenario())
    assert load_events(path)[0].output["type"] == "MARK"
    assert load_events(path)[0].output["original_type"] == "RuntimeError"


def test_uncaught_error_not_duplicated_via_original_type(tmp_path):
    path = tmp_path / "c.jsonl"
    with pytest.raises(ValueError):
        with Cassette.record(path) as cassette:
            # Error escapes the with-block; __exit__ must not add uncaught_exception
            # because original_type matches the escaping error's real name.
            cassette.call(
                EventType.TOOL_CALL,
                "t",
                {"x": 1},
                _raise(ValueError("boom")),
                error_serializer=_mark,
            )
    errors = [e for e in load_events(path) if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].output == {"type": "MARK", "message": "boom", "original_type": "ValueError"}


def test_replayer_accepts_and_ignores_error_serializer(tmp_path):
    path = tmp_path / "c.jsonl"
    with Cassette.record(path) as cassette:
        cassette.call(EventType.TOOL_CALL, "t", {"x": 1}, lambda: "ok")
    with Cassette.replay(path) as replayer:
        out = replayer.call(
            EventType.TOOL_CALL, "t", {"x": 1}, error_serializer=lambda error: {"bad": 1}
        )
    assert out == "ok"


def test_hybrid_forwards_error_serializer_on_injected_raise(tmp_path):
    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    rule = InjectionRule(Raise(ValueError("boom")), event_type="tool_call", name="t")
    with pytest.raises(ValueError, match="boom"):
        with Cassette.fork(source, output, injections=(rule,)) as hybrid:
            hybrid.call(EventType.TOOL_CALL, "t", {"x": 1}, None, error_serializer=_mark)
    errors = [e for e in load_events(output) if e.type == EventType.ERROR]
    assert errors[-1].output == {"type": "MARK", "message": "boom", "original_type": "ValueError"}


def test_hybrid_forwards_error_serializer_on_live(tmp_path):
    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    with pytest.raises(RuntimeError, match="live boom"):
        with Cassette.fork(source, output) as hybrid:
            hybrid.call(
                EventType.TOOL_CALL,
                "t",
                {"x": 1},
                _raise(RuntimeError("live boom")),
                error_serializer=_mark,
            )
    errors = [e for e in load_events(output) if e.type == EventType.ERROR]
    assert errors[-1].output["type"] == "MARK"
