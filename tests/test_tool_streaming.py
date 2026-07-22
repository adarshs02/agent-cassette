"""Tests for Phase B — generator and async-generator tool record/replay."""

from __future__ import annotations

import asyncio
import inspect
from enum import IntEnum
from functools import partial
from time import perf_counter
from typing import Any

import pytest

from agent_cassette import (
    Cassette,
    Delay,
    EventType,
    InjectionRule,
    Raise,
    RateLimitError,
    RecordedCallError,
    Return,
    wrap_tool,
)
from agent_cassette.json_codec import StrictJSONError
from agent_cassette.storage import load_events
from agent_cassette.tools import _restore_tool_stream

_INPUT = {"args": ["q"], "kwargs": {}}


class _Hostile:
    """A value whose rendering methods must never be invoked during serialization."""

    def __str__(self) -> str:
        raise AssertionError("__str__ called")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ called")

    def model_dump(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("model_dump called")


class _Grade(IntEnum):
    A = 1


class _StrSub(str):
    pass


class _IntSub(int):
    pass


class _FloatSub(float):
    pass


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


class _HostileList(list):
    """A list subclass whose iteration must never run (exact-type reject first)."""

    def __iter__(self):
        raise AssertionError("__iter__ called")


class _HostileDict(dict):
    """A dict subclass whose items() must never run (exact-type reject first)."""

    def items(self):
        raise AssertionError("items called")


class _CustomError(Exception):
    """A tool exception type that is not on the replay allowlist."""


def _success_envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "__agent_cassette_tool_stream__": True,
        "version": 1,
        "items": [],
        "completion": "exhausted",
        "return": None,
    }
    envelope.update(overrides)
    return envelope


def _error_envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "__agent_cassette_tool_stream__": True,
        "version": 1,
        "items": [],
        "completion": "error",
        "phase": "iteration",
        "error": {"type": "ValueError", "message": "boom"},
    }
    envelope.update(overrides)
    return envelope


def _collect_sync(iterator: Any) -> tuple[list[Any], Any]:
    """Drain a sync iterator, returning its items and restored StopIteration value."""
    items: list[Any] = []
    while True:
        try:
            items.append(next(iterator))
        except StopIteration as stop:
            return items, stop.value


def _seed_stream_event(
    path: Any, output: Any, *, name: str = "stream", is_error: bool = False
) -> None:
    """Write one raw tool-stream event so replay can be exercised against it."""
    with Cassette.record(path) as cassette:
        if is_error:
            cassette.add(
                EventType.ERROR,
                name,
                input=_INPUT,
                output=output,
                metadata={"streaming": True, "_agent_cassette": {"call_type": "tool_call"}},
            )
        else:
            cassette.add(
                EventType.TOOL_CALL,
                name,
                input=_INPUT,
                output=output,
                metadata={"streaming": True},
            )


# --------------------------------------------------------------------------- #
# Core record/replay
# --------------------------------------------------------------------------- #


def test_sync_stream_records_and_replays_multiple_items(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield query
        yield query.upper()

    with Cassette.record(path) as cassette:
        assert list(wrap_tool(stream, cassette)("hi")) == ["hi", "HI"]

    with Cassette.replay(path) as replayer:
        assert list(wrap_tool(stream, replayer)("hi")) == ["hi", "HI"]
        assert replayer.remaining == 0


def test_async_stream_records_and_replays_multiple_items(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield query
        yield query.upper()

    async def scenario():
        async with Cassette.record(path) as cassette:
            assert [item async for item in wrap_tool(stream, cassette)("hi")] == ["hi", "HI"]
        async with Cassette.replay(path) as replayer:
            out = [item async for item in wrap_tool(stream, replayer)("hi")]
            assert replayer.remaining == 0
            return out

    assert asyncio.run(scenario()) == ["hi", "HI"]


def test_sync_empty_stream_round_trips(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        return
        yield  # pragma: no cover -- makes stream a generator function

    with Cassette.record(path) as cassette:
        assert list(wrap_tool(stream, cassette)("q")) == []

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.TOOL_CALL
    assert events[0].output["items"] == []
    assert events[0].output["completion"] == "exhausted"

    with Cassette.replay(path) as replayer:
        assert list(wrap_tool(stream, replayer)("q")) == []


def test_async_empty_stream_round_trips(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        return
        yield  # pragma: no cover -- makes stream an async generator function

    async def scenario():
        async with Cassette.record(path) as cassette:
            assert [item async for item in wrap_tool(stream, cassette)("q")] == []
        async with Cassette.replay(path) as replayer:
            return [item async for item in wrap_tool(stream, replayer)("q")]

    assert asyncio.run(scenario()) == []


def test_success_envelope_shape(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield 1

    with Cassette.record(path) as cassette:
        list(wrap_tool(stream, cassette)("q"))

    output = load_events(path)[0].output
    assert set(output) == {
        "__agent_cassette_tool_stream__",
        "version",
        "items",
        "completion",
        "return",
    }
    assert output["__agent_cassette_tool_stream__"] is True
    assert output["version"] == 1
    assert output["items"] == [1]
    assert output["completion"] == "exhausted"
    assert output["return"] is None


def test_sync_return_value_is_recorded_and_restored(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        return "DONE"

    with Cassette.record(path) as cassette:
        items, value = _collect_sync(wrap_tool(stream, cassette)("q"))
    assert items == ["a"]
    assert value == "DONE"
    assert load_events(path)[0].output["return"] == "DONE"

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        items, value = _collect_sync(wrap_tool(forbidden, replayer, name="stream")("q"))
    assert items == ["a"]
    assert value == "DONE"


# --------------------------------------------------------------------------- #
# Zero-live replay
# --------------------------------------------------------------------------- #


def test_sync_replay_never_constructs_or_runs_generator(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield query

    with Cassette.record(path) as cassette:
        list(wrap_tool(stream, cassette)("q"))

    constructed: list[bool] = []

    def forbidden(query):
        constructed.append(True)
        raise AssertionError("generator body ran during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        assert list(wrap_tool(forbidden, replayer, name="stream")("q")) == ["q"]

    assert constructed == []


def test_async_replay_never_constructs_or_runs_generator(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield query

    async def record():
        async with Cassette.record(path) as cassette:
            assert [item async for item in wrap_tool(stream, cassette)("q")] == ["q"]

    asyncio.run(record())

    constructed: list[bool] = []

    async def forbidden(query):
        constructed.append(True)
        raise AssertionError("generator body ran during replay")
        yield  # pragma: no cover

    async def replay():
        async with Cassette.replay(path) as replayer:
            return [item async for item in wrap_tool(forbidden, replayer, name="stream")("q")]

    assert asyncio.run(replay()) == ["q"]
    assert constructed == []


# --------------------------------------------------------------------------- #
# Early close
# --------------------------------------------------------------------------- #


def test_sync_close_records_prefix_runs_finally_and_is_idempotent(tmp_path):
    path = tmp_path / "c.jsonl"
    finalized: list[bool] = []

    def stream(query):
        try:
            yield "a"
            yield "b"  # pragma: no cover -- never reached; closed after first item
        finally:
            finalized.append(True)

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        proxy.close()
        proxy.close()  # idempotent

    assert finalized == [True]
    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.TOOL_CALL
    assert events[0].output["completion"] == "closed"
    assert events[0].output["items"] == ["a"]

    with Cassette.replay(path) as replayer:
        assert list(wrap_tool(stream, replayer)("q")) == ["a"]


def test_async_aclose_records_prefix_runs_finally_and_is_idempotent(tmp_path):
    path = tmp_path / "c.jsonl"
    finalized: list[bool] = []

    async def stream(query):
        try:
            yield "a"
            yield "b"  # pragma: no cover
        finally:
            finalized.append(True)

    async def record():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.__anext__() == "a"
            await proxy.aclose()
            await proxy.aclose()  # idempotent

    asyncio.run(record())

    assert finalized == [True]
    events = load_events(path)
    assert len(events) == 1
    assert events[0].output["completion"] == "closed"
    assert events[0].output["items"] == ["a"]

    async def replay():
        async with Cassette.replay(path) as replayer:
            return [item async for item in wrap_tool(stream, replayer)("q")]

    assert asyncio.run(replay()) == ["a"]


# --------------------------------------------------------------------------- #
# Abandoned streams persist nothing
# --------------------------------------------------------------------------- #


def test_sync_unstarted_stream_persists_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield query  # pragma: no cover -- never iterated

    with Cassette.record(path) as cassette:
        wrap_tool(stream, cassette)("q")

    assert load_events(path) == []


def test_sync_started_but_unclosed_stream_persists_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        yield "b"

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"  # started, never exhausted or closed

    assert load_events(path) == []


def test_async_started_but_unclosed_stream_persists_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield "a"
        yield "b"

    async def scenario():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.__anext__() == "a"

    asyncio.run(scenario())
    assert load_events(path) == []


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_sync_start_error_replays(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        raise ValueError("early")
        yield  # pragma: no cover

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        with pytest.raises(ValueError, match="early"):
            next(proxy)

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].output["phase"] == "start"
    assert events[0].output["items"] == []
    assert events[0].metadata["_agent_cassette"]["call_type"] == "tool_call"

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(forbidden, replayer, name="stream")("q")
        with pytest.raises(ValueError, match="early"):
            next(proxy)


def test_sync_mid_stream_allowlisted_error_replays_after_prefix(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        raise TimeoutError("boom")

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(TimeoutError, match="boom"):
            next(proxy)

    assert load_events(path)[0].output["phase"] == "iteration"

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(forbidden, replayer, name="stream")("q")
        assert next(proxy) == "a"
        with pytest.raises(TimeoutError, match="boom"):
            next(proxy)


def test_async_mid_stream_allowlisted_error_replays_after_prefix(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield "a"
        raise ConnectionError("dropped")

    async def record():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(ConnectionError, match="dropped"):
                await proxy.__anext__()

    asyncio.run(record())

    async def replay():
        async with Cassette.replay(path) as replayer:
            proxy = wrap_tool(stream, replayer)("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(ConnectionError, match="dropped"):
                await proxy.__anext__()

    asyncio.run(replay())


def test_ratelimit_error_replays_same_type(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        raise RateLimitError("slow down")

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(RateLimitError):
            next(proxy)

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(stream, replayer)("q")
        assert next(proxy) == "a"
        with pytest.raises(RateLimitError):
            next(proxy)


def test_unknown_exception_replays_as_recorded_call_error_after_prefix(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        raise _CustomError("weird")

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(_CustomError, match="weird"):
            next(proxy)

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(forbidden, replayer, name="stream")("q")
        assert next(proxy) == "a"
        with pytest.raises(RecordedCallError):
            next(proxy)


def test_close_error_records_close_phase_and_reraises(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        try:
            yield "a"
        except GeneratorExit:
            raise RuntimeError("cleanup failed") from None

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(RuntimeError, match="cleanup failed"):
            proxy.close()

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].output["phase"] == "close"
    assert events[0].output["items"] == ["a"]

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(forbidden, replayer, name="stream")("q")
        assert next(proxy) == "a"
        with pytest.raises(RuntimeError, match="cleanup failed"):
            next(proxy)


# --------------------------------------------------------------------------- #
# Async cancellation (the one Phase B extension to the cancellation limit)
# --------------------------------------------------------------------------- #


def test_async_cancellation_records_partial_and_replays_cancelled(tmp_path):
    path = tmp_path / "c.jsonl"
    finalized: list[bool] = []

    async def stream(query):
        try:
            yield "a"
            raise asyncio.CancelledError()
        finally:
            finalized.append(True)

    async def record():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(asyncio.CancelledError):
                await proxy.__anext__()

    asyncio.run(record())

    assert finalized == [True]
    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].output["completion"] == "error"
    assert events[0].output["phase"] == "cancellation"
    assert events[0].output["items"] == ["a"]

    async def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    async def replay():
        async with Cassette.replay(path) as replayer:
            proxy = wrap_tool(forbidden, replayer, name="stream")("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(asyncio.CancelledError):
                await proxy.__anext__()

    asyncio.run(replay())


# --------------------------------------------------------------------------- #
# Strict serialization (input, items, return)
# --------------------------------------------------------------------------- #


def test_invalid_input_fails_before_generator_construction(tmp_path):
    path = tmp_path / "c.jsonl"
    constructed: list[bool] = []

    def stream(bad):
        constructed.append(True)  # pragma: no cover
        yield 1  # pragma: no cover

    with Cassette.record(path) as cassette:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, cassette)(object())

    assert constructed == []
    assert load_events(path) == []


def test_async_invalid_input_fails_before_generator_construction(tmp_path):
    path = tmp_path / "c.jsonl"
    constructed: list[bool] = []

    async def stream(bad):
        constructed.append(True)  # pragma: no cover
        yield 1  # pragma: no cover

    with Cassette.record(path) as cassette:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, cassette)(object())

    assert constructed == []
    assert load_events(path) == []


def test_invalid_item_fails_with_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield object()

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        with pytest.raises(StrictJSONError):
            next(proxy)

    assert load_events(path) == []


def test_invalid_sync_return_fails_with_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        return object()

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(StrictJSONError):
            next(proxy)

    assert load_events(path) == []


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        (1, 2),
        _Grade.A,
        _StrSub("x"),
        _IntSub(5),
        _FloatSub(1.5),
        _ListSub([1]),
        _DictSub({"k": 1}),
        {_StrSub("k"): 1},
        _Hostile(),
    ],
)
def test_invalid_item_types_do_not_bypass_exact_serialization(tmp_path, bad):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield bad

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        with pytest.raises(StrictJSONError):
            next(proxy)

    assert load_events(path) == []


def test_cyclic_item_is_rejected(tmp_path):
    path = tmp_path / "c.jsonl"
    cycle: list[Any] = []
    cycle.append(cycle)

    def stream(query):
        yield cycle

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        with pytest.raises(StrictJSONError):
            next(proxy)

    assert load_events(path) == []


def test_over_deep_item_is_rejected(tmp_path):
    path = tmp_path / "c.jsonl"
    deep: Any = 1
    for _ in range(100):
        deep = [deep]

    def stream(query):
        yield deep

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        with pytest.raises(StrictJSONError):
            next(proxy)

    assert load_events(path) == []


def test_yielded_mutable_value_is_detached_from_recorded_event(tmp_path):
    path = tmp_path / "c.jsonl"
    buffer = {"n": 1}

    def stream(query):
        yield buffer
        buffer["n"] = 999  # mutate the source after yielding

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        emitted = next(proxy)
        emitted["n"] = 555  # mutate the value handed to the caller
        list(proxy)  # exhaust to record

    # Neither the source mutation nor the caller mutation reaches the cassette.
    assert load_events(path)[0].output["items"] == [{"n": 1}]


# --------------------------------------------------------------------------- #
# Malformed cassette data fails closed
# --------------------------------------------------------------------------- #


def _valid_envelope() -> dict[str, Any]:
    return {
        "__agent_cassette_tool_stream__": True,
        "version": 1,
        "items": ["a"],
        "completion": "exhausted",
        "return": None,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update({"__agent_cassette_tool_stream__": False}),
        lambda e: e.pop("__agent_cassette_tool_stream__"),
        lambda e: e.update({"version": 2}),
        lambda e: e.update({"extra": 1}),
        lambda e: e.update({"completion": "bogus"}),
        lambda e: e.update({"items": {"not": "a list"}}),
    ],
)
def test_malformed_success_envelope_fails_closed(tmp_path, mutate):
    path = tmp_path / "c.jsonl"
    envelope = _valid_envelope()
    mutate(envelope)
    _seed_stream_event(path, envelope)

    def stream(query):
        yield "a"  # pragma: no cover

    with Cassette.replay(path) as replayer:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, replayer, name="stream")("q")


@pytest.mark.parametrize(
    "envelope",
    [
        {
            "__agent_cassette_tool_stream__": True,
            "version": 1,
            "items": [],
            "completion": "error",
            "phase": "nonsense",
            "error": {"type": "ValueError", "message": "x"},
        },
        {
            "__agent_cassette_tool_stream__": True,
            "version": 1,
            "items": ["a"],
            "completion": "error",
            "phase": "start",  # start errors must have no items
            "error": {"type": "ValueError", "message": "x"},
        },
        {
            "__agent_cassette_tool_stream__": True,
            "version": 1,
            "items": [],
            "completion": "error",
            "phase": "iteration",
            "error": {"type": "ValueError"},  # missing message
        },
    ],
)
def test_malformed_error_envelope_fails_closed(tmp_path, envelope):
    path = tmp_path / "c.jsonl"
    _seed_stream_event(path, envelope, is_error=True)

    def stream(query):
        yield "a"  # pragma: no cover

    with Cassette.replay(path) as replayer:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, replayer, name="stream")("q")


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #


def test_stream_inside_nested_span_records_lineage(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield query

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(stream, cassette)
        with cassette.span("outer"):
            with cassette.span("inner"):
                list(wrapped("q"))

    events = load_events(path)
    assert events[0].parent_id == "outer"
    assert events[0].span_id == "inner"


# --------------------------------------------------------------------------- #
# Hybrid replay-prefix-then-live and injection
# --------------------------------------------------------------------------- #


def test_stream_fork_replays_prefix_then_runs_live(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    calls = 0

    def stream(query):
        nonlocal calls
        calls += 1
        yield query
        yield f"{query}!"

    with Cassette.record(source) as cassette:
        list(wrap_tool(stream, cassette, name="stream")("first"))
    assert calls == 1

    with Cassette.fork(source, output, at=1) as forked:
        wrapped = wrap_tool(stream, forked, name="stream")
        replayed = list(wrapped("first"))
        live = list(wrapped("second"))

    assert calls == 2
    assert replayed == ["first", "first!"]
    assert live == ["second", "second!"]


def test_stream_return_injection_skips_live_body(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls = 0

    def stream(query):
        nonlocal calls
        calls += 1  # pragma: no cover
        yield query  # pragma: no cover

    rule = InjectionRule(Return(["x", "y"]), event_type="tool_call", name="stream", occurrence=1)
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        result = list(wrap_tool(stream, cassette, name="stream")("q"))

    assert result == ["x", "y"]
    assert calls == 0


def test_stream_return_injection_rejects_non_list_value(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    def stream(query):
        yield query  # pragma: no cover

    rule = InjectionRule(Return((1, 2)), event_type="tool_call", name="stream")
    with pytest.raises(StrictJSONError):
        with Cassette.fork(source, output, injections=(rule,)) as cassette:
            wrap_tool(stream, cassette, name="stream")("q")


def test_stream_raise_injection_defers_error_to_first_iteration(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls = 0

    def stream(query):
        nonlocal calls
        calls += 1  # pragma: no cover
        yield query  # pragma: no cover

    rule = InjectionRule(Raise(TimeoutError("slow")), event_type="tool_call", name="stream")
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        proxy = wrap_tool(stream, cassette, name="stream")("q")
        with pytest.raises(TimeoutError, match="slow"):
            next(proxy)

    assert calls == 0

    # The forked cassette recorded a replayable deferred error.
    with Cassette.replay(output) as replayer:
        proxy = wrap_tool(stream, replayer, name="stream")("q")
        with pytest.raises(TimeoutError, match="slow"):
            next(proxy)


def test_stream_delay_injection_composes_with_return(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    def stream(query):
        yield query  # pragma: no cover

    rule = InjectionRule(
        Delay(0.01, then=Return(["cached"])),
        event_type="tool_call",
        name="stream",
    )
    started = perf_counter()
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        result = list(wrap_tool(stream, cassette, name="stream")("q"))
    elapsed = perf_counter() - started

    assert result == ["cached"]
    assert elapsed >= 0.01


# --------------------------------------------------------------------------- #
# Callable / name matrix and signature preservation
# --------------------------------------------------------------------------- #


def test_generator_decorator_form(tmp_path):
    path = tmp_path / "c.jsonl"

    with Cassette.record(path) as cassette:

        @cassette.tool
        def stream(query):
            yield query

        assert list(stream("q")) == ["q"]

    assert load_events(path)[0].name == "stream"


def test_bound_generator_method(tmp_path):
    path = tmp_path / "c.jsonl"

    class Service:
        def stream(self, query):
            yield query

    service = Service()
    with Cassette.record(path) as cassette:
        assert list(wrap_tool(service.stream, cassette, name="svc")("q")) == ["q"]

    with Cassette.replay(path) as replayer:
        assert list(wrap_tool(service.stream, replayer, name="svc")("q")) == ["q"]


def test_named_generator_partial(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(prefix, query):
        yield f"{prefix}:{query}"

    bound = partial(stream, "p")
    with Cassette.record(path) as cassette:
        assert list(wrap_tool(bound, cassette, name="stream")("q")) == ["p:q"]


def test_generator_wrapper_preserves_signature_and_metadata():
    def stream(query, *, limit=10):
        """Stream docs."""
        yield query

    wrapped = wrap_tool(stream, object())
    assert wrapped.__name__ == "stream"
    assert wrapped.__doc__ == "Stream docs."
    assert wrapped.__wrapped__ is stream  # type: ignore[attr-defined]
    assert inspect.signature(wrapped) == inspect.signature(stream)


# --------------------------------------------------------------------------- #
# send/asend and unsupported bidirectional controls
# --------------------------------------------------------------------------- #


def test_sync_send_none_advances_and_non_none_rejected(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        yield "b"

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert proxy.send(None) == "a"
        with pytest.raises(NotImplementedError):
            proxy.send(123)
        with pytest.raises(NotImplementedError):
            proxy.throw(ValueError)
        proxy.close()


def test_replay_send_none_advances_and_non_none_rejected(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        yield "b"

    with Cassette.record(path) as cassette:
        list(wrap_tool(stream, cassette)("q"))

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(stream, replayer)("q")
        assert proxy.send(None) == "a"
        with pytest.raises(NotImplementedError):
            proxy.send(123)
        with pytest.raises(NotImplementedError):
            proxy.throw(ValueError)


def test_async_asend_none_advances_and_non_none_rejected(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield "a"
        yield "b"

    async def scenario():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.asend(None) == "a"
            with pytest.raises(NotImplementedError):
                await proxy.asend(123)
            with pytest.raises(NotImplementedError):
                await proxy.athrow(ValueError)
            await proxy.aclose()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Idempotency / no duplicate persistence
# --------------------------------------------------------------------------- #


def test_exhaustion_then_close_and_next_do_not_duplicate(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert list(proxy) == ["a"]
        proxy.close()  # no-op after exhaustion
        with pytest.raises(StopIteration):
            next(proxy)

    assert len(load_events(path)) == 1


def test_error_then_next_and_close_do_not_duplicate(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield "a"
        raise ValueError("boom")

    with Cassette.record(path) as cassette:
        proxy = wrap_tool(stream, cassette)("q")
        assert next(proxy) == "a"
        with pytest.raises(ValueError):
            next(proxy)
        proxy.close()  # no-op after error
        with pytest.raises(StopIteration):
            next(proxy)

    assert len(load_events(path)) == 1


# --------------------------------------------------------------------------- #
# Replay-envelope strict validation (Phase B correction)
# --------------------------------------------------------------------------- #
#
# Python-only invalid shapes (tuples, subclasses, cycles, over-depth, non-finite)
# cannot be seeded through a real cassette (event storage rejects them), so these
# exercise the private restore helper directly with a hand-built envelope.


def test_restore_rejects_nested_tuple_in_items():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_success_envelope(items=[[1, (2, 3)]]), asynchronous=False)


@pytest.mark.parametrize(
    "bad_return",
    [
        (1, 2),
        _Grade.A,
        _StrSub("x"),
        _IntSub(5),
        _FloatSub(1.5),
        _ListSub([1]),
        _DictSub({"k": 1}),
        float("nan"),
        float("inf"),
    ],
)
def test_restore_rejects_invalid_return_values(bad_return):
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(
            _success_envelope(items=["a"], **{"return": bad_return}), asynchronous=False
        )


def test_restore_rejects_cyclic_return():
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_success_envelope(**{"return": cycle}), asynchronous=False)


def test_restore_rejects_over_deep_return():
    deep: Any = 1
    for _ in range(100):
        deep = [deep]
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_success_envelope(**{"return": deep}), asynchronous=False)


def test_restore_rejects_str_subclass_key_in_item():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_success_envelope(items=[{_StrSub("k"): 1}]), asynchronous=False)


def test_restore_rejects_hostile_list_subclass_without_iterating():
    envelope = _success_envelope(items=[_HostileList([1, 2])])
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(envelope, asynchronous=False)


def test_restore_rejects_hostile_dict_subclass_without_reading_items():
    envelope = _success_envelope(items=[_HostileDict({"k": 1})])
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(envelope, asynchronous=False)


def test_restore_rejects_list_completion_without_typeerror():
    # completion=[] is JSON-representable, so a real seeded cassette exercises the
    # production replay path; it must fail closed with StrictJSONError, not TypeError.
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_success_envelope(completion=[]), asynchronous=False)


def test_restore_rejects_list_phase_without_typeerror():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(_error_envelope(phase=[]), asynchronous=False)


def test_restore_rejects_closed_with_non_none_return():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(
            _success_envelope(completion="closed", **{"return": "x"}), asynchronous=False
        )


def test_restore_rejects_async_exhausted_with_non_none_return():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(
            _success_envelope(completion="exhausted", **{"return": "x"}), asynchronous=True
        )


def test_restore_rejects_cancellation_phase_with_non_cancelled_type():
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(
            _error_envelope(phase="cancellation", error={"type": "ValueError", "message": "x"}),
            asynchronous=True,
        )


@pytest.mark.parametrize("phase", ["start", "iteration", "close"])
def test_restore_rejects_cancelled_type_outside_cancellation_phase(phase):
    items = [] if phase == "start" else ["a"]
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(
            _error_envelope(
                phase=phase,
                items=items,
                error={"type": "CancelledError", "message": "x"},
            ),
            asynchronous=False,
        )


def test_seeded_list_completion_replays_closed_via_wrapper(tmp_path):
    # End-to-end through the wrapper on a real cassette: JSON-representable garbage
    # must raise StrictJSONError at call time, never a raw TypeError.
    path = tmp_path / "c.jsonl"
    _seed_stream_event(path, _success_envelope(completion=[]))

    def stream(query):
        yield "a"  # pragma: no cover

    with Cassette.replay(path) as replayer:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, replayer, name="stream")("q")


def test_seeded_closed_with_return_fails_closed_via_wrapper(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed_stream_event(path, _success_envelope(completion="closed", **{"return": "x"}))

    def stream(query):
        yield "a"  # pragma: no cover

    with Cassette.replay(path) as replayer:
        with pytest.raises(StrictJSONError):
            wrap_tool(stream, replayer, name="stream")("q")


def test_replayed_nested_item_mutation_does_not_mutate_loaded_event(tmp_path):
    path = tmp_path / "c.jsonl"

    def stream(query):
        yield {"n": 1}

    with Cassette.record(path) as cassette:
        list(wrap_tool(stream, cassette)("q"))

    def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    with Cassette.replay(path) as replayer:
        proxy = wrap_tool(forbidden, replayer, name="stream")("q")
        emitted = next(proxy)
        emitted["n"] = 999  # mutate the replayed item
        # The loaded event that the Replayer still holds must be untouched.
        assert replayer.events[0].output["items"][0] == {"n": 1}


# --------------------------------------------------------------------------- #
# Async parity (Phase B correction)
# --------------------------------------------------------------------------- #


def test_async_start_error_replays(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        raise ValueError("early")
        yield  # pragma: no cover

    async def record():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            with pytest.raises(ValueError, match="early"):
                await proxy.__anext__()

    asyncio.run(record())

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].output["phase"] == "start"
    assert events[0].output["items"] == []

    async def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    async def replay():
        async with Cassette.replay(path) as replayer:
            proxy = wrap_tool(forbidden, replayer, name="stream")("q")
            with pytest.raises(ValueError, match="early"):
                await proxy.__anext__()

    asyncio.run(replay())


def test_async_unknown_exception_replays_as_recorded_call_error(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield "a"
        raise _CustomError("weird")

    async def record():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(_CustomError, match="weird"):
                await proxy.__anext__()

    asyncio.run(record())

    async def forbidden(query):
        raise AssertionError("live body entered during replay")
        yield  # pragma: no cover

    async def replay():
        async with Cassette.replay(path) as replayer:
            proxy = wrap_tool(forbidden, replayer, name="stream")("q")
            assert await proxy.__anext__() == "a"
            with pytest.raises(RecordedCallError):
                await proxy.__anext__()

    asyncio.run(replay())


def test_async_invalid_item_closes_and_persists_no_event(tmp_path):
    path = tmp_path / "c.jsonl"
    finalized: list[bool] = []

    async def stream(query):
        try:
            yield object()
        finally:
            finalized.append(True)

    async def scenario():
        async with Cassette.record(path) as cassette:
            proxy = wrap_tool(stream, cassette)("q")
            with pytest.raises(StrictJSONError):
                await proxy.__anext__()

    asyncio.run(scenario())
    assert finalized == [True]
    assert load_events(path) == []


def test_async_unstarted_stream_persists_no_event(tmp_path):
    path = tmp_path / "c.jsonl"

    async def stream(query):
        yield query  # pragma: no cover

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrap_tool(stream, cassette)("q")

    asyncio.run(scenario())
    assert load_events(path) == []


@pytest.mark.parametrize("missing", ["items", "return", "completion", "version"])
def test_restore_missing_key_fails_closed_not_keyerror(missing):
    envelope = _success_envelope()
    envelope.pop(missing)
    with pytest.raises(StrictJSONError):
        _restore_tool_stream(envelope, asynchronous=False)
