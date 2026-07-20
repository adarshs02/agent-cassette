"""Tests for `wrap_tool` — Phase A sync/async Python tool record and replay."""

from __future__ import annotations

import asyncio
import inspect
from enum import IntEnum
from functools import partial
from time import perf_counter
from typing import Any, cast

import pytest

from agent_cassette import (
    Cassette,
    Delay,
    EventType,
    InjectionRule,
    Raise,
    RecordedCallError,
    ReplayMismatchError,
    Return,
    wrap_tool,
)
from agent_cassette.json_codec import StrictJSONError
from agent_cassette.storage import load_events


class Hostile:
    """An object whose value-rendering methods must never be invoked."""

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


class _FloatSub(float):
    pass


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


class _HostileList(list):
    """A list subclass whose iteration must never be reached (exact-type reject first)."""

    def __iter__(self):
        raise AssertionError("__iter__ called")


class _HostileDict(dict):
    """A dict subclass whose items() must never be reached (exact-type reject first)."""

    def items(self):
        raise AssertionError("items called")


_SUBCLASS_FACTORIES = [
    lambda: _Grade.A,
    lambda: _StrSub("x"),
    lambda: _FloatSub(1.5),
    lambda: _ListSub([1]),
    lambda: _DictSub({"k": 1}),
]


class _CustomToolError(Exception):
    """A tool exception type that is not on the replay allowlist."""


# --------------------------------------------------------------------------- #
# Core replay
# --------------------------------------------------------------------------- #


def test_sync_tool_records_and_replays(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def search(query):
        nonlocal calls
        calls += 1
        return {"query": query, "results": ["a", "b"]}

    with Cassette.record(path) as cassette:
        recorded = wrap_tool(search, cassette)("agents")

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(search, replayer)("agents")

    assert recorded == replayed == {"query": "agents", "results": ["a", "b"]}
    assert calls == 1


def test_async_tool_records_and_replays(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    async def fetch(url):
        nonlocal calls
        calls += 1
        return {"url": url}

    async def scenario():
        async with Cassette.record(path) as cassette:
            recorded = await wrap_tool(fetch, cassette)("https://x")
        async with Cassette.replay(path) as cassette:
            replayed = await wrap_tool(fetch, cassette)("https://x")
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded == replayed == {"url": "https://x"}
    assert calls == 1


def test_sync_replay_never_executes_live_body(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def search(query):
        return {"query": query}

    with Cassette.record(path) as cassette:
        wrap_tool(search, cassette)("agents")

    def forbidden(query):
        raise AssertionError("live body entered during replay")

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(forbidden, replayer, name="search")
        result = replayed("agents")

    assert result == {"query": "agents"}


def test_async_replay_never_executes_live_body(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def fetch(url):
        return {"url": url}

    async def forbidden(url):
        raise AssertionError("live body entered during replay")

    async def scenario():
        async with Cassette.record(path) as cassette:
            await wrap_tool(fetch, cassette)("https://x")
        async with Cassette.replay(path) as cassette:
            replayed = wrap_tool(forbidden, cassette, name="fetch")
            return await replayed("https://x")

    result = asyncio.run(scenario())
    assert result == {"url": "https://x"}


def test_explicit_tool_name_controls_matching(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def fetch(url):
        return {"url": url}

    with Cassette.record(path) as cassette:
        wrap_tool(fetch, cassette, name="web.fetch")("https://x")

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(fetch, replayer, name="web.fetch")
        assert replayed("https://x") == {"url": "https://x"}

    with pytest.raises(ReplayMismatchError):
        with Cassette.replay(path) as replayer:
            wrap_tool(fetch, replayer)("https://x")


def test_tool_records_one_tool_call_event(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def search(query):
        return {"query": query}

    with Cassette.record(path) as cassette:
        wrap_tool(search, cassette)("agents")

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.TOOL_CALL
    assert events[0].name == "search"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


def test_sync_allowlisted_exception_replays_same_type(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def flaky(query):
        raise ValueError("bad query")

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(flaky, cassette)
        with pytest.raises(ValueError, match="bad query"):
            wrapped("agents")

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(flaky, replayer)
        with pytest.raises(ValueError, match="bad query"):
            replayed("agents")


def test_async_allowlisted_exception_replays_same_type(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def flaky(query):
        raise TimeoutError("slow backend")

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(flaky, cassette)
            with pytest.raises(TimeoutError, match="slow backend"):
                await wrapped("agents")

        async with Cassette.replay(path) as cassette:
            replayed = wrap_tool(flaky, cassette)
            with pytest.raises(TimeoutError, match="slow backend"):
                await replayed("agents")

    asyncio.run(scenario())


def test_custom_exception_replays_as_recorded_call_error(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def flaky(query):
        raise _CustomToolError("custom failure")

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(flaky, cassette)
        with pytest.raises(_CustomToolError):
            wrapped("agents")

    events = load_events(path)
    assert events[-1].type == EventType.ERROR
    assert events[-1].metadata["_agent_cassette"]["call_type"] == "tool_call"

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(flaky, replayer)
        with pytest.raises(RecordedCallError) as excinfo:
            replayed("agents")
        assert excinfo.value.recorded_type == "_CustomToolError"


# --------------------------------------------------------------------------- #
# Decorators and callable support
# --------------------------------------------------------------------------- #


def test_bare_tool_decorator(tmp_path):
    path = tmp_path / "cassette.jsonl"

    with Cassette.record(path) as cassette:

        @cassette.tool
        def search(query):
            return {"query": query}

        assert search("agents") == {"query": "agents"}

    with Cassette.replay(path) as replayer:

        @replayer.tool
        def search(query):
            raise AssertionError("live body entered during replay")

        assert search("agents") == {"query": "agents"}


def test_named_tool_decorator(tmp_path):
    path = tmp_path / "cassette.jsonl"

    with Cassette.record(path) as cassette:

        @cassette.tool(name="web.search")
        def search(query):
            return {"query": query}

        assert search("agents") == {"query": "agents"}

    events = load_events(path)
    assert events[0].name == "web.search"


def test_bound_sync_method(tmp_path):
    path = tmp_path / "cassette.jsonl"

    class Client:
        def search(self, query):
            return {"query": query}

    client = Client()

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(client.search, cassette)
        assert wrapped("agents") == {"query": "agents"}

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(client.search, replayer)
        assert replayed("agents") == {"query": "agents"}


def test_bound_async_method(tmp_path):
    path = tmp_path / "cassette.jsonl"

    class Client:
        async def fetch(self, url):
            return {"url": url}

    client = Client()

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(client.fetch, cassette)
            assert await wrapped("https://x") == {"url": "https://x"}

        async with Cassette.replay(path) as cassette:
            replayed = wrap_tool(client.fetch, cassette)
            assert await replayed("https://x") == {"url": "https://x"}

    asyncio.run(scenario())


def test_named_sync_partial(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def search(base_url, query):
        return {"url": base_url, "query": query}

    bound = partial(search, "https://x")
    signature_check = wrap_tool(bound, object(), name="search")

    assert inspect.signature(signature_check) == inspect.signature(bound)
    assert cast(Any, signature_check).__wrapped__ is bound

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(bound, cassette, name="search")
        assert wrapped("agents") == {"url": "https://x", "query": "agents"}

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(bound, replayer, name="search")
        assert replayed("agents") == {"url": "https://x", "query": "agents"}


def test_named_async_partial(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def fetch(base_url, query):
        return {"url": base_url, "query": query}

    bound = partial(fetch, "https://x")
    signature_check = wrap_tool(bound, object(), name="fetch")

    assert inspect.signature(signature_check) == inspect.signature(bound)
    assert cast(Any, signature_check).__wrapped__ is bound

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(bound, cassette, name="fetch")
            assert await wrapped("agents") == {"url": "https://x", "query": "agents"}

        async with Cassette.replay(path) as cassette:
            replayed = wrap_tool(bound, cassette, name="fetch")
            assert await replayed("agents") == {"url": "https://x", "query": "agents"}

    asyncio.run(scenario())


def test_lambda_requires_name():
    with pytest.raises(ValueError, match="name"):
        wrap_tool(lambda query: query, object())


def test_partial_requires_name():
    def search(base, query):
        return query

    bound = partial(search, "base")
    with pytest.raises(ValueError, match="name"):
        wrap_tool(bound, object())


def test_callable_object_rejected():
    class Tool:
        def __call__(self, query):
            return query

    with pytest.raises(ValueError, match="callable object"):
        wrap_tool(Tool(), object())


def test_builtin_rejected():
    with pytest.raises(ValueError, match="builtin"):
        wrap_tool(len, object())


def test_non_callable_rejected():
    with pytest.raises(TypeError, match="callable"):
        wrap_tool(cast(Any, 42), object())


def test_empty_name_rejected():
    def search(query):
        return query

    with pytest.raises(ValueError, match="nonempty"):
        wrap_tool(search, object(), name="   ")


def test_wrapper_preserves_signature_and_metadata():
    def search(query, *, limit=10):
        """Search docs."""
        return {"query": query, "limit": limit}

    wrapped = wrap_tool(search, object())

    assert wrapped.__name__ == "search"
    assert wrapped.__doc__ == "Search docs."
    assert cast(Any, wrapped).__wrapped__ is search
    assert inspect.signature(wrapped) == inspect.signature(search)


# --------------------------------------------------------------------------- #
# Generator rejection
# --------------------------------------------------------------------------- #


def test_generator_tool_rejected_before_execution():
    entered = False

    def stream(query):
        nonlocal entered
        entered = True
        yield query

    with pytest.raises(ValueError, match="generator"):
        wrap_tool(stream, object())

    assert entered is False


def test_async_generator_tool_rejected_before_execution():
    entered = False

    async def astream(query):
        nonlocal entered
        entered = True
        yield query

    with pytest.raises(ValueError, match="generator"):
        wrap_tool(astream, object())

    assert entered is False


# --------------------------------------------------------------------------- #
# Strict serialization
# --------------------------------------------------------------------------- #


def test_json_native_inputs_round_trip(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def echo(*args, **kwargs):
        return {"args": list(args), "kwargs": kwargs}

    payload_args = (None, "text", True, 7, 1.5, ["a", 1, False], {"ok": True})
    payload_kwargs = {"flag": True, "items": [1, 2, 3]}

    with Cassette.record(path) as cassette:
        recorded = wrap_tool(echo, cassette)(*payload_args, **payload_kwargs)

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(echo, replayer)(*payload_args, **payload_kwargs)

    assert recorded == replayed


def test_non_string_input_key_rejected_before_execution(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped({1: "a"})

    assert calls == 0


def test_tuple_input_rejected_before_execution(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped((1, 2))

    assert calls == 0


def test_non_finite_input_rejected_before_execution(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped(float("nan"))

    assert calls == 0


def test_cyclic_input_rejected_before_execution(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    cyclic: dict = {}
    cyclic["self"] = cyclic

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped(cyclic)

    assert calls == 0


def test_arbitrary_input_does_not_call_str_repr_or_model_dump(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped(Hostile())

    assert calls == 0


@pytest.mark.parametrize(
    "output",
    [None, "text", True, 7, 1.5, ["a", 1, False], {"ok": True, "items": [1, 2]}],
)
def test_supported_outputs_preserve_type_and_value(tmp_path, output):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return output

    with Cassette.record(path) as cassette:
        recorded = wrap_tool(tool, cassette)()

    with Cassette.replay(path) as replayer:
        replayed = wrap_tool(tool, replayer)()

    assert type(recorded) is type(output)
    assert recorded == output
    assert type(replayed) is type(output)
    assert replayed == output


def test_tuple_output_rejected(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return (1, 2)

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


def test_rich_output_rejected_without_str_or_repr(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return Hostile()

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


def test_non_finite_output_rejected(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return float("inf")

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


def test_cyclic_output_rejected(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def tool():
        cyclic: dict = {}
        cyclic["self"] = cyclic
        return cyclic

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


@pytest.mark.parametrize("factory", _SUBCLASS_FACTORIES)
def test_subclass_input_rejected_before_execution(tmp_path, factory):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped(factory())

    assert calls == 0


def test_str_subclass_dict_key_input_rejected_before_execution(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped({_StrSub("k"): 1})

    assert calls == 0


def test_hostile_container_subclass_rejected_without_iteration(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def tool(payload):
        nonlocal calls
        calls += 1
        return payload

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        # StrictJSONError (not AssertionError) proves the overridden __iter__/items ran never.
        with pytest.raises(StrictJSONError):
            wrapped(_HostileList([1, 2]))
        with pytest.raises(StrictJSONError):
            wrapped(_HostileDict({"k": 1}))

    assert calls == 0


@pytest.mark.parametrize("factory", _SUBCLASS_FACTORIES)
def test_subclass_output_rejected(tmp_path, factory):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return factory()

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


def test_str_subclass_dict_key_output_rejected(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def tool():
        return {_StrSub("k"): 1}

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(tool, cassette)
        with pytest.raises(StrictJSONError):
            wrapped()

    assert load_events(path) == []


# --------------------------------------------------------------------------- #
# Redaction and spans
# --------------------------------------------------------------------------- #


def test_tool_input_is_redacted_before_persistence(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def login(api_key):
        return {"ok": True}

    with Cassette.record(path) as cassette:
        wrap_tool(login, cassette)(api_key="super-secret")

    events = load_events(path)
    assert events[0].input == {"args": [], "kwargs": {"api_key": "[REDACTED]"}}


def test_tool_inside_nested_span_records_parent_and_span(tmp_path):
    path = tmp_path / "cassette.jsonl"

    def search(query):
        return query

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(search, cassette)
        with cassette.span("outer"):
            with cassette.span("inner"):
                wrapped("agents")

    events = load_events(path)
    assert events[0].parent_id == "outer"
    assert events[0].span_id == "inner"


# --------------------------------------------------------------------------- #
# Hybrid and injection
# --------------------------------------------------------------------------- #


def test_tool_fork_replays_prefix_then_runs_live(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    calls = 0

    def search(query):
        nonlocal calls
        calls += 1
        return {"query": query, "live": True}

    with Cassette.record(source) as cassette:
        wrap_tool(search, cassette, name="search")("first")

    assert calls == 1

    with Cassette.fork(source, output, at=1) as forked:
        wrapped = wrap_tool(search, forked, name="search")
        replayed_result = wrapped("first")
        live_result = wrapped("second")

    assert calls == 2
    assert replayed_result == {"query": "first", "live": True}
    assert live_result == {"query": "second", "live": True}


def test_tool_return_injection_skips_live_body(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls = 0

    def search(query):
        nonlocal calls
        calls += 1
        return {"query": query}

    rule = InjectionRule(Return({"ok": True}), event_type="tool_call", name="search", occurrence=1)
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        wrapped = wrap_tool(search, cassette, name="search")
        result = wrapped("agents")

    assert result == {"ok": True}
    assert calls == 0


def test_tool_raise_injection_skips_live_body(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls = 0

    def fetch(url):
        nonlocal calls
        calls += 1
        return {"url": url}

    rule = InjectionRule(Raise(TimeoutError("slow")), event_type="tool_call", name="fetch")
    with pytest.raises(TimeoutError, match="slow"):
        with Cassette.fork(source, output, injections=(rule,)) as cassette:
            wrapped = wrap_tool(fetch, cassette, name="fetch")
            wrapped("https://x")

    assert calls == 0


def test_tool_delay_injection_composes_with_return(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls = 0

    def fetch(url):
        nonlocal calls
        calls += 1
        return {"url": url}

    rule = InjectionRule(
        Delay(0.01, then=Return({"cached": True})),
        event_type="tool_call",
        name="fetch",
    )
    started = perf_counter()
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        wrapped = wrap_tool(fetch, cassette, name="fetch")
        result = wrapped("https://x")
    elapsed = perf_counter() - started

    assert result == {"cached": True}
    assert calls == 0
    assert elapsed >= 0.01


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_async_tools_replay_non_strict_and_consume_all_events(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def fetch(url):
        return {"url": url}

    async def record_scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(fetch, cassette)
            await asyncio.gather(wrapped("a"), wrapped("b"), wrapped("c"))

    asyncio.run(record_scenario())

    async def replay_scenario():
        async with Cassette.replay(path, strict=False) as replayer:
            wrapped = wrap_tool(fetch, replayer)
            results = await asyncio.gather(wrapped("a"), wrapped("b"), wrapped("c"))
            assert replayer.remaining == 0
            return results

    results = asyncio.run(replay_scenario())
    assert {result["url"] for result in results} == {"a", "b", "c"}


def test_identical_async_tool_inputs_use_first_unconsumed_match(tmp_path):
    path = tmp_path / "cassette.jsonl"

    with Cassette.record(path) as cassette:
        cassette.add(
            EventType.TOOL_CALL, "fetch", input={"args": ["x"], "kwargs": {}}, output="first"
        )
        cassette.add(
            EventType.TOOL_CALL, "fetch", input={"args": ["x"], "kwargs": {}}, output="second"
        )

    async def fetch(url):
        raise AssertionError("live body entered during replay")

    async def scenario():
        async with Cassette.replay(path, strict=False) as replayer:
            wrapped = wrap_tool(fetch, replayer, name="fetch")
            results = await asyncio.gather(wrapped("x"), wrapped("x"))
            assert replayer.remaining == 0
            return results

    results = asyncio.run(scenario())
    # Identical concurrent inputs are matched first-unconsumed; both recorded
    # events are drained exactly once each, but which task receives which
    # output is an accepted, documented ambiguity (Phase A design §5).
    assert sorted(results) == ["first", "second"]


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_caught_cancelled_tool_records_no_tool_event(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def flaky(query):
        raise asyncio.CancelledError()

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(flaky, cassette)
            try:
                await wrapped("agents")
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())

    assert load_events(path) == []


def test_uncaught_cancelled_tool_records_generic_uncaught_exception(tmp_path):
    path = tmp_path / "cassette.jsonl"

    async def flaky(query):
        raise asyncio.CancelledError()

    async def scenario():
        async with Cassette.record(path) as cassette:
            wrapped = wrap_tool(flaky, cassette)
            await wrapped("agents")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    events = load_events(path)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].name == "uncaught_exception"
    # No `_agent_cassette.call_type` metadata exists on this generic exit event;
    # it is filtered from replay solely because its event type is ERROR.
    assert events[0].metadata == {}


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_tool_invocation_after_context_uses_inherited_session_behavior(tmp_path):
    path = tmp_path / "cassette.jsonl"
    calls = 0

    def search(query):
        nonlocal calls
        calls += 1
        return {"query": query}

    with Cassette.record(path) as cassette:
        wrapped = wrap_tool(search, cassette)

    # wrap_tool adds no lifecycle guard; invoking the wrapper after the `with`
    # block still records, because it is inherited Recorder behavior (still
    # appends and writes), not something wrap_tool enforces or prevents.
    result = wrapped("post-context")

    assert result == {"query": "post-context"}
    assert calls == 1
    events = load_events(path)
    assert events[-1].name == "search"
    assert events[-1].output == {"query": "post-context"}
