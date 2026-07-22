"""Phase C2 — LangChain registered-tool replay bridge.

Uses the installed real ``@tool``/``Tool``/``StructuredTool``/``BaseTool`` APIs with
no network or credential. Skips cleanly when ``langchain_core`` is absent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import (  # noqa: E402
    BaseTool,
    StructuredTool,
    Tool,
    ToolException,
    tool,
)

from agent_cassette import (  # noqa: E402
    Cassette,
    Delay,
    EventType,
    InjectionRule,
    Raise,
    RecordedCallError,
    ReplayMismatchError,
    Return,
    wrap_langchain_tools,
)
from agent_cassette.integrations.langchain_tools import (  # noqa: E402
    _build_request,
    _clean_config,
    _decode_tool_result,
    _encode_tool_result,
)
from agent_cassette.json_codec import StrictJSONError  # noqa: E402
from agent_cassette.storage import load_events  # noqa: E402

# --------------------------------------------------------------------------- #
# Tool factories
# --------------------------------------------------------------------------- #


def _search(counter, *, forbidden=False):
    @tool
    def search(query: str) -> str:
        "search docs"
        counter.append(query)
        if forbidden:
            raise AssertionError("live tool body executed during replay")
        return "result:" + query

    return search


def _async_search(counter, *, forbidden=False):
    @tool
    async def search(query: str) -> str:
        "async search"
        counter.append(query)
        if forbidden:
            raise AssertionError("live async body executed during replay")
        return "async:" + query

    return search


def _artifact_tool(counter, *, forbidden=False):
    @tool(response_format="content_and_artifact")
    def build(query: str):
        "artifact tool"
        counter.append(query)
        if forbidden:
            raise AssertionError("live artifact body executed during replay")
        return ("content:" + query, {"artifact": [1, 2, 3]})

    return build


# --------------------------------------------------------------------------- #
# Core record / replay
# --------------------------------------------------------------------------- #


def test_sync_invoke_record_replay_zero_live(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        rec = bridged.invoke({"query": "agents"})
    assert calls == ["agents"]
    assert rec == "result:agents"

    events = load_events(path)
    assert [e.type for e in events] == [EventType.TOOL_CALL]
    assert events[0].name == "langchain.tool.search"
    assert events[0].metadata["tool_bridge"] is True
    assert events[0].metadata["integration"] == "langchain"

    replay_calls: list[str] = []
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search(replay_calls, forbidden=True)], replayer)
        rep = bridged.invoke({"query": "agents"})
        assert replayer.remaining == 0
    assert rep == "result:agents"
    assert replay_calls == []


def test_sync_run_record_replay(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        rec = bridged.run("agents")
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
        rep = bridged.run("agents")
        assert replayer.remaining == 0
    assert rec == rep == "result:agents"


def test_async_ainvoke_record_replay_zero_live(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []

    async def scenario():
        with Cassette.record(path) as cassette:
            (bridged,) = wrap_langchain_tools([_async_search(calls)], cassette)
            rec = await bridged.ainvoke({"query": "agents"})
        replay_calls: list[str] = []
        with Cassette.replay(path) as replayer:
            (bridged,) = wrap_langchain_tools(
                [_async_search(replay_calls, forbidden=True)], replayer
            )
            rep = await bridged.ainvoke({"query": "agents"})
            assert replayer.remaining == 0
        return rec, rep, replay_calls

    rec, rep, replay_calls = asyncio.run(scenario())
    assert rec == rep == "async:agents"
    assert replay_calls == []


def test_sync_only_tool_async_fallback(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []

    async def scenario():
        with Cassette.record(path) as cassette:
            (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
            rec = await bridged.ainvoke({"query": "agents"})  # sync tool via async fallback
        with Cassette.replay(path) as replayer:
            (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
            rep = await bridged.ainvoke({"query": "agents"})
            assert replayer.remaining == 0
        return rec, rep

    rec, rep = asyncio.run(scenario())
    assert rec == rep == "result:agents"


def test_async_concurrent_unique_inputs_non_strict(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []

    async def record():
        with Cassette.record(path) as cassette:
            (bridged,) = wrap_langchain_tools([_async_search(calls)], cassette)
            await asyncio.gather(bridged.ainvoke({"query": "x"}), bridged.ainvoke({"query": "y"}))

    asyncio.run(record())

    async def replay():
        with Cassette.replay(path, strict=False) as replayer:
            (bridged,) = wrap_langchain_tools([_async_search([], forbidden=True)], replayer)
            results = await asyncio.gather(
                bridged.ainvoke({"query": "x"}), bridged.ainvoke({"query": "y"})
            )
            assert replayer.remaining == 0
            return results

    results = asyncio.run(replay())
    assert set(results) == {"async:x", "async:y"}


def test_args_schema_validation_runs_on_replay_and_consumes_no_event(tmp_path):
    path = tmp_path / "c.jsonl"
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search([])], cassette)
        bridged.invoke({"query": "agents"})

    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
        with pytest.raises(Exception):  # noqa: B017 -- SDK validation error type varies
            bridged.invoke({"wrong_field": 1})
        assert replayer.remaining == 1  # invalid schema input consumed no event
        assert bridged.invoke({"query": "agents"}) == "result:agents"
        assert replayer.remaining == 0


# --------------------------------------------------------------------------- #
# content_and_artifact
# --------------------------------------------------------------------------- #


def test_content_and_artifact_round_trip(tmp_path):
    from langchain_core.messages import ToolMessage

    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_artifact_tool(calls)], cassette)
        rec = bridged.invoke(
            {"args": {"query": "agents"}, "type": "tool_call", "id": "1", "name": "build"}
        )
    assert isinstance(rec, ToolMessage)
    assert rec.content == "content:agents"
    assert rec.artifact == {"artifact": [1, 2, 3]}

    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_artifact_tool([], forbidden=True)], replayer)
        rep = bridged.invoke(
            {"args": {"query": "agents"}, "type": "tool_call", "id": "1", "name": "build"}
        )
        assert replayer.remaining == 0
    assert isinstance(rep, ToolMessage)
    assert rep.content == "content:agents"
    assert rep.artifact == {"artifact": [1, 2, 3]}


def test_content_result_detached_from_loaded_event(tmp_path):
    path = tmp_path / "c.jsonl"

    @tool
    def emit(query: str) -> dict:
        "emit dict"
        return {"query": query, "hits": ["a", "b"]}

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([emit], cassette)
        bridged.invoke({"query": "x"})

    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([emit], replayer)
        result = bridged.invoke({"query": "x"})
        result["hits"].append("MUTATED")
        stored = replayer.events[0]
        assert stored.output["value"] == {"query": "x", "hits": ["a", "b"]}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_allowlisted_error_replays(tmp_path):
    path = tmp_path / "c.jsonl"

    @tool
    def boom(query: str) -> str:
        "boom"
        raise ValueError("bad query")

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([boom], cassette)
        with pytest.raises(ValueError, match="bad query"):
            bridged.invoke({"query": "x"})

    ran: list[bool] = []

    @tool
    def forbidden(query: str) -> str:
        "forbidden"
        ran.append(True)
        raise AssertionError("live executed on replay")

    forbidden.name = "boom"
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([forbidden], replayer)
        with pytest.raises(ValueError, match="bad query"):
            bridged.invoke({"query": "x"})
    assert ran == []


def test_unknown_error_replays_as_recorded_call_error(tmp_path):
    path = tmp_path / "c.jsonl"

    class _Custom(Exception):
        pass

    @tool
    def boom(query: str) -> str:
        "boom"
        raise _Custom("weird")

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([boom], cassette)
        with pytest.raises(_Custom):
            bridged.invoke({"query": "x"})

    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([boom], replayer)
        with pytest.raises(RecordedCallError):
            bridged.invoke({"query": "x"})


def test_tool_exception_translates_and_handler_reruns(tmp_path):
    path = tmp_path / "c.jsonl"
    handled: list[int] = []

    def handler(error):
        handled.append(1)
        return "handled: " + str(error)

    def make(counter, *, forbidden=False):
        @tool(response_format="content")
        def risky(query: str) -> str:
            "risky"
            counter.append(query)
            if forbidden:
                raise AssertionError("live executed on replay")
            raise ToolException("tool failed")

        risky.handle_tool_error = handler
        return risky

    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([make(calls)], cassette)
        rec = bridged.invoke({"query": "x"})
    assert rec == "handled: tool failed"
    assert handled == [1]

    handled.clear()
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([make([], forbidden=True)], replayer)
        rep = bridged.invoke({"query": "x"})
    assert rep == "handled: tool failed"
    assert handled == [1]  # handler ran again on replay


# --------------------------------------------------------------------------- #
# Hybrid injection
# --------------------------------------------------------------------------- #


def test_hybrid_return_injection(tmp_path):
    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    calls: list[str] = []
    rule = InjectionRule(Return("injected"), event_type="tool_call", name="langchain.tool.search")
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        result = bridged.invoke({"query": "x"})
    assert result == "injected"
    assert calls == []


def test_hybrid_raise_injection(tmp_path):
    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    calls: list[str] = []
    rule = InjectionRule(
        Raise(ValueError("nope")), event_type="tool_call", name="langchain.tool.search"
    )
    with pytest.raises(ValueError, match="nope"):
        with Cassette.fork(source, output, injections=(rule,)) as cassette:
            (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
            bridged.invoke({"query": "x"})
    assert calls == []


def test_hybrid_delay_then_return(tmp_path):
    from time import perf_counter

    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    calls: list[str] = []
    rule = InjectionRule(
        Delay(0.01, then=Return("cached")),
        event_type="tool_call",
        name="langchain.tool.search",
    )
    started = perf_counter()
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        result = bridged.invoke({"query": "x"})
    elapsed = perf_counter() - started
    assert result == "cached"
    assert calls == []
    assert elapsed >= 0.01


def test_hybrid_return_artifact_tuple_injection_persists_and_replays(tmp_path):
    from langchain_core.messages import ToolMessage

    source = tmp_path / "s.jsonl"
    output = tmp_path / "f.jsonl"
    with Cassette.record(source):
        pass
    calls: list[str] = []
    rule = InjectionRule(
        Return(("injected-content", {"a": 1})),
        event_type="tool_call",
        name="langchain.tool.build",
    )
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        (bridged,) = wrap_langchain_tools([_artifact_tool(calls)], cassette)
        rec = bridged.invoke(
            {"args": {"query": "x"}, "type": "tool_call", "id": "1", "name": "build"}
        )
    assert isinstance(rec, ToolMessage)
    assert rec.content == "injected-content"
    assert rec.artifact == {"a": 1}
    assert calls == []

    with Cassette.replay(output) as replayer:
        (bridged,) = wrap_langchain_tools([_artifact_tool([], forbidden=True)], replayer)
        rep = bridged.invoke(
            {"args": {"query": "x"}, "type": "tool_call", "id": "1", "name": "build"}
        )
    assert isinstance(rep, ToolMessage)
    assert rep.content == "injected-content"
    assert rep.artifact == {"a": 1}


# --------------------------------------------------------------------------- #
# Request / result codec (unit)
# --------------------------------------------------------------------------- #


class _Grade(int):
    pass


class _Hostile:
    def __str__(self):
        raise AssertionError("__str__")

    def __repr__(self):
        raise AssertionError("__repr__")


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"x": (1, 2)},
        {"x": _Grade(3)},
        {"x": float("nan")},
        {"x": _Hostile()},
        {"x": {1: "int key"}},
    ],
)
def test_build_request_rejects_unsupported(bad_kwargs):
    with pytest.raises(StrictJSONError):
        _build_request((), bad_kwargs)


def test_build_request_rejects_cyclic():
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(StrictJSONError):
        _build_request((), {"x": cycle})


def test_clean_config_drops_volatile_keys():
    cleaned = _clean_config(
        {"callbacks": object(), "run_id": "r", "run_name": "n", "tags": ["a"], "metadata": {"k": 1}}
    )
    assert cleaned == {"tags": ["a"], "metadata": {"k": 1}}


def test_clean_config_rejects_non_dict():
    with pytest.raises(StrictJSONError):
        _clean_config(["not", "a", "dict"])


def test_build_request_removes_run_manager_and_config():
    request = _build_request(
        (), {"query": "x", "run_manager": object(), "config": {"tags": ["t"], "run_id": "1"}}
    )
    assert request == {"args": [], "kwargs": {"query": "x"}, "config": {"tags": ["t"]}}


def test_encode_content_and_artifact_requires_tuple():
    with pytest.raises(StrictJSONError):
        _encode_tool_result(["not", "a", "tuple"], "content_and_artifact")
    with pytest.raises(StrictJSONError):
        _encode_tool_result(("only-one",), "content_and_artifact")


def test_encode_rejects_unsupported_content():
    with pytest.raises(StrictJSONError):
        _encode_tool_result((1, 2), "content")  # tuple is not JSON-native


def _result_envelope(**overrides):
    envelope = {
        "__agent_cassette_langchain_tool_result__": True,
        "version": 1,
        "kind": "json",
        "value": "ok",
    }
    envelope.update(overrides)
    return envelope


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update({"__agent_cassette_langchain_tool_result__": False}),
        lambda e: e.update({"version": 2}),
        lambda e: e.update({"kind": "bogus"}),
        lambda e: e.update({"extra": 1}),
    ],
)
def test_decode_malformed_envelope_fails_closed(mutate):
    envelope = _result_envelope()
    mutate(envelope)
    with pytest.raises(StrictJSONError):
        _decode_tool_result(envelope, "content")


def test_decode_kind_response_format_mismatch_fails_closed():
    with pytest.raises(StrictJSONError):
        _decode_tool_result(_result_envelope(), "content_and_artifact")
    envelope = _result_envelope(kind="content_and_artifact", value=["c", {"a": 1}])
    with pytest.raises(StrictJSONError):
        _decode_tool_result(envelope, "content")


def test_decode_content_and_artifact_shape():
    envelope = _result_envelope(kind="content_and_artifact", value=["c", {"a": 1}])
    assert _decode_tool_result(envelope, "content_and_artifact") == ("c", {"a": 1})
    bad = _result_envelope(kind="content_and_artifact", value=["only-one"])
    with pytest.raises(StrictJSONError):
        _decode_tool_result(bad, "content_and_artifact")


# --------------------------------------------------------------------------- #
# Cloning, identity, wrap-time validation
# --------------------------------------------------------------------------- #


def test_clone_preserves_type_and_leaves_original_untouched():
    calls: list[str] = []
    original = _search(calls)
    (clone,) = wrap_langchain_tools([original], object())
    assert clone is not original
    assert type(clone) is type(original)
    assert clone.name == original.name
    assert clone.args_schema is original.args_schema
    assert clone.description == original.description
    # original still runs its real body
    assert original.invoke({"query": "z"}) == "result:z"
    assert calls == ["z"]


def test_supported_tool_families_wrap():
    t = Tool(name="t", description="d", func=lambda q: "r:" + q)
    st = StructuredTool.from_function(func=lambda query: "s:" + query, name="st", description="d")

    class CustomTool(BaseTool):
        name: str = "custom"
        description: str = "d"

        def _run(self, query: str, **kwargs: Any) -> str:
            return "c:" + query

    bridged = wrap_langchain_tools([t, st, CustomTool()], object())
    assert [type(b).__name__ for b in bridged] == ["Tool", "StructuredTool", "CustomTool"]


def test_duplicate_identity_returns_same_clone():
    original = _search([])
    a, b = wrap_langchain_tools([original, original], object())
    assert a is b


def test_same_cassette_rewrap_is_idempotent():
    cassette = object()
    (clone,) = wrap_langchain_tools([_search([])], cassette)
    (again,) = wrap_langchain_tools([clone], cassette)
    assert again is clone


def test_different_cassette_rewrap_rejected():
    (clone,) = wrap_langchain_tools([_search([])], object())
    with pytest.raises(ValueError, match="different cassette"):
        wrap_langchain_tools([clone], object())


@pytest.mark.parametrize("bad", ["notalist", iter([_search([])]), {"a": _search([])}])
def test_bad_sequence_rejected(bad):
    with pytest.raises(TypeError):
        wrap_langchain_tools(bad, object())


def test_non_basetool_entry_rejected():
    with pytest.raises(TypeError):
        wrap_langchain_tools([object()], object())


def test_bad_name_prefix_rejected():
    with pytest.raises(ValueError):
        wrap_langchain_tools([_search([])], object(), name_prefix="   ")


def test_custom_public_method_override_rejected():
    class OverridingTool(BaseTool):
        name: str = "over"
        description: str = "d"

        def _run(self, query: str, **kwargs: Any) -> str:
            return query

        def invoke(self, *args: Any, **kwargs: Any) -> Any:  # user override of a public method
            return "bypassed"

    with pytest.raises(TypeError, match="public"):
        wrap_langchain_tools([OverridingTool()], object())


def test_unsupported_response_format_rejected():
    tool_obj = _search([])
    # Pydantic enforces the response_format Literal, so bypass it to simulate an
    # unsupported value reaching the bridge.
    object.__setattr__(tool_obj, "response_format", "weird")
    with pytest.raises(ValueError):
        wrap_langchain_tools([tool_obj], object())


def test_non_str_response_format_rejected_without_running_its_dunders():
    class _HostileFormat:
        def __eq__(self, other):
            raise AssertionError("__eq__ called on response_format")

        def __repr__(self):
            raise AssertionError("__repr__ called on response_format")

        def __hash__(self):
            return 0

    tool_obj = _search([])
    object.__setattr__(tool_obj, "response_format", _HostileFormat())
    with pytest.raises(ValueError):
        wrap_langchain_tools([tool_obj], object())


# --------------------------------------------------------------------------- #
# Optional-import isolation
# --------------------------------------------------------------------------- #


def test_core_import_does_not_import_langchain_core():
    import subprocess
    import sys
    from textwrap import dedent

    code = dedent(
        """
        import sys
        import agent_cassette  # noqa: F401
        assert "langchain_core" not in sys.modules, "core import pulled in langchain_core"
        assert hasattr(agent_cassette, "wrap_langchain_tools")
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------- #
# Phase C2 correction — bridge identity, async fallback, error trust
# --------------------------------------------------------------------------- #

import functools  # noqa: E402


def test_forged_non_tool_rejected_without_getattr_or_marker_access():
    class _Forged:
        _agent_cassette_tool_bridge = "forged"

        def __getattr__(self, name):  # pragma: no cover - must never run
            raise AssertionError(f"__getattr__ ran: {name}")

    with pytest.raises(TypeError):
        wrap_langchain_tools([_Forged()], object())


def test_tampered_clone_rewrap_rejected():
    cassette = object()
    (clone,) = wrap_langchain_tools([_search([])], cassette)
    object.__setattr__(clone, "_run", lambda *a, **k: "tampered")  # replace the boundary
    with pytest.raises(ValueError, match="boundary was replaced"):
        wrap_langchain_tools([clone], cassette)


def test_forged_wrong_type_marker_rejected():
    tool_obj = _search([])
    object.__setattr__(tool_obj, "_agent_cassette_tool_bridge", "not-a-state")
    with pytest.raises(ValueError, match="forged or collided"):
        wrap_langchain_tools([tool_obj], object())


def test_spoofed_public_override_rejected():
    class _Spoofed(BaseTool):
        name: str = "spoof"
        description: str = "d"

        def _run(self, query: str, **kwargs: Any) -> str:
            return query

        @functools.wraps(BaseTool.invoke)  # copies __module__/__wrapped__ from the SDK
        def invoke(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return "bypassed"

    with pytest.raises(TypeError, match="public"):
        wrap_langchain_tools([_Spoofed()], object())


class _StrKey(str):
    pass


def test_config_and_kwargs_key_subclass_rejected():
    with pytest.raises(StrictJSONError):
        _clean_config({_StrKey("callbacks"): 1})
    with pytest.raises(StrictJSONError):
        _build_request((), {_StrKey("query"): 1})


_CUSTOM_STATE: dict[str, Any] = {"calls": [], "forbidden": False}


class _CustomSyncTool(BaseTool):
    name: str = "custom_sync"
    description: str = "d"

    def _run(self, query: str, run_manager: Any = None, **kwargs: Any) -> str:
        _CUSTOM_STATE["calls"].append(query)
        # A declared run_manager must still be injected through the wrapper's signature.
        _CUSTOM_STATE["got_run_manager"] = run_manager is not None
        if _CUSTOM_STATE["forbidden"]:
            raise AssertionError("live custom tool executed during replay")
        return "custom:" + query


def test_custom_sync_tool_ainvoke_one_event_zero_live(tmp_path):
    path = tmp_path / "c.jsonl"
    _CUSTOM_STATE["calls"] = []
    _CUSTOM_STATE["forbidden"] = False

    async def scenario():
        with Cassette.record(path) as cassette:
            (bridged,) = wrap_langchain_tools([_CustomSyncTool()], cassette)
            rec = await bridged.ainvoke({"query": "agents"})  # async fallback into sync _run
        assert _CUSTOM_STATE["got_run_manager"] is True  # signature-driven injection preserved
        # exactly one event: no inner+outer double record
        assert len([e for e in load_events(path) if e.type == EventType.TOOL_CALL]) == 1
        _CUSTOM_STATE["forbidden"] = True
        with Cassette.replay(path) as replayer:
            (bridged,) = wrap_langchain_tools([_CustomSyncTool()], replayer)
            rep = await bridged.ainvoke({"query": "agents"})
            assert replayer.remaining == 0
        return rec, rep

    rec, rep = asyncio.run(scenario())
    assert rec == rep == "custom:agents"


def test_lookalike_tool_exception_stays_recorded_call_error(tmp_path):
    path = tmp_path / "c.jsonl"

    class ToolException(Exception):  # user lookalike, NOT the SDK class
        pass

    @tool
    def boom(query: str) -> str:
        "boom"
        raise ToolException("lookalike")

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([boom], cassette)
        with pytest.raises(ToolException):
            bridged.invoke({"query": "x"})

    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([boom], replayer)
        with pytest.raises(RecordedCallError):  # not reconstructed as the SDK ToolException
            bridged.invoke({"query": "x"})


def test_two_turn_loop_zero_live_all_consumed(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        first = bridged.invoke({"query": "a"})
        second = bridged.invoke({"query": "b"})
    assert calls == ["a", "b"]

    replay_calls: list[str] = []
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search(replay_calls, forbidden=True)], replayer)
        rfirst = bridged.invoke({"query": "a"})
        rsecond = bridged.invoke({"query": "b"})
        assert replayer.remaining == 0
    assert (first, second) == (rfirst, rsecond) == ("result:a", "result:b")
    assert replay_calls == []


def test_sequential_batch_record_replay(tmp_path):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        rec = bridged.batch([{"query": "a"}, {"query": "b"}], config={"max_concurrency": 1})
    assert rec == ["result:a", "result:b"]
    with Cassette.replay(path, strict=False) as replayer:
        (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
        rep = bridged.batch([{"query": "a"}, {"query": "b"}], config={"max_concurrency": 1})
        assert replayer.remaining == 0
    assert rep == ["result:a", "result:b"]


def test_hard_zero_live_with_socket_disabled(tmp_path, monkeypatch):
    import socket

    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        bridged.invoke({"query": "agents"})

    def _blocked(*args, **kwargs):
        raise RuntimeError("network egress disabled")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    replay_calls: list[str] = []
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search(replay_calls, forbidden=True)], replayer)
        result = bridged.invoke({"query": "agents"})
        assert replayer.remaining == 0
    assert result == "result:agents"
    assert replay_calls == []


def test_nested_leaf_ordering_mismatch_fails_closed(tmp_path):
    # An orchestration tool that calls a bridged leaf inside its own body records
    # the leaf event BEFORE the outer result; bridging the outer body would consume
    # events out of order. We leave orchestration live and only bridge the leaf; a
    # bridged outer body must fail closed rather than scan past the nested event.
    path = tmp_path / "c.jsonl"
    leaf_calls: list[str] = []

    with Cassette.record(path) as cassette:
        (leaf,) = wrap_langchain_tools([_search(leaf_calls)], cassette)

        @tool
        def outer(query: str) -> str:
            "orchestration tool (must stay live)"
            inner = leaf.invoke({"query": query})
            return "outer:" + inner

        # If a user WRONGLY bridges the orchestration tool too:
        (bridged_outer,) = wrap_langchain_tools([outer], cassette)
        bridged_outer.invoke({"query": "x"})

    # On replay the outer bridge tries to consume its TOOL_CALL first, but the
    # recorded order is leaf-then-outer, so it fails closed (mismatch), not scan-ahead.
    with pytest.raises(ReplayMismatchError):
        with Cassette.replay(path) as replayer:
            (leaf,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)

            @tool
            def outer(query: str) -> str:
                "orchestration"
                inner = leaf.invoke({"query": query})
                return "outer:" + inner

            (bridged_outer,) = wrap_langchain_tools([outer], replayer)
            bridged_outer.invoke({"query": "x"})


def test_real_callback_lifecycle_counts(tmp_path):
    from langchain_core.callbacks import BaseCallbackHandler

    class _Counter(BaseCallbackHandler):
        def __init__(self):
            self.starts = 0
            self.ends = 0

        def on_tool_start(self, *args, **kwargs):
            self.starts += 1

        def on_tool_end(self, *args, **kwargs):
            self.ends += 1

    path = tmp_path / "c.jsonl"
    rec_cb = _Counter()
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search([])], cassette)
        bridged.invoke({"query": "agents"}, config={"callbacks": [rec_cb]})
    assert (rec_cb.starts, rec_cb.ends) == (1, 1)

    rep_cb = _Counter()
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
        bridged.invoke({"query": "agents"}, config={"callbacks": [rep_cb]})
        assert replayer.remaining == 0
    assert (rep_cb.starts, rep_cb.ends) == (1, 1)  # callbacks run once on replay too


# --------------------------------------------------------------------------- #
# Phase C2 final correction — default _arun integrity + missing gates
# --------------------------------------------------------------------------- #

from types import MethodType  # noqa: E402


def test_tampered_default_arun_rewrap_rejected():
    # sync-only tool -> default _arun path (arun_wrapper None); an instance _arun
    # installed after bridging must be rejected, not silently ignored.
    cassette = object()
    (clone,) = wrap_langchain_tools([_CustomSyncTool()], cassette)

    async def evil(*args, **kwargs):
        return "bypass"

    object.__setattr__(clone, "_arun", MethodType(evil, clone))
    with pytest.raises(ValueError, match="_arun"):
        wrap_langchain_tools([clone], cassette)


def test_tampered_installed_arun_rewrap_rejected():
    cassette = object()

    @tool
    async def atool(query: str) -> str:
        "a"
        return query

    (clone,) = wrap_langchain_tools([atool], cassette)  # async tool -> arun_wrapper installed

    async def evil(*args, **kwargs):
        return "bypass"

    object.__setattr__(clone, "_arun", MethodType(evil, clone))
    with pytest.raises(ValueError, match="_arun"):
        wrap_langchain_tools([clone], cassette)


def test_none_marker_collision_rejected():
    tool_obj = _search([])
    object.__setattr__(tool_obj, "_agent_cassette_tool_bridge", None)  # collision, not absence
    with pytest.raises(ValueError, match="forged or collided"):
        wrap_langchain_tools([tool_obj], object())


def test_intact_idempotent_rewrap_both_async_shapes():
    cassette = object()
    (sync_clone,) = wrap_langchain_tools([_CustomSyncTool()], cassette)
    assert wrap_langchain_tools([sync_clone], cassette)[0] is sync_clone

    @tool
    async def atool(query: str) -> str:
        "a"
        return query

    (async_clone,) = wrap_langchain_tools([atool], cassette)
    assert wrap_langchain_tools([async_clone], cassette)[0] is async_clone


def test_fake_agent_loop_record_replay(tmp_path):
    from langchain_core.messages import ToolMessage

    path = tmp_path / "c.jsonl"

    def run_loop(tools, forbidden):
        by_name = {t.name: t for t in tools}
        # fake model: one tool call, then a final answer built from the ToolMessage
        tool_call = {
            "name": "search",
            "args": {"query": "agents"},
            "id": "call_1",
            "type": "tool_call",
        }
        message = by_name["search"].invoke(tool_call)
        assert isinstance(message, ToolMessage)
        return {"final": "answer:" + str(message.content), "messages": [message.content]}

    calls: list[str] = []
    with Cassette.record(path) as cassette:
        rec = run_loop(wrap_langchain_tools([_search(calls)], cassette), forbidden=False)
    assert calls == ["agents"]

    replay_calls: list[str] = []
    with Cassette.replay(path) as replayer:
        rep = run_loop(
            wrap_langchain_tools([_search(replay_calls, forbidden=True)], replayer), True
        )
        assert replayer.remaining == 0
    assert rec == rep
    assert replay_calls == []


def test_sequential_abatch_record_replay(tmp_path):
    path = tmp_path / "c.jsonl"

    async def scenario():
        calls: list[str] = []
        with Cassette.record(path) as cassette:
            (bridged,) = wrap_langchain_tools([_async_search(calls)], cassette)
            rec = await bridged.abatch(
                [{"query": "a"}, {"query": "b"}], config={"max_concurrency": 1}
            )
        with Cassette.replay(path, strict=False) as replayer:
            (bridged,) = wrap_langchain_tools([_async_search([], forbidden=True)], replayer)
            rep = await bridged.abatch(
                [{"query": "a"}, {"query": "b"}], config={"max_concurrency": 1}
            )
            assert replayer.remaining == 0
        return rec, rep

    rec, rep = asyncio.run(scenario())
    assert rec == rep == ["async:a", "async:b"]


def test_coexists_with_langchain_callback_handler(tmp_path):
    from agent_cassette import langchain_callback_handler

    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    with Cassette.record(path) as cassette:
        handler = langchain_callback_handler(cassette)
        (bridged,) = wrap_langchain_tools([_search(calls)], cassette)
        bridged.invoke({"query": "agents"}, config={"callbacks": [handler]})

    events = load_events(path)
    # the bridged TOOL_CALL is replayable; any observational callback event is marked
    bridged_events = [e for e in events if e.metadata.get("tool_bridge")]
    assert len(bridged_events) == 1
    observational = [
        e for e in events if e.metadata.get("_agent_cassette", {}).get("observational") is True
    ]
    # observational events (if the handler emitted any) are filtered from strict replay
    with Cassette.replay(path) as replayer:
        assert replayer.remaining == len(events) - len(observational)
        (bridged,) = wrap_langchain_tools([_search([], forbidden=True)], replayer)
        bridged.invoke({"query": "agents"})
        assert replayer.remaining == 0


def test_invalid_output_persists_no_success_event(tmp_path):
    path = tmp_path / "c.jsonl"

    @tool
    def bad(query: str):
        "returns a tuple, which is not JSON-native for a content tool"
        return (1, 2)

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([bad], cassette)
        with pytest.raises(StrictJSONError):
            bridged.invoke({"query": "x"})

    events = load_events(path)
    assert not any(e.type == EventType.TOOL_CALL and e.metadata.get("tool_bridge") for e in events)
    # nor a replayable error event
    assert not any(e.type == EventType.ERROR for e in events)


def test_artifact_result_detached_from_loaded_event(tmp_path):
    path = tmp_path / "c.jsonl"
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_artifact_tool([])], cassette)
        bridged.invoke({"args": {"query": "x"}, "type": "tool_call", "id": "1", "name": "build"})
    with Cassette.replay(path) as replayer:
        (bridged,) = wrap_langchain_tools([_artifact_tool([], forbidden=True)], replayer)
        message = bridged.invoke(
            {"args": {"query": "x"}, "type": "tool_call", "id": "1", "name": "build"}
        )
        message.artifact["artifact"].append("MUTATED")
        stored = replayer.events[0].output["value"]  # [content, artifact]
        assert stored[1] == {"artifact": [1, 2, 3]}


@pytest.mark.parametrize(
    "bad_return,response_format",
    [
        ((1, 2), "content"),  # tuple not JSON-native for content
        (["only-one"], "content_and_artifact"),  # wrong tuple shape
        (("c", {"a": 1}, "extra"), "content_and_artifact"),  # 3-tuple
        (float("nan"), "content"),
        (_Grade(3), "content"),
    ],
)
def test_encode_invalid_results_rejected(bad_return, response_format):
    with pytest.raises(StrictJSONError):
        _encode_tool_result(bad_return, response_format)


def test_encode_cyclic_and_deep_content_rejected():
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(StrictJSONError):
        _encode_tool_result(cycle, "content")
    deep: Any = 1
    for _ in range(100):
        deep = [deep]
    with pytest.raises(StrictJSONError):
        _encode_tool_result(deep, "content")


def test_config_captured_and_cleaned_in_request(tmp_path):
    path = tmp_path / "c.jsonl"
    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_search([])], cassette)
        bridged.invoke({"query": "x"}, config={"tags": ["t"], "metadata": {"k": 1}})
    request = load_events(path)[0].input
    # Volatile keys (callbacks/run_id/run_name) are dropped; behaviour keys retained.
    assert request["config"].get("tags") == ["t"]
    assert request["config"].get("metadata") == {"k": 1}
    assert "callbacks" not in request["config"]
    assert "run_id" not in request["config"]


def test_custom_basetool_run_manager_injection(tmp_path):
    # A conventional custom BaseTool declaring run_manager still gets it injected
    # through the wrapper's preserved signature.
    path = tmp_path / "c.jsonl"
    seen: dict[str, Any] = {}

    class _RunManagerTool(BaseTool):
        name: str = "rm"
        description: str = "d"

        def _run(self, query: str, run_manager: Any = None, **kwargs: Any) -> str:
            seen["run_manager"] = run_manager is not None
            return "rm:" + query

    with Cassette.record(path) as cassette:
        (bridged,) = wrap_langchain_tools([_RunManagerTool()], cassette)
        bridged.invoke({"query": "x"})
    assert seen["run_manager"] is True
