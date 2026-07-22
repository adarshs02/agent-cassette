"""Phase C1 — OpenAI Agents FunctionTool replay bridge.

Uses the installed SDK's real ``FunctionTool``/``function_tool`` with a deterministic
fake ``Runner`` so no network call or credential is needed. Skips cleanly when the
optional ``agents`` extra is absent.
"""

from __future__ import annotations

import asyncio

import pytest

agents = pytest.importorskip("agents")

from agents import FunctionTool, function_tool  # noqa: E402
from agents.tool import (  # noqa: E402
    ToolOutputFileContent,
    ToolOutputImage,
    ToolOutputText,
)

from agent_cassette import Cassette, EventType, InjectionRule, Raise, Return  # noqa: E402
from agent_cassette.integrations.openai_agents import (  # noqa: E402
    _AgentsToolBridgeState,
    _decode_agents_tool_result,
    _encode_agents_tool_result,
    patch_openai_agents,
)
from agent_cassette.json_codec import StrictJSONError  # noqa: E402
from agent_cassette.storage import load_events  # noqa: E402

_STRUCTURED = (ToolOutputText, ToolOutputImage, ToolOutputFileContent)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class _Ctx:
    def __init__(self, agent, tool, call_id, arguments):
        self.tool_call_id = call_id
        self.tool_name = getattr(tool, "name", None)
        self.tool_arguments = arguments
        self.agent = agent
        self.context = None


class _Agent:
    def __init__(self, name, tools, handoff_to=None):
        self.name = name
        self.tools = list(tools)
        self.handoff_to = handoff_to


def _install_runner(monkeypatch, *, run=None, run_sync=None, run_streamed=None):
    namespace: dict[str, object] = {}
    if run is not None:
        namespace["run"] = staticmethod(run)
    if run_sync is not None:
        namespace["run_sync"] = staticmethod(run_sync)
    if run_streamed is not None:
        namespace["run_streamed"] = staticmethod(run_streamed)
    fake_runner = type("_FakeRunner", (), namespace)
    monkeypatch.setattr(agents, "Runner", fake_runner, raising=False)
    return fake_runner


async def _drive_single_tool(agent, prompt, *, hooks=None, call_id="call_1", arguments=None):
    """One tool-call turn: start, llm, tool_start, invoke, tool_end, agent_end."""
    tool = agent.tools[0]
    arguments = arguments if arguments is not None else '{"query":"agents"}'
    ctx = _Ctx(agent, tool, call_id, arguments)
    await hooks.on_agent_start(ctx, agent)
    await hooks.on_llm_start(ctx, agent, "system", [{"role": "user", "content": prompt}])
    await hooks.on_llm_end(ctx, agent, {"call": tool.name})
    await hooks.on_tool_start(ctx, agent, tool)
    result = await tool.on_invoke_tool(ctx, arguments)
    await hooks.on_tool_end(ctx, agent, tool, result)
    await hooks.on_agent_end(ctx, agent, "done")
    return result


def _search_tool(counter):
    @function_tool
    def search(query: str) -> dict:
        "search docs"
        counter.append(query)
        return {"query": query, "hits": ["a", "b"]}

    return search


# --------------------------------------------------------------------------- #
# Envelope codec (unit)
# --------------------------------------------------------------------------- #


def test_encode_decode_json_round_trip():
    envelope = _encode_agents_tool_result({"ok": True, "n": [1, 2]}, _STRUCTURED)
    assert envelope["__agent_cassette_openai_agents_tool_result__"] is True
    assert envelope["version"] == 1
    assert envelope["kind"] == "json"
    assert _decode_agents_tool_result(envelope, _STRUCTURED) == {"ok": True, "n": [1, 2]}


def test_encode_decode_structured_round_trip():
    envelope = _encode_agents_tool_result(ToolOutputText(text="hi"), _STRUCTURED)
    assert envelope["kind"] == "structured"
    restored = _decode_agents_tool_result(envelope, _STRUCTURED)
    assert isinstance(restored, ToolOutputText)
    assert restored.text == "hi"


def test_encode_decode_structured_list_round_trip():
    envelope = _encode_agents_tool_result(
        [ToolOutputText(text="a"), ToolOutputImage(image_url="http://x/y.png")], _STRUCTURED
    )
    assert envelope["kind"] == "structured_list"
    restored = _decode_agents_tool_result(envelope, _STRUCTURED)
    assert [type(item) for item in restored] == [ToolOutputText, ToolOutputImage]


def test_empty_list_encodes_as_json():
    envelope = _encode_agents_tool_result([], _STRUCTURED)
    assert envelope["kind"] == "json"
    assert _decode_agents_tool_result(envelope, _STRUCTURED) == []


@pytest.mark.parametrize(
    "bad",
    [
        (1, 2),
        {1: "int key"},
        float("nan"),
        object(),
        ToolOutputText,  # the class, not an instance
    ],
)
def test_unsupported_json_output_rejected(bad):
    with pytest.raises(StrictJSONError):
        _encode_agents_tool_result(bad, _STRUCTURED)


def test_mixed_structured_and_json_list_rejected():
    with pytest.raises(StrictJSONError):
        _encode_agents_tool_result([ToolOutputText(text="a"), {"plain": 1}], _STRUCTURED)


def test_legacy_unmarked_output_decodes_as_json():
    assert _decode_agents_tool_result({"legacy": True}, _STRUCTURED) == {"legacy": True}
    assert _decode_agents_tool_result(["a", "b"], _STRUCTURED) == ["a", "b"]


def test_injected_structured_object_passthrough_on_decode():
    obj = ToolOutputText(text="injected")
    assert _decode_agents_tool_result(obj, _STRUCTURED) is obj


def test_injected_structured_list_passthrough_on_decode():
    objs = [ToolOutputText(text="a"), ToolOutputImage(image_url="http://x/y.png")]
    assert _decode_agents_tool_result(objs, _STRUCTURED) is objs


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update({"__agent_cassette_openai_agents_tool_result__": False}),
        lambda e: e.update({"version": 2}),
        lambda e: e.update({"kind": "bogus"}),
        lambda e: e.update({"extra": 1}),
    ],
)
def test_malformed_envelope_fails_closed(mutate):
    envelope = _encode_agents_tool_result({"ok": 1}, _STRUCTURED)
    mutate(envelope)
    with pytest.raises(StrictJSONError):
        _decode_agents_tool_result(envelope, _STRUCTURED)


def test_unknown_structured_discriminator_fails_without_import():
    envelope = {
        "__agent_cassette_openai_agents_tool_result__": True,
        "version": 1,
        "kind": "structured",
        "value": {"type": "nonexistent_kind", "text": "x"},
    }
    with pytest.raises(StrictJSONError):
        _decode_agents_tool_result(envelope, _STRUCTURED)


# --------------------------------------------------------------------------- #
# End-to-end record/replay
# --------------------------------------------------------------------------- #


def test_record_replay_two_turn_zero_live(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    recorded_calls: list[str] = []
    tool = _search_tool(recorded_calls)
    original_callback = tool.on_invoke_tool
    agent = _Agent("researcher", [tool])
    _install_runner(monkeypatch, run=_drive_single_tool)

    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        rec = asyncio.run(agents.Runner.run(agent, "find"))

    assert recorded_calls == ["agents"]
    assert rec == {"query": "agents", "hits": ["a", "b"]}
    # tool must be restored to its original callback after the context.
    assert tool.on_invoke_tool is original_callback

    replay_calls: list[str] = []
    replay_tool = _search_tool(replay_calls)
    replay_agent = _Agent("researcher", [replay_tool])

    def forbidden_runner(agent, prompt, *, hooks=None):
        return _drive_single_tool(agent, prompt, hooks=hooks)

    _install_runner(monkeypatch, run=forbidden_runner)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        rep = asyncio.run(agents.Runner.run(replay_agent, "find"))

    assert rep == {"query": "agents", "hits": ["a", "b"]}
    assert replay_calls == []  # live callback never executed on replay


def test_event_order_is_tool_call_then_bridged_tool_result(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _search_tool([])
    agent = _Agent("researcher", [tool])
    _install_runner(monkeypatch, run=_drive_single_tool)

    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "find"))

    events = load_events(path)
    types = [e.type for e in events]
    # exactly one TOOL_CALL immediately followed by exactly one TOOL_RESULT
    assert types.count(EventType.TOOL_RESULT) == 1
    call_index = types.index(EventType.TOOL_CALL)
    assert types[call_index + 1] == EventType.TOOL_RESULT
    result_event = next(e for e in events if e.type == EventType.TOOL_RESULT)
    assert result_event.metadata["tool_bridge"] is True
    assert result_event.metadata["stage"] == "invoke"
    assert result_event.input == {"agent": "researcher", "tool_call_id": "call_1"}


def test_replay_callback_never_awaited_sentinel(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _search_tool([])
    agent = _Agent("researcher", [tool])
    _install_runner(monkeypatch, run=_drive_single_tool)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "find"))

    @function_tool
    def forbidden(query: str) -> dict:
        "forbidden"
        raise AssertionError("live tool callback executed during replay")

    forbidden.name = "search"
    replay_agent = _Agent("researcher", [forbidden])
    _install_runner(monkeypatch, run=_drive_single_tool)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        result = asyncio.run(agents.Runner.run(replay_agent, "find"))
    assert result == {"query": "agents", "hits": ["a", "b"]}


def test_json_result_detached_from_loaded_event(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _search_tool([])
    agent = _Agent("researcher", [tool])
    _install_runner(monkeypatch, run=_drive_single_tool)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "find"))

    _install_runner(monkeypatch, run=_drive_single_tool)
    with Cassette.replay(path) as replayer, patch_openai_agents(replayer):
        result = asyncio.run(agents.Runner.run(_Agent("researcher", [tool]), "find"))
        result["hits"].append("MUTATED")
        stored = next(e for e in replayer.events if e.type == EventType.TOOL_RESULT)
        assert stored.output["value"] == {"query": "agents", "hits": ["a", "b"]}


# --------------------------------------------------------------------------- #
# Structured outputs end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory,check",
    [
        (
            lambda: ToolOutputText(text="hi"),
            lambda r: isinstance(r, ToolOutputText) and r.text == "hi",
        ),
        (
            lambda: ToolOutputImage(image_url="http://x/y.png"),
            lambda r: isinstance(r, ToolOutputImage),
        ),
        (
            lambda: ToolOutputFileContent(file_url="http://x/y.pdf"),
            lambda r: isinstance(r, ToolOutputFileContent),
        ),
    ],
)
def test_structured_output_round_trip(tmp_path, monkeypatch, factory, check):
    path = tmp_path / "c.jsonl"
    output = factory()

    @function_tool
    def make(msg: str):
        "structured"
        return output

    seen: list = []

    async def drive(agent, prompt, *, hooks=None):
        tool = agent.tools[0]
        ctx = _Ctx(agent, tool, "call_1", '{"msg":"x"}')
        await hooks.on_agent_start(ctx, agent)
        await hooks.on_tool_start(ctx, agent, tool)
        result = await tool.on_invoke_tool(ctx, '{"msg":"x"}')
        await hooks.on_tool_end(ctx, agent, tool, result)
        seen.append(result)
        await hooks.on_agent_end(ctx, agent, "done")
        return result

    agent = _Agent("a", [make])
    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))

    _install_runner(monkeypatch, run=drive)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        rep = asyncio.run(agents.Runner.run(agent, "go"))
    assert check(rep)


def test_structured_list_output_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"

    @function_tool
    def make(msg: str):
        "structured list"
        return [ToolOutputText(text="a"), ToolOutputText(text="b")]

    async def drive(agent, prompt, *, hooks=None):
        tool = agent.tools[0]
        ctx = _Ctx(agent, tool, "call_1", '{"msg":"x"}')
        await hooks.on_tool_start(ctx, agent, tool)
        result = await tool.on_invoke_tool(ctx, '{"msg":"x"}')
        await hooks.on_tool_end(ctx, agent, tool, result)
        return result

    agent = _Agent("a", [make])
    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))
    _install_runner(monkeypatch, run=drive)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        rep = asyncio.run(agents.Runner.run(agent, "go"))
    assert [type(x) for x in rep] == [ToolOutputText, ToolOutputText]
    assert [x.text for x in rep] == ["a", "b"]


def test_unsupported_output_persists_no_success_event(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"

    @function_tool
    def make(msg: str):
        "unsupported"
        return (1, 2)  # tuple: not JSON-native, not structured

    async def drive(agent, prompt, *, hooks=None):
        tool = agent.tools[0]
        ctx = _Ctx(agent, tool, "call_1", '{"msg":"x"}')
        await hooks.on_tool_start(ctx, agent, tool)
        with pytest.raises(StrictJSONError):
            await tool.on_invoke_tool(ctx, '{"msg":"x"}')

    agent = _Agent("a", [make])
    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))

    events = load_events(path)
    assert not any(e.type == EventType.TOOL_RESULT for e in events)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


def test_allowlisted_tool_exception_replays(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"

    @function_tool(failure_error_function=None)
    def boom(query: str) -> str:
        "boom"
        raise ValueError("tool exploded")

    async def drive(agent, prompt, *, hooks=None):
        tool = agent.tools[0]
        ctx = _Ctx(agent, tool, "call_1", '{"query":"x"}')
        await hooks.on_tool_start(ctx, agent, tool)
        with pytest.raises(ValueError, match="tool exploded"):
            await tool.on_invoke_tool(ctx, '{"query":"x"}')

    agent = _Agent("a", [boom])
    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))

    ran: list[bool] = []

    @function_tool(failure_error_function=None)
    def forbidden(query: str) -> str:
        "forbidden"
        ran.append(True)
        raise AssertionError("live executed on replay")

    forbidden.name = "boom"

    async def replay_drive(agent, prompt, *, hooks=None):
        tool = agent.tools[0]
        ctx = _Ctx(agent, tool, "call_1", '{"query":"x"}')
        await hooks.on_tool_start(ctx, agent, tool)
        with pytest.raises(ValueError, match="tool exploded"):
            await tool.on_invoke_tool(ctx, '{"query":"x"}')

    _install_runner(monkeypatch, run=replay_drive)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        asyncio.run(agents.Runner.run(_Agent("a", [forbidden]), "go"))
    assert ran == []


# --------------------------------------------------------------------------- #
# Non-FunctionTool families and lifecycle
# --------------------------------------------------------------------------- #


class _CustomTool:
    """A local tool family that is not an SDK FunctionTool."""

    def __init__(self, name):
        self.name = name


def test_non_functiontool_retains_lifecycle_capture(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _CustomTool("lookup")
    agent = _Agent("a", [tool])

    async def drive(agent, prompt, *, hooks=None):
        ctx = _Ctx(agent, tool, "call_1", "{}")
        await hooks.on_tool_start(ctx, agent, tool)
        await hooks.on_tool_end(ctx, agent, tool, {"raw": "result"})

    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))

    events = load_events(path)
    result_events = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert len(result_events) == 1
    # lifecycle metadata, not tool_bridge
    assert result_events[0].metadata.get("lifecycle") is True
    assert "tool_bridge" not in result_events[0].metadata


# --------------------------------------------------------------------------- #
# User hooks, wrapping, restoration
# --------------------------------------------------------------------------- #


def test_user_hooks_receive_events_and_late_added_tool_is_bridged(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    tool = _search_tool(calls)
    agent = _Agent("a", [])  # starts with no tools

    class UserHooks:
        def __init__(self):
            self.starts = 0
            self.ends = 0

        async def on_agent_start(self, context, agent):
            self.starts += 1
            agent.tools.append(tool)  # add a tool during on_agent_start

        async def on_tool_end(self, context, agent, tool, result):
            self.ends += 1

    user = UserHooks()

    async def drive(agent, prompt, *, hooks=None):
        await hooks.on_agent_start(_Ctx(agent, None, None, None), agent)
        t = agent.tools[0]
        ctx = _Ctx(agent, t, "call_1", '{"query":"agents"}')
        await hooks.on_tool_start(ctx, agent, t)
        result = await t.on_invoke_tool(ctx, '{"query":"agents"}')
        await hooks.on_tool_end(ctx, agent, t, result)
        return result

    _install_runner(monkeypatch, run=lambda a, p, *, hooks=None: drive(a, p, hooks=hooks))
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go", hooks=user))

    assert user.starts == 1
    assert user.ends == 1  # user still receives on_tool_end
    events = load_events(path)
    # the late-added tool was bridged: exactly one bridged TOOL_RESULT
    bridged = [
        e for e in events if e.type == EventType.TOOL_RESULT and e.metadata.get("tool_bridge")
    ]
    assert len(bridged) == 1


def test_handoff_destination_tool_is_bridged(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    calls: list[str] = []
    dest_tool = _search_tool(calls)
    source = _Agent("source", [])
    dest = _Agent("dest", [dest_tool])

    async def drive(agent, prompt, *, hooks=None):
        await hooks.on_agent_start(_Ctx(agent, None, None, None), agent)
        await hooks.on_handoff(_Ctx(agent, None, None, None), source, dest)
        t = dest.tools[0]
        ctx = _Ctx(dest, t, "call_1", '{"query":"agents"}')
        await hooks.on_tool_start(ctx, dest, t)
        result = await t.on_invoke_tool(ctx, '{"query":"agents"}')
        await hooks.on_tool_end(ctx, dest, t, result)
        return result

    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(source, "go"))

    events = load_events(path)
    bridged = [
        e for e in events if e.type == EventType.TOOL_RESULT and e.metadata.get("tool_bridge")
    ]
    assert len(bridged) == 1


def test_shared_tool_wrapped_once():
    tool = _search_tool([])
    agent_a = _Agent("a", [tool])
    agent_b = _Agent("b", [tool])
    bridge = _AgentsToolBridgeState(
        cassette=object(), function_tool_cls=FunctionTool, structured_types=()
    )
    bridge.patch_agent(agent_a)
    wrapper = tool.on_invoke_tool
    bridge.patch_agent(agent_b)
    assert tool.on_invoke_tool is wrapper  # not re-wrapped
    assert bridge.is_bridged(tool)
    bridge.restore_all()
    assert not bridge.is_bridged(tool)


def test_callbacks_and_runner_restored_after_exceptional_exit(tmp_path, monkeypatch):
    tool = _search_tool([])
    original_callback = tool.on_invoke_tool
    agent = _Agent("a", [tool])

    async def drive(agent, prompt, *, hooks=None):
        await hooks.on_agent_start(_Ctx(agent, None, None, None), agent)
        raise RuntimeError("boom during run")

    runner = _install_runner(monkeypatch, run=drive)
    original_run = runner.run
    path = tmp_path / "c.jsonl"
    with pytest.raises(RuntimeError, match="boom during run"):
        with Cassette.record(path) as cassette, patch_openai_agents(cassette):
            asyncio.run(agents.Runner.run(agent, "go"))

    assert tool.on_invoke_tool is original_callback  # restored despite exception
    assert runner.run is original_run  # Runner descriptor restored


def test_user_replaced_callback_not_overwritten_and_still_lifecycle():
    tool = _search_tool([])
    agent = _Agent("a", [tool])
    bridge = _AgentsToolBridgeState(
        cassette=object(), function_tool_cls=FunctionTool, structured_types=()
    )
    bridge.patch_agent(agent)

    async def user_replacement(ctx, arguments):
        return "user"

    tool.on_invoke_tool = user_replacement  # user swaps after install
    assert not bridge.is_bridged(tool)  # bridge no longer owns the callback
    bridge.restore_all()
    assert tool.on_invoke_tool is user_replacement  # not overwritten


# --------------------------------------------------------------------------- #
# Concurrency, run_sync, run_streamed
# --------------------------------------------------------------------------- #


def test_concurrent_calls_unique_ids_non_strict(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _search_tool([])
    agent = _Agent("a", [tool])

    async def drive(agent, prompt, *, hooks=None):
        async def one(call_id, q):
            ctx = _Ctx(agent, tool, call_id, f'{{"query":"{q}"}}')
            await hooks.on_tool_start(ctx, agent, tool)
            result = await tool.on_invoke_tool(ctx, f'{{"query":"{q}"}}')
            await hooks.on_tool_end(ctx, agent, tool, result)
            return result

        return await asyncio.gather(one("c1", "x"), one("c2", "y"))

    _install_runner(monkeypatch, run=drive)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(agent, "go"))

    _install_runner(monkeypatch, run=drive)
    with Cassette.replay(path, strict=False) as replayer, patch_openai_agents(replayer):
        results = asyncio.run(agents.Runner.run(_Agent("a", [tool]), "go"))
        assert replayer.remaining == 0
    assert {r["query"] for r in results} == {"x", "y"}


def test_run_sync_and_run_streamed_paths(tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    tool = _search_tool([])
    agent = _Agent("a", [tool])

    def run_sync(agent, prompt, *, hooks=None):
        return asyncio.run(_drive_single_tool(agent, prompt, hooks=hooks))

    class _Streaming:
        def __init__(self, agent, prompt, hooks):
            self._agent, self._prompt, self._hooks = agent, prompt, hooks

        async def consume(self):
            return await _drive_single_tool(self._agent, self._prompt, hooks=self._hooks)

    def run_streamed(agent, prompt, *, hooks=None):
        return _Streaming(agent, prompt, hooks)

    _install_runner(monkeypatch, run_sync=run_sync, run_streamed=run_streamed)
    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        sync_result = agents.Runner.run_sync(agent, "go")
        streamed = agents.Runner.run_streamed(agent, "go2")
        streamed_result = asyncio.run(streamed.consume())

    assert sync_result == {"query": "agents", "hits": ["a", "b"]}
    assert streamed_result == {"query": "agents", "hits": ["a", "b"]}
    assert sum(1 for e in load_events(path) if e.type == EventType.TOOL_RESULT) == 2


# --------------------------------------------------------------------------- #
# Hybrid injection on tool_result
# --------------------------------------------------------------------------- #


def test_hybrid_return_injection_on_tool_result(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls: list[str] = []
    tool = _search_tool(calls)
    agent = _Agent("a", [tool])
    rule = InjectionRule(Return({"injected": True}), event_type="tool_result", name="search")

    _install_runner(monkeypatch, run=_drive_single_tool)
    with (
        Cassette.fork(source, output, injections=(rule,)) as cassette,
        patch_openai_agents(cassette),
    ):
        result = asyncio.run(agents.Runner.run(agent, "go"))

    assert result == {"injected": True}
    assert calls == []  # live tool body skipped by injection


def test_hybrid_raise_injection_on_tool_result(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls: list[str] = []
    tool = _search_tool(calls)
    agent = _Agent("a", [tool])
    rule = InjectionRule(Raise(TimeoutError("slow")), event_type="tool_result", name="search")

    _install_runner(monkeypatch, run=_drive_single_tool)
    with pytest.raises(TimeoutError, match="slow"):
        with (
            Cassette.fork(source, output, injections=(rule,)) as cassette,
            patch_openai_agents(cassette),
        ):
            asyncio.run(agents.Runner.run(agent, "go"))
    assert calls == []


# --------------------------------------------------------------------------- #
# Agent-as-tool exclusion
# --------------------------------------------------------------------------- #


def test_agent_as_tool_not_bridged():
    tool = _search_tool([])
    tool._is_agent_tool = True  # mark as agent-as-tool
    agent = _Agent("a", [tool])
    bridge = _AgentsToolBridgeState(
        cassette=object(), function_tool_cls=FunctionTool, structured_types=()
    )
    bridge.patch_agent(agent)
    assert not bridge.is_bridged(tool)


def test_agent_as_tool_fallback_flag():
    tool = _search_tool([])
    # No explicit _is_agent_tool flag, but an agent instance is present.
    tool._is_agent_tool = None
    tool._agent_instance = object()
    agent = _Agent("a", [tool])
    bridge = _AgentsToolBridgeState(
        cassette=object(), function_tool_cls=FunctionTool, structured_types=()
    )
    bridge.patch_agent(agent)
    assert not bridge.is_bridged(tool)


# --------------------------------------------------------------------------- #
# Legacy pre-C1 cassette replay
# --------------------------------------------------------------------------- #


def test_legacy_unmarked_tool_result_replays(tmp_path, monkeypatch):
    path = tmp_path / "legacy.jsonl"
    # Seed a pre-C1 lifecycle cassette: on_tool_start wrote TOOL_CALL, on_tool_end
    # wrote an unmarked TOOL_RESULT (raw value, no C1 envelope).
    with Cassette.record(path) as cassette:
        cassette.call(
            EventType.TOOL_CALL,
            "search",
            {"agent": "a", "arguments": '{"query":"agents"}', "tool_call_id": "call_1"},
            lambda: None,
            metadata={"provider": "openai-agents", "lifecycle": True},
        )
        cassette.call(
            EventType.TOOL_RESULT,
            "search",
            {"agent": "a", "tool_call_id": "call_1"},
            lambda: {"query": "agents", "hits": ["a", "b"]},
            metadata={"provider": "openai-agents", "lifecycle": True},
        )

    calls: list[str] = []
    tool = _search_tool(calls)
    tool.name = "search"
    agent = _Agent("a", [tool])

    async def tool_only_drive(agent, prompt, *, hooks=None):
        t = agent.tools[0]
        ctx = _Ctx(agent, t, "call_1", '{"query":"agents"}')
        await hooks.on_tool_start(ctx, agent, t)
        result = await t.on_invoke_tool(ctx, '{"query":"agents"}')
        await hooks.on_tool_end(ctx, agent, t, result)
        return result

    _install_runner(monkeypatch, run=tool_only_drive)
    with Cassette.replay(path) as replaying, patch_openai_agents(replaying):
        result = asyncio.run(agents.Runner.run(agent, "find"))

    assert result == {"query": "agents", "hits": ["a", "b"]}
    assert calls == []  # live callback never executed on legacy replay


def test_hybrid_return_injection_structured_list_on_tool_result(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    calls: list[str] = []
    tool = _search_tool(calls)
    agent = _Agent("a", [tool])
    injected = [ToolOutputText(text="a"), ToolOutputText(text="b")]
    rule = InjectionRule(Return(injected), event_type="tool_result", name="search")

    _install_runner(monkeypatch, run=_drive_single_tool)
    with (
        Cassette.fork(source, output, injections=(rule,)) as cassette,
        patch_openai_agents(cassette),
    ):
        result = asyncio.run(agents.Runner.run(agent, "go"))

    assert [type(item) for item in result] == [ToolOutputText, ToolOutputText]
    assert [item.text for item in result] == ["a", "b"]
    assert calls == []  # live tool body skipped by injection

    # The forked cassette stored a structured_list envelope; it replays to SDK types.
    _install_runner(monkeypatch, run=_drive_single_tool)
    with Cassette.replay(output) as replaying, patch_openai_agents(replaying):
        replayed = asyncio.run(agents.Runner.run(_Agent("a", [tool]), "go"))
    assert [type(item) for item in replayed] == [ToolOutputText, ToolOutputText]
