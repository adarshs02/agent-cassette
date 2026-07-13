from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass

from agent_cassette import Cassette, EventType
from agent_cassette.integrations.openai_agents import patch_openai_agents
from agent_cassette.storage import load_events


@dataclass
class _Named:
    name: str


class _ToolContext:
    tool_arguments = '{"query":"agents"}'
    tool_call_id = "call_1"


class _ExistingHooks:
    def __init__(self) -> None:
        self.started = False

    async def on_agent_start(self, context, agent):
        self.started = True


class _Runner:
    @staticmethod
    async def run(agent, prompt, *, hooks=None):
        context = _ToolContext()
        tool = _Named("search")
        await hooks.on_agent_start(context, agent)
        await hooks.on_llm_start(context, agent, "system", [{"role": "user", "content": prompt}])
        await hooks.on_llm_end(context, agent, {"output": "call search"})
        await hooks.on_tool_start(context, agent, tool)
        await hooks.on_tool_end(context, agent, tool, ["result"])
        await hooks.on_handoff(context, agent, _Named("writer"))
        await hooks.on_agent_end(context, agent, "done")
        return "done"


def test_openai_agents_runner_hooks_are_injected_and_restored(tmp_path, monkeypatch):
    agents = types.ModuleType("agents")
    agents.Runner = _Runner
    monkeypatch.setitem(sys.modules, "agents", agents)
    original = _Runner.run
    existing = _ExistingHooks()
    cassette_path = tmp_path / "agents.jsonl"

    with Cassette.record(cassette_path) as cassette, patch_openai_agents(cassette):
        result = asyncio.run(agents.Runner.run(_Named("researcher"), "find facts", hooks=existing))

    assert result == "done"
    assert existing.started
    assert _Runner.run is original
    events = load_events(cassette_path)
    assert [event.type for event in events] == [
        EventType.CUSTOM,
        EventType.CUSTOM,
        EventType.CUSTOM,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.CUSTOM,
        EventType.CUSTOM,
    ]
    assert events[3].name == "search"
    assert events[5].input == {"from": "researcher", "to": "writer"}


def test_openai_agents_lifecycle_replays_without_unconsumed_events(tmp_path, monkeypatch):
    agents = types.ModuleType("agents")
    agents.Runner = _Runner
    monkeypatch.setitem(sys.modules, "agents", agents)
    cassette_path = tmp_path / "agents-replay.jsonl"

    with Cassette.record(cassette_path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(_Named("researcher"), "find facts"))
    with Cassette.replay(cassette_path) as cassette, patch_openai_agents(cassette):
        assert asyncio.run(agents.Runner.run(_Named("researcher"), "find facts")) == "done"
