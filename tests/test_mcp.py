from __future__ import annotations

import asyncio

from agent_cassette import Cassette
from agent_cassette.integrations.mcp import wrap_mcp


class _Result:
    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]

    def model_dump(self, mode=None):
        return {"content": self.content}


class _AsyncSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name, arguments, **kwargs):
        self.calls += 1
        return _Result(f"{name}:{arguments['query']}")


def test_mcp_tool_calls_record_and_replay_without_live_session(tmp_path):
    path = tmp_path / "mcp.jsonl"
    session = _AsyncSession()

    async def scenario():
        async with Cassette.record(path) as cassette:
            recorded = await wrap_mcp(session, cassette).call_tool("search", {"query": "agents"})
        async with Cassette.replay(path) as cassette:
            replayed = await wrap_mcp(None, cassette).call_tool("search", {"query": "agents"})
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded.content == replayed.content
    assert replayed.content[0].text == "search:agents"
    assert session.calls == 1
