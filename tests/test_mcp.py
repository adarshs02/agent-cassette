from __future__ import annotations

import asyncio

import pytest

from agent_cassette import Cassette
from agent_cassette.integrations.mcp import wrap_mcp
from agent_cassette.json_codec import StrictJSONError


class _Result:
    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]

    def model_dump(self, mode=None):
        return {"content": self.content}


# Pin the fake under the real ``mcp`` root so it serializes via the trusted
# SDK path.
_Result.__module__ = "mcp"


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


class _McpTypesResult:
    """Fake mirroring mcp SDK 2.x, whose wire types live under ``mcp_types``.

    ``mcp.types.CallToolResult.__module__`` is ``"mcp_types._types"`` on
    mcp>=2 (the module is only re-exported from ``mcp.types``), and the field
    is named ``is_error`` rather than the pre-2.0 ``isError``.
    """

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [{"type": "text", "text": text}]
        self.is_error = is_error

    def model_dump(self, mode=None):
        return {"content": self.content, "is_error": self.is_error}


# Pin the fake under the ``mcp_types`` root -- not ``mcp`` -- so the test
# fails if trust is not extended to where mcp SDK 2.x actually defines its
# result types.
_McpTypesResult.__module__ = "mcp_types._types"


class _McpTypesSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name, arguments, **kwargs):
        self.calls += 1
        return _McpTypesResult(f"{name}:{arguments['query']}")


def test_mcp_types_root_is_trusted_for_mcp_sdk_2x_results(tmp_path):
    path = tmp_path / "mcp.jsonl"
    session = _McpTypesSession()

    async def scenario():
        async with Cassette.record(path) as cassette:
            recorded = await wrap_mcp(session, cassette).call_tool("search", {"query": "agents"})
        async with Cassette.replay(path) as cassette:
            replayed = await wrap_mcp(None, cassette).call_tool("search", {"query": "agents"})
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded.content == replayed.content
    assert replayed.content[0].text == "search:agents"
    assert replayed.is_error is False
    assert session.calls == 1


class _LookAlikeResult:
    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]

    def model_dump(self, mode=None):
        return {"content": self.content}


class _LookAlikeSession:
    def __init__(self, module: str) -> None:
        self.calls = 0
        self._module = module

    async def call_tool(self, name, arguments, **kwargs):
        self.calls += 1
        result_type = type("_LookAlikeResult", (_LookAlikeResult,), {})
        result_type.__module__ = self._module
        return result_type(f"{name}:{arguments['query']}")


@pytest.mark.parametrize("module", ["mcp_typesX._types", "evil_mcp_types"])
def test_mcp_types_lookalike_module_is_still_rejected(tmp_path, module):
    """Trust is exact-root or root + '.' -- a mere name prefix must not match."""
    path = tmp_path / "mcp.jsonl"
    session = _LookAlikeSession(module)

    async def scenario():
        async with Cassette.record(path) as cassette:
            with pytest.raises(StrictJSONError):
                await wrap_mcp(session, cassette).call_tool("search", {"query": "agents"})

    asyncio.run(scenario())
    assert session.calls == 1


def test_real_mcp_sdk_call_tool_result_records_and_replays(tmp_path):
    mcp_types = pytest.importorskip("mcp.types")

    class _RealSession:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, name, arguments, **kwargs):
            self.calls += 1
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=f"{name}:{arguments['query']}")],
                is_error=False,
            )

    path = tmp_path / "mcp.jsonl"
    session = _RealSession()

    async def scenario():
        async with Cassette.record(path) as cassette:
            recorded = await wrap_mcp(session, cassette).call_tool("search", {"query": "agents"})
        async with Cassette.replay(path) as cassette:
            replayed = await wrap_mcp(None, cassette).call_tool("search", {"query": "agents"})
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert replayed.content[0].text == "search:agents"
    assert replayed.is_error is False
    assert session.calls == 1
