"""Sync facade over an async MCP session (live, fake, or replayed) on a private loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from agent_cassette import wrap_mcp
from retrace.config import Settings


class DataHubToolError(RuntimeError):
    """The DataHub MCP server reported a tool error."""


class DataHubUnavailable(RuntimeError):
    """The DataHub MCP server could not be started."""


class LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="retrace-mcp"
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def _text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts = [getattr(item, "text", None) for item in content]
    return "\n".join(p for p in parts if p)


def _is_error(result: Any) -> bool:
    # The installed mcp SDK's wire model names this field ``is_error``;
    # ``isError`` remains accepted as a constructor alias only. Check both
    # spellings so this keeps working across mcp SDK versions.
    for attr in ("is_error", "isError"):
        value = getattr(result, attr, None)
        if value is not None:
            return bool(value)
    return False


def parse_result(result: Any) -> Any:
    if _is_error(result):
        raise DataHubToolError(_text(result) or "DataHub tool error")
    text = _text(result)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


class _JSONResultSession:
    """Adapts a session so ``call_tool`` results are plain JSON, not SDK models.

    ``wrap_mcp`` records tool results through Agent Cassette's SDK-trusted
    serializer, which only trusts objects whose defining module lives under
    the ``mcp`` package root (``_TRUSTED_ROOTS = ("mcp",)``). The installed
    mcp SDK defines its wire types in the standalone ``mcp_types`` package
    (mirrored, not re-defined, under ``mcp.types``), so ``CallToolResult``'s
    real ``__module__`` is ``mcp_types._types`` and falls outside that trust
    boundary. Dumping the result to a plain dict here — before it ever
    reaches Agent Cassette — keeps recording on the always-trusted plain-JSON
    path, with no SDK trust decision required at all.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = await self._session.call_tool(name, arguments)
        dump = getattr(result, "model_dump", None)
        return dump(mode="json") if callable(dump) else result


def _server_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env["DATAHUB_GMS_URL"] = settings.datahub_gms_url
    if settings.datahub_gms_token:
        env["DATAHUB_GMS_TOKEN"] = settings.datahub_gms_token
    env["TOOLS_IS_MUTATION_ENABLED"] = "true"
    return env


class DataHubConnection:
    def __init__(self, runner: LoopThread, session: Any, shutdown: Callable[[], None]) -> None:
        self._runner = runner
        self._session = session
        self._shutdown = shutdown

    @classmethod
    def from_session(cls, session: Any, cassette: Any | None = None) -> DataHubConnection:
        runner = LoopThread()
        if cassette is None:
            return cls(runner, session, runner.stop)
        proxy = wrap_mcp(_JSONResultSession(session), cassette, asynchronous=True)
        return cls(runner, proxy, runner.stop)

    @classmethod
    def replay(cls, cassette: Any) -> DataHubConnection:
        runner = LoopThread()
        return cls(runner, wrap_mcp(None, cassette, asynchronous=True), runner.stop)

    @classmethod
    def live(cls, settings: Settings, cassette: Any | None = None) -> DataHubConnection:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        runner = LoopThread()
        ready: concurrent.futures.Future[Any] = concurrent.futures.Future()
        holder: dict[str, asyncio.Event] = {}
        params = StdioServerParameters(
            command="uvx", args=[settings.mcp_server_spec], env=_server_env(settings)
        )

        async def serve() -> None:
            holder["stop"] = asyncio.Event()
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        ready.set_result(session)
                        await holder["stop"].wait()
            except BaseException as error:
                if not ready.done():
                    ready.set_exception(error)
                raise

        task = asyncio.run_coroutine_threadsafe(serve(), runner.loop)
        try:
            session = ready.result(timeout=120)
        except Exception as error:
            runner.stop()
            raise DataHubUnavailable(
                f"could not start {settings.mcp_server_spec}: {error}"
            ) from error

        def shutdown() -> None:
            runner.loop.call_soon_threadsafe(holder["stop"].set)
            try:
                task.result(timeout=10)
            except Exception:
                pass
            runner.stop()

        if cassette is None:
            return cls(runner, session, shutdown)
        proxy = wrap_mcp(_JSONResultSession(session), cassette, asynchronous=True)
        return cls(runner, proxy, shutdown)

    def call(self, name: str, args: dict[str, Any], timeout: float = 45) -> Any:
        return parse_result(self._runner.run(self._session.call_tool(name, args), timeout))

    def close(self) -> None:
        self._shutdown()
