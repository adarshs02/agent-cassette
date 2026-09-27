"""Sync facade over an async MCP session (live, fake, or replayed) on a private loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
from collections.abc import Callable, Coroutine, Iterable
from typing import Any, TextIO

from agent_cassette import wrap_mcp
from retrace.config import Settings


class DataHubToolError(RuntimeError):
    """The DataHub MCP server reported a tool error."""


class DataHubUnavailable(RuntimeError):
    """The DataHub MCP server could not be started."""


DEFAULT_TIMEOUT = 45.0
# The outer (thread-side) wait must outlast the inner, recorded timeout so the inner
# TimeoutError is what surfaces and gets recorded as the call's event.
OUTER_TIMEOUT_MARGIN = 15.0

# Tools and parameters Retrace sends; the live server must accept all of them.
REQUIRED_TOOLS: dict[str, tuple[str, ...]] = {
    "search": ("query", "num_results"),
    "get_entities": ("urns",),
    "get_lineage": ("urn", "upstream", "max_hops"),
    "save_document": ("document_type", "title", "content", "topics", "related_assets"),
    "add_tags": ("tag_urns", "entity_urns"),
}


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def check_tool_contract(tools: Iterable[Any]) -> list[str]:
    """Return what the server's tool listing lacks versus REQUIRED_TOOLS (empty if ok)."""
    listed: dict[str, set[str]] = {}
    for tool in tools:
        name = _field(tool, "name")
        schema = _field(tool, "input_schema", "inputSchema") or {}
        properties = _field(schema, "properties") or {}
        if isinstance(name, str):
            listed[name] = set(properties) if isinstance(properties, dict) else set()
    problems: list[str] = []
    for name, params in REQUIRED_TOOLS.items():
        if name not in listed:
            problems.append(f"missing tool {name}")
            continue
        problems += [f"{name}: missing parameter {p}" for p in params if p not in listed[name]]
    return problems


async def _list_all_tools(session: Any) -> list[Any]:
    tools: list[Any] = []
    cursor = None
    for _ in range(100):  # bounded pagination
        if cursor is None:
            listing = await session.list_tools()
        else:
            from mcp.types import PaginatedRequestParams

            listing = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        tools.extend(_field(listing, "tools") or [])
        cursor = _field(listing, "next_cursor", "nextCursor")
        if not cursor:
            break
    return tools


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

    def __init__(self, session: Any, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._session = session
        self._timeout = timeout

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        # The timeout lives inside wrap_mcp's live_call, so a timeout is recorded as this
        # call's error event and replays as the same TimeoutError.
        try:
            result = await asyncio.wait_for(self._session.call_tool(name, arguments), self._timeout)
        except asyncio.TimeoutError:  # a distinct class from TimeoutError on Python 3.10
            raise TimeoutError(
                f"DataHub MCP call {name} timed out after {self._timeout:g}s"
            ) from None
        dump = getattr(result, "model_dump", None)
        return dump(mode="json") if callable(dump) else result


def _server_errlog(settings: Settings) -> TextIO:
    """Open the file the DataHub MCP server's stderr is redirected to.

    The server is chatty on stderr; routed here (append mode) instead of our
    own stderr so its debug lines don't flood Retrace's terminal output.
    """
    path = settings.mcp_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed explicitly by the caller


def _shutdown_server(
    runner: LoopThread,
    task: concurrent.futures.Future[Any],
    stop_event: asyncio.Event,
    errlog: TextIO,
) -> None:
    """Tear down the live() server thread, always closing the errlog handle.

    ``runner.stop()`` can raise (e.g. the loop thread failing to join); the
    errlog file descriptor must still be closed so it isn't leaked.
    """
    runner.loop.call_soon_threadsafe(stop_event.set)
    try:
        task.result(timeout=10)
    except Exception:
        pass
    try:
        runner.stop()
    finally:
        errlog.close()


def _server_env(settings: Settings) -> dict[str, str]:
    """An allowlisted environment for the MCP server: never pass model credentials."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "HOME") or key.startswith(("UV_", "XDG_"))
    }
    env["DATAHUB_GMS_URL"] = settings.datahub_gms_url
    if settings.datahub_gms_token:
        env["DATAHUB_GMS_TOKEN"] = settings.datahub_gms_token
    env["TOOLS_IS_MUTATION_ENABLED"] = "true"
    return env


class DataHubConnection:
    def __init__(
        self,
        runner: LoopThread,
        session: Any,
        shutdown: Callable[[], None],
        *,
        timeout: float = DEFAULT_TIMEOUT,
        inner_timeout: bool = False,
    ) -> None:
        self._runner = runner
        self._session = session
        self._shutdown = shutdown
        self._timeout = timeout
        self._inner_timeout = inner_timeout

    @classmethod
    def _wrap(
        cls,
        runner: LoopThread,
        session: Any,
        shutdown: Callable[[], None],
        cassette: Any | None,
        timeout: float,
    ) -> DataHubConnection:
        if cassette is None:
            return cls(runner, session, shutdown, timeout=timeout)
        proxy = wrap_mcp(_JSONResultSession(session, timeout), cassette, asynchronous=True)
        return cls(runner, proxy, shutdown, timeout=timeout, inner_timeout=True)

    @classmethod
    def from_session(
        cls, session: Any, cassette: Any | None = None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> DataHubConnection:
        runner = LoopThread()
        return cls._wrap(runner, session, runner.stop, cassette, timeout)

    @classmethod
    def replay(cls, cassette: Any) -> DataHubConnection:
        runner = LoopThread()
        return cls(
            runner, wrap_mcp(None, cassette, asynchronous=True), runner.stop, inner_timeout=True
        )

    @classmethod
    def live(
        cls, settings: Settings, cassette: Any | None = None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> DataHubConnection:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        runner = LoopThread()
        ready: concurrent.futures.Future[Any] = concurrent.futures.Future()
        holder: dict[str, asyncio.Event] = {}
        params = StdioServerParameters(
            command="uvx", args=[settings.mcp_server_spec], env=_server_env(settings)
        )
        errlog = _server_errlog(settings)

        async def serve() -> None:
            holder["stop"] = asyncio.Event()
            try:
                async with stdio_client(params, errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        problems = check_tool_contract(await _list_all_tools(session))
                        if problems:
                            raise DataHubUnavailable(
                                f"{settings.mcp_server_spec} does not match Retrace's tool "
                                f"contract: {'; '.join(problems)}"
                            )
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
            try:
                runner.stop()
            finally:
                errlog.close()
            raise DataHubUnavailable(
                f"could not start {settings.mcp_server_spec}: {error}"
            ) from error

        def shutdown() -> None:
            _shutdown_server(runner, task, holder["stop"], errlog)

        return cls._wrap(runner, session, shutdown, cassette, timeout)

    def call(self, name: str, args: dict[str, Any]) -> Any:
        wait = self._timeout + OUTER_TIMEOUT_MARGIN if self._inner_timeout else self._timeout
        return parse_result(self._runner.run(self._session.call_tool(name, args), wait))

    def close(self) -> None:
        self._shutdown()
