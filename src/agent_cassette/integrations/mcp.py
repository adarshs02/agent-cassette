"""Transparent record and replay wrapper for MCP client sessions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from agent_cassette.events import EventType

Session = TypeVar("Session")


class _ReplayObject(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class MCPClientProxy:
    """Proxy that records and replays ``call_tool`` operations."""

    def __init__(self, session: Any, cassette: Any, *, asynchronous: bool) -> None:
        self._session = session
        self._cassette = cassette
        self._asynchronous = asynchronous

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._session, name) if self._session is not None else None
        if name != "call_tool":
            if target is None:
                raise AttributeError(f"MCP session attribute {name!r} requires a live session")
            return target
        return self._async_call_tool(target) if self._asynchronous else self._sync_call_tool(target)

    def _sync_call_tool(self, call_tool: Callable[..., Any] | None) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            name = _tool_name(args, kwargs)
            request = {"args": _to_data(args), "kwargs": _to_data(kwargs)}

            def live_call() -> Any:
                if call_tool is None:
                    raise RuntimeError("Replay unexpectedly attempted a live MCP tool call")
                return call_tool(*args, **kwargs)

            result = self._cassette.call(
                EventType.TOOL_CALL,
                f"mcp.{name}",
                request,
                live_call,
                metadata={"provider": "mcp", "tool": name},
                serializer=_to_data,
            )
            return _to_replay_object(result)

        return call

    def _async_call_tool(self, call_tool: Callable[..., Any] | None) -> Callable[..., Any]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            name = _tool_name(args, kwargs)
            request = {"args": _to_data(args), "kwargs": _to_data(kwargs)}

            async def live_call() -> Any:
                if call_tool is None:
                    raise RuntimeError("Replay unexpectedly attempted a live MCP tool call")
                return await call_tool(*args, **kwargs)

            result = await self._cassette.acall(
                EventType.TOOL_CALL,
                f"mcp.{name}",
                request,
                live_call,
                metadata={"provider": "mcp", "tool": name},
                serializer=_to_data,
            )
            return _to_replay_object(result)

        return call


def wrap_mcp(
    session: Session | None, cassette: Any, *, asynchronous: bool | None = None
) -> Session:
    """Wrap an MCP client session for automatic tool-call capture and replay."""
    if asynchronous is None:
        asynchronous = session is None or inspect.iscoroutinefunction(
            getattr(session, "call_tool", None)
        )
    return MCPClientProxy(session, cassette, asynchronous=asynchronous)  # type: ignore[return-value]


def _tool_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    value = args[0] if args else kwargs.get("name")
    if not isinstance(value, str) or not value:
        raise TypeError("MCP call_tool requires a tool name")
    return value


def _to_data(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if isinstance(value, dict):
        return {str(key): _to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _to_replay_object(value: Any) -> Any:
    if isinstance(value, dict):
        return _ReplayObject({key: _to_replay_object(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_replay_object(item) for item in value]
    return value


__all__ = ["MCPClientProxy", "wrap_mcp"]
