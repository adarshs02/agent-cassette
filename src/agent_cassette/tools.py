"""Explicit wrappers for deterministic Python-tool record and replay."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable
from functools import partial, wraps
from typing import Any, ParamSpec, TypeVar, cast, overload

from agent_cassette.events import EventType
from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError

P = ParamSpec("P")
R = TypeVar("R")


def _root_callable(function: Callable[..., Any]) -> Callable[..., Any]:
    root = function
    while isinstance(root, partial):
        root = root.func

    if not (inspect.isfunction(root) or inspect.ismethod(root)):
        raise ValueError(
            "callable objects and builtin callables are not supported in Phase A; "
            "wrap a Python function or bound method"
        )
    return root


def _resolve_name(function: Callable[..., Any], name: str | None) -> str:
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name must be a nonempty string")
        return name

    candidate = getattr(function, "__name__", None)
    if not isinstance(candidate, str) or not candidate or candidate == "<lambda>":
        raise ValueError(
            "tool name is required for lambdas, partials, and callables without a usable __name__"
        )
    return candidate


def _copy_tool_json(
    value: Any,
    *,
    depth: int = 0,
    active: set[int] | None = None,
) -> Any:
    """Validate and copy exact-builtin JSON only. Subclasses are rejected so that a
    recorded value and its replay have identical Python types (no IntEnum -> int drift)."""
    value_type = type(value)

    if value is None or value_type in (str, bool, int):
        return value

    if value_type is float:
        if not math.isfinite(value):
            raise StrictJSONError("non-finite floats are not valid tool JSON")
        return value

    if value_type not in (list, dict):
        raise StrictJSONError(
            f"unsupported tool JSON value type: {value_type.__module__}.{value_type.__qualname__}"
        )

    if depth >= MAX_JSON_DEPTH:
        raise StrictJSONError(f"maximum tool JSON depth {MAX_JSON_DEPTH} exceeded")

    if active is None:
        active = set()

    value_id = id(value)
    if value_id in active:
        raise StrictJSONError("cyclic values are not valid tool JSON")

    active.add(value_id)
    try:
        if value_type is list:
            return [_copy_tool_json(item, depth=depth + 1, active=active) for item in value]

        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise StrictJSONError("tool JSON object keys must be plain strings")
            copied[key] = _copy_tool_json(item, depth=depth + 1, active=active)
        return copied
    finally:
        active.remove(value_id)


def _serialize_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _copy_tool_json({"args": list(args), "kwargs": kwargs}),
    )


def _serialize_output(value: Any) -> Any:
    return _copy_tool_json(value)


def wrap_tool(
    function: Callable[P, R],
    cassette: Any,
    *,
    name: str | None = None,
) -> Callable[P, R]:
    """Wrap a sync or async Python tool for cassette record and replay."""
    if not callable(function):
        raise TypeError("tool must be callable")

    root = _root_callable(function)

    if inspect.isgeneratorfunction(root) or inspect.isasyncgenfunction(root):
        raise ValueError(
            "generator and async-generator tools are not supported in Phase A; "
            "streaming tool support is Phase B"
        )

    tool_name = _resolve_name(function, name)

    if inspect.iscoroutinefunction(root):
        asynchronous = cast(Callable[P, Awaitable[Any]], function)

        @wraps(function)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            input_value = _serialize_input(args, kwargs)
            return await cassette.acall(
                EventType.TOOL_CALL,
                tool_name,
                input_value,
                lambda: asynchronous(*args, **kwargs),
                serializer=_serialize_output,
            )

        return cast(Callable[P, R], async_wrapper)

    synchronous = cast(Callable[P, Any], function)

    @wraps(function)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        input_value = _serialize_input(args, kwargs)
        return cassette.call(
            EventType.TOOL_CALL,
            tool_name,
            input_value,
            lambda: synchronous(*args, **kwargs),
            serializer=_serialize_output,
        )

    return cast(Callable[P, R], sync_wrapper)


class _ToolSessionMixin:
    @overload
    def tool(
        self,
        function: Callable[P, R],
        *,
        name: str | None = None,
    ) -> Callable[P, R]: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def tool(
        self,
        function: Callable[P, R] | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        """Bind wrap_tool to this cassette session."""
        if function is not None:
            return wrap_tool(function, self, name=name)

        def decorate(value: Callable[P, R]) -> Callable[P, R]:
            return wrap_tool(value, self, name=name)

        return decorate


__all__ = ["wrap_tool"]
