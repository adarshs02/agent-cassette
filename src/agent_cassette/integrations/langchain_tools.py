"""Registered-tool replay bridge for LangChain ``BaseTool`` objects.

Imported lazily by :mod:`agent_cassette` so the core package never imports
``langchain_core``. ``wrap_langchain_tools`` returns bridged shallow clones of
installed tools: during record each tool's real ``_run``/``_arun`` executes once
and its result is captured; during replay LangChain's agent loop, args-schema
validation, callback lifecycle, output formatting, and error handling all still
run, but the original tool body is never called. This complements the Runnable
boundary (:func:`wrap_langchain`) and the observational
``langchain_callback_handler``; it is not a tracer.
"""

from __future__ import annotations

import functools
from types import MethodType
from typing import Any, cast

from agent_cassette.events import EventType
from agent_cassette.integrations._serialization import serialize_recorded_value
from agent_cassette.json_codec import StrictJSONError
from agent_cassette.replay import RecordedCallError

_TOOL_RESULT_MARKER = "__agent_cassette_langchain_tool_result__"
_TOOL_RESULT_VERSION = 1
_OMITTED_CONFIG_KEYS = frozenset({"callbacks", "run_id", "run_name"})
_BRIDGE_CASSETTE_ATTR = "_agent_cassette_tool_bridge"
_PUBLIC_METHODS = ("invoke", "ainvoke", "run", "arun")
_VALID_RESPONSE_FORMATS = ("content", "content_and_artifact")


# --------------------------------------------------------------------------- #
# Request / result codec (strict; not the broad Runnable codec)
# --------------------------------------------------------------------------- #


def _clean_config(config: Any) -> Any:
    if config is None:
        return None
    if type(config) is not dict:
        raise StrictJSONError("langchain tool config must be a plain dict or None")
    # Drop only the volatile top-level keys; retain tags, metadata, configurable, etc.
    return {key: value for key, value in config.items() if key not in _OMITTED_CONFIG_KEYS}


def _build_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    match_kwargs = {
        key: value for key, value in kwargs.items() if key not in ("run_manager", "config")
    }
    return cast(
        dict[str, Any],
        serialize_recorded_value(
            {
                "args": list(args),
                "kwargs": match_kwargs,
                "config": _clean_config(kwargs.get("config")),
            }
        ),
    )


def _tool_metadata(tool_name: str) -> dict[str, Any]:
    return {
        "integration": "langchain",
        "operation": "tool",
        "tool_bridge": True,
        "tool": tool_name,
    }


def _encode_tool_result(value: Any, response_format: str) -> dict[str, Any]:
    if response_format == "content_and_artifact":
        if type(value) is not tuple or len(value) != 2:
            raise StrictJSONError("a content_and_artifact tool must return a length-2 tuple")
        data: Any = [serialize_recorded_value(value[0]), serialize_recorded_value(value[1])]
        kind = "content_and_artifact"
    else:
        data = serialize_recorded_value(value)
        kind = "json"
    return cast(
        dict[str, Any],
        serialize_recorded_value(
            {
                _TOOL_RESULT_MARKER: True,
                "version": _TOOL_RESULT_VERSION,
                "kind": kind,
                "value": data,
            }
        ),
    )


def _decode_tool_result(recorded: Any, response_format: str) -> Any:
    # A Hybrid Return injection may hand back a raw two-tuple (artifact tool) directly.
    if type(recorded) is tuple:
        if response_format != "content_and_artifact" or len(recorded) != 2:
            raise StrictJSONError("injected tuple does not match the tool response format")
        return (serialize_recorded_value(recorded[0]), serialize_recorded_value(recorded[1]))
    if type(recorded) is dict and _TOOL_RESULT_MARKER in recorded:
        detached = serialize_recorded_value(recorded)
        if detached.get(_TOOL_RESULT_MARKER) is not True:
            raise StrictJSONError("langchain tool result marker is invalid")
        version = detached.get("version")
        if type(version) is not int or version != _TOOL_RESULT_VERSION:
            raise StrictJSONError("langchain tool result has an unsupported version")
        kind = detached.get("kind")
        if kind not in ("json", "content_and_artifact"):
            raise StrictJSONError("langchain tool result kind is invalid")
        if set(detached) != {_TOOL_RESULT_MARKER, "version", "kind", "value"}:
            raise StrictJSONError("langchain tool result envelope has unexpected keys")
        value = detached["value"]
        if kind == "json":
            if response_format != "content":
                raise StrictJSONError("a json tool result requires a content-format tool")
            return value
        if response_format != "content_and_artifact":
            raise StrictJSONError("a content_and_artifact result requires that response format")
        if type(value) is not list or len(value) != 2:
            raise StrictJSONError("content_and_artifact value must be a two-item list")
        return (value[0], value[1])
    # A Hybrid Return injection may hand back a raw JSON value for a content tool.
    if response_format == "content":
        return serialize_recorded_value(recorded)
    raise StrictJSONError("a content_and_artifact tool requires a 2-tuple or marked envelope")


def _translate_replayed_error(
    error: RecordedCallError, tool_exception_cls: type | None
) -> BaseException:
    # Only the exact recorded ToolException is translated back so LangChain's
    # handle_tool_error can run again; never a module-qualified/lookalike name.
    if tool_exception_cls is not None and error.recorded_type == "ToolException":
        return tool_exception_cls(error.recorded_message)
    return error


# --------------------------------------------------------------------------- #
# Bridge installation
# --------------------------------------------------------------------------- #


def _install_tool_bridge(
    clone: Any,
    cassette: Any,
    event_name: str,
    tool_name: str,
    response_format: str,
    tool_exception_cls: type | None,
) -> None:
    original_run = clone._run
    original_arun = clone._arun

    @functools.wraps(original_run)
    def run_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = _build_request(args, kwargs)
        executed = {"live": False}

        def live() -> Any:
            executed["live"] = True
            return original_run(*args, **kwargs)

        try:
            result = cassette.call(
                EventType.TOOL_CALL,
                event_name,
                request,
                live,
                metadata=_tool_metadata(tool_name),
                serializer=lambda value: _encode_tool_result(value, response_format),
            )
        except RecordedCallError as error:
            raise _translate_replayed_error(error, tool_exception_cls) from None
        if executed["live"]:
            return result
        return _decode_tool_result(result, response_format)

    @functools.wraps(original_arun)
    async def arun_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = _build_request(args, kwargs)
        executed = {"live": False}

        async def live() -> Any:
            executed["live"] = True
            return await original_arun(*args, **kwargs)

        try:
            result = await cassette.acall(
                EventType.TOOL_CALL,
                event_name,
                request,
                live,
                metadata=_tool_metadata(tool_name),
                serializer=lambda value: _encode_tool_result(value, response_format),
            )
        except RecordedCallError as error:
            raise _translate_replayed_error(error, tool_exception_cls) from None
        if executed["live"]:
            return result
        return _decode_tool_result(result, response_format)

    object.__setattr__(clone, "_run", MethodType(run_wrapper, clone))
    object.__setattr__(clone, "_arun", MethodType(arun_wrapper, clone))


def _defining_module(cls: type, method: str) -> str:
    func = getattr(cls, method, None)
    func = getattr(func, "__func__", func)
    return getattr(func, "__module__", "") or ""


def _bridge_one(
    tool: Any,
    index: int,
    cassette: Any,
    name_prefix: str,
    base_tool_cls: type,
    tool_exception_cls: type | None,
) -> Any:
    existing = getattr(tool, _BRIDGE_CASSETTE_ATTR, None)
    if existing is not None:
        # Idempotent same-cassette rewrap; a different cassette must not nest boundaries.
        if existing is cassette:
            return tool
        raise ValueError("tool is already bridged to a different cassette session")
    if not isinstance(tool, base_tool_cls):
        raise TypeError(
            f"tool at index {index} must be a LangChain BaseTool, got {type(tool).__name__}"
        )
    name = getattr(tool, "name", None)
    if type(name) is not str or not name:
        raise TypeError(f"tool at index {index} has an invalid name of type {type(name).__name__}")
    for method in _PUBLIC_METHODS:
        module = _defining_module(type(tool), method)
        if not (module == "langchain_core" or module.startswith("langchain_core.")):
            raise TypeError(
                f"tool at index {index} overrides the public {method!r} method "
                f"({type(tool).__name__}); custom public-method overrides are not supported"
            )
    response_format = getattr(tool, "response_format", "content")
    # Guard the type before membership/formatting so a hostile object's __eq__/__repr__
    # can never run here (or, being closed over, on the replay hot path).
    if type(response_format) is not str or response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"tool at index {index} has an unsupported response_format of type "
            f"{type(response_format).__name__}"
        )
    # Clone through the fixed SDK implementation, never a user-overridden model_copy.
    clone = cast(Any, base_tool_cls).model_copy(tool, deep=False)
    if clone is tool or type(clone) is not type(tool):
        raise RuntimeError("BaseTool.model_copy did not return a distinct same-type clone")
    event_name = f"{name_prefix}.{name}"
    _install_tool_bridge(clone, cassette, event_name, name, response_format, tool_exception_cls)
    object.__setattr__(clone, _BRIDGE_CASSETTE_ATTR, cassette)
    return clone


def wrap_langchain_tools(
    tools: Any, cassette: Any, *, name_prefix: str = "langchain.tool"
) -> list[Any]:
    """Return bridged shallow clones of installed LangChain ``BaseTool`` objects."""
    from langchain_core.tools import BaseTool

    tool_exception_cls: type | None
    try:
        from langchain_core.tools import ToolException

        tool_exception_cls = ToolException
    except Exception:  # noqa: BLE001 -- optional; translation simply disabled without it
        tool_exception_cls = None

    if type(tools) is not list and type(tools) is not tuple:
        raise TypeError(
            f"wrap_langchain_tools requires a list or tuple of tools, got {type(tools).__name__}"
        )
    if type(name_prefix) is not str or not name_prefix.strip():
        raise ValueError("name_prefix must be a nonempty string")

    bridged: list[Any] = []
    by_identity: dict[int, Any] = {}
    for index, tool in enumerate(tools):
        if id(tool) in by_identity:
            # Preserve duplicate identity: the same original yields the same clone.
            bridged.append(by_identity[id(tool)])
            continue
        clone = _bridge_one(tool, index, cassette, name_prefix, BaseTool, tool_exception_cls)
        by_identity[id(tool)] = clone
        bridged.append(clone)
    return bridged


__all__ = ["wrap_langchain_tools"]
