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
from dataclasses import dataclass
from types import MethodType
from typing import Any, cast

from agent_cassette.events import EventType
from agent_cassette.integrations._serialization import serialize_recorded_value
from agent_cassette.json_codec import StrictJSONError
from agent_cassette.replay import RecordedCallError

_TOOL_RESULT_MARKER = "__agent_cassette_langchain_tool_result__"
_TOOL_RESULT_VERSION = 1
# Unique code-owned recorded-error marker so a user exception merely *named*
# ``ToolException`` is never reconstructed as LangChain's fixed class.
_TOOL_EXCEPTION_MARKER = "AgentCassetteLangChainToolExceptionV1"
_OMITTED_CONFIG_KEYS = frozenset({"callbacks", "run_id", "run_name"})
_BRIDGE_STATE_ATTR = "_agent_cassette_tool_bridge"
_PUBLIC_METHODS = ("invoke", "ainvoke", "run", "arun")
_VALID_RESPONSE_FORMATS = ("content", "content_and_artifact")


_MISSING = object()  # private sentinel: distinguish an absent marker from a None collision


@dataclass(frozen=True, slots=True)
class _ToolBridgeState:
    """Immutable, verifiable per-clone bridge marker (exact type, not a bare cassette ref).

    Exactly one of two async shapes holds: ``arun_wrapper`` is the installed instance
    ``_arun`` wrapper, or it is ``None`` and ``default_arun`` is the fixed inherited
    ``BaseTool._arun`` descriptor captured at install (with no instance ``_arun``).
    """

    cassette: Any
    run_wrapper: Any
    arun_wrapper: Any
    default_arun: Any


# --------------------------------------------------------------------------- #
# Request / result codec (strict; not the broad Runnable codec)
# --------------------------------------------------------------------------- #


def _clean_config(config: Any) -> Any:
    if config is None:
        return None
    if type(config) is not dict:
        raise StrictJSONError("langchain tool config must be a plain dict or None")
    # Validate every key is an exact str BEFORE membership filtering, so a str
    # subclass named callbacks/run_id/run_name cannot silently disappear.
    for key in config:
        if type(key) is not str:
            raise StrictJSONError("langchain tool config keys must be exact strings")
    return {key: value for key, value in config.items() if key not in _OMITTED_CONFIG_KEYS}


def _build_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    for key in kwargs:
        if type(key) is not str:
            raise StrictJSONError("langchain tool kwargs keys must be exact strings")
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
    # The only value that is not strict-copied first is the explicitly supported
    # raw exact-tuple Hybrid artifact injection (tuples are not JSON-native).
    if type(recorded) is tuple:
        if response_format != "content_and_artifact" or len(recorded) != 2:
            raise StrictJSONError("injected tuple does not match the tool response format")
        return (serialize_recorded_value(recorded[0]), serialize_recorded_value(recorded[1]))
    # Strict-copy the whole value before any marker/key lookup so membership never
    # touches hostile or key-subclass data.
    detached = serialize_recorded_value(recorded)
    if type(detached) is dict and _TOOL_RESULT_MARKER in detached:
        if detached.get(_TOOL_RESULT_MARKER) is not True:
            raise StrictJSONError("langchain tool result marker is invalid")
        version = detached.get("version")
        if type(version) is not int or version != _TOOL_RESULT_VERSION:
            raise StrictJSONError("langchain tool result has an unsupported version")
        kind = detached.get("kind")
        if type(kind) is not str or kind not in ("json", "content_and_artifact"):
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
        return detached
    raise StrictJSONError("a content_and_artifact tool requires a 2-tuple or marked envelope")


def _make_error_serializer(tool_exception_cls: type | None) -> Any:
    def serialize(error: BaseException) -> dict[str, Any]:
        if tool_exception_cls is not None and isinstance(error, tool_exception_cls):
            # Emit a unique code-owned marker plus the real class name so the
            # dedup check works and only *this* exact SDK ToolException translates back.
            return {
                "type": _TOOL_EXCEPTION_MARKER,
                "message": str(error),
                "original_type": type(error).__name__,
            }
        return {"type": type(error).__name__, "message": str(error)}

    return serialize


def _translate_replayed_error(
    error: RecordedCallError, tool_exception_cls: type | None
) -> BaseException:
    # Translate only the unique code-owned marker, never the ambiguous name
    # "ToolException" (a user lookalike stays a RecordedCallError).
    if tool_exception_cls is not None and error.recorded_type == _TOOL_EXCEPTION_MARKER:
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
    install_arun: bool,
    default_arun: Any,
) -> _ToolBridgeState:
    original_run = clone._run
    original_arun = clone._arun
    error_serializer = _make_error_serializer(tool_exception_cls)

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
                error_serializer=error_serializer,
            )
        except RecordedCallError as error:
            raise _translate_replayed_error(error, tool_exception_cls) from None
        if executed["live"]:
            return result
        return _decode_tool_result(result, response_format)

    run_bound = MethodType(run_wrapper, clone)
    object.__setattr__(clone, "_run", run_bound)

    arun_bound: Any = None
    if install_arun:

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
                    error_serializer=error_serializer,
                )
            except RecordedCallError as error:
                raise _translate_replayed_error(error, tool_exception_cls) from None
            if executed["live"]:
                return result
            return _decode_tool_result(result, response_format)

        arun_bound = MethodType(arun_wrapper, clone)
        object.__setattr__(clone, "_arun", arun_bound)

    # When the default-_arun path was chosen, remember the fixed descriptor so a later
    # instance _arun (or a class change) is detectable; otherwise there is no default.
    return _ToolBridgeState(cassette, run_bound, arun_bound, None if install_arun else default_arun)


def _bridge_one(
    tool: Any,
    index: int,
    cassette: Any,
    name_prefix: str,
    base_tool_cls: type,
    fixed_classes: tuple[type, ...],
    tool_exception_cls: type | None,
) -> Any:
    # Validate the fixed-class BaseTool BEFORE any marker access, so a forged
    # attribute or hostile __getattr__ on a non-tool cannot be reached.
    if not isinstance(tool, base_tool_cls):
        raise TypeError(
            f"tool at index {index} must be a LangChain BaseTool, got {type(tool).__name__}"
        )
    # Read bridge state only from the validated object's instance dictionary. Use a
    # private sentinel so a marker attribute whose value is None is treated as a
    # collision (rejected), not as "absent".
    state = vars(tool).get(_BRIDGE_STATE_ATTR, _MISSING)
    if state is not _MISSING:
        if type(state) is not _ToolBridgeState:
            raise ValueError("tool bridge marker was forged or collided with a foreign attribute")
        if state.cassette is not cassette:
            raise ValueError("tool is already bridged to a different cassette session")
        if vars(tool).get("_run") is not state.run_wrapper:
            raise ValueError("tool _run replay boundary was replaced")
        if state.arun_wrapper is not None:
            if vars(tool).get("_arun") is not state.arun_wrapper:
                raise ValueError("tool _arun replay boundary was replaced")
        else:
            # Default-_arun path: an instance _arun must be absent (LangChain would
            # resolve it before its default offload and bypass the wrapped _run), and
            # the class must still resolve _arun to the fixed descriptor captured at install.
            if "_arun" in vars(tool):
                raise ValueError("tool _arun replay boundary was replaced")
            if getattr(type(tool), "_arun", None) is not state.default_arun:
                raise ValueError("tool _arun class descriptor changed after bridging")
        return tool  # idempotent same-cassette rewrap with intact boundary
    name = getattr(tool, "name", None)
    if type(name) is not str or not name:
        raise TypeError(f"tool at index {index} has an invalid name of type {type(name).__name__}")
    tool_type = type(tool)
    # Reject public-method overrides by resolved-descriptor identity (defeats
    # functools.wraps and __module__ spoofing): each resolved method must be one
    # of the fixed installed SDK descriptors.
    for method in _PUBLIC_METHODS:
        resolved = getattr(tool_type, method, None)
        if resolved not in {getattr(cls, method, None) for cls in fixed_classes}:
            raise TypeError(
                f"tool at index {index} overrides the public {method!r} method "
                f"({tool_type.__name__}); custom public-method overrides are not supported"
            )
    response_format = getattr(tool, "response_format", "content")
    if type(response_format) is not str or response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"tool at index {index} has an unsupported response_format of type "
            f"{type(response_format).__name__}"
        )
    # Clone through the fixed SDK implementation, never a user-overridden model_copy.
    clone = cast(Any, base_tool_cls).model_copy(tool, deep=False)
    if clone is tool or type(clone) is not tool_type:
        raise RuntimeError("BaseTool.model_copy did not return a distinct same-type clone")
    # Only install an _arun boundary for a genuinely custom/SDK async implementation.
    # A conventional tool inheriting the default BaseTool._arun offloads through the
    # already-wrapped _run, so wrapping _arun too would double-record.
    default_arun = getattr(base_tool_cls, "_arun", None)
    install_arun = getattr(tool_type, "_arun", None) is not default_arun
    event_name = f"{name_prefix}.{name}"
    state = _install_tool_bridge(
        clone,
        cassette,
        event_name,
        name,
        response_format,
        tool_exception_cls,
        install_arun,
        default_arun,
    )
    object.__setattr__(clone, _BRIDGE_STATE_ATTR, state)
    return clone


def wrap_langchain_tools(
    tools: Any, cassette: Any, *, name_prefix: str = "langchain.tool"
) -> list[Any]:
    """Return bridged shallow clones of installed LangChain ``BaseTool`` objects."""
    from langchain_core.tools import BaseTool, StructuredTool, Tool

    fixed_classes = (BaseTool, Tool, StructuredTool)

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
        clone = _bridge_one(
            tool, index, cassette, name_prefix, BaseTool, fixed_classes, tool_exception_cls
        )
        by_identity[id(tool)] = clone
        bridged.append(clone)
    return bridged


__all__ = ["wrap_langchain_tools"]
