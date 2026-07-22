"""Lifecycle capture, hook injection, and FunctionTool replay for the OpenAI Agents SDK.

Phase C1 adds a tool-replay bridge: ordinary SDK ``FunctionTool`` callbacks record
and replay their results at the public ``FunctionTool.on_invoke_tool(context,
arguments)`` boundary. During replay the SDK agent loop still runs against replayed
model responses, hooks, guardrails, and handoffs, but the original Python tool
callback is never invoked. Agent-as-tool ``FunctionTool`` values are intentionally
left unbridged so their nested run replays in order. All ``agents``/``openai``
imports stay lazy: importing ``agent_cassette`` never imports either SDK.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, cast

from agent_cassette.events import EventType
from agent_cassette.integrations._serialization import (
    serialize_recorded_value,
    serialize_sdk_value,
)
from agent_cassette.json_codec import StrictJSONError

_TRUSTED_ROOTS = ("agents", "openai")

_TOOL_RESULT_MARKER = "__agent_cassette_openai_agents_tool_result__"
_TOOL_RESULT_VERSION = 1
_STRUCTURED_TYPE_FIELD = "type"
_STRUCTURED_CLASS_NAMES = ("ToolOutputText", "ToolOutputImage", "ToolOutputFileContent")


def _to_data(value: Any) -> Any:
    return serialize_sdk_value(value, trusted_roots=_TRUSTED_ROOTS)


class OpenAIAgentsUnavailableError(ImportError):
    """Raised when OpenAI Agents integration is requested without the SDK."""


# --------------------------------------------------------------------------- #
# Tool-result envelope (C1-specific; not the provider/stream/LangChain markers)
# --------------------------------------------------------------------------- #


def _encode_agents_tool_result(value: Any, structured_types: tuple[type, ...]) -> dict[str, Any]:
    """Serialize a FunctionTool callback result into the versioned C1 envelope.

    ``json`` accepts only the exact JSON-native contract (via
    ``serialize_recorded_value``); ``structured``/``structured_list`` accept only
    exact instances of the lazily captured SDK ``ToolOutput*`` classes. Anything
    mixed, subclassed, or otherwise unsupported is rejected before a success event
    is persisted -- no ``str``/``repr``/arbitrary ``model_dump`` fallback.
    """
    if structured_types and type(value) in structured_types:
        kind = "structured"
        data: Any = serialize_sdk_value(value, trusted_roots=_TRUSTED_ROOTS)
    elif (
        structured_types
        and type(value) is list
        and value
        and all(type(element) in structured_types for element in value)
    ):
        kind = "structured_list"
        data = [serialize_sdk_value(element, trusted_roots=_TRUSTED_ROOTS) for element in value]
    else:
        kind = "json"
        data = serialize_recorded_value(value)
    envelope = {
        _TOOL_RESULT_MARKER: True,
        "version": _TOOL_RESULT_VERSION,
        "kind": kind,
        "value": data,
    }
    # Strict-copy and validate the completed envelope before persistence.
    return cast(dict[str, Any], serialize_recorded_value(envelope))


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Recursive, type-exact JSON equality (so ``True`` never equals ``1``)."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if left.keys() != right.keys():
            return False
        return all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(item, other) for item, other in zip(left, right, strict=True)
        )
    return bool(left == right)


def _reconstruct_structured(value: Any, structured_types: tuple[type, ...]) -> Any:
    if type(value) is not dict:
        raise StrictJSONError("structured tool output must be a JSON object")
    discriminator = value.get(_STRUCTURED_TYPE_FIELD)
    selected: type | None = None
    for candidate in structured_types:
        if _structured_discriminator(candidate) == discriminator:
            selected = candidate
            break
    if selected is None:
        # Never import a class named by cassette data; only fixed captured classes.
        raise StrictJSONError("unknown structured tool output type discriminator")
    try:
        instance = selected.model_validate(value)  # type: ignore[attr-defined]
        reserialized = serialize_sdk_value(instance, trusted_roots=_TRUSTED_ROOTS)
    except Exception:  # noqa: BLE001 -- SDK/pydantic validation failure; hide payload cause
        # Raise ``from None`` so no chained SDK/pydantic exception can expose the
        # rejected cassette payload through its traceback or repr.
        raise StrictJSONError(
            f"structured tool output failed SDK validation: {selected.__qualname__}"
        ) from None
    # Require an exact, recursive round-trip: reject any dropped key, inserted
    # default, or coerced value that Pydantic would otherwise accept silently.
    if not _exact_json_equal(reserialized, value):
        raise StrictJSONError(
            f"structured tool output did not round-trip exactly: {selected.__qualname__}"
        ) from None
    return instance


def _decode_agents_tool_result(recorded: Any, structured_types: tuple[type, ...]) -> Any:
    """Restore a recorded/injected tool result, failing closed on a malformed envelope."""
    # A Hybrid Return injection may hand back a raw structured SDK object -- or an exact
    # list of them -- directly. Loaded replay data is never a live SDK instance, so these
    # checks only ever match an injected raw value, symmetric with the encoder.
    if structured_types and type(recorded) in structured_types:
        return recorded
    if (
        structured_types
        and type(recorded) is list
        and recorded
        and all(type(element) in structured_types for element in recorded)
    ):
        return recorded
    if type(recorded) is dict and _TOOL_RESULT_MARKER in recorded:
        detached = serialize_recorded_value(recorded)
        if detached.get(_TOOL_RESULT_MARKER) is not True:
            raise StrictJSONError("openai-agents tool result marker is invalid")
        version = detached.get("version")
        if type(version) is not int or version != _TOOL_RESULT_VERSION:
            raise StrictJSONError("openai-agents tool result has an unsupported version")
        kind = detached.get("kind")
        if kind not in ("json", "structured", "structured_list"):
            raise StrictJSONError("openai-agents tool result kind is invalid")
        if set(detached) != {_TOOL_RESULT_MARKER, "version", "kind", "value"}:
            raise StrictJSONError("openai-agents tool result envelope has unexpected keys")
        value = detached["value"]
        if kind == "json":
            return value
        if kind == "structured":
            return _reconstruct_structured(value, structured_types)
        if type(value) is not list:
            raise StrictJSONError("structured_list tool output value must be a list")
        return [_reconstruct_structured(element, structured_types) for element in value]
    # Legacy pre-C1 TOOL_RESULT (unmarked JSON) or an injected raw JSON value:
    # strict-copy and return.
    return serialize_recorded_value(recorded)


def _structured_discriminator(cls: type) -> Any:
    field = getattr(cls, "model_fields", {}).get(_STRUCTURED_TYPE_FIELD)
    return getattr(field, "default", None)


def _resolve_structured_classes(agents_module: Any) -> tuple[type, ...]:
    classes: list[type] = []
    tool_module: Any = None
    try:
        tool_module = importlib.import_module("agents.tool")
    except Exception:  # noqa: BLE001 -- optional; fall back to the top-level module
        tool_module = None
    for name in _STRUCTURED_CLASS_NAMES:
        cls = getattr(agents_module, name, None)
        if cls is None and tool_module is not None:
            cls = getattr(tool_module, name, None)
        if (
            isinstance(cls, type)
            and _STRUCTURED_TYPE_FIELD in getattr(cls, "model_fields", {})
            and hasattr(cls, "model_validate")
        ):
            classes.append(cls)
    return tuple(classes)


# --------------------------------------------------------------------------- #
# Tool bridge state
# --------------------------------------------------------------------------- #


class _AgentsToolBridgeState:
    """Owns per-session FunctionTool callback patching and call metadata."""

    def __init__(
        self,
        cassette: Any,
        function_tool_cls: type | None,
        structured_types: tuple[type, ...],
    ) -> None:
        self._cassette = cassette
        self._function_tool_cls = function_tool_cls
        self._structured_types = structured_types
        # id(tool) -> (tool, original_callback, installed_wrapper)
        self._patched: dict[int, tuple[Any, Any, Any]] = {}
        # (id(tool), tool_call_id) -> result-match input dict
        self._call_meta: dict[tuple[int, Any], dict[str, Any]] = {}

    def patch_agent(self, agent: Any) -> None:
        if self._function_tool_cls is None or agent is None:
            return
        tools = getattr(agent, "tools", None)
        if not tools:
            return
        for tool in tools:
            self._maybe_wrap(tool)

    def _maybe_wrap(self, tool: Any) -> None:
        # isinstance (not exact type): a user's SDK-compatible FunctionTool subclass is
        # still an ordinary function tool and must get zero-live replay.
        if self._function_tool_cls is None or not isinstance(tool, self._function_tool_cls):
            return
        if self._is_agent_as_tool(tool):
            return
        if id(tool) in self._patched:
            return
        original = getattr(tool, "on_invoke_tool", None)
        if not callable(original):
            return
        wrapper = self._make_wrapper(tool, original)
        setattr(tool, "on_invoke_tool", wrapper)  # noqa: B010 -- attribute name is fixed
        self._patched[id(tool)] = (tool, original, wrapper)

    @staticmethod
    def _is_agent_as_tool(tool: Any) -> bool:
        # Conservative across the supported SDK range: exclude when the flag is set OR
        # an agent instance is present (an ordinary FunctionTool has _agent_instance None),
        # so a False flag alongside an agent instance is still excluded.
        if getattr(tool, "_is_agent_tool", None) is True:
            return True
        return getattr(tool, "_agent_instance", None) is not None

    def is_bridged(self, tool: Any) -> bool:
        entry = self._patched.get(id(tool))
        return entry is not None and getattr(tool, "on_invoke_tool", None) is entry[2]

    def record_call_meta(self, tool: Any, tool_call_id: Any, agent_name: Any) -> None:
        # Only bridged tools consume call metadata; never retain it for shell/custom/
        # agent-as-tool calls, so a long run does not grow _call_meta until context exit.
        if not self.is_bridged(tool):
            return
        self._call_meta[(id(tool), tool_call_id)] = {
            "agent": agent_name,
            "tool_call_id": tool_call_id,
        }

    def discard_call_meta(self, tool: Any, tool_call_id: Any) -> None:
        self._call_meta.pop((id(tool), tool_call_id), None)

    def _pop_call_meta(self, tool: Any, tool_call_id: Any, context: Any) -> dict[str, Any]:
        meta = self._call_meta.pop((id(tool), tool_call_id), None)
        if meta is not None:
            return meta
        agent = getattr(context, "agent", None)
        return {
            "agent": _name(agent) if agent is not None else None,
            "tool_call_id": tool_call_id,
        }

    def restore_all(self) -> None:
        for tool, original, wrapper in self._patched.values():
            # Do not overwrite a callback that user code replaced after installation.
            if getattr(tool, "on_invoke_tool", None) is wrapper:
                setattr(tool, "on_invoke_tool", original)  # noqa: B010 -- attribute name is fixed
        self._patched.clear()
        self._call_meta.clear()

    def _make_wrapper(self, tool: Any, original: Any) -> Any:
        cassette = self._cassette
        structured_types = self._structured_types
        tool_name = _name(tool)

        @wraps(original)
        async def bridged(context: Any, arguments: Any) -> Any:
            tool_call_id = getattr(context, "tool_call_id", None)
            meta = self._pop_call_meta(tool, tool_call_id, context)
            result_input = serialize_recorded_value(
                {"agent": meta["agent"], "tool_call_id": meta["tool_call_id"]}
            )
            executed = {"live": False}

            async def live_thunk() -> Any:
                executed["live"] = True
                return await original(context, arguments)

            recorded = await cassette.acall(
                EventType.TOOL_RESULT,
                tool_name,
                result_input,
                live_thunk,
                metadata={"provider": "openai-agents", "tool_bridge": True, "stage": "invoke"},
                serializer=lambda value: _encode_agents_tool_result(value, structured_types),
            )
            if executed["live"]:
                # Live record / Hybrid live transition: preserve the real object identity.
                return recorded
            return _decode_agents_tool_result(recorded, structured_types)

        return bridged


# --------------------------------------------------------------------------- #
# Run hooks
# --------------------------------------------------------------------------- #


class AgentCassetteRunHooks:
    """Duck-typed OpenAI Agents ``RunHooks`` implementation."""

    def __init__(self, cassette: Any, bridge: _AgentsToolBridgeState | None = None) -> None:
        self.cassette = cassette
        self._bridge = bridge

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        # Patch after user hooks (which run first) so tools they add are bridged
        # before the first tool executes.
        if self._bridge is not None:
            self._bridge.patch_agent(agent)
        self._add(EventType.CUSTOM, "agent.start", input={"agent": _name(agent)})

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.end",
            input={"agent": _name(agent)},
            output=output,
        )

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        if self._bridge is not None:
            self._bridge.patch_agent(to_agent)
        self._add(
            EventType.CUSTOM,
            "agent.handoff",
            input={"from": _name(from_agent), "to": _name(to_agent)},
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_call_id = getattr(context, "tool_call_id", None)
        if self._bridge is not None:
            self._bridge.record_call_meta(tool, tool_call_id, _name(agent))
        self._add(
            EventType.TOOL_CALL,
            _name(tool),
            input={
                "agent": _name(agent),
                "arguments": getattr(context, "tool_arguments", None),
                "tool_call_id": tool_call_id,
            },
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        # A bridged FunctionTool already recorded its TOOL_RESULT at the callback
        # boundary; do not write a second one. Non-bridged tools keep lifecycle capture.
        tool_call_id = getattr(context, "tool_call_id", None)
        if self._bridge is not None:
            # Defensively drop any lingering call-state key before returning/recording.
            self._bridge.discard_call_meta(tool, tool_call_id)
            if self._bridge.is_bridged(tool):
                return
        self._add(
            EventType.TOOL_RESULT,
            _name(tool),
            input={"agent": _name(agent), "tool_call_id": tool_call_id},
            output=result,
        )

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: str | None, input_items: list[Any]
    ) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.llm.start",
            input={
                "agent": _name(agent),
                "system_prompt": system_prompt,
                "items": input_items,
            },
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.llm.end",
            input={"agent": _name(agent)},
            output=response,
        )

    def _add(self, event_type: EventType, name: str, **values: Any) -> None:
        metadata = dict(values.pop("metadata", {}))
        metadata.update({"provider": "openai-agents", "lifecycle": True})
        input_value = values.pop("input", None)
        output_value = values.pop("output", None)
        # Serialize (and validate) the COMPLETE input payload once here, and the
        # output once via the serializer, so SDK values are dumped exactly once
        # (no double-dump) and an invalid payload fails before it is persisted.
        serialized_input = _to_data(input_value) if input_value is not None else None
        self.cassette.call(
            event_type,
            name,
            serialized_input,
            lambda: output_value,
            metadata=metadata,
            serializer=_to_data,
        )


class _CompositeRunHooks:
    def __init__(self, *hooks: Any) -> None:
        self.hooks = hooks

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("on_"):
            raise AttributeError(name)

        async def dispatch(*args: Any, **kwargs: Any) -> None:
            for hook in self.hooks:
                callback = getattr(hook, name, None)
                if callback is not None:
                    result = callback(*args, **kwargs)
                    if inspect.isawaitable(result):
                        await result

        return dispatch


@contextmanager
def patch_openai_agents(cassette: Any) -> Iterator[None]:
    """Inject cassette lifecycle hooks and the FunctionTool replay bridge for one context."""
    try:
        agents = importlib.import_module("agents")
        runner = agents.Runner
    except (ModuleNotFoundError, AttributeError) as error:
        raise OpenAIAgentsUnavailableError(
            "OpenAI Agents capture requires `pip install agent-cassette[agents]`."
        ) from error

    function_tool_cls = getattr(agents, "FunctionTool", None)
    structured_types = _resolve_structured_classes(agents)
    bridge = _AgentsToolBridgeState(cassette, function_tool_cls, structured_types)

    originals: dict[str, Any] = {}
    for method_name in ("run", "run_sync", "run_streamed"):
        original = getattr(runner, method_name, None)
        if original is None:
            continue
        originals[method_name] = inspect.getattr_static(runner, method_name)
        setattr(runner, method_name, staticmethod(_runner_wrapper(original, cassette, bridge)))
    try:
        yield
    finally:
        for method_name, descriptor in originals.items():
            setattr(runner, method_name, descriptor)
        bridge.restore_all()


def _runner_wrapper(original: Any, cassette: Any, bridge: _AgentsToolBridgeState) -> Any:
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def async_run(*args: Any, **kwargs: Any) -> Any:
            bridge.patch_agent(_starting_agent(args, kwargs))
            kwargs["hooks"] = _merge_hooks(kwargs.get("hooks"), cassette, bridge)
            return await original(*args, **kwargs)

        return async_run

    @wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        bridge.patch_agent(_starting_agent(args, kwargs))
        kwargs["hooks"] = _merge_hooks(kwargs.get("hooks"), cassette, bridge)
        return original(*args, **kwargs)

    return run


def _starting_agent(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("starting_agent")


def _merge_hooks(existing: Any, cassette: Any, bridge: _AgentsToolBridgeState) -> Any:
    hooks = AgentCassetteRunHooks(cassette, bridge)
    return hooks if existing is None else _CompositeRunHooks(existing, hooks)


def _name(value: Any) -> str:
    return str(getattr(value, "name", type(value).__name__))


__all__ = [
    "AgentCassetteRunHooks",
    "OpenAIAgentsUnavailableError",
    "patch_openai_agents",
]
