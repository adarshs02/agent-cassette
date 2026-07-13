"""Lifecycle capture and automatic hook injection for the OpenAI Agents SDK."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from agent_cassette.events import EventType


class OpenAIAgentsUnavailableError(ImportError):
    """Raised when OpenAI Agents integration is requested without the SDK."""


class AgentCassetteRunHooks:
    """Duck-typed OpenAI Agents ``RunHooks`` implementation."""

    def __init__(self, cassette: Any) -> None:
        self.cassette = cassette

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        self._add(EventType.CUSTOM, "agent.start", input={"agent": _name(agent)})

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.end",
            input={"agent": _name(agent)},
            output=_to_data(output),
        )

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.handoff",
            input={"from": _name(from_agent), "to": _name(to_agent)},
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self._add(
            EventType.TOOL_CALL,
            _name(tool),
            input={
                "agent": _name(agent),
                "arguments": _to_data(getattr(context, "tool_arguments", None)),
                "tool_call_id": getattr(context, "tool_call_id", None),
            },
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self._add(
            EventType.TOOL_RESULT,
            _name(tool),
            input={"agent": _name(agent), "tool_call_id": getattr(context, "tool_call_id", None)},
            output=_to_data(result),
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
                "items": _to_data(input_items),
            },
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self._add(
            EventType.CUSTOM,
            "agent.llm.end",
            input={"agent": _name(agent)},
            output=_to_data(response),
        )

    def _add(self, event_type: EventType, name: str, **values: Any) -> None:
        metadata = dict(values.pop("metadata", {}))
        metadata.update({"provider": "openai-agents", "lifecycle": True})
        input_value = values.pop("input", None)
        output_value = values.pop("output", None)
        self.cassette.call(
            event_type,
            name,
            input_value,
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
    """Inject cassette lifecycle hooks into ``Runner`` methods for one context."""
    try:
        agents = importlib.import_module("agents")
        runner = agents.Runner
    except (ModuleNotFoundError, AttributeError) as error:
        raise OpenAIAgentsUnavailableError(
            "OpenAI Agents capture requires `pip install agent-cassette[agents]`."
        ) from error

    originals: dict[str, Any] = {}
    for method_name in ("run", "run_sync", "run_streamed"):
        original = getattr(runner, method_name, None)
        if original is None:
            continue
        originals[method_name] = inspect.getattr_static(runner, method_name)
        setattr(runner, method_name, staticmethod(_runner_wrapper(original, cassette)))
    try:
        yield
    finally:
        for method_name, descriptor in originals.items():
            setattr(runner, method_name, descriptor)


def _runner_wrapper(original: Any, cassette: Any) -> Any:
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def async_run(*args: Any, **kwargs: Any) -> Any:
            kwargs["hooks"] = _merge_hooks(kwargs.get("hooks"), cassette)
            return await original(*args, **kwargs)

        return async_run

    @wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        kwargs["hooks"] = _merge_hooks(kwargs.get("hooks"), cassette)
        return original(*args, **kwargs)

    return run


def _merge_hooks(existing: Any, cassette: Any) -> Any:
    hooks = AgentCassetteRunHooks(cassette)
    return hooks if existing is None else _CompositeRunHooks(existing, hooks)


def _name(value: Any) -> str:
    return str(getattr(value, "name", type(value).__name__))


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


__all__ = [
    "AgentCassetteRunHooks",
    "OpenAIAgentsUnavailableError",
    "patch_openai_agents",
]
