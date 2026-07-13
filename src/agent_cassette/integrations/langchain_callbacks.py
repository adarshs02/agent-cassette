"""Optional LangChain lifecycle tracing callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agent_cassette.events import EventType
from agent_cassette.integrations.langchain import _encode

_INTERNAL_METADATA_KEY = "_agent_cassette"
_PARSER_HINTS = ("output_parser", "outputparser", "parser")


@dataclass(frozen=True, slots=True)
class _Run:
    category: str
    name: str
    input: Any
    parent_id: str | None
    started: float
    tags: Any
    metadata: Any


class AgentCassetteCallbackHandler(BaseCallbackHandler):
    """Record one observational event when each supported LangChain run finishes."""

    def __init__(self, cassette: Any) -> None:
        super().__init__()
        self._cassette = cassette
        self._enabled = callable(getattr(cassette, "add", None))
        self._runs: dict[str, _Run] = {}
        self._lock = RLock()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        category = _chain_category(serialized, metadata, kwargs.get("name"))
        self._start(
            category,
            serialized,
            inputs,
            run_id,
            parent_run_id,
            tags,
            metadata,
            kwargs.get("name"),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "success", outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "error", _safe_error(error))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            "model",
            serialized,
            prompts,
            run_id,
            parent_run_id,
            tags,
            metadata,
            kwargs.get("name"),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            "model",
            serialized,
            messages,
            run_id,
            parent_run_id,
            tags,
            metadata,
            kwargs.get("name"),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "success", response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "error", _safe_error(error))

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            "retriever",
            serialized,
            query,
            run_id,
            parent_run_id,
            tags,
            metadata,
            kwargs.get("name"),
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "success", documents)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "error", _safe_error(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            "tool",
            serialized,
            inputs if inputs is not None else input_str,
            run_id,
            parent_run_id,
            tags,
            metadata,
            kwargs.get("name"),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "success", output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, parent_run_id, "error", _safe_error(error))

    def _start(
        self,
        category: str,
        serialized: dict[str, Any] | None,
        input_value: Any,
        run_id: UUID,
        parent_run_id: UUID | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        callback_name: Any,
    ) -> None:
        if not self._enabled:
            return
        key = str(run_id)
        run = _Run(
            category=category,
            name=_run_name(category, serialized, callback_name),
            input=_safe_value(input_value),
            parent_id=str(parent_run_id) if parent_run_id is not None else None,
            started=perf_counter(),
            tags=_safe_value(tags or []),
            metadata=_safe_value(metadata or {}),
        )
        with self._lock:
            self._runs.setdefault(key, run)

    def _finish(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        outcome: str,
        output: Any,
    ) -> None:
        if not self._enabled:
            return
        key = str(run_id)
        with self._lock:
            run = self._runs.pop(key, None)
            if run is None:
                return
        parent_id = run.parent_id
        if parent_id is None and parent_run_id is not None:
            parent_id = str(parent_run_id)
        metadata = {
            "integration": "langchain",
            "lifecycle": True,
            "category": run.category,
            "outcome": outcome,
            "run_name": run.name,
            "tags": run.tags,
            "langchain_metadata": run.metadata,
            _INTERNAL_METADATA_KEY: {"observational": True},
        }
        self._cassette.add(
            EventType.CUSTOM,
            f"langchain.{run.category}.{outcome}",
            input=run.input,
            output=output if outcome == "error" else _safe_value(output),
            metadata=metadata,
            duration_ms=(perf_counter() - run.started) * 1000,
            span_id=key,
            parent_id=parent_id,
        )


def langchain_callback_handler(cassette: Any) -> BaseCallbackHandler:
    """Return an optional callback handler for LangChain lifecycle events."""
    return AgentCassetteCallbackHandler(cassette)


def _safe_value(value: Any) -> Any:
    try:
        return _encode(value)
    except (TypeError, ValueError, RecursionError):
        value_type = type(value)
        module = value_type.__module__
        return {
            "__agent_cassette_unserializable__": {
                "module": module if isinstance(module, str) else "unknown",
                "type": value_type.__qualname__,
            }
        }


def _safe_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _chain_category(
    serialized: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    callback_name: Any,
) -> str:
    hints: list[str] = []
    if isinstance(callback_name, str):
        hints.append(callback_name)
    if isinstance(serialized, dict):
        for key in ("name", "id"):
            value = serialized.get(key)
            if isinstance(value, str):
                hints.append(value)
            elif isinstance(value, list):
                hints.extend(item for item in value if isinstance(item, str))
    if isinstance(metadata, dict):
        for key in ("name", "run_type", "type"):
            value = metadata.get(key)
            if isinstance(value, str):
                hints.append(value)
    combined = " ".join(hints).lower().replace("-", "_")
    return "parser" if any(hint in combined for hint in _PARSER_HINTS) else "chain"


def _run_name(category: str, serialized: dict[str, Any] | None, callback_name: Any) -> str:
    if isinstance(callback_name, str) and callback_name:
        return callback_name
    if isinstance(serialized, dict):
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        identifiers = serialized.get("id")
        if isinstance(identifiers, list):
            for identifier in reversed(identifiers):
                if isinstance(identifier, str) and identifier:
                    return identifier
    return category
