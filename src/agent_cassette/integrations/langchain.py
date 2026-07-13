"""Safe record and replay support for LangChain ``Runnable`` objects.

This module is intentionally imported lazily by :mod:`agent_cassette` so the
core package remains usable when ``langchain-core`` is not installed.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import AsyncIterator, Iterator, Mapping
from time import perf_counter
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from agent_cassette.events import EventType
from agent_cassette.hybrid import Hybrid
from agent_cassette.replay import RateLimitError, RecordedCallError, Replayer

_MARKER = "__agent_cassette_langchain__"
_SERIALIZER_VERSION = 1
_MAX_DEPTH = 64
_OMITTED_CONFIG_KEYS = frozenset({"callbacks", "run_id", "run_name"})


class LangChainSerializationError(ValueError):
    """Raised when a value cannot be represented safely in a cassette."""


class LangChainReplayIntrospectionError(RuntimeError):
    """Raised when live-only Runnable introspection is requested during replay."""


_LANGCHAIN_TYPES: dict[type[Any], str] = {
    HumanMessage: "HumanMessage",
    AIMessage: "AIMessage",
    SystemMessage: "SystemMessage",
    ChatMessage: "ChatMessage",
    FunctionMessage: "FunctionMessage",
    ToolMessage: "ToolMessage",
    HumanMessageChunk: "HumanMessageChunk",
    AIMessageChunk: "AIMessageChunk",
    SystemMessageChunk: "SystemMessageChunk",
    ChatMessageChunk: "ChatMessageChunk",
    FunctionMessageChunk: "FunctionMessageChunk",
    ToolMessageChunk: "ToolMessageChunk",
    Document: "Document",
    Generation: "Generation",
    GenerationChunk: "GenerationChunk",
    ChatGeneration: "ChatGeneration",
    ChatGenerationChunk: "ChatGenerationChunk",
    LLMResult: "LLMResult",
}
_LANGCHAIN_CLASSES = {name: value_type for value_type, name in _LANGCHAIN_TYPES.items()}
_EXCEPTION_TYPES: dict[type[BaseException], str] = {
    ConnectionError: "ConnectionError",
    RateLimitError: "RateLimitError",
    RuntimeError: "RuntimeError",
    TimeoutError: "TimeoutError",
    ValueError: "ValueError",
    asyncio.CancelledError: "CancelledError",
}


def _envelope(value_type: str, data: Any) -> dict[str, Any]:
    return {
        _MARKER: True,
        "version": _SERIALIZER_VERSION,
        "type": value_type,
        "data": data,
    }


def _encode(value: Any) -> Any:
    return _encode_at(value, depth=0, active=set(), path="$ ".rstrip())


def _encode_at(value: Any, *, depth: int, active: set[int], path: str) -> Any:
    if depth > _MAX_DEPTH:
        raise LangChainSerializationError(
            f"maximum serialization depth {_MAX_DEPTH} exceeded at {path}"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LangChainSerializationError(f"non-finite float at {path}")
        return value

    value_id = id(value)
    if value_id in active:
        raise LangChainSerializationError(f"cyclic value at {path}")

    exact_type = type(value)
    if exact_type in _LANGCHAIN_TYPES:
        active.add(value_id)
        try:
            dump = getattr(value, "model_dump", None)
            if not callable(dump):
                raise LangChainSerializationError(
                    f"{exact_type.__name__} at {path} does not support model_dump"
                )
            model_data = {
                field_name: getattr(value, field_name) for field_name in exact_type.model_fields
            }
            if exact_type is LLMResult:
                # RunInfo is an internal child of LLMResult. Keep it as
                # validated data; concrete Generation values remain typed.
                model_data["run"] = dump(mode="python").get("run")
            return _envelope(
                _LANGCHAIN_TYPES[exact_type],
                _encode_at(model_data, depth=depth + 1, active=active, path=f"{path}.data"),
            )
        finally:
            active.remove(value_id)

    if exact_type in _EXCEPTION_TYPES:
        exception_data: dict[str, Any] = {"message": str(value)}
        if exact_type is RateLimitError:
            exception_data["retry_after"] = cast(RateLimitError, value).retry_after
        return _envelope(_EXCEPTION_TYPES[exact_type], exception_data)

    if isinstance(value, Mapping):
        active.add(value_id)
        try:
            pairs: list[dict[str, Any]] = []
            encoded_dict: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise LangChainSerializationError(
                        f"dictionary key at {path} must be a string, got {type(key).__name__}"
                    )
                encoded = _encode_at(
                    item,
                    depth=depth + 1,
                    active=active,
                    path=f"{path}.{key}",
                )
                # One-entry dictionaries keep the original key visible to the
                # Recorder's recursive redactor while avoiding marker clashes.
                pairs.append({key: encoded})
                encoded_dict[key] = encoded
            # A user dictionary containing our marker must not be confused with
            # a code-owned envelope when replayed.
            return _envelope("dict", pairs) if _MARKER in value else encoded_dict
        finally:
            active.remove(value_id)

    if isinstance(value, list):
        active.add(value_id)
        try:
            return [
                _encode_at(item, depth=depth + 1, active=active, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(value_id)

    if isinstance(value, tuple):
        active.add(value_id)
        try:
            return _envelope(
                "tuple",
                [
                    _encode_at(item, depth=depth + 1, active=active, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                ],
            )
        finally:
            active.remove(value_id)

    raise LangChainSerializationError(
        f"unsupported value of type {exact_type.__module__}.{exact_type.__qualname__} at {path}"
    )


def _decode(value: Any) -> Any:
    return _decode_at(value, depth=0, path="$ ".rstrip())


def _decode_at(value: Any, *, depth: int, path: str) -> Any:
    if depth > _MAX_DEPTH:
        raise LangChainSerializationError(
            f"maximum deserialization depth {_MAX_DEPTH} exceeded at {path}"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LangChainSerializationError(f"non-finite float at {path}")
        return value
    if isinstance(value, list):
        return [
            _decode_at(item, depth=depth + 1, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        raise LangChainSerializationError(
            f"invalid cassette value of type {type(value).__name__} at {path}"
        )
    if _MARKER not in value:
        if not all(isinstance(key, str) for key in value):
            raise LangChainSerializationError(f"dictionary key at {path} must be a string")
        return {
            key: _decode_at(item, depth=depth + 1, path=f"{path}.{key}")
            for key, item in value.items()
        }

    required = {_MARKER, "version", "type", "data"}
    if set(value) != required or value.get(_MARKER) is not True:
        raise LangChainSerializationError(f"malformed LangChain envelope at {path}")
    if value.get("version") != _SERIALIZER_VERSION:
        raise LangChainSerializationError(
            f"unsupported LangChain envelope version {value.get('version')!r} at {path}"
        )
    encoded_type = value.get("type")
    if not isinstance(encoded_type, str):
        raise LangChainSerializationError(f"LangChain envelope type at {path} must be a string")

    # Decode escaped user dictionaries before generic traversal. In particular,
    # the one-entry mapping whose key is our marker is user data, not an envelope.
    if encoded_type == "dict":
        raw_data = value.get("data")
        if not isinstance(raw_data, list):
            raise LangChainSerializationError(f"dict envelope data at {path} must be a list")
        restored: dict[str, Any] = {}
        for index, entry in enumerate(raw_data):
            if not isinstance(entry, dict) or len(entry) != 1:
                raise LangChainSerializationError(
                    f"dict envelope entry {index} at {path} is invalid"
                )
            key, item = next(iter(entry.items()))
            if not isinstance(key, str):
                raise LangChainSerializationError(
                    f"dict envelope key {index} at {path} must be a string"
                )
            restored[key] = _decode_at(item, depth=depth + 1, path=f"{path}.data[{index}].{key}")
        return restored

    data = _decode_at(value.get("data"), depth=depth + 1, path=f"{path}.data")

    if encoded_type == "tuple":
        if not isinstance(data, list):
            raise LangChainSerializationError(f"tuple envelope data at {path} must be a list")
        return tuple(data)
    if encoded_type in _LANGCHAIN_CLASSES:
        if not isinstance(data, dict):
            raise LangChainSerializationError(
                f"{encoded_type} envelope data at {path} must be a dictionary"
            )
        try:
            return _LANGCHAIN_CLASSES[encoded_type](**data)
        except (TypeError, ValueError) as error:
            raise LangChainSerializationError(
                f"invalid {encoded_type} envelope data at {path}: {error}"
            ) from error
    if encoded_type in {"ConnectionError", "RuntimeError", "TimeoutError", "ValueError"}:
        if not isinstance(data, dict) or not isinstance(data.get("message"), str):
            raise LangChainSerializationError(f"{encoded_type} envelope data at {path} is invalid")
        exception_classes = {
            "ConnectionError": ConnectionError,
            "RuntimeError": RuntimeError,
            "TimeoutError": TimeoutError,
            "ValueError": ValueError,
        }
        return exception_classes[encoded_type](data["message"])
    if encoded_type == "RateLimitError":
        if not isinstance(data, dict) or not isinstance(data.get("message"), str):
            raise LangChainSerializationError(f"RateLimitError envelope data at {path} is invalid")
        retry_after = data.get("retry_after")
        if retry_after is not None and not isinstance(retry_after, (int, float)):
            raise LangChainSerializationError(
                f"RateLimitError retry_after at {path} must be a number or null"
            )
        return RateLimitError(data["message"], retry_after=retry_after)
    if encoded_type == "CancelledError":
        if not isinstance(data, dict) or not isinstance(data.get("message"), str):
            raise LangChainSerializationError(f"CancelledError envelope data at {path} is invalid")
        return asyncio.CancelledError(data["message"])
    raise LangChainSerializationError(
        f"unsupported LangChain envelope type {encoded_type!r} at {path}"
    )


def _clean_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, list):
        return [_clean_config(item) for item in config]
    if not isinstance(config, Mapping):
        raise LangChainSerializationError(
            f"Runnable config must be a dictionary, list, or None, got {type(config).__name__}"
        )
    return {key: value for key, value in config.items() if key not in _OMITTED_CONFIG_KEYS}


def _request(
    inputs: Any,
    config: Any,
    kwargs: dict[str, Any],
    *,
    return_exceptions: bool | None = None,
) -> dict[str, Any]:
    request = {
        "inputs": _encode(inputs),
        "config": _encode(_clean_config(config)),
        "kwargs": _encode(kwargs),
    }
    if return_exceptions is not None:
        request["return_exceptions"] = return_exceptions
    return request


def _metadata(operation: str, *, streaming: bool = False) -> dict[str, Any]:
    return {"integration": "langchain", "operation": operation, "streaming": streaming}


def _error_metadata(operation: str) -> dict[str, Any]:
    metadata = _metadata(operation, streaming=True)
    metadata["_agent_cassette"] = {"call_type": EventType.CUSTOM.value}
    return metadata


def _stream_output(
    chunks: list[Any],
    error: BaseException | None = None,
    *,
    phase: str = "iteration",
) -> dict[str, Any]:
    output: dict[str, Any] = {
        _MARKER: True,
        "version": _SERIALIZER_VERSION,
        "type": "stream",
        "data": {"chunks": [_encode(chunk) for chunk in chunks]},
    }
    if error is not None:
        if phase not in {"start", "iteration"}:
            raise LangChainSerializationError(f"invalid LangChain stream failure phase {phase!r}")
        output["data"]["phase"] = phase
        error_type = type(error)
        if error_type in _EXCEPTION_TYPES:
            output["data"]["error"] = _encode(error)
        else:
            output["data"]["error"] = _envelope(
                "RecordedCallError",
                {"recorded_type": error_type.__name__, "message": str(error)},
            )
    return output


def _restore_stream(value: Any) -> tuple[list[Any], BaseException | None]:
    if (
        not isinstance(value, dict)
        or set(value) != {_MARKER, "version", "type", "data"}
        or value.get(_MARKER) is not True
        or value.get("version") != _SERIALIZER_VERSION
        or value.get("type") != "stream"
        or not isinstance(value.get("data"), dict)
    ):
        raise LangChainSerializationError("recorded LangChain stream payload is invalid")
    data = value["data"]
    data_keys = set(data)
    if data_keys not in (
        {"chunks"},
        {"chunks", "error"},
        {"chunks", "error", "phase"},
    ):
        raise LangChainSerializationError("recorded LangChain stream data has unknown keys")
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise LangChainSerializationError("recorded LangChain stream chunks must be a list")
    restored_chunks = [_decode(chunk) for chunk in chunks]
    error_payload = data.get("error")
    phase = data.get("phase", "iteration")
    if phase not in {"start", "iteration"}:
        raise LangChainSerializationError("recorded LangChain stream phase is invalid")
    if phase == "start" and chunks:
        raise LangChainSerializationError(
            "recorded LangChain start failure cannot contain stream chunks"
        )
    if error_payload is None:
        return restored_chunks, None
    if (
        isinstance(error_payload, dict)
        and set(error_payload) == {_MARKER, "version", "type", "data"}
        and error_payload.get(_MARKER) is True
        and error_payload.get("version") == _SERIALIZER_VERSION
        and error_payload.get("type") == "RecordedCallError"
        and isinstance(error_payload.get("data"), dict)
    ):
        error_data = error_payload["data"]
        if set(error_data) != {"recorded_type", "message"} or not all(
            isinstance(error_data.get(key), str) for key in ("recorded_type", "message")
        ):
            raise LangChainSerializationError(
                "recorded LangChain RecordedCallError data is invalid"
            )
        return restored_chunks, RecordedCallError(
            error_data["recorded_type"],
            error_data["message"],
        )
    if isinstance(error_payload, dict) and error_payload.get("type") == "RecordedCallError":
        raise LangChainSerializationError(
            "recorded LangChain RecordedCallError envelope is invalid"
        )
    restored_error = _decode(error_payload)
    if not isinstance(restored_error, BaseException):
        raise LangChainSerializationError("recorded LangChain stream error is invalid")
    return restored_chunks, restored_error


def _stream_phase(value: dict[str, Any]) -> str:
    data = cast(dict[str, Any], value["data"])
    return cast(str, data.get("phase", "iteration"))


def _prepare_stream(
    cassette: Any, name: str, request: dict[str, Any], operation: str
) -> tuple[bool, Any]:
    metadata = _metadata(operation, streaming=True)
    prepare = getattr(cassette, "prepare_stream", None)
    if callable(prepare):
        prepare_kwargs: dict[str, Any] = {"metadata": metadata}
        if isinstance(cassette, Hybrid):
            prepare_kwargs["serializer"] = _normalize_injected_stream
            prepare_kwargs["error_serializer"] = lambda error: _stream_output(
                [], error, phase="start"
            )
        replayed, output = cast(
            tuple[bool, Any],
            prepare(EventType.CUSTOM, name, request, **prepare_kwargs),
        )
        return replayed, output
    consume = getattr(cassette, "consume", None)
    if callable(consume) and hasattr(cassette, "position"):
        event = consume(EventType.CUSTOM, name, request)
        return True, event.output
    return False, None


def _hybrid_mode(cassette: Hybrid) -> str | None:
    if not cassette.recorder.events:
        return None
    lineage = cassette.recorder.events[-1].metadata.get("_agent_cassette")
    return lineage.get("mode") if isinstance(lineage, dict) else None


def _normalize_injected_stream(output: Any) -> dict[str, Any]:
    if (
        isinstance(output, dict)
        and set(output) == {_MARKER, "version", "type", "data"}
        and output.get(_MARKER) is True
        and output.get("version") == _SERIALIZER_VERSION
        and output.get("type") == "stream"
    ):
        _restore_stream(output)
        return output
    chunks = list(output) if isinstance(output, (list, tuple)) else [output]
    return _stream_output(chunks)


def _restore_call_result(cassette: Any, result: Any, *, executed: bool) -> Any:
    if executed:
        return result
    if isinstance(cassette, Hybrid):
        return _decode(result) if _hybrid_mode(cassette) == "replayed" else result
    if isinstance(cassette, Replayer):
        return _decode(result)
    # Custom replay implementations follow the Replayer contract and return
    # serialized data without executing the supplied callable.
    return _decode(result)


class _ReplayStream(Iterator[Any]):
    def __init__(self, chunks: list[Any], error: BaseException | None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __iter__(self) -> _ReplayStream:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                error, self._error = self._error, None
                raise error from None
            raise

    def close(self) -> None:
        self._error = None


class _RecordingStream(Iterator[Any]):
    def __init__(
        self,
        stream: Iterator[Any],
        cassette: Any,
        name: str,
        request: dict[str, Any],
        operation: str,
        started: float,
    ) -> None:
        self._stream = stream
        self._cassette = cassette
        self._name = name
        self._request = request
        self._operation = operation
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __iter__(self) -> _RecordingStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            self._fail(error)
            raise
        self._chunks.append(chunk)
        return chunk

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        except Exception as error:
            self._fail(error)
            raise
        else:
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        output = _stream_output(self._chunks)
        self._cassette.add(
            EventType.CUSTOM,
            self._name,
            input=self._request,
            output=output,
            metadata=_metadata(self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )
        self._finished = True

    def _fail(self, error: BaseException) -> None:
        if self._finished:
            return
        output = _stream_output(self._chunks, error)
        self._cassette.add(
            EventType.ERROR,
            self._name,
            input=self._request,
            output=output,
            metadata=_error_metadata(self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )
        self._finished = True


class _AsyncReplayStream(AsyncIterator[Any]):
    def __init__(self, chunks: list[Any], error: BaseException | None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __aiter__(self) -> _AsyncReplayStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as stop:
            if self._error is not None:
                error, self._error = self._error, None
                raise error from None
            raise StopAsyncIteration from stop

    async def aclose(self) -> None:
        self._error = None


class _AsyncRecordingStream(AsyncIterator[Any]):
    def __init__(
        self,
        stream: AsyncIterator[Any],
        cassette: Any,
        name: str,
        request: dict[str, Any],
        operation: str,
        started: float,
    ) -> None:
        self._stream = stream
        self._cassette = cassette
        self._name = name
        self._request = request
        self._operation = operation
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __aiter__(self) -> _AsyncRecordingStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except asyncio.CancelledError as error:
            self._fail(error)
            raise
        except Exception as error:
            self._fail(error)
            raise
        self._chunks.append(chunk)
        return chunk

    async def aclose(self) -> None:
        try:
            close = getattr(self._stream, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except asyncio.CancelledError as error:
            self._fail(error)
            raise
        except Exception as error:
            self._fail(error)
            raise
        else:
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        output = _stream_output(self._chunks)
        self._cassette.add(
            EventType.CUSTOM,
            self._name,
            input=self._request,
            output=output,
            metadata=_metadata(self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )
        self._finished = True

    def _fail(self, error: BaseException) -> None:
        if self._finished:
            return
        output = _stream_output(self._chunks, error)
        self._cassette.add(
            EventType.ERROR,
            self._name,
            input=self._request,
            output=output,
            metadata=_error_metadata(self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )
        self._finished = True


class LangChainRunnable(Runnable[Any, Any]):
    """Runnable-compatible record/replay wrapper."""

    def __init__(self, runnable: Runnable[Any, Any] | None, cassette: Any, name: str) -> None:
        self._runnable = runnable
        self._cassette = cassette
        self._cassette_name = name

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        runnable = self.__dict__.get("_runnable")
        if runnable is None:
            raise AttributeError(
                f"Runnable attribute {attribute!r} is unavailable during offline replay"
            )
        value = getattr(runnable, attribute)
        if callable(value):
            raise AttributeError(
                f"Callable Runnable attribute {attribute!r} cannot bypass cassette capture"
            )
        return value

    def _live(self) -> Runnable[Any, Any]:
        if self._runnable is None:
            raise RuntimeError("LangChain replay unexpectedly attempted live Runnable execution")
        return self._runnable

    def _introspection(self) -> Runnable[Any, Any]:
        if self._runnable is None:
            raise LangChainReplayIntrospectionError(
                "Runnable introspection is unavailable during offline replay "
                "without a live Runnable"
            )
        return self._runnable

    @property
    def InputType(self) -> Any:  # noqa: N802
        return self._introspection().InputType

    @property
    def OutputType(self) -> Any:  # noqa: N802
        return self._introspection().OutputType

    @property
    def config_specs(self) -> list[Any]:
        return self._introspection().config_specs

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        return self._introspection().get_name(suffix, name=name)

    def get_input_schema(self, config: RunnableConfig | None = None) -> type[Any]:
        return self._introspection().get_input_schema(config)

    def get_output_schema(self, config: RunnableConfig | None = None) -> type[Any]:
        return self._introspection().get_output_schema(config)

    def get_graph(self, config: RunnableConfig | None = None) -> Any:
        return self._introspection().get_graph(config)

    def get_prompts(self, config: RunnableConfig | None = None) -> list[Any]:
        return self._introspection().get_prompts(config)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return self._call(
            "invoke", input, config, kwargs, lambda: self._live().invoke(input, config, **kwargs)
        )

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        request = _request(input, config, kwargs)
        executed = False

        async def live() -> Any:
            nonlocal executed
            executed = True
            return await self._live().ainvoke(input, config, **kwargs)

        result = await self._cassette.acall(
            EventType.CUSTOM,
            f"{self._cassette_name}.ainvoke",
            request,
            live,
            metadata=_metadata("ainvoke"),
            serializer=_encode,
        )
        return _restore_call_result(self._cassette, result, executed=executed)

    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        operation = "stream"
        request = _request(input, config, kwargs)
        name = f"{self._cassette_name}.{operation}"
        replayed, output = _prepare_stream(self._cassette, name, request, operation)
        if replayed:
            chunks, error = _restore_stream(output)
            if error is not None and _stream_phase(output) == "start":
                raise error from None
            return _ReplayStream(chunks, error)
        started = perf_counter()
        try:
            stream = iter(self._live().stream(input, config, **kwargs))
        except Exception as error:
            self._record_start_error(name, request, operation, error, started)
            raise
        return _RecordingStream(stream, self._cassette, name, request, operation, started)

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        operation = "astream"
        request = _request(input, config, kwargs)
        name = f"{self._cassette_name}.{operation}"
        replayed, output = _prepare_stream(self._cassette, name, request, operation)
        if replayed:
            inner: AsyncIterator[Any] = _AsyncReplayStream(*_restore_stream(output))
        else:
            started = perf_counter()
            try:
                live_stream = self._live().astream(input, config, **kwargs)
                inner = _AsyncRecordingStream(
                    live_stream.__aiter__(), self._cassette, name, request, operation, started
                )
            except Exception as error:
                self._record_start_error(name, request, operation, error, started)
                raise
        try:
            async for chunk in inner:
                yield chunk
        finally:
            close = getattr(inner, "aclose", None)
            if callable(close):
                await close()

    def batch(
        self,
        inputs: list[Any],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        result = self._call(
            "batch",
            inputs,
            config,
            kwargs,
            lambda: self._live().batch(
                inputs, config, return_exceptions=return_exceptions, **kwargs
            ),
            return_exceptions=return_exceptions,
        )
        return cast(list[Any], result)

    async def abatch(
        self,
        inputs: list[Any],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        request = _request(inputs, config, kwargs, return_exceptions=return_exceptions)
        executed = False

        async def live() -> list[Any]:
            nonlocal executed
            executed = True
            return await self._live().abatch(
                inputs, config, return_exceptions=return_exceptions, **kwargs
            )

        result = await self._cassette.acall(
            EventType.CUSTOM,
            f"{self._cassette_name}.abatch",
            request,
            live,
            metadata=_metadata("abatch"),
            serializer=_encode,
        )
        return cast(list[Any], _restore_call_result(self._cassette, result, executed=executed))

    def _call(
        self,
        operation: str,
        inputs: Any,
        config: Any,
        kwargs: dict[str, Any],
        live: Any,
        *,
        return_exceptions: bool | None = None,
    ) -> Any:
        request = _request(inputs, config, kwargs, return_exceptions=return_exceptions)
        executed = False

        def execute() -> Any:
            nonlocal executed
            executed = True
            return live()

        result = self._cassette.call(
            EventType.CUSTOM,
            f"{self._cassette_name}.{operation}",
            request,
            execute,
            metadata=_metadata(operation),
            serializer=_encode,
        )
        return _restore_call_result(self._cassette, result, executed=executed)

    def _record_start_error(
        self,
        name: str,
        request: dict[str, Any],
        operation: str,
        error: BaseException,
        started: float,
    ) -> None:
        self._cassette.add(
            EventType.ERROR,
            name,
            input=request,
            output=_stream_output([], error, phase="start"),
            metadata=_error_metadata(operation),
            duration_ms=(perf_counter() - started) * 1000,
        )


def wrap_langchain(
    runnable: Runnable[Any, Any] | None,
    cassette: Any,
    *,
    name: str = "langchain.runnable",
) -> Runnable[Any, Any]:
    """Wrap a LangChain Runnable for deterministic record and offline replay."""
    if runnable is not None and not isinstance(runnable, Runnable):
        raise TypeError("wrap_langchain requires a LangChain Runnable or None")
    if not isinstance(name, str) or not name:
        raise ValueError("LangChain cassette name must be a non-empty string")
    return LangChainRunnable(runnable, cassette, name)


__all__ = [
    "LangChainReplayIntrospectionError",
    "LangChainRunnable",
    "LangChainSerializationError",
    "wrap_langchain",
]
