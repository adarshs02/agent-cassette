"""Transparent record and replay wrapper for the OpenAI Python client."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from time import perf_counter
from typing import Any, TypeVar, cast

from agent_cassette.events import EventType

Client = TypeVar("Client")
_SUPPORTED_OPERATIONS = {"responses.create", "chat.completions.create"}
_RESOURCE_PREFIXES = {"responses", "chat", "chat.completions"}


class OpenAIStreamingUnsupportedError(NotImplementedError):
    """Raised when a provider stream does not expose the expected iterator API."""


class OpenAIRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


class _ReplayObject(dict[str, Any]):
    """Dictionary fallback that also supports response-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _RecordingStream(Iterator[Any]):
    def __init__(
        self, stream: Any, cassette: Any, operation: str, request: Any, started: float
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)
        self._cassette = cassette
        self._operation = operation
        self._request = request
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __iter__(self) -> _RecordingStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            self._fail(error)
            raise
        self._chunks.append(chunk)
        return chunk

    def __enter__(self) -> _RecordingStream:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            self._iterator = iter(cast(Any, enter()))
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        exit_stream = getattr(self._stream, "__exit__", None)
        try:
            if callable(exit_stream):
                exit_stream(exc_type, exc, traceback)
            else:
                close = getattr(self._stream, "close", None)
                if callable(close):
                    close()
        finally:
            self._finish()

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.MODEL_CALL,
            f"openai.{self._operation}",
            input=self._request,
            output=_serialize_stream(self._chunks),
            metadata=_metadata(self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )

    def _fail(self, error: Exception) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.ERROR,
            f"openai.{self._operation}",
            input=self._request,
            output=_serialize_stream_error(self._chunks, error),
            metadata=_stream_error_metadata(self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )


class _AsyncRecordingStream(AsyncIterator[Any]):
    def __init__(
        self, stream: Any, cassette: Any, operation: str, request: Any, started: float
    ) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._cassette = cassette
        self._operation = operation
        self._request = request
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __aiter__(self) -> _AsyncRecordingStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as error:
            self._fail(error)
            raise
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self) -> _AsyncRecordingStream:
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            entered = await cast(Any, enter())
            self._iterator = entered.__aiter__()
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        exit_stream = getattr(self._stream, "__aexit__", None)
        try:
            if callable(exit_stream):
                await cast(Any, exit_stream(exc_type, exc, traceback))
            else:
                await self._close_stream()
        finally:
            self._finish()

    async def aclose(self) -> None:
        try:
            await self._close_stream()
        finally:
            self._finish()

    async def _close_stream(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if not callable(close):
            close = getattr(self._stream, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.MODEL_CALL,
            f"openai.{self._operation}",
            input=self._request,
            output=_serialize_stream(self._chunks),
            metadata=_metadata(self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )

    def _fail(self, error: Exception) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.ERROR,
            f"openai.{self._operation}",
            input=self._request,
            output=_serialize_stream_error(self._chunks, error),
            metadata=_stream_error_metadata(self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )


class _ReplayStream(Iterator[Any]):
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
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


class _AsyncReplayStream(AsyncIterator[Any]):
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
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


class _ResourceProxy:
    def __init__(self, target: Any, path: str, cassette: Any, *, asynchronous: bool) -> None:
        self._target = target
        self._path = path
        self._cassette = cassette
        self._asynchronous = asynchronous

    def __getattr__(self, name: str) -> Any:
        operation = f"{self._path}.{name}"
        if name in {"with_raw_response", "with_streaming_response"}:
            raise OpenAIRawResponseUnsupportedError(
                f"OpenAI {name} transport helpers are not replay-safe; use standard create calls"
            )
        target_attribute = getattr(self._target, name) if self._target is not None else None
        if operation in _RESOURCE_PREFIXES:
            return _ResourceProxy(
                target_attribute,
                operation,
                self._cassette,
                asynchronous=self._asynchronous,
            )
        if operation in _SUPPORTED_OPERATIONS:
            return self._wrap_create(target_attribute, operation)
        if target_attribute is None:
            raise AttributeError(
                f"OpenAI operation {operation!r} is unavailable without a live client"
            )
        return target_attribute

    def _wrap_create(self, create: Callable[..., Any] | None, operation: str) -> Callable[..., Any]:
        if self._asynchronous:

            async def async_create(*args: Any, **kwargs: Any) -> Any:
                request = _serialize_request(args, kwargs)
                if kwargs.get("stream"):
                    started = perf_counter()
                    replayed, recorded = _prepare_stream(
                        self._cassette,
                        EventType.MODEL_CALL,
                        f"openai.{operation}",
                        request,
                        metadata=_metadata(operation, streaming=True),
                    )
                    if replayed:
                        chunks, error = _restore_stream(recorded)
                        return _AsyncReplayStream(chunks, error)
                    if create is None:
                        raise RuntimeError("Replay unexpectedly attempted a live OpenAI request")
                    try:
                        stream = await create(*args, **kwargs)
                    except Exception as error:
                        _record_stream_start_error(
                            self._cassette, operation, request, error, started
                        )
                        raise
                    try:
                        return _AsyncRecordingStream(
                            stream, self._cassette, operation, request, started
                        )
                    except (AttributeError, TypeError) as error:
                        raise OpenAIStreamingUnsupportedError(
                            "OpenAI streaming capture requires an async iterable stream; "
                            "call with stream=False"
                        ) from error

                async def live_call() -> Any:
                    if create is None:
                        raise RuntimeError("Replay unexpectedly attempted a live OpenAI request")
                    return await create(*args, **kwargs)

                recorded = await self._cassette.acall(
                    EventType.MODEL_CALL,
                    f"openai.{operation}",
                    request,
                    live_call,
                    metadata=_metadata(operation),
                    serializer=_serialize_response,
                )
                return _restore_response(recorded)

            return async_create

        def sync_create(*args: Any, **kwargs: Any) -> Any:
            request = _serialize_request(args, kwargs)
            if kwargs.get("stream"):
                started = perf_counter()
                replayed, recorded = _prepare_stream(
                    self._cassette,
                    EventType.MODEL_CALL,
                    f"openai.{operation}",
                    request,
                    metadata=_metadata(operation, streaming=True),
                )
                if replayed:
                    chunks, error = _restore_stream(recorded)
                    return _ReplayStream(chunks, error)
                if create is None:
                    raise RuntimeError("Replay unexpectedly attempted a live OpenAI request")
                try:
                    stream = create(*args, **kwargs)
                except Exception as error:
                    _record_stream_start_error(self._cassette, operation, request, error, started)
                    raise
                try:
                    return _RecordingStream(stream, self._cassette, operation, request, started)
                except TypeError as error:
                    raise OpenAIStreamingUnsupportedError(
                        "OpenAI streaming capture requires an iterable stream; "
                        "call with stream=False"
                    ) from error

            def live_call() -> Any:
                if create is None:
                    raise RuntimeError("Replay unexpectedly attempted a live OpenAI request")
                return create(*args, **kwargs)

            recorded = self._cassette.call(
                EventType.MODEL_CALL,
                f"openai.{operation}",
                request,
                live_call,
                metadata=_metadata(operation),
                serializer=_serialize_response,
            )
            return _restore_response(recorded)

        return sync_create


class OpenAIClientProxy:
    """Proxy that intercepts supported model-creation operations."""

    def __init__(self, client: Any, cassette: Any, *, asynchronous: bool = False) -> None:
        self._client = client
        self._cassette = cassette
        self._asynchronous = asynchronous

    def __getattr__(self, name: str) -> Any:
        target_attribute = getattr(self._client, name) if self._client is not None else None
        if name in _RESOURCE_PREFIXES:
            return _ResourceProxy(
                target_attribute,
                name,
                self._cassette,
                asynchronous=self._asynchronous,
            )
        if name in {"with_options", "copy"}:

            def derive(*args: Any, **kwargs: Any) -> OpenAIClientProxy:
                derived = target_attribute(*args, **kwargs) if callable(target_attribute) else None
                return OpenAIClientProxy(
                    derived,
                    self._cassette,
                    asynchronous=self._asynchronous,
                )

            return derive

        if target_attribute is None:
            raise AttributeError(f"OpenAI client attribute {name!r} requires a live client")
        return target_attribute


def wrap_openai(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap an OpenAI client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. In that case, set ``asynchronous``
    explicitly when replaying calls made by ``AsyncOpenAI``.
    """
    if asynchronous is None:
        asynchronous = _is_async_client(client)
    return OpenAIClientProxy(client, cassette, asynchronous=asynchronous)  # type: ignore[return-value]


def _is_async_client(client: Any) -> bool:
    if client is None:
        return False
    client_name = type(client).__name__.lower()
    if "async" in client_name:
        return True
    responses = getattr(client, "responses", None)
    create = getattr(responses, "create", None)
    return inspect.iscoroutinefunction(create)


def _prepare_stream(
    cassette: Any,
    event_type: EventType,
    name: str,
    input: Any,
    *,
    metadata: dict[str, Any],
) -> tuple[bool, Any]:
    prepare = getattr(cassette, "prepare_stream", None)
    if callable(prepare):
        return cast(tuple[bool, Any], prepare(event_type, name, input, metadata=metadata))
    if hasattr(cassette, "position"):
        consume = getattr(cassette, "consume", None)
        if callable(consume):
            event: Any = consume(event_type, name, input)
            return True, event.output
        return True, cassette.call(event_type, name, input, metadata=metadata)
    return False, None


def _metadata(operation: str, *, streaming: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": "openai", "operation": operation}
    if streaming:
        metadata["streaming"] = True
    return metadata


def _stream_error_metadata(operation: str) -> dict[str, Any]:
    metadata = _metadata(operation, streaming=True)
    metadata["_agent_cassette"] = {"call_type": EventType.MODEL_CALL.value}
    return metadata


def _record_stream_start_error(
    cassette: Any, operation: str, request: Any, error: Exception, started: float
) -> None:
    cassette.add(
        EventType.ERROR,
        f"openai.{operation}",
        input=request,
        output=_serialize_stream_error([], error),
        metadata=_stream_error_metadata(operation),
        duration_ms=(perf_counter() - started) * 1000,
    )


def _serialize_stream(chunks: list[Any]) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk) for chunk in chunks],
    }


def _serialize_stream_error(chunks: list[Any], error: Exception) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk) for chunk in chunks],
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _restore_stream(recorded: Any) -> tuple[list[Any], Exception | None]:
    if not isinstance(recorded, dict) or not recorded.get("__agent_cassette_stream__"):
        raise ValueError("Recorded OpenAI stream has an invalid payload")
    chunks = recorded.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Recorded OpenAI stream chunks must be a list")
    error_payload = recorded.get("error")
    error = _restore_stream_error(error_payload) if isinstance(error_payload, dict) else None
    return [_restore_response(chunk) for chunk in chunks], error


def _restore_stream_error(payload: dict[str, Any]) -> Exception:
    error_type = str(payload.get("type", "RuntimeError"))
    message = str(payload.get("message", "recorded stream failed"))
    allowed = {
        "ConnectionError": ConnectionError,
        "RuntimeError": RuntimeError,
        "TimeoutError": TimeoutError,
        "ValueError": ValueError,
    }
    return allowed.get(error_type, RuntimeError)(message)


def _serialize_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [_to_data(argument) for argument in args],
        "kwargs": {key: _to_data(value) for key, value in kwargs.items()},
    }


def _serialize_response(response: Any) -> dict[str, Any]:
    response_type = type(response)
    return {
        "__agent_cassette_response__": True,
        "module": response_type.__module__,
        "class": response_type.__qualname__,
        "data": _to_data(response),
    }


def _restore_response(recorded: Any) -> Any:
    if not isinstance(recorded, dict) or not recorded.get("__agent_cassette_response__"):
        return recorded
    return _to_replay_object(recorded.get("data"))


def _to_data(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
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
