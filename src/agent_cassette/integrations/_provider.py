"""Provider-agnostic record and replay machinery for SDK client proxies."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TypeVar, cast

from agent_cassette.events import EventType
from agent_cassette.replay import RateLimitError

Client = TypeVar("Client")


class ProviderStreamingUnsupportedError(NotImplementedError):
    """Raised when a provider stream does not expose the expected iterator API."""


class ProviderRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


@dataclass(frozen=True)
class ProviderSpec:
    """Describe how one provider SDK is intercepted."""

    provider: str
    operations: frozenset[str]
    prefixes: frozenset[str]
    raw_response_attrs: frozenset[str] = frozenset({"with_raw_response", "with_streaming_response"})
    unsupported_operations: dict[str, str] = field(default_factory=dict)
    stream_operations: frozenset[str] = frozenset()
    async_operations: frozenset[str] = frozenset()
    response_attributes: frozenset[str] = frozenset()
    derive_methods: frozenset[str] = frozenset({"with_options", "copy"})
    async_probe_path: tuple[str, ...] = ()
    streaming_error: type[Exception] = ProviderStreamingUnsupportedError
    raw_response_error: type[Exception] = ProviderRawResponseUnsupportedError

    def event_name(self, operation: str) -> str:
        return f"{self.provider}.{operation}"


class _ReplayObject(dict[str, Any]):
    """Dictionary fallback that also supports response-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _RecordingStream(Iterator[Any]):
    def __init__(
        self,
        stream: Any,
        cassette: Any,
        spec: ProviderSpec,
        operation: str,
        request: Any,
        started: float,
    ) -> None:
        self._stream = stream
        try:
            self._iterator: Any = iter(stream)
        except TypeError:
            if not hasattr(stream, "__enter__"):
                raise  # genuinely not a stream; fail fast like before
            self._iterator = None  # entered lazily via __enter__
        self._cassette = cassette
        self._spec = spec
        self._operation = operation
        self._request = request
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __iter__(self) -> _RecordingStream:
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            raise TypeError("stream must be entered as a context manager before iteration")
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
            self._spec.event_name(self._operation),
            input=self._request,
            output=_serialize_stream(self._chunks, self._spec.response_attributes),
            metadata=_metadata(self._spec, self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )

    def _fail(self, error: Exception) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.ERROR,
            self._spec.event_name(self._operation),
            input=self._request,
            output=_serialize_stream_error(self._chunks, error, self._spec.response_attributes),
            metadata=_stream_error_metadata(self._spec, self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )


class _AsyncRecordingStream(AsyncIterator[Any]):
    def __init__(
        self,
        stream: Any,
        cassette: Any,
        spec: ProviderSpec,
        operation: str,
        request: Any,
        started: float,
    ) -> None:
        # ``stream`` is normally the concrete async iterable/context-manager
        # already, because the async wrapper awaits the underlying coroutine
        # (e.g. mistralai's ``chat.stream_async`` or an ``AsyncOpenAI``-style
        # ``create(stream=True)``) before constructing this object. The
        # ``inspect.isawaitable`` branch in ``_resolve`` remains load-bearing
        # for any stream that is itself still an awaitable when handed over.
        self._stream = stream
        self._resolved = False
        self._iterator: Any = None
        self._cassette = cassette
        self._spec = spec
        self._operation = operation
        self._request = request
        self._started = started
        self._chunks: list[Any] = []
        self._finished = False

    def __aiter__(self) -> _AsyncRecordingStream:
        return self

    async def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        stream = self._stream
        if inspect.isawaitable(stream):
            stream = await stream
            self._stream = stream
        try:
            self._iterator = stream.__aiter__()
        except (TypeError, AttributeError) as error:
            if not hasattr(stream, "__aenter__"):
                raise self._spec.streaming_error(
                    f"{self._spec.provider} streaming capture requires an async iterable "
                    "stream; call with stream=False"
                ) from error
            self._iterator = None  # entered lazily via __aenter__

    async def __anext__(self) -> Any:
        await self._resolve()
        if self._iterator is None:
            raise TypeError("stream must be entered as a context manager before iteration")
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
        await self._resolve()
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
            self._spec.event_name(self._operation),
            input=self._request,
            output=_serialize_stream(self._chunks, self._spec.response_attributes),
            metadata=_metadata(self._spec, self._operation, streaming=True),
            duration_ms=(perf_counter() - self._started) * 1000,
        )

    def _fail(self, error: Exception) -> None:
        if self._finished:
            return
        self._finished = True
        self._cassette.add(
            EventType.ERROR,
            self._spec.event_name(self._operation),
            input=self._request,
            output=_serialize_stream_error(self._chunks, error, self._spec.response_attributes),
            metadata=_stream_error_metadata(self._spec, self._operation),
            duration_ms=(perf_counter() - self._started) * 1000,
        )


class _ReplayStream(Iterator[Any]):
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __iter__(self) -> _ReplayStream:
        return self

    def __enter__(self) -> _ReplayStream:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        return None

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

    async def __aenter__(self) -> _AsyncReplayStream:
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        return None

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as stop:
            if self._error is not None:
                error, self._error = self._error, None
                raise error from None
            raise StopAsyncIteration from stop


class _ResourceProxy:
    def __init__(
        self, target: Any, path: str, cassette: Any, spec: ProviderSpec, *, asynchronous: bool
    ) -> None:
        self._target = target
        self._path = path
        self._cassette = cassette
        self._spec = spec
        self._asynchronous = asynchronous

    def __getattr__(self, name: str) -> Any:
        operation = f"{self._path}.{name}"
        spec = self._spec
        if name in spec.raw_response_attrs:
            raise spec.raw_response_error(
                f"{spec.provider} {name} transport helpers are not replay-safe; "
                "use standard create calls"
            )
        if operation in spec.unsupported_operations:
            raise spec.streaming_error(spec.unsupported_operations[operation])
        target_attribute = getattr(self._target, name) if self._target is not None else None
        if operation in spec.prefixes:
            return _ResourceProxy(
                target_attribute,
                operation,
                self._cassette,
                spec,
                asynchronous=self._asynchronous,
            )
        if operation in spec.operations:
            return self._wrap_create(target_attribute, operation)
        if target_attribute is None:
            raise AttributeError(
                f"{spec.provider} operation {operation!r} is unavailable without a live client"
            )
        return target_attribute

    def _wrap_create(self, create: Callable[..., Any] | None, operation: str) -> Callable[..., Any]:
        spec = self._spec
        event_name = spec.event_name(operation)
        is_async = self._asynchronous or operation in spec.async_operations
        stream_operation = operation in spec.stream_operations

        if is_async:

            async def async_create(*args: Any, **kwargs: Any) -> Any:
                request = _serialize_request(args, kwargs)
                if stream_operation or kwargs.get("stream"):
                    started = perf_counter()
                    replayed, recorded = _prepare_stream(
                        self._cassette,
                        EventType.MODEL_CALL,
                        event_name,
                        request,
                        metadata=_metadata(spec, operation, streaming=True),
                    )
                    if replayed:
                        chunks, error = _restore_stream(recorded, spec)
                        return _AsyncReplayStream(chunks, error)
                    if create is None:
                        raise RuntimeError(
                            f"Replay unexpectedly attempted a live {spec.provider} request"
                        )
                    try:
                        stream = await create(*args, **kwargs)
                    except Exception as error:
                        _record_stream_start_error(
                            self._cassette, spec, operation, request, error, started
                        )
                        raise
                    return _AsyncRecordingStream(
                        stream, self._cassette, spec, operation, request, started
                    )

                async def live_call() -> Any:
                    if create is None:
                        raise RuntimeError(
                            f"Replay unexpectedly attempted a live {spec.provider} request"
                        )
                    return await create(*args, **kwargs)

                recorded = await self._cassette.acall(
                    EventType.MODEL_CALL,
                    event_name,
                    request,
                    live_call,
                    metadata=_metadata(spec, operation),
                    serializer=lambda response: _serialize_response(
                        response, spec.response_attributes
                    ),
                )
                return _restore_response(recorded)

            return async_create

        def sync_create(*args: Any, **kwargs: Any) -> Any:
            request = _serialize_request(args, kwargs)
            if stream_operation or kwargs.get("stream"):
                started = perf_counter()
                replayed, recorded = _prepare_stream(
                    self._cassette,
                    EventType.MODEL_CALL,
                    event_name,
                    request,
                    metadata=_metadata(spec, operation, streaming=True),
                )
                if replayed:
                    chunks, error = _restore_stream(recorded, spec)
                    return _ReplayStream(chunks, error)
                if create is None:
                    raise RuntimeError(
                        f"Replay unexpectedly attempted a live {spec.provider} request"
                    )
                try:
                    stream = create(*args, **kwargs)
                except Exception as error:
                    _record_stream_start_error(
                        self._cassette, spec, operation, request, error, started
                    )
                    raise
                try:
                    return _RecordingStream(
                        stream, self._cassette, spec, operation, request, started
                    )
                except TypeError as error:
                    raise spec.streaming_error(
                        f"{spec.provider} streaming capture requires an iterable stream; "
                        "call with stream=False"
                    ) from error

            def live_call() -> Any:
                if create is None:
                    raise RuntimeError(
                        f"Replay unexpectedly attempted a live {spec.provider} request"
                    )
                return create(*args, **kwargs)

            recorded = self._cassette.call(
                EventType.MODEL_CALL,
                event_name,
                request,
                live_call,
                metadata=_metadata(spec, operation),
                serializer=lambda response: _serialize_response(response, spec.response_attributes),
            )
            return _restore_response(recorded)

        return sync_create


class ProviderClientProxy:
    """Proxy that intercepts supported model-creation operations."""

    def __init__(
        self, client: Any, cassette: Any, spec: ProviderSpec, *, asynchronous: bool = False
    ) -> None:
        self._client = client
        self._cassette = cassette
        self._spec = spec
        self._asynchronous = asynchronous

    def __getattr__(self, name: str) -> Any:
        spec = self._spec
        target_attribute = getattr(self._client, name) if self._client is not None else None
        if name in spec.prefixes:
            return _ResourceProxy(
                target_attribute,
                name,
                self._cassette,
                spec,
                asynchronous=self._asynchronous,
            )
        if name in spec.derive_methods:

            def derive(*args: Any, **kwargs: Any) -> ProviderClientProxy:
                derived = target_attribute(*args, **kwargs) if callable(target_attribute) else None
                return ProviderClientProxy(
                    derived,
                    self._cassette,
                    spec,
                    asynchronous=self._asynchronous,
                )

            return derive

        if target_attribute is None:
            raise AttributeError(
                f"{spec.provider} client attribute {name!r} requires a live client"
            )
        return target_attribute


def wrap_provider(
    client: Client | None,
    cassette: Any,
    spec: ProviderSpec,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a provider client so supported calls record or replay automatically."""
    if asynchronous is None:
        asynchronous = _is_async_client(client, spec)
    return ProviderClientProxy(client, cassette, spec, asynchronous=asynchronous)  # type: ignore[return-value]


def _is_async_client(client: Any, spec: ProviderSpec) -> bool:
    if client is None:
        return False
    client_name = type(client).__name__.lower()
    if "async" in client_name:
        return True
    probe: Any = client
    for part in spec.async_probe_path:
        probe = getattr(probe, part, None)
    return inspect.iscoroutinefunction(probe)


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


def _metadata(spec: ProviderSpec, operation: str, *, streaming: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": spec.provider, "operation": operation}
    if streaming:
        metadata["streaming"] = True
    return metadata


def _stream_error_metadata(spec: ProviderSpec, operation: str) -> dict[str, Any]:
    metadata = _metadata(spec, operation, streaming=True)
    metadata["_agent_cassette"] = {"call_type": EventType.MODEL_CALL.value}
    return metadata


def _record_stream_start_error(
    cassette: Any,
    spec: ProviderSpec,
    operation: str,
    request: Any,
    error: Exception,
    started: float,
) -> None:
    cassette.add(
        EventType.ERROR,
        spec.event_name(operation),
        input=request,
        output=_serialize_stream_error([], error, spec.response_attributes),
        metadata=_stream_error_metadata(spec, operation),
        duration_ms=(perf_counter() - started) * 1000,
    )


def _serialize_stream(
    chunks: list[Any], response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk, response_attributes) for chunk in chunks],
    }


def _serialize_stream_error(
    chunks: list[Any], error: Exception, response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk, response_attributes) for chunk in chunks],
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _restore_stream(recorded: Any, spec: ProviderSpec) -> tuple[list[Any], Exception | None]:
    if not isinstance(recorded, dict) or not recorded.get("__agent_cassette_stream__"):
        raise ValueError(f"Recorded {spec.provider} stream has an invalid payload")
    chunks = recorded.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"Recorded {spec.provider} stream chunks must be a list")
    error_payload = recorded.get("error")
    error = _restore_stream_error(error_payload) if isinstance(error_payload, dict) else None
    return [_restore_response(chunk) for chunk in chunks], error


def _restore_stream_error(payload: dict[str, Any]) -> Exception:
    error_type = str(payload.get("type", "RuntimeError"))
    message = str(payload.get("message", "recorded stream failed"))
    allowed = {
        "ConnectionError": ConnectionError,
        "RateLimitError": RateLimitError,
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


def _serialize_response(
    response: Any, response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    response_type = type(response)
    data = _to_data(response)
    if response_attributes and isinstance(data, dict):
        for attribute in response_attributes:
            if attribute not in data:
                data[attribute] = _to_data(getattr(response, attribute, None))
    return {
        "__agent_cassette_response__": True,
        "module": response_type.__module__,
        "class": response_type.__qualname__,
        "data": data,
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
