"""Explicit wrappers for deterministic Python-tool record and replay.

``wrap_tool`` supports four callable shapes. Plain sync and async functions
record and replay a single ``TOOL_CALL`` value (Phase A). Sync generator and
async-generator functions record and replay a *stream*: their yielded items,
terminal return value, terminal error, or a deliberate early close, all through
one versioned tool-stream envelope (Phase B). Replay never constructs or runs
the wrapped generator; the session type (``Recorder``/``Replayer``/``Hybrid``)
alone decides record, replay, or inject. Every input, yielded item, and return
value is validated and detached by the exact-type :func:`_copy_tool_json` codec
before it is stored or returned -- no ``model_dump``/``str``/``repr`` or other
duck-typed conversion is ever called, and subclasses are rejected so a recorded
value and its replay have identical Python types.
"""

from __future__ import annotations

import asyncio
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
            "callable objects and builtin callables are not supported for tool replay; "
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


# --------------------------------------------------------------------------- #
# Phase B -- generator / async-generator tool streaming
# --------------------------------------------------------------------------- #

_TOOL_STREAM_MARKER = "__agent_cassette_tool_stream__"
_TOOL_STREAM_VERSION = 1
_STREAM_COMPLETIONS = frozenset({"exhausted", "closed", "error"})
_STREAM_ERROR_PHASES = frozenset({"start", "iteration", "close", "cancellation"})


class _RestoredToolStream:
    """A validated, replay-ready tool stream restored from a cassette envelope."""

    __slots__ = ("items", "completion", "return_value", "terminal_error")

    def __init__(
        self,
        items: list[Any],
        completion: str,
        return_value: Any,
        terminal_error: BaseException | None,
    ) -> None:
        self.items = items
        self.completion = completion
        self.return_value = return_value
        self.terminal_error = terminal_error


def _stream_success_metadata() -> dict[str, Any]:
    return {"streaming": True}


def _stream_error_metadata() -> dict[str, Any]:
    # ``call_type`` keeps the ERROR event replayable (see replay._is_replayable);
    # ``Hybrid.add``/``Recorder.add`` preserve this internal key.
    return {"streaming": True, "_agent_cassette": {"call_type": EventType.TOOL_CALL.value}}


def _stream_success_envelope(
    items: list[Any], completion: str, return_value: Any
) -> dict[str, Any]:
    return {
        _TOOL_STREAM_MARKER: True,
        "version": _TOOL_STREAM_VERSION,
        "items": items,
        "completion": completion,
        "return": return_value,
    }


def _stream_error_envelope(items: list[Any], phase: str, error: BaseException) -> dict[str, Any]:
    return {
        _TOOL_STREAM_MARKER: True,
        "version": _TOOL_STREAM_VERSION,
        "items": items,
        "completion": "error",
        "phase": phase,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _encode_return_stream(value: Any) -> dict[str, Any]:
    """Encode a Hybrid ``Return`` injection as a natural-exhaustion tool stream.

    The injected value must be an exact plain ``list`` of individually valid
    yielded items; a tuple, subclass, generator, or a caller-supplied internal
    envelope is rejected rather than silently accepted.
    """
    if type(value) is not list:
        raise StrictJSONError("tool stream Return injection requires a plain list of yielded items")
    items = [_copy_tool_json(item) for item in value]
    return _stream_success_envelope(items, "exhausted", None)


def _encode_start_error_stream(error: BaseException) -> dict[str, Any]:
    """Encode a Hybrid ``Raise`` injection as a deferred zero-item start error."""
    return _stream_error_envelope([], "start", error)


def _restore_tool_error(phase: str, type_name: str, message: str) -> BaseException:
    if phase == "cancellation":
        return asyncio.CancelledError(message)
    # Local import: replay imports tools for _ToolSessionMixin, so a module-level
    # import would be circular. The allowlist mirrors Phase A / the provider path.
    from agent_cassette.replay import RateLimitError, RecordedCallError

    allowed: dict[str, type[Exception]] = {
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "TimeoutError": TimeoutError,
        "ConnectionError": ConnectionError,
        "RateLimitError": RateLimitError,
    }
    exception_type = allowed.get(type_name)
    if exception_type is not None:
        return exception_type(message)
    return RecordedCallError(type_name, message)


def _restore_tool_stream(recorded: Any, *, asynchronous: bool) -> _RestoredToolStream:
    """Strictly validate a recorded tool-stream envelope; fail closed on any deviation.

    The complete envelope is first driven through the exact-type ``_copy_tool_json``
    codec, so nested tuples, subclasses, hostile containers, cycles, over-depth values,
    non-finite floats, and non-``str`` keys are rejected -- without invoking any user
    ``__str__``/``__repr__``/``model_dump`` -- and every returned item and return value
    is a detached copy that never aliases ``Replayer.events``. ``completion``/``phase``
    are checked to be exact ``str`` before set membership so a list/dict there fails
    closed with ``StrictJSONError`` rather than leaking an unhashable ``TypeError``.
    """
    detached = _copy_tool_json(recorded)
    if type(detached) is not dict:
        raise StrictJSONError("recorded tool stream must be a JSON object")
    if detached.get(_TOOL_STREAM_MARKER) is not True:
        raise StrictJSONError("recorded tool stream is missing its marker")
    version = detached.get("version")
    if type(version) is not int or version != _TOOL_STREAM_VERSION:
        raise StrictJSONError("recorded tool stream has an unsupported version")
    completion = detached.get("completion")
    if type(completion) is not str or completion not in _STREAM_COMPLETIONS:
        raise StrictJSONError("recorded tool stream completion is invalid")
    items = detached.get("items")
    if type(items) is not list:
        raise StrictJSONError("recorded tool stream items must be a list")
    keys = set(detached)
    if completion == "error":
        if keys != {_TOOL_STREAM_MARKER, "version", "items", "completion", "phase", "error"}:
            raise StrictJSONError("recorded tool stream error envelope has unexpected keys")
        phase = detached.get("phase")
        if type(phase) is not str or phase not in _STREAM_ERROR_PHASES:
            raise StrictJSONError("recorded tool stream error phase is invalid")
        error = detached.get("error")
        if type(error) is not dict or set(error) != {"type", "message"}:
            raise StrictJSONError("recorded tool stream error payload is invalid")
        error_type = error["type"]
        message = error["message"]
        if type(error_type) is not str or type(message) is not str:
            raise StrictJSONError("recorded tool stream error fields must be strings")
        if phase == "start" and items:
            raise StrictJSONError("a start-phase tool stream error must have no items")
        if phase == "cancellation" and error_type != "CancelledError":
            raise StrictJSONError("a cancellation tool stream error must be CancelledError")
        if phase != "cancellation" and error_type == "CancelledError":
            raise StrictJSONError("CancelledError is only valid for the cancellation phase")
        terminal = _restore_tool_error(phase, error_type, message)
        return _RestoredToolStream(items, completion, None, terminal)
    if keys != {_TOOL_STREAM_MARKER, "version", "items", "completion", "return"}:
        raise StrictJSONError("recorded tool stream envelope has unexpected keys")
    return_value = detached.get("return")
    if completion == "closed" and return_value is not None:
        raise StrictJSONError("a closed tool stream must have a null return value")
    if asynchronous and return_value is not None:
        raise StrictJSONError("an async tool stream must have a null return value")
    return _RestoredToolStream(items, completion, return_value, None)


def _dispatch_tool_stream(session: Any, name: str, input_value: Any) -> tuple[bool, Any]:
    """Replay/inject through the session, or signal a live stream is required.

    Mirrors the provider stream capability probe without importing the session
    types: ``Hybrid`` exposes ``prepare_stream`` (tool streams defer injected
    errors to first iteration); ``Replayer`` exposes ``consume`` + ``position``;
    a plain ``Recorder`` has neither and records live.
    """
    prepare = getattr(session, "prepare_stream", None)
    if callable(prepare):
        return cast(
            tuple[bool, Any],
            prepare(
                EventType.TOOL_CALL,
                name,
                input_value,
                metadata=_stream_success_metadata(),
                serializer=_encode_return_stream,
                error_serializer=_encode_start_error_stream,
                defer_errors=True,
            ),
        )
    if hasattr(session, "position"):
        consume = getattr(session, "consume", None)
        if callable(consume):
            event = consume(EventType.TOOL_CALL, name, input_value)
            return True, event.output
    return False, None


class _ReplayToolStream:
    """Synchronous replay of a recorded tool stream; never runs live code."""

    def __init__(self, restored: _RestoredToolStream) -> None:
        self._items = iter(restored.items)
        self._return = restored.return_value
        self._terminal = restored.terminal_error

    def __iter__(self) -> _ReplayToolStream:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            if self._terminal is not None:
                error, self._terminal = self._terminal, None
                raise error from None
            raise StopIteration(self._return) from None

    def send(self, value: Any) -> Any:
        if value is None:
            return self.__next__()
        raise NotImplementedError(
            "send() with a non-None value requires a bidirectional transcript (not in Phase B)"
        )

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("throw() requires a bidirectional transcript (not in Phase B)")

    def close(self) -> None:
        self._terminal = None
        self._items = iter(())


class _AsyncReplayToolStream:
    """Asynchronous replay of a recorded tool stream; never awaits live code."""

    def __init__(self, restored: _RestoredToolStream) -> None:
        self._items = iter(restored.items)
        self._terminal = restored.terminal_error

    def __aiter__(self) -> _AsyncReplayToolStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            if self._terminal is not None:
                error, self._terminal = self._terminal, None
                raise error from None
            raise StopAsyncIteration from None

    async def asend(self, value: Any) -> Any:
        if value is None:
            return await self.__anext__()
        raise NotImplementedError(
            "asend() with a non-None value requires a bidirectional transcript (not in Phase B)"
        )

    async def athrow(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("athrow() requires a bidirectional transcript (not in Phase B)")

    async def aclose(self) -> None:
        self._terminal = None
        self._items = iter(())


class _RecordingToolStream:
    """Record a live sync generator's items, terminal return, error, or close."""

    def __init__(self, generator: Any, session: Any, name: str, input_value: Any) -> None:
        self._generator = generator
        self._session = session
        self._name = name
        self._input = input_value
        self._items: list[Any] = []
        self._finished = False

    def __iter__(self) -> _RecordingToolStream:
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        try:
            item = next(self._generator)
        except StopIteration as stop:
            self._record_success("exhausted", stop.value)
            raise
        except Exception as error:
            self._record_error("iteration" if self._items else "start", error)
            raise
        self._items.append(self._detach(item))
        return _copy_tool_json(self._items[-1])

    def send(self, value: Any) -> Any:
        if value is None:
            return self.__next__()
        raise NotImplementedError(
            "send() with a non-None value requires a bidirectional transcript (not in Phase B)"
        )

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("throw() requires a bidirectional transcript (not in Phase B)")

    def close(self) -> None:
        if self._finished:
            return
        try:
            self._generator.close()
        except Exception as error:
            self._record_error("close", error)
            raise
        self._record_closed()

    def _detach(self, value: Any) -> Any:
        try:
            return _copy_tool_json(value)
        except StrictJSONError:
            # Invalid item/return: persist nothing and close the live generator.
            if not self._finished:
                self._finished = True
                _safe_close(self._generator)
            raise

    def _record_success(self, completion: str, return_value: Any) -> None:
        if self._finished:
            return
        encoded_return = self._detach(return_value) if completion == "exhausted" else None
        self._finished = True
        self._session.add(
            EventType.TOOL_CALL,
            self._name,
            input=self._input,
            output=_stream_success_envelope(self._items, completion, encoded_return),
            metadata=_stream_success_metadata(),
        )

    def _record_closed(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._session.add(
            EventType.TOOL_CALL,
            self._name,
            input=self._input,
            output=_stream_success_envelope(self._items, "closed", None),
            metadata=_stream_success_metadata(),
        )

    def _record_error(self, phase: str, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        self._session.add(
            EventType.ERROR,
            self._name,
            input=self._input,
            output=_stream_error_envelope(self._items, phase, error),
            metadata=_stream_error_metadata(),
        )


class _AsyncRecordingToolStream:
    """Record a live async generator's items, terminal error, cancellation, or close."""

    def __init__(self, generator: Any, session: Any, name: str, input_value: Any) -> None:
        self._generator = generator
        self._session = session
        self._name = name
        self._input = input_value
        self._items: list[Any] = []
        self._finished = False

    def __aiter__(self) -> _AsyncRecordingToolStream:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        try:
            item = await self._generator.__anext__()
        except StopAsyncIteration:
            self._record_success("exhausted")
            raise
        except asyncio.CancelledError as error:
            await self._record_cancellation(error)
            raise
        except Exception as error:
            self._record_error("iteration" if self._items else "start", error)
            raise
        try:
            stored = _copy_tool_json(item)
        except StrictJSONError:
            if not self._finished:
                self._finished = True
                await self._safe_aclose()
            raise
        self._items.append(stored)
        return _copy_tool_json(stored)

    async def asend(self, value: Any) -> Any:
        if value is None:
            return await self.__anext__()
        raise NotImplementedError(
            "asend() with a non-None value requires a bidirectional transcript (not in Phase B)"
        )

    async def athrow(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("athrow() requires a bidirectional transcript (not in Phase B)")

    async def aclose(self) -> None:
        if self._finished:
            return
        try:
            await self._generator.aclose()
        except asyncio.CancelledError as error:
            await self._record_cancellation(error)
            raise
        except Exception as error:
            self._record_error("close", error)
            raise
        self._record_closed()

    async def _safe_aclose(self) -> None:
        try:
            await self._generator.aclose()
        except (Exception, asyncio.CancelledError):
            pass

    def _record_success(self, completion: str) -> None:
        # Async generators cannot return a value, so the recorded return is None.
        if self._finished:
            return
        self._finished = True
        self._session.add(
            EventType.TOOL_CALL,
            self._name,
            input=self._input,
            output=_stream_success_envelope(self._items, completion, None),
            metadata=_stream_success_metadata(),
        )

    def _record_closed(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._session.add(
            EventType.TOOL_CALL,
            self._name,
            input=self._input,
            output=_stream_success_envelope(self._items, "closed", None),
            metadata=_stream_success_metadata(),
        )

    def _record_error(self, phase: str, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        self._session.add(
            EventType.ERROR,
            self._name,
            input=self._input,
            output=_stream_error_envelope(self._items, phase, error),
            metadata=_stream_error_metadata(),
        )

    async def _record_cancellation(self, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        await self._safe_aclose()
        self._session.add(
            EventType.ERROR,
            self._name,
            input=self._input,
            output=_stream_error_envelope(self._items, "cancellation", error),
            metadata=_stream_error_metadata(),
        )


def _safe_close(generator: Any) -> None:
    try:
        generator.close()
    except Exception:
        pass


def wrap_tool(
    function: Callable[P, R],
    cassette: Any,
    *,
    name: str | None = None,
) -> Callable[P, R]:
    """Wrap a sync/async function or (async) generator for cassette record and replay."""
    if not callable(function):
        raise TypeError("tool must be callable")

    root = _root_callable(function)
    tool_name = _resolve_name(function, name)

    if inspect.isasyncgenfunction(root):
        async_generator = cast(Callable[P, Any], function)

        @wraps(function)
        def async_stream_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            input_value = _serialize_input(args, kwargs)
            replayed, recorded = _dispatch_tool_stream(cassette, tool_name, input_value)
            if replayed:
                return _AsyncReplayToolStream(_restore_tool_stream(recorded, asynchronous=True))
            return _AsyncRecordingToolStream(
                async_generator(*args, **kwargs), cassette, tool_name, input_value
            )

        return cast(Callable[P, R], async_stream_wrapper)

    if inspect.isgeneratorfunction(root):
        generator = cast(Callable[P, Any], function)

        @wraps(function)
        def sync_stream_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            input_value = _serialize_input(args, kwargs)
            replayed, recorded = _dispatch_tool_stream(cassette, tool_name, input_value)
            if replayed:
                return _ReplayToolStream(_restore_tool_stream(recorded, asynchronous=False))
            return _RecordingToolStream(
                generator(*args, **kwargs), cassette, tool_name, input_value
            )

        return cast(Callable[P, R], sync_stream_wrapper)

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
