from __future__ import annotations

import asyncio
import builtins
import json
import math
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation, GenerationChunk, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda, chain

from agent_cassette import Cassette, EventType, InjectionRule, Raise, Return, wrap_langchain
from agent_cassette.integrations.langchain import (
    LangChainReplayIntrospectionError,
    LangChainSerializationError,
    _decode,
    _encode,
    _restore_stream,
    _stream_output,
    _stream_phase,
)
from agent_cassette.replay import RecordedCallError
from agent_cassette.storage import load_events


def test_invoke_records_lcel_and_replays_without_live_runnable(tmp_path: Path) -> None:
    path = tmp_path / "invoke.jsonl"
    calls = 0

    def first(value: int) -> int:
        nonlocal calls
        calls += 1
        return value + 1

    runnable = RunnableLambda(first) | RunnableLambda(lambda value: {"answer": value * 2})
    with Cassette.record(path) as recorder:
        wrapped = wrap_langchain(runnable, recorder, name="example")
        assert isinstance(wrapped, Runnable)
        assert wrapped.invoke(2) == {"answer": 6}

    with Cassette.replay(path) as replayer:
        assert wrap_langchain(None, replayer, name="example").invoke(2) == {"answer": 6}

    assert calls == 1
    event = load_events(path)[0]
    assert event.type == EventType.CUSTOM
    assert event.name == "example.invoke"
    assert event.metadata == {
        "integration": "langchain",
        "operation": "invoke",
        "streaming": False,
    }


def test_ainvoke_batch_and_abatch_replay(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "methods.jsonl"

        def double(value: int) -> int:
            return value * 2

        runnable: Runnable[int, int] = RunnableLambda(double)

        async with Cassette.record(path) as recorder:
            wrapped = wrap_langchain(runnable, recorder)
            assert await wrapped.ainvoke(2) == 4
            assert wrapped.batch([1, 2]) == [2, 4]
            assert await wrapped.abatch([3, 4]) == [6, 8]

        async with Cassette.replay(path) as replayer:
            wrapped = wrap_langchain(None, replayer)
            assert await wrapped.ainvoke(2) == 4
            assert wrapped.batch([1, 2]) == [2, 4]
            assert await wrapped.abatch([3, 4]) == [6, 8]

    asyncio.run(exercise())


def test_chain_decorator_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "chain.jsonl"

    @chain
    def decorated(value: str) -> AIMessage:
        return AIMessage(content=value.upper())

    with Cassette.record(path) as recorder:
        result = wrap_langchain(decorated, recorder).invoke("hello")
        assert isinstance(result, AIMessage)

    with Cassette.replay(path) as replayer:
        result = wrap_langchain(None, replayer).invoke("hello")
        assert isinstance(result, AIMessage)
        assert result.content == "HELLO"


def test_matching_keeps_config_and_kwargs_but_omits_callback_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.jsonl"
    config = {
        "callbacks": [],
        "run_id": uuid4(),
        "run_name": "first",
        "tags": ["stable"],
        "metadata": {"tenant": "one"},
        "configurable": {"temperature": 0},
        "recursion_limit": 10,
        "max_concurrency": 2,
        "future_key": "retained",
    }

    def add_suffix(value: str, suffix: str = "") -> str:
        return value + suffix

    runnable = RunnableLambda(add_suffix)
    with Cassette.record(path) as recorder:
        assert wrap_langchain(runnable, recorder).invoke("a", config, suffix="b") == "ab"

    event = load_events(path)[0]
    assert "callbacks" not in event.input["config"]
    assert "run_id" not in event.input["config"]
    assert "run_name" not in event.input["config"]
    assert event.input["config"]["future_key"] == "retained"

    replay_config = dict(config, callbacks=[object()], run_id=uuid4(), run_name="changed")
    with Cassette.replay(path) as replayer:
        assert wrap_langchain(None, replayer).invoke("a", replay_config, suffix="b") == "ab"


def test_batch_return_exceptions_is_atomic_and_typed(tmp_path: Path) -> None:
    path = tmp_path / "batch.jsonl"

    def maybe_fail(value: int) -> int:
        if value < 0:
            raise ValueError("negative")
        return value

    with Cassette.record(path) as recorder:
        results = wrap_langchain(RunnableLambda(maybe_fail), recorder).batch(
            [1, -1, 2], return_exceptions=True
        )
        assert isinstance(results[1], ValueError)
    assert len(load_events(path)) == 1

    with Cassette.replay(path) as replayer:
        results = wrap_langchain(None, replayer).batch([1, -1, 2], return_exceptions=True)
        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "negative"


def test_sync_and_async_failures_record_and_replay(tmp_path: Path) -> None:
    sync_path = tmp_path / "sync-error.jsonl"

    def fail_sync(_: str) -> str:
        raise ValueError("sync failed")

    with Cassette.record(sync_path) as recorder:
        with pytest.raises(ValueError, match="sync failed"):
            wrap_langchain(RunnableLambda(fail_sync), recorder).invoke("x")
    with Cassette.replay(sync_path) as replayer:
        with pytest.raises(ValueError, match="sync failed"):
            wrap_langchain(None, replayer).invoke("x")

    async def exercise_async() -> None:
        async_path = tmp_path / "async-error.jsonl"

        async def fail_async(_: str) -> str:
            raise RuntimeError("async failed")

        async with Cassette.record(async_path) as recorder:
            with pytest.raises(RuntimeError, match="async failed"):
                await wrap_langchain(RunnableLambda(fail_async), recorder).ainvoke("x")
        async with Cassette.replay(async_path) as replayer:
            with pytest.raises(RuntimeError, match="async failed"):
                await wrap_langchain(None, replayer).ainvoke("x")

    asyncio.run(exercise_async())


@pytest.mark.parametrize(
    "value",
    [
        HumanMessage(content="hello", name="user"),
        AIMessage(content="answer", tool_calls=[{"name": "f", "args": {}, "id": "1"}]),
        AIMessageChunk(content="part"),
        ToolMessage(content="done", tool_call_id="1"),
        Document(page_content="text", metadata={"source": "unit"}),
        Generation(text="plain"),
        GenerationChunk(text="chunk"),
        ChatGeneration(message=AIMessage(content="chat")),
        LLMResult(generations=[[Generation(text="result")]], llm_output={"tokens": 1}),
    ],
)
def test_safe_langchain_value_round_trip(value: object) -> None:
    restored = _decode(_encode(value))
    assert type(restored) is type(value)
    assert restored == value


def test_serializer_rejects_unsafe_values_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.jsonl"
    recursive: list[object] = []
    recursive.append(recursive)

    for value, message in [
        (recursive, "cyclic"),
        ({1: "bad"}, "must be a string"),
        (math.inf, "non-finite"),
        (object(), "unsupported value"),
    ]:
        with Cassette.record(path) as cassette:
            with pytest.raises(LangChainSerializationError, match=message):
                wrap_langchain(RunnableLambda(lambda item: item), cassette).invoke(value)
        assert load_events(path) == []


def test_serializer_depth_bound() -> None:
    value: object = "end"
    for _ in range(66):
        value = [value]
    with pytest.raises(LangChainSerializationError, match="maximum serialization depth"):
        _encode(value)


def test_reserved_marker_dictionary_round_trips_at_depth_boundary() -> None:
    value: object = {"__agent_cassette_langchain__": "user-value"}
    for _ in range(63):
        value = [value]
    assert _decode(_encode(value)) == value


def test_marker_dictionary_redaction_preserves_keys_in_raw_input_and_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "redacted.jsonl"
    value = {
        "__agent_cassette_langchain__": "user-value",
        "api_key": "sk-input-output-secret",
        "nested": {"password": "password-secret"},
    }
    with Cassette.record(path) as recorder:
        result = wrap_langchain(RunnableLambda(lambda item: item), recorder).invoke(value)
        assert result == value

    raw = path.read_text(encoding="utf-8")
    assert "sk-input-output-secret" not in raw
    assert "password-secret" not in raw
    assert "[REDACTED]" in raw
    event = load_events(path)[0]
    restored = _decode(event.output)
    assert restored == {
        "__agent_cassette_langchain__": "user-value",
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }
    assert event.input["inputs"]["type"] == "dict"
    assert "[REDACTED]" in json.dumps(event.input)


def test_marker_dictionary_respects_disabled_redaction(tmp_path: Path) -> None:
    path = tmp_path / "unredacted.jsonl"
    value = {"__agent_cassette_langchain__": "user", "api_key": "keep-me"}
    with Cassette.record(path, redact_secrets=False) as recorder:
        wrap_langchain(RunnableLambda(lambda item: item), recorder).invoke(value)
    assert "keep-me" in path.read_text(encoding="utf-8")
    assert _decode(load_events(path)[0].output) == value


def test_hostile_envelope_never_imports_cassette_named_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> Any:
        imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    payload = {
        "__agent_cassette_langchain__": True,
        "version": 1,
        "type": "os.system",
        "data": {"command": "never"},
    }
    with pytest.raises(LangChainSerializationError, match="unsupported LangChain envelope type"):
        _decode(payload)
    assert "os.system" not in imported


def test_recorded_call_stream_error_envelope_is_strict() -> None:
    payload = _stream_output([], LookupError("unknown"))
    chunks, error = _restore_stream(payload)
    assert chunks == []
    assert isinstance(error, RecordedCallError)

    extra_key = json.loads(json.dumps(payload))
    extra_key["data"]["error"]["extra"] = True
    with pytest.raises(LangChainSerializationError, match="envelope is invalid"):
        _restore_stream(extra_key)

    wrong_version = json.loads(json.dumps(payload))
    wrong_version["data"]["error"]["version"] = 2
    with pytest.raises(LangChainSerializationError, match="envelope is invalid"):
        _restore_stream(wrong_version)

    bad_data = json.loads(json.dumps(payload))
    bad_data["data"]["error"]["data"]["extra"] = "bad"
    with pytest.raises(LangChainSerializationError, match="data is invalid"):
        _restore_stream(bad_data)

    invalid_phase = _stream_output([], ValueError("bad phase"))
    invalid_phase["data"]["phase"] = "unknown"
    with pytest.raises(LangChainSerializationError, match="phase is invalid"):
        _restore_stream(invalid_phase)

    start_with_chunks = _stream_output(["impossible"], ValueError("bad start"))
    start_with_chunks["data"]["phase"] = "start"
    with pytest.raises(LangChainSerializationError, match="cannot contain"):
        _restore_stream(start_with_chunks)


def test_old_stream_error_payload_defaults_to_iteration_phase() -> None:
    payload = _stream_output(["partial"], ValueError("old iteration failure"))
    del payload["data"]["phase"]
    chunks, error = _restore_stream(payload)
    assert chunks == ["partial"]
    assert isinstance(error, ValueError)
    assert _stream_phase(payload) == "iteration"


class _StreamingRunnable(Runnable[str, str]):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.async_closed = False

    def invoke(self, input: str, config: object = None, **kwargs: object) -> str:
        return input

    def stream(self, input: str, config: object = None, **kwargs: object) -> Iterator[str]:
        try:
            yield input[0]
            if self.fail:
                raise ValueError("stream failed")
            yield input[1:]
        finally:
            self.closed = True

    async def astream(
        self, input: str, config: object = None, **kwargs: object
    ) -> AsyncIterator[str]:
        try:
            yield input[0]
            await asyncio.sleep(0)
            if self.fail:
                raise ValueError("async stream failed")
            yield input[1:]
        finally:
            self.async_closed = True


def test_stream_replay_failure_close_and_abandonment(tmp_path: Path) -> None:
    failed_path = tmp_path / "stream-failed.jsonl"
    live = _StreamingRunnable(fail=True)
    with Cassette.record(failed_path) as cassette:
        stream = wrap_langchain(live, cassette).stream("ab")
        assert next(stream) == "a"
        with pytest.raises(ValueError, match="stream failed"):
            next(stream)

    with Cassette.replay(failed_path) as cassette:
        stream = wrap_langchain(None, cassette).stream("ab")
        assert next(stream) == "a"
        with pytest.raises(ValueError, match="stream failed"):
            next(stream)

    close_path = tmp_path / "stream-close.jsonl"
    live = _StreamingRunnable()
    with Cassette.record(close_path) as cassette:
        stream = wrap_langchain(live, cassette).stream("ab")
        assert next(stream) == "a"
        stream.close()
    assert live.closed
    with Cassette.replay(close_path) as cassette:
        assert list(wrap_langchain(None, cassette).stream("ab")) == ["a"]

    abandoned_path = tmp_path / "abandoned.jsonl"
    with Cassette.record(abandoned_path) as cassette:
        wrap_langchain(_StreamingRunnable(), cassette).stream("ab")
    assert load_events(abandoned_path) == []


def test_sync_stream_start_failure_replays_before_iterator_return(tmp_path: Path) -> None:
    class StartFailingRunnable(Runnable[str, str]):
        def __init__(self) -> None:
            self.starts = 0

        def invoke(self, input: str, config: object = None, **kwargs: object) -> str:
            return input

        def stream(self, input: str, config: object = None, **kwargs: object) -> Iterator[str]:
            self.starts += 1
            raise ConnectionError("failed before iterator")

    path = tmp_path / "stream-start-error.jsonl"
    live = StartFailingRunnable()
    with Cassette.record(path) as recorder:
        with pytest.raises(ConnectionError, match="failed before iterator"):
            wrap_langchain(live, recorder).stream("request")
    assert live.starts == 1
    event = load_events(path)[0]
    assert event.output["data"]["phase"] == "start"
    assert event.output["data"]["chunks"] == []

    with Cassette.replay(path) as replayer:
        with pytest.raises(ConnectionError, match="failed before iterator"):
            # The error is raised by stream() itself; no iterator is consumed.
            wrap_langchain(None, replayer).stream("request")


def test_hybrid_typed_invoke_batch_and_stream_injections(tmp_path: Path) -> None:
    invoke_source = tmp_path / "invoke-source.jsonl"
    invoke_fork = tmp_path / "invoke-fork.jsonl"
    with Cassette.record(invoke_source) as recorder:
        wrap_langchain(RunnableLambda(lambda item: item), recorder).invoke("request")
    invoke_rule = InjectionRule(
        Return(AIMessage(content="injected")),
        type=EventType.CUSTOM,
        name="langchain.runnable.invoke",
    )
    with Cassette.fork(invoke_source, invoke_fork, injections=(invoke_rule,)) as hybrid:
        result = wrap_langchain(None, hybrid).invoke("request")
        assert isinstance(result, AIMessage)
        assert result.content == "injected"
    with Cassette.replay(invoke_fork) as replayer:
        replayed = wrap_langchain(None, replayer).invoke("request")
        assert isinstance(replayed, AIMessage)
        assert replayed.content == "injected"

    batch_source = tmp_path / "batch-source.jsonl"
    batch_fork = tmp_path / "batch-fork.jsonl"
    with Cassette.record(batch_source) as recorder:
        wrap_langchain(RunnableLambda(lambda item: item), recorder).batch(["request"])
    batch_rule = InjectionRule(
        Return([ValueError("injected batch error")]),
        type=EventType.CUSTOM,
        name="langchain.runnable.batch",
    )
    with Cassette.fork(batch_source, batch_fork, injections=(batch_rule,)) as hybrid:
        results = wrap_langchain(None, hybrid).batch(["request"])
        assert isinstance(results[0], ValueError)
    with Cassette.replay(batch_fork) as replayer:
        results = wrap_langchain(None, replayer).batch(["request"])
        assert isinstance(results[0], ValueError)
        assert str(results[0]) == "injected batch error"

    stream_source = tmp_path / "stream-source.jsonl"
    stream_fork = tmp_path / "stream-fork.jsonl"
    with Cassette.record(stream_source) as recorder:
        list(wrap_langchain(_StreamingRunnable(), recorder).stream("source"))
    stream_rule = InjectionRule(
        Return([AIMessageChunk(content="injected chunk")]),
        type=EventType.CUSTOM,
        name="langchain.runnable.stream",
    )
    with Cassette.fork(stream_source, stream_fork, injections=(stream_rule,)) as hybrid:
        chunks = list(wrap_langchain(None, hybrid).stream("source"))
        assert isinstance(chunks[0], AIMessageChunk)
        assert chunks[0].content == "injected chunk"
    saved_output = load_events(stream_fork)[0].output
    assert saved_output["version"] == 1
    assert saved_output["type"] == "stream"
    with Cassette.replay(stream_fork) as replayer:
        chunks = list(wrap_langchain(None, replayer).stream("source"))
        assert isinstance(chunks[0], AIMessageChunk)
        assert chunks[0].content == "injected chunk"

    secret_fork = tmp_path / "stream-secret-fork.jsonl"
    secret_rule = InjectionRule(
        Return(
            [
                {
                    "__agent_cassette_langchain__": "user-value",
                    "api_key": "never-write-stream-secret",
                }
            ]
        ),
        type=EventType.CUSTOM,
        name="langchain.runnable.stream",
    )
    with Cassette.fork(stream_source, secret_fork, injections=(secret_rule,)) as hybrid:
        chunks = list(wrap_langchain(None, hybrid).stream("source"))
        assert chunks[0]["api_key"] == "never-write-stream-secret"
        raw_during_context = secret_fork.read_text(encoding="utf-8")
        assert "never-write-stream-secret" not in raw_during_context
        assert "[REDACTED]" in raw_during_context
    assert "never-write-stream-secret" not in secret_fork.read_text(encoding="utf-8")
    with Cassette.replay(secret_fork) as replayer:
        chunks = list(wrap_langchain(None, replayer).stream("source"))
        assert chunks == [
            {
                "__agent_cassette_langchain__": "user-value",
                "api_key": "[REDACTED]",
            }
        ]

    ordinary_fork = tmp_path / "stream-ordinary-dict-fork.jsonl"
    ordinary_chunk = {"type": "stream", "value": "ordinary user data"}
    ordinary_rule = InjectionRule(
        Return(ordinary_chunk),
        type=EventType.CUSTOM,
        name="langchain.runnable.stream",
    )
    with Cassette.fork(stream_source, ordinary_fork, injections=(ordinary_rule,)) as hybrid:
        assert list(wrap_langchain(None, hybrid).stream("source")) == [ordinary_chunk]
    with Cassette.replay(ordinary_fork) as replayer:
        assert list(wrap_langchain(None, replayer).stream("source")) == [ordinary_chunk]


@pytest.mark.parametrize(
    ("error", "replayed_type"),
    [
        (ValueError("safe injected stream error"), ValueError),
        (LookupError("unknown injected stream error"), RecordedCallError),
    ],
)
def test_hybrid_stream_raise_is_safe_and_replayable(
    tmp_path: Path, error: Exception, replayed_type: type[Exception]
) -> None:
    source = tmp_path / f"raise-{type(error).__name__}-source.jsonl"
    fork = tmp_path / f"raise-{type(error).__name__}-fork.jsonl"
    with Cassette.record(source) as recorder:
        list(wrap_langchain(_StreamingRunnable(), recorder).stream("source"))
    rule = InjectionRule(
        Raise(error),
        type=EventType.CUSTOM,
        name="langchain.runnable.stream",
    )
    with Cassette.fork(source, fork, injections=(rule,)) as hybrid:
        with pytest.raises(type(error), match=str(error)):
            list(wrap_langchain(None, hybrid).stream("source"))

    events = load_events(fork)
    assert len(events) == 1
    assert events[0].type == EventType.ERROR
    assert events[0].metadata["_agent_cassette"]["mode"] == "injected"
    assert events[0].metadata["_agent_cassette"]["call_type"] == "custom"
    assert events[0].output["version"] == 1
    assert events[0].output["type"] == "stream"
    assert events[0].output["data"]["phase"] == "start"
    _restore_stream(events[0].output)

    with Cassette.replay(fork) as replayer:
        with pytest.raises(replayed_type, match=str(error)):
            list(wrap_langchain(None, replayer).stream("source"))


def test_constrained_live_attribute_delegation(tmp_path: Path) -> None:
    sequence = RunnableLambda(lambda item: item) | RunnableLambda(lambda item: item)
    sequence_attributes: Any = sequence
    path = tmp_path / "attributes.jsonl"
    with Cassette.record(path) as recorder:
        wrapped = wrap_langchain(sequence, recorder)
        assert wrapped.steps == sequence_attributes.steps
        sequence_attributes._private_test_value = "secret"
        with pytest.raises(AttributeError):
            _ = wrapped._private_test_value

    with Cassette.replay(path) as replayer:
        wrapped = wrap_langchain(None, replayer)
        with pytest.raises(AttributeError, match="offline replay"):
            _ = wrapped.steps

    class ExtendedRunnable(Runnable[str, str]):
        label = "visible"

        def invoke(self, input: str, config: object = None, **kwargs: object) -> str:
            return input

        def custom_execute(self, value: str) -> str:
            return f"unrecorded:{value}"

    callable_path = tmp_path / "callable-attributes.jsonl"
    with Cassette.record(callable_path) as recorder:
        wrapped = wrap_langchain(ExtendedRunnable(), recorder)
        assert wrapped.label == "visible"
        with pytest.raises(AttributeError, match="cannot bypass cassette capture"):
            _ = wrapped.custom_execute


def test_astream_replay_failure_close_and_abandonment(tmp_path: Path) -> None:
    async def exercise() -> None:
        failed_path = tmp_path / "astream-failed.jsonl"
        live = _StreamingRunnable(fail=True)
        async with Cassette.record(failed_path) as cassette:
            stream = wrap_langchain(live, cassette).astream("ab")
            assert await anext(stream) == "a"
            with pytest.raises(ValueError, match="async stream failed"):
                await anext(stream)

        async with Cassette.replay(failed_path) as cassette:
            stream = wrap_langchain(None, cassette).astream("ab")
            assert await anext(stream) == "a"
            with pytest.raises(ValueError, match="async stream failed"):
                await anext(stream)

        close_path = tmp_path / "astream-close.jsonl"
        live = _StreamingRunnable()
        async with Cassette.record(close_path) as cassette:
            stream = wrap_langchain(live, cassette).astream("ab")
            assert await anext(stream) == "a"
            await stream.aclose()
        assert live.async_closed
        async with Cassette.replay(close_path) as cassette:
            replayed = wrap_langchain(None, cassette).astream("ab")
            assert [item async for item in replayed] == ["a"]

        abandoned_path = tmp_path / "abandoned-async.jsonl"
        async with Cassette.record(abandoned_path) as cassette:
            wrap_langchain(_StreamingRunnable(), cassette).astream("ab")
        assert load_events(abandoned_path) == []

    asyncio.run(exercise())


def test_astream_task_cancellation_records_partial_chunks_and_closes(tmp_path: Path) -> None:
    class CancellableRunnable(Runnable[str, str]):
        def __init__(self) -> None:
            self.waiting = asyncio.Event()
            self.closed = False

        def invoke(self, input: str, config: object = None, **kwargs: object) -> str:
            return input

        async def astream(
            self, input: str, config: object = None, **kwargs: object
        ) -> AsyncIterator[str]:
            try:
                yield "partial"
                await self.waiting.wait()
                yield "unreachable"
            finally:
                self.closed = True

    async def exercise() -> None:
        path = tmp_path / "cancelled.jsonl"
        live = CancellableRunnable()
        async with Cassette.record(path) as recorder:
            stream = wrap_langchain(live, recorder).astream("request")
            assert await anext(stream) == "partial"
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert live.closed
        events = load_events(path)
        assert len(events) == 1
        assert events[0].type == EventType.ERROR

        async with Cassette.replay(path) as replayer:
            stream = wrap_langchain(None, replayer).astream("request")
            assert await anext(stream) == "partial"
            with pytest.raises(asyncio.CancelledError):
                await anext(stream)

    asyncio.run(exercise())


def test_replay_only_introspection_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with Cassette.replay(path) as cassette:
        wrapped = wrap_langchain(None, cassette)
        with pytest.raises(LangChainReplayIntrospectionError, match="introspection"):
            _ = wrapped.InputType


def test_schema_v1_event_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "v1.jsonl"
    event: dict[str, Any] = {
        "id": "old",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "type": "custom",
        "name": "legacy",
        "input": None,
        "output": None,
        "metadata": {},
        "duration_ms": None,
        "cost": None,
        "parent_id": None,
        "span_id": None,
        "schema_version": 1,
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert load_events(path)[0].schema_version == 1


def test_root_import_is_lazy_without_langchain(tmp_path: Path) -> None:
    script = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'langchain_core' or name.startswith('langchain_core.'):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import agent_cassette
assert callable(agent_cassette.wrap_langchain)
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
