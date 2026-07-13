from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from agent_cassette import Cassette, EventType, langchain_callback_handler, wrap_langchain
from agent_cassette.recorder import Recorder
from agent_cassette.storage import load_events


def _start_chain(
    handler: BaseCallbackHandler,
    run_id: UUID,
    *,
    parent_id: UUID | None = None,
    name: str = "example-chain",
    metadata: dict[str, Any] | None = None,
) -> None:
    handler.on_chain_start(
        {"name": name},
        {"request": name},
        run_id=run_id,
        parent_run_id=parent_id,
        metadata=metadata,
    )


def test_nested_lifecycle_events_preserve_late_child_relationships(tmp_path: Path) -> None:
    path = tmp_path / "nested.jsonl"
    parent = uuid4()
    child = uuid4()
    with Cassette.record(path) as recorder:
        handler = langchain_callback_handler(recorder)
        _start_chain(handler, parent)
        handler.on_tool_start(
            {"name": "lookup"},
            "question",
            run_id=child,
            parent_run_id=parent,
        )
        handler.on_chain_end({"answer": "pending child"}, run_id=parent)
        handler.on_tool_end("done", run_id=child)

    events = load_events(path)
    assert [event.name for event in events] == [
        "langchain.chain.success",
        "langchain.tool.success",
    ]
    assert events[0].span_id == str(parent)
    assert events[0].parent_id is None
    assert events[1].span_id == str(child)
    assert events[1].parent_id == str(parent)
    assert all(event.metadata["_agent_cassette"] == {"observational": True} for event in events)


def test_all_supported_categories_and_duplicate_terminal_are_recorded_once(tmp_path: Path) -> None:
    path = tmp_path / "categories.jsonl"
    with Cassette.record(path) as recorder:
        handler = langchain_callback_handler(recorder)

        chain_id = uuid4()
        _start_chain(handler, chain_id)
        handler.on_chain_end({"ok": True}, run_id=chain_id)
        handler.on_chain_error(RuntimeError("late"), run_id=chain_id)

        parser_id = uuid4()
        _start_chain(
            handler,
            parser_id,
            name="StructuredOutputParser",
            metadata={"type": "output_parser"},
        )
        handler.on_chain_end({"parsed": True}, run_id=parser_id)

        model_id = uuid4()
        handler.on_llm_start({"name": "model"}, ["hello"], run_id=model_id)
        handler.on_llm_end({"generations": []}, run_id=model_id)

        retriever_id = uuid4()
        handler.on_retriever_start({"name": "search"}, "query", run_id=retriever_id)
        handler.on_retriever_end([], run_id=retriever_id)

        tool_id = uuid4()
        handler.on_tool_start({"name": "tool"}, "{}", run_id=tool_id, inputs={"x": 1})
        handler.on_tool_error(ValueError("bad tool"), run_id=tool_id)

        cancelled_id = uuid4()
        _start_chain(handler, cancelled_id, name="cancelled-chain")
        handler.on_chain_error(asyncio.CancelledError(), run_id=cancelled_id)

    events = load_events(path)
    assert [event.metadata["category"] for event in events] == [
        "chain",
        "parser",
        "model",
        "retriever",
        "tool",
        "chain",
    ]
    assert [event.metadata["outcome"] for event in events] == [
        "success",
        "success",
        "success",
        "success",
        "error",
        "error",
    ]
    assert events[-2].output == {"type": "ValueError", "message": "bad tool"}
    assert events[-1].output == {"type": "CancelledError", "message": ""}


def test_observational_events_are_excluded_from_strict_replay_only_for_exact_flag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observational.jsonl"
    with Cassette.record(path) as recorder:
        recorder.add(
            EventType.CUSTOM,
            "trace",
            metadata={"_agent_cassette": {"observational": True}},
        )
        recorder.add(EventType.CUSTOM, "actual", input={"x": 1}, output="done")

    with Cassette.replay(path) as replayer:
        assert replayer.call(EventType.CUSTOM, "actual", {"x": 1}) == "done"

    not_exact = tmp_path / "not-exact.jsonl"
    with Cassette.record(not_exact) as recorder:
        recorder.add(
            EventType.CUSTOM,
            "integer-flag",
            metadata={"_agent_cassette": {"observational": 1}},
        )
    with Cassette.replay(not_exact) as replayer:
        assert replayer.call(EventType.CUSTOM, "integer-flag") is None


def test_callback_is_noop_during_replay(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    with Cassette.record(path) as recorder:
        recorder.add(EventType.CUSTOM, "actual", output="saved")

    before = path.read_bytes()
    with Cassette.replay(path) as replayer:
        handler = langchain_callback_handler(replayer)
        run_id = uuid4()
        _start_chain(handler, run_id)
        handler.on_chain_end({"ignored": True}, run_id=run_id)
        assert replayer.call(EventType.CUSTOM, "actual") == "saved"
    assert path.read_bytes() == before


class _PeerHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.errors = 0

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.starts += 1

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self.ends += 1

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self.errors += 1


def test_callback_composes_with_sync_async_streaming_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "runnable.jsonl"
    peer = _PeerHandler()
    runnable = RunnableLambda(lambda value: value.upper())
    with Cassette.record(path) as recorder:
        handler = langchain_callback_handler(recorder)
        config: RunnableConfig = {"callbacks": [handler, peer]}
        assert runnable.invoke("one", config=config) == "ONE"
        assert asyncio.run(runnable.ainvoke("two", config=config)) == "TWO"
        assert list(runnable.stream("three", config=config)) == ["THREE"]

        failing = RunnableLambda(lambda _: (_ for _ in ()).throw(ValueError("failed")))
        with pytest.raises(ValueError, match="failed"):
            failing.invoke("four", config=config)

    events = load_events(path)
    assert len(events) == 4
    assert [event.metadata["outcome"] for event in events] == [
        "success",
        "success",
        "success",
        "error",
    ]
    assert peer.starts == 4
    assert peer.ends == 3
    assert peer.errors == 1


def test_handler_adds_no_duplicate_provider_boundary(tmp_path: Path) -> None:
    path = tmp_path / "provider.jsonl"
    model_id = uuid4()
    with Cassette.record(path) as recorder:
        handler = langchain_callback_handler(recorder)
        recorder.add(EventType.MODEL_CALL, "provider.chat", input="hello", output="world")
        handler.on_llm_start({"name": "provider-chat"}, ["hello"], run_id=model_id)
        handler.on_llm_end({"text": "world"}, run_id=model_id)

    events = load_events(path)
    assert sum(event.type == EventType.MODEL_CALL for event in events) == 1
    assert sum(event.metadata.get("lifecycle") is True for event in events) == 1
    with Cassette.replay(path) as replayer:
        assert replayer.call(EventType.MODEL_CALL, "provider.chat", "hello") == "world"


def test_unsupported_values_never_use_arbitrary_repr(tmp_path: Path) -> None:
    class Unsafe:
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    path = tmp_path / "safe.jsonl"
    run_id = uuid4()
    with Cassette.record(path) as recorder:
        handler = langchain_callback_handler(recorder)
        handler.on_chain_start(None, Unsafe(), run_id=run_id)  # type: ignore[arg-type]
        handler.on_chain_end(Unsafe(), run_id=run_id)  # type: ignore[arg-type]

    event = load_events(path)[0]
    marker = "__agent_cassette_unserializable__"
    assert marker in event.input
    assert marker in event.output
    assert event.input[marker]["type"].endswith("Unsafe")


def test_concurrent_recorder_and_callback_writes_leave_valid_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    count = 40
    with Recorder(path) as recorder:
        handler = langchain_callback_handler(recorder)

        def record(index: int) -> None:
            run_id = UUID(int=index + 1)
            _start_chain(handler, run_id, name=f"chain-{index}")
            recorder.add(EventType.CUSTOM, f"direct-{index}", output=index)
            handler.on_chain_end({"index": index}, run_id=run_id)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(record, range(count)))

    events = load_events(path)
    assert len(events) == count * 2
    assert len({event.id for event in events}) == count * 2
    assert {event.name for event in events if event.name.startswith("direct-")} == {
        f"direct-{index}" for index in range(count)
    }


def test_async_gather_preserves_concurrent_nested_lifecycle_relationships(
    tmp_path: Path,
) -> None:
    path = tmp_path / "async-concurrent.jsonl"
    count = 12

    async def record() -> None:
        async with Cassette.record(path) as recorder:
            handler = langchain_callback_handler(recorder)

            async def trace(index: int) -> None:
                parent = UUID(int=index + 1)
                child = UUID(int=count + index + 1)
                _start_chain(handler, parent, name=f"parent-{index}")
                await asyncio.sleep(0)
                handler.on_tool_start(
                    {"name": f"child-{index}"},
                    str(index),
                    run_id=child,
                    parent_run_id=parent,
                )
                await asyncio.sleep(0)
                handler.on_tool_end({"index": index}, run_id=child)
                await asyncio.sleep(0)
                handler.on_chain_end({"index": index}, run_id=parent)

            await asyncio.gather(*(trace(index) for index in range(count)))

    asyncio.run(record())
    events = load_events(path)
    assert len(events) == count * 2
    assert len({event.id for event in events}) == count * 2
    by_span = {event.span_id: event for event in events}
    assert len(by_span) == count * 2
    for index in range(count):
        parent_id = str(UUID(int=index + 1))
        child_id = str(UUID(int=count + index + 1))
        parent = by_span[parent_id]
        child = by_span[child_id]
        assert parent.parent_id is None
        assert parent.metadata["run_name"] == f"parent-{index}"
        assert child.parent_id == parent_id
        assert child.metadata["run_name"] == f"child-{index}"


def test_callback_and_runnable_wrapper_record_distinct_boundary_and_trace(tmp_path: Path) -> None:
    path = tmp_path / "wrapped.jsonl"
    with Cassette.record(path) as recorder:
        callback = langchain_callback_handler(recorder)
        runnable = wrap_langchain(RunnableLambda(lambda value: value + "!"), recorder)
        assert runnable.invoke("hello", config={"callbacks": [callback]}) == "hello!"

    events = load_events(path)
    boundary = [event for event in events if event.name == "langchain.runnable.invoke"]
    traces = [event for event in events if event.metadata.get("lifecycle") is True]
    assert len(boundary) == 1
    assert len(traces) == 1
    with Cassette.replay(path) as replayer:
        callback = langchain_callback_handler(replayer)
        assert (
            wrap_langchain(None, replayer).invoke("hello", config={"callbacks": [callback]})
            == "hello!"
        )
