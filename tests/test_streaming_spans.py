from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from agent_cassette import Cassette, EventType, wrap_openai


@dataclass
class FakeChunk:
    sequence: int
    text: str

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return {"sequence": self.sequence, "text": self.text}


class SyncStream:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self) -> SyncStream:
        return self

    def __next__(self) -> FakeChunk:
        return next(self._chunks)

    def close(self) -> None:
        self.closed = True


class AsyncStream:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> FakeChunk:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> SyncStream:
        self.calls += 1
        return SyncStream([FakeChunk(1, "hel"), FakeChunk(2, "lo")])


class FakeAsyncResponses:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: Any) -> AsyncStream:
        self.calls += 1
        return AsyncStream([FakeChunk(1, "async "), FakeChunk(2, "stream")])


class FailingStream:
    def __init__(self) -> None:
        self._first = True

    def __iter__(self):
        return self

    def __next__(self):
        if self._first:
            self._first = False
            return FakeChunk(1, "partial")
        raise TimeoutError("stream interrupted")


class FailingResponses:
    def create(self, **kwargs: Any) -> FailingStream:
        return FailingStream()


class StartFailingResponses:
    def create(self, **kwargs: Any) -> FailingStream:
        raise ConnectionError("stream could not start")


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FailingOpenAI:
    def __init__(self) -> None:
        self.responses = FailingResponses()


class StartFailingOpenAI:
    def __init__(self) -> None:
        self.responses = StartFailingResponses()


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.responses = FakeAsyncResponses()


def test_events_persist_incrementally_without_final_duplicates(tmp_path):
    path = tmp_path / "incremental.jsonl"

    with Cassette.record(path) as recorder:
        recorder.add(EventType.CUSTOM, "first", output=1)
        assert len(path.read_text().splitlines()) == 1
        recorder.add(EventType.CUSTOM, "second", output=2)
        assert len(path.read_text().splitlines()) == 2

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["name"] for event in events] == ["first", "second"]


def test_nested_sync_and_async_spans_propagate_relationships(tmp_path):
    path = tmp_path / "spans.jsonl"

    async def record() -> None:
        async with Cassette.record(path) as recorder:
            with recorder.span("outer") as outer:
                recorder.call(EventType.CUSTOM, "sync", None, lambda: "done")
                async with recorder.span("inner") as inner:
                    await recorder.acall(
                        EventType.CUSTOM, "async", None, lambda: asyncio.sleep(0, result="done")
                    )
            assert outer == "outer"
            assert inner == "inner"
            assert recorder.events[0].span_id == "outer"
            assert recorder.events[0].parent_id is None
            assert recorder.events[1].span_id == "inner"
            assert recorder.events[1].parent_id == "outer"

    asyncio.run(record())


def test_sync_stream_records_and_replays_without_live_call(tmp_path):
    path = tmp_path / "sync-stream.jsonl"
    live = FakeOpenAI()

    with Cassette.record(path) as cassette:
        stream = wrap_openai(live, cassette).responses.create(
            model="gpt-test", input="hello", stream=True
        )
        recorded = list(stream)
        assert len(cassette.events) == 1

    replay_live = FakeOpenAI()
    with Cassette.replay(path) as cassette:
        stream = wrap_openai(replay_live, cassette).responses.create(
            model="gpt-test", input="hello", stream=True
        )
        replayed = list(stream)

    assert [(chunk.sequence, chunk.text) for chunk in recorded] == [(1, "hel"), (2, "lo")]
    assert [(chunk.sequence, chunk.text) for chunk in replayed] == [(1, "hel"), (2, "lo")]
    assert live.responses.calls == 1
    assert replay_live.responses.calls == 0


def test_hybrid_stream_replays_prefix_and_records_live_suffix(tmp_path):
    source = tmp_path / "source-stream.jsonl"
    replayed_output = tmp_path / "replayed-stream.jsonl"
    live_output = tmp_path / "live-stream.jsonl"
    baseline = FakeOpenAI()
    with Cassette.record(source) as cassette:
        list(
            wrap_openai(baseline, cassette).responses.create(
                model="gpt-test", input="hello", stream=True
            )
        )

    replay_client = FakeOpenAI()
    with Cassette.fork(source, replayed_output, at=1) as cassette:
        replayed = list(
            wrap_openai(replay_client, cassette).responses.create(
                model="gpt-test", input="hello", stream=True
            )
        )
    live_client = FakeOpenAI()
    with Cassette.fork(source, live_output, at=0) as cassette:
        live = list(
            wrap_openai(live_client, cassette).responses.create(
                model="gpt-test", input="changed", stream=True
            )
        )

    assert [chunk.text for chunk in replayed] == ["hel", "lo"]
    assert [chunk.text for chunk in live] == ["hel", "lo"]
    assert replay_client.responses.calls == 0
    assert live_client.responses.calls == 1
    assert len(json.loads(live_output.read_text().splitlines()[0])["output"]["chunks"]) == 2


def test_stream_failure_is_recorded_and_replayed(tmp_path):
    path = tmp_path / "failed-stream.jsonl"
    live = FailingOpenAI()

    with Cassette.record(path) as cassette:
        stream = wrap_openai(live, cassette).responses.create(
            model="gpt-test", input="hello", stream=True
        )
        assert next(stream).text == "partial"
        try:
            next(stream)
        except TimeoutError:
            pass

    with Cassette.replay(path) as cassette:
        replayed = wrap_openai(None, cassette).responses.create(
            model="gpt-test", input="hello", stream=True
        )
        assert next(replayed).text == "partial"
        try:
            next(replayed)
        except TimeoutError as error:
            assert str(error) == "stream interrupted"
        else:
            raise AssertionError("recorded stream failure did not replay")


def test_stream_start_failure_is_recorded_and_replayed(tmp_path):
    path = tmp_path / "stream-start-error.jsonl"

    with Cassette.record(path) as cassette:
        try:
            wrap_openai(StartFailingOpenAI(), cassette).responses.create(
                model="gpt-test", input="hello", stream=True
            )
        except ConnectionError:
            pass

    with Cassette.replay(path) as cassette:
        replayed = wrap_openai(None, cassette).responses.create(
            model="gpt-test", input="hello", stream=True
        )
        try:
            next(replayed)
        except ConnectionError as error:
            assert str(error) == "stream could not start"
        else:
            raise AssertionError("recorded stream start failure did not replay")


def test_async_stream_records_and_replays_without_live_call(tmp_path):
    path = tmp_path / "async-stream.jsonl"
    live = FakeAsyncOpenAI()
    replay_live = FakeAsyncOpenAI()

    async def collect(stream: Any) -> list[Any]:
        return [chunk async for chunk in stream]

    async def scenario() -> tuple[list[Any], list[Any]]:
        async with Cassette.record(path) as cassette:
            stream = await wrap_openai(live, cassette).responses.create(
                model="gpt-test", input="hello", stream=True
            )
            recorded = await collect(stream)
        async with Cassette.replay(path) as cassette:
            stream = await wrap_openai(replay_live, cassette, asynchronous=True).responses.create(
                model="gpt-test", input="hello", stream=True
            )
            replayed = await collect(stream)
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert [(chunk.sequence, chunk.text) for chunk in recorded] == [
        (1, "async "),
        (2, "stream"),
    ]
    assert [(chunk.sequence, chunk.text) for chunk in replayed] == [
        (1, "async "),
        (2, "stream"),
    ]
    assert live.responses.calls == 1
    assert replay_live.responses.calls == 0
