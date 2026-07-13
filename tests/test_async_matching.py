from __future__ import annotations

import asyncio

from agent_cassette import Cassette, EventType


def test_async_record_and_replay_never_awaits_live_code(tmp_path):
    path = tmp_path / "async.jsonl"
    live_calls = 0

    async def scenario():
        nonlocal live_calls

        async def live_model():
            nonlocal live_calls
            live_calls += 1
            return {"answer": "recorded"}

        async with Cassette.record(path) as cassette:
            recorded = await cassette.acall(
                EventType.MODEL_CALL,
                "answer",
                {"prompt": "hi"},
                live_model,
            )
        async with Cassette.replay(path) as cassette:
            replayed = await cassette.acall(
                EventType.MODEL_CALL,
                "answer",
                {"prompt": "hi"},
                live_model,
            )
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded == replayed == {"answer": "recorded"}
    assert live_calls == 1


def test_replay_can_ignore_dynamic_dotted_paths(tmp_path):
    path = tmp_path / "dynamic.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(
            EventType.MODEL_CALL,
            "answer",
            input={"request": {"id": "first", "prompt": "hi"}},
            output="hello",
        )

    with Cassette.replay(path, ignore_paths=("request.id",)) as cassette:
        assert (
            cassette.call(
                EventType.MODEL_CALL,
                "answer",
                {"request": {"id": "second", "prompt": "hi"}},
            )
            == "hello"
        )


def test_subset_matching_allows_new_optional_parameters(tmp_path):
    path = tmp_path / "subset.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.TOOL_CALL, "search", input={"query": "agents"}, output=[])

    with Cassette.replay(path, match="subset") as cassette:
        assert (
            cassette.call(
                EventType.TOOL_CALL,
                "search",
                {"query": "agents", "timeout": 10},
            )
            == []
        )
