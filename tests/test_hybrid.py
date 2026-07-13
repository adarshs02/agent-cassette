from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass

import pytest

from agent_cassette.events import EventType
from agent_cassette.hybrid import (
    LINEAGE_METADATA_KEY,
    Hybrid,
    HybridConfigurationError,
    HybridLiveCallError,
    InjectionRule,
    Raise,
    Return,
)
from agent_cassette.integrations.openai import wrap_openai
from agent_cassette.recorder import Recorder
from agent_cassette.replay import ReplayMismatchError
from agent_cassette.storage import load_events


def _source(path, outputs=("one", "two", "three")):
    with Recorder(path) as recorder:
        for index, output in enumerate(outputs, start=1):
            recorder.add(
                EventType.TOOL_CALL,
                f"step-{index}",
                input={"index": index},
                output=output,
                metadata={"original": index},
            )


def test_replays_prefix_then_permanently_records_live_suffix(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)
    live_calls = []

    def live(value):
        return lambda: live_calls.append(value) or value

    with Hybrid(source, output, prefix=1) as cassette:
        assert cassette.call(EventType.TOOL_CALL, "step-1", {"index": 1}, live("unused")) == "one"
        assert (
            cassette.call(EventType.TOOL_CALL, "step-2", {"index": 2}, live("live-two"))
            == "live-two"
        )
        assert (
            cassette.call(EventType.TOOL_CALL, "step-3", {"index": 3}, live("live-three"))
            == "live-three"
        )
        assert cassette.is_live

    assert live_calls == ["live-two", "live-three"]
    events = load_events(output)
    assert [event.output for event in events] == ["one", "live-two", "live-three"]
    assert [event.metadata[LINEAGE_METADATA_KEY]["mode"] for event in events] == [
        "replayed",
        "live",
        "live",
    ]
    assert events[0].metadata[LINEAGE_METADATA_KEY]["source"] == source.name
    assert str(tmp_path) not in json.dumps(events[0].metadata)
    assert events[0].metadata["original"] == 1


def test_prefix_none_replays_until_exhaustion_then_goes_live(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source, outputs=("one", "two"))
    calls = 0

    def live():
        nonlocal calls
        calls += 1
        return "three-live"

    with Hybrid(source, output, prefix=None) as cassette:
        assert cassette.call(EventType.TOOL_CALL, "step-1", {"index": 1}, live) == "one"
        assert cassette.call(EventType.TOOL_CALL, "step-2", {"index": 2}, live) == "two"
        assert cassette.call(EventType.TOOL_CALL, "step-3", {"index": 3}, live) == "three-live"

    assert calls == 1
    assert [event.output for event in load_events(output)] == ["one", "two", "three-live"]


def test_live_mismatch_policy_forks_without_consuming_or_resuming_replay(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)

    with Hybrid(source, output, mismatch="live") as cassette:
        assert cassette.call(EventType.TOOL_CALL, "different", {}, lambda: "forked") == "forked"
        assert (
            cassette.call(EventType.TOOL_CALL, "step-1", {"index": 1}, lambda: "still-live")
            == "still-live"
        )

    assert [event.output for event in load_events(output)] == ["forked", "still-live"]


def test_raise_mismatch_policy_is_useful_and_does_not_call_live(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)
    called = False

    def live():
        nonlocal called
        called = True

    with pytest.raises(ReplayMismatchError, match="expected name 'step-1', received 'different'"):
        with Hybrid(source, output, mismatch="raise") as cassette:
            cassette.call(EventType.TOOL_CALL, "different", {}, live)

    assert not called
    assert load_events(output) == []


def test_missing_live_callable_has_safe_context_and_output_is_saved(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)

    with pytest.raises(HybridLiveCallError, match="tool_call 'step-2'.*no live function"):
        with Hybrid(source, output, prefix=1) as cassette:
            assert cassette.call(EventType.TOOL_CALL, "step-1", {"index": 1}) == "one"
            cassette.call(EventType.TOOL_CALL, "step-2", {"index": 2})

    assert [event.output for event in load_events(output)] == ["one"]


def test_source_and_output_must_be_different_and_rules_validate(tmp_path):
    path = tmp_path / "same.jsonl"
    _source(path)

    with pytest.raises(HybridConfigurationError, match="must differ"):
        Hybrid(path, path)
    hard_link = tmp_path / "hard-link.jsonl"
    hard_link.hardlink_to(path)
    with pytest.raises(HybridConfigurationError, match="must differ"):
        Hybrid(path, hard_link)
    with pytest.raises(HybridConfigurationError, match="positive integer"):
        InjectionRule(Return("bad"), occurrence=0)
    with pytest.raises(HybridConfigurationError, match="Exception instance"):
        Raise("unsafe")  # type: ignore[arg-type]


def test_ordered_return_injection_selects_type_name_and_occurrence(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)
    rules = [
        InjectionRule(Return("second-injected"), type="tool_call", name="search", occurrence=2),
        InjectionRule(Return("first-injected"), type="tool_call", name="search", occurrence=1),
    ]

    with Hybrid(source, output, prefix=0, injections=rules) as cassette:
        assert cassette.call(EventType.CUSTOM, "search", None, lambda: "custom") == "custom"
        assert cassette.call(EventType.TOOL_CALL, "search", 1, lambda: "unused") == "first-injected"
        assert (
            cassette.call(EventType.TOOL_CALL, "search", 2, lambda: "unused") == "second-injected"
        )
        assert cassette.call(EventType.TOOL_CALL, "search", 3, lambda: "live") == "live"

    events = load_events(output)
    assert [event.output for event in events] == [
        "custom",
        "first-injected",
        "second-injected",
        "live",
    ]
    assert events[1].metadata[LINEAGE_METADATA_KEY]["rule"] == 2
    assert events[2].metadata[LINEAGE_METADATA_KEY]["rule"] == 1
    assert all(event.metadata[LINEAGE_METADATA_KEY]["source"] == source.name for event in events)


def test_injected_raise_is_recorded_and_switches_permanently_live(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)
    rule = InjectionRule(Raise(TimeoutError("planned timeout")), name="step-2")

    with Hybrid(source, output, injections=[rule]) as cassette:
        assert cassette.call(EventType.TOOL_CALL, "step-1", {"index": 1}) == "one"
        with pytest.raises(TimeoutError, match="planned timeout"):
            cassette.call(EventType.TOOL_CALL, "step-2", {"index": 2}, lambda: "unused")
        assert (
            cassette.call(EventType.TOOL_CALL, "step-3", {"index": 3}, lambda: "live-three")
            == "live-three"
        )

    events = load_events(output)
    assert [event.type for event in events] == [
        EventType.TOOL_CALL,
        EventType.ERROR,
        EventType.TOOL_CALL,
    ]
    assert events[1].output == {"type": "TimeoutError", "message": "planned timeout"}
    assert events[1].metadata[LINEAGE_METADATA_KEY]["action"] == "raise"


def test_sync_and_async_call_signatures_remain_parallel():
    sync_parameters = inspect.signature(Hybrid.call).parameters
    async_parameters = inspect.signature(Hybrid.acall).parameters

    assert list(sync_parameters) == list(async_parameters)
    assert [parameter.kind for parameter in sync_parameters.values()] == [
        parameter.kind for parameter in async_parameters.values()
    ]


def test_async_replay_injection_and_live_suffix_match_sync_behavior(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "fork.jsonl"
    _source(source)
    calls = 0

    async def scenario():
        nonlocal calls

        async def live():
            nonlocal calls
            calls += 1
            return "live-three"

        async with Hybrid(
            source,
            output,
            injections=[InjectionRule(Return("injected-two"), name="step-2")],
        ) as cassette:
            first = await cassette.acall(EventType.TOOL_CALL, "step-1", {"index": 1}, live)
            second = await cassette.acall(EventType.TOOL_CALL, "step-2", {"index": 2}, live)
            third = await cassette.acall(EventType.TOOL_CALL, "step-3", {"index": 3}, live)
        return first, second, third

    assert asyncio.run(scenario()) == ("one", "injected-two", "live-three")
    assert calls == 1
    assert [event.output for event in load_events(output)] == ["one", "injected-two", "live-three"]


def test_prepare_stream_serializer_is_once_and_optional(tmp_path):
    source = tmp_path / "source.jsonl"
    serialized_output = tmp_path / "serialized.jsonl"
    plain_output = tmp_path / "plain.jsonl"
    _source(source, outputs=("unused",))
    action_value = {"value": "raw"}
    rule = InjectionRule(Return(action_value), name="step-1")
    calls = 0

    def serialize(value):
        nonlocal calls
        calls += 1
        return {"serialized": value}

    with Hybrid(source, serialized_output, injections=(rule,)) as cassette:
        replayed, value = cassette.prepare_stream(
            EventType.TOOL_CALL,
            "step-1",
            {"index": 1},
            serializer=serialize,
        )
        assert replayed is True
        assert value == {"serialized": action_value}
        assert load_events(serialized_output)[0].output == value
    assert calls == 1

    with Hybrid(
        source,
        plain_output,
        injections=(InjectionRule(Return(action_value), name="step-1"),),
    ) as cassette:
        replayed, value = cassette.prepare_stream(EventType.TOOL_CALL, "step-1", {"index": 1})
        assert replayed is True
        assert value == action_value
    assert load_events(plain_output)[0].output == action_value


@dataclass
class _Response:
    id: str
    output_text: str

    def model_dump(self, mode=None):
        return {"id": self.id, "output_text": self.output_text}


class _Responses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Response(id=f"live-{self.calls}", output_text=kwargs["input"])


class _OpenAI:
    def __init__(self):
        self.responses = _Responses()


def test_hybrid_composes_with_openai_wrapper(tmp_path):
    source = tmp_path / "openai-source.jsonl"
    output = tmp_path / "openai-fork.jsonl"
    original_client = _OpenAI()
    with Recorder(source) as recorder:
        wrap_openai(original_client, recorder).responses.create(model="test", input="one")

    live_client = _OpenAI()
    with Hybrid(source, output, prefix=1) as cassette:
        client = wrap_openai(live_client, cassette)
        replayed = client.responses.create(model="test", input="one")
        live = client.responses.create(model="test", input="two")

    assert replayed.output_text == "one"
    assert live.output_text == "two"
    assert live_client.responses.calls == 1
    assert [event.metadata[LINEAGE_METADATA_KEY]["mode"] for event in load_events(output)] == [
        "replayed",
        "live",
    ]
