import asyncio
import json
from time import perf_counter

import pytest

from agent_cassette import (
    Cassette,
    Delay,
    EventType,
    InjectionRule,
    Raise,
    RateLimitError,
    Return,
)
from agent_cassette.cli import _load_injections
from agent_cassette.hybrid import HybridConfigurationError


def _record_baseline(path):
    with Cassette.record(path) as cassette:
        cassette.call(EventType.TOOL_CALL, "search", {"q": "a"}, lambda: "first")
        cassette.call(EventType.TOOL_CALL, "search", {"q": "b"}, lambda: "second")


def test_delay_injection_sleeps_then_runs_live(tmp_path):
    source = tmp_path / "baseline.jsonl"
    output = tmp_path / "delayed.jsonl"
    _record_baseline(source)

    rule = InjectionRule(Delay(0.05), type="tool_call", name="search")
    with Cassette.fork(source, output, injections=(rule,)) as cassette:
        started = perf_counter()
        result = cassette.call(EventType.TOOL_CALL, "search", {"q": "a"}, lambda: "live")
        elapsed = perf_counter() - started

    assert result == "live"
    assert elapsed >= 0.05
    events = json.loads(output.read_text().splitlines()[0])
    lineage = events["metadata"]["_agent_cassette"]
    assert lineage["mode"] == "injected"
    assert lineage["action"] == "delay"
    assert lineage["delay_seconds"] == 0.05


def test_delay_injection_with_then_raise(tmp_path):
    source = tmp_path / "baseline.jsonl"
    output = tmp_path / "delay-raise.jsonl"
    _record_baseline(source)

    rule = InjectionRule(
        Delay(0.01, then=Raise(TimeoutError("slow backend"))),
        type="tool_call",
        name="search",
    )
    with pytest.raises(TimeoutError, match="slow backend"):
        with Cassette.fork(source, output, injections=(rule,)) as cassette:
            cassette.call(EventType.TOOL_CALL, "search", {"q": "a"}, lambda: "live")


def test_async_delay_injection_with_then_return(tmp_path):
    source = tmp_path / "baseline.jsonl"
    output = tmp_path / "delay-async.jsonl"
    _record_baseline(source)

    rule = InjectionRule(Delay(0.01, then=Return("fallback")), type="tool_call")

    async def scenario():
        async with Cassette.fork(source, output, injections=(rule,)) as cassette:
            return await cassette.acall(EventType.TOOL_CALL, "search", {"q": "a"})

    assert asyncio.run(scenario()) == "fallback"


def test_rate_limit_error_injection_records_and_replays(tmp_path):
    source = tmp_path / "baseline.jsonl"
    output = tmp_path / "rate-limit.jsonl"
    _record_baseline(source)

    rule = InjectionRule(
        Raise(RateLimitError("too many requests", retry_after=2.5)),
        type="tool_call",
    )
    with pytest.raises(RateLimitError, match="too many requests"):
        with Cassette.fork(source, output, injections=(rule,)) as cassette:
            cassette.call(EventType.TOOL_CALL, "search", {"q": "a"}, lambda: "live")

    with pytest.raises(RateLimitError, match="too many requests"):
        with Cassette.replay(output, strict=False) as cassette:
            cassette.call(EventType.TOOL_CALL, "search", {"q": "a"})


def test_rate_limit_error_is_a_connection_error():
    error = RateLimitError("slow down", retry_after=1.0)
    assert isinstance(error, ConnectionError)
    assert error.retry_after == 1.0


def test_delay_validation():
    with pytest.raises(HybridConfigurationError, match="seconds"):
        Delay(-1)
    with pytest.raises(HybridConfigurationError, match="then-action"):
        Delay(1, then="oops")
    with pytest.raises(HybridConfigurationError, match="then-action"):
        Delay(1, then=Delay(1))


def test_cli_injections_parse_delay_and_rate_limit(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            [
                {"type": "tool_call", "action": "delay", "seconds": 0.5},
                {
                    "type": "tool_call",
                    "action": "delay",
                    "seconds": 0.1,
                    "then": {"action": "raise", "error": "TimeoutError", "message": "slow"},
                },
                {
                    "type": "model_call",
                    "action": "raise",
                    "error": "RateLimitError",
                    "message": "rate limited",
                    "retry_after": 3,
                },
            ]
        ),
        encoding="utf-8",
    )

    rules = _load_injections(rules_path)

    assert isinstance(rules[0].action, Delay)
    assert rules[0].action.seconds == 0.5
    assert rules[0].action.then is None
    assert isinstance(rules[1].action.then, Raise)
    assert isinstance(rules[1].action.then.error, TimeoutError)
    assert isinstance(rules[2].action, Raise)
    assert isinstance(rules[2].action.error, RateLimitError)
    assert rules[2].action.error.retry_after == 3.0


def test_cli_injections_reject_nested_delay(tmp_path):
    rules_path = tmp_path / "nested.json"
    rules_path.write_text(
        json.dumps([{"action": "delay", "seconds": 1, "then": {"action": "delay", "seconds": 1}}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="injection action"):
        _load_injections(rules_path)
