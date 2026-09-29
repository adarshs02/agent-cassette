from typing import Any

from anthropic.types import Message
from retrace.agent.executor import ToolExecutor
from retrace.agent.loop import run_agent
from retrace.agent.state import IncidentState, Stage
from retrace.datahub.fake import FakeDataHubSession
from retrace.datahub.mcp_client import DataHubConnection
from retrace.pipeline.workspace import prepare
from retrace.testing import ScriptedModel, unit_cents_script
from retrace.tools.datahub import DataHubTools
from retrace.tools.repair import Repairer, Transforms
from retrace.tools.warehouse import Warehouse


def _run(tmp_path, baseline, script, **kwargs):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(FakeDataHubSession())
    state = IncidentState(scenario="unit_cents", report="KPI looks off")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    model = ScriptedModel(script)
    try:
        stats = run_agent(model, ex, state, model="claude-sonnet-5", sleep=lambda s: None, **kwargs)
    finally:
        conn.close()
    return state, stats, model


def test_scripted_unit_cents_run_reaches_written_back(tmp_path, baseline):
    state, stats, model = _run(tmp_path, baseline, unit_cents_script())
    assert state.stage is Stage.WRITTEN_BACK, state.failure_reason
    assert state.root_cause.asset == "raw.raw_orders"
    assert stats.turns == len(unit_cents_script())
    first = model.requests[0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["messages"][0]["role"] == "user"


def test_turn_budget(tmp_path, baseline):
    loop_forever = [[("get_metric_history", {"days": d})] for d in range(1, 60)]
    state, stats, _ = _run(tmp_path, baseline, loop_forever, max_turns=5)
    assert state.stage is Stage.FAILED and "turn budget" in state.failure_reason
    assert stats.turns == 5


def test_text_only_replies_are_nudged_then_fail(tmp_path, baseline):
    state, _, _ = _run(tmp_path, baseline, [[], [], [], []], max_nudges=2)
    assert state.stage is Stage.FAILED and "without finishing" in state.failure_reason


def test_identical_repeats_fail_after_nudges(tmp_path, baseline):
    same = [[("get_metric_history", {"days": 7})]] * 6
    state, _, _ = _run(tmp_path, baseline, same, max_nudges=2)
    assert state.stage is Stage.FAILED and "repeated" in state.failure_reason


def test_rate_limit_is_retried(tmp_path, baseline):
    from agent_cassette import RateLimitError

    class Flaky(ScriptedModel):
        def create(self, **kwargs):
            if len(self.requests) == 1 and not getattr(self, "_failed", False):
                self._failed = True
                raise RateLimitError("rate limited")
            return super().create(**kwargs)

    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(FakeDataHubSession())
    state = IncidentState(scenario="unit_cents", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "s"),
        DataHubTools(conn),
    )
    try:
        run_agent(Flaky(unit_cents_script()), ex, state, model="m", sleep=lambda s: None)
    finally:
        conn.close()
    assert state.stage is Stage.WRITTEN_BACK


def test_bad_tool_args_do_not_crash_run(tmp_path, baseline):
    script = [[("get_metric_history", {"days": -5})], *unit_cents_script()]
    state, stats, _ = _run(tmp_path, baseline, script)
    assert state.stage is Stage.WRITTEN_BACK, state.failure_reason
    assert stats.turns == len(script)


def test_extra_calls_after_finish_in_batch_are_skipped(tmp_path, baseline):
    script = list(unit_cents_script())
    finish_call = script[-1][0]
    script[-1] = [finish_call, ("get_metric_history", {"days": 7})]
    state, stats, _ = _run(tmp_path, baseline, script)
    assert state.stage is Stage.WRITTEN_BACK, state.failure_reason
    assert stats.turns == len(script)
    # The skipped call was never dispatched, so it never recorded evidence.
    assert not any(item.kind == "metric_history" for item in state.evidence)
    # Only the "finish" call in the last batch was actually dispatched.
    assert stats.tool_calls == len(script)


class _EmptyThenScripted(ScriptedModel):
    """A ScriptedModel subclass whose very first reply has empty content."""

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        if not getattr(self, "_served_empty", False):
            self._served_empty = True
            return Message.model_validate(
                {
                    "id": "msg_000",
                    "type": "message",
                    "role": "assistant",
                    "model": kwargs["model"],
                    "content": [],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        index = len(self.requests) - 1
        step = self._script[index - 1] if index <= len(self._script) else []
        content: list[dict[str, Any]] = [
            {"type": "tool_use", "id": f"toolu_{index:03d}_{j}", "name": name, "input": args}
            for j, (name, args) in enumerate(step)
        ] or [{"type": "text", "text": "Thinking."}]
        return Message.model_validate(
            {
                "id": f"msg_{index:03d}",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": content,
                "stop_reason": "tool_use" if step else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )


def test_empty_content_reply_is_nudged_not_crashed(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(FakeDataHubSession())
    state = IncidentState(scenario="unit_cents", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    model = _EmptyThenScripted(unit_cents_script())
    try:
        run_agent(model, ex, state, model="claude-sonnet-5", sleep=lambda s: None)
    finally:
        conn.close()
    assert state.stage is Stage.WRITTEN_BACK, state.failure_reason
    second_request_messages = model.requests[1]["messages"]
    first_assistant = next(m for m in second_request_messages if m["role"] == "assistant")
    assert first_assistant["content"] == [{"type": "text", "text": "(no content)"}]


def _executor(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(FakeDataHubSession())
    state = IncidentState(scenario="unit_cents", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    return state, ex, conn


class _RawModel:
    """Returns fully custom responses: (content blocks, stop_reason) per turn."""

    def __init__(self, turns: list[tuple[list[dict[str, Any]], str]]) -> None:
        self._turns = turns
        self.requests: list[dict[str, Any]] = []
        self.snapshots: list[list[dict[str, Any]]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        import copy

        self.requests.append(kwargs)
        self.snapshots.append(copy.deepcopy(kwargs["messages"]))
        index = len(self.requests)
        content, stop = (
            self._turns[index - 1]
            if index <= len(self._turns)
            else ([{"type": "text", "text": "done"}], "end_turn")
        )
        return Message.model_validate(
            {
                "id": f"msg_{index:03d}",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": content,
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )


def test_thinking_blocks_are_passed_back_verbatim_in_order(tmp_path, baseline):
    first = [
        {"type": "thinking", "thinking": "look at history", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_metric_history", "input": {"days": 7}},
    ]
    finish = [
        {"type": "tool_use", "id": "toolu_2", "name": "finish", "input": {"report": "r"}},
    ]
    state, ex, conn = _executor(tmp_path, baseline)
    model = _RawModel([(first, "tool_use"), (finish, "tool_use")])
    try:
        run_agent(model, ex, state, model="m", sleep=lambda s: None)
    finally:
        conn.close()
    assistant = model.snapshots[1][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {"type": "thinking", "thinking": "look at history", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_metric_history", "input": {"days": 7}},
    ]


def test_max_tokens_is_raised():
    from retrace.agent import loop

    assert loop.MAX_TOKENS == 16000


def test_get_is_none_safe_for_dict_missing_a_key():
    """A replayed/recorded usage dict from before the cache fields existed
    must not raise KeyError when a newer field is looked up."""
    from retrace.agent.loop import _get

    usage = {"input_tokens": 10, "output_tokens": 2}
    assert _get(usage, "input_tokens") == 10
    assert _get(usage, "cache_read_input_tokens") is None
    assert _get(usage, "cache_creation_input_tokens") is None
    assert int(_get(usage, "cache_read_input_tokens") or 0) == 0


def test_get_is_none_safe_for_objects_missing_an_attribute():
    from retrace.agent.loop import _get

    class _Bare:
        input_tokens = 5

    assert _get(_Bare(), "input_tokens") == 5
    assert _get(_Bare(), "cache_read_input_tokens") is None


def test_run_agent_survives_dict_shaped_usage_without_cache_keys(tmp_path, baseline):
    """An older recorded/replayed response whose usage dict predates the cache
    fields must not crash run_agent; the missing fields must read as 0."""
    state, ex, conn = _executor(tmp_path, baseline)

    class _DictModel:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.messages = self

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.requests.append(kwargs)
            return {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "finish", "input": {"report": "r"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 2},  # no cache_* keys
            }

    try:
        stats = run_agent(_DictModel(), ex, state, model="m", sleep=lambda s: None, max_turns=1)
    finally:
        conn.close()
    assert stats.input_tokens == 10
    assert stats.output_tokens == 2
    assert stats.cache_read_input_tokens == 0
    assert stats.cache_creation_input_tokens == 0


def test_cache_tokens_are_accumulated_from_usage(tmp_path, baseline):
    state, ex, conn = _executor(tmp_path, baseline)

    class _CacheModel:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.messages = self

        def create(self, **kwargs: Any) -> Message:
            self.requests.append(kwargs)
            index = len(self.requests)
            usage = (
                {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 30,
                }
                if index == 1
                else {
                    "input_tokens": 60,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 400,
                    "cache_creation_input_tokens": 0,
                }
            )
            content = [
                {
                    "type": "tool_use",
                    "id": f"toolu_{index}",
                    "name": "get_metric_history",
                    "input": {"days": 7},
                }
            ]
            return Message.model_validate(
                {
                    "id": f"msg_{index:03d}",
                    "type": "message",
                    "role": "assistant",
                    "model": kwargs["model"],
                    "content": content,
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": usage,
                }
            )

    try:
        stats = run_agent(_CacheModel(), ex, state, model="m", sleep=lambda s: None, max_turns=2)
    finally:
        conn.close()
    assert stats.turns == 2
    assert stats.cache_read_input_tokens == 600
    assert stats.cache_creation_input_tokens == 30
    assert stats.to_dict()["cache_read_input_tokens"] == 600
    assert stats.to_dict()["cache_creation_input_tokens"] == 30


def test_truncated_tool_use_is_not_dispatched(tmp_path, baseline):
    truncated = [
        {"type": "tool_use", "id": "toolu_1", "name": "get_metric_history", "input": {"days": 7}},
    ]
    state, ex, conn = _executor(tmp_path, baseline)
    model = _RawModel([(truncated, "max_tokens")])
    try:
        stats = run_agent(model, ex, state, model="m", sleep=lambda s: None, max_turns=2)
    finally:
        conn.close()
    assert model.requests[0]["max_tokens"] == 16000
    assert state.evidence == []
    assert stats.tool_calls == 0
    reply = model.snapshots[1][-1]
    assert reply["role"] == "user"
    assert reply["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "response was truncated at max_tokens; retry with a shorter reply",
            "is_error": True,
        }
    ]


def test_refusal_fails_the_run(tmp_path, baseline):
    state, ex, conn = _executor(tmp_path, baseline)
    model = _RawModel([([{"type": "text", "text": "I can't help."}], "refusal")])
    try:
        run_agent(model, ex, state, model="m", sleep=lambda s: None)
    finally:
        conn.close()
    assert state.stage is Stage.FAILED
    assert state.failure_reason == "model refused"
    assert len(model.requests) == 1


def test_overloaded_error_is_retried(tmp_path, baseline):
    import anthropic
    import httpx
    from retrace.agent.loop import RETRYABLE

    overloaded = getattr(anthropic, "OverloadedError", None)
    if overloaded is None:
        import pytest

        pytest.skip("installed SDK has no OverloadedError")
    assert overloaded in RETRYABLE

    class Flaky(ScriptedModel):
        def create(self, **kwargs):
            if not getattr(self, "_failed", False):
                self._failed = True
                request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                raise overloaded(
                    "overloaded", response=httpx.Response(529, request=request), body=None
                )
            return super().create(**kwargs)

    state, ex, conn = _executor(tmp_path, baseline)
    try:
        run_agent(Flaky(unit_cents_script()), ex, state, model="m", sleep=lambda s: None)
    finally:
        conn.close()
    assert state.stage is Stage.WRITTEN_BACK, state.failure_reason


def test_claim_tool_descriptions_state_asset_format():
    from retrace.agent.prompts import TOOL_SCHEMAS

    by_name = {t["name"]: t for t in TOOL_SCHEMAS}
    exact_format = (
        " asset: schema-qualified table name (schema.table) or its DataHub URN; field: column name."
    )
    for name in ("confirm_root_cause", "escalate_upstream"):
        description = by_name[name]["description"]
        assert exact_format in description, f"Tool '{name}' missing exact format: {description}"
