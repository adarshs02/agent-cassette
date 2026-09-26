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
