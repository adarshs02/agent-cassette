import pytest
from retrace.agent.executor import ToolExecutor
from retrace.agent.loop import run_agent
from retrace.agent.state import IncidentState, Stage
from retrace.cassette import open_clients
from retrace.config import Settings
from retrace.datahub.fake import FakeDataHubSession
from retrace.datahub.mcp_client import DataHubConnection
from retrace.pipeline.workspace import prepare
from retrace.testing import ScriptedModel, unit_cents_script
from retrace.tools.datahub import DataHubTools
from retrace.tools.repair import Repairer, Transforms
from retrace.tools.warehouse import Warehouse

from agent_cassette import EventType, InjectionRule, Raise, RateLimitError, compare_cassettes


def _run(mode, root, baseline, cassette, injections=(), source=None):
    ws = prepare(root / "ws", fault="unit_cents")
    state = IncidentState(scenario="unit_cents", report="KPI looks off")
    with open_clients(
        mode,
        settings=Settings(),
        cassette_path=cassette,
        injections=injections,
        source=source,
        model_factory=lambda: ScriptedModel(unit_cents_script()),
        datahub_factory=lambda c: DataHubConnection.from_session(FakeDataHubSession(), c),
    ) as clients:
        ex = ToolExecutor(
            state,
            Warehouse(ws.warehouse, baseline),
            Transforms(ws),
            Repairer(ws, baseline, root / "scratch"),
            DataHubTools(clients.datahub),
        )
        stats = run_agent(
            clients.anthropic, ex, state, model="claude-sonnet-5", sleep=lambda s: None
        )
    return state, stats


def test_record_then_replay_offline(tmp_path, baseline, monkeypatch):
    cassette = tmp_path / "unit_cents.jsonl"
    recorded, rstats = _run("record", tmp_path / "rec", baseline, cassette)
    assert recorded.stage is Stage.WRITTEN_BACK
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    replayed, pstats = _run("replay", tmp_path / "rep", baseline, cassette)
    assert replayed.stage is Stage.WRITTEN_BACK
    assert replayed.patched == recorded.patched
    assert pstats.to_dict() == rstats.to_dict()


def test_recordings_are_byte_identical_across_workspaces(tmp_path, baseline):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _run("record", tmp_path / "one", baseline, a)
    _run("record", tmp_path / "two", baseline, b)
    report = compare_cassettes(a, b)
    assert report.to_text() == compare_cassettes(a, a).to_text()


def test_datahub_outage_injection_is_replayable(tmp_path, baseline):
    base = tmp_path / "base.jsonl"
    _run("record", tmp_path / "base", baseline, base)
    outage = tuple(
        InjectionRule(
            Raise(TimeoutError("DataHub MCP call timed out")),
            event_type=EventType.TOOL_CALL,
            occurrence=n,
        )
        for n in range(2, 41)
    )
    chaos = tmp_path / "chaos.jsonl"
    recorded, _ = _run("record", tmp_path / "chaos", baseline, chaos, outage, base)
    assert recorded.stage is not Stage.NO_INCIDENT
    assert recorded.stage is not Stage.WRITTEN_BACK
    replayed, _ = _run("replay", tmp_path / "chaos_rep", baseline, chaos)
    assert replayed.stage is recorded.stage


def test_rate_limit_injection_still_succeeds(tmp_path, baseline):
    base = tmp_path / "base.jsonl"
    _run("record", tmp_path / "base", baseline, base)
    rule = (
        InjectionRule(
            Raise(RateLimitError("rate limited")), event_type=EventType.MODEL_CALL, occurrence=3
        ),
    )
    limited = tmp_path / "rl.jsonl"
    recorded, _ = _run("record", tmp_path / "rl", baseline, limited, rule, base)
    assert recorded.stage is Stage.WRITTEN_BACK
    replayed, _ = _run("replay", tmp_path / "rl_rep", baseline, limited)
    assert replayed.stage is Stage.WRITTEN_BACK


def test_injections_require_existing_source(tmp_path):
    with (
        pytest.raises(FileNotFoundError),
        open_clients(
            "record",
            settings=Settings(),
            cassette_path=tmp_path / "x.jsonl",
            injections=(InjectionRule(Raise(TimeoutError("x")), event_type=EventType.TOOL_CALL),),
            source=tmp_path / "missing.jsonl",
        ),
    ):
        pass
