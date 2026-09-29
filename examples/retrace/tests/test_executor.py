import pytest
from retrace.agent.executor import ToolExecutor
from retrace.agent.state import IncidentState, Stage
from retrace.datahub.catalog import dataset_urn
from retrace.datahub.fake import FakeDataHubSession
from retrace.datahub.mcp_client import DataHubConnection
from retrace.faults import get_fault
from retrace.pipeline.workspace import prepare
from retrace.tools.datahub import DataHubTools
from retrace.tools.repair import Repairer, Transforms
from retrace.tools.warehouse import Warehouse


@pytest.fixture()
def executor(tmp_path, baseline):
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
    yield ex
    conn.close()


def test_fact_tools_record_evidence(executor):
    out = executor.dispatch(
        "datahub_lineage", {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream"}
    )
    assert out["evidence_id"] == "ev_001"
    out = executor.dispatch(
        "profile_column",
        {"table": "raw.raw_orders", "column": "amount", "group_by": "payment_processor"},
    )
    assert out["evidence_id"] == "ev_002"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("no_such_tool", {}),
        ("run_sql", {}),
        ("run_sql", {"query": 5}),
        ("profile_column", {"table": "raw.raw_orders"}),
        ("propose_repair", {"file": "stg_orders.sql", "new_sql": "x"}),
        ("finish", {"report": "done"}),
        (
            "confirm_root_cause",
            {
                "asset": "raw.raw_orders",
                "field": "amount",
                "summary": "s",
                "evidence_ids": ["ev_404"],
            },
        ),
    ],
)
def test_dispatch_rejects_bad_arguments(executor, name, args):
    out = executor.dispatch(name, args)
    assert "error" in out
    assert executor.state.stage is Stage.INVESTIGATING


def test_full_happy_path(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch(
        "profile_column",
        {"table": "raw.raw_orders", "column": "amount", "group_by": "payment_processor"},
    )
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "cents",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "stage": "ROOT_CAUSE_CONFIRMED"}
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    out = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
    assert out["passed"] is True and ex.state.stage is Stage.VERIFIED
    assert ex.dispatch("write_back", {"summary": "fixed"})["written_back"] is True
    assert ex.state.stage is Stage.WRITTEN_BACK
    assert ex.dispatch("finish", {"report": "all good"}) == {"finished": True}
    assert ex.finished and ex.state.final_report == "all good"


def test_repair_budget_exhaustion_fails_run(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "s",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    bad = ex.transforms.read("stg_orders")
    for _ in range(3):
        out = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": bad})
    assert "error" in out and "budget" in out["error"]
    assert ex.state.stage is Stage.FAILED and ex.finished


@pytest.mark.parametrize("days", [-5, 0, 91, 10**18, True, False])
def test_get_metric_history_rejects_bad_days(executor, days):
    out = executor.dispatch("get_metric_history", {"days": days})
    assert "error" in out
    assert executor.state.stage is Stage.INVESTIGATING


@pytest.mark.parametrize("last_days", [-1, 0, 91, 10**18, True, False])
def test_compare_to_baseline_rejects_bad_last_days(executor, last_days):
    out = executor.dispatch("compare_to_baseline", {"last_days": last_days})
    assert "error" in out
    assert executor.state.stage is Stage.INVESTIGATING


def test_bad_day_args_do_not_break_subsequent_calls(executor):
    bad = executor.dispatch("get_metric_history", {"days": -5})
    assert "error" in bad
    good = executor.dispatch("get_metric_history", {"days": 7})
    assert good["evidence_id"] == "ev_001"


def test_dispatch_after_finished_is_rejected(executor):
    ex = executor
    ex.finished = True
    assert ex.dispatch("get_metric_history", {"days": 7}) == {"error": "run already finished"}
    assert len(ex.state.evidence) == 0


def test_second_write_back_after_escalation_errors(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    out = ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "upstream freshness issue",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "stage": "ESCALATED"}
    assert ex.dispatch("write_back", {"summary": "first"})["written_back"] is True
    out = ex.dispatch("write_back", {"summary": "second"})
    assert "error" in out and "already written back" in out["error"]


def test_confirm_root_cause_after_a_passing_repair_errors_and_claim_unchanged(executor):
    # A root cause can be revised freely up until a repair actually passes; once
    # verified, the claim is locked (a second confirm before that point is now a
    # revision, not an error -- see test_revise_root_cause_before_repair_is_accepted).
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "first",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    passed = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
    assert passed["passed"] is True and ex.state.stage is Stage.VERIFIED
    original = ex.state.root_cause
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "second",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert "error" in out
    assert "already confirmed and repaired" in out["error"]
    assert ex.state.root_cause == original
    assert ex.state.stage is Stage.VERIFIED


def test_revise_root_cause_before_repair_is_accepted(executor):
    ex = executor
    _confirm(ex)  # confirms raw.raw_orders
    original = ex.state.root_cause
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "staging.stg_orders",
            "field": "amount",
            "summary": "revised: the cast happens in staging, not raw",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "revised": True, "stage": "ROOT_CAUSE_CONFIRMED"}
    assert ex.state.root_cause != original
    assert ex.state.root_cause.asset == "staging.stg_orders"


def test_revise_root_cause_after_failed_repair_changes_allowed_repair_targets(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_customers",
            "summary": "first guess: the customer join",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    bad = ex.transforms.read("stg_customers")
    out = ex.dispatch("propose_repair", {"file": "stg_customers.sql", "new_sql": bad})
    assert out["passed"] is False and ex.state.stage is Stage.REPAIRING

    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "revised: it is actually the orders feed",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "revised": True, "stage": "REPAIRING"}
    assert ex.state.root_cause.asset == "raw.raw_orders"

    # stg_orders.sql is not downstream of raw.raw_customers (the stale claim), only
    # of raw.raw_orders (the revised one). Passing here proves propose_repair's
    # allowed-targets check used the revised asset, not the stale one.
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    out = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
    assert out["passed"] is True and ex.state.stage is Stage.VERIFIED


def test_confirm_root_cause_after_escalate_errors(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "upstream freshness issue",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert ex.state.stage is Stage.ESCALATED
    original = ex.state.root_cause
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "too late",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert "error" in out
    assert out["error"] == "cannot confirm a root cause in stage ESCALATED"
    assert ex.state.root_cause == original


def test_confirm_root_cause_after_write_back_errors(executor):
    ex = executor
    _confirm(ex)
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    passed = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
    assert passed["passed"] is True and ex.state.stage is Stage.VERIFIED
    assert ex.dispatch("write_back", {"summary": "fixed"})["written_back"] is True
    assert ex.state.stage is Stage.WRITTEN_BACK
    original = ex.state.root_cause
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "too late",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert "error" in out
    assert out["error"] == "root cause already confirmed and repaired"
    assert ex.state.root_cause == original


def test_confirm_root_cause_after_no_incident_errors(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault=None)
    conn = DataHubConnection.from_session(FakeDataHubSession())
    state = IncidentState(scenario="none", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    try:
        evidence = ex.dispatch("compare_to_baseline", {"last_days": 14})
        out = ex.dispatch(
            "declare_no_incident",
            {"summary": "within band", "evidence_ids": [evidence["evidence_id"]]},
        )
        assert out["accepted"] is True and ex.state.stage is Stage.NO_INCIDENT
        original = ex.state.root_cause
        out = ex.dispatch(
            "confirm_root_cause",
            {
                "asset": "raw.raw_orders",
                "field": "amount",
                "summary": "too late",
                "evidence_ids": [evidence["evidence_id"]],
            },
        )
        assert "error" in out
        assert out["error"] == "cannot confirm a root cause in stage NO_INCIDENT"
        assert ex.state.root_cause == original
    finally:
        conn.close()


def test_second_escalate_errors(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "first",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    out = ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "second",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert "error" in out


def test_confirm_then_escalate_clears_root_cause_and_claims_escalation(executor):
    ex = executor
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "root",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    out = ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "actually upstream",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "stage": "ESCALATED"}
    assert ex.state.root_cause is None
    assert ex.state.claimed() is ex.state.escalation


def _confirm(ex):
    ex.dispatch(
        "datahub_lineage",
        {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
    )
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    out = ex.dispatch(
        "confirm_root_cause",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "summary": "root",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out["accepted"] is True


def test_escalate_after_failed_repair_is_accepted(executor):
    ex = executor
    _confirm(ex)
    bad = ex.transforms.read("stg_orders")
    out = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": bad})
    assert out["passed"] is False and ex.state.stage is Stage.REPAIRING
    out = ex.dispatch(
        "escalate_upstream",
        {
            "asset": "raw.raw_orders",
            "field": "amount",
            "reason": "the feed itself is wrong",
            "evidence_ids": ["ev_001", "ev_002"],
        },
    )
    assert out == {"accepted": True, "stage": "ESCALATED"}
    assert ex.state.stage is Stage.ESCALATED and ex.state.root_cause is None
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    out = ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
    assert "error" in out and ex.state.stage is Stage.ESCALATED


def test_declare_no_incident_rejects_string_evidence_ids(executor):
    ex = executor
    ex.dispatch("get_metric_history", {"days": 7})
    out = ex.dispatch("declare_no_incident", {"summary": "fine", "evidence_ids": "ev_001"})
    assert "error" in out and "evidence_ids must be a list of strings" in out["error"]
    assert ex.state.stage is Stage.INVESTIGATING


def test_claim_rejects_string_evidence_ids(executor):
    ex = executor
    ex.dispatch("profile_column", {"table": "raw.raw_orders", "column": "amount"})
    out = ex.dispatch(
        "confirm_root_cause",
        {"asset": "raw.raw_orders", "summary": "s", "evidence_ids": "ev_001"},
    )
    assert "error" in out and "evidence_ids must be a list of strings" in out["error"]
    assert ex.state.root_cause is None


class _DocUrnSession(FakeDataHubSession):
    async def call_tool(self, name, arguments=None):
        if name == "save_document":
            from retrace.datahub.fake import _ok

            self.calls.append((name, dict(arguments or {})))
            return _ok({"success": True, "urn": "urn:li:document:retrace-1"})
        return await super().call_tool(name, arguments)


def test_write_back_records_document_urn(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(_DocUrnSession())
    state = IncidentState(scenario="unit_cents", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    try:
        _confirm(ex)
        sql = get_fault("unit_cents").reference_patch["stg_orders"]
        ex.dispatch("propose_repair", {"file": "stg_orders.sql", "new_sql": sql})
        out = ex.dispatch("write_back", {"summary": "fixed"})
    finally:
        conn.close()
    assert out == {"written_back": True, "stage": "WRITTEN_BACK"}
    assert state.writeback_urns == ["urn:li:document:retrace-1"]
    assert state.to_dict()["writeback_urns"] == ["urn:li:document:retrace-1"]


def test_writeback_urns_default_empty():
    state = IncidentState(scenario="s", report="r")
    assert state.writeback_urns == [] and state.to_dict()["writeback_urns"] == []


class _PatternSession(FakeDataHubSession):
    """Fails call_tool per a queue of booleans (True = raise), falling back to
    the real fake behaviour once the queue is exhausted."""

    def __init__(self, pattern: list[bool]) -> None:
        super().__init__()
        self._pattern = list(pattern)

    async def call_tool(self, name, arguments=None):
        fail_now = self._pattern.pop(0) if self._pattern else False
        if fail_now:
            raise TimeoutError("DataHub down")
        return await super().call_tool(name, arguments)


def _executor_with_pattern(tmp_path, baseline, pattern):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    conn = DataHubConnection.from_session(_PatternSession(pattern))
    state = IncidentState(scenario="unit_cents", report="r")
    ex = ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, tmp_path / "scratch"),
        DataHubTools(conn),
    )
    return state, ex, conn


def test_three_consecutive_datahub_errors_fail_the_run(tmp_path, baseline):
    state, ex, conn = _executor_with_pattern(tmp_path, baseline, [True, True, True])
    try:
        ex.dispatch("datahub_search", {"query": "revenue"})
        ex.dispatch("datahub_search", {"query": "revenue"})
        out = ex.dispatch("datahub_search", {"query": "revenue"})
    finally:
        conn.close()
    assert out == {"error": "DataHub unavailable: 3 consecutive errors"}
    assert ex.finished is True
    assert state.stage is Stage.FAILED
    assert state.failure_reason == "DataHub unavailable: 3 consecutive errors"


def test_datahub_error_streak_resets_on_success(tmp_path, baseline):
    state, ex, conn = _executor_with_pattern(tmp_path, baseline, [True, True, False, True, True])
    try:
        outs = [ex.dispatch("datahub_search", {"query": "revenue"}) for _ in range(5)]
    finally:
        conn.close()
    assert "error" in outs[0] and "error" in outs[1]
    assert "error" not in outs[2]  # the interleaved success resets the streak
    assert "error" in outs[3] and "error" in outs[4]
    # Never 3 in a row, so the outage stop must never trip.
    assert ex.finished is False
    assert state.stage is not Stage.FAILED
