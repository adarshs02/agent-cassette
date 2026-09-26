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


def test_second_confirm_root_cause_errors_and_claim_unchanged(executor):
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
    assert ex.state.root_cause == original


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
