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
