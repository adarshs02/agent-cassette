import hashlib

import pytest
from retrace.faults import get_fault
from retrace.pipeline.workspace import prepare
from retrace.tools.repair import Repairer, RepairRejected, Transforms, allowed_repair_targets


def _digest(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.sql")}


def test_allowed_targets():
    assert allowed_repair_targets("raw.raw_customers") == {
        "stg_customers",
        "fct_revenue",
        "exec_metric",
    }
    assert "stg_customers" not in allowed_repair_targets("raw.raw_orders")


def test_reference_fix_passes_and_originals_untouched(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    before = _digest(ws.transforms)
    repairer = Repairer(ws, baseline, tmp_path / "scratch")
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    outcome = repairer.propose("stg_orders.sql", sql, "raw.raw_orders", accepted={})
    assert outcome.passed, outcome.failed_checks
    assert "cloudpay_v2" in outcome.diff and outcome.diff.startswith("--- a/stg_orders.sql")
    assert _digest(ws.transforms) == before
    assert repairer.attempts == 1


def test_failing_fix_reports_checks(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    repairer = Repairer(ws, baseline, tmp_path / "scratch")
    sql = (
        Transforms(ws)
        .read("stg_orders")
        .replace("CAST(amount AS DOUBLE) AS amount", "CAST(amount AS DOUBLE) / 100.0 AS amount")
    )
    outcome = repairer.propose("stg_orders", sql, "raw.raw_orders", accepted={})
    assert not outcome.passed
    assert "historical_revenue_immutable" in outcome.failed_checks


def test_build_error_is_reported(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    outcome = Repairer(ws, baseline, tmp_path / "s").propose(
        "stg_orders.sql", "SELECT FROM WHERE", "raw.raw_orders", accepted={}
    )
    assert not outcome.passed and "stg_orders.sql failed" in outcome.error
    assert str(tmp_path) not in outcome.error


@pytest.mark.parametrize(
    "file", ["../../x.sql", "/etc/passwd", "transforms/stg_orders.sql", "nope.sql", ""]
)
def test_rejects_path_traversal_and_unknown_files(tmp_path, baseline, file):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    repairer = Repairer(ws, baseline, tmp_path / "s")
    with pytest.raises(RepairRejected):
        repairer.propose(file, "SELECT 1", "raw.raw_orders", accepted={})
    assert repairer.attempts == 0


def test_rejects_target_not_downstream_of_root_cause(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    with pytest.raises(RepairRejected, match="downstream"):
        Repairer(ws, baseline, tmp_path / "s").propose(
            "stg_customers.sql", "SELECT 1", "raw.raw_orders", accepted={}
        )
