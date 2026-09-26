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
        "stg_orders.sql",
        "CREATE OR REPLACE TABLE staging.stg_orders AS SELECT * FROM raw.nope",
        "raw.raw_orders",
        accepted={},
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


def test_rejects_bad_accepted_keys_before_writing(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    repairer = Repairer(ws, baseline, tmp_path / "scratch")
    with pytest.raises(RepairRejected):
        repairer.propose(
            "stg_orders.sql",
            "SELECT 1",
            "raw.raw_orders",
            accepted={"../../evil": "SELECT 1"},
        )
    assert repairer.attempts == 0
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "evil.sql").exists()
    assert not any(tmp_path.rglob("evil*"))


def test_valid_accepted_target_is_allowed_and_covered_in_diff(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    repairer = Repairer(ws, baseline, tmp_path / "scratch")
    sql = get_fault("unit_cents").reference_patch["stg_orders"]
    first = repairer.propose("stg_orders.sql", sql, "raw.raw_orders", accepted={})
    assert first.passed, first.failed_checks

    second = repairer.propose("stg_orders.sql", sql, "raw.raw_orders", accepted={"stg_orders": sql})
    assert second.passed, second.failed_checks
    assert "a/stg_orders.sql" in second.diff
    assert repairer.attempts == 2


def _raw_amounts(ws):
    return sorted(p.read_bytes() for p in ws.sources.glob("raw_orders_*.csv"))


def test_multi_statement_exfil_patch_is_rejected(tmp_path, baseline):
    import duckdb

    ws = prepare(tmp_path / "ws", fault="unit_cents")
    before = _raw_amounts(ws)
    exfil = tmp_path / "exfil.csv"
    sql = f"COPY (SELECT 'x') TO '{exfil}'; UPDATE raw.raw_orders SET amount='1'"
    outcome = Repairer(ws, baseline, tmp_path / "scratch").propose(
        "stg_orders.sql", sql, "raw.raw_orders", accepted={}
    )
    assert not outcome.passed
    assert outcome.error == (
        "stg_orders.sql must be a single CREATE [OR REPLACE] TABLE staging.stg_orders"
    )
    assert not exfil.exists()
    assert _raw_amounts(ws) == before
    con = duckdb.connect(str(tmp_path / "scratch" / "attempt_1" / "warehouse.duckdb"), True)
    try:
        ones = con.execute("SELECT COUNT(*) FROM raw.raw_orders WHERE amount = '1'").fetchone()
        total = con.execute("SELECT COUNT(*) FROM raw.raw_orders").fetchone()
    finally:
        con.close()
    assert total[0] > 0 and ones[0] < total[0]


@pytest.mark.parametrize(
    "sql",
    [
        "INSTALL httpfs",
        "ATTACH 'x.duckdb' AS other",
        "CREATE OR REPLACE TABLE staging.stg_orders AS SELECT 1; INSTALL httpfs",
        "CREATE OR REPLACE TABLE raw.raw_orders AS SELECT 1 AS amount",
        "CREATE OR REPLACE VIEW staging.stg_orders AS SELECT 1",
        "UPDATE raw.raw_orders SET amount = '1'",
    ],
)
def test_non_create_table_patches_are_rejected(tmp_path, baseline, sql):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    outcome = Repairer(ws, baseline, tmp_path / "scratch").propose(
        "stg_orders.sql", sql, "raw.raw_orders", accepted={}
    )
    assert not outcome.passed
    assert "must be a single CREATE [OR REPLACE] TABLE staging.stg_orders" in outcome.error


def test_build_sandbox_blocks_external_file_reads(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    sql = f"CREATE OR REPLACE TABLE staging.stg_orders AS SELECT * FROM read_csv('{secret}')"
    outcome = Repairer(ws, baseline, tmp_path / "scratch").propose(
        "stg_orders.sql", sql, "raw.raw_orders", accepted={}
    )
    assert not outcome.passed
    assert outcome.error.startswith("stg_orders.sql failed")


def test_unparseable_patch_is_rejected_with_detail(tmp_path, baseline):
    ws = prepare(tmp_path / "ws", fault="unit_cents")
    outcome = Repairer(ws, baseline, tmp_path / "s").propose(
        "stg_orders.sql", "SELECT FROM WHERE (", "raw.raw_orders", accepted={}
    )
    assert not outcome.passed
    assert outcome.error.startswith("stg_orders.sql must be a single CREATE")
