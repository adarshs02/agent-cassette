import pytest
from retrace.faults import evaluate_rules, fault_names, get_fault
from retrace.pipeline.build import build
from retrace.pipeline.checks import check_names, failed_names, run_checks
from retrace.pipeline.workspace import prepare

REPAIRABLE = ["unit_cents", "schema_rename", "join_fanout", "tz_shift"]


def _apply(ws, patch):
    for name, sql in patch.items():
        (ws.transforms / f"{name}.sql").write_text(sql)
    build(ws.sources, ws.transforms, ws.warehouse)


def test_registry_lists_repairable_faults():
    assert set(REPAIRABLE) <= set(fault_names())
    with pytest.raises(KeyError, match="unknown fault"):
        get_fault("nope")


@pytest.mark.parametrize("name", REPAIRABLE)
def test_fault_breaks_expected_checks(name, tmp_path, baseline):
    fault = get_fault(name)
    assert set(fault.must_fail) <= set(check_names())
    ws = prepare(tmp_path / name, fault=name)
    failed = failed_names(run_checks(ws.warehouse, baseline))
    assert set(fault.must_fail) <= set(failed), failed


@pytest.mark.parametrize("name", REPAIRABLE)
def test_reference_patch_restores_every_check(name, tmp_path, baseline):
    fault = get_fault(name)
    ws = prepare(tmp_path / name, fault=name)
    _apply(ws, fault.reference_patch)
    results = run_checks(ws.warehouse, baseline)
    assert failed_names(results) == [], [r for r in results if not r.passed]
    assert all(r.passed for r in evaluate_rules(fault, fault.reference_patch))


def test_unit_cents_keeps_pre_cutover_rows_identical(tmp_path):
    healthy = prepare(tmp_path / "h", fault=None)
    faulty = prepare(tmp_path / "f", fault="unit_cents")
    part1 = "raw_orders_part1.csv"
    assert (healthy.sources / part1).read_bytes() == (faulty.sources / part1).read_bytes()


def test_blanket_divide_is_rejected(tmp_path, baseline):
    from retrace.faults import patched_transform

    fault = get_fault("unit_cents")
    blanket = patched_transform(
        "stg_orders", "CAST(amount AS DOUBLE) AS amount", "CAST(amount AS DOUBLE) / 100.0 AS amount"
    )
    ws = prepare(tmp_path / "blanket", fault="unit_cents")
    _apply(ws, blanket)
    assert "historical_revenue_immutable" in failed_names(run_checks(ws.warehouse, baseline))
    assert not all(r.passed for r in evaluate_rules(fault, blanket))


@pytest.mark.parametrize("name", REPAIRABLE)
def test_report_does_not_leak_ground_truth(name):
    fault = get_fault(name)
    report = fault.report.lower()
    for token in ("cents", "cloudpay", "currency_code", "duplicate", "timezone", "fx", "null"):
        assert token not in report
    assert fault.ground_truth.field not in report
