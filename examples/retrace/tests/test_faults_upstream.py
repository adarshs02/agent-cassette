import pytest
from retrace.faults import fault_names, get_fault
from retrace.pipeline.checks import failed_names, run_checks
from retrace.pipeline.workspace import prepare

ESCALATE = ["stale_feed", "null_surge"]
CONTROLS = ["control_healthy", "control_distractor"]


def test_all_eight_registered_in_order():
    assert fault_names() == [
        "unit_cents",
        "schema_rename",
        "join_fanout",
        "tz_shift",
        "stale_feed",
        "null_surge",
        "control_healthy",
        "control_distractor",
    ]


@pytest.mark.parametrize("name", ESCALATE)
def test_escalate_fault_breaks_raw_level_checks(name, tmp_path, baseline):
    fault = get_fault(name)
    assert fault.ground_truth.outcome == "escalate"
    assert fault.reference_patch is None
    ws = prepare(tmp_path / name, fault=name)
    failed = failed_names(run_checks(ws.warehouse, baseline))
    assert set(fault.must_fail) <= set(failed), failed


@pytest.mark.parametrize("name", CONTROLS)
def test_controls_pass_every_check(name, tmp_path, baseline):
    fault = get_fault(name)
    assert fault.ground_truth.outcome == "no_incident"
    ws = prepare(tmp_path / name, fault=name)
    assert failed_names(run_checks(ws.warehouse, baseline)) == []


def test_extending_fx_fill_does_not_fix_stale_feed(tmp_path, baseline):
    from retrace.faults import patched_transform
    from retrace.pipeline.build import build

    ws = prepare(tmp_path / "stale", fault="stale_feed")
    patch = patched_transform("stg_fx_rates", "s.day - r.rate_day <= 2", "s.day - r.rate_day <= 30")
    for name, sql in patch.items():
        (ws.transforms / f"{name}.sql").write_text(sql)
    build(ws.sources, ws.transforms, ws.warehouse)
    assert "fx_feed_fresh" in failed_names(run_checks(ws.warehouse, baseline))


@pytest.mark.parametrize("name", ESCALATE)
def test_escalate_reports_do_not_leak(name):
    report = get_fault(name).report.lower()
    for token in ("fx", "feed", "null", "rate", "missing", "cloudpay"):
        assert token not in report
