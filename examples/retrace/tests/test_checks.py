from retrace.pipeline.baseline import build_healthy_baseline, load_baseline
from retrace.pipeline.checks import check_names, failed_names, run_checks


def test_nineteen_checks_registered():
    assert len(check_names()) == 19
    assert check_names()[0] == "stg_order_id_unique"
    assert check_names()[-1] == "raw_no_null_amount"


def test_healthy_passes_every_check(healthy_ws, baseline):
    results = run_checks(healthy_ws.warehouse, baseline)
    assert failed_names(results) == [], [r for r in results if not r.passed]


def test_committed_baseline_matches_fixture():
    assert load_baseline() == build_healthy_baseline()


def test_broken_warehouse_reports_failures_not_exceptions(tmp_path, baseline):
    import duckdb

    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    results = run_checks(path, baseline)
    assert len(results) == 19
    assert all(not r.passed for r in results)
    assert all("check errored" in r.detail for r in results)
