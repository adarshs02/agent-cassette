"""Grader-level tests: the format-robustness variant grade for sql_repair faults."""

import csv

from retrace.evals.graders import _variant_verify
from retrace.faults import get_fault
from retrace.pipeline.checks import failed_names, run_checks
from retrace.pipeline.generate import generate
from retrace.pipeline.workspace import prepare

_HEURISTIC_PATCH = {
    "stg_orders": (
        "-- Adapted from Project Blackbox "
        "(https://github.com/alejandro-publius/blackbox-datahub), "
        "Apache-2.0. Modified for Retrace.\n"
        "CREATE OR REPLACE TABLE staging.stg_orders AS\n"
        "SELECT\n"
        "    order_id,\n"
        "    CAST(order_ts AS TIMESTAMP) AS order_ts,\n"
        "    CAST(CAST(order_ts AS TIMESTAMP) AS DATE) AS order_day,\n"
        "    customer_id,\n"
        "    currency,\n"
        "    CASE WHEN amount NOT LIKE '%.%' THEN CAST(amount AS DOUBLE) / 100.0\n"
        "    ELSE CAST(amount AS DOUBLE) END AS amount,\n"
        "    payment_processor,\n"
        "    status\n"
        "FROM raw.raw_orders\n"
        "WHERE status = 'completed';"
    )
}


def _rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_variant_actually_rewrites_some_legacy_pos_rows(tmp_path):
    plain = tmp_path / "plain"
    varied = tmp_path / "varied"
    generate(plain, fault="unit_cents", variant=False)
    generate(varied, fault="unit_cents", variant=True)
    before = _rows(plain / "raw_orders_part1.csv") + _rows(plain / "raw_orders_part2.csv")
    after = _rows(varied / "raw_orders_part1.csv") + _rows(varied / "raw_orders_part2.csv")
    changed = sum(
        1
        for b, a in zip(before, after)
        if b["payment_processor"] == "legacy_pos" and b["amount"] != a["amount"]
    )
    assert changed > 0, "variant must be non-vacuous: it should reformat some legacy_pos rows"
    # The value itself is unchanged, only its string format.
    for b, a in zip(before, after):
        if b["amount"] != a["amount"]:
            assert float(b["amount"]) == float(a["amount"])


def test_healthy_control_with_variant_still_passes_all_checks_with_original_transforms(
    tmp_path, baseline
):
    # control_healthy has no .variant of its own, so requesting variant=True is a
    # safe no-op: the build must still be the ordinary healthy build.
    ws = prepare(tmp_path / "ws", fault="control_healthy", variant=True)
    assert failed_names(run_checks(ws.warehouse, baseline)) == []


def test_reference_patch_passes_the_variant_grade(tmp_path, baseline):
    fault = get_fault("unit_cents")
    grade = _variant_verify(fault, fault.reference_patch, baseline, tmp_path / "scratch")
    assert grade.passed, grade.detail


def test_format_heuristic_patch_fails_the_variant_grade_on_historical_immutability(
    tmp_path, baseline
):
    fault = get_fault("unit_cents")
    grade = _variant_verify(fault, _HEURISTIC_PATCH, baseline, tmp_path / "scratch")
    assert not grade.passed
    assert "historical_revenue_immutable" in grade.detail
