"""Grader-level tests: the format-robustness variant grade (Fix 1b) and
alternate acceptable root-cause fields (Fix 2)."""

import csv

from retrace.agent.state import Claim, IncidentState, Stage
from retrace.evals.graders import _claim_grades, _variant_verify
from retrace.faults import get_fault
from retrace.pipeline.build import build
from retrace.pipeline.checks import failed_names, run_checks
from retrace.pipeline.generate import generate, generate_frames, write_sources
from retrace.pipeline.workspace import TRANSFORMS_DIR

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
        for b, a in zip(before, after, strict=True)
        if b["payment_processor"] == "legacy_pos" and b["amount"] != a["amount"]
    )
    assert changed > 0, "variant must be non-vacuous: it should reformat some legacy_pos rows"
    # The value itself is unchanged, only its string format.
    for b, a in zip(before, after, strict=True):
        if b["amount"] != a["amount"]:
            assert float(b["amount"]) == float(a["amount"])


def test_variant_alone_still_passes_every_check_against_original_transforms(tmp_path, baseline):
    # Apply unit_cents' variant with no fault injected at all: just the legacy_pos
    # ".00" -> "40" reformat, run through the ORIGINAL (unpatched) transforms. Since
    # the reformat doesn't change the value, every check should still be green.
    frames = generate_frames()
    before = [row["amount"] for row in frames.orders if row["payment_processor"] == "legacy_pos"]
    get_fault("unit_cents").variant(frames)
    after = [row["amount"] for row in frames.orders if row["payment_processor"] == "legacy_pos"]
    changed = sum(1 for b, a in zip(before, after, strict=True) if b != a)
    assert changed > 0, "variant must be non-vacuous: it should reformat some legacy_pos rows"

    sources = tmp_path / "sources"
    write_sources(frames, sources)
    warehouse = tmp_path / "warehouse.duckdb"
    build(sources, TRANSFORMS_DIR, warehouse)
    assert failed_names(run_checks(warehouse, baseline)) == []


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


def _stale_feed_field_grade(field: str):
    fault = get_fault("stale_feed")
    state = IncidentState(scenario="stale_feed", report="r", stage=Stage.ESCALATED)
    state.escalation = Claim(
        asset="raw.raw_fx_rates", field=field, summary="s", evidence_ids=["ev_001"]
    )
    grades = {g.name: g for g in _claim_grades(fault, state)}
    return grades["root_cause_field"]


def test_stale_feed_accepts_the_ground_truth_field():
    assert _stale_feed_field_grade("rate_day").passed


def test_stale_feed_accepts_the_alt_field():
    assert _stale_feed_field_grade("usd_rate").passed


def test_stale_feed_rejects_an_unrelated_field():
    grade = _stale_feed_field_grade("currency")
    assert not grade.passed
    assert "rate_day" in grade.detail
    assert "usd_rate" in grade.detail
