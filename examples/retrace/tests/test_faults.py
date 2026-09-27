import pytest
from retrace.faults import (
    evaluate_rules,
    fault_names,
    get_fault,
    mentions_all,
    mentions_any,
    strip_sql_comments,
)
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


def test_strip_sql_comments_removes_line_and_block_comments():
    sql = (
        "SELECT amount -- cloudpay_v2 divide by 100\n"
        "/* block\ncomment mentions cloudpay_v2 too */\n"
        "FROM raw.raw_orders;"
    )
    stripped = strip_sql_comments(sql)
    assert "cloudpay_v2" not in stripped
    assert "block" not in stripped and "comment" not in stripped
    assert "FROM raw.raw_orders" in stripped


def test_strip_sql_comments_does_not_mangle_string_literals_containing_dashes():
    sql = "SELECT 'a--b' AS lit -- a real comment\nFROM raw.raw_orders;"
    stripped = strip_sql_comments(sql)
    assert "'a--b'" in stripped
    assert "a real comment" not in stripped


def test_strip_sql_comments_falls_back_when_unparseable():
    # An unterminated block comment can't be tokenized at all; the regex
    # fallback must still drop the preceding line comment without raising.
    broken = "SELECT 1 -- cloudpay_v2\n/* unterminated"
    stripped = strip_sql_comments(broken)
    assert "cloudpay_v2" not in stripped


def test_mentions_all_ignores_a_needle_that_only_appears_in_a_comment():
    rule = mentions_all("cloudpay_v2", "100")
    patch = {
        "stg_orders": (
            "CREATE OR REPLACE TABLE staging.stg_orders AS\n"
            "SELECT\n"
            "    -- fixes cloudpay_v2 cents\n"
            "    CASE WHEN amount NOT LIKE '%.%' THEN CAST(amount AS DOUBLE) / 100.0\n"
            "    ELSE CAST(amount AS DOUBLE) END AS amount\n"
            "FROM raw.raw_orders;"
        )
    }
    result = rule(patch)
    assert not result.passed
    assert "cloudpay_v2" in result.detail


def test_mentions_any_still_matches_a_needle_inside_a_string_literal():
    rule = mentions_any("a--b")
    result = rule({"stg_orders": "SELECT 'a--b' AS lit -- unrelated comment"})
    assert result.passed


def test_unit_cents_rule_rejects_heuristic_patch_that_only_names_processor_in_a_comment():
    # Reproduces the real-Claude-round bug: a format heuristic (not scoped to
    # cloudpay_v2) that only namedrops the processor in a SQL comment used to
    # pass mentions_all("cloudpay_v2", "100") because comments weren't stripped.
    fault = get_fault("unit_cents")
    heuristic_patch = {
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
            "    -- cloudpay_v2 amounts arrive as integer cents; divide by 100\n"
            "    CASE WHEN amount NOT LIKE '%.%' THEN CAST(amount AS DOUBLE) / 100.0\n"
            "    ELSE CAST(amount AS DOUBLE) END AS amount,\n"
            "    payment_processor,\n"
            "    status\n"
            "FROM raw.raw_orders\n"
            "WHERE status = 'completed';"
        )
    }
    assert not all(r.passed for r in evaluate_rules(fault, heuristic_patch))
