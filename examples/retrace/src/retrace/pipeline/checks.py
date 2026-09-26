# Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub),
# Apache-2.0. Modified for Retrace.
"""Pipeline invariants. Healthy data passes all; each fault fails at least one."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb

from retrace.pipeline.generate import ANCHOR_DAY, CURRENCIES, CUTOVER_DAY

BAND = 0.02
RECENT_DAYS = 14
MAX_SANE_AMOUNT = 5000.0
AOV_MEDIAN_RANGE = (15.0, 150.0)
KPI_RATIO_RANGE = (0.8, 1.25)
DIURNAL_MIN_SHARE = 0.65
FX_MAX_STALENESS_DAYS = 2


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Check = Callable[[duckdb.DuckDBPyConnection, dict], tuple[bool, str]]
_CHECKS: list[tuple[str, Check]] = []


def check(name: str) -> Callable[[Check], Check]:
    def register(fn: Check) -> Check:
        _CHECKS.append((name, fn))
        return fn

    return register


def check_names() -> list[str]:
    return [name for name, _ in _CHECKS]


def failed_names(results: list[CheckResult]) -> list[str]:
    return [r.name for r in results if not r.passed]


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    return None if row is None else row[0]


def _count_is_zero(con: duckdb.DuckDBPyConnection, sql: str, label: str) -> tuple[bool, str]:
    count = _scalar(con, sql)
    return count == 0, f"{count} {label}"


@check("stg_order_id_unique")
def _order_id_unique(con, baseline):
    return _count_is_zero(
        con,
        "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM staging.stg_orders",
        "duplicate order_id values",
    )


@check("stg_no_null_amount")
def _no_null_amount(con, baseline):
    return _count_is_zero(
        con, "SELECT COUNT(*) FROM staging.stg_orders WHERE amount IS NULL", "null amounts"
    )


@check("stg_no_null_currency")
def _no_null_currency(con, baseline):
    return _count_is_zero(
        con, "SELECT COUNT(*) FROM staging.stg_orders WHERE currency IS NULL", "null currencies"
    )


@check("stg_no_null_order_ts")
def _no_null_ts(con, baseline):
    return _count_is_zero(
        con, "SELECT COUNT(*) FROM staging.stg_orders WHERE order_ts IS NULL", "null order_ts"
    )


@check("stg_currency_known")
def _currency_known(con, baseline):
    known = ", ".join(f"'{c}'" for c in CURRENCIES)
    return _count_is_zero(
        con,
        f"SELECT COUNT(*) FROM staging.stg_orders "
        f"WHERE currency IS NOT NULL AND currency NOT IN ({known})",
        "unknown currency codes",
    )


@check("stg_amount_positive")
def _amount_positive(con, baseline):
    return _count_is_zero(
        con,
        "SELECT COUNT(*) FROM staging.stg_orders WHERE amount IS NOT NULL AND amount <= 0",
        "non-positive amounts",
    )


@check("stg_max_amount_sane")
def _max_amount(con, baseline):
    top = _scalar(con, "SELECT MAX(amount) FROM staging.stg_orders")
    return top is not None and top < MAX_SANE_AMOUNT, f"max amount {top}"


@check("fct_order_count_matches_stg")
def _order_count_matches(con, baseline):
    fct = _scalar(con, "SELECT SUM(order_count) FROM marts.fct_revenue")
    stg = _scalar(con, "SELECT COUNT(*) FROM staging.stg_orders")
    return fct == stg, f"fct={fct} stg={stg}"


@check("fct_one_row_per_day")
def _one_row_per_day(con, baseline):
    rows, days = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT day) FROM marts.fct_revenue"
    ).fetchone()
    return rows == days == 90, f"rows={rows} distinct_days={days}"


def _recent(con: duckdb.DuckDBPyConnection) -> list[tuple[str, float | None, int]]:
    return con.execute(
        "SELECT CAST(day AS VARCHAR), revenue_usd, order_count FROM marts.fct_revenue "
        f"ORDER BY day DESC LIMIT {RECENT_DAYS}"
    ).fetchall()


def _band(con, baseline, column: int, key: str) -> tuple[bool, str]:
    bad = []
    for row in _recent(con):
        day, value = row[0], row[column]
        base = baseline["days"].get(day)
        if base is None or value is None or not base[key]:
            bad.append(f"{day}:missing")
            continue
        ratio = value / base[key]
        if abs(ratio - 1) > BAND:
            bad.append(f"{day}:{ratio:.3f}")
    return not bad, "ok" if not bad else "out of band: " + ", ".join(sorted(bad)[:6])


@check("revenue_within_baseline_band")
def _revenue_band(con, baseline):
    return _band(con, baseline, 1, "revenue_usd")


@check("order_count_within_baseline_band")
def _count_band(con, baseline):
    return _band(con, baseline, 2, "order_count")


@check("historical_revenue_immutable")
def _historical(con, baseline):
    rows = con.execute(
        "SELECT CAST(day AS VARCHAR), revenue_usd, order_count FROM marts.fct_revenue "
        f"WHERE day < DATE '{CUTOVER_DAY.isoformat()}' ORDER BY day"
    ).fetchall()
    bad = []
    for day, revenue, count in rows:
        base = baseline["days"].get(day)
        if base is None or revenue is None:
            bad.append(f"{day}:missing")
        elif abs(revenue / base["revenue_usd"] - 1) > 1e-4 or count != base["order_count"]:
            bad.append(day)
    return not bad, "ok" if not bad else f"{len(bad)} historical days changed, e.g. {bad[:3]}"


@check("aov_median_in_range")
def _aov_median(con, baseline):
    low, high = AOV_MEDIAN_RANGE
    bad = con.execute(
        "SELECT CAST(day AS VARCHAR) FROM marts.fct_revenue "
        f"WHERE aov_median_usd IS NULL OR aov_median_usd NOT BETWEEN {low} AND {high} "
        "ORDER BY day"
    ).fetchall()
    return not bad, f"{len(bad)} days out of range"


@check("fx_feed_fresh")
def _fx_fresh(con, baseline):
    cutoff = (ANCHOR_DAY - timedelta(days=FX_MAX_STALENESS_DAYS)).isoformat()
    stale = con.execute(
        "SELECT currency, CAST(MAX(CAST(rate_day AS DATE)) AS VARCHAR) FROM raw.raw_fx_rates "
        f"GROUP BY currency HAVING MAX(CAST(rate_day AS DATE)) < DATE '{cutoff}' "
        "ORDER BY currency"
    ).fetchall()
    return not stale, "ok" if not stale else f"stale currencies: {stale}"


@check("fx_rates_positive")
def _fx_positive(con, baseline):
    return _count_is_zero(
        con,
        "SELECT COUNT(*) FROM raw.raw_fx_rates "
        "WHERE TRY_CAST(usd_rate AS DOUBLE) IS NULL OR TRY_CAST(usd_rate AS DOUBLE) <= 0",
        "invalid rates",
    )


@check("diurnal_profile")
def _diurnal(con, baseline):
    rows = con.execute(
        "SELECT payment_processor, AVG(CASE WHEN hour(order_ts) >= 15 OR hour(order_ts) <= 3 "
        "THEN 1.0 ELSE 0.0 END) FROM staging.stg_orders "
        f"WHERE order_day >= DATE '{CUTOVER_DAY.isoformat()}' GROUP BY 1 ORDER BY 1"
    ).fetchall()
    low = [f"{p}:{s:.2f}" for p, s in rows if s < DIURNAL_MIN_SHARE]
    return not low, "ok" if not low else f"peak-hour share too low: {low}"


@check("metric_matches_fct_latest")
def _metric_matches(con, baseline):
    row = con.execute(
        "SELECT m.revenue, f.revenue_usd FROM marts.exec_metric AS m "
        "JOIN marts.fct_revenue AS f ON f.day = m.kpi_day"
    ).fetchone()
    ok = row is not None and None not in row and abs(row[0] - row[1]) < 1e-6
    return ok, f"metric/fct = {row}"


@check("kpi_anomaly_ratio_sane")
def _kpi_ratio(con, baseline):
    ratio = _scalar(con, "SELECT anomaly_ratio FROM marts.exec_metric")
    low, high = KPI_RATIO_RANGE
    return ratio is not None and low <= ratio <= high, f"anomaly_ratio={ratio}"


@check("raw_no_null_amount")
def _raw_null_amount(con, baseline):
    return _count_is_zero(
        con, "SELECT COUNT(*) FROM raw.raw_orders WHERE amount IS NULL", "null raw amounts"
    )


def run_checks(warehouse: Path, baseline: dict) -> list[CheckResult]:
    con = duckdb.connect(str(warehouse), read_only=True)
    results: list[CheckResult] = []
    try:
        con.execute("SET threads TO 1")
        for name, fn in _CHECKS:
            try:
                passed, detail = fn(con, baseline)
            except duckdb.Error as error:
                passed, detail = False, f"check errored: {error}"
            results.append(CheckResult(name, bool(passed), detail))
    finally:
        con.close()
    return results
