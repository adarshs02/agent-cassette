"""Faults that a targeted SQL change can correct."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from retrace.faults import (
    Fault,
    GroundTruth,
    files_within,
    mentions_all,
    mentions_any,
    patched_transform,
    register,
)
from retrace.pipeline.generate import CUTOVER_DAY, SEED, TS_FORMAT, Frames

NEW_PROCESSOR = "cloudpay_v2"


def _inject_unit_cents(frames: Frames) -> None:
    for row in frames.orders:
        if row["payment_processor"] == NEW_PROCESSOR and row["amount"]:
            row["amount"] = str(int(round(float(row["amount"]) * 100)))


register(
    Fault(
        name="unit_cents",
        report="The exec revenue KPI on this morning's dashboard looks way off. Can you check?",
        inject=_inject_unit_cents,
        ground_truth=GroundTruth("sql_repair", "raw.raw_orders", "amount"),
        must_fail=("stg_max_amount_sane", "revenue_within_baseline_band"),
        reference_patch=patched_transform(
            "stg_orders",
            "    CAST(amount AS DOUBLE) AS amount,\n",
            "    CASE\n"
            f"        WHEN payment_processor = '{NEW_PROCESSOR}' THEN "
            "CAST(amount AS DOUBLE) / 100.0\n"
            "        ELSE CAST(amount AS DOUBLE)\n"
            "    END AS amount,\n",
        ),
        repair_rules=(
            files_within("stg_orders", "fct_revenue"),
            mentions_all(NEW_PROCESSOR, "100"),
        ),
    )
)


def _inject_schema_rename(frames: Frames) -> None:
    cutover = CUTOVER_DAY.isoformat()
    for row in frames.orders:
        if row["order_ts"][:10] >= cutover:
            row["currency_code"] = row.pop("currency")


register(
    Fault(
        name="schema_rename",
        report="Exec revenue dropped sharply over the last few days and nobody launched "
        "anything. Please investigate.",
        inject=_inject_schema_rename,
        ground_truth=GroundTruth("sql_repair", "raw.raw_orders", "currency"),
        must_fail=("stg_no_null_currency", "revenue_within_baseline_band"),
        reference_patch=patched_transform(
            "stg_orders", "    currency,\n", "    COALESCE(currency, currency_code) AS currency,\n"
        ),
        repair_rules=(files_within("stg_orders"), mentions_all("currency_code")),
    )
)


def _inject_join_fanout(frames: Frames) -> None:
    rng = random.Random(SEED + 3)
    extras = []
    for row in frames.customers:
        if rng.random() < 0.15:
            dup = dict(row)
            dup["segment"] = rng.choice(("consumer", "smb", "enterprise"))
            dup["created_at"] = "2026-08-01"
            extras.append(dup)
    frames.customers.extend(extras)


register(
    Fault(
        name="join_fanout",
        report="Finance says our revenue numbers are higher than what they reconcile to, "
        "including for older months. Can you look?",
        inject=_inject_join_fanout,
        ground_truth=GroundTruth("sql_repair", "raw.raw_customers", "customer_id"),
        must_fail=("fct_order_count_matches_stg", "historical_revenue_immutable"),
        reference_patch=patched_transform(
            "stg_customers",
            "FROM raw.raw_customers;",
            "FROM raw.raw_customers\n"
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1;",
        ),
        repair_rules=(
            files_within("stg_customers", "fct_revenue"),
            mentions_any("row_number", "distinct", "qualify", "any_value", "group by"),
        ),
    )
)


def _inject_tz_shift(frames: Frames) -> None:
    for row in frames.orders:
        if row["payment_processor"] == NEW_PROCESSOR:
            ts = datetime.strptime(row["order_ts"], TS_FORMAT) - timedelta(hours=7)
            row["order_ts"] = ts.strftime(TS_FORMAT)


_SHIFT = (
    f"CASE WHEN payment_processor = '{NEW_PROCESSOR}' THEN INTERVAL 7 HOUR ELSE INTERVAL 0 HOUR END"
)

register(
    Fault(
        name="tz_shift",
        report="Yesterday's exec revenue looks low and some earlier days look a bit high. "
        "Something off?",
        inject=_inject_tz_shift,
        ground_truth=GroundTruth("sql_repair", "raw.raw_orders", "order_ts"),
        must_fail=("diurnal_profile", "historical_revenue_immutable"),
        reference_patch=patched_transform(
            "stg_orders",
            "    CAST(order_ts AS TIMESTAMP) AS order_ts,\n"
            "    CAST(CAST(order_ts AS TIMESTAMP) AS DATE) AS order_day,\n",
            f"    CAST(order_ts AS TIMESTAMP) + {_SHIFT} AS order_ts,\n"
            f"    CAST(CAST(order_ts AS TIMESTAMP) + {_SHIFT} AS DATE) AS order_day,\n",
        ),
        repair_rules=(
            files_within("stg_orders"),
            mentions_all(NEW_PROCESSOR),
            mentions_any("interval", "hour", "timezone", "at time zone"),
        ),
    )
)
