# Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub),
# Apache-2.0. Modified for Retrace.
"""Static metadata describing the pipeline: docs, contracts, owners, lineage.

Nothing here references any fault; faults must be discovered from data.
"""

from __future__ import annotations

from retrace.pipeline.lineage import OUTPUT_TABLES, direct_dependencies

PLATFORM = "duckdb"
ENV = "PROD"

TABLES = [
    "raw.raw_orders",
    "raw.raw_customers",
    "raw.raw_fx_rates",
    "staging.stg_orders",
    "staging.stg_customers",
    "staging.stg_fx_rates",
    "marts.fct_revenue",
    "marts.exec_metric",
]

COLUMNS: dict[str, list[str]] = {
    "raw.raw_orders": [
        "order_id",
        "order_ts",
        "customer_id",
        "currency",
        "amount",
        "payment_processor",
        "status",
    ],
    "raw.raw_customers": ["customer_id", "segment", "country", "created_at"],
    "raw.raw_fx_rates": ["rate_day", "currency", "usd_rate"],
    "staging.stg_orders": [
        "order_id",
        "order_ts",
        "order_day",
        "customer_id",
        "currency",
        "amount",
        "payment_processor",
        "status",
    ],
    "staging.stg_customers": ["customer_id", "segment", "country", "created_at"],
    "staging.stg_fx_rates": ["rate_day", "currency", "usd_rate", "source_rate_day"],
    "marts.fct_revenue": [
        "day",
        "order_count",
        "revenue_usd",
        "aov_usd",
        "aov_median_usd",
        "enterprise_revenue_usd",
    ],
    "marts.exec_metric": ["kpi_day", "revenue", "trailing_28d_median_revenue", "anomaly_ratio"],
}

DATASET_DOCS: dict[str, str] = {
    "raw.raw_orders": (
        "Order events extracted nightly from the payments platform. Contract "
        "(payments-platform v1.3): one row per order attempt; amount is a decimal in major "
        "currency units; order_ts is UTC."
    ),
    "raw.raw_customers": "Customer master extract from CRM. Contract: one row per customer.",
    "raw.raw_fx_rates": (
        "Daily FX reference rates versus USD. EUR/GBP/CAD come from an external vendor feed "
        "that is expected to update every day; USD is fixed at 1.0."
    ),
    "staging.stg_orders": "Completed orders only, typed. Canonical input for revenue reporting.",
    "staging.stg_customers": "Typed customer dimension.",
    "staging.stg_fx_rates": (
        "FX rates on a daily spine; a missing day is forward-filled from the last known rate "
        "for at most 2 days, after which usd_rate is NULL."
    ),
    "marts.fct_revenue": "Daily revenue fact, USD-normalized. Grain: one row per day.",
    "marts.exec_metric": (
        "Executive Revenue KPI shown on the executive dashboard each morning. anomaly_ratio "
        "compares the latest day with the trailing median (kpi_day-35 .. kpi_day-8)."
    ),
}

FIELD_DOCS: dict[str, dict[str, str]] = {
    "raw.raw_orders": {
        "order_id": "Unique order identifier.",
        "order_ts": "Order capture timestamp in UTC, format YYYY-MM-DD HH:MM:SS.",
        "customer_id": "Foreign key to raw_customers.customer_id.",
        "currency": "ISO-4217 currency code (USD/EUR/GBP/CAD).",
        "amount": (
            "Order amount in MAJOR currency units as a decimal with two places (49.99 means 49.99)."
        ),
        "payment_processor": (
            "Provider that captured the order: legacy_pos, shopgate, "
            "cloudpay_v2 (rolled out 2026-08-07)."
        ),
        "status": "Terminal order state: completed | cancelled | refunded.",
    },
    "raw.raw_customers": {
        "customer_id": "Primary key. Exactly one row per customer.",
        "segment": "consumer | smb | enterprise.",
    },
    "raw.raw_fx_rates": {
        "rate_day": "Date the rate is effective.",
        "currency": "ISO-4217 code.",
        "usd_rate": "USD per one unit of currency.",
    },
    "marts.fct_revenue": {
        "revenue_usd": "Sum of USD-normalized completed order amounts. Feeds the executive KPI.",
        "order_count": "Completed orders that day.",
    },
    "marts.exec_metric": {
        "revenue": "Executive revenue for kpi_day (USD).",
        "anomaly_ratio": "revenue / trailing_28d_median_revenue. Healthy is about 1.0.",
    },
}

OWNERS: dict[str, tuple[str, str, list[str]]] = {
    "jordan.lee": (
        "Jordan Lee",
        "Staff Engineer, Payments Platform",
        ["raw.raw_orders", "raw.raw_customers", "raw.raw_fx_rates"],
    ),
    "priya.desai": (
        "Priya Desai",
        "Analytics Engineering Lead",
        [
            "staging.stg_orders",
            "staging.stg_customers",
            "staging.stg_fx_rates",
            "marts.fct_revenue",
            "marts.exec_metric",
        ],
    ),
}

TAGS: dict[str, list[str]] = {
    "marts.exec_metric": ["kpi", "executive-reporting"],
    "marts.fct_revenue": ["revenue"],
}


def dataset_urn(table: str) -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},{table},{ENV})"


def lineage_edges() -> list[tuple[str, str]]:
    edges = []
    for name, deps in direct_dependencies().items():
        for upstream in sorted(deps):
            edges.append((upstream, OUTPUT_TABLES[name]))
    return edges
