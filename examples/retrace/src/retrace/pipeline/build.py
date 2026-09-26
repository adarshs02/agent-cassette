"""Load raw CSVs into DuckDB and run the SQL transforms in order."""

from __future__ import annotations

from pathlib import Path

import duckdb

TRANSFORM_ORDER: tuple[str, ...] = (
    "stg_orders",
    "stg_customers",
    "stg_fx_rates",
    "fct_revenue",
    "exec_metric",
)


class BuildError(RuntimeError):
    """A transform failed to execute."""


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def build(sources: Path, transforms: Path, warehouse: Path) -> None:
    if warehouse.exists():
        warehouse.unlink()
    con = duckdb.connect(str(warehouse))
    try:
        con.execute("SET threads TO 1")
        for schema in ("raw", "staging", "marts"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.execute(
            "CREATE OR REPLACE TABLE raw.raw_orders AS SELECT * FROM read_csv("
            f"'{_quote(sources)}/raw_orders_*.csv', union_by_name = true, "
            "all_varchar = true, header = true)"
        )
        for table in ("raw_customers", "raw_fx_rates"):
            con.execute(
                f"CREATE OR REPLACE TABLE raw.{table} AS SELECT * FROM read_csv("
                f"'{_quote(sources / (table + '.csv'))}', all_varchar = true, header = true)"
            )
        for name in TRANSFORM_ORDER:
            sql = (transforms / f"{name}.sql").read_text()
            try:
                con.execute(sql)
            except duckdb.Error as error:
                raise BuildError(f"{name}.sql failed: {error}") from error
    finally:
        con.close()
