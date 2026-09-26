"""Load raw CSVs into DuckDB and run the SQL transforms in order."""

from __future__ import annotations

from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

TRANSFORM_ORDER: tuple[str, ...] = (
    "stg_orders",
    "stg_customers",
    "stg_fx_rates",
    "fct_revenue",
    "exec_metric",
)

# The single table each transform may create. lineage.py re-exports this.
OUTPUT_TABLES: dict[str, str] = {
    "stg_orders": "staging.stg_orders",
    "stg_customers": "staging.stg_customers",
    "stg_fx_rates": "staging.stg_fx_rates",
    "fct_revenue": "marts.fct_revenue",
    "exec_metric": "marts.exec_metric",
}


class BuildError(RuntimeError):
    """A transform failed to execute."""


def _created_table(statement: exp.Expression) -> str | None:
    if not isinstance(statement, exp.Create):
        return None
    if str(statement.args.get("kind") or "").upper() != "TABLE":
        return None
    target = statement.this
    table = target if isinstance(target, exp.Table) else target.find(exp.Table)
    if table is None or not table.db or table.catalog:
        return None
    return f"{table.db}.{table.name}".lower()


def validate_transform(name: str, sql: str) -> None:
    """Allow exactly one ``CREATE [OR REPLACE] TABLE <OUTPUT_TABLES[name]> ...``."""
    expected = OUTPUT_TABLES[name]
    message = f"{name}.sql must be a single CREATE [OR REPLACE] TABLE {expected}"
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.SqlglotError as error:
        detail = " ".join(str(error).split())[:300]
        raise BuildError(f"{message} (could not parse: {detail})") from None
    if len(statements) != 1 or _created_table(statements[0]) != expected:
        raise BuildError(message)


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
        # Sandbox: transforms run after the raw load and may not touch files, the
        # network, or extensions, nor loosen these settings again.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        for name in TRANSFORM_ORDER:
            sql = (transforms / f"{name}.sql").read_text()
            validate_transform(name, sql)
            try:
                con.execute(sql)
            except duckdb.Error as error:
                raise BuildError(f"{name}.sql failed: {error}") from error
    finally:
        con.close()
