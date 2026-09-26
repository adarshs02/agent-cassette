"""Read-only fact tools over the DuckDB warehouse."""

from __future__ import annotations

import json
import re
import threading
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from retrace.pipeline.checks import BAND, failed_names
from retrace.pipeline.checks import run_checks as run_pipeline_checks

MAX_RESULT_CHARS = 8000
MAX_ROWS = 200
QUERY_TIMEOUT_S = 5.0
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class SqlRejected(ValueError):
    """The SQL is not a single read-only query."""


class SqlTimeout(RuntimeError):
    """The query exceeded the time budget."""


def ensure_read_only(query: str) -> None:
    try:
        statements = sqlglot.parse(query, read="duckdb")
    except sqlglot.errors.ParseError as error:
        raise SqlRejected(f"could not parse SQL: {error}") from error
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlRejected("exactly one statement is allowed")
    if not isinstance(statements[0], exp.Query):
        raise SqlRejected("only SELECT/WITH queries are allowed")


def _jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _truncate(payload: dict[str, Any]) -> dict[str, Any]:
    while len(json.dumps(payload)) > MAX_RESULT_CHARS and payload["rows"]:
        payload["rows"] = payload["rows"][: len(payload["rows"]) // 2]
        payload["truncated"] = True
    return payload


class Warehouse:
    def __init__(self, path: Path, baseline: dict) -> None:
        self.path = path
        self.baseline = baseline

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[list[str], list[tuple]]:
        con = duckdb.connect(str(self.path), read_only=True)
        timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
        try:
            con.execute("SET threads TO 1")
            timer.start()
            cursor = con.execute(sql, params)
            columns = [d[0] for d in cursor.description or []]
            return columns, cursor.fetchall()
        except duckdb.InterruptException as error:
            raise SqlTimeout(f"query exceeded {QUERY_TIMEOUT_S}s") from error
        finally:
            timer.cancel()
            con.close()

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        columns, rows = self._execute(sql, params)
        return [{c: _jsonable(v) for c, v in zip(columns, row, strict=True)} for row in rows]

    def run_sql(self, query: str) -> dict[str, Any]:
        ensure_read_only(query)
        try:
            columns, rows = self._execute(query)
        except duckdb.Error as error:
            raise SqlRejected(f"query failed: {error}") from error
        truncated = len(rows) > MAX_ROWS
        payload = {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows[:MAX_ROWS]],
            "row_count": len(rows),
            "truncated": truncated,
        }
        return _truncate(payload)

    def metric_history(self, days: int = 30) -> dict[str, Any]:
        kpi = self._rows("SELECT * FROM marts.exec_metric")
        daily = self._rows(
            "SELECT day, order_count, revenue_usd, aov_median_usd FROM marts.fct_revenue "
            "ORDER BY day DESC LIMIT ?",
            (int(days),),
        )
        return {"kpi": kpi[0] if kpi else None, "daily": list(reversed(daily))}

    def compare_to_baseline(self, last_days: int = 14) -> dict[str, Any]:
        rows = self._execute(
            "SELECT CAST(day AS VARCHAR), revenue_usd, order_count FROM marts.fct_revenue "
            "ORDER BY day DESC LIMIT ?",
            (int(last_days),),
        )[1]
        out, deviations = [], []
        for day, revenue, count in reversed(rows):
            base = self.baseline["days"].get(day)
            if base is None or revenue is None:
                out.append({"day": day, "revenue_ratio": None, "order_count_ratio": None})
                deviations.append(1.0)
                continue
            revenue_ratio = revenue / base["revenue_usd"]
            count_ratio = count / base["order_count"]
            out.append(
                {
                    "day": day,
                    "revenue_ratio": round(revenue_ratio, 4),
                    "order_count_ratio": round(count_ratio, 4),
                }
            )
            deviations.append(max(abs(revenue_ratio - 1), abs(count_ratio - 1)))
        return {
            "days": out,
            "max_abs_deviation": round(max(deviations, default=1.0), 4),
            "band": BAND,
        }

    def _columns(self, schema: str, table: str) -> list[str]:
        return [
            r[0]
            for r in self._execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                (schema, table),
            )[1]
        ]

    def profile_column(
        self, table: str, column: str, group_by: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        schema, _, name = table.partition(".")
        for ident in (schema, name, column, *([group_by] if group_by else [])):
            if not _IDENT.match(ident or ""):
                raise ValueError(f"invalid identifier: {ident!r}")
        columns = self._columns(schema, name)
        if not columns:
            raise ValueError(f"unknown table {table!r}")
        for ident in (column, *([group_by] if group_by else [])):
            if ident not in columns:
                raise ValueError(f"unknown column {ident!r} in {table}; columns: {columns}")
        where = ""
        if since is not None:
            day = date.fromisoformat(since).isoformat()
            date_expr = next(
                (
                    e
                    for c, e in (
                        ("order_day", "order_day"),
                        ("day", "day"),
                        ("rate_day", "TRY_CAST(rate_day AS DATE)"),
                        ("order_ts", "TRY_CAST(order_ts AS DATE)"),
                    )
                    if c in columns
                ),
                None,
            )
            if date_expr is None:
                raise ValueError(f"{table} has no date column for 'since'")
            where = f"WHERE {date_expr} >= DATE '{day}'"
        grp = group_by or "'all'"
        groups = self._rows(
            f"SELECT CAST({grp} AS VARCHAR) AS grp, COUNT(*) AS row_count, "
            f"COUNT({column}) AS non_null, COUNT(DISTINCT {column}) AS distinct_values, "
            f"MIN(TRY_CAST({column} AS DOUBLE)) AS min_num, "
            f"MAX(TRY_CAST({column} AS DOUBLE)) AS max_num, "
            f"MEDIAN(TRY_CAST({column} AS DOUBLE)) AS median_num, "
            f"MIN(CAST({column} AS VARCHAR)) AS min_text, "
            f"MAX(CAST({column} AS VARCHAR)) AS max_text "
            f"FROM {schema}.{name} {where} GROUP BY 1 ORDER BY 1"
        )
        return {"table": table, "column": column, "since": since, "groups": groups}

    def run_checks(self) -> dict[str, Any]:
        results = run_pipeline_checks(self.path, self.baseline)
        failed = failed_names(results)
        return {
            "passed": not failed,
            "failed": failed,
            "results": [r.to_dict() for r in results],
        }
