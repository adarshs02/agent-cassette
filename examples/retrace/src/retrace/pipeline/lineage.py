"""Table-level lineage derived from the SQL transforms."""

from __future__ import annotations

import re
from pathlib import Path

import sqlglot
from sqlglot import exp

from retrace.pipeline.build import OUTPUT_TABLES, TRANSFORM_ORDER
from retrace.pipeline.workspace import TRANSFORMS_DIR

# OUTPUT_TABLES lives in build.py (build validates transforms against it); re-exported here.
_URN = re.compile(r"urn:li:dataset:\(urn:li:dataPlatform:[^,]+,([^,]+),[^)]+\)")


def table_name(asset: str) -> str:
    match = _URN.match(asset.strip())
    return match.group(1) if match else asset.strip()


def reads_tables(sql: str, output: str | None = None) -> set[str]:
    statement = sqlglot.parse_one(sql, read="duckdb")
    tables = {f"{t.db}.{t.name}" for t in statement.find_all(exp.Table) if t.db}
    if output is not None:
        tables.discard(output)
    return tables


def direct_dependencies(transforms_dir: Path = TRANSFORMS_DIR) -> dict[str, set[str]]:
    return {
        name: reads_tables((transforms_dir / f"{name}.sql").read_text(), OUTPUT_TABLES[name])
        for name in TRANSFORM_ORDER
    }


def dependency_closure(transforms_dir: Path = TRANSFORMS_DIR) -> dict[str, set[str]]:
    direct = direct_dependencies(transforms_dir)
    producer = {table: name for name, table in OUTPUT_TABLES.items()}
    closure: dict[str, set[str]] = {}
    for name in TRANSFORM_ORDER:
        acc = set(direct[name])
        for table in direct[name]:
            if table in producer:
                acc |= closure[producer[table]]
        closure[name] = acc
    return closure
