"""Committed observability baseline from the healthy fixture."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb

from retrace.pipeline.workspace import prepare

BASELINE_PATH = Path(__file__).parent / "baselines" / "healthy.json"


def compute_baseline(warehouse: Path) -> dict:
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        rows = con.execute(
            "SELECT CAST(day AS VARCHAR), revenue_usd, order_count FROM marts.fct_revenue "
            "ORDER BY day"
        ).fetchall()
    finally:
        con.close()
    return {
        "days": {
            day: {"revenue_usd": round(revenue, 4), "order_count": int(count)}
            for day, revenue, count in rows
        }
    }


def build_healthy_baseline() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ws = prepare(Path(tmp) / "healthy", fault=None)
        return compute_baseline(ws.warehouse)


def write_baseline(path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_healthy_baseline(), indent=2, sort_keys=True) + "\n")


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    write_baseline()
