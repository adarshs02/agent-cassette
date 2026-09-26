import duckdb
import pytest
from retrace.pipeline.build import BuildError, build
from retrace.pipeline.workspace import Workspace, prepare


def test_healthy_build(tmp_path):
    ws = prepare(tmp_path / "ws", fault=None)
    con = duckdb.connect(str(ws.warehouse), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM marts.fct_revenue").fetchone()[0] == 90
    kpi_day, ratio = con.execute(
        "SELECT CAST(kpi_day AS VARCHAR), anomaly_ratio FROM marts.exec_metric"
    ).fetchone()
    assert kpi_day == "2026-08-09"
    assert 0.8 <= ratio <= 1.25
    nulls = con.execute(
        "SELECT COUNT(*) FROM staging.stg_fx_rates WHERE usd_rate IS NULL"
    ).fetchone()[0]
    assert nulls == 0


def test_build_error_names_transform(tmp_path):
    ws = prepare(tmp_path / "ws", fault=None)
    (ws.transforms / "fct_revenue.sql").write_text("SELECT * FROM nope;")
    with pytest.raises(BuildError, match="fct_revenue.sql failed"):
        build(ws.sources, ws.transforms, ws.warehouse)


def test_workspace_create_copies_transforms(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    assert sorted(p.name for p in ws.transforms.glob("*.sql")) == [
        "exec_metric.sql",
        "fct_revenue.sql",
        "stg_customers.sql",
        "stg_fx_rates.sql",
        "stg_orders.sql",
    ]
