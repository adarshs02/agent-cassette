import pytest
from retrace.tools import warehouse as wh
from retrace.tools.warehouse import SqlRejected, SqlTimeout, Warehouse


@pytest.fixture()
def tools(healthy_ws, baseline):
    return Warehouse(healthy_ws.warehouse, baseline)


def test_run_sql_returns_json_rows(tools):
    out = tools.run_sql("SELECT day, revenue_usd FROM marts.fct_revenue ORDER BY day LIMIT 2")
    assert out["columns"] == ["day", "revenue_usd"]
    assert out["row_count"] == 2
    assert isinstance(out["rows"][0][0], str)


@pytest.mark.parametrize(
    "query",
    [
        "CREATE TABLE x AS SELECT 1",
        "COPY marts.fct_revenue TO 'x.csv'",
        "ATTACH 'other.duckdb'",
        "PRAGMA database_list",
        "SELECT 1; DROP TABLE marts.fct_revenue",
        "DELETE FROM marts.fct_revenue",
        "not sql at all (((",
    ],
)
def test_run_sql_rejects_non_queries(tools, query):
    with pytest.raises(SqlRejected):
        tools.run_sql(query)


def test_run_sql_truncates(tools):
    out = tools.run_sql("SELECT * FROM staging.stg_orders")
    assert out["truncated"] is True
    assert len(str(out)) < wh.MAX_RESULT_CHARS + 2000


def test_timeout(tools, monkeypatch):
    monkeypatch.setattr(wh, "QUERY_TIMEOUT_S", 0.2)
    with pytest.raises(SqlTimeout):
        tools.run_sql("SELECT COUNT(*) FROM range(100000000000)")


def test_compare_to_baseline_healthy_is_flat(tools):
    out = tools.compare_to_baseline(14)
    assert len(out["days"]) == 14
    assert out["max_abs_deviation"] == 0.0


def test_profile_column_grouped(tools):
    out = tools.profile_column(
        "staging.stg_orders", "amount", group_by="payment_processor", since="2026-08-07"
    )
    groups = {g["grp"] for g in out["groups"]}
    assert groups == {"cloudpay_v2", "legacy_pos", "shopgate"}


@pytest.mark.parametrize(
    ("table", "column", "group_by"),
    [
        ("staging.stg_orders; --", "amount", None),
        ("staging.stg_orders", "nope", None),
        ("staging.stg_orders", "amount", "x) OR (1"),
    ],
)
def test_profile_column_rejects_bad_identifiers(tools, table, column, group_by):
    with pytest.raises(ValueError):
        tools.profile_column(table, column, group_by=group_by)


def test_run_checks_tool(tools):
    out = tools.run_checks()
    assert out["passed"] is True and out["failed"] == []
