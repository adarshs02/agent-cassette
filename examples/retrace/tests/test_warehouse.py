import duckdb
import pytest
from retrace.tools import warehouse as wh
from retrace.tools.warehouse import SqlRejected, SqlTimeout, Warehouse


@pytest.fixture()
def tools(healthy_ws, baseline):
    return Warehouse(healthy_ws.warehouse, baseline)


def _reference_rows(warehouse_path, sql: str) -> list[list]:
    """Run `sql` directly against the warehouse (bypassing run_sql/its
    rewrite entirely) to compute an expected result from an explicit,
    fully-disambiguating ORDER BY."""
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    return [list(row) for row in rows]


def test_run_sql_returns_json_rows(tools):
    out = tools.run_sql("SELECT day, revenue_usd FROM marts.fct_revenue ORDER BY day LIMIT 2")
    assert out["columns"] == ["day", "revenue_usd"]
    assert out["row_count"] == 2
    assert isinstance(out["rows"][0][0], str)


def test_run_sql_tie_plus_limit_is_deterministic(tools, healthy_ws):
    # raw_fx_rates has 4 currencies tied on every rate_day, so ORDER BY
    # rate_day DESC LIMIT 6 cuts mid-tie: DuckDB gives no guarantee about
    # which of the tied rows come back, and that choice can differ across
    # platforms/thread counts. Compute the expected rows with an explicit,
    # fully-disambiguating ORDER BY (ties broken by currency ascending).
    expected = _reference_rows(
        healthy_ws.warehouse,
        "SELECT currency, rate_day FROM raw.raw_fx_rates "
        "ORDER BY rate_day DESC, currency ASC LIMIT 6",
    )
    out = tools.run_sql(
        "SELECT currency, rate_day FROM raw.raw_fx_rates ORDER BY rate_day DESC LIMIT 6"
    )
    assert out["rows"] == expected
    # The model's own DESC direction on rate_day must survive the rewrite.
    days = [row[1] for row in out["rows"]]
    assert days == sorted(days, reverse=True)


def test_run_sql_distinct_no_order_by_is_sorted(tools, healthy_ws):
    # No ORDER BY at all: DuckDB doesn't promise any particular row order.
    # NULLS LAST matches DuckDB's default for an ascending ordinal.
    expected = _reference_rows(
        healthy_ws.warehouse, "SELECT DISTINCT currency FROM raw.raw_orders ORDER BY 1"
    )
    out = tools.run_sql("SELECT DISTINCT currency FROM raw.raw_orders")
    assert out["rows"] == expected


def test_run_sql_union_is_deterministic(tools, healthy_ws):
    query = "SELECT currency FROM raw.raw_fx_rates UNION SELECT currency FROM raw.raw_orders"
    expected = _reference_rows(healthy_ws.warehouse, query + " ORDER BY 1")
    first = tools.run_sql(query)
    second = tools.run_sql(query)
    assert first["rows"] == expected
    assert first["rows"] == second["rows"]


def test_run_sql_with_cte_order_by_desc_limit_is_deterministic(tools, healthy_ws):
    query = (
        "WITH recent AS (SELECT currency, rate_day FROM raw.raw_fx_rates) "
        "SELECT currency, rate_day FROM recent ORDER BY rate_day DESC LIMIT 6"
    )
    expected = _reference_rows(
        healthy_ws.warehouse,
        "WITH recent AS (SELECT currency, rate_day FROM raw.raw_fx_rates) "
        "SELECT currency, rate_day FROM recent ORDER BY rate_day DESC, currency ASC LIMIT 6",
    )
    first = tools.run_sql(query)
    second = tools.run_sql(query)
    assert first["rows"] == expected
    assert first["rows"] == second["rows"]
    days = [row[1] for row in first["rows"]]
    assert days == sorted(days, reverse=True)


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


def test_run_sql_rejects_host_file_access(tools):
    with pytest.raises(SqlRejected):
        tools.run_sql("SELECT * FROM read_csv('/etc/passwd')")


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
