from retrace.pipeline.lineage import dependency_closure, direct_dependencies, table_name


def test_table_name_accepts_urn_and_plain():
    urn = "urn:li:dataset:(urn:li:dataPlatform:duckdb,raw.raw_orders,PROD)"
    assert table_name(urn) == "raw.raw_orders"
    assert table_name(" raw.raw_orders ") == "raw.raw_orders"


def test_direct_dependencies():
    deps = direct_dependencies()
    assert deps["stg_orders"] == {"raw.raw_orders"}
    assert deps["fct_revenue"] == {
        "staging.stg_orders",
        "staging.stg_customers",
        "staging.stg_fx_rates",
    }


def test_closure_reaches_raw_tables():
    closure = dependency_closure()
    assert "raw.raw_orders" in closure["exec_metric"]
    assert "raw.raw_customers" in closure["fct_revenue"]
    assert closure["stg_fx_rates"] == {"raw.raw_fx_rates"}
