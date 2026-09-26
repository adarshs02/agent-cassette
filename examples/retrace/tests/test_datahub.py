import json

import pytest
from mcp.types import CallToolResult, TextContent
from retrace.datahub.catalog import TABLES, dataset_urn, lineage_edges
from retrace.datahub.fake import FakeDataHubSession
from retrace.datahub.mcp_client import DataHubConnection, DataHubToolError, parse_result
from retrace.tools.datahub import DataHubTools

from agent_cassette import Cassette

KPI = dataset_urn("marts.exec_metric")


def test_lineage_edges_follow_transforms():
    edges = set(lineage_edges())
    assert ("raw.raw_orders", "staging.stg_orders") in edges
    assert ("staging.stg_customers", "marts.fct_revenue") in edges
    assert ("marts.fct_revenue", "marts.exec_metric") in edges
    assert len(TABLES) == 8


def test_parse_result_handles_error_and_text():
    ok = CallToolResult(content=[TextContent(type="text", text='{"a": 1}')], isError=False)
    assert parse_result(ok) == {"a": 1}
    text = CallToolResult(content=[TextContent(type="text", text="plain words")], isError=False)
    assert parse_result(text) == "plain words"
    bad = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    with pytest.raises(DataHubToolError, match="boom"):
        parse_result(bad)


def test_tools_over_fake_session():
    conn = DataHubConnection.from_session(FakeDataHubSession())
    try:
        tools = DataHubTools(conn)
        hits = tools.search("executive revenue")
        assert any("exec_metric" in json.dumps(h) for h in hits["searchResults"])
        up = tools.lineage(KPI, "upstream", 3)
        assert "raw.raw_orders" in json.dumps(up)
        ds = tools.get_dataset(dataset_urn("raw.raw_orders"))
        assert "amount" in json.dumps(ds)
        assert tools.write_back(dataset_urn("raw.raw_orders"), "t", "c") == {"success": True}
        assert "error" in tools.lineage(KPI, "sideways", 3)
    finally:
        conn.close()


def test_tool_errors_become_error_dicts():
    class Broken(FakeDataHubSession):
        async def call_tool(self, name, arguments=None):
            raise TimeoutError("datahub down")

    conn = DataHubConnection.from_session(Broken())
    try:
        assert DataHubTools(conn).search("x") == {"error": "TimeoutError: datahub down"}
    finally:
        conn.close()


def test_record_then_replay_without_session(tmp_path):
    path = tmp_path / "dh.jsonl"
    with Cassette.record(path) as cassette:
        conn = DataHubConnection.from_session(FakeDataHubSession(), cassette)
        recorded = DataHubTools(conn).lineage(KPI, "upstream", 3)
        conn.close()
    with Cassette.replay(path) as cassette:
        conn = DataHubConnection.replay(cassette)
        replayed = DataHubTools(conn).lineage(KPI, "upstream", 3)
        conn.close()
    assert json.loads(json.dumps(replayed)) == json.loads(json.dumps(recorded))
