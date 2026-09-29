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


def _tool(name, *params, as_dict=False):
    schema = {"type": "object", "properties": {p: {"type": "string"} for p in params}}
    if as_dict:
        return {"name": name, "inputSchema": schema}
    from mcp.types import Tool

    return Tool(name=name, inputSchema=schema)


FULL_CONTRACT = [
    ("search", "query", "num_results", "filters"),
    ("get_entities", "urns"),
    ("get_lineage", "urn", "upstream", "max_hops"),
    ("save_document", "document_type", "title", "content", "topics", "related_assets"),
    ("add_tags", "tag_urns", "entity_urns"),
    ("list_schema_fields", "urn"),
]


def test_tool_contract_accepts_matching_listing():
    from retrace.datahub.mcp_client import check_tool_contract

    assert check_tool_contract([_tool(*spec) for spec in FULL_CONTRACT]) == []
    assert check_tool_contract([_tool(*spec, as_dict=True) for spec in FULL_CONTRACT]) == []


def test_tool_contract_reports_missing_tools_and_params():
    from retrace.datahub.mcp_client import check_tool_contract

    tools = [
        _tool("search", "query"),
        _tool("get_entities", "urns"),
        _tool("get_lineage", "urn", "direction", "max_hops"),
        _tool("add_tags", "tag_urns", "entity_urns"),
    ]
    problems = check_tool_contract(tools)
    assert "search: missing parameter num_results" in problems
    assert "get_lineage: missing parameter upstream" in problems
    assert "missing tool save_document" in problems
    assert len(problems) == 3


def test_live_rejects_server_with_wrong_contract(monkeypatch, tmp_path):
    import contextlib

    import pytest
    from mcp.types import ListToolsResult
    from retrace.config import Settings
    from retrace.datahub import mcp_client

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, **kwargs):
            return ListToolsResult(tools=[_tool("search", "query")])

    @contextlib.asynccontextmanager
    async def fake_stdio(params, errlog=None):
        yield (None, None)

    monkeypatch.setattr("mcp.ClientSession", Session)
    monkeypatch.setattr("mcp.client.stdio.stdio_client", fake_stdio)
    with pytest.raises(mcp_client.DataHubUnavailable, match="missing tool save_document"):
        mcp_client.DataHubConnection.live(Settings(mcp_log_path=tmp_path / "mcp-server.log"))


def test_server_env_is_an_allowlist(monkeypatch):
    import os

    from retrace.config import Settings
    from retrace.datahub.mcp_client import _server_env

    for key in list(os.environ):
        if key.startswith(("UV_", "XDG_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/x")
    env = _server_env(Settings(datahub_gms_url="http://gms:8080", datahub_gms_token="tok"))
    assert "ANTHROPIC_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "UV_CACHE_DIR": "/tmp/uv",
        "XDG_CACHE_HOME": "/tmp/xdg",
        "DATAHUB_GMS_URL": "http://gms:8080",
        "DATAHUB_GMS_TOKEN": "tok",
        "TOOLS_IS_MUTATION_ENABLED": "true",
    }
    assert "DATAHUB_GMS_TOKEN" not in _server_env(Settings())


def test_server_errlog_opens_configured_path(tmp_path):
    from retrace.config import Settings
    from retrace.datahub.mcp_client import _server_errlog

    log_path = tmp_path / "nested" / "mcp-server.log"
    handle = _server_errlog(Settings(mcp_log_path=log_path))
    try:
        assert log_path.exists()
        assert handle.mode == "a"
        handle.write("hello\n")
        handle.flush()
        assert log_path.read_text() == "hello\n"
    finally:
        handle.close()


def test_server_errlog_default_path_is_under_work_dir():
    from retrace.config import DEFAULT_MCP_LOG_PATH, Settings

    assert Settings().mcp_log_path == DEFAULT_MCP_LOG_PATH
    assert DEFAULT_MCP_LOG_PATH.parts[-2:] == ("work", "mcp-server.log")


def test_shutdown_server_closes_errlog_even_if_runner_stop_raises(tmp_path):
    import asyncio
    import concurrent.futures

    import pytest
    from retrace.datahub.mcp_client import _shutdown_server

    class FakeLoop:
        def call_soon_threadsafe(self, callback):
            callback()

    class FakeRunner:
        def __init__(self):
            self.loop = FakeLoop()

        def stop(self):
            raise RuntimeError("loop thread would not join")

    stop_event = asyncio.Event()
    task: concurrent.futures.Future[None] = concurrent.futures.Future()
    task.set_result(None)
    errlog = open(tmp_path / "mcp-server.log", "a")

    with pytest.raises(RuntimeError, match="loop thread would not join"):
        _shutdown_server(FakeRunner(), task, stop_event, errlog)

    assert stop_event.is_set()
    assert errlog.closed


class _SlowSession(FakeDataHubSession):
    async def call_tool(self, name, arguments=None):
        import asyncio

        await asyncio.sleep(2)
        return await super().call_tool(name, arguments)


def test_mcp_timeout_is_recorded_and_replayed(tmp_path):
    import pytest

    path = tmp_path / "dh.jsonl"
    with Cassette.record(path) as cassette:
        conn = DataHubConnection.from_session(_SlowSession(), cassette, timeout=0.05)
        try:
            with pytest.raises(TimeoutError):
                conn.call("search", {"query": "x", "num_results": 10})
        finally:
            conn.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    calls = [e for e in events if e.get("name") == "mcp.search"]
    assert len(calls) == 1
    assert "TimeoutError" in json.dumps(calls[0])
    with Cassette.replay(path) as cassette:
        conn = DataHubConnection.replay(cassette)
        try:
            with pytest.raises(TimeoutError):
                conn.call("search", {"query": "x", "num_results": 10})
        finally:
            conn.close()


# --- Compaction: drop facets/aggregations/platform objects, keep urn/name/description ---

REAL_FACETS = [
    {
        "field": "platform",
        "displayName": "Platform",
        "aggregations": [{"value": "urn:li:dataPlatform:duckdb", "count": 8}],
    }
]


def test_compact_search_strips_facets_and_flattens_hits():
    from retrace.tools.datahub import _compact_search

    real = {
        "start": 0,
        "count": 10,
        "total": 1,
        "searchResults": [
            {
                "entity": {
                    "urn": "urn:li:dataset:(urn:li:dataPlatform:duckdb,marts.exec_metric,PROD)",
                    "properties": {"name": "marts.exec_metric"},
                }
            }
        ],
        "facets": REAL_FACETS,
    }
    compacted = _compact_search(real)
    assert compacted == {
        "total": 1,
        "searchResults": [
            {
                "urn": "urn:li:dataset:(urn:li:dataPlatform:duckdb,marts.exec_metric,PROD)",
                "name": "marts.exec_metric",
            }
        ],
    }
    assert len(json.dumps(compacted)) < len(json.dumps(real))


def test_compact_search_passes_through_unknown_shapes():
    from retrace.tools.datahub import _compact_search

    assert _compact_search({"error": "boom"}) == {"error": "boom"}
    assert _compact_search("plain text") == "plain text"


def test_compact_lineage_strips_facets_keeps_urn_name_degree():
    from retrace.tools.datahub import _compact_lineage

    real = {
        "upstreams": {
            "total": 7,
            "facets": REAL_FACETS,
            "searchResults": [
                {
                    "entity": {"urn": dataset_urn("raw.raw_orders"), "name": "raw.raw_orders"},
                    "degree": 1,
                }
            ],
        }
    }
    compacted = _compact_lineage(real, "upstream")
    assert compacted == {
        "upstreams": [
            {"urn": dataset_urn("raw.raw_orders"), "name": "raw.raw_orders", "degree": 1}
        ],
        "total": 7,
    }
    assert len(json.dumps(compacted)) < len(json.dumps(real))


def test_compact_lineage_passes_through_unknown_shapes():
    from retrace.tools.datahub import _compact_lineage

    assert _compact_lineage({"error": "boom"}, "upstream") == {"error": "boom"}
    assert _compact_lineage({"downstreams": {"total": 0}}, "upstream") == {
        "downstreams": {"total": 0}
    }


def test_compact_entities_strips_platform_flattens_owners_and_fields():
    from retrace.tools.datahub import _compact_entities

    real = [
        {
            "urn": dataset_urn("raw.raw_orders"),
            "name": "raw.raw_orders",
            "platform": {"urn": "urn:li:dataPlatform:duckdb", "properties": {"logoUrl": "x"}},
            "properties": {"name": "raw.raw_orders", "description": "Order events."},
            "schemaMetadata": {
                "fields": [
                    {"fieldPath": "amount", "description": "Amount.", "nativeDataType": "double"}
                ]
            },
            "ownership": {"owners": [{"owner": {"urn": "urn:li:corpuser:jordan.lee"}}]},
        }
    ]
    compacted = _compact_entities(real)
    assert compacted == [
        {
            "urn": dataset_urn("raw.raw_orders"),
            "name": "raw.raw_orders",
            "description": "Order events.",
            "fields": [{"fieldPath": "amount", "description": "Amount.", "type": "double"}],
            "owners": ["urn:li:corpuser:jordan.lee"],
        }
    ]
    assert len(json.dumps(compacted)) < len(json.dumps(real))


def test_compact_entities_passes_through_error_dict():
    from retrace.tools.datahub import _compact_entities

    assert _compact_entities({"error": "not found"}) == {"error": "not found"}


def test_tag_name_handles_bare_strings_and_nested_shapes():
    from retrace.tools.datahub import _tag_name

    assert _tag_name("urn:li:tag:pii") == "urn:li:tag:pii"
    assert _tag_name({"tag": {"properties": {"name": "pii"}}}) == "pii"
    assert _tag_name({"tag": {"name": "pii"}}) == "pii"
    assert _tag_name({"tag": {"urn": "urn:li:tag:pii"}}) == "urn:li:tag:pii"
    assert _tag_name({"name": "pii"}) == "pii"  # no "tag" wrapper
    assert _tag_name({"urn": "urn:li:tag:pii"}) == "urn:li:tag:pii"
    assert _tag_name(123) is None
    assert _tag_name({}) is None


def test_compact_tags_reads_tags_and_global_tags_shapes():
    from retrace.tools.datahub import _compact_tags

    assert _compact_tags({"tags": {"tags": ["urn:li:tag:pii"]}}) == ["urn:li:tag:pii"]
    assert _compact_tags({"globalTags": {"tags": [{"tag": {"properties": {"name": "pii"}}}]}}) == [
        "pii"
    ]
    assert _compact_tags({}) == []
    assert _compact_tags({"tags": {"tags": []}}) == []


def test_compact_entity_keeps_bare_string_owners_and_tags_together():
    """The fake produces owners as bare corpuser URN strings (not
    {"owner": {...}} dicts, unlike the real-shape test above); a tags block
    must survive compaction alongside them."""
    from retrace.tools.datahub import _compact_entity

    entity = {
        "urn": dataset_urn("raw.raw_orders"),
        "name": "raw.raw_orders",
        "ownership": {"owners": ["urn:li:corpuser:jordan.lee"]},
        "globalTags": {"tags": [{"tag": {"urn": "urn:li:tag:retrace-incident"}}]},
    }
    compacted = _compact_entity(entity)
    assert compacted["owners"] == ["urn:li:corpuser:jordan.lee"]
    assert compacted["tags"] == ["urn:li:tag:retrace-incident"]


def test_tools_over_fake_session_owners_survive_compaction():
    """End-to-end: DataHubTools.get_dataset over the fake session (which
    emits bare-string owner URNs, per test_compact_entity above) keeps the
    owners list after compaction."""
    conn = DataHubConnection.from_session(FakeDataHubSession())
    try:
        ds = DataHubTools(conn).get_dataset(dataset_urn("raw.raw_orders"))
        assert ds[0]["owners"] == ["urn:li:corpuser:jordan.lee"]
    finally:
        conn.close()


def test_compact_lineage_keeps_gate_working():
    """The lineage gate (check_claim_evidence) substring-matches the table
    name in json.dumps(evidence.payload); compaction must not lose it, even
    against the real (facet-heavy, entity-nested) response shape."""
    from retrace.agent.evidence import EvidenceStore, check_claim_evidence
    from retrace.agent.state import IncidentState
    from retrace.tools.datahub import _compact_lineage

    real = {
        "upstreams": {
            "total": 1,
            "facets": REAL_FACETS,
            "searchResults": [
                {
                    "entity": {"urn": dataset_urn("raw.raw_orders"), "name": "raw.raw_orders"},
                    "degree": 2,
                }
            ],
        }
    }
    compacted = _compact_lineage(real, "upstream")
    state = IncidentState(scenario="t", report="r")
    store = EvidenceStore(state)
    lineage_ev = store.add("datahub", "lineage", "upstream of KPI", compacted)
    profile_ev = store.add("warehouse", "profile", "amount by processor", {"rows": []})
    check_claim_evidence(store, "raw.raw_orders", [lineage_ev.id, profile_ev.id])  # no raise


def test_tools_over_fake_session_use_compacted_shapes():
    """DataHubTools.search/get_dataset/lineage apply compaction over the fake
    session too (no "properties"/"schemaMetadata"/"ownership" nesting left)."""
    conn = DataHubConnection.from_session(FakeDataHubSession())
    try:
        tools = DataHubTools(conn)
        hits = tools.search("executive revenue")
        assert any(
            h["urn"] == dataset_urn("marts.exec_metric") and h["name"] == "marts.exec_metric"
            for h in hits["searchResults"]
        )
        assert "properties" not in json.dumps(hits)
        up = tools.lineage(KPI, "upstream", 3)
        assert "properties" not in json.dumps(up)
        assert any(e["name"] == "raw.raw_orders" for e in up["upstreams"])
        ds = tools.get_dataset(dataset_urn("raw.raw_orders"))
        assert "schemaMetadata" not in json.dumps(ds)
        assert any(f["fieldPath"] == "amount" for f in ds[0]["fields"])
    finally:
        conn.close()
