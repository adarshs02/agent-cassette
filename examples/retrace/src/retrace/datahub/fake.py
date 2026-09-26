"""In-process stand-in for mcp-server-datahub, built from the static catalog."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent

from retrace.datahub.catalog import (
    COLUMNS,
    DATASET_DOCS,
    FIELD_DOCS,
    OWNERS,
    TABLES,
    dataset_urn,
    lineage_edges,
)
from retrace.pipeline.lineage import table_name


def _ok(payload: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], isError=False
    )


def _err(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


def _entity(table: str) -> dict[str, Any]:
    owner = next((u for u, (_, _, tables) in OWNERS.items() if table in tables), None)
    return {
        "urn": dataset_urn(table),
        "name": table,
        "description": DATASET_DOCS.get(table, ""),
        "schemaMetadata": {
            "fields": [
                {"fieldPath": c, "description": FIELD_DOCS.get(table, {}).get(c, "")}
                for c in COLUMNS[table]
            ]
        },
        "ownership": {"owners": [f"urn:li:corpuser:{owner}"] if owner else []},
    }


def _walk(start: str, upstream: bool, max_hops: int) -> list[dict[str, Any]]:
    edges = lineage_edges()
    frontier, seen, found = {start}, {start}, []
    for degree in range(1, max_hops + 1):
        nxt = set()
        for up, down in edges:
            src, dst = (down, up) if upstream else (up, down)
            if src in frontier and dst not in seen:
                seen.add(dst)
                nxt.add(dst)
                found.append({"entity": {"urn": dataset_urn(dst), "name": dst}, "degree": degree})
        frontier = nxt
    return sorted(found, key=lambda r: (r["degree"], r["entity"]["name"]))


class FakeDataHubSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "search":
            cleaned = str(args.get("query", "")).lower().replace("/q", " ").replace("+", " ")
            tokens = [t for t in cleaned.split() if t]
            hits = [
                t
                for t in TABLES
                if any(tok in (t + " " + DATASET_DOCS.get(t, "")).lower() for tok in tokens)
            ]
            return _ok(
                {
                    "total": len(hits),
                    "searchResults": [
                        {
                            "entity": {
                                "urn": dataset_urn(t),
                                "name": t,
                                "properties": {"description": DATASET_DOCS.get(t, "")},
                            }
                        }
                        for t in hits
                    ],
                }
            )
        if name == "get_entities":
            urns = args.get("urns")
            urns = [urns] if isinstance(urns, str) else list(urns or [])
            tables = [table_name(u) for u in urns]
            unknown = [t for t in tables if t not in TABLES]
            if unknown:
                return _err(f"entities not found: {unknown}")
            return _ok([_entity(t) for t in tables])
        if name == "get_lineage":
            start = table_name(str(args.get("urn", "")))
            if start not in TABLES:
                return _err(f"unknown urn {args.get('urn')!r}")
            upstream = bool(args.get("upstream", True))
            results = _walk(start, upstream, int(args.get("max_hops", 1)))
            key = "upstreams" if upstream else "downstreams"
            return _ok({key: {"total": len(results), "searchResults": results}})
        if name in ("save_document", "add_tags"):
            return _ok({"success": True})
        return _err(f"unknown tool {name!r}")
