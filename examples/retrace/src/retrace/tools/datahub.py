"""Agent-facing DataHub tools over an MCP connection."""

from __future__ import annotations

import concurrent.futures
from typing import Any

from retrace.datahub.mcp_client import DataHubConnection

_DATAHUB_ERRORS = (
    RuntimeError,
    TimeoutError,
    concurrent.futures.TimeoutError,
    ConnectionError,
    OSError,
    ValueError,
)


def _document_urn(doc: Any) -> str | None:
    """Best-effort: pull the created document's URN out of a save_document result."""
    if isinstance(doc, str):
        return doc if doc.startswith("urn:li:") else None
    if not isinstance(doc, dict):
        return None
    for key in ("urn", "documentUrn", "document_urn"):
        value = doc.get(key)
        if isinstance(value, str) and value.startswith("urn:li:"):
            return value
    for key in ("document", "data", "result"):
        found = _document_urn(doc.get(key))
        if found:
            return found
    return None


class DataHubTools:
    def __init__(self, conn: DataHubConnection) -> None:
        self._conn = conn

    def _call(self, name: str, args: dict[str, Any]) -> Any:
        try:
            return self._conn.call(name, args)
        except _DATAHUB_ERRORS as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def search(self, query: str) -> Any:
        return self._call("search", {"query": f"/q {query}", "num_results": 10})

    def get_dataset(self, urn: str) -> Any:
        return self._call("get_entities", {"urns": [urn]})

    def lineage(self, urn: str, direction: str, max_hops: int = 3) -> Any:
        if direction not in ("upstream", "downstream"):
            return {"error": "direction must be 'upstream' or 'downstream'"}
        return self._call(
            "get_lineage",
            {"urn": urn, "upstream": direction == "upstream", "max_hops": int(max_hops)},
        )

    def write_back(self, asset_urn: str, title: str, content: str) -> Any:
        doc = self._call(
            "save_document",
            {
                "document_type": "Analysis",
                "title": title,
                "content": content,
                "topics": ["retrace-incident"],
                "related_assets": [asset_urn],
            },
        )
        if isinstance(doc, dict) and "error" in doc:
            return doc
        tagged = self._call(
            "add_tags", {"tag_urns": ["urn:li:tag:retrace-incident"], "entity_urns": [asset_urn]}
        )
        urn = _document_urn(doc)
        if urn is None:
            return tagged
        if isinstance(tagged, dict):
            return {**tagged, "document_urn": urn}
        return {"result": tagged, "document_urn": urn}
