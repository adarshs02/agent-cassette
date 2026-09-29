"""Agent-facing DataHub tools over an MCP connection."""

from __future__ import annotations

import concurrent.futures
from typing import Any

from retrace.datahub.catalog import INCIDENT_TAG
from retrace.datahub.mcp_client import DataHubConnection

_DATAHUB_ERRORS = (
    RuntimeError,
    TimeoutError,
    concurrent.futures.TimeoutError,
    ConnectionError,
    OSError,
    ValueError,
)

_DIRECTION_KEYS = {"upstream": "upstreams", "downstream": "downstreams"}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def _entity_dict(raw: Any) -> dict[str, Any]:
    """Best-effort: pull the 'entity' sub-object out of a search/lineage hit.

    The real mcp-server-datahub nests the entity under an "entity" key
    (``{"entity": {...}, "degree": 1}``); the fake and get_entities results are
    already flat entity dicts. Unknown shapes fall back to the raw dict.
    """
    if not isinstance(raw, dict):
        return {}
    entity = raw.get("entity")
    return entity if isinstance(entity, dict) else raw


def _entity_name(entity: dict[str, Any]) -> str | None:
    props = entity.get("properties")
    props = props if isinstance(props, dict) else {}
    return _str_or_none(entity.get("name")) or _str_or_none(props.get("name"))


def _entity_description(entity: dict[str, Any]) -> str | None:
    props = entity.get("properties")
    props = props if isinstance(props, dict) else {}
    return _str_or_none(entity.get("description")) or _str_or_none(props.get("description"))


def _compact_search_hit(hit: Any) -> Any:
    if not isinstance(hit, dict):
        return hit
    entity = _entity_dict(hit)
    return _drop_empty(
        {
            "urn": _str_or_none(entity.get("urn")),
            "name": _entity_name(entity),
            "description": _entity_description(entity),
        }
    )


def _compact_search(result: Any) -> Any:
    """Trim a `search` result to what the agent actually cites: per-hit
    urn/name/description plus the total. Drops facets/aggregations. Anything
    that doesn't look like a search result (e.g. an ``{"error": ...}`` dict)
    passes through unchanged."""
    if not isinstance(result, dict):
        return result
    hits = result.get("searchResults")
    if not isinstance(hits, list):
        return result
    return {"total": result.get("total"), "searchResults": [_compact_search_hit(h) for h in hits]}


def _compact_lineage_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    entity = _entity_dict(entry)
    out = _drop_empty({"urn": _str_or_none(entity.get("urn")), "name": _entity_name(entity)})
    degree = entry.get("degree")
    if degree is not None:
        out["degree"] = degree
    return out


def _compact_lineage(result: Any, direction: str) -> Any:
    """Trim a `get_lineage` result to {direction: [{urn, name, degree}], total}.
    Drops facets and the extra searchResults/direction nesting. Passes through
    unchanged if the shape doesn't match (e.g. an error dict)."""
    if not isinstance(result, dict):
        return result
    key = _DIRECTION_KEYS.get(direction)
    section = result.get(key) if key else None
    if not isinstance(section, dict):
        return result
    hits = section.get("searchResults")
    if not isinstance(hits, list):
        return result
    return {key: [_compact_lineage_entry(e) for e in hits], "total": section.get("total")}


def _compact_field(field: Any) -> Any:
    if not isinstance(field, dict):
        return field
    path = _str_or_none(field.get("fieldPath")) or _str_or_none(field.get("name"))
    out = _drop_empty({"fieldPath": path, "description": _str_or_none(field.get("description"))})
    field_type = field.get("type") or field.get("nativeDataType")
    if isinstance(field_type, str) and field_type:
        out["type"] = field_type
    return out


def _owner_urn(entry: Any) -> str | None:
    if isinstance(entry, str):
        return _str_or_none(entry)
    if isinstance(entry, dict):
        owner = entry.get("owner")
        if isinstance(owner, str):
            return _str_or_none(owner)
        if isinstance(owner, dict):
            return _str_or_none(owner.get("urn"))
        return _str_or_none(entry.get("urn"))
    return None


def _compact_owners(entity: dict[str, Any]) -> list[str]:
    ownership = entity.get("ownership")
    owners = ownership.get("owners") if isinstance(ownership, dict) else None
    if not isinstance(owners, list):
        return []
    return [urn for o in owners if (urn := _owner_urn(o))]


def _tag_name(entry: Any) -> str | None:
    if isinstance(entry, str):
        return _str_or_none(entry)
    if not isinstance(entry, dict):
        return None
    tag = entry.get("tag")
    if isinstance(tag, dict):
        props = tag.get("properties")
        props = props if isinstance(props, dict) else {}
        return (
            _str_or_none(props.get("name"))
            or _str_or_none(tag.get("name"))
            or _str_or_none(tag.get("urn"))
        )
    return _str_or_none(entry.get("name")) or _str_or_none(entry.get("urn"))


def _compact_tags(entity: dict[str, Any]) -> list[str]:
    for key in ("tags", "globalTags"):
        block = entity.get(key)
        items = block.get("tags") if isinstance(block, dict) else block
        if isinstance(items, list):
            names = [name for t in items if (name := _tag_name(t))]
            if names:
                return names
    return []


def _compact_entity(entity: Any) -> Any:
    if not isinstance(entity, dict):
        return entity
    schema = entity.get("schemaMetadata")
    fields = schema.get("fields") if isinstance(schema, dict) else None
    return _drop_empty(
        {
            "urn": _str_or_none(entity.get("urn")),
            "name": _entity_name(entity),
            "description": _entity_description(entity),
            "fields": [_compact_field(f) for f in fields] if isinstance(fields, list) else None,
            "owners": _compact_owners(entity),
            "tags": _compact_tags(entity),
        }
    )


def _compact_entities(result: Any) -> Any:
    """Trim a `get_entities` result to per-entity urn/name/description/fields/
    owners/tags. Drops platform objects and other noise. get_entities always
    returns a list on success; anything else (e.g. an error dict) passes
    through unchanged."""
    if not isinstance(result, list):
        return result
    return [_compact_entity(e) for e in result]


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
        result = self._call("search", {"query": f"/q {query}", "num_results": 10})
        return _compact_search(result)

    def get_dataset(self, urn: str) -> Any:
        result = self._call("get_entities", {"urns": [urn]})
        return _compact_entities(result)

    def lineage(self, urn: str, direction: str, max_hops: int = 3) -> Any:
        if direction not in ("upstream", "downstream"):
            return {"error": "direction must be 'upstream' or 'downstream'"}
        result = self._call(
            "get_lineage",
            {"urn": urn, "upstream": direction == "upstream", "max_hops": int(max_hops)},
        )
        return _compact_lineage(result, direction)

    def write_back(self, asset_urn: str, title: str, content: str) -> Any:
        doc = self._call(
            "save_document",
            {
                "document_type": "Analysis",
                "title": title,
                "content": content,
                "topics": [INCIDENT_TAG],
                "related_assets": [asset_urn],
            },
        )
        if isinstance(doc, dict) and "error" in doc:
            return doc
        tagged = self._call(
            "add_tags",
            {"tag_urns": [f"urn:li:tag:{INCIDENT_TAG}"], "entity_urns": [asset_urn]},
        )
        urn = _document_urn(doc)
        if urn is None:
            return tagged
        if isinstance(tagged, dict):
            return {**tagged, "document_urn": urn}
        return {"result": tagged, "document_urn": urn}
