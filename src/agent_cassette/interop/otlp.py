"""Dependency-free OTLP JSON import and export with OpenInference conventions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType

_KIND_TO_EVENT = {
    "LLM": EventType.MODEL_CALL,
    "TOOL": EventType.TOOL_CALL,
    "AGENT": EventType.CUSTOM,
    "CHAIN": EventType.CUSTOM,
    "RETRIEVER": EventType.CUSTOM,
    "EMBEDDING": EventType.CUSTOM,
    "RERANKER": EventType.CUSTOM,
    "GUARDRAIL": EventType.CUSTOM,
    "EVALUATOR": EventType.CUSTOM,
    "UNKNOWN": EventType.CUSTOM,
}
_EVENT_TO_KIND = {
    EventType.MODEL_CALL: "LLM",
    EventType.TOOL_CALL: "TOOL",
    EventType.TOOL_RESULT: "TOOL",
    EventType.ERROR: "UNKNOWN",
    EventType.CUSTOM: "AGENT",
}


def _stable_hex(namespace: str, value: str, length: int) -> str:
    return hashlib.sha256(f"agent-cassette:{namespace}:{value}".encode()).hexdigest()[:length]


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _timestamp_to_ns(timestamp: str) -> int:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _ns_to_timestamp(value: Any) -> str:
    nanoseconds = int(value)
    moment = datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=timezone.utc)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read_json(source: Mapping[str, Any] | str | bytes | Path) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, bytes):
        value = json.loads(source.decode("utf-8"))
    elif isinstance(source, Path):
        value = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        stripped = source.lstrip()
        value = (
            json.loads(source)
            if stripped.startswith(("{", "["))
            else json.loads(Path(source).read_text(encoding="utf-8"))
        )
    else:
        raise TypeError("OTLP source must be a mapping, JSON text, bytes, or path")
    if not isinstance(value, Mapping):
        raise ValueError("OTLP JSON root must be an object")
    return value


def _write_json(destination: str | Path, value: Mapping[str, Any]) -> None:
    Path(destination).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def export_otlp(events: Iterable[Event], destination: str | Path | None = None) -> dict[str, Any]:
    """Export events as deterministic OTLP JSON and optionally write it to disk."""
    materialized = list(events)
    trace_seed = _json_value([event.to_dict() for event in materialized])
    trace_id = _stable_hex("trace", trace_seed, 32)
    span_ids = {event.id: _stable_hex("span", event.id, 16) for event in materialized}
    span_owners: dict[str, str] = {}
    for event in materialized:
        if event.span_id is not None:
            span_owners.setdefault(event.span_id, event.id)
    spans: list[dict[str, Any]] = []
    for event in materialized:
        start = _timestamp_to_ns(event.timestamp)
        end = start + round((event.duration_ms or 0.0) * 1_000_000)
        attributes = [
            _attribute("openinference.span.kind", _EVENT_TO_KIND[event.type]),
            _attribute("agent_cassette.event.id", event.id),
            _attribute("agent_cassette.event.type", event.type.value),
            _attribute("agent_cassette.metadata", _json_value(event.metadata)),
        ]
        if event.span_id is not None:
            attributes.append(_attribute("agent_cassette.span_id", event.span_id))
        if event.parent_id is not None:
            attributes.append(_attribute("agent_cassette.parent_id", event.parent_id))
        if event.input is not None:
            attributes.extend(
                [
                    _attribute("input.value", _json_value(event.input)),
                    _attribute("input.mime_type", "application/json"),
                ]
            )
        if event.output is not None:
            attributes.extend(
                [
                    _attribute("output.value", _json_value(event.output)),
                    _attribute("output.mime_type", "application/json"),
                ]
            )
        if event.cost is not None:
            attributes.append(_attribute("llm.cost.total", event.cost))
        span: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_ids[event.id],
            "name": event.name,
            "kind": 1,
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(end),
            "attributes": attributes,
            "status": {"code": 2 if event.type == EventType.ERROR else 1},
        }
        if event.parent_id is not None:
            parent_event_id = (
                event.parent_id if event.parent_id in span_ids else span_owners.get(event.parent_id)
            )
            if parent_event_id is not None:
                span["parentSpanId"] = span_ids[parent_event_id]
        spans.append(span)
    result = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attribute("service.name", "agent-cassette")]},
                "scopeSpans": [
                    {"scope": {"name": "agent_cassette", "version": "0.10-beta"}, "spans": spans}
                ],
            }
        ]
    }
    if destination is not None:
        _write_json(destination, result)
    return result


def _decode_value(container: Any) -> Any:
    if not isinstance(container, Mapping):
        raise ValueError("attribute value must be an object")
    for key in ("stringValue", "boolValue", "intValue", "doubleValue"):
        if key in container:
            value = container[key]
            return int(value) if key == "intValue" else value
    if "arrayValue" in container:
        values = container["arrayValue"].get("values", [])
        return [_decode_value(value) for value in values]
    if "kvlistValue" in container:
        return _attributes(container["kvlistValue"].get("values", []))
    raise ValueError("unsupported OTLP attribute value")


def _attributes(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        raise ValueError("attributes must be a list")
    result: dict[str, Any] = {}
    for attribute in values:
        if not isinstance(attribute, Mapping) or "key" not in attribute or "value" not in attribute:
            raise ValueError("invalid OTLP attribute")
        result[str(attribute["key"])] = _decode_value(attribute["value"])
    return result


def _decode_json_attribute(attributes: Mapping[str, Any], key: str) -> Any:
    value = attributes.get(key)
    if value is None:
        return None
    mime_type = attributes.get(key.replace(".value", ".mime_type"))
    if mime_type == "application/json" or isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _all_spans(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    spans: list[Mapping[str, Any]] = []
    resources = document.get("resourceSpans", [])
    if not isinstance(resources, list):
        raise ValueError("resourceSpans must be a list")
    for resource in resources:
        if not isinstance(resource, Mapping):
            raise ValueError("resource span must be an object")
        scopes = resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", []))
        if not isinstance(scopes, list):
            raise ValueError("scopeSpans must be a list")
        for scope in scopes:
            if not isinstance(scope, Mapping) or not isinstance(scope.get("spans", []), list):
                raise ValueError("scope span must contain a spans list")
            spans.extend(scope.get("spans", []))
    return spans


def import_otlp(
    source: Mapping[str, Any] | str | bytes | Path, *, strict: bool = True
) -> list[Event]:
    """Import OTLP JSON, rejecting malformed data or skipping it when permissive."""
    document = _read_json(source)
    try:
        spans = _all_spans(document)
    except (TypeError, ValueError):
        if strict:
            raise
        return []
    span_to_event_id: dict[str, str] = {}
    prepared: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for span in spans:
        try:
            if not isinstance(span, Mapping):
                raise ValueError("span must be an object")
            attributes = _attributes(span.get("attributes", []))
            span_id = str(span["spanId"])
            event_id = str(attributes.get("agent_cassette.event.id") or f"otlp-{span_id}")
            span_to_event_id[span_id] = event_id
            prepared.append((span, attributes))
        except (KeyError, TypeError, ValueError):
            if strict:
                raise ValueError("invalid OTLP span identity") from None
    events: list[Event] = []
    for span, attributes in prepared:
        try:
            kind = str(attributes.get("openinference.span.kind", "UNKNOWN")).upper()
            stored_type = attributes.get("agent_cassette.event.type")
            if stored_type is not None:
                event_type = EventType(stored_type)
            elif span.get("status", {}).get("code") in (2, "STATUS_CODE_ERROR"):
                event_type = EventType.ERROR
            elif kind in _KIND_TO_EVENT:
                event_type = _KIND_TO_EVENT[kind]
            elif strict:
                raise ValueError(f"unsupported OpenInference span kind: {kind}")
            else:
                event_type = EventType.CUSTOM
            start = int(span["startTimeUnixNano"])
            end = int(span.get("endTimeUnixNano", start))
            if end < start:
                raise ValueError("span end precedes start")
            metadata_value = attributes.get("agent_cassette.metadata", "{}")
            metadata = (
                json.loads(metadata_value) if isinstance(metadata_value, str) else metadata_value
            )
            if not isinstance(metadata, dict):
                raise ValueError("agent_cassette.metadata must decode to an object")
            parent_span_id = span.get("parentSpanId")
            stored_parent_id = attributes.get("agent_cassette.parent_id")
            parent_id = (
                str(stored_parent_id)
                if stored_parent_id is not None
                else span_to_event_id.get(str(parent_span_id))
                if parent_span_id
                else None
            )
            if parent_span_id and parent_id is None and strict:
                raise ValueError(f"unknown parent span: {parent_span_id}")
            otlp_span_id = str(span["spanId"])
            span_id = str(attributes.get("agent_cassette.span_id", otlp_span_id))
            events.append(
                Event(
                    id=span_to_event_id[otlp_span_id],
                    timestamp=_ns_to_timestamp(start),
                    type=event_type,
                    name=str(span.get("name", "unnamed")),
                    input=_decode_json_attribute(attributes, "input.value"),
                    output=_decode_json_attribute(attributes, "output.value"),
                    metadata=metadata,
                    duration_ms=(end - start) / 1_000_000,
                    cost=float(attributes["llm.cost.total"])
                    if "llm.cost.total" in attributes
                    else None,
                    parent_id=parent_id,
                    span_id=span_id,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if strict:
                raise ValueError(
                    f"invalid OTLP span {span.get('spanId', '<unknown>')}: {error}"
                ) from error
    return events


def events_to_otlp(
    events: Iterable[Event], destination: str | Path | None = None
) -> dict[str, Any]:
    """Alias for :func:`export_otlp`."""
    return export_otlp(events, destination)


def otlp_to_events(
    source: Mapping[str, Any] | str | bytes | Path, *, strict: bool = True
) -> list[Event]:
    """Alias for :func:`import_otlp`."""
    return import_otlp(source, strict=strict)
