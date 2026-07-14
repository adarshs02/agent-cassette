"""Dependency-free OTLP JSON import and export with OpenInference conventions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.json_codec import (
    StrictJSONError,
    strict_json_loads,
    validate_json_value,
)

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
    validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError("non-finite OTLP attributes are not allowed")
        encoded = {"doubleValue": value}
    elif isinstance(value, str):
        encoded = {"stringValue": value}
    else:
        raise StrictJSONError(
            f"unsupported OTLP attribute type: {type(value).__module__}.{type(value).__qualname__}"
        )
    return {"key": key, "value": encoded}


def _timestamp_to_ns(timestamp: str) -> int:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1_000_000_000)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("timestamp must be a valid ISO 8601 datetime") from error


def _ns_to_timestamp(value: Any) -> str:
    nanoseconds = _integer(value, "timestamp")
    try:
        moment = datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=timezone.utc)
        return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("timestamp is outside the supported datetime range") from error


def _decode_json_text(value: str) -> Any:
    try:
        decoded = strict_json_loads(value)
        validate_json_value(decoded)
        return decoded
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid OTLP JSON at line {error.lineno}, column {error.colno}"
        ) from error
    except StrictJSONError as error:
        raise ValueError(f"invalid OTLP JSON: {error}") from error
    except RecursionError as error:
        raise ValueError("invalid OTLP JSON: maximum parser depth exceeded") from error


def _decode_utf8(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("OTLP JSON must be valid UTF-8") from error


def _read_json(source: Mapping[str, Any] | str | bytes | Path) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        value: Any = source
        validate_json_value(value)
    elif isinstance(source, bytes):
        value = _decode_json_text(_decode_utf8(source))
    elif isinstance(source, Path):
        value = _decode_json_text(_decode_utf8(source.read_bytes()))
    elif isinstance(source, str):
        stripped = source.lstrip()
        value = (
            _decode_json_text(source)
            if stripped.startswith(("{", "["))
            else _decode_json_text(_decode_utf8(Path(source).read_bytes()))
        )
    else:
        raise TypeError("OTLP source must be a mapping, JSON text, bytes, or path")
    if not isinstance(value, Mapping):
        raise ValueError("OTLP JSON root must be an object")
    return value


def _write_json(destination: str | Path, value: Mapping[str, Any]) -> None:
    validate_json_value(value)
    serialized = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
            if key == "stringValue":
                return _string(value, "string attribute")
            if key == "boolValue":
                if not isinstance(value, bool):
                    raise ValueError("boolean attribute must be true or false")
                return value
            if key == "intValue":
                return _integer(value, "integer attribute")
            return _finite_number(value, "double attribute")
    if "arrayValue" in container:
        array_value = container["arrayValue"]
        if not isinstance(array_value, Mapping):
            raise ValueError("arrayValue must be an object")
        values = array_value.get("values", [])
        if not isinstance(values, list):
            raise ValueError("arrayValue values must be a list")
        return [_decode_value(value) for value in values]
    if "kvlistValue" in container:
        kvlist_value = container["kvlistValue"]
        if not isinstance(kvlist_value, Mapping):
            raise ValueError("kvlistValue must be an object")
        return _attributes(kvlist_value.get("values", []))
    raise ValueError("unsupported OTLP attribute value")


def _attributes(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        raise ValueError("attributes must be a list")
    result: dict[str, Any] = {}
    for attribute in values:
        if not isinstance(attribute, Mapping) or "key" not in attribute or "value" not in attribute:
            raise ValueError("invalid OTLP attribute")
        result[_string(attribute["key"], "attribute key")] = _decode_value(attribute["value"])
    return result


def _decode_json_attribute(attributes: Mapping[str, Any], key: str) -> Any:
    value = attributes.get(key)
    if value is None:
        return None
    mime_type = attributes.get(key.replace(".value", ".mime_type"))
    if mime_type == "application/json" or isinstance(value, str):
        try:
            decoded = strict_json_loads(value)
            validate_json_value(decoded)
            return decoded
        except json.JSONDecodeError:
            return value
        except (StrictJSONError, RecursionError) as error:
            raise ValueError("invalid embedded OTLP JSON") from error
    return value


def _all_spans(document: Mapping[str, Any], *, strict: bool) -> list[Mapping[str, Any]]:
    spans: list[Mapping[str, Any]] = []
    resources = document.get("resourceSpans", [])
    if not isinstance(resources, list):
        if strict:
            raise ValueError("resourceSpans must be a list")
        return spans
    for resource in resources:
        if not isinstance(resource, Mapping):
            if strict:
                raise ValueError("resource span must be an object")
            continue
        scopes = resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", []))
        if not isinstance(scopes, list):
            if strict:
                raise ValueError("scopeSpans must be a list")
            continue
        for scope in scopes:
            if not isinstance(scope, Mapping):
                if strict:
                    raise ValueError("scope span must be an object")
                continue
            scope_spans = scope.get("spans", [])
            if not isinstance(scope_spans, list):
                if strict:
                    raise ValueError("scope span must contain a spans list")
                continue
            for span in scope_spans:
                if not isinstance(span, Mapping):
                    if strict:
                        raise ValueError("span must be an object")
                    continue
                spans.append(span)
    return spans


def import_otlp(
    source: Mapping[str, Any] | str | bytes | Path, *, strict: bool = True
) -> list[Event]:
    """Import OTLP JSON, rejecting malformed data or skipping it when permissive."""
    document = _read_json(source)
    spans = _all_spans(document, strict=strict)
    span_to_event_id: dict[str, str] = {}
    prepared: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for span in spans:
        try:
            if not isinstance(span, Mapping):
                raise ValueError("span must be an object")
            attributes = _attributes(span.get("attributes", []))
            span_id = _string(span["spanId"], "span identity")
            stored_event_id = attributes.get("agent_cassette.event.id")
            event_id = (
                _string(stored_event_id, "event identity")
                if stored_event_id is not None
                else f"otlp-{span_id}"
            )
            span_to_event_id[span_id] = event_id
            prepared.append((span, attributes))
        except (KeyError, TypeError, ValueError):
            if strict:
                raise ValueError("invalid OTLP span identity") from None
    events: list[Event] = []
    for span_index, (span, attributes) in enumerate(prepared):
        try:
            kind = _string(
                attributes.get("openinference.span.kind", "UNKNOWN"), "span kind"
            ).upper()
            status = span.get("status", {})
            if not isinstance(status, Mapping):
                raise ValueError("span status must be an object")
            stored_type = attributes.get("agent_cassette.event.type")
            if stored_type is not None:
                try:
                    event_type = EventType(_string(stored_type, "event type"))
                except ValueError as error:
                    raise ValueError("unsupported event type") from error
            elif status.get("code") in (2, "STATUS_CODE_ERROR"):
                event_type = EventType.ERROR
            elif kind in _KIND_TO_EVENT:
                event_type = _KIND_TO_EVENT[kind]
            elif strict:
                raise ValueError("unsupported OpenInference span kind")
            else:
                event_type = EventType.CUSTOM
            start = _integer(span["startTimeUnixNano"], "span start time")
            end = _integer(span.get("endTimeUnixNano", start), "span end time")
            if end < start:
                raise ValueError("span end precedes start")
            duration_ms = _finite_number(end - start, "span duration nanoseconds") / 1_000_000
            metadata_value = attributes.get("agent_cassette.metadata", "{}")
            metadata = (
                _decode_json_text(metadata_value)
                if isinstance(metadata_value, str)
                else metadata_value
            )
            if not isinstance(metadata, dict):
                raise ValueError("agent_cassette.metadata must decode to an object")
            parent_span_id_value = span.get("parentSpanId")
            parent_span_id = (
                _string(parent_span_id_value, "parent span identity")
                if parent_span_id_value is not None
                else None
            )
            stored_parent_id = attributes.get("agent_cassette.parent_id")
            parent_id = (
                _string(stored_parent_id, "parent identity")
                if stored_parent_id is not None
                else span_to_event_id.get(parent_span_id)
                if parent_span_id is not None
                else None
            )
            if parent_span_id is not None and parent_id is None and strict:
                raise ValueError("unknown parent span")
            otlp_span_id = _string(span["spanId"], "span identity")
            span_id = _string(
                attributes.get("agent_cassette.span_id", otlp_span_id), "cassette span identity"
            )
            events.append(
                Event(
                    id=span_to_event_id[otlp_span_id],
                    timestamp=_ns_to_timestamp(start),
                    type=event_type,
                    name=_string(span.get("name", "unnamed"), "span name"),
                    input=_decode_json_attribute(attributes, "input.value"),
                    output=_decode_json_attribute(attributes, "output.value"),
                    metadata=metadata,
                    duration_ms=duration_ms,
                    cost=_finite_number(attributes["llm.cost.total"], "span cost")
                    if "llm.cost.total" in attributes
                    else None,
                    parent_id=parent_id,
                    span_id=span_id,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            if strict:
                reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                raise ValueError(f"invalid OTLP span at index {span_index}: {reason}") from error
    return events


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


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
