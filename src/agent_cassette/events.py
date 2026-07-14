"""Versioned event types used by cassette files."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agent_cassette.json_codec import copy_json_value, validate_json_value

SCHEMA_VERSION = 1
_EVENT_FIELDS = frozenset(
    {
        "id",
        "timestamp",
        "type",
        "name",
        "input",
        "output",
        "metadata",
        "duration_ms",
        "cost",
        "parent_id",
        "span_id",
        "schema_version",
    }
)
_REQUIRED_EVENT_FIELDS = frozenset({"id", "timestamp", "type", "name"})

EventMigration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[int, EventMigration] = {}


def register_migration(from_version: int, migration: EventMigration) -> None:
    """Register an upgrade for event dictionaries at ``from_version``.

    A migration receives an event dictionary at ``from_version`` and must
    return a dictionary at ``from_version + 1``. Registering the same source
    version twice is rejected to keep the chain unambiguous.
    """
    if isinstance(from_version, bool) or not isinstance(from_version, int) or from_version < 1:
        raise ValueError("migration source version must be a positive integer")
    if from_version in _MIGRATIONS:
        raise ValueError(f"a migration from version {from_version} is already registered")
    _MIGRATIONS[from_version] = migration


def unregister_migration(from_version: int) -> EventMigration:
    """Remove and return a previously registered migration."""
    try:
        return _MIGRATIONS.pop(from_version)
    except KeyError as error:
        raise KeyError(f"no migration registered from version {from_version}") from error


def migrate_event_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an event dictionary to the current schema version.

    Missing ``schema_version`` values default to 1. Versions newer than this
    release and versions with no registered migration path are rejected.
    """
    current = dict(data)
    version = current.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("Invalid cassette schema version")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Cassette schema version {version} is newer than this release "
            f"(supports up to {SCHEMA_VERSION})"
        )
    while version < SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"No migration registered from cassette schema version {version}")
        migrated = migration(dict(current))
        if not isinstance(migrated, dict):
            raise ValueError(f"Migration from version {version} must produce a dictionary")
        current = dict(migrated)
        migrated_version = current.get("schema_version", version)
        if migrated_version != version + 1:
            raise ValueError(
                f"Migration from version {version} produced an invalid version "
                f"instead of {version + 1}"
            )
        version = migrated_version
    current["schema_version"] = SCHEMA_VERSION
    return current


class EventType(str, Enum):
    """Built-in event categories."""

    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass(slots=True)
class Event:
    """One observable step in an agent execution."""

    id: str
    timestamp: str
    type: EventType
    name: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    cost: float | None = None
    parent_id: str | None = None
    span_id: str | None = None
    schema_version: int = field(default_factory=lambda: SCHEMA_VERSION)

    def __post_init__(self) -> None:
        _validate_event(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event dictionary."""
        _validate_event(self)
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "name": self.name,
            "input": copy_json_value(self.input),
            "output": copy_json_value(self.output),
            "metadata": copy_json_value(self.metadata),
            "duration_ms": self.duration_ms,
            "cost": self.cost,
            "parent_id": self.parent_id,
            "span_id": self.span_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Build an event, migrating older schema versions when possible."""
        if not isinstance(data, dict):
            raise TypeError("cassette event must be a JSON object")
        values = migrate_event_dict(data)
        if any(not isinstance(key, str) for key in values):
            raise TypeError("cassette event field names must be strings")
        if set(values) - _EVENT_FIELDS:
            raise ValueError("cassette event contains unknown fields")
        missing = _REQUIRED_EVENT_FIELDS - set(values)
        if missing:
            raise ValueError(
                "cassette event is missing required fields: " + ", ".join(sorted(missing))
            )
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cassette event migration did not reach the supported schema")
        try:
            values["type"] = EventType(values["type"])
        except (TypeError, ValueError) as error:
            raise ValueError("cassette event type is not supported") from error
        return cls(**values)


def _validate_event(event: Event) -> None:
    _require_nonempty_string("id", event.id)
    _require_aware_timestamp(event.timestamp)
    if not isinstance(event.type, EventType):
        raise TypeError("event type must be an EventType")
    _require_nonempty_string("name", event.name)
    if not isinstance(event.metadata, dict):
        raise TypeError("event metadata must be a dictionary")
    _require_optional_nonnegative_number("duration_ms", event.duration_ms)
    _require_optional_nonnegative_number("cost", event.cost)
    _require_optional_string("parent_id", event.parent_id)
    _require_optional_string("span_id", event.span_id)
    if (
        isinstance(event.schema_version, bool)
        or not isinstance(event.schema_version, int)
        or event.schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"event schema_version must equal supported version {SCHEMA_VERSION}")
    validate_json_value(event.input)
    validate_json_value(event.output)
    validate_json_value(event.metadata)


def _require_nonempty_string(field_name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"event {field_name} must be a nonempty string")
    if not value:
        raise ValueError(f"event {field_name} must be a nonempty string")


def _require_optional_string(field_name: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"event {field_name} must be a string or None")


def _require_optional_nonnegative_number(field_name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"event {field_name} must be a number or None")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0:
        raise ValueError(f"event {field_name} must be finite and nonnegative")


def _require_aware_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("event timestamp must be a string")
    normalized = f"{value[:-1]}+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("event timestamp must be a valid ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event timestamp must include a timezone offset")
