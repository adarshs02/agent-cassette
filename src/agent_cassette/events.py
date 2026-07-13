"""Versioned event types used by cassette files."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1

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
        raise ValueError(f"Invalid cassette schema version: {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Cassette schema version {version} is newer than this release "
            f"(supports up to {SCHEMA_VERSION})"
        )
    while version < SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"No migration registered from cassette schema version {version}")
        current = dict(migration(dict(current)))
        migrated_version = current.get("schema_version", version)
        if migrated_version != version + 1:
            raise ValueError(
                f"Migration from version {version} produced version "
                f"{migrated_version!r} instead of {version + 1}"
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
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event dictionary."""
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Build an event, migrating older schema versions when possible."""
        values = migrate_event_dict(data)
        values["type"] = EventType(values["type"])
        return cls(**values)
