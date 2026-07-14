"""Cassette schema normalization and migration helpers."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from agent_cassette.deprecations import AgentCassetteDeprecationWarning
from agent_cassette.events import (
    EventMigration,
    migrate_event_dict,
    register_migration,
    unregister_migration,
)
from agent_cassette.storage import load_events, save_events


def migrate_cassette(source: str | Path, destination: str | Path | None = None) -> Path:
    """Upgrade and atomically rewrite a cassette using the current schema.

    Events at older schema versions are upgraded through the registered
    migration chain; events newer than this release are rejected.
    """
    source_path = Path(source)
    if destination is None:
        warn(
            "in-place cassette migration is deprecated; pass a separate destination path",
            AgentCassetteDeprecationWarning,
            stacklevel=2,
        )
    destination_path = Path(destination) if destination is not None else source_path
    events = load_events(source_path)
    save_events(destination_path, events)
    return destination_path


__all__ = [
    "EventMigration",
    "migrate_cassette",
    "migrate_event_dict",
    "register_migration",
    "unregister_migration",
]
