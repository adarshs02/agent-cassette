"""Cassette schema normalization and migration helpers."""

from __future__ import annotations

from pathlib import Path

from agent_cassette.events import (
    EventMigration,
    migrate_event_dict,
    register_migration,
    unregister_migration,
)
from agent_cassette.storage import _same_file, load_events, save_events


def migrate_cassette(source: str | Path, destination: str | Path) -> Path:
    """Upgrade and atomically rewrite a cassette to a separate destination.

    Events at older schema versions are upgraded through the registered
    migration chain; events newer than this release are rejected. The
    destination must differ from the source cassette.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if _same_file(source_path, destination_path):
        raise ValueError("migration destination must differ from source cassette")
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
