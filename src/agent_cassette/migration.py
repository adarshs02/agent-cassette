"""Cassette schema normalization and migration helpers."""

from __future__ import annotations

from pathlib import Path

from agent_cassette.events import SCHEMA_VERSION
from agent_cassette.storage import load_events, save_events


def migrate_cassette(source: str | Path, destination: str | Path | None = None) -> Path:
    """Validate and atomically rewrite a cassette using the current schema."""
    source_path = Path(source)
    destination_path = Path(destination) if destination is not None else source_path
    events = load_events(source_path)
    if any(event.schema_version != SCHEMA_VERSION for event in events):
        raise ValueError("cassette contains events newer than this Agent Cassette release")
    save_events(destination_path, events)
    return destination_path


__all__ = ["migrate_cassette"]
