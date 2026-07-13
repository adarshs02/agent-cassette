"""Read and write JSONL cassette files."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from agent_cassette.events import Event


def load_events(path: str | Path) -> list[Event]:
    """Load all events from a cassette file."""
    cassette_path = Path(path)
    if not cassette_path.exists():
        raise FileNotFoundError(f"Cassette does not exist: {cassette_path}")

    events: list[Event] = []
    with cassette_path.open(encoding="utf-8") as cassette_file:
        for line_number, line in enumerate(cassette_file, start=1):
            if not line.strip():
                continue
            try:
                events.append(Event.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid cassette event at {cassette_path}:{line_number}: {error}"
                ) from error
    return events


def append_event(path: str | Path, event: Event) -> None:
    """Append and flush one event so completed calls survive process failure."""
    cassette_path = Path(path)
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    with cassette_path.open("a", encoding="utf-8") as cassette_file:
        cassette_file.write(_event_line(event))
        cassette_file.flush()
        os.fsync(cassette_file.fileno())


def save_events(path: str | Path, events: Iterable[Event]) -> None:
    """Atomically replace a cassette with the supplied events."""
    cassette_path = Path(path)
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cassette_path.parent,
            prefix=f".{cassette_path.name}.",
            delete=False,
        ) as cassette_file:
            temporary_path = Path(cassette_file.name)
            for event in events:
                cassette_file.write(_event_line(event))
            cassette_file.flush()
            os.fsync(cassette_file.fileno())
        temporary_path.replace(cassette_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _event_line(event: Event) -> str:
    return json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n"
