"""Read and write JSONL cassette files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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


def save_events(path: str | Path, events: Iterable[Event]) -> None:
    """Write events atomically enough for local test workflows."""
    cassette_path = Path(path)
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    with cassette_path.open("w", encoding="utf-8") as cassette_file:
        for event in events:
            cassette_file.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")
