"""Deterministic large-cassette read/write benchmark.

Timing fields are measurements, not release gates. Stable count, byte size, and
SHA-256 fields let CI verify that benchmark data generation is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from agent_cassette import Event, EventType
from agent_cassette.storage import load_events, save_events

REPORT_SCHEMA_VERSION = 1


def _events(count: int) -> list[Event]:
    return [
        Event(
            id=f"event-{index:08d}",
            timestamp="2025-01-01T00:00:00+00:00",
            type=EventType.MODEL_CALL,
            name="benchmark.model",
            input={"index": index, "prompt": f"question-{index:08d}"},
            output={"answer": f"answer-{index:08d}"},
            metadata={"benchmark": True, "partition": index % 16},
            duration_ms=(index % 100) / 10,
            cost=(index % 10) / 100_000,
        )
        for index in range(count)
    ]


def run_benchmark(output: Path, event_count: int) -> dict[str, object]:
    """Write and read deterministic events, returning a versioned JSON report."""
    events = _events(event_count)

    started = perf_counter()
    save_events(output, events)
    write_seconds = perf_counter() - started

    started = perf_counter()
    loaded = load_events(output)
    read_seconds = perf_counter() - started
    if len(loaded) != event_count:
        raise RuntimeError("large-cassette benchmark loaded an unexpected event count")

    cassette_bytes = output.read_bytes()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": "large-cassette",
        "parameters": {"event_count": event_count},
        "result": {
            "event_count": len(loaded),
            "cassette_bytes": len(cassette_bytes),
            "cassette_sha256": hashlib.sha256(cassette_bytes).hexdigest(),
            "write_seconds": round(write_seconds, 6),
            "read_seconds": round(read_seconds, 6),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.events < 1:
        parser.error("--events must be a positive integer")

    report = run_benchmark(arguments.output, arguments.events)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
