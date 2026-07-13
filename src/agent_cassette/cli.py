"""Command-line interface for inspecting and comparing trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agent_cassette.diff import compare_cassettes
from agent_cassette.storage import load_events


def _inspect(path: Path, *, as_json: bool) -> int:
    events = load_events(path)
    summary = {
        "path": str(path),
        "events": len(events),
        "types": {},
        "duration_ms": sum(event.duration_ms or 0 for event in events),
        "cost": sum(event.cost or 0 for event in events),
    }
    for event in events:
        summary["types"][event.type.value] = summary["types"].get(event.type.value, 0) + 1
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Cassette: {path}")
        print(f"Events: {summary['events']}")
        for event_type, count in sorted(summary["types"].items()):
            print(f"  {event_type}: {count}")
        print(f"Duration: {summary['duration_ms']:.2f} ms")
        print(f"Cost: ${summary['cost']:.6f}")
    return 0


def _diff(baseline: Path, candidate: Path, *, as_json: bool) -> int:
    report = compare_cassettes(baseline, candidate)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if as_json else report.to_text())
    return 0 if report.identical else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(prog="agent-cassette", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="summarize a cassette")
    inspect_parser.add_argument("cassette", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    diff_parser = subparsers.add_parser("diff", help="compare two trajectories")
    diff_parser.add_argument("baseline", type=Path)
    diff_parser.add_argument("candidate", type=Path)
    diff_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the Agent Cassette CLI."""
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "inspect":
        return _inspect(parsed.cassette, as_json=parsed.as_json)
    return _diff(parsed.baseline, parsed.candidate, as_json=parsed.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
