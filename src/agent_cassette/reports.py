"""Deterministic reports suitable for CI artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_cassette.assertions import AssertionReport
from agent_cassette.diff import DiffReport


@dataclass(slots=True)
class CIReport:
    """Aggregate named assertion and diff reports without process-global state."""

    assertions: dict[str, AssertionReport] = field(default_factory=dict)
    diffs: dict[str, DiffReport] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.assertions.values()) and all(
            report.identical for report in self.diffs.values()
        )

    def add_assertions(self, name: str, report: AssertionReport) -> None:
        self._require_new_name(name)
        self.assertions[name] = report

    def add_diff(self, name: str, report: DiffReport) -> None:
        self._require_new_name(name)
        self.diffs[name] = report

    def _require_new_name(self, name: str) -> None:
        if not name:
            raise ValueError("report name cannot be empty")
        if name in self.assertions or name in self.diffs:
            raise ValueError(f"duplicate report name: {name}")

    def to_dict(self) -> dict[str, Any]:
        """Return stable, name-sorted report data."""
        return {
            "passed": self.passed,
            "assertions": {
                name: self.assertions[name].to_dict() for name in sorted(self.assertions)
            },
            "diffs": {name: self.diffs[name].to_dict() for name in sorted(self.diffs)},
            "metadata": {name: self.metadata[name] for name in sorted(self.metadata)},
        }

    def to_json(self) -> str:
        """Serialize with stable key ordering and formatting."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"

    def write_json(self, path: str | Path) -> None:
        """Atomically replace a JSON report in its destination directory."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(self.to_json())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
