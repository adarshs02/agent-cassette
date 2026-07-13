"""Structured trajectory comparison for cassette files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_cassette.events import Event
from agent_cassette.storage import load_events


@dataclass(slots=True)
class EventDifference:
    """Fields changed at one trajectory step."""

    index: int
    fields: dict[str, dict[str, Any]]


@dataclass(slots=True)
class DiffReport:
    """Machine- and human-readable cassette comparison."""

    baseline_steps: int
    candidate_steps: int
    first_divergence: int | None
    differences: list[EventDifference] = field(default_factory=list)
    telemetry_differences: list[EventDifference] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """Whether both trajectories are behaviorally identical."""
        return not self.differences and self.baseline_steps == self.candidate_steps

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "identical": self.identical,
            "baseline_steps": self.baseline_steps,
            "candidate_steps": self.candidate_steps,
            "first_divergence": self.first_divergence,
            "differences": [asdict(difference) for difference in self.differences],
            "telemetry_differences": [
                asdict(difference) for difference in self.telemetry_differences
            ],
        }

    def to_text(self) -> str:
        """Render a concise terminal report."""
        if self.identical:
            return f"Trajectories are identical ({self.baseline_steps} events)."
        lines = [
            f"First divergence: step {self.first_divergence}",
            f"Steps: {self.baseline_steps} baseline, {self.candidate_steps} candidate",
        ]
        if self.differences:
            first = self.differences[0]
            for field_name, change in first.fields.items():
                lines.append(f"  {field_name}: {change['baseline']!r} -> {change['candidate']!r}")
        elif self.baseline_steps != self.candidate_steps:
            lines.append("  trajectory length changed")
        return "\n".join(lines)


_BEHAVIOR_FIELDS = ("type", "name", "input", "output", "cost", "metadata")
_TELEMETRY_FIELDS = ("duration_ms",)


def _event_changes(
    baseline: Event, candidate: Event, fields: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    baseline_data = baseline.to_dict()
    candidate_data = candidate.to_dict()
    for field_name in fields:
        if baseline_data[field_name] != candidate_data[field_name]:
            changes[field_name] = {
                "baseline": baseline_data[field_name],
                "candidate": candidate_data[field_name],
            }
    return changes


def compare_cassettes(baseline_path: str | Path, candidate_path: str | Path) -> DiffReport:
    """Compare two trajectories and identify their first behavioral divergence."""
    baseline = load_events(baseline_path)
    candidate = load_events(candidate_path)
    differences: list[EventDifference] = []
    telemetry_differences: list[EventDifference] = []
    for index, (baseline_event, candidate_event) in enumerate(
        zip(baseline, candidate, strict=False), start=1
    ):
        changes = _event_changes(baseline_event, candidate_event, _BEHAVIOR_FIELDS)
        if changes:
            differences.append(EventDifference(index=index, fields=changes))
        telemetry_changes = _event_changes(baseline_event, candidate_event, _TELEMETRY_FIELDS)
        if telemetry_changes:
            telemetry_differences.append(EventDifference(index=index, fields=telemetry_changes))

    first_divergence = differences[0].index if differences else None
    if first_divergence is None and len(baseline) != len(candidate):
        first_divergence = min(len(baseline), len(candidate)) + 1
    return DiffReport(
        baseline_steps=len(baseline),
        candidate_steps=len(candidate),
        first_divergence=first_divergence,
        differences=differences,
        telemetry_differences=telemetry_differences,
    )
