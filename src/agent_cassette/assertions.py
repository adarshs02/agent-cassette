"""Composable assertions for agent trajectories."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.storage import load_events

Trajectory = Iterable[Event] | str | Path
Check = Callable[[Sequence[Event]], "AssertionResult"]


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """The result of evaluating one trajectory check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssertionReport:
    """Deterministic collection of trajectory assertion results."""

    results: tuple[AssertionResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }

    def to_text(self) -> str:
        status = "passed" if self.passed else "failed"
        lines = [f"Trajectory assertions {status}: {len(self.results)} check(s)"]
        lines.extend(
            f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.message}"
            for result in self.results
        )
        return "\n".join(lines)


def _events(trajectory: Trajectory) -> list[Event]:
    if isinstance(trajectory, (str, Path)):
        return load_events(trajectory)
    return list(trajectory)


def check_trajectory(trajectory: Trajectory, *checks: Check) -> AssertionReport:
    """Evaluate checks in caller-specified order without short-circuiting."""
    events = _events(trajectory)
    return AssertionReport(tuple(check(events) for check in checks))


def assert_trajectory(trajectory: Trajectory, *checks: Check) -> AssertionReport:
    """Evaluate checks and raise ``AssertionError`` with the full report on failure."""
    report = check_trajectory(trajectory, *checks)
    if not report.passed:
        raise AssertionError(report.to_text())
    return report


def _result(name: str, passed: bool, message: str, **details: Any) -> AssertionResult:
    return AssertionResult(name=name, passed=passed, message=message, details=details)


def no_errors() -> Check:
    """Require a trajectory with no error events."""

    def check(events: Sequence[Event]) -> AssertionResult:
        indexes = [index for index, event in enumerate(events) if event.type == EventType.ERROR]
        return _result(
            "no_errors",
            not indexes,
            "no error events" if not indexes else f"found {len(indexes)} error event(s)",
            error_indexes=indexes,
        )

    return check


def event_count(
    expected: int | None = None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    event_type: EventType | str | None = None,
) -> Check:
    """Require an exact or bounded number of optionally filtered events."""
    if expected is None and minimum is None and maximum is None:
        raise ValueError("event_count requires expected, minimum, or maximum")
    if expected is not None and (minimum is not None or maximum is not None):
        raise ValueError("expected cannot be combined with minimum or maximum")
    kind = EventType(event_type) if event_type is not None else None

    def check(events: Sequence[Event]) -> AssertionResult:
        actual = sum(1 for event in events if kind is None or event.type == kind)
        passed = (
            actual == expected
            if expected is not None
            else ((minimum is None or actual >= minimum) and (maximum is None or actual <= maximum))
        )
        requirement = (
            f"exactly {expected}" if expected is not None else f"between {minimum} and {maximum}"
        )
        return _result(
            "event_count",
            passed,
            f"expected {requirement}; observed {actual}",
            actual=actual,
            expected=expected,
            minimum=minimum,
            maximum=maximum,
            event_type=kind.value if kind else None,
        )

    return check


def contains_event(event_type: EventType | str | None = None, *, name: str | None = None) -> Check:
    """Require at least one event matching type and/or name."""
    if event_type is None and name is None:
        raise ValueError("contains_event requires event_type or name")
    kind = EventType(event_type) if event_type is not None else None

    def check(events: Sequence[Event]) -> AssertionResult:
        indexes = [
            index
            for index, event in enumerate(events)
            if (kind is None or event.type == kind) and (name is None or event.name == name)
        ]
        description = f"type={kind.value if kind else '*'}, name={name or '*'}"
        return _result(
            "contains_event",
            bool(indexes),
            f"found matching event ({description})"
            if indexes
            else f"missing event ({description})",
            matching_indexes=indexes,
            event_type=kind.value if kind else None,
            event_name=name,
        )

    return check


def event_sequence(*expected: EventType | str | tuple[EventType | str, str]) -> Check:
    """Require expected event descriptors to occur in order, allowing gaps."""
    descriptors = tuple(
        (EventType(item[0]), item[1]) if isinstance(item, tuple) else (EventType(item), None)
        for item in expected
    )

    def check(events: Sequence[Event]) -> AssertionResult:
        matched: list[int] = []
        next_index = 0
        for kind, name in descriptors:
            for index in range(next_index, len(events)):
                event = events[index]
                if event.type == kind and (name is None or event.name == name):
                    matched.append(index)
                    next_index = index + 1
                    break
            else:
                rendered = [(item_kind.value, item_name) for item_kind, item_name in descriptors]
                return _result(
                    "event_sequence",
                    False,
                    f"sequence stopped after {len(matched)} of {len(descriptors)} event(s)",
                    expected=rendered,
                    matching_indexes=matched,
                )
        return _result(
            "event_sequence",
            True,
            f"found all {len(descriptors)} event(s) in order",
            expected=[(kind.value, name) for kind, name in descriptors],
            matching_indexes=matched,
        )

    return check


def max_total_cost(limit: float) -> Check:
    """Require the sum of recorded event costs not to exceed a limit."""
    if limit < 0:
        raise ValueError("cost limit cannot be negative")

    def check(events: Sequence[Event]) -> AssertionResult:
        total = math.fsum(event.cost or 0.0 for event in events)
        passed = total <= limit or math.isclose(total, limit, rel_tol=1e-12, abs_tol=1e-15)
        return _result(
            "max_total_cost",
            passed,
            f"total cost {total:g} {'<=' if passed else '>'} {limit:g}",
            total=total,
            limit=limit,
        )

    return check


def max_total_duration_ms(limit: float) -> Check:
    """Require the sum of recorded event durations not to exceed a limit."""
    if limit < 0:
        raise ValueError("duration limit cannot be negative")

    def check(events: Sequence[Event]) -> AssertionResult:
        total = sum(event.duration_ms or 0.0 for event in events)
        return _result(
            "max_total_duration_ms",
            total <= limit,
            f"total duration {total:g}ms {'<=' if total <= limit else '>'} {limit:g}ms",
            total=total,
            limit=limit,
        )

    return check
