"""Composable assertions for agent trajectories."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_cassette.events import Event, EventType
from agent_cassette.integrations._serialization import serialize_recorded_value
from agent_cassette.matching import (
    DEFAULT_FUZZY_THRESHOLD,
    MatchMode,
    inputs_match,
    normalize_input,
    validate_fuzzy_threshold,
)
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


_ANY_INPUT = ...  # Ellipsis sentinel: omitted with_input matches any recorded input


def _is_tool_call_boundary(event: Event) -> bool:
    """One logical tool call: a ``TOOL_CALL`` event, or an ERROR whose recorded logical
    call type is exactly ``tool_call`` (a failed Phase A/B/LangChain/MCP call). A
    ``TOOL_RESULT``, a ``tool_result`` ERROR, a provider call, or an observational
    ``CUSTOM`` event is not a tool call."""
    if event.type is EventType.TOOL_CALL:
        return True
    if event.type is EventType.ERROR:
        internal = event.metadata.get("_agent_cassette")
        return isinstance(internal, dict) and internal.get("call_type") == "tool_call"
    return False


def _validate_count(label: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{label} must be a non-negative integer")


def _validate_ignore_paths(ignore_paths: tuple[str, ...]) -> None:
    if type(ignore_paths) is not tuple or not all(
        type(path) is str and path for path in ignore_paths
    ):
        raise ValueError("ignore_paths must be a tuple of nonempty strings")


def _tool_match_indexes(
    events: Sequence[Event],
    name: str,
    *,
    expected: Any,
    input_filtered: bool,
    match: MatchMode,
    ignore_paths: tuple[str, ...],
    fuzzy_threshold: float,
) -> list[int]:
    normalized_expected = normalize_input(expected, ignore_paths) if input_filtered else None
    indexes: list[int] = []
    for index, event in enumerate(events):
        if not _is_tool_call_boundary(event) or event.name != name:
            continue
        if input_filtered and not inputs_match(
            normalized_expected,
            normalize_input(event.input, ignore_paths),
            mode=match,
            fuzzy_threshold=fuzzy_threshold,
        ):
            continue
        indexes.append(index)
    return indexes


def tool_called(
    name: str,
    *,
    with_input: Any = _ANY_INPUT,
    times: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
    match: MatchMode = "exact",
    ignore_paths: tuple[str, ...] = (),
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> Check:
    """Require a logical tool-call boundary named ``name`` to have been recorded.

    A boundary is a ``TOOL_CALL`` event (or an ERROR whose logical ``call_type`` is
    ``tool_call``) matched by exact ``name`` -- one per invocation across ``wrap_tool``,
    MCP, the OpenAI Agents lifecycle, and the LangChain bridge; ``TOOL_RESULT`` is never
    counted. With no count arguments the requirement is at least one; ``times`` is exact;
    ``minimum``/``maximum`` are inclusive bounds (zero permitted). ``with_input`` (omitted =
    any input) is detached and strictly validated at creation, then matched with
    ``normalize_input``/``inputs_match`` exactly like the Replayer.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("tool_called requires a nonempty tool name")
    if match not in ("exact", "subset", "normalized", "fuzzy"):
        raise ValueError("match must be one of exact, subset, normalized, fuzzy")
    validate_fuzzy_threshold(fuzzy_threshold)
    _validate_ignore_paths(ignore_paths)
    _validate_count("times", times)
    _validate_count("minimum", minimum)
    _validate_count("maximum", maximum)
    if times is not None and (minimum is not None or maximum is not None):
        raise ValueError("times cannot be combined with minimum or maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    input_filtered = with_input is not _ANY_INPUT
    expected = serialize_recorded_value(with_input) if input_filtered else None

    def check(events: Sequence[Event]) -> AssertionResult:
        indexes = _tool_match_indexes(
            events,
            name,
            expected=expected,
            input_filtered=input_filtered,
            match=match,
            ignore_paths=ignore_paths,
            fuzzy_threshold=fuzzy_threshold,
        )
        actual = len(indexes)
        if times is not None:
            passed = actual == times
            requirement = f"exactly {times}"
        elif minimum is None and maximum is None:
            passed = actual >= 1
            requirement = "at least 1"
        else:
            low = minimum if minimum is not None else 0
            passed = actual >= low and (maximum is None or actual <= maximum)
            requirement = f"minimum {low}" + (f", maximum {maximum}" if maximum is not None else "")
        return _result(
            "tool_called",
            passed,
            f"tool {name!r}: expected {requirement} call(s); observed {actual}",
            tool_name=name,
            matching_indexes=indexes,
            actual=actual,
            times=times,
            minimum=minimum,
            maximum=maximum,
            input_filtered=input_filtered,
            match=match,
            ignore_paths=list(ignore_paths),
        )

    return check


def tool_not_called(
    name: str,
    *,
    with_input: Any = _ANY_INPUT,
    match: MatchMode = "exact",
    ignore_paths: tuple[str, ...] = (),
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> Check:
    """Require zero recorded tool-call boundaries named ``name`` (optionally by input).

    Same boundary and input-matching rules as :func:`tool_called`; passes only when the
    number of matches is exactly zero.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("tool_not_called requires a nonempty tool name")
    if match not in ("exact", "subset", "normalized", "fuzzy"):
        raise ValueError("match must be one of exact, subset, normalized, fuzzy")
    validate_fuzzy_threshold(fuzzy_threshold)
    _validate_ignore_paths(ignore_paths)
    input_filtered = with_input is not _ANY_INPUT
    expected = serialize_recorded_value(with_input) if input_filtered else None

    def check(events: Sequence[Event]) -> AssertionResult:
        indexes = _tool_match_indexes(
            events,
            name,
            expected=expected,
            input_filtered=input_filtered,
            match=match,
            ignore_paths=ignore_paths,
            fuzzy_threshold=fuzzy_threshold,
        )
        actual = len(indexes)
        return _result(
            "tool_not_called",
            actual == 0,
            f"tool {name!r}: expected 0 call(s); observed {actual}",
            tool_name=name,
            matching_indexes=indexes,
            actual=actual,
            input_filtered=input_filtered,
            match=match,
            ignore_paths=list(ignore_paths),
        )

    return check
