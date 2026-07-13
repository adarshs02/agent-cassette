from __future__ import annotations

import pytest

from agent_cassette.assertions import (
    assert_trajectory,
    check_trajectory,
    contains_event,
    event_count,
    event_sequence,
    max_total_cost,
    max_total_duration_ms,
    no_errors,
)
from agent_cassette.events import Event, EventType


def make_event(
    identifier: str,
    kind: EventType,
    name: str,
    *,
    cost: float | None = None,
    duration: float | None = None,
) -> Event:
    return Event(
        id=identifier,
        timestamp="2026-01-01T00:00:00Z",
        type=kind,
        name=name,
        cost=cost,
        duration_ms=duration,
    )


def test_check_trajectory_aggregates_all_checks_in_order():
    events = [
        make_event("1", EventType.MODEL_CALL, "plan", cost=0.2, duration=12),
        make_event("2", EventType.TOOL_CALL, "search", cost=0.1, duration=8),
        make_event("3", EventType.TOOL_RESULT, "search", duration=3),
    ]

    report = check_trajectory(
        events,
        no_errors(),
        event_count(3),
        contains_event(EventType.TOOL_CALL, name="search"),
        event_sequence(EventType.MODEL_CALL, (EventType.TOOL_CALL, "search")),
        max_total_cost(0.3),
        max_total_duration_ms(23),
    )

    assert report.passed
    assert [result.name for result in report.results] == [
        "no_errors",
        "event_count",
        "contains_event",
        "event_sequence",
        "max_total_cost",
        "max_total_duration_ms",
    ]
    assert report.to_dict()["passed"] is True


def test_assert_trajectory_reports_every_failure():
    events = [make_event("1", EventType.ERROR, "boom", cost=2, duration=50)]

    with pytest.raises(AssertionError) as raised:
        assert_trajectory(events, no_errors(), event_count(2), max_total_cost(1))

    message = str(raised.value)
    assert "FAIL no_errors" in message
    assert "FAIL event_count" in message
    assert "FAIL max_total_cost" in message


def test_checks_accept_cassette_path_and_filter_counts(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"id":"1","timestamp":"2026-01-01T00:00:00Z","type":"custom",'
        '"name":"one","schema_version":1}\n',
        encoding="utf-8",
    )

    report = check_trajectory(
        path,
        event_count(minimum=1, maximum=1, event_type="custom"),
        contains_event(name="one"),
    )

    assert report.passed


def test_check_factories_validate_invalid_configuration():
    with pytest.raises(ValueError, match="requires"):
        event_count()
    with pytest.raises(ValueError, match="combined"):
        event_count(1, minimum=1)
    with pytest.raises(ValueError, match="requires"):
        contains_event()
    with pytest.raises(ValueError, match="negative"):
        max_total_cost(-1)
    with pytest.raises(ValueError, match="negative"):
        max_total_duration_ms(-1)


def test_sequence_allows_gaps_but_preserves_order():
    events = [
        make_event("1", EventType.TOOL_CALL, "first"),
        make_event("2", EventType.CUSTOM, "gap"),
        make_event("3", EventType.MODEL_CALL, "second"),
    ]

    assert check_trajectory(
        events, event_sequence((EventType.TOOL_CALL, "first"), EventType.MODEL_CALL)
    ).passed
    assert not check_trajectory(
        events, event_sequence(EventType.MODEL_CALL, EventType.TOOL_CALL)
    ).passed
