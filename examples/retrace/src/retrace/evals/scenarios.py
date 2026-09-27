"""Eval scenarios: one per fault, plus a deterministic gate and robustness forks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_cassette import EventType, InjectionRule, Raise, RateLimitError
from retrace.faults import fault_names

Kind = Literal["agent", "gate", "robustness"]


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: str
    kind: Kind
    injections: tuple[InjectionRule, ...] = ()
    base: str | None = None


def datahub_outage(from_call: int = 3, through: int = 60) -> tuple[InjectionRule, ...]:
    return tuple(
        InjectionRule(
            Raise(TimeoutError("DataHub MCP call timed out")),
            event_type=EventType.TOOL_CALL,
            occurrence=n,
        )
        for n in range(from_call, through + 1)
    )


SCENARIOS: list[Scenario] = [
    *(Scenario(name, name, "agent") for name in fault_names()),
    Scenario("bad_repair_rejected", "unit_cents", "gate"),
    Scenario(
        "datahub_timeout",
        "unit_cents",
        "robustness",
        datahub_outage(from_call=1),
        base="unit_cents",
    ),
    Scenario(
        "rate_limit_midrun",
        "unit_cents",
        "robustness",
        (
            InjectionRule(
                Raise(RateLimitError("rate limited")), event_type=EventType.MODEL_CALL, occurrence=4
            ),
        ),
        base="unit_cents",
    ),
]


def get_scenario(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario {name!r}; known: {[s.name for s in SCENARIOS]}")
