"""Legal stage transitions."""

from __future__ import annotations

from retrace.agent.evidence import GateError
from retrace.agent.state import IncidentState, Stage

ALLOWED: dict[Stage, frozenset[Stage]] = {
    Stage.INVESTIGATING: frozenset(
        {Stage.ROOT_CAUSE_CONFIRMED, Stage.NO_INCIDENT, Stage.ESCALATED}
    ),
    Stage.ROOT_CAUSE_CONFIRMED: frozenset({Stage.REPAIRING, Stage.ESCALATED}),
    Stage.REPAIRING: frozenset({Stage.VERIFIED, Stage.ESCALATED}),
    Stage.VERIFIED: frozenset({Stage.WRITTEN_BACK}),
    Stage.WRITTEN_BACK: frozenset(),
    Stage.NO_INCIDENT: frozenset(),
    Stage.ESCALATED: frozenset(),
    Stage.FAILED: frozenset(),
}


def advance(state: IncidentState, target: Stage) -> None:
    if target is state.stage:
        return
    if target not in ALLOWED[state.stage]:
        raise GateError(f"cannot move from {state.stage.value} to {target.value}")
    state.stage = target


def fail(state: IncidentState, reason: str) -> None:
    state.stage = Stage.FAILED
    state.failure_reason = reason
