"""Incident state carried through one agent run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Stage(str, Enum):
    INVESTIGATING = "INVESTIGATING"
    ROOT_CAUSE_CONFIRMED = "ROOT_CAUSE_CONFIRMED"
    REPAIRING = "REPAIRING"
    VERIFIED = "VERIFIED"
    WRITTEN_BACK = "WRITTEN_BACK"
    NO_INCIDENT = "NO_INCIDENT"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


TERMINAL = frozenset({Stage.WRITTEN_BACK, Stage.NO_INCIDENT, Stage.ESCALATED, Stage.FAILED})
FINISHABLE = frozenset({Stage.WRITTEN_BACK, Stage.NO_INCIDENT, Stage.ESCALATED, Stage.VERIFIED})

Source = Literal["datahub", "warehouse", "pipeline"]


@dataclass
class EvidenceItem:
    id: str
    source: Source
    kind: str
    summary: str
    payload: Any


@dataclass
class Claim:
    asset: str
    field: str | None
    summary: str
    evidence_ids: list[str]


@dataclass
class IncidentState:
    scenario: str
    report: str
    stage: Stage = Stage.INVESTIGATING
    evidence: list[EvidenceItem] = field(default_factory=list)
    root_cause: Claim | None = None
    escalation: Claim | None = None
    no_incident_summary: str | None = None
    repair_attempts: int = 0
    patched: dict[str, str] = field(default_factory=dict)
    verification: list[dict[str, Any]] | None = None
    written_back: bool = False
    final_report: str | None = None
    failure_reason: str | None = None

    def claimed(self) -> Claim | None:
        return self.root_cause or self.escalation

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data
