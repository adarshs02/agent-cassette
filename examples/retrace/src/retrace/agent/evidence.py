"""Evidence store and the citation gates that keep the agent honest."""

from __future__ import annotations

import json
from typing import Any

from retrace.agent.state import EvidenceItem, IncidentState, Source
from retrace.pipeline.checks import BAND
from retrace.pipeline.lineage import table_name


class GateError(ValueError):
    """A workflow step was attempted without the evidence it requires."""


class EvidenceStore:
    def __init__(self, state: IncidentState) -> None:
        self._state = state

    def add(self, source: Source, kind: str, summary: str, payload: Any) -> EvidenceItem:
        item = EvidenceItem(
            id=f"ev_{len(self._state.evidence) + 1:03d}",
            source=source,
            kind=kind,
            summary=summary,
            payload=payload,
        )
        self._state.evidence.append(item)
        return item

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return next((e for e in self._state.evidence if e.id == evidence_id), None)

    def all(self) -> list[EvidenceItem]:
        return list(self._state.evidence)


def _resolve(store: EvidenceStore, evidence_ids: list[str]) -> list[EvidenceItem]:
    missing = [i for i in evidence_ids if store.get(i) is None]
    if missing:
        raise GateError(f"unknown evidence ids: {missing}")
    return [item for i in evidence_ids if (item := store.get(i)) is not None]


def check_claim_evidence(store: EvidenceStore, asset: str, evidence_ids: list[str]) -> None:
    items = _resolve(store, evidence_ids)
    if not any(i.source == "datahub" for i in items):
        raise GateError("cite at least one DataHub evidence item (metadata or lineage)")
    if not any(i.source == "warehouse" for i in items):
        raise GateError("cite at least one warehouse data evidence item (profile, sql, baseline)")
    name = table_name(asset)
    lineage = [e for e in store.all() if e.source == "datahub" and e.kind == "lineage"]
    if not any(name in json.dumps(e.payload, default=str) for e in lineage):
        raise GateError(
            f"{name} does not appear in any DataHub lineage evidence; "
            "trace lineage upstream from the KPI first"
        )


def check_no_incident_evidence(store: EvidenceStore, evidence_ids: list[str]) -> None:
    items = _resolve(store, evidence_ids)
    calm = [
        i
        for i in items
        if i.kind == "baseline_comparison"
        and isinstance(i.payload, dict)
        and i.payload.get("max_abs_deviation") is not None
        and i.payload["max_abs_deviation"] <= BAND
    ]
    if not calm:
        raise GateError(
            f"cite a baseline comparison showing max deviation <= {BAND} to declare no incident"
        )
