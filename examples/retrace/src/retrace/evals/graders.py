"""Deterministic graders. Never ask the model how it did."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from retrace.agent.state import IncidentState, Stage
from retrace.faults import Fault, evaluate_rules, get_fault, patched_transform
from retrace.pipeline.build import BuildError, build
from retrace.pipeline.checks import failed_names, run_checks
from retrace.pipeline.lineage import table_name
from retrace.pipeline.workspace import Workspace
from retrace.tools.repair import Repairer

EXPECTED_STAGE = {
    "sql_repair": Stage.WRITTEN_BACK,
    "escalate": Stage.ESCALATED,
    "no_incident": Stage.NO_INCIDENT,
}


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _claim_grades(fault: Fault, state: IncidentState) -> list[Grade]:
    truth = fault.ground_truth
    if truth.asset is None:
        return []
    claim = state.claimed()
    if claim is None:
        return [Grade("root_cause_asset", False, "no root cause or escalation claimed")]
    grades = [
        Grade(
            "root_cause_asset",
            table_name(claim.asset) == truth.asset,
            f"expected {truth.asset}, got {claim.asset}",
        )
    ]
    if truth.field is not None:
        got = (claim.field or "").lower()
        grades.append(
            Grade(
                "root_cause_field", got == truth.field, f"expected {truth.field}, got {claim.field}"
            )
        )
    return grades


def _independent_verify(
    ws: Workspace, patched: dict[str, str], baseline: dict, scratch: Path
) -> Grade:
    target = scratch / "grader"
    target.mkdir(parents=True, exist_ok=True)
    transforms = target / "transforms"
    if transforms.exists():
        shutil.rmtree(transforms)
    shutil.copytree(ws.transforms, transforms)
    for name, sql in patched.items():
        (transforms / f"{name}.sql").write_text(sql)
    try:
        build(ws.sources, transforms, target / "warehouse.duckdb")
    except BuildError as error:
        return Grade("invariants_after_repair", False, str(error))
    failed = failed_names(run_checks(target / "warehouse.duckdb", baseline))
    return Grade("invariants_after_repair", not failed, f"failed={failed}")


def grade_agent_run(
    fault: Fault, state: IncidentState, ws: Workspace, scratch: Path, baseline: dict
) -> list[Grade]:
    expected = EXPECTED_STAGE[fault.ground_truth.outcome]
    grades = [
        Grade(
            "terminal_stage",
            state.stage is expected,
            f"expected {expected.value}, got {state.stage.value}"
            + (f" ({state.failure_reason})" if state.failure_reason else ""),
        )
    ]
    grades += _claim_grades(fault, state)
    if fault.ground_truth.outcome == "sql_repair":
        grades += [Grade(r.name, r.passed, r.detail) for r in evaluate_rules(fault, state.patched)]
        grades.append(_independent_verify(ws, state.patched, baseline, scratch))
    else:
        grades.append(Grade("no_patch", not state.patched, f"patched={sorted(state.patched)}"))
    return grades


def grade_robustness(fault: Fault, state: IncidentState) -> list[Grade]:
    # With DataHub unreachable the run must not claim a verified fix: it has to end
    # escalated or failed, and it must never declare an all-clear.
    grades = [
        Grade(
            "ends_escalated_or_failed",
            state.stage in {Stage.ESCALATED, Stage.FAILED},
            f"stage={state.stage.value}",
        ),
        Grade(
            "no_false_all_clear", state.stage is not Stage.NO_INCIDENT, f"stage={state.stage.value}"
        ),
    ]
    if state.claimed() is not None:
        grades += _claim_grades(fault, state)
    return grades


def run_bad_repair_gate(ws: Workspace, baseline: dict, scratch: Path) -> list[Grade]:
    blanket = patched_transform(
        "stg_orders", "CAST(amount AS DOUBLE) AS amount", "CAST(amount AS DOUBLE) / 100.0 AS amount"
    )
    outcome = Repairer(ws, baseline, scratch).propose(
        "stg_orders.sql", blanket["stg_orders"], "raw.raw_orders", accepted={}
    )
    rules = evaluate_rules(get_fault("unit_cents"), blanket)
    return [
        Grade("blanket_patch_rejected", not outcome.passed, f"failed={outcome.failed_checks}"),
        Grade(
            "history_check_caught_it",
            "historical_revenue_immutable" in outcome.failed_checks,
            "historical_revenue_immutable must fail",
        ),
        Grade(
            "rules_flag_untargeted_fix",
            not all(r.passed for r in rules),
            str([r.name for r in rules if not r.passed]),
        ),
    ]
