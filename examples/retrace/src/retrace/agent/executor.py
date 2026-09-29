"""Dispatch model tool calls to fact and workflow tools, enforcing gates."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import duckdb

from retrace.agent.evidence import (
    EvidenceStore,
    GateError,
    check_claim_evidence,
    check_no_incident_evidence,
)
from retrace.agent.stages import advance, fail
from retrace.agent.state import FINISHABLE, Claim, IncidentState, Stage
from retrace.datahub.catalog import dataset_urn
from retrace.pipeline.lineage import table_name
from retrace.tools.datahub import DataHubTools
from retrace.tools.repair import Repairer, RepairRejected, Transforms
from retrace.tools.warehouse import MAX_RESULT_CHARS, SqlRejected, SqlTimeout, Warehouse


def _clip(value: Any) -> Any:
    text = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    if len(text) <= MAX_RESULT_CHARS:
        return value
    return {"truncated": True, "preview": text[:MAX_RESULT_CHARS]}


def _is_error(value: Any) -> bool:
    return isinstance(value, dict) and "error" in value


# DataHub tools whose errors count toward the consecutive-outage stop below.
_DATAHUB_TOOL_NAMES = frozenset(
    {"datahub_search", "datahub_get_dataset", "datahub_lineage", "write_back"}
)
MAX_DATAHUB_ERRORS = 3


class ToolExecutor:
    def __init__(
        self,
        state: IncidentState,
        warehouse: Warehouse,
        transforms: Transforms,
        repairer: Repairer,
        datahub: DataHubTools,
        max_repair_attempts: int = 3,
    ) -> None:
        self.state = state
        self.store = EvidenceStore(state)
        self.warehouse = warehouse
        self.transforms = transforms
        self.repairer = repairer
        self.datahub = datahub
        self.max_repair_attempts = max_repair_attempts
        self.finished = False
        self._datahub_error_streak = 0
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "datahub_search": self.t_datahub_search,
            "datahub_get_dataset": self.t_datahub_get_dataset,
            "datahub_lineage": self.t_datahub_lineage,
            "get_metric_history": self.t_get_metric_history,
            "compare_to_baseline": self.t_compare_to_baseline,
            "profile_column": self.t_profile_column,
            "run_sql": self.t_run_sql,
            "list_transforms": self.t_list_transforms,
            "read_transform": self.t_read_transform,
            "run_checks": self.t_run_checks,
            "confirm_root_cause": self.t_confirm_root_cause,
            "escalate_upstream": self.t_escalate_upstream,
            "declare_no_incident": self.t_declare_no_incident,
            "propose_repair": self.t_propose_repair,
            "write_back": self.t_write_back,
            "finish": self.t_finish,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.finished:
            return {"error": "run already finished"}
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        if not isinstance(args, dict):
            return {"error": "tool input must be an object"}
        try:
            result = handler(**args)
        except TypeError as error:
            result = {"error": f"bad arguments for {name}: {error}"}
        except (GateError, RepairRejected, SqlRejected, SqlTimeout, ValueError) as error:
            result = {"error": str(error)}
        except (duckdb.Error, OSError) as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        return self._track_datahub_outage(name, result)

    def _track_datahub_outage(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Fail the run once a DataHub tool has errored MAX_DATAHUB_ERRORS times in a
        row (any successful DataHub call resets the streak); a hopeless outage would
        otherwise burn the whole turn budget without ever making progress."""
        if name not in _DATAHUB_TOOL_NAMES:
            return result
        if not _is_error(result):
            self._datahub_error_streak = 0
            return result
        self._datahub_error_streak += 1
        if self._datahub_error_streak < MAX_DATAHUB_ERRORS:
            return result
        reason = f"DataHub unavailable: {MAX_DATAHUB_ERRORS} consecutive errors"
        fail(self.state, reason)
        self.finished = True
        return {"error": reason}

    def _fact(self, source: str, kind: str, summary: str, result: Any) -> dict[str, Any]:
        if _is_error(result):
            return result
        item = self.store.add(source, kind, summary, result)  # type: ignore[arg-type]
        return {"evidence_id": item.id, "result": _clip(result)}

    @staticmethod
    def _need_str(**values: Any) -> None:
        for key, value in values.items():
            if not isinstance(value, str) or not value:
                raise TypeError(f"{key} must be a non-empty string")

    @staticmethod
    def _need_day_count(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not (1 <= value <= 90):
            raise ValueError(f"{name} must be between 1 and 90")
        return value

    def t_datahub_search(self, query: str) -> dict[str, Any]:
        self._need_str(query=query)
        return self._fact("datahub", "search", f"search {query!r}", self.datahub.search(query))

    def t_datahub_get_dataset(self, urn: str) -> dict[str, Any]:
        self._need_str(urn=urn)
        return self._fact(
            "datahub", "metadata", f"dataset {table_name(urn)}", self.datahub.get_dataset(urn)
        )

    def t_datahub_lineage(self, urn: str, direction: str, max_hops: int = 3) -> dict[str, Any]:
        self._need_str(urn=urn, direction=direction)
        return self._fact(
            "datahub",
            "lineage",
            f"{direction} lineage of {table_name(urn)}",
            self.datahub.lineage(urn, direction, max_hops),
        )

    def t_get_metric_history(self, days: int = 30) -> dict[str, Any]:
        days = self._need_day_count("days", days)
        return self._fact(
            "warehouse",
            "metric_history",
            "KPI and recent revenue",
            self.warehouse.metric_history(days),
        )

    def t_compare_to_baseline(self, last_days: int = 14) -> dict[str, Any]:
        last_days = self._need_day_count("last_days", last_days)
        return self._fact(
            "warehouse",
            "baseline_comparison",
            "revenue vs baseline",
            self.warehouse.compare_to_baseline(last_days),
        )

    def t_profile_column(
        self, table: str, column: str, group_by: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        self._need_str(table=table, column=column)
        return self._fact(
            "warehouse",
            "profile",
            f"profile {table}.{column}",
            self.warehouse.profile_column(table, column, group_by, since),
        )

    def t_run_sql(self, query: str) -> dict[str, Any]:
        self._need_str(query=query)
        return self._fact("warehouse", "sql", "ad-hoc query", self.warehouse.run_sql(query))

    def t_list_transforms(self) -> dict[str, Any]:
        return {"transforms": self.transforms.list()}

    def t_read_transform(self, name: str) -> dict[str, Any]:
        self._need_str(name=name)
        return self._fact("pipeline", "transform", name, self.transforms.read(name))

    def t_run_checks(self) -> dict[str, Any]:
        return self._fact("pipeline", "checks", "invariant suite", self.warehouse.run_checks())

    @staticmethod
    def _need_ids(evidence_ids: Any) -> list[str]:
        # A bare string must not be split into characters by list().
        if not isinstance(evidence_ids, list) or not all(isinstance(i, str) for i in evidence_ids):
            raise TypeError("evidence_ids must be a list of strings")
        return list(evidence_ids)

    def _claim(self, asset: str, field: str | None, summary: str, evidence_ids: list[str]) -> Claim:
        self._need_str(asset=asset, summary=summary)
        evidence_ids = self._need_ids(evidence_ids)
        check_claim_evidence(self.store, asset, evidence_ids)
        return Claim(table_name(asset), field, summary, list(evidence_ids))

    _ROOT_CAUSE_LOCKED_STAGES = frozenset(
        {Stage.VERIFIED, Stage.WRITTEN_BACK, Stage.ESCALATED, Stage.NO_INCIDENT, Stage.FAILED}
    )

    def t_confirm_root_cause(
        self, asset: str, summary: str, evidence_ids: list[str], field: str | None = None
    ) -> dict[str, Any]:
        # A root cause may be revised (while ROOT_CAUSE_CONFIRMED or REPAIRING) as
        # long as no repair has passed yet. Once a repair passes (state.patched is
        # non-empty) or the run has moved to a terminal/verified stage, it is locked.
        if self.state.patched:
            raise GateError("root cause already confirmed and repaired")
        if self.state.stage in self._ROOT_CAUSE_LOCKED_STAGES:
            raise GateError(f"cannot confirm a root cause in stage {self.state.stage.value}")
        revising = self.state.root_cause is not None
        claim = self._claim(asset, field, summary, evidence_ids)
        if revising:
            self.state.root_cause = claim
            return {"accepted": True, "revised": True, "stage": self.state.stage.value}
        advance(self.state, Stage.ROOT_CAUSE_CONFIRMED)
        self.state.root_cause = claim
        return {"accepted": True, "stage": self.state.stage.value}

    def t_escalate_upstream(
        self, asset: str, reason: str, evidence_ids: list[str], field: str | None = None
    ) -> dict[str, Any]:
        if self.state.escalation is not None:
            raise GateError("already escalated")
        claim = self._claim(asset, field, reason, evidence_ids)
        advance(self.state, Stage.ESCALATED)
        self.state.escalation = claim
        self.state.root_cause = None
        return {"accepted": True, "stage": self.state.stage.value}

    def t_declare_no_incident(self, summary: str, evidence_ids: list[str]) -> dict[str, Any]:
        self._need_str(summary=summary)
        check_no_incident_evidence(self.store, self._need_ids(evidence_ids))
        advance(self.state, Stage.NO_INCIDENT)
        self.state.no_incident_summary = summary
        return {"accepted": True, "stage": self.state.stage.value}

    def t_propose_repair(self, file: str, new_sql: str) -> dict[str, Any]:
        self._need_str(file=file, new_sql=new_sql)
        if self.state.stage not in (Stage.ROOT_CAUSE_CONFIRMED, Stage.REPAIRING):
            raise GateError("confirm a root cause before proposing a repair")
        assert self.state.root_cause is not None
        outcome = self.repairer.propose(
            file, new_sql, self.state.root_cause.asset, dict(self.state.patched)
        )
        advance(self.state, Stage.REPAIRING)
        self.state.repair_attempts = self.repairer.attempts
        if outcome.passed:
            self.state.patched = outcome.patched
            self.state.verification = outcome.results
            advance(self.state, Stage.VERIFIED)
            return {"passed": True, "diff": outcome.diff, "stage": self.state.stage.value}
        if self.repairer.attempts >= self.max_repair_attempts:
            fail(self.state, f"repair budget of {self.max_repair_attempts} attempts exhausted")
            self.finished = True
            return {"error": f"repair budget exhausted; last failures: {outcome.failed_checks}"}
        return {
            "passed": False,
            "failed_checks": outcome.failed_checks,
            "build_error": outcome.error,
            "diff": outcome.diff,
            "attempts_left": self.max_repair_attempts - self.repairer.attempts,
        }

    def t_write_back(self, summary: str) -> dict[str, Any]:
        self._need_str(summary=summary)
        if self.state.written_back:
            raise GateError("already written back")
        claim = self.state.claimed()
        if self.state.stage not in (Stage.VERIFIED, Stage.ESCALATED) or claim is None:
            raise GateError("write_back is allowed after a verified repair or an escalation")
        result = self.datahub.write_back(
            dataset_urn(claim.asset), f"Retrace incident: {self.state.scenario}", summary
        )
        # The document URN is kept on state for live cleanup; it is never sent to the
        # model (it is a server-generated id, which would break replay determinism).
        urn = result.get("document_urn") if isinstance(result, dict) else None
        if isinstance(urn, str) and urn not in self.state.writeback_urns:
            self.state.writeback_urns.append(urn)
        if _is_error(result):
            return {"error": result["error"]}
        self.state.written_back = True
        if self.state.stage is Stage.VERIFIED:
            advance(self.state, Stage.WRITTEN_BACK)
        return {"written_back": True, "stage": self.state.stage.value}

    def t_finish(self, report: str) -> dict[str, Any]:
        self._need_str(report=report)
        if self.state.stage not in FINISHABLE:
            raise GateError(f"cannot finish from {self.state.stage.value}")
        self.state.final_report = report
        self.finished = True
        return {"finished": True}
