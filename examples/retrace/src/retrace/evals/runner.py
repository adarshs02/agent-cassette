"""Run scenarios live (recording cassettes) or offline (replaying them)."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_cassette import ReplayMismatchError
from retrace.agent.executor import ToolExecutor
from retrace.agent.loop import run_agent
from retrace.agent.state import IncidentState
from retrace.cassette import open_clients
from retrace.config import Settings
from retrace.datahub.mcp_client import DataHubConnection
from retrace.evals.graders import Grade, grade_agent_run, grade_robustness, run_bad_repair_gate
from retrace.evals.scenarios import SCENARIOS, Scenario, get_scenario
from retrace.faults import get_fault
from retrace.pipeline.baseline import load_baseline
from retrace.pipeline.workspace import Workspace, prepare
from retrace.tools.datahub import DataHubTools
from retrace.tools.repair import Repairer, Transforms, unified_diff
from retrace.tools.warehouse import Warehouse

MANIFEST = "MANIFEST.json"
Status = Literal["passed", "failed", "error", "skipped"]


@dataclass
class ScenarioResult:
    scenario: str
    status: Status
    grades: list[Grade] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    final_stage: str | None = None
    diff: str = ""
    wall_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["grades"] = [g.to_dict() for g in self.grades]
        return data


def build_executor(
    state: IncidentState,
    ws: Workspace,
    datahub: DataHubConnection,
    baseline: dict,
    scratch: Path,
    settings: Settings,
) -> ToolExecutor:
    return ToolExecutor(
        state,
        Warehouse(ws.warehouse, baseline),
        Transforms(ws),
        Repairer(ws, baseline, scratch),
        DataHubTools(datahub),
        max_repair_attempts=settings.max_repair_attempts,
    )


def _manifest(cassette_dir: Path) -> list[str] | None:
    path = cassette_dir / MANIFEST
    return json.loads(path.read_text())["scenarios"] if path.exists() else None


def run_scenario(
    scenario: Scenario,
    mode: Literal["live", "replay"],
    *,
    settings: Settings,
    cassette_dir: Path,
    work_root: Path,
    model_factory: Callable[[], Any] | None = None,
    datahub_factory: Callable[[Any | None], DataHubConnection] | None = None,
    ingest: bool = True,
    record: bool = True,
) -> ScenarioResult:
    started = time.perf_counter()
    fault = get_fault(scenario.fault)
    baseline = load_baseline()
    root = work_root / scenario.name
    ws = prepare(root, fault=fault.name)
    scratch = root / "scratch"

    if scenario.kind == "gate":
        grades = run_bad_repair_gate(ws, baseline, scratch)
        return _finish(scenario, grades, None, {}, "", started)

    cassette = cassette_dir / f"{scenario.name}.jsonl"
    if mode == "replay" and not cassette.exists():
        manifest = _manifest(cassette_dir)
        if manifest is not None and scenario.name in manifest:
            return ScenarioResult(scenario.name, "error", error=f"missing cassette {cassette.name}")
        return ScenarioResult(scenario.name, "skipped", error="no cassette recorded yet")

    if mode == "live" and ingest:
        from retrace.datahub.ingest import ingest_workspace

        ingest_workspace(ws, settings)

    source = cassette_dir / f"{scenario.base}.jsonl" if scenario.base else None
    clients_mode = "replay" if mode == "replay" else ("record" if record else "live")
    state = IncidentState(scenario=scenario.name, report=fault.report)
    try:
        with open_clients(
            clients_mode,
            settings=settings,
            cassette_path=cassette,
            injections=scenario.injections if mode == "live" else (),
            source=source,
            model_factory=model_factory,
            datahub_factory=datahub_factory,
        ) as clients:
            executor = build_executor(state, ws, clients.datahub, baseline, scratch, settings)
            stats = run_agent(
                clients.anthropic,
                executor,
                state,
                model=settings.model,
                max_turns=settings.max_turns,
                max_nudges=settings.max_nudges,
                sleep=time.sleep if mode == "live" else (lambda s: None),
            )
    except ReplayMismatchError as error:
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-15:])
        return ScenarioResult(
            scenario.name,
            "error",
            final_stage=state.stage.value,
            wall_s=round(time.perf_counter() - started, 2),
            error=f"replay diverged: {error}\n{tail}",
        )
    except Exception as error:  # noqa: BLE001 - one scenario must not abort the suite
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-15:])
        return ScenarioResult(
            scenario.name,
            "error",
            final_stage=state.stage.value,
            wall_s=round(time.perf_counter() - started, 2),
            error=f"{type(error).__name__}: {error}\n{tail}",
        )

    if scenario.kind == "robustness" and scenario.name == "datahub_timeout":
        grades = grade_robustness(fault, state)
    else:
        grades = grade_agent_run(fault, state, ws, scratch, baseline)
    diff = unified_diff(ws.transforms, state.patched) if state.patched else ""
    return _finish(scenario, grades, state, stats.to_dict(), diff, started)


def _finish(
    scenario: Scenario,
    grades: list[Grade],
    state: IncidentState | None,
    stats: dict[str, int],
    diff: str,
    started: float,
) -> ScenarioResult:
    status: Status = "passed" if all(g.passed for g in grades) else "failed"
    return ScenarioResult(
        scenario.name,
        status,
        grades,
        stats,
        final_stage=state.stage.value if state else None,
        diff=diff,
        wall_s=round(time.perf_counter() - started, 2),
    )


def run_eval(
    mode: Literal["live", "replay"],
    names: list[str] | None,
    *,
    settings: Settings,
    cassette_dir: Path,
    work_root: Path,
    trials: int = 1,
    **factories: Any,
) -> list[ScenarioResult]:
    selected = [get_scenario(n) for n in names] if names else list(SCENARIOS)
    results: list[ScenarioResult] = []
    for scenario in selected:
        # Only agent scenarios repeat; robustness forks and the gate run once.
        repeats = max(1, trials) if mode == "live" and scenario.kind == "agent" else 1
        for trial in range(repeats):
            result = run_scenario(
                scenario,
                mode,
                settings=settings,
                cassette_dir=cassette_dir,
                work_root=work_root / f"t{trial + 1}",
                record=(trial == 0),
                **factories,
            )
            if trial > 0:
                result.scenario = f"{scenario.name}#t{trial + 1}"
            results.append(result)
    if mode == "live" and not names:
        cassette_dir.mkdir(parents=True, exist_ok=True)
        recorded = sorted(p.stem for p in cassette_dir.glob("*.jsonl"))
        (cassette_dir / MANIFEST).write_text(json.dumps({"scenarios": recorded}, indent=2) + "\n")
    return results
