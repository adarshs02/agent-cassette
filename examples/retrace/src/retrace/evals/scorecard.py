"""Markdown scorecard and full JSON results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from retrace.evals.runner import ScenarioResult

FAULTS = ("unit_cents", "schema_rename", "join_fanout", "tz_shift", "stale_feed", "null_surge")
CONTROLS = ("control_healthy", "control_distractor")


def _cost(results: list[ScenarioResult]) -> str:
    price_in = os.environ.get("RETRACE_PRICE_INPUT_PER_MTOK")
    price_out = os.environ.get("RETRACE_PRICE_OUTPUT_PER_MTOK")
    if not price_in or not price_out:
        return "n/a (set RETRACE_PRICE_INPUT_PER_MTOK / RETRACE_PRICE_OUTPUT_PER_MTOK)"
    tokens_in = sum(r.stats.get("input_tokens", 0) for r in results)
    tokens_out = sum(r.stats.get("output_tokens", 0) for r in results)
    return f"${tokens_in / 1e6 * float(price_in) + tokens_out / 1e6 * float(price_out):.2f}"


def _next_run(results_dir: Path) -> int:
    runs = [int(p.stem.split("_")[1]) for p in results_dir.glob("run_*.json")]
    return max(runs, default=0) + 1


def _table_cell(text: str, limit: int = 120) -> str:
    """One markdown-table-safe line: first line only, `|` escaped, length-capped."""
    first_line = text.splitlines()[0] if text else ""
    return first_line.replace("|", "\\|")[:limit]


def _match_column(results: list[ScenarioResult], expected: dict[str, str]) -> dict[str, str]:
    """Per-scenario "yes"/"no"/"—" against manifest["expected"] (recorded trial-1 status)."""
    column = {}
    for r in results:
        if r.scenario not in expected:
            column[r.scenario] = "—"  # em dash: no recorded expectation to compare
        else:
            column[r.scenario] = "yes" if r.status == expected[r.scenario] else "no"
    return column


def write_results(
    results: list[ScenarioResult],
    mode: str,
    results_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    run = _next_run(results_dir)
    (results_dir / f"run_{run:04d}.json").write_text(
        json.dumps({"mode": mode, "results": [r.to_dict() for r in results]}, indent=2) + "\n"
    )
    by_name = {r.scenario: r for r in results}
    faults_ok = sum(1 for f in FAULTS if by_name.get(f) and by_name[f].status == "passed")
    controls_fp = sum(1 for c in CONTROLS if by_name.get(c) and by_name[c].status == "failed")
    not_graded = [r.scenario for r in results if r.status in ("skipped", "error")]
    expected = (manifest or {}).get("expected") if mode == "replay" else None
    lines = [
        f"# Retrace eval — {mode} (run {run})",
        "",
        f"- Faults correct: {faults_ok}/{len(FAULTS)}",
        f"- Control false positives: {controls_fp}/{len(CONTROLS)}",
        f"- Not graded (skipped/error): {', '.join(not_graded) if not_graded else 'none'}",
        f"- Total wall time: {sum(r.wall_s for r in results):.1f}s",
        f"- Cost: {_cost(results)}",
    ]
    match_column: dict[str, str] = {}
    if expected is not None:
        match_column = _match_column(results, expected)
        comparable = [v for v in match_column.values() if v != "—"]
        matched = sum(1 for v in comparable if v == "yes")
        lines.append(f"- Matches recording: {matched}/{len(comparable)}")
    lines.append("")
    header = (
        "| scenario | status | stage | failed grades | turns | tool calls | tokens in | "
        "tokens out | wall s |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    if expected is not None:
        header += " matches recording |"
        separator += "---|"
    lines += [header, separator]
    for r in results:
        failed = ", ".join(g.name for g in r.grades if not g.passed) or (r.error or "")
        row = (
            f"| {r.scenario} | {r.status} | {r.final_stage or ''} | {_table_cell(failed)} | "
            f"{r.stats.get('turns', '')} | {r.stats.get('tool_calls', '')} | "
            f"{r.stats.get('input_tokens', '')} | {r.stats.get('output_tokens', '')} | "
            f"{r.wall_s} |"
        )
        if expected is not None:
            row += f" {match_column[r.scenario]} |"
        lines.append(row)
    errored = [r for r in results if r.status == "error" and r.error]
    if errored:
        lines += ["", "## Errors", ""]
        for r in errored:
            lines += [f"### {r.scenario}", "", "````", r.error, "````", ""]
    path = results_dir / "scorecard.md"
    path.write_text("\n".join(lines) + "\n")
    return path
