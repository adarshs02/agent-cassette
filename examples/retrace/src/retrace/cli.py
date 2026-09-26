"""retrace command line."""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

from retrace.config import APP_ROOT, CASSETTE_DIR, RESULTS_DIR, Settings

WORK_ROOT = APP_ROOT / "work"


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "model", None):
        settings = dataclasses.replace(settings, model=args.model)
    return settings


def _eval(args: argparse.Namespace) -> int:
    from retrace.evals.runner import run_eval
    from retrace.evals.scorecard import write_results

    mode = "live" if args.live else "replay"
    names = [n for n in args.scenarios.split(",") if n] if args.scenarios else None
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    try:
        results = run_eval(
            mode,
            names,
            settings=_settings(args),
            cassette_dir=CASSETTE_DIR,
            work_root=WORK_ROOT,
            trials=args.trials,
        )
    except KeyError as error:
        print(error, file=sys.stderr)
        return 2
    path = write_results(results, mode, RESULTS_DIR)
    print(path.read_text())
    skipped = [r.scenario for r in results if r.status == "skipped"]
    if skipped:
        print(f"WARNING: skipped (no cassette yet): {skipped}", file=sys.stderr)
    return 0 if all(r.status in ("passed", "skipped") for r in results) else 1


def _run(args: argparse.Namespace) -> int:
    from retrace.evals.runner import run_scenario
    from retrace.evals.scenarios import Scenario

    with tempfile.TemporaryDirectory() as tmp:
        result = run_scenario(
            Scenario(args.fault, args.fault, "agent"),
            "live",
            settings=_settings(args),
            cassette_dir=Path(tmp) / "c",
            work_root=Path(tmp) / "w",
            record=False,
        )
    print(f"status={result.status} stage={result.final_stage}")
    for grade in result.grades:
        print(f"  {'PASS' if grade.passed else 'FAIL'} {grade.name}: {grade.detail}")
    if result.diff:
        print(result.diff)
    if result.error:
        print(result.error, file=sys.stderr)
    return 0 if result.status == "passed" else 1


def _ingest(args: argparse.Namespace) -> int:
    from retrace.datahub.ingest import ingest_workspace
    from retrace.pipeline.workspace import prepare

    with tempfile.TemporaryDirectory() as tmp:
        count = ingest_workspace(prepare(Path(tmp) / "ws", fault=args.fault), _settings(args))
    print(f"emitted {count} metadata proposals")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrace")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="run the eval suite")
    mode = ev.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="real Claude + DataHub; records")
    mode.add_argument("--replay", action="store_true", help="offline from cassettes")
    ev.add_argument("--scenarios", help="comma-separated scenario names")
    ev.add_argument("--trials", type=int, default=1)
    ev.add_argument("--model")
    ev.set_defaults(func=_eval)

    run = sub.add_parser("run", help="one live, unrecorded run")
    run.add_argument("--fault", required=True)
    run.add_argument("--model")
    run.set_defaults(func=_run)

    ing = sub.add_parser("ingest", help="push pipeline metadata to DataHub")
    ing.add_argument("--fault")
    ing.set_defaults(func=_ingest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
