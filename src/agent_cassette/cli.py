"""Record, replay, test, compare, and inspect AI-agent trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from agent_cassette.assertions import (
    check_trajectory,
    contains_event,
    max_total_cost,
    max_total_duration_ms,
    no_errors,
)
from agent_cassette.cassette import Cassette
from agent_cassette.deprecations import AgentCassetteDeprecationWarning
from agent_cassette.diagnostics import doctor
from agent_cassette.diff import compare_cassettes
from agent_cassette.events import EventType
from agent_cassette.hybrid import (
    Delay,
    Hybrid,
    HybridConfigurationError,
    InjectionRule,
    Raise,
    RateLimitError,
    Return,
)
from agent_cassette.interop import export_otlp, import_otlp
from agent_cassette.matching import MatchMode
from agent_cassette.migration import migrate_cassette
from agent_cassette.project_init import (
    ProjectInitError,
    initialize_project,
    load_project_config_from_root,
    render_init_report,
)
from agent_cassette.replay import ReplayMismatchError
from agent_cassette.reports import CIReport
from agent_cassette.runner import RunnerUsageError, run_python, validate_python_command
from agent_cassette.storage import (
    CassetteCorruptionError,
    RecoveryReport,
    load_events,
    recover_cassette,
    save_events,
)
from agent_cassette.viewer import write_viewer


def _inspect(path: Path, *, as_json: bool) -> int:
    events = load_events(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "events": len(events),
        "types": {},
        "duration_ms": sum(event.duration_ms or 0 for event in events),
        "cost": sum(event.cost or 0 for event in events),
    }
    for event in events:
        summary["types"][event.type.value] = summary["types"].get(event.type.value, 0) + 1
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Cassette: {path}")
        print(f"Events: {summary['events']}")
        for event_type, count in sorted(summary["types"].items()):
            print(f"  {event_type}: {count}")
        print(f"Duration: {summary['duration_ms']:.2f} ms")
        print(f"Cost: ${summary['cost']:.6f}")
    return 0


def _diff(baseline: Path, candidate: Path, *, as_json: bool, report_json: Path | None) -> int:
    report = compare_cassettes(baseline, candidate)
    if report_json is not None:
        ci_report = CIReport(metadata={"baseline": str(baseline), "candidate": str(candidate)})
        ci_report.add_diff("trajectory", report)
        ci_report.write_json(report_json)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if as_json else report.to_text())
    return 0 if report.identical else 1


def _recover(source: Path, destination: Path, *, as_json: bool) -> int:
    report: RecoveryReport = recover_cassette(source, destination)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    elif report.recovered:
        print(
            f"Recovered {report.events} event(s) to {report.destination}; "
            f"discarded {report.discarded_bytes} byte(s) at offset "
            f"{report.discarded_offset}."
        )
    else:
        print(
            f"Validated {report.events} event(s) and wrote {report.destination}; "
            "no incomplete tail found."
        )
    return 0


def _check(parsed: argparse.Namespace) -> int:
    checks = []
    if parsed.no_errors or not any(
        (parsed.require, parsed.max_cost is not None, parsed.max_duration_ms is not None)
    ):
        checks.append(no_errors())
    checks.extend(_required_check(value) for value in parsed.require)
    if parsed.max_cost is not None:
        checks.append(max_total_cost(parsed.max_cost))
    if parsed.max_duration_ms is not None:
        checks.append(max_total_duration_ms(parsed.max_duration_ms))
    assertions = check_trajectory(parsed.cassette, *checks)
    print(assertions.to_text())
    if parsed.report_json is not None:
        report = CIReport(metadata={"cassette": str(parsed.cassette)})
        report.add_assertions("trajectory", assertions)
        report.write_json(parsed.report_json)
    return 0 if assertions.passed else 1


def _required_check(value: str):
    event_type, separator, name = value.partition(":")
    return contains_event(EventType(event_type), name=name if separator else None)


_ALLOWED_INJECTED_ERRORS: dict[str, type[Exception]] = {
    "ConnectionError": ConnectionError,
    "RateLimitError": RateLimitError,
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}


def _parse_injection_action(
    item: dict[str, Any], *, nested: bool = False
) -> Return | Raise | Delay:
    action_name = item.get("action")
    if action_name == "return":
        return Return(item.get("value"))
    if action_name == "raise":
        error_name = str(item.get("error", "RuntimeError"))
        if error_name not in _ALLOWED_INJECTED_ERRORS:
            raise ValueError(f"unsupported injected error: {error_name}")
        message = str(item.get("message", error_name))
        if error_name == "RateLimitError":
            retry_after = item.get("retry_after")
            return Raise(
                RateLimitError(
                    message,
                    retry_after=float(retry_after) if retry_after is not None else None,
                )
            )
        return Raise(_ALLOWED_INJECTED_ERRORS[error_name](message))
    if action_name == "delay" and not nested:
        then_item = item.get("then")
        then = None
        if then_item is not None:
            if not isinstance(then_item, dict):
                raise ValueError("delay 'then' must be a JSON object")
            then_action = _parse_injection_action(then_item, nested=True)
            assert isinstance(then_action, (Return, Raise))
            then = then_action
        return Delay(float(item.get("seconds", 0)), then=then)
    allowed = "'return' or 'raise'" if nested else "'return', 'raise', or 'delay'"
    raise ValueError(f"injection action must be {allowed}")


def _load_injections(path: Path | None) -> tuple[InjectionRule, ...]:
    if path is None:
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("injection file must contain a JSON list")
    rules = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each injection rule must be a JSON object")
        action = _parse_injection_action(item)
        rules.append(
            InjectionRule(
                action,
                event_type=item.get("type"),
                name=item.get("name"),
                occurrence=item.get("occurrence", 1),
            )
        )
    return tuple(rules)


def _run(parsed: argparse.Namespace) -> int:
    command = validate_python_command(parsed.python_command)
    if parsed.command == "record":
        context = Cassette.record(parsed.cassette)
    elif parsed.command == "replay":
        loaded = load_project_config_from_root(Path.cwd())
        config = loaded[0] if loaded is not None else None
        if loaded is not None:
            for warning in loaded[1]:
                print(f"Configuration warning: {warning}", file=sys.stderr)
        strict = (
            parsed.replay_strict
            if parsed.replay_strict is not None
            else config.strict
            if config is not None
            else True
        )
        match_value = parsed.match or (config.match if config is not None else "exact")
        context = Cassette.replay(
            parsed.cassette,
            strict=strict,
            match=cast(MatchMode, match_value),
        )
    else:
        try:
            context = Hybrid(
                parsed.source,
                parsed.output,
                prefix=parsed.at,
                mismatch=parsed.mismatch,
                injections=_load_injections(parsed.inject),
            )
        except (json.JSONDecodeError, UnicodeError, OSError, ValueError) as error:
            raise _CLIInputError(str(error)) from error
    child_error: Exception | None = None
    try:
        with context as cassette:
            try:
                child_status = run_python(command, cassette)
            except ReplayMismatchError:
                raise
            except Exception as error:
                child_error = error
                raise
    except Exception as error:
        if error is child_error:
            raise _ChildExecutionError(error) from error
        raise
    return 0 if child_status == 0 else 1


class _CLIInputError(ValueError):
    """Expected invalid CLI input that is safe to render without a traceback."""


class _ChildExecutionError(Exception):
    """Private marker distinguishing user-code failures from CLI failures."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(str(error))


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(prog="agent-cassette", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="summarize a cassette")
    inspect_parser.add_argument("cassette", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    diff_parser = subparsers.add_parser("diff", help="compare two trajectories")
    diff_parser.add_argument("baseline", type=Path)
    diff_parser.add_argument("candidate", type=Path)
    diff_parser.add_argument("--json", action="store_true", dest="as_json")
    diff_parser.add_argument("--report-json", type=Path)

    check_parser = subparsers.add_parser("check", help="assert trajectory properties")
    check_parser.add_argument("cassette", type=Path)
    check_parser.add_argument("--no-errors", action="store_true")
    check_parser.add_argument("--require", action="append", default=[], metavar="TYPE[:NAME]")
    check_parser.add_argument("--max-cost", type=float)
    check_parser.add_argument("--max-duration-ms", type=float)
    check_parser.add_argument("--report-json", type=Path)

    view_parser = subparsers.add_parser("view", help="write a secure standalone HTML viewer")
    view_parser.add_argument("cassette", type=Path)
    view_parser.add_argument("--output", "-o", required=True, type=Path)

    export_parser = subparsers.add_parser("export-otlp", help="export cassette as OTLP JSON")
    export_parser.add_argument("cassette", type=Path)
    export_parser.add_argument("output", type=Path)

    import_parser = subparsers.add_parser("import-otlp", help="import OTLP JSON as a cassette")
    import_parser.add_argument("input", type=Path)
    import_parser.add_argument("cassette", type=Path)
    import_parser.add_argument("--permissive", action="store_true")

    migrate_parser = subparsers.add_parser("migrate", help="rewrite using the current schema")
    migrate_parser.add_argument("cassette", type=Path)
    migrate_parser.add_argument("--output", "-o", type=Path)

    recover_parser = subparsers.add_parser(
        "recover", help="recover only an incomplete final JSONL fragment"
    )
    recover_parser.add_argument("source", type=Path)
    recover_parser.add_argument("output", type=Path)
    recover_parser.add_argument("--json", action="store_true", dest="as_json")

    doctor_parser = subparsers.add_parser("doctor", help="diagnose installation and integrations")
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    init_parser = subparsers.add_parser("init", help="initialize Agent Cassette in a project")
    init_parser.add_argument("project", nargs="?", type=Path, default=Path.cwd())
    init_parser.add_argument(
        "--detect", action="store_true", help="detect integrations from Python manifests"
    )
    init_mode = init_parser.add_mutually_exclusive_group()
    init_mode.add_argument("--check", action="store_true", help="check without writing")
    init_mode.add_argument("--dry-run", action="store_true", help="preview without writing")
    init_parser.add_argument("--json", action="store_true", dest="as_json")

    for name, help_text in (
        ("record", "run Python and record supported agent calls"),
        ("replay", "run Python with offline agent-call replay"),
    ):
        run_parser = subparsers.add_parser(name, help=help_text)
        if name == "replay":
            run_parser.add_argument(
                "--match",
                choices=("exact", "subset", "normalized", "fuzzy"),
                default=None,
                help="override project replay matching",
            )
            strict_group = run_parser.add_mutually_exclusive_group()
            strict_group.add_argument(
                "--strict",
                action="store_true",
                dest="replay_strict",
                help="require all cassette events to be consumed",
            )
            strict_group.add_argument(
                "--no-strict",
                action="store_false",
                dest="replay_strict",
                help="allow unconsumed cassette events",
            )
            run_parser.set_defaults(replay_strict=None)
        run_parser.add_argument("cassette", type=Path)
        run_parser.add_argument("python_command", nargs=argparse.REMAINDER)

    fork_parser = subparsers.add_parser("fork", help="replay a prefix then continue live")
    fork_parser.add_argument("source", type=Path)
    fork_parser.add_argument("output", type=Path)
    fork_parser.add_argument("--at", type=int, default=None, metavar="EVENTS")
    fork_parser.add_argument("--mismatch", choices=("raise", "live"), default="raise")
    fork_parser.add_argument("--inject", type=Path, help="JSON failure-injection rules")
    fork_parser.add_argument("python_command", nargs=argparse.REMAINDER)
    return parser


def _dispatch(parsed: argparse.Namespace) -> int:
    if parsed.command == "inspect":
        return _inspect(parsed.cassette, as_json=parsed.as_json)
    if parsed.command == "diff":
        return _diff(
            parsed.baseline,
            parsed.candidate,
            as_json=parsed.as_json,
            report_json=parsed.report_json,
        )
    if parsed.command == "check":
        return _check(parsed)
    if parsed.command == "view":
        write_viewer(parsed.output, parsed.cassette)
        print(parsed.output)
        return 0
    if parsed.command == "export-otlp":
        export_otlp(load_events(parsed.cassette), parsed.output)
        print(parsed.output)
        return 0
    if parsed.command == "import-otlp":
        save_events(parsed.cassette, import_otlp(parsed.input, strict=not parsed.permissive))
        print(parsed.cassette)
        return 0
    if parsed.command == "migrate":
        destination = parsed.output
        if destination is None:
            with warnings.catch_warnings():
                warnings.simplefilter("always", AgentCassetteDeprecationWarning)
                warnings.warn(
                    "in-place cassette migration is deprecated; pass --output PATH",
                    AgentCassetteDeprecationWarning,
                    stacklevel=2,
                )
            destination = parsed.cassette
        print(migrate_cassette(parsed.cassette, destination))
        return 0
    if parsed.command == "recover":
        return _recover(parsed.source, parsed.output, as_json=parsed.as_json)
    if parsed.command == "doctor":
        return doctor(as_json=parsed.as_json)
    if parsed.command == "init":
        mode = "check" if parsed.check else "dry-run" if parsed.dry_run else "apply"
        report, exit_code = initialize_project(parsed.project, detect=parsed.detect, mode=mode)
        print(render_init_report(report, as_json=parsed.as_json))
        return exit_code
    return _run(parsed)


def _render_expected_error(parsed: argparse.Namespace, error: BaseException) -> None:
    message = str(error)
    if parsed.command == "replay" and isinstance(error, ProjectInitError):
        message = f"Invalid project configuration: {message}"
    if getattr(parsed, "as_json", False):
        print(
            json.dumps(
                {
                    "command": parsed.command,
                    "error": message,
                    "schema_version": 1,
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Error: {message}", file=sys.stderr)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the Agent Cassette CLI with stable process-compatible exit codes."""
    parsed = build_parser().parse_args(arguments)
    try:
        return _dispatch(parsed)
    except _ChildExecutionError as wrapped:
        raise wrapped.error.with_traceback(wrapped.error.__traceback__) from None
    except ReplayMismatchError as error:
        _render_expected_error(parsed, error)
        return 1
    except (
        CassetteCorruptionError,
        HybridConfigurationError,
        ProjectInitError,
        RunnerUsageError,
        _CLIInputError,
        OSError,
        UnicodeError,
    ) as error:
        _render_expected_error(parsed, error)
        return 2
    except ValueError as error:
        if parsed.command in {"record", "replay", "fork"}:
            raise
        _render_expected_error(parsed, error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
