"""Phase E named record / replay / rerecord orchestration.

Named runs resolve a config-owned golden cassette by name, run the caller's Python child
in-process, and write a single machine report atomically to a separate file (the child's
own stdout/stderr is never touched). Record is create-only and publishes a temporary
cassette to the golden path only after full validation; rerecord is the sole explicit
golden-update path. Reports carry only code-owned, payload-free data.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_cassette.cassette import Cassette
from agent_cassette.machine import (
    EXIT_CHANGES,
    EXIT_INVALID,
    EXIT_OK,
    Envelope,
    MachineInputError,
    NextAction,
    validate_cassette_name,
)
from agent_cassette.project_init import ProjectConfig, ProjectInitError
from agent_cassette.project_loop import (
    MANIFEST_DIR,
    REPORTS_DIRNAME,
    ProjectLoopError,
    _load_config,
    resolve_project_root,
)
from agent_cassette.replay import ReplayMismatchError, _is_replayable
from agent_cassette.runner import RunnerUsageError, run_python, validate_python_command
from agent_cassette.storage import CassetteCorruptionError, load_events


@dataclass(frozen=True, slots=True)
class NamedRunResult:
    """What the CLI needs: the envelope, the written report path, and any child exception
    to re-raise (preserving its traceback) after the report is durably written."""

    envelope: Envelope
    report_path: Path | None
    child_exception: BaseException | None = None


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def _resolve_context(project: str | Path | None) -> tuple[Path, ProjectConfig]:
    root = resolve_project_root(project)
    config, _warnings = _load_config(root)
    if config is None:
        raise ProjectLoopError("a named run requires a valid project configuration")
    return root, config


def _resolve_named_cassette(root: Path, config: ProjectConfig, name: str) -> Path:
    validate_cassette_name(name)
    cassette_dir = (root / config.cassette_dir).absolute()
    candidate = (cassette_dir / f"{name}.jsonl").absolute()
    if os.path.commonpath([str(cassette_dir), str(candidate)]) != str(cassette_dir):
        raise ProjectLoopError("resolved cassette escapes the configured cassette directory")
    return candidate


def _default_report_path(root: Path, command: str, name: str) -> Path:
    return (root / MANIFEST_DIR / REPORTS_DIRNAME / f"{command}-{name}.json").absolute()


def _resolve_report_path(root: Path, command: str, name: str, override: str | Path | None) -> Path:
    if override is None:
        return _default_report_path(root, command, name)
    destination = Path(override)
    destination = destination if destination.is_absolute() else (root / destination)
    # normpath collapses ``..`` lexically so a ``../../x`` override cannot pass the
    # containment check and escape the project root.
    normalized = Path(os.path.normpath(str(destination)))
    if os.path.commonpath([str(root), str(normalized)]) != str(root):
        raise ProjectLoopError("--report-json must be beneath the project root")
    return normalized


# --------------------------------------------------------------------------- #
# Atomic report writing (rejects symlink / non-regular / directory destinations)
# --------------------------------------------------------------------------- #


def _write_report(path: Path, envelope: Envelope) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise ProjectLoopError(f"cannot inspect report destination: {error}") from error
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise ProjectLoopError("report destination is not a regular file")
    contents = envelope.to_json()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _cassette_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# record --name (create-only, atomic publish on full success)
# --------------------------------------------------------------------------- #


def run_named_record(
    project: str | Path | None, name: str, command: list[str], report_override: str | Path | None
) -> NamedRunResult:
    try:
        root, config = _resolve_context(project)
        golden = _resolve_named_cassette(root, config, name)
        report_path = _resolve_report_path(root, "record", name, report_override)
        arguments = validate_python_command(command)
    except (MachineInputError, ProjectLoopError, ProjectInitError, RunnerUsageError) as error:
        return _invalid_named(project, "record", name, error)

    if os.path.lexists(golden):
        return _finish(
            _named_envelope(
                "record",
                "invalid",
                EXIT_INVALID,
                root,
                golden,
                name,
                data={"blockers": [{"code": "cassette-exists", "path": str(golden)}]},
            ),
            report_path,
        )

    golden.parent.mkdir(parents=True, exist_ok=True)
    temporary = golden.parent / f".{name}.{uuid4().hex}.tmp"
    child_exception: BaseException | None = None
    child_status: int | None = None
    try:
        with Cassette.record(temporary) as recorder:
            child_status = run_python(arguments, recorder)
    except Exception as error:  # child code raised
        child_exception = error

    if child_exception is not None:
        temporary.unlink(missing_ok=True)
        # The child exception is re-raised after the report is written, so the process
        # exits 1 (child failure) -- exit_code must equal that, not EXIT_INVALID.
        envelope = _named_envelope(
            "record",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={
                "error_code": "child-exception",
                "exception_type": type(child_exception).__name__,
            },
        )
        return _finish(envelope, report_path, child_exception=child_exception)

    if child_status != 0:
        temporary.unlink(missing_ok=True)
        envelope = _named_envelope(
            "record",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={"child_exit_code": child_status},
            next_actions=(_retry_record_action(name),),
        )
        return _finish(envelope, report_path)

    valid, replayable = _validate_recorded(temporary)
    if not valid or replayable < 1:
        temporary.unlink(missing_ok=True)
        envelope = _named_envelope(
            "record",
            "invalid",
            EXIT_INVALID,
            root,
            golden,
            name,
            data={"error_code": "empty-or-invalid-recording", "replayable_events": replayable},
        )
        return _finish(envelope, report_path)

    # Create-only publish: os.link fails closed if a golden appeared during the (possibly
    # long/networked) child run, so record never overwrites an existing cassette.
    try:
        os.link(temporary, golden)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _finish(
            _named_envelope(
                "record",
                "invalid",
                EXIT_INVALID,
                root,
                golden,
                name,
                data={"blockers": [{"code": "cassette-exists", "path": str(golden)}]},
            ),
            report_path,
        )
    temporary.unlink(missing_ok=True)
    envelope = _named_envelope(
        "record",
        "current",
        EXIT_OK,
        root,
        golden,
        name,
        data={
            "cassette": {"name": name, "path": str(golden), "sha256": _cassette_sha256(golden)},
            "replayable_events": replayable,
            "child_exit_code": 0,
        },
        next_actions=(_offline_replay_action(name),),
    )
    return _finish(envelope, report_path)


def _validate_recorded(path: Path) -> tuple[bool, int]:
    try:
        events = load_events(path)
    except (CassetteCorruptionError, OSError, ValueError):
        return False, 0
    return True, sum(1 for event in events if _is_replayable(event))


# --------------------------------------------------------------------------- #
# replay --name (offline; structured success / mismatch report)
# --------------------------------------------------------------------------- #


def run_named_replay(
    project: str | Path | None,
    name: str,
    command: list[str],
    report_override: str | Path | None,
    *,
    match: str | None,
    strict: bool | None,
) -> NamedRunResult:
    try:
        root, config = _resolve_context(project)
        golden = _resolve_named_cassette(root, config, name)
        report_path = _resolve_report_path(root, "replay", name, report_override)
        arguments = validate_python_command(command)
    except (MachineInputError, ProjectLoopError, ProjectInitError, RunnerUsageError) as error:
        return _invalid_named(project, "replay", name, error)

    if not os.path.lexists(golden):
        return _finish(
            _named_envelope(
                "replay",
                "invalid",
                EXIT_INVALID,
                root,
                golden,
                name,
                data={"blockers": [{"code": "cassette-missing", "path": str(golden)}]},
            ),
            report_path,
        )

    effective_match = match or config.match
    effective_strict = strict if strict is not None else config.strict
    replayer = Cassette.replay(golden, strict=effective_strict, match=effective_match)  # type: ignore[arg-type]
    mismatch: ReplayMismatchError | None = None
    child_exception: BaseException | None = None
    child_status: int | None = None
    try:
        with replayer:
            child_status = run_python(arguments, replayer)
    except ReplayMismatchError as error:
        mismatch = error
    except Exception as error:
        child_exception = error

    sha = _cassette_sha256(golden)
    cassette_summary = {"name": name, "path": str(golden), "sha256": sha}
    replayable = len(replayer.events)
    remaining = replayer.remaining
    consumed = replayable - remaining

    if child_exception is not None:
        envelope = _named_envelope(
            "replay",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={
                "cassette": cassette_summary,
                "error_code": "child-exception",
                "exception_type": type(child_exception).__name__,
            },
        )
        return _finish(envelope, report_path, child_exception=child_exception)

    if mismatch is not None:
        envelope = _named_envelope(
            "replay",
            "mismatch",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={"cassette": cassette_summary, "failure": _failure_data(mismatch)},
            next_actions=_mismatch_actions(name),
        )
        return _finish(envelope, report_path)

    if child_status != 0:
        envelope = _named_envelope(
            "replay",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={"cassette": cassette_summary, "child_exit_code": child_status},
        )
        return _finish(envelope, report_path)

    envelope = _named_envelope(
        "replay",
        "current",
        EXIT_OK,
        root,
        golden,
        name,
        data={
            "cassette": cassette_summary,
            "replayable_events": replayable,
            "consumed_events": consumed,
            "remaining_events": remaining,
            "child_exit_code": 0,
        },
    )
    return _finish(envelope, report_path)


def _failure_data(error: ReplayMismatchError) -> dict[str, Any]:
    return {
        "kind": error.kind,
        "event_index": error.event_index,
        "expected": error.expected,
        "actual": error.actual,
        "changed_paths": list(error.changed_paths),
        "changed_paths_truncated": error.changed_paths_truncated,
        "match": error.match,
        "remaining": error.remaining,
    }


# --------------------------------------------------------------------------- #
# rerecord --name (explicit golden update; atomic replace on full success)
# --------------------------------------------------------------------------- #


def run_named_rerecord(
    project: str | Path | None, name: str, command: list[str], report_override: str | Path | None
) -> NamedRunResult:
    try:
        root, config = _resolve_context(project)
        golden = _resolve_named_cassette(root, config, name)
        report_path = _resolve_report_path(root, "rerecord", name, report_override)
        arguments = validate_python_command(command)
    except (MachineInputError, ProjectLoopError, ProjectInitError, RunnerUsageError) as error:
        return _invalid_named(project, "rerecord", name, error)

    if not os.path.lexists(golden):
        return _finish(
            _named_envelope(
                "rerecord",
                "invalid",
                EXIT_INVALID,
                root,
                golden,
                name,
                data={"blockers": [{"code": "cassette-missing", "path": str(golden)}]},
            ),
            report_path,
        )

    old_sha = _cassette_sha256(golden)
    old_valid, old_count = _validate_recorded(golden)
    temporary = golden.parent / f".{name}.{uuid4().hex}.tmp"
    child_exception: BaseException | None = None
    child_status: int | None = None
    try:
        with Cassette.record(temporary) as recorder:
            child_status = run_python(arguments, recorder)
    except Exception as error:
        child_exception = error

    if child_exception is not None:
        temporary.unlink(missing_ok=True)  # golden untouched
        envelope = _named_envelope(
            "rerecord",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={
                "error_code": "child-exception",
                "exception_type": type(child_exception).__name__,
            },
        )
        return _finish(envelope, report_path, child_exception=child_exception)

    if child_status != 0:
        temporary.unlink(missing_ok=True)
        envelope = _named_envelope(
            "rerecord",
            "child-failed",
            EXIT_CHANGES,
            root,
            golden,
            name,
            data={"child_exit_code": child_status},
        )
        return _finish(envelope, report_path)

    valid, replayable = _validate_recorded(temporary)
    if not valid or replayable < 1:
        temporary.unlink(missing_ok=True)
        envelope = _named_envelope(
            "rerecord",
            "invalid",
            EXIT_INVALID,
            root,
            golden,
            name,
            data={"error_code": "empty-or-invalid-recording", "replayable_events": replayable},
        )
        return _finish(envelope, report_path)

    os.replace(temporary, golden)  # atomic; old golden only now replaced
    new_sha = _cassette_sha256(golden)
    envelope = _named_envelope(
        "rerecord",
        "changed",
        EXIT_OK,
        root,
        golden,
        name,
        data={
            "cassette": {"name": name, "path": str(golden), "sha256": new_sha},
            "previous_sha256": old_sha,
            "previous_replayable_events": old_count if old_valid else None,
            "replayable_events": replayable,
            "child_exit_code": 0,
        },
    )
    return _finish(envelope, report_path)


# --------------------------------------------------------------------------- #
# Shared envelope / action helpers
# --------------------------------------------------------------------------- #


def _named_envelope(
    command: str,
    status: str,
    exit_code: int,
    root: Path,
    golden: Path,
    name: str,
    *,
    data: dict[str, Any],
    next_actions: tuple[NextAction, ...] = (),
) -> Envelope:
    return Envelope(
        command=command,
        status=status,
        exit_code=exit_code,
        project=str(root),
        data=data,
        next_actions=next_actions,
    )


def _finish(
    envelope: Envelope, report_path: Path, *, child_exception: BaseException | None = None
) -> NamedRunResult:
    _write_report(report_path, envelope)
    return NamedRunResult(
        envelope=envelope, report_path=report_path, child_exception=child_exception
    )


def _invalid_named(
    project: str | Path | None, command: str, name: str, error: Exception
) -> NamedRunResult:
    # Best-effort root/report path; if even those fail, surface an envelope without a file.
    try:
        root = resolve_project_root(project)
    except Exception:
        root = Path.cwd()
    envelope = Envelope(
        command=command,
        status="invalid",
        exit_code=EXIT_INVALID,
        project=str(root),
        data={"error_code": _error_code(error)},
        next_actions=(
            NextAction(id="inspect-status", argv=("agent-cassette", "status", ".", "--json")),
        ),
    )
    return NamedRunResult(envelope=envelope, report_path=None)


def _error_code(error: Exception) -> str:
    if isinstance(error, MachineInputError):
        return "invalid-name"
    if isinstance(error, RunnerUsageError):
        return "invalid-command"
    return "invalid-project"


def _retry_record_action(name: str) -> NextAction:
    return NextAction(
        id="retry-record",
        argv=("agent-cassette", "record", "--name", name, "--"),
        network="required",
        requires_child_command=True,
    )


def _offline_replay_action(name: str) -> NextAction:
    return NextAction(
        id="offline-replay",
        argv=("agent-cassette", "replay", "--name", name, "--"),
        requires_child_command=True,
    )


def _mismatch_actions(name: str) -> tuple[NextAction, ...]:
    return (
        NextAction(id="inspect-status", argv=("agent-cassette", "status", ".", "--json")),
        NextAction(
            id="retry-after-fix",
            argv=("agent-cassette", "replay", "--name", name, "--"),
            requires_child_command=True,
        ),
        NextAction(
            id="inspect-cassette",
            argv=("agent-cassette", "inspect", "<cassette-path>", "--json"),
        ),
        NextAction(
            id="rerecord-golden",
            argv=("agent-cassette", "rerecord", "--name", name, "--"),
            network="required",
            approval_required=True,
            requires_child_command=True,
        ),
    )
