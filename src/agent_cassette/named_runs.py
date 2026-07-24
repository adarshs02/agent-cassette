"""Phase E named record / replay / rerecord orchestration.

All project filesystem access goes through :mod:`agent_cassette.secure_fs` (directory-FD /
no-follow), never a path-based ``open``/``mkdir``/``rename`` that could follow a symlinked
or swapped parent out of the project. Child recordings are staged in a private temp
directory outside the consumer tree and copied into the project through file descriptors;
the golden cassette read for replay/rerecord is snapshotted once and never reopened. Named
``record`` is create-only, ``rerecord`` is the sole atomic golden-update path, and the
report is a separate, atomically written machine channel (child stdout/stderr untouched).
Reports carry only code-owned, payload-free data.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from agent_cassette.project_init import ProjectConfig, ProjectInitError, _relative_parts
from agent_cassette.project_loop import (
    MANIFEST_DIR,
    REPORTS_DIRNAME,
    ProjectLoopError,
    _load_config,
    resolve_project_root,
)
from agent_cassette.replay import ReplayMismatchError, _is_replayable
from agent_cassette.runner import RunnerUsageError, run_python, validate_python_command
from agent_cassette.secure_fs import (
    SecureFilesystemError,
    atomic_replace,
    identity_of,
    open_root,
    preflight_parent,
    publish_create_only,
    snapshot,
)
from agent_cassette.storage import load_events_from_bytes


@dataclass(frozen=True, slots=True)
class NamedRunResult:
    envelope: Envelope
    report_path: Path | None
    child_exception: BaseException | None = None


# --------------------------------------------------------------------------- #
# Resolution (lexical) — the FD layer enforces the actual security
# --------------------------------------------------------------------------- #


def _resolve_context(project: str | Path | None) -> tuple[Path, ProjectConfig]:
    root = resolve_project_root(project)
    config, _warnings = _load_config(root)
    if config is None:
        raise ProjectLoopError("a named run requires a valid project configuration")
    return root, config


def _cassette_relative(config: ProjectConfig, name: str) -> str:
    validate_cassette_name(name)
    parts = _relative_parts(config.cassette_dir)
    return "/".join((*parts, f"{name}.jsonl"))


def _report_relative(root: Path, command: str, name: str, override: str | Path | None) -> str:
    if override is None:
        return "/".join((MANIFEST_DIR, REPORTS_DIRNAME, f"{command}-{name}.json"))
    destination = Path(override)
    destination = destination if destination.is_absolute() else (root / destination)
    normalized = Path(os.path.normpath(str(destination)))
    if normalized == root or os.path.commonpath([str(root), str(normalized)]) != str(root):
        raise ProjectLoopError("--report-json must be a file beneath the project root")
    return str(normalized.relative_to(root))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_recording(data: bytes) -> tuple[bool, int]:
    try:
        events = load_events_from_bytes(data)
    except Exception:
        return False, 0
    return True, sum(1 for event in events if _is_replayable(event))


# --------------------------------------------------------------------------- #
# record --name (create-only; staged outside the tree, published via FD)
# --------------------------------------------------------------------------- #


def run_named_record(
    project: str | Path | None, name: str, command: list[str], report_override: str | Path | None
) -> NamedRunResult:
    try:
        root, config = _resolve_context(project)
        cassette_rel = _cassette_relative(config, name)
        report_rel = _report_relative(root, "record", name, report_override)
        arguments = validate_python_command(command)
        root_fd = open_root(root)
    except (
        MachineInputError,
        ProjectLoopError,
        ProjectInitError,
        RunnerUsageError,
        SecureFilesystemError,
    ) as error:
        return _invalid_named(project, "record", error)
    try:
        if not _report_parent_trusted(root_fd, report_rel):
            return _untrusted_report("record", root, report_rel)
        golden = root / cassette_rel
        try:
            already_exists = identity_of(root_fd, cassette_rel) is not None
        except SecureFilesystemError:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "record",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-unsafe", "path": str(golden)}]},
                ),
            )
        if already_exists:  # create-only pre-check
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "record",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-exists", "path": str(golden)}]},
                ),
            )
        staged, child_status, child_exception = _staged_record(arguments)
        try:
            if child_exception is not None:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
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
                    ),
                    child_exception=child_exception,
                )
            if child_status != 0:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "record",
                        "child-failed",
                        EXIT_CHANGES,
                        root,
                        golden,
                        name,
                        data={"child_exit_code": child_status},
                        next_actions=(_retry_record_action(name),),
                    ),
                )
            data = staged.read_bytes()  # staged file lives in our private 0o700 dir
            valid, replayable = _validate_recording(data)
            if not valid or replayable < 1:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "record",
                        "invalid",
                        EXIT_INVALID,
                        root,
                        golden,
                        name,
                        data={
                            "error_code": "empty-or-invalid-recording",
                            "replayable_events": replayable,
                        },
                    ),
                )
            try:
                publish_create_only(root_fd, cassette_rel, data)  # never overwrites
            except SecureFilesystemError:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "record",
                        "invalid",
                        EXIT_INVALID,
                        root,
                        golden,
                        name,
                        data={
                            "blockers": [{"code": "cassette-unsafe-or-exists", "path": str(golden)}]
                        },
                    ),
                )
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "record",
                    "current",
                    EXIT_OK,
                    root,
                    golden,
                    name,
                    data={
                        "cassette": {"name": name, "path": str(golden), "sha256": _sha256(data)},
                        "replayable_events": replayable,
                        "child_exit_code": 0,
                    },
                    next_actions=(_offline_replay_action(name),),
                ),
            )
        finally:
            _cleanup_staging(staged)
    finally:
        os.close(root_fd)


# --------------------------------------------------------------------------- #
# replay --name (offline; golden snapshotted once, structured mismatch)
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
        cassette_rel = _cassette_relative(config, name)
        report_rel = _report_relative(root, "replay", name, report_override)
        arguments = validate_python_command(command)
        root_fd = open_root(root)
    except (
        MachineInputError,
        ProjectLoopError,
        ProjectInitError,
        RunnerUsageError,
        SecureFilesystemError,
    ) as error:
        return _invalid_named(project, "replay", error)
    try:
        if not _report_parent_trusted(root_fd, report_rel):
            return _untrusted_report("replay", root, report_rel)
        golden = root / cassette_rel
        try:
            snap = snapshot(root_fd, cassette_rel)
        except SecureFilesystemError:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "replay",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-unsafe", "path": str(golden)}]},
                ),
            )
        if snap is None:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "replay",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-missing", "path": str(golden)}]},
                ),
            )
        data, _identity = snap
        cassette_summary = {"name": name, "path": str(golden), "sha256": _sha256(data)}
        valid, _count = _validate_recording(data)  # reject a corrupt golden before staging
        if not valid:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "replay",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-invalid", "path": str(golden)}]},
                ),
            )
        replayer, mismatch, child_exception, child_status = _staged_replay(
            data,
            arguments,
            match or config.match,
            strict if strict is not None else config.strict,
        )
        # counts come from the Replayer, whose events are already _is_replayable-filtered
        replayable = len(replayer.events)
        remaining = replayer.remaining
        consumed = replayable - remaining
        if child_exception is not None:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
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
                ),
                child_exception=child_exception,
            )
        if mismatch is not None:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "replay",
                    "mismatch",
                    EXIT_CHANGES,
                    root,
                    golden,
                    name,
                    data={"cassette": cassette_summary, "failure": _failure_data(mismatch)},
                    next_actions=_mismatch_actions(name, str(golden)),
                ),
            )
        if child_status != 0:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "replay",
                    "child-failed",
                    EXIT_CHANGES,
                    root,
                    golden,
                    name,
                    data={"cassette": cassette_summary, "child_exit_code": child_status},
                ),
            )
        return _finish(
            root_fd,
            report_rel,
            _named_envelope(
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
            ),
        )
    finally:
        os.close(root_fd)


# --------------------------------------------------------------------------- #
# rerecord --name (explicit atomic golden update; original preserved on failure)
# --------------------------------------------------------------------------- #


def run_named_rerecord(
    project: str | Path | None, name: str, command: list[str], report_override: str | Path | None
) -> NamedRunResult:
    try:
        root, config = _resolve_context(project)
        cassette_rel = _cassette_relative(config, name)
        report_rel = _report_relative(root, "rerecord", name, report_override)
        arguments = validate_python_command(command)
        root_fd = open_root(root)
    except (
        MachineInputError,
        ProjectLoopError,
        ProjectInitError,
        RunnerUsageError,
        SecureFilesystemError,
    ) as error:
        return _invalid_named(project, "rerecord", error)
    try:
        if not _report_parent_trusted(root_fd, report_rel):
            return _untrusted_report("rerecord", root, report_rel)
        golden = root / cassette_rel
        try:
            original = snapshot(root_fd, cassette_rel)  # rejects non-regular/multi-link/symlink
        except SecureFilesystemError:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "rerecord",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-unsafe", "path": str(golden)}]},
                ),
            )
        if original is None:
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "rerecord",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-missing", "path": str(golden)}]},
                ),
            )
        original_bytes, original_identity = original
        old_valid, old_count = _validate_recording(original_bytes)
        if not old_valid:  # never execute live code against an invalid golden
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "rerecord",
                    "invalid",
                    EXIT_INVALID,
                    root,
                    golden,
                    name,
                    data={"blockers": [{"code": "cassette-invalid", "path": str(golden)}]},
                ),
            )
        old_sha = _sha256(original_bytes)
        staged, child_status, child_exception = _staged_record(arguments)
        try:
            if child_exception is not None:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
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
                    ),
                    child_exception=child_exception,
                )
            if child_status != 0:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "rerecord",
                        "child-failed",
                        EXIT_CHANGES,
                        root,
                        golden,
                        name,
                        data={"child_exit_code": child_status},
                    ),
                )
            data = staged.read_bytes()
            valid, replayable = _validate_recording(data)
            if not valid or replayable < 1:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "rerecord",
                        "invalid",
                        EXIT_INVALID,
                        root,
                        golden,
                        name,
                        data={
                            "error_code": "empty-or-invalid-recording",
                            "replayable_events": replayable,
                        },
                    ),
                )
            try:
                # revalidate the original identity immediately before the atomic replace
                atomic_replace(root_fd, cassette_rel, data, expect_identity=original_identity)
            except SecureFilesystemError:
                return _finish(
                    root_fd,
                    report_rel,
                    _named_envelope(
                        "rerecord",
                        "invalid",
                        EXIT_INVALID,
                        root,
                        golden,
                        name,
                        data={
                            "blockers": [
                                {"code": "cassette-changed-or-unsafe", "path": str(golden)}
                            ]
                        },
                    ),
                )
            return _finish(
                root_fd,
                report_rel,
                _named_envelope(
                    "rerecord",
                    "changed",
                    EXIT_OK,
                    root,
                    golden,
                    name,
                    data={
                        "cassette": {"name": name, "path": str(golden), "sha256": _sha256(data)},
                        "previous_sha256": old_sha,
                        "previous_replayable_events": old_count,
                        "replayable_events": replayable,
                        "child_exit_code": 0,
                    },
                ),
            )
        finally:
            _cleanup_staging(staged)
    finally:
        os.close(root_fd)


# --------------------------------------------------------------------------- #
# Staging (private, outside the consumer tree)
# --------------------------------------------------------------------------- #


def _staged_record(arguments: list[str]) -> tuple[Path, int | None, BaseException | None]:
    staging = Path(tempfile.mkdtemp(prefix="agent-cassette-record-"))  # mode 0o700
    staged = staging / "cassette.jsonl"
    child_status: int | None = None
    child_exception: BaseException | None = None
    try:
        with Cassette.record(staged) as recorder:
            child_status = run_python(arguments, recorder)
    except Exception as error:  # child code raised
        child_exception = error
    return staged, child_status, child_exception


def _staged_replay(
    data: bytes, arguments: list[str], match: str, strict: bool
) -> tuple[Any, ReplayMismatchError | None, BaseException | None, int | None]:
    staging = Path(tempfile.mkdtemp(prefix="agent-cassette-replay-"))
    staged = staging / "cassette.jsonl"
    mismatch: ReplayMismatchError | None = None
    child_exception: BaseException | None = None
    child_status: int | None = None
    try:
        staged.write_bytes(data)  # immutable private snapshot; consumer path never reopened
        replayer = Cassette.replay(staged, strict=strict, match=match)  # type: ignore[arg-type]
        try:
            with replayer:
                child_status = run_python(arguments, replayer)
        except ReplayMismatchError as error:
            mismatch = error
        except Exception as error:
            child_exception = error
    finally:
        _cleanup_staging(staged)
    return replayer, mismatch, child_exception, child_status


def _cleanup_staging(staged: Path) -> None:
    shutil.rmtree(staged.parent, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Reports (atomic, contained) and envelope helpers
# --------------------------------------------------------------------------- #


def _report_parent_trusted(root_fd: int, report_rel: str) -> bool:
    # Preflight the report's parent chain through the root FD; an untrusted (symlinked /
    # swapped) parent means we never write and never run the child.
    try:
        preflight_parent(root_fd, report_rel)
        return True
    except SecureFilesystemError:
        return False


def _finish(
    root_fd: int,
    report_rel: str,
    envelope: Envelope,
    *,
    child_exception: BaseException | None = None,
) -> NamedRunResult:
    root = Path(envelope.project)
    try:
        atomic_replace(root_fd, report_rel, envelope.to_json().encode("utf-8"))
        return NamedRunResult(envelope, root / report_rel, child_exception)
    except SecureFilesystemError:
        # The report location became untrusted after preflight: never write outside.
        return NamedRunResult(
            _untrusted_envelope(envelope.command, root, report_rel), None, child_exception
        )


def _untrusted_report(command: str, root: Path, report_rel: str) -> NamedRunResult:
    return NamedRunResult(_untrusted_envelope(command, root, report_rel), None)


def _untrusted_envelope(command: str, root: Path, report_rel: str) -> Envelope:
    return Envelope(
        command=command,
        status="invalid",
        exit_code=EXIT_INVALID,
        project=str(root),
        data={"blockers": [{"code": "report-destination-unsafe", "path": report_rel}]},
        next_actions=(_status_action(),),
    )


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


def _invalid_named(project: str | Path | None, command: str, error: Exception) -> NamedRunResult:
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
        next_actions=(_status_action(),),
    )
    return NamedRunResult(envelope, None)


def _error_code(error: Exception) -> str:
    if isinstance(error, MachineInputError):
        return "invalid-name"
    if isinstance(error, RunnerUsageError):
        return "invalid-command"
    if isinstance(error, SecureFilesystemError):
        return "project-unsafe"
    return "invalid-project"


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


def _status_action() -> NextAction:
    return NextAction(id="inspect-status", argv=("agent-cassette", "status", ".", "--json"))


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


def _mismatch_actions(name: str, cassette_path: str) -> tuple[NextAction, ...]:
    return (
        _status_action(),
        NextAction(
            id="retry-after-fix",
            argv=("agent-cassette", "replay", "--name", name, "--"),
            requires_child_command=True,
        ),
        NextAction(
            id="inspect-cassette", argv=("agent-cassette", "inspect", cassette_path, "--json")
        ),
        NextAction(
            id="rerecord-golden",
            argv=("agent-cassette", "rerecord", "--name", name, "--"),
            network="required",
            approval_required=True,
            requires_child_command=True,
        ),
    )
