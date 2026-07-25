"""Phase E project loop: setup, status, agent-manifest, and CI scaffolding.

Builds on :mod:`agent_cassette.project_init`'s directory-FD / no-follow preflight and
atomic-publish primitives (never a weaker ``Path.write_text`` path). Setup owns a small
set of files, records their SHA-256 in ``.agent-cassette/manifest.json``, and never
overwrites a file whose bytes differ from the generated content. Nothing here imports or
executes consumer code, reads environment values, discovers credentials, installs
dependencies, or runs a subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_cassette.machine import EXIT_CHANGES, EXIT_INVALID, EXIT_OK, Envelope, NextAction
from agent_cassette.project_init import (
    CONFIG_NAME,
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_PROVIDERS,
    SUPPORTED_TEST_FRAMEWORKS,
    ProjectConfig,
    ProjectInitError,
    _Action,
    _apply,
    _detect_integrations_with_warnings,
    _open_absolute_directory,
    _open_existing_components,
    _parse_project_config,
    _PlannedFile,
    _preflight,
    _read_optional_regular_at,
    _relative_parts,
    _render_config,
    _smoke_test,
)
from agent_cassette.replay import _is_replayable
from agent_cassette.storage import load_events_from_bytes

MANIFEST_DIR = ".agent-cassette"
MANIFEST_PATH = ".agent-cassette/manifest.json"
REPORTS_DIRNAME = "reports"
MANIFEST_SCHEMA_VERSION = 1
WORKFLOW_PATH = ".github/workflows/agent-cassette-replay.yml"
WORKFLOW_MARKER = "# agent-cassette: generated replay workflow v1"
_MATCH_MODES = ("exact", "subset", "normalized", "fuzzy")
_SETUP_MODE = Literal["dry-run", "apply", "check"]

# Runner support that is genuinely automatic vs. requiring an explicit wrapper/integration.
_AUTOMATIC_CAPTURE = ("openai", "anthropic", "openai-agents")
_EXPLICIT_CAPTURE = {
    "mistral": "wrap_mistral / patch_mistral",
    "gemini": "wrap_gemini / patch_gemini",
    "mcp": "wrap_mcp",
    "langchain": "wrap_langchain / wrap_langchain_tools / langchain_callback_handler",
}


class ProjectLoopError(ValueError):
    """Invalid Phase E project-loop input or unsafe state (envelope-rendered, no traceback)."""


@dataclass(frozen=True, slots=True)
class Overrides:
    """Explicit setup overrides; ``None`` means "use existing config or detection"."""

    providers: tuple[str, ...] | None = None
    frameworks: tuple[str, ...] | None = None
    test_frameworks: tuple[str, ...] | None = None
    cassette_dir: str | None = None
    match: str | None = None
    strict: bool | None = None


# --------------------------------------------------------------------------- #
# Project-root discovery
# --------------------------------------------------------------------------- #


def resolve_project_root(explicit: str | Path | None) -> Path:
    """Explicit ``--project`` wins; otherwise walk up from the cwd to the config."""
    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    current = Path.cwd().absolute()
    for candidate in (current, *current.parents):
        if os.path.lexists(candidate / CONFIG_NAME):
            return candidate
    return current


def _config_present(root: Path) -> bool:
    try:
        root_fd = _open_absolute_directory(root)
    except ProjectInitError:
        return False
    try:
        return _read_optional_regular_at(root_fd, CONFIG_NAME) is not None
    except ProjectInitError:
        return False
    finally:
        os.close(root_fd)


def _load_config(root: Path) -> tuple[ProjectConfig | None, tuple[str, ...]]:
    root_fd = _open_absolute_directory(root)
    try:
        existing = _read_optional_regular_at(root_fd, CONFIG_NAME)
    finally:
        os.close(root_fd)
    if existing is None:
        return None, ()
    config, warnings = _parse_project_config(existing[0])
    return config, warnings


# --------------------------------------------------------------------------- #
# Effective configuration and override conflict detection
# --------------------------------------------------------------------------- #


def _canonical_category(
    values: tuple[str, ...], supported: tuple[str, ...], label: str
) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values:
        if type(value) is not str or value not in supported:
            raise ProjectLoopError(f"unsupported {label}: choose from {', '.join(supported)}")
        if value not in cleaned:
            cleaned.append(value)
    return tuple(sorted(cleaned))


def _effective_config(
    existing: ProjectConfig | None, detected: dict[str, list[str]], overrides: Overrides
) -> tuple[ProjectConfig, list[str]]:
    """Resolve the config setup would write, and any override-vs-existing conflicts."""
    conflicts: list[str] = []

    def category(
        explicit: tuple[str, ...] | None,
        supported: tuple[str, ...],
        existing_values: tuple[str, ...],
        detected_values: list[str],
        label: str,
    ) -> tuple[str, ...]:
        if explicit is not None:
            chosen = _canonical_category(explicit, supported, label)
            if existing is not None and chosen != existing_values:
                conflicts.append(f"{label} override disagrees with existing {CONFIG_NAME}")
            return chosen
        if existing is not None:
            return existing_values
        return _canonical_category(tuple(detected_values), supported, label)

    providers = category(
        overrides.providers,
        SUPPORTED_PROVIDERS,
        existing.providers if existing else (),
        detected["providers"],
        "provider",
    )
    frameworks = category(
        overrides.frameworks,
        SUPPORTED_FRAMEWORKS,
        existing.frameworks if existing else (),
        detected["frameworks"],
        "framework",
    )
    test_frameworks = category(
        overrides.test_frameworks,
        SUPPORTED_TEST_FRAMEWORKS,
        existing.test_frameworks if existing else (),
        detected["test_frameworks"],
        "test-framework",
    )

    cassette_dir = _resolve_scalar(
        overrides.cassette_dir,
        existing.cassette_dir if existing else None,
        "tests/cassettes",
        existing is not None,
        "cassette_dir",
        conflicts,
        _validate_cassette_dir,
    )
    match = _resolve_scalar(
        overrides.match,
        existing.match if existing else None,
        "exact",
        existing is not None,
        "match",
        conflicts,
        _validate_match,
    )
    strict = _resolve_scalar(
        overrides.strict,
        existing.strict if existing else None,
        True,
        existing is not None,
        "strict",
        conflicts,
        lambda value: value,
    )

    config = ProjectConfig(
        cassette_dir=cassette_dir,
        match=match,
        strict=strict,
        providers=providers,
        frameworks=frameworks,
        test_frameworks=test_frameworks,
    )
    return config, conflicts


def _resolve_scalar(
    explicit: Any,
    existing_value: Any,
    default: Any,
    has_existing: bool,
    label: str,
    conflicts: list[str],
    validate: Any,
) -> Any:
    if explicit is not None:
        value = validate(explicit)
        if has_existing and value != existing_value:
            conflicts.append(f"{label} override disagrees with existing {CONFIG_NAME}")
        return value
    if has_existing:
        return existing_value
    return default


def _validate_cassette_dir(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or ".." in _relative_parts(value)
    ):
        raise ProjectLoopError("cassette_dir must be a non-empty relative path")
    return value


def _validate_match(value: str) -> str:
    if type(value) is not str or value not in _MATCH_MODES:
        raise ProjectLoopError(f"match must be one of: {', '.join(_MATCH_MODES)}")
    return value


# --------------------------------------------------------------------------- #
# Owned-file plan and manifest
# --------------------------------------------------------------------------- #


def _owned_files(config: ProjectConfig, *, include_ci: bool) -> tuple[tuple[str, str, str], ...]:
    """Return (relative_path, kind, generated_content) for every setup-owned file."""
    files = [
        (CONFIG_NAME, "config", _render_config(config)),
        (f"{config.cassette_dir}/.gitkeep", "gitkeep", ""),
        ("tests/test_agent_cassette_smoke.py", "smoke", _smoke_test()),
    ]
    if include_ci:
        files.append((WORKFLOW_PATH, "workflow", _render_workflow(config)))
    return tuple(files)


def _render_manifest(owned: tuple[tuple[str, str, str], ...]) -> str:
    entries = {
        relative: {"kind": kind, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        for relative, kind, content in owned
    }
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "files": entries}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _setup_plans(config: ProjectConfig, *, include_ci: bool) -> tuple[_PlannedFile, ...]:
    owned = _owned_files(config, include_ci=include_ci)
    plans = [_PlannedFile(relative, content) for relative, _kind, content in owned]
    plans.append(_PlannedFile(MANIFEST_PATH, _render_manifest(owned)))  # written last
    return tuple(plans)


def _render_workflow(config: ProjectConfig) -> str:
    uses_uv = True  # dependency install strategy is chosen statically at run time in CI
    del uses_uv
    return f"""{WORKFLOW_MARKER}
name: agent-cassette-replay

on:
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  replay:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ""
      ANTHROPIC_API_KEY: ""
      MISTRAL_API_KEY: ""
      GEMINI_API_KEY: ""
      GOOGLE_API_KEY: ""
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          if [ -f uv.lock ]; then
            pipx run uv sync --frozen --all-extras --dev
          else
            python -m pip install --upgrade pip
            python -m pip install . pytest agent-cassette
          fi
      - name: Verify Agent Cassette setup
        run: pipx run agent-cassette setup . --check --json
      - name: Replay tests offline
        run: pytest --cassette-mode=replay
"""


# --------------------------------------------------------------------------- #
# setup command
# --------------------------------------------------------------------------- #


def run_setup(
    project: str | Path | None, *, mode: _SETUP_MODE, overrides: Overrides, include_ci: bool
) -> Envelope:
    root = resolve_project_root(project)
    warnings: list[str] = []
    try:
        existing, config_warnings = _load_config(root)
        warnings.extend(config_warnings)
        detected, detection_warnings = _detect_integrations_with_warnings(root)
        warnings.extend(detection_warnings)
        config, conflicts = _effective_config(existing, detected, overrides)
        root_fd = _open_absolute_directory(root)
        try:
            # setup owns the workflow only when --github-ci is given or the manifest already
            # records it; a workflow created by the standalone `ci` command is not pulled
            # into setup's ledger (so a later `setup --check` never conflicts over it).
            if not include_ci:
                include_ci = _manifest_tracks_workflow(root_fd)
            if include_ci and "pytest" not in config.test_frameworks:
                # Never invent a test command; the replay workflow runs pytest specifically.
                return Envelope(
                    command="setup",
                    status="changes-needed",
                    exit_code=EXIT_CHANGES,
                    project=str(root),
                    warnings=tuple(warnings),
                    data={"blockers": [{"code": "pytest-not-configured", "path": CONFIG_NAME}]},
                )
            plans = _setup_plans(config, include_ci=include_ci)
            actions = _preflight(root_fd, plans)
            preflight_conflicts = [action for action in actions if action.status == "conflict"]
            if conflicts or preflight_conflicts:
                return _setup_envelope(
                    "conflict",
                    EXIT_INVALID,
                    root,
                    warnings,
                    actions,
                    conflicts,
                    preflight_conflicts,
                )
            changes = [action for action in actions if action.status == "create"]
            if mode == "dry-run":
                return _setup_envelope(
                    "would-change" if changes else "current",
                    EXIT_OK,
                    root,
                    warnings,
                    actions,
                    [],
                    [],
                )
            if mode == "check":
                if changes:
                    return _setup_envelope(
                        "changes-needed", EXIT_CHANGES, root, warnings, actions, [], []
                    )
                return _setup_envelope("current", EXIT_OK, root, warnings, actions, [], [])
            _apply(root_fd, plans, actions)
            status = "changed" if changes else "current"
            return _setup_envelope(status, EXIT_OK, root, warnings, actions, [], [])
        finally:
            os.close(root_fd)
    except (ProjectLoopError, ProjectInitError):
        # Fail closed with a stable code; never surface the underlying message/payload.
        return Envelope(
            command="setup",
            status="invalid",
            exit_code=EXIT_INVALID,
            project=str(root),
            warnings=tuple(warnings),
            data={"error_code": "invalid-setup"},
            next_actions=(_inspect_status_action(),),
            changes=(),
        )


def _manifest_tracks_workflow(root_fd: int) -> bool:
    manifest = _read_manifest(root_fd)  # raises on an invalid manifest (unsafe state)
    return manifest is not None and WORKFLOW_PATH in manifest["files"]


def _setup_envelope(
    status: str,
    exit_code: int,
    root: Path,
    warnings: list[str],
    actions: list[_Action],
    override_conflicts: list[str],
    preflight_conflicts: list[_Action],
) -> Envelope:
    changes = tuple(action.path for action in actions if action.status == "create")
    data: dict[str, Any] = {
        "files": {action.path: action.status for action in actions},
    }
    conflict_paths = [action.path for action in preflight_conflicts]
    if override_conflicts or preflight_conflicts:
        data["conflicts"] = sorted(override_conflicts) + sorted(conflict_paths)
        actions_out: tuple[NextAction, ...] = (_inspect_status_action(),)
    elif status in ("would-change", "changes-needed"):
        actions_out = (_apply_setup_action(),)
    else:
        actions_out = (_inspect_status_action(),)
    return Envelope(
        command="setup",
        status=status,
        exit_code=exit_code,
        project=str(root),
        warnings=tuple(warnings),
        changes=changes,
        next_actions=actions_out,
        data=data,
    )


def _apply_setup_action() -> NextAction:
    return NextAction(
        id="setup-apply",
        argv=("agent-cassette", "setup", ".", "--apply", "--json"),
        mutates=(MANIFEST_PATH, CONFIG_NAME),
        network="forbidden",
    )


def _inspect_status_action() -> NextAction:
    return NextAction(id="inspect-status", argv=("agent-cassette", "status", ".", "--json"))


# --------------------------------------------------------------------------- #
# Safe read helpers (read-only, no-follow)
# --------------------------------------------------------------------------- #


def _read_relative(root_fd: int, relative: str) -> bytes | None:
    """Read an owned file's bytes via the no-follow directory-FD path, or None if absent."""
    parts = _relative_parts(relative)
    parent_fd, _identities = _open_existing_components(root_fd, parts[:-1])
    if parent_fd is None:
        return None
    try:
        existing = _read_optional_regular_at(parent_fd, parts[-1])
    finally:
        os.close(parent_fd)
    return None if existing is None else existing[0]


def _read_manifest(root_fd: int) -> dict[str, Any] | None:
    try:
        raw = _read_relative(root_fd, MANIFEST_PATH)
    except ProjectInitError as error:
        # A symlinked/hard-linked/non-regular manifest is an unsafe state, not a parse error.
        raise ProjectLoopError("manifest is not a safe regular file") from error
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProjectLoopError("manifest is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ProjectLoopError("manifest schema is unsupported")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ProjectLoopError("manifest is missing a files table")
    return payload


# --------------------------------------------------------------------------- #
# status command
# --------------------------------------------------------------------------- #


def run_status(project: str | Path | None) -> Envelope:
    root = resolve_project_root(project)
    warnings: list[str] = []
    blockers: list[dict[str, str]] = []
    try:
        root_fd = _open_absolute_directory(root)
    except ProjectInitError:
        return Envelope(
            command="status",
            status="invalid",
            exit_code=EXIT_INVALID,
            project=str(root),
            warnings=tuple(warnings),
            data={"blockers": [{"code": "project-dir-unsafe", "path": "."}]},
        )
    try:
        existing, config_warnings, config_valid = _load_config_status(root_fd, blockers)
        warnings.extend(config_warnings)
        detected, detection_warnings = _detect_integrations_with_warnings(root)
        warnings.extend(detection_warnings)
        effective = existing if existing is not None else _detected_config(detected)
        try:
            manifest = _read_manifest(root_fd)
        except ProjectLoopError:
            manifest = None
            blockers.append({"code": "manifest-invalid", "path": MANIFEST_PATH})
        # managed files are the setup-owned set (from the manifest); a standalone `ci`
        # workflow is reported through ci_ready, not folded into setup readiness.
        owns_workflow = manifest is not None and WORKFLOW_PATH in manifest.get("files", {})
        managed = _managed_files(root_fd, effective, manifest, blockers, include_ci=owns_workflow)
        cassettes = _cassette_inventory(root_fd, effective, blockers)
        workflow_current = _workflow_matches(root_fd, effective, blockers)
        readiness = _readiness(existing, manifest, managed, cassettes, workflow_current, effective)
        data = {
            "config": {
                "present": existing is not None,
                "valid": config_valid,
                "cassette_dir": effective.cassette_dir,
                "match": effective.match,
                "strict": effective.strict,
                "providers": {
                    "configured": list(effective.providers),
                    "detected": detected["providers"],
                },
                "frameworks": {
                    "configured": list(effective.frameworks),
                    "detected": detected["frameworks"],
                },
                "test_frameworks": {
                    "configured": list(effective.test_frameworks),
                    "detected": detected["test_frameworks"],
                },
            },
            "managed_files": managed,
            "cassettes": cassettes,
            "capture_coverage": _capture_coverage(effective),
            "readiness": readiness,
            "blockers": _unique_blockers(blockers),
        }
    finally:
        os.close(root_fd)

    if any(blocker["code"] in _INVALID_BLOCKERS for blocker in blockers):
        status, exit_code = "invalid", EXIT_INVALID
    elif readiness["setup_ready"]:
        status, exit_code = "current", EXIT_OK
    else:
        status, exit_code = "changes-needed", EXIT_CHANGES
    actions = () if exit_code == EXIT_OK else (_apply_setup_action(),)
    return Envelope(
        command="status",
        status=status,
        exit_code=exit_code,
        project=str(root),
        warnings=tuple(warnings),
        next_actions=actions,
        data=data,
    )


def _unique_blockers(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for blocker in blockers:
        key = (blocker["code"], blocker.get("path", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(blocker)
    return sorted(unique, key=lambda item: (item["code"], item.get("path", "")))


_INVALID_BLOCKERS = frozenset(
    {
        "config-invalid",
        "config-unsafe",
        "manifest-invalid",
        "cassette-dir-unsafe",
        "cassette-file-unsafe",
        "managed-file-unsafe",
        "workflow-unsafe",
    }
)


def _load_config_status(
    root_fd: int, blockers: list[dict[str, str]]
) -> tuple[ProjectConfig | None, tuple[str, ...], bool]:
    try:
        existing = _read_optional_regular_at(root_fd, CONFIG_NAME)
    except ProjectInitError:
        # A symlinked/hard-linked/non-regular config cannot be trusted; never read it.
        blockers.append({"code": "config-unsafe", "path": CONFIG_NAME})
        return None, (), False
    if existing is None:
        return None, (), True
    try:
        config, warnings = _parse_project_config(existing[0])
    except ProjectInitError:
        blockers.append({"code": "config-invalid", "path": CONFIG_NAME})
        return None, (), False
    return config, warnings, True


def _detected_config(detected: dict[str, list[str]]) -> ProjectConfig:
    return ProjectConfig(
        providers=tuple(detected["providers"]),
        frameworks=tuple(detected["frameworks"]),
        test_frameworks=tuple(detected["test_frameworks"]),
    )


def _managed_files(
    root_fd: int,
    config: ProjectConfig,
    manifest: dict[str, Any] | None,
    blockers: list[dict[str, str]],
    *,
    include_ci: bool,
) -> dict[str, dict[str, str]]:
    recorded = manifest["files"] if manifest else {}
    result: dict[str, dict[str, str]] = {}
    for relative, kind, _content in _owned_files(config, include_ci=include_ci):
        try:
            on_disk = _read_relative(root_fd, relative)
        except ProjectInitError:
            # A symlinked/hard-linked managed file is never adopted as "current"; setup's
            # apply fails closed on it. Report it, don't silently drop it.
            blockers.append({"code": "managed-file-unsafe", "path": relative})
            result[relative] = {"kind": kind, "state": "modified"}
            continue
        recorded_sha = (
            recorded.get(relative, {}).get("sha256") if isinstance(recorded, dict) else None
        )
        if on_disk is None:
            state = "missing"
        elif recorded_sha is None:
            state = "unmanaged"
        elif hashlib.sha256(on_disk).hexdigest() == recorded_sha:
            state = "current"
        else:
            state = "modified"
        result[relative] = {"kind": kind, "state": state}
    return result


def _cassette_inventory(
    root_fd: int, config: ProjectConfig, blockers: list[dict[str, str]]
) -> list[dict[str, Any]]:
    parts = _relative_parts(config.cassette_dir)
    parent_fd, _identities = _open_existing_components(root_fd, parts)
    if parent_fd is None:
        return []
    try:
        try:
            names = sorted(os.listdir(parent_fd))
        except OSError:
            blockers.append({"code": "cassette-dir-unsafe", "path": config.cassette_dir})
            return []
        inventory: list[dict[str, Any]] = []
        for entry in names:
            if not entry.endswith(".jsonl"):
                continue
            try:
                existing = _read_optional_regular_at(parent_fd, entry)
            except ProjectInitError:
                # A symlinked/hard-linked cassette cannot be trusted; record it as invalid
                # (no hash, zero replayable events) and block, never silently omit it.
                blockers.append({"code": "cassette-file-unsafe", "path": "/".join((*parts, entry))})
                inventory.append(
                    {"name": entry[: -len(".jsonl")], "valid": False, "replayable_events": 0}
                )
                continue
            if existing is None:
                continue
            inventory.append(_describe_cassette(entry, existing[0]))
        return inventory
    finally:
        os.close(parent_fd)


def _describe_cassette(entry: str, raw: bytes) -> dict[str, Any]:
    # Validate/count/hash the exact bytes already read through the directory FD; never
    # reopen the cassette by path (a swapped path could feed a different file to loader).
    name = entry[: -len(".jsonl")]
    record: dict[str, Any] = {
        "name": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        events = load_events_from_bytes(raw)  # fails closed on corruption
    except Exception:
        record["valid"] = False
        record["replayable_events"] = 0
        return record
    record["valid"] = True
    record["replayable_events"] = sum(1 for event in events if _is_replayable(event))
    return record


def _capture_coverage(config: ProjectConfig) -> dict[str, Any]:
    configured = set(config.providers) | set(config.frameworks)
    automatic = sorted(name for name in _AUTOMATIC_CAPTURE if name in configured)
    explicit = {
        name: _EXPLICIT_CAPTURE[name] for name in sorted(configured) if name in _EXPLICIT_CAPTURE
    }
    return {"automatic": automatic, "requires_explicit_integration": explicit}


def _workflow_matches(root_fd: int, config: ProjectConfig, blockers: list[dict[str, str]]) -> bool:
    try:
        on_disk = _read_relative(root_fd, WORKFLOW_PATH)
    except ProjectInitError:
        blockers.append({"code": "workflow-unsafe", "path": WORKFLOW_PATH})
        return False
    return on_disk is not None and on_disk == _render_workflow(config).encode("utf-8")


def _readiness(
    existing: ProjectConfig | None,
    manifest: dict[str, Any] | None,
    managed: dict[str, dict[str, str]],
    cassettes: list[dict[str, Any]],
    workflow_current: bool,
    config: ProjectConfig,
) -> dict[str, bool]:
    managed_ok = manifest is not None and all(
        entry["state"] == "current" for entry in managed.values()
    )
    setup_ready = existing is not None and managed_ok
    replay_ready = any(
        item.get("valid") and item.get("replayable_events", 0) > 0 for item in cassettes
    )
    ci_ready = workflow_current and bool(config.test_frameworks)
    return {
        "setup_ready": setup_ready,
        "record_ready": setup_ready,
        "replay_ready": replay_ready,
        "ci_ready": ci_ready,
    }


# --------------------------------------------------------------------------- #
# agent-manifest command
# --------------------------------------------------------------------------- #


def run_agent_manifest(project: str | Path | None) -> Envelope:
    root = resolve_project_root(project)
    data: dict[str, Any] = {
        "commands": _AGENT_MANIFEST_COMMANDS,
        "exit_codes": {
            "0": "success, current, or dry-run result",
            "1": "check needs changes, replay mismatch, or child returned nonzero",
            "2": "invalid input/config/cassette/report path, conflict, or unsafe state",
        },
        "next_action_schema": {
            "id": "stable-kebab-id",
            "argv": ["argument", "vector", "never", "a", "shell", "string"],
            "mutates": ["relative", "paths"],
            "network": "forbidden|allowed|required",
            "approval_required": "bool",
            "requires_child_command": "bool (argv ends at -- ; caller owns the child command)",
        },
        "owned_file_rules": [
            "setup owns .agent-cassette.toml, the cassette dir .gitkeep, the smoke test, "
            "the manifest, and (with --github-ci) the replay workflow",
            "a file whose bytes differ from the manifest hash is never overwritten (conflict)",
            "a missing owned file may be created; existing valid config is authoritative",
        ],
        "capture_coverage": {
            "automatic": list(_AUTOMATIC_CAPTURE),
            "requires_explicit_integration": dict(_EXPLICIT_CAPTURE),
        },
        "safety": [
            "setup/status/ci never run consumer code, read env values, or install dependencies",
            "named record is live; replay is offline only at supported wrapped/bridged boundaries",
            "reports are a separate machine channel; child stdout/stderr is untouched",
            "rerecord is the only explicit golden-update path; it never runs implicitly",
        ],
    }
    # Include the current status summary when a config exists, but stay useful uninitialized.
    # Config presence is checked through the secure root FD, never os.path.lexists.
    if _config_present(root):
        summary = run_status(project)
        data["project_status"] = {
            "status": summary.status,
            "readiness": summary.data.get("readiness", {}),
        }
    return Envelope(
        command="agent-manifest",
        status="current",
        exit_code=EXIT_OK,
        project=str(root),
        data=data,
    )


_AGENT_MANIFEST_COMMANDS: dict[str, list[str]] = {
    "setup": ["--dry-run", "--apply", "--check", "--json", "--github-ci"],
    "status": ["--json"],
    "agent-manifest": ["--json"],
    "record": ["--name", "--project", "--report-json", "--"],
    "replay": ["--name", "--project", "--report-json", "--match", "--strict", "--no-strict", "--"],
    "rerecord": ["--name", "--project", "--report-json", "--"],
    "ci": ["--github", "--dry-run", "--apply", "--check", "--json"],
}


# --------------------------------------------------------------------------- #
# ci command (standalone replay-only workflow scaffold)
# --------------------------------------------------------------------------- #


def run_ci(project: str | Path | None, *, mode: _SETUP_MODE) -> Envelope:
    root = resolve_project_root(project)
    warnings: list[str] = []
    try:
        existing, config_warnings = _load_config(root)
        warnings.extend(config_warnings)
        if existing is None:
            return _ci_blocker(root, warnings, "config-missing", CONFIG_NAME)
        if "pytest" not in existing.test_frameworks:
            return _ci_blocker(root, warnings, "pytest-not-configured", CONFIG_NAME)
        plan = (_PlannedFile(WORKFLOW_PATH, _render_workflow(existing)),)
        root_fd = _open_absolute_directory(root)
        try:
            actions = _preflight(root_fd, plan)
            conflicts = [action for action in actions if action.status == "conflict"]
            changes = [action for action in actions if action.status == "create"]
            if conflicts:
                return _ci_envelope("conflict", EXIT_INVALID, root, warnings, actions)
            if mode == "dry-run":
                return _ci_envelope(
                    "would-change" if changes else "current", EXIT_OK, root, warnings, actions
                )
            if mode == "check":
                if changes:
                    return _ci_envelope("changes-needed", EXIT_CHANGES, root, warnings, actions)
                return _ci_envelope("current", EXIT_OK, root, warnings, actions)
            _apply(root_fd, plan, actions)
            return _ci_envelope(
                "changed" if changes else "current", EXIT_OK, root, warnings, actions
            )
        finally:
            os.close(root_fd)
    except (ProjectLoopError, ProjectInitError):
        return _ci_blocker(root, warnings, "invalid-ci", WORKFLOW_PATH)


def _ci_envelope(
    status: str, exit_code: int, root: Path, warnings: list[str], actions: list[_Action]
) -> Envelope:
    changes = tuple(action.path for action in actions if action.status == "create")
    data = {"files": {action.path: action.status for action in actions}}
    if status in ("would-change", "changes-needed"):
        next_actions: tuple[NextAction, ...] = (
            NextAction(
                id="ci-apply",
                argv=("agent-cassette", "ci", ".", "--github", "--apply", "--json"),
                mutates=(WORKFLOW_PATH,),
            ),
        )
    else:
        next_actions = ()
    return Envelope(
        command="ci",
        status=status,
        exit_code=exit_code,
        project=str(root),
        warnings=tuple(warnings),
        changes=changes,
        next_actions=next_actions,
        data=data,
    )


def _ci_blocker(root: Path, warnings: list[str], code: str, path: str) -> Envelope:
    exit_code = EXIT_INVALID if code in ("invalid-ci", "config-missing") else EXIT_CHANGES
    return Envelope(
        command="ci",
        status="invalid" if exit_code == EXIT_INVALID else "changes-needed",
        exit_code=exit_code,
        project=str(root),
        warnings=tuple(warnings),
        data={"blockers": [{"code": code, "path": path}]},
    )
