"""Safe, deterministic project initialization for Agent Cassette."""

from __future__ import annotations

import ast
import configparser
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

try:  # pragma: no cover - selected by the running Python version
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


CONFIG_NAME = ".agent-cassette.toml"
CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SUPPORTED_PROVIDERS = ("anthropic", "mcp", "openai", "openai-agents")
SUPPORTED_FRAMEWORKS = ("langchain",)
SUPPORTED_TEST_FRAMEWORKS = ("pytest", "unittest")
_MATCH_MODES = ("exact", "fuzzy", "normalized", "subset")
_KNOWN_CONFIG_KEYS = {
    "schema_version",
    "cassette_dir",
    "match",
    "strict",
    "providers",
    "frameworks",
    "test_frameworks",
}
_SMOKE_MARKER = "# agent-cassette: generated smoke test v1"
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


class ProjectInitError(ValueError):
    """Raised when a project cannot be initialized safely."""


@dataclass(frozen=True)
class ProjectConfig:
    """Validated schema-v1 project configuration."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    cassette_dir: str = "tests/cassettes"
    match: str = "exact"
    strict: bool = True
    providers: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    test_frameworks: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PlannedFile:
    relative: str
    content: str


@dataclass(frozen=True)
class _Action:
    path: str
    status: Literal["create", "unchanged", "conflict"]
    reason: str | None = None
    parent_identities: tuple[tuple[tuple[str, ...], tuple[int, int]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"path": self.path, "status": self.status}
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass
class _CreatedFile:
    parent_fd: int
    object_fd: int
    name: str
    identity: tuple[int, int]


@dataclass
class _CreatedDirectory:
    parent_fd: int
    object_fd: int
    name: str
    identity: tuple[int, int]


def load_project_config(path: str | Path) -> tuple[ProjectConfig, tuple[str, ...]]:
    """Load and validate a project config, returning forward-compatible warnings."""
    config_path = Path(path).expanduser().absolute()
    data = _read_config_path(config_path)
    return _parse_project_config(data, config_path.name)


def load_project_config_from_root(
    root: str | Path,
) -> tuple[ProjectConfig, tuple[str, ...]] | None:
    """Load a root project config safely, or return ``None`` when it is absent."""
    root_path = Path(root).expanduser().absolute()
    config_path = root_path / CONFIG_NAME
    if not os.path.lexists(config_path):
        return None
    data = _read_config_path(config_path)
    return _parse_project_config(data)


def _read_config_path(config_path: Path) -> bytes:
    if not _secure_primitives_available():
        return _read_regular_portable(config_path)
    parent_fd = _open_absolute_directory(config_path.parent)
    try:
        data, _ = _read_regular_at(parent_fd, config_path.name)
    finally:
        os.close(parent_fd)
    return data


def _read_regular_portable(path: Path) -> bytes:
    """Read data only, rejecting symlinks/non-files before and after open."""
    try:
        before = os.lstat(path)
    except (OSError, ValueError) as error:
        raise ProjectInitError(f"cannot inspect {path.name}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ProjectInitError(f"{path.name} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as error:
        raise ProjectInitError(f"cannot safely open {path.name}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
            raise ProjectInitError(f"{path.name} changed while it was being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    except OSError as error:
        raise ProjectInitError(f"cannot safely read {path.name}: {error}") from error
    finally:
        os.close(descriptor)


def _parse_project_config(
    data: bytes, display_name: str = CONFIG_NAME
) -> tuple[ProjectConfig, tuple[str, ...]]:
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ProjectInitError(f"cannot read {display_name}: {error}") from error
    if not isinstance(raw, dict):
        raise ProjectInitError(f"{display_name} must contain a TOML table")

    unknown = sorted(set(raw) - _KNOWN_CONFIG_KEYS)
    warnings = tuple(f"unknown configuration key ignored: {key}" for key in unknown)
    schema = raw.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ProjectInitError("schema_version must be the integer 1")
    if schema > CONFIG_SCHEMA_VERSION:
        raise ProjectInitError(
            f"schema_version {schema} is newer than supported version {CONFIG_SCHEMA_VERSION}"
        )
    if schema != CONFIG_SCHEMA_VERSION:
        raise ProjectInitError(f"unsupported schema_version: {schema}")

    cassette_dir = raw.get("cassette_dir")
    if not isinstance(cassette_dir, str) or not cassette_dir:
        raise ProjectInitError("cassette_dir must be a non-empty relative path")
    _validate_relative_config_path(cassette_dir)
    match = raw.get("match")
    if not isinstance(match, str) or match not in _MATCH_MODES:
        raise ProjectInitError(f"match must be one of: {', '.join(_MATCH_MODES)}")
    strict = raw.get("strict")
    if not isinstance(strict, bool):
        raise ProjectInitError("strict must be true or false")
    providers = _string_list(raw.get("providers"), "providers", SUPPORTED_PROVIDERS)
    frameworks = _string_list(raw.get("frameworks"), "frameworks", SUPPORTED_FRAMEWORKS)
    test_frameworks = _string_list(
        raw.get("test_frameworks", []), "test_frameworks", SUPPORTED_TEST_FRAMEWORKS
    )
    return (
        ProjectConfig(
            schema_version=schema,
            cassette_dir=cassette_dir,
            match=match,
            strict=strict,
            providers=providers,
            frameworks=frameworks,
            test_frameworks=test_frameworks,
        ),
        warnings,
    )


def initialize_project(
    project: str | Path,
    *,
    detect: bool = False,
    mode: Literal["apply", "dry-run", "check"] = "apply",
) -> tuple[dict[str, Any], int]:
    """Plan or apply project scaffolding and return a stable report plus exit code."""
    if mode not in ("apply", "dry-run", "check"):
        raise ValueError(f"unsupported initialization mode: {mode}")
    lexical_root = Path(project).expanduser().absolute()
    warnings: list[str] = []
    root_fd: int | None = None
    try:
        root_fd = _open_absolute_directory(lexical_root)
        root = lexical_root
        if detect:
            detected, detection_warnings = _detect_integrations_with_warnings(root)
            warnings.extend(detection_warnings)
        else:
            detected = {"providers": [], "frameworks": [], "test_frameworks": []}
        existing_data = _read_optional_regular_at(root_fd, CONFIG_NAME)
        if existing_data is not None:
            config_bytes, config_identity = existing_data
            existing_config, config_warnings = _parse_project_config(config_bytes)
            warnings.extend(config_warnings)
            config = existing_config
        else:
            config_bytes = None
            config_identity = None
            config = ProjectConfig(
                providers=tuple(detected["providers"]),
                frameworks=tuple(detected["frameworks"]),
                test_frameworks=tuple(detected["test_frameworks"]),
            )
        plans = _scaffold(config, include_config=config_bytes is None)
        actions = _preflight(root_fd, plans)
        if config_bytes is not None:
            actions.insert(0, _Action(CONFIG_NAME, "unchanged"))
    except ProjectInitError as error:
        if root_fd is not None:
            os.close(root_fd)
        report = _report(
            lexical_root,
            mode=mode,
            detect=detect,
            detected={"providers": [], "frameworks": [], "test_frameworks": []},
            actions=(),
            warnings=warnings,
            status="invalid",
            error=str(error),
        )
        return report, 2

    conflicts = [action for action in actions if action.status == "conflict"]
    changes = [action for action in actions if action.status == "create"]
    if conflicts:
        status = "conflict"
        exit_code = 2
    elif mode == "check":
        status = "changes-needed" if changes else "current"
        exit_code = 1 if changes else 0
    elif mode == "dry-run":
        status = "would-change" if changes else "current"
        exit_code = 0
    else:
        try:
            assert root_fd is not None
            if config_identity is not None:
                current_config = _read_optional_regular_at(root_fd, CONFIG_NAME)
                if (
                    current_config is None
                    or current_config[0] != config_bytes
                    or current_config[1] != config_identity
                ):
                    raise ProjectInitError(f"{CONFIG_NAME} changed after preflight")
            _apply(root_fd, plans, actions)
        except ProjectInitError as error:
            report = _report(
                root,
                mode=mode,
                detect=detect,
                detected=detected,
                actions=actions,
                warnings=warnings,
                status="invalid",
                error=str(error),
            )
            os.close(root_fd)
            return report, 2
        except BaseException:
            os.close(root_fd)
            raise
        status = "changed" if changes else "current"
        exit_code = 0

    assert root_fd is not None
    os.close(root_fd)
    return (
        _report(
            root,
            mode=mode,
            detect=detect,
            detected=detected,
            actions=actions,
            warnings=warnings,
            status=status,
        ),
        exit_code,
    )


def detect_integrations(root: str | Path) -> dict[str, list[str]]:
    """Detect supported integrations using static project-manifest parsing only."""
    detected, _ = _detect_integrations_with_warnings(Path(root).expanduser().absolute())
    return detected


def _detect_integrations_with_warnings(
    project_root: Path,
) -> tuple[dict[str, list[str]], list[str]]:
    names: set[str] = set()
    warnings: list[str] = []
    source_test_frameworks: set[str] = set()
    root_fd = _open_absolute_directory(project_root)
    try:
        entries = sorted(os.listdir(root_fd))
        manifests = [
            name
            for name in entries
            if name in {"pyproject.toml", "setup.cfg", "setup.py"}
            or (name.startswith("requirements") and name.endswith(".txt"))
        ]
        for name in manifests:
            try:
                data, _ = _read_regular_at(root_fd, name)
                if name == "pyproject.toml":
                    names.update(_dependencies_from_pyproject_bytes(data, name))
                elif name.startswith("requirements"):
                    names.update(_dependencies_from_requirements_bytes(data, name))
                elif name == "setup.cfg":
                    names.update(
                        _dependencies_from_setup_cfg_text(_decode_manifest(data, name), name)
                    )
                else:
                    names.update(
                        _dependencies_from_setup_py_text(_decode_manifest(data, name), name)
                    )
            except ProjectInitError as error:
                warnings.append(f"manifest {name} ignored: {error}")
        source_test_frameworks, source_warnings = _detect_test_source_frameworks(root_fd, entries)
        warnings.extend(source_warnings)
    except OSError as error:
        warnings.append(f"project manifest listing failed: {error}")
    finally:
        os.close(root_fd)

    providers: set[str] = set()
    frameworks: set[str] = set()
    test_frameworks = set(source_test_frameworks)
    for name in names:
        normalized = _normalize_dependency(name)
        if normalized == "openai":
            providers.add("openai")
        elif normalized == "anthropic":
            providers.add("anthropic")
        elif normalized in {"mcp", "mcp-sdk"}:
            providers.add("mcp")
        elif normalized == "openai-agents":
            providers.add("openai-agents")
        if normalized == "langchain" or normalized.startswith("langchain-"):
            frameworks.add("langchain")
        if normalized == "pytest" or normalized.startswith("pytest-"):
            test_frameworks.add("pytest")
    return (
        {
            "providers": sorted(providers),
            "frameworks": sorted(frameworks),
            "test_frameworks": sorted(test_frameworks),
        },
        sorted(warnings),
    )


def _detect_test_source_frameworks(
    root_fd: int, root_entries: list[str]
) -> tuple[set[str], list[str]]:
    frameworks: set[str] = set()
    warnings: list[str] = []
    for name in root_entries:
        if name.startswith("test") and name.endswith(".py"):
            _inspect_test_source(root_fd, name, name, frameworks, warnings)
    try:
        tests_fd = _open_directory_at(root_fd, "tests")
    except FileNotFoundError:
        return frameworks, warnings
    except OSError as error:
        warnings.append(f"test source directory tests ignored: {error}")
        return frameworks, warnings
    try:
        _scan_test_directory(tests_fd, ("tests",), frameworks, warnings, depth=0)
    finally:
        os.close(tests_fd)
    return frameworks, warnings


def _scan_test_directory(
    directory_fd: int,
    prefix: tuple[str, ...],
    frameworks: set[str],
    warnings: list[str],
    *,
    depth: int,
) -> None:
    if depth > 8:
        warnings.append(f"test source directory {'/'.join(prefix)} ignored: nesting too deep")
        return
    try:
        entries = sorted(os.listdir(directory_fd))
    except OSError as error:
        warnings.append(f"test source directory {'/'.join(prefix)} ignored: {error}")
        return
    for name in entries:
        relative = "/".join((*prefix, name))
        try:
            item_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            warnings.append(f"test source {relative} ignored: {error}")
            continue
        if stat.S_ISDIR(item_stat.st_mode):
            try:
                child = _open_directory_at(directory_fd, name)
            except OSError as error:
                warnings.append(f"test source directory {relative} ignored: {error}")
                continue
            try:
                _scan_test_directory(
                    child,
                    (*prefix, name),
                    frameworks,
                    warnings,
                    depth=depth + 1,
                )
            finally:
                os.close(child)
        elif name.startswith("test") and name.endswith(".py"):
            _inspect_test_source(directory_fd, name, relative, frameworks, warnings)


def _inspect_test_source(
    parent_fd: int,
    name: str,
    relative: str,
    frameworks: set[str],
    warnings: list[str],
) -> None:
    try:
        data, _ = _read_regular_at(parent_fd, name)
        text = _decode_manifest(data, relative)
        tree = ast.parse(text, filename=relative)
    except (ProjectInitError, SyntaxError) as error:
        message = error.msg if isinstance(error, SyntaxError) else str(error)
        warnings.append(f"test source {relative} ignored: {message}")
        return
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            root_module = module.split(".", 1)[0]
            if root_module in SUPPORTED_TEST_FRAMEWORKS:
                frameworks.add(root_module)


def render_init_report(report: dict[str, Any], *, as_json: bool) -> str:
    """Render a deterministic machine or human initialization report."""
    if as_json:
        return json.dumps(report, indent=2, sort_keys=True)
    lines = [f"Agent Cassette init: {report['status']}", f"Project: {report['project']}"]
    detected = report["detected"]
    providers = ", ".join(detected["providers"]) or "none"
    frameworks = ", ".join(detected["frameworks"]) or "none"
    test_frameworks = ", ".join(detected["test_frameworks"]) or "none"
    lines.extend(
        (
            f"Detected providers: {providers}",
            f"Detected frameworks: {frameworks}",
            f"Detected test frameworks: {test_frameworks}",
        )
    )
    for action in report["actions"]:
        line = f"  {action['status']}: {action['path']}"
        if "reason" in action:
            line += f" ({action['reason']})"
        lines.append(line)
    for warning in report["warnings"]:
        lines.append(f"Warning: {warning}")
    if report["error"] is not None:
        lines.append(f"Error: {report['error']}")
    if report["next_steps"]:
        lines.append("Next steps:")
        lines.extend(f"  {step}" for step in report["next_steps"])
    return "\n".join(lines)


def _report(
    root: Path,
    *,
    mode: str,
    detect: bool,
    detected: dict[str, list[str]],
    actions: tuple[_Action, ...] | list[_Action],
    warnings: list[str],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    next_steps: list[str] = []
    if status not in ("invalid", "conflict"):
        next_steps.append("uv run pytest tests/test_agent_cassette_smoke.py")
        if "langchain" in detected["frameworks"]:
            next_steps.append("wrap a Runnable with agent_cassette.wrap_langchain")
        if detected["providers"]:
            next_steps.append("record one provider call, then rerun it in replay mode")
    return {
        "actions": [action.to_dict() for action in actions],
        "command": "init",
        "detect": detect,
        "detected": {
            "frameworks": sorted(detected["frameworks"]),
            "providers": sorted(detected["providers"]),
            "test_frameworks": sorted(detected["test_frameworks"]),
        },
        "error": error,
        "mode": mode,
        "next_steps": next_steps,
        "project": str(root),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "warnings": sorted(warnings),
    }


def _scaffold(config: ProjectConfig, *, include_config: bool) -> tuple[_PlannedFile, ...]:
    plans: list[_PlannedFile] = []
    if include_config:
        plans.append(_PlannedFile(CONFIG_NAME, _render_config(config)))
    plans.extend(
        (
            _PlannedFile(f"{config.cassette_dir}/.gitkeep", ""),
            _PlannedFile(
                "tests/test_agent_cassette_smoke.py",
                _smoke_test(),
            ),
        )
    )
    return tuple(plans)


def _render_config(config: ProjectConfig) -> str:
    providers = ", ".join(json.dumps(value) for value in sorted(config.providers))
    frameworks = ", ".join(json.dumps(value) for value in sorted(config.frameworks))
    test_frameworks = ", ".join(json.dumps(value) for value in sorted(config.test_frameworks))
    return (
        "# Generated by agent-cassette init. Safe to edit.\n"
        f"schema_version = {config.schema_version}\n"
        f"cassette_dir = {json.dumps(config.cassette_dir)}\n"
        f"match = {json.dumps(config.match)}\n"
        f"strict = {'true' if config.strict else 'false'}\n"
        f"providers = [{providers}]\n"
        f"frameworks = [{frameworks}]\n"
        f"test_frameworks = [{test_frameworks}]\n"
    )


def _smoke_test() -> str:
    return f'''{_SMOKE_MARKER}
"""Offline record/replay smoke test generated by Agent Cassette."""

from agent_cassette import Cassette, EventType


def test_agent_cassette_offline_round_trip(tmp_path):
    cassette_path = tmp_path / "agent-cassette-smoke.jsonl"
    request = {{"prompt": "hello"}}

    with Cassette.record(cassette_path) as recorder:
        recorded = recorder.call(
            EventType.CUSTOM,
            "smoke",
            request,
            lambda: {{"answer": "world"}},
        )

    live_called = False

    def live_call():
        nonlocal live_called
        live_called = True
        raise AssertionError("offline replay called live code")

    with Cassette.replay(cassette_path) as replayer:
        replayed = replayer.call(EventType.CUSTOM, "smoke", request, live_call)

    assert recorded == replayed == {{"answer": "world"}}
    assert live_called is False
'''


def _preflight(root_fd: int, plans: tuple[_PlannedFile, ...]) -> list[_Action]:
    actions: list[_Action] = []
    for plan in plans:
        parts = _relative_parts(plan.relative)
        try:
            parent_fd, parent_identities = _open_existing_components(root_fd, parts[:-1])
            if parent_fd is None:
                actions.append(
                    _Action(
                        plan.relative,
                        "create",
                        parent_identities=parent_identities,
                    )
                )
                continue
            try:
                existing = _read_optional_regular_at(parent_fd, parts[-1])
            finally:
                os.close(parent_fd)
        except ProjectInitError as error:
            actions.append(_Action(plan.relative, "conflict", str(error)))
            continue
        if existing is None:
            actions.append(_Action(plan.relative, "create", parent_identities=parent_identities))
            continue
        try:
            expected = plan.content.encode("utf-8")
        except UnicodeError as error:  # pragma: no cover - code-owned strings
            raise ProjectInitError(f"cannot encode scaffold {plan.relative}: {error}") from error
        if existing[0] == expected:
            actions.append(_Action(plan.relative, "unchanged", parent_identities=parent_identities))
        else:
            actions.append(
                _Action(plan.relative, "conflict", "existing file differs from scaffold")
            )
    return actions


def _apply(root_fd: int, plans: tuple[_PlannedFile, ...], actions: list[_Action]) -> None:
    if any(action.status == "conflict" for action in actions):
        raise ProjectInitError("scaffold contains conflicting files")
    plan_paths = {plan.relative for plan in plans}
    expected = [
        action for action in actions if action.path != CONFIG_NAME or action.path in plan_paths
    ]
    refreshed = _preflight(root_fd, plans)
    if refreshed != expected:
        raise ProjectInitError("project changed after preflight")

    plan_by_path = {plan.relative: plan for plan in plans}
    refreshed_by_path = {action.path: action for action in refreshed}
    created_files: list[_CreatedFile] = []
    created_dirs: list[_CreatedDirectory] = []
    created_directory_identities: dict[tuple[str, ...], tuple[int, int]] = {}
    try:
        for action in actions:
            if action.status != "create":
                continue
            plan = plan_by_path.get(action.path)
            if plan is None:
                continue
            parts = _relative_parts(action.path)
            current_action = refreshed_by_path[action.path]
            parent_fd = _open_or_create_components(
                root_fd,
                parts[:-1],
                dict(current_action.parent_identities),
                created_dirs,
                created_directory_identities,
            )
            try:
                rollback_parent_fd = os.dup(parent_fd)
                object_fd: int | None = None
                try:
                    identity, object_fd = _publish_file(
                        parent_fd, parts[-1], plan.content.encode("utf-8")
                    )
                    try:
                        record = _CreatedFile(
                            rollback_parent_fd,
                            object_fd,
                            parts[-1],
                            identity,
                        )
                        created_files.append(record)
                    except BaseException:
                        try:
                            _remove_created_file_now(
                                rollback_parent_fd,
                                object_fd,
                                parts[-1],
                                identity,
                            )
                        finally:
                            closing_object_fd = object_fd
                            object_fd = None
                            _close_fd_quietly(closing_object_fd)
                            closing_parent_fd = rollback_parent_fd
                            rollback_parent_fd = -1
                            _close_fd_quietly(closing_parent_fd)
                        raise
                    object_fd = None
                    rollback_parent_fd = -1
                except BaseException:
                    if object_fd is not None:
                        _close_fd_quietly(object_fd)
                    if rollback_parent_fd >= 0:
                        _close_fd_quietly(rollback_parent_fd)
                    raise
            finally:
                os.close(parent_fd)
    except BaseException as error:
        _rollback(created_files, created_dirs)
        if isinstance(error, ProjectInitError):
            raise
        if isinstance(error, (OSError, ValueError)):
            raise ProjectInitError(f"initialization write failed safely: {error}") from error
        raise
    else:
        _close_created_records(created_files, created_dirs)


def _publish_file(parent_fd: int, name: str, content: bytes) -> tuple[tuple[int, int], int]:
    """Publish without replacement using a same-directory temporary hard link."""
    temporary = f".agent-cassette-{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644, dir_fd=parent_fd)
    keep_descriptor = False
    published = False
    identity: tuple[int, int] | None = None
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS invariant
                raise OSError("short write while creating scaffold")
            view = view[written:]
        os.fsync(descriptor)
        identity = _identity(os.fstat(descriptor))
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as error:
            raise ProjectInitError(f"target appeared during initialization: {name}") from error
        except OSError as error:
            raise ProjectInitError(
                f"atomic no-replace publication failed for {name}: {error}"
            ) from error
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError as error:
            raise ProjectInitError(
                f"temporary scaffold cleanup failed for {name}: {error}"
            ) from error
        keep_descriptor = True
        assert identity is not None
        return identity, descriptor
    except BaseException:
        if published and identity is not None:
            try:
                _remove_created_file_now(parent_fd, descriptor, name, identity)
            except BaseException:
                pass
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except BaseException:
            pass
        raise
    finally:
        if not keep_descriptor:
            _close_fd_quietly(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except BaseException:
            pass


def _rollback(created_files: list[_CreatedFile], created_dirs: list[_CreatedDirectory]) -> None:
    for created_file in reversed(created_files):
        try:
            _remove_created_file_now(
                created_file.parent_fd,
                created_file.object_fd,
                created_file.name,
                created_file.identity,
            )
        except OSError:
            pass
    for created_dir in reversed(created_dirs):
        try:
            candidate = _find_identity_name(
                created_dir.parent_fd,
                created_dir.object_fd,
                created_dir.name,
                created_dir.identity,
            )
            if candidate is not None:
                os.rmdir(candidate, dir_fd=created_dir.parent_fd)
        except OSError:
            pass
    _close_created_records(created_files, created_dirs)


def _remove_created_file_now(
    parent_fd: int,
    object_fd: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    open_identity = _identity(os.fstat(object_fd))
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if open_identity == identity == _identity(current):
        os.unlink(name, dir_fd=parent_fd)


def _remove_created_directory_now(
    parent_fd: int,
    object_fd: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    open_identity = _identity(os.fstat(object_fd))
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if open_identity == identity == _identity(current):
        os.rmdir(name, dir_fd=parent_fd)


def _remove_created_directory_guard(
    parent_fd: int,
    object_fd: int | None,
    name: str,
    identity: tuple[int, int] | None,
) -> None:
    if object_fd is not None and identity is not None:
        _remove_created_directory_now(parent_fd, object_fd, name, identity)
        return
    if identity is None:
        return
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(current) == identity:
        os.rmdir(name, dir_fd=parent_fd)


def _close_created_records(
    created_files: list[_CreatedFile], created_dirs: list[_CreatedDirectory]
) -> None:
    for created_file in created_files:
        _close_fd_quietly(created_file.parent_fd)
        _close_fd_quietly(created_file.object_fd)
    for created_dir in created_dirs:
        _close_fd_quietly(created_dir.parent_fd)
        _close_fd_quietly(created_dir.object_fd)


def _close_fd_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _find_identity_name(
    parent_fd: int,
    object_fd: int,
    intended: str,
    identity: tuple[int, int],
) -> str | None:
    try:
        if _identity(os.fstat(object_fd)) != identity:
            return None
    except OSError:
        return None
    names = [intended]
    try:
        names.extend(name for name in os.listdir(parent_fd) if name != intended)
    except OSError:
        return None
    for name in names:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            continue
        if _identity(current) == identity:
            return name
    return None


def _open_or_create_components(
    root_fd: int,
    parts: tuple[str, ...],
    expected: dict[tuple[str, ...], tuple[int, int]],
    created: list[_CreatedDirectory],
    created_identities: dict[tuple[str, ...], tuple[int, int]],
) -> int:
    current = os.dup(root_fd)
    prefix: tuple[str, ...] = ()
    for part in parts:
        prefix = (*prefix, part)
        required_identity = expected.get(prefix) or created_identities.get(prefix)
        child: int | None = None
        try:
            if required_identity is not None:
                child = _open_directory_at(current, part)
                if _identity(os.fstat(child)) != required_identity:
                    os.close(child)
                    raise ProjectInitError(
                        f"scaffold parent changed after preflight: {'/'.join(prefix)}"
                    )
            else:
                os.mkdir(part, 0o755, dir_fd=current)
                created_here = True
                identity: tuple[int, int] | None = None
                rollback_parent_fd: int | None = None
                rollback_object_fd: int | None = None
                try:
                    created_stat = os.stat(part, dir_fd=current, follow_symlinks=False)
                    created_identity = _identity(created_stat)
                    identity = created_identity
                    child = _open_directory_at(current, part)
                    child_stat = os.fstat(child)
                    if created_identity != _identity(child_stat):
                        raise ProjectInitError(
                            f"scaffold parent changed during creation: {part}"
                        ) from None
                    identity = _identity(child_stat)
                    rollback_parent_fd = os.dup(current)
                    rollback_object_fd = os.dup(child)
                    record = _CreatedDirectory(
                        rollback_parent_fd,
                        rollback_object_fd,
                        part,
                        identity,
                    )
                    created_identities[prefix] = identity
                    created.append(record)
                    created_here = False
                except BaseException:
                    if rollback_parent_fd is not None:
                        _close_fd_quietly(rollback_parent_fd)
                    if rollback_object_fd is not None:
                        _close_fd_quietly(rollback_object_fd)
                    if created_here:
                        try:
                            _remove_created_directory_guard(
                                current,
                                child,
                                part,
                                identity,
                            )
                        except BaseException:
                            pass
                    if child is not None:
                        _close_fd_quietly(child)
                    raise
        except FileExistsError as error:
            os.close(current)
            raise ProjectInitError(
                f"scaffold parent appeared after preflight: {'/'.join(prefix)}"
            ) from error
        except BaseException as error:
            os.close(current)
            if isinstance(error, ProjectInitError):
                raise
            if isinstance(error, (OSError, ValueError)):
                raise ProjectInitError(f"unsafe scaffold parent {part}: {error}") from error
            raise
        os.close(current)
        assert child is not None
        current = child
    return current


def _open_existing_components(
    root_fd: int, parts: tuple[str, ...]
) -> tuple[int | None, tuple[tuple[tuple[str, ...], tuple[int, int]], ...]]:
    try:
        current = os.dup(root_fd)
    except OSError as error:
        raise ProjectInitError(f"cannot duplicate project directory descriptor: {error}") from error
    identities: list[tuple[tuple[str, ...], tuple[int, int]]] = []
    prefix: tuple[str, ...] = ()
    for part in parts:
        prefix = (*prefix, part)
        try:
            child = _open_directory_at(current, part)
        except FileNotFoundError:
            os.close(current)
            return None, tuple(identities)
        except OSError as error:
            os.close(current)
            raise ProjectInitError(f"unsafe scaffold parent {part}: {error}") from error
        identities.append((prefix, _identity(os.fstat(child))))
        os.close(current)
        current = child
    return current, tuple(identities)


def _open_absolute_directory(path: Path) -> int:
    _require_secure_filesystem_primitives()
    if not path.is_absolute():  # pragma: no cover - callers normalize first
        raise ProjectInitError(f"project path must be absolute: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        current = os.open(path.anchor, flags)
        traversed = Path(path.anchor)
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
            traversed /= part
        return current
    except (OSError, ValueError) as error:
        try:
            os.close(current)
        except (OSError, UnboundLocalError):
            pass
        raise ProjectInitError(f"cannot securely open project directory {path}: {error}") from error


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)


def _read_optional_regular_at(parent_fd: int, name: str) -> tuple[bytes, tuple[int, int]] | None:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ProjectInitError(f"cannot safely inspect {name}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ProjectInitError(f"{name} is not a regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ProjectInitError(f"cannot safely open {name}: {error}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or _identity(before) != _identity(file_stat):
            raise ProjectInitError(f"{name} changed while it was being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), _identity(file_stat)
    except OSError as error:
        raise ProjectInitError(f"cannot safely read {name}: {error}") from error
    finally:
        os.close(descriptor)


def _read_regular_at(parent_fd: int, name: str) -> tuple[bytes, tuple[int, int]]:
    result = _read_optional_regular_at(parent_fd, name)
    if result is None:
        raise ProjectInitError(f"cannot read {name}: file does not exist")
    return result


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ProjectInitError(f"unsafe scaffold path: {relative}")
    return path.parts


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _require_secure_filesystem_primitives() -> None:
    if not _secure_primitives_available():
        raise ProjectInitError(
            "secure project initialization is unsupported on this platform; "
            "directory-FD and no-follow filesystem operations are required"
        )


def _secure_primitives_available() -> bool:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    dir_fd_functions = (os.open, os.stat, os.mkdir, os.unlink, os.rmdir, os.link)
    return not (
        os.name != "posix"
        or any(not hasattr(os, name) for name in required_flags)
        or any(function not in os.supports_dir_fd for function in dir_fd_functions)
        or os.listdir not in os.supports_fd
    )


def _validate_relative_config_path(value: str) -> None:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProjectInitError("cassette_dir must not contain ASCII control characters")
    if "\\" in value:
        raise ProjectInitError("cassette_dir must use forward slashes")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in (".", "")
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ProjectInitError("cassette_dir must be a safe relative path")


def _string_list(value: Any, key: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProjectInitError(f"{key} must be an array of strings")
    invalid = sorted(set(value) - set(allowed))
    if invalid:
        raise ProjectInitError(f"unsupported {key}: {', '.join(invalid)}")
    return tuple(sorted(set(value)))


def _dependencies_from_pyproject(path: Path) -> set[str]:
    return _dependencies_from_pyproject_bytes(_read_path_bytes(path), path.name)


def _dependencies_from_pyproject_bytes(data: bytes, name: str) -> set[str]:
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ProjectInitError(f"cannot parse {name}: {error}") from error
    dependencies: set[str] = set()
    project = parsed.get("project", {})
    if isinstance(project, dict):
        dependencies.update(_names_from_values(project.get("dependencies")))
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for values in optional.values():
                dependencies.update(_names_from_values(values))
    groups = parsed.get("dependency-groups", {})
    if isinstance(groups, dict):
        for values in groups.values():
            dependencies.update(_names_from_values(values))
    tool = parsed.get("tool", {})
    if isinstance(tool, dict):
        poetry = tool.get("poetry", {})
        if isinstance(poetry, dict):
            for table_name in ("dependencies", "dev-dependencies"):
                table = poetry.get(table_name, {})
                if isinstance(table, dict):
                    dependencies.update(str(name) for name in table)
            poetry_groups = poetry.get("group", {})
            if isinstance(poetry_groups, dict):
                for group in poetry_groups.values():
                    if isinstance(group, dict) and isinstance(group.get("dependencies"), dict):
                        dependencies.update(str(name) for name in group["dependencies"])
    return dependencies


def _dependencies_from_requirements(path: Path) -> set[str]:
    return _dependencies_from_requirements_bytes(_read_path_bytes(path), path.name)


def _dependencies_from_requirements_bytes(data: bytes, name: str) -> set[str]:
    lines = _decode_manifest(data, name).splitlines()
    dependencies: set[str] = set()
    for line in lines:
        candidate = line.split("#", 1)[0].strip()
        if not candidate or candidate.startswith(("-", ".", "/")):
            continue
        match = _DEPENDENCY_NAME.match(candidate)
        if match:
            dependencies.add(match.group())
    return dependencies


def _dependencies_from_setup_cfg(path: Path) -> set[str]:
    return _dependencies_from_setup_cfg_text(
        _decode_manifest(_read_path_bytes(path), path.name), path.name
    )


def _dependencies_from_setup_cfg_text(text: str, name: str) -> set[str]:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text, source=name)
    except configparser.Error as error:
        raise ProjectInitError(f"cannot parse {name}: {error}") from error
    dependencies: set[str] = set()
    if parser.has_option("options", "install_requires"):
        dependencies.update(_names_from_text(parser.get("options", "install_requires")))
    if parser.has_section("options.extras_require"):
        for _, value in parser.items("options.extras_require"):
            dependencies.update(_names_from_text(value))
    return dependencies


def _dependencies_from_setup_py(path: Path) -> set[str]:
    return _dependencies_from_setup_py_text(
        _decode_manifest(_read_path_bytes(path), path.name), path.name
    )


def _dependencies_from_setup_py_text(text: str, name: str) -> set[str]:
    try:
        tree = ast.parse(text, filename=name)
    except SyntaxError as error:
        raise ProjectInitError(f"cannot parse {name}: {error.msg}") from error
    dependencies: set[str] = set()
    constants: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_setup_call(node.func):
            continue
        for keyword in node.keywords:
            if keyword.arg not in ("install_requires", "extras_require"):
                continue
            if isinstance(keyword.value, ast.Name) and keyword.value.id in constants:
                value = constants[keyword.value.id]
            else:
                try:
                    value = ast.literal_eval(keyword.value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    continue
            dependencies.update(_names_from_literal(value))
    return dependencies


def _read_path_bytes(path: Path) -> bytes:
    absolute = path.expanduser().absolute()
    parent_fd = _open_absolute_directory(absolute.parent)
    try:
        data, _ = _read_regular_at(parent_fd, absolute.name)
        return data
    finally:
        os.close(parent_fd)


def _decode_manifest(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeError as error:
        raise ProjectInitError(f"cannot decode {name} as UTF-8: {error}") from error


def _is_setup_call(function: ast.expr) -> bool:
    return (isinstance(function, ast.Name) and function.id == "setup") or (
        isinstance(function, ast.Attribute) and function.attr == "setup"
    )


def _names_from_literal(value: Any) -> set[str]:
    if isinstance(value, str):
        return _names_from_text(value)
    if isinstance(value, (list, tuple, set)):
        return _names_from_values(value)
    if isinstance(value, dict):
        dependencies: set[str] = set()
        for nested in value.values():
            dependencies.update(_names_from_literal(nested))
        return dependencies
    return set()


def _names_from_values(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    dependencies: set[str] = set()
    for value in values:
        if isinstance(value, str):
            match = _DEPENDENCY_NAME.match(value.strip())
            if match:
                dependencies.add(match.group())
    return dependencies


def _names_from_text(value: str) -> set[str]:
    dependencies: set[str] = set()
    for line in value.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith(("-", "#")):
            continue
        match = _DEPENDENCY_NAME.match(candidate)
        if match:
            dependencies.add(match.group())
    return dependencies


def _normalize_dependency(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ProjectConfig",
    "ProjectInitError",
    "detect_integrations",
    "initialize_project",
    "load_project_config",
    "load_project_config_from_root",
    "render_init_report",
]
