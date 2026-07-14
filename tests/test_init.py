from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_cassette.cli import build_parser, main
from agent_cassette.project_init import (
    ProjectInitError,
    detect_integrations,
    initialize_project,
    load_project_config,
    load_project_config_from_root,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _open_fd_count() -> int | None:
    descriptor_directory = Path("/dev/fd")
    return len(os.listdir(descriptor_directory)) if descriptor_directory.is_dir() else None


def test_init_detect_dry_run_reports_without_writing(tmp_path: Path) -> None:
    _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = ["openai>=1", "langchain-core>=0.3"]\n',
    )
    before = _snapshot(tmp_path)

    report, exit_code = initialize_project(tmp_path, detect=True, mode="dry-run")

    assert exit_code == 0
    assert report["schema_version"] == 1
    assert report["command"] == "init"
    assert report["status"] == "would-change"
    assert report["detected"] == {
        "frameworks": ["langchain"],
        "providers": ["openai"],
        "test_frameworks": [],
    }
    assert [action["path"] for action in report["actions"]] == [
        ".agent-cassette.toml",
        "tests/cassettes/.gitkeep",
        "tests/test_agent_cassette_smoke.py",
    ]
    assert all(action["status"] == "create" for action in report["actions"])
    assert _snapshot(tmp_path) == before


def test_init_apply_is_idempotent_and_check_reports_current(tmp_path: Path) -> None:
    first, first_exit = initialize_project(tmp_path, detect=True)
    first_snapshot = _snapshot(tmp_path)
    second, second_exit = initialize_project(tmp_path, detect=True)
    second_snapshot = _snapshot(tmp_path)
    checked, check_exit = initialize_project(tmp_path, detect=True, mode="check")

    assert first_exit == second_exit == check_exit == 0
    assert first["status"] == "changed"
    assert second["status"] == checked["status"] == "current"
    assert first_snapshot == second_snapshot
    assert [action["status"] for action in second["actions"]] == [
        "unchanged",
        "unchanged",
        "unchanged",
    ]


def test_check_returns_one_when_changes_are_needed(tmp_path: Path) -> None:
    report, exit_code = initialize_project(tmp_path, mode="check")

    assert exit_code == 1
    assert report["status"] == "changes-needed"
    assert not (tmp_path / ".agent-cassette.toml").exists()


def test_existing_valid_config_is_accepted_without_reformatting(tmp_path: Path) -> None:
    config = """# user maintained
schema_version = 1
cassette_dir = "fixtures/cassettes"
match = "subset"
strict = false
providers = ["anthropic"]
frameworks = []
"""
    _write(tmp_path / ".agent-cassette.toml", config)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 0
    assert report["actions"][0] == {
        "path": ".agent-cassette.toml",
        "status": "unchanged",
    }
    assert (tmp_path / ".agent-cassette.toml").read_text(encoding="utf-8") == config


def test_unknown_config_keys_warn_but_are_accepted(tmp_path: Path) -> None:
    _write(
        tmp_path / ".agent-cassette.toml",
        """schema_version = 1
cassette_dir = "tests/cassettes"
match = "exact"
strict = true
providers = []
frameworks = []
future_option = "preserve me"
""",
    )

    report, exit_code = initialize_project(tmp_path, mode="dry-run")

    assert exit_code == 0
    assert report["warnings"] == ["unknown configuration key ignored: future_option"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("schema_version = 2", "newer than supported"),
        ('schema_version = "1"', "integer 1"),
        ('cassette_dir = "../outside"', "safe relative path"),
        ('match = "semantic"', "match must be one of"),
        ("strict = 1", "strict must be true or false"),
        ('providers = ["unknown"]', "unsupported providers"),
        ('frameworks = ["llama-index"]', "unsupported frameworks"),
        ('test_frameworks = ["nose"]', "unsupported test_frameworks"),
    ],
)
def test_invalid_known_config_values_are_rejected(
    tmp_path: Path, replacement: str, message: str
) -> None:
    values = {
        "schema_version": "schema_version = 1",
        "cassette_dir": 'cassette_dir = "tests/cassettes"',
        "match": 'match = "exact"',
        "strict": "strict = true",
        "providers": "providers = []",
        "frameworks": "frameworks = []",
        "test_frameworks": "test_frameworks = []",
    }
    key = replacement.split(" =", 1)[0]
    values[key] = replacement
    _write(tmp_path / ".agent-cassette.toml", "\n".join(values.values()) + "\n")

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "invalid"
    assert message in report["error"]
    assert not (tmp_path / "tests").exists()


def test_malformed_config_is_rejected_before_writes(tmp_path: Path) -> None:
    _write(tmp_path / ".agent-cassette.toml", "schema_version = [\n")

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "invalid"
    assert "cannot read .agent-cassette.toml" in report["error"]
    assert not (tmp_path / "tests").exists()


def test_unowned_smoke_test_conflict_prevents_all_writes(tmp_path: Path) -> None:
    original = "def test_user_code():\n    assert True\n"
    _write(tmp_path / "tests/test_agent_cassette_smoke.py", original)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "conflict"
    assert not (tmp_path / ".agent-cassette.toml").exists()
    assert not (tmp_path / "tests/cassettes").exists()
    assert (tmp_path / "tests/test_agent_cassette_smoke.py").read_text() == original


def test_owned_but_changed_smoke_test_is_never_auto_updated(tmp_path: Path) -> None:
    original = "# agent-cassette: generated smoke test v1\n# stale\n"
    _write(
        tmp_path / "tests/test_agent_cassette_smoke.py",
        original,
    )

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "conflict"
    assert (tmp_path / "tests/test_agent_cassette_smoke.py").read_text() == original
    assert not (tmp_path / ".agent-cassette.toml").exists()


def test_existing_nonempty_gitkeep_is_a_conflict(tmp_path: Path) -> None:
    _write(tmp_path / "tests/cassettes/.gitkeep", "user content\n")

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "conflict"
    assert not (tmp_path / ".agent-cassette.toml").exists()


def test_file_in_scaffold_parent_path_is_preflight_conflict(tmp_path: Path) -> None:
    _write(tmp_path / "tests", "not a directory\n")

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] == "conflict"
    assert not (tmp_path / ".agent-cassette.toml").exists()
    assert "unsafe scaffold parent" in report["actions"][1]["reason"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlinked_project_root_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    report, exit_code = initialize_project(link)

    assert exit_code == 2
    assert report["status"] == "invalid"
    assert "cannot securely open project directory" in report["error"]
    assert not (real / ".agent-cassette.toml").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
@pytest.mark.parametrize("relative", [".agent-cassette.toml", "tests", "tests/cassettes"])
def test_symlinked_target_or_parent_is_rejected(tmp_path: Path, relative: str) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-{relative.replace('/', '-')}"
    outside.mkdir()
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert report["status"] in {"invalid", "conflict"}
    messages = [str(report["error"])] + [str(action.get("reason")) for action in report["actions"]]
    assert any(
        "not a regular file" in value or "unsafe scaffold parent" in value for value in messages
    )
    assert not (outside / "test_agent_cassette_smoke.py").exists()


def test_nonexistent_or_file_project_is_rejected(tmp_path: Path) -> None:
    missing_report, missing_exit = initialize_project(tmp_path / "missing")
    file_path = tmp_path / "project.py"
    _write(file_path, "")
    file_report, file_exit = initialize_project(file_path)

    assert missing_exit == file_exit == 2
    assert "No such file or directory" in missing_report["error"]
    assert "Not a directory" in file_report["error"]


def test_apply_rolls_back_only_paths_created_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    existing_tests = tmp_path / "tests"
    existing_tests.mkdir()
    calls = 0
    real_publish = project_init._publish_file

    def fail_second_write(parent_fd: int, name: str, content: bytes) -> tuple[tuple[int, int], int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated failure")
        return real_publish(parent_fd, name, content)

    monkeypatch.setattr(project_init, "_publish_file", fail_second_write)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "simulated failure" in report["error"]
    assert existing_tests.is_dir()
    assert not (tmp_path / ".agent-cassette.toml").exists()
    assert not (tmp_path / "tests/cassettes").exists()


def test_failed_temporary_write_leaves_no_scaffold_or_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    def fail_write(descriptor: int, content: object) -> int:
        raise OSError("simulated short storage")

    monkeypatch.setattr(project_init.os, "write", fail_write)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "simulated short storage" in report["error"]
    assert list(tmp_path.iterdir()) == []


def test_dup_failure_after_first_file_publication_rolls_back_without_fd_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    original_dup = project_init.os.dup
    original_open_components = project_init._open_or_create_components
    active = False
    calls = 0
    before = _open_fd_count()

    def fail_second_dup(descriptor: int) -> int:
        nonlocal calls
        if active:
            calls += 1
        if active and calls == 1:
            raise OSError("dup failed after file publication")
        return original_dup(descriptor)

    def activate_failure(*arguments, **keywords):
        nonlocal active
        active = True
        try:
            return original_open_components(*arguments, **keywords)
        finally:
            active = False

    monkeypatch.setattr(project_init.os, "dup", fail_second_dup)
    monkeypatch.setattr(project_init, "_open_or_create_components", activate_failure)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "dup failed after file publication" in report["error"]
    assert list(tmp_path.iterdir()) == []
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


def test_file_record_construction_failure_removes_just_published_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    before = _open_fd_count()

    def fail_record(*values: object) -> None:
        raise OSError("file record construction failed")

    monkeypatch.setattr(project_init, "_CreatedFile", fail_record)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "file record construction failed" in report["error"]
    assert list(tmp_path.iterdir()) == []
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


def test_keyboard_interrupt_after_link_cleans_target_temp_and_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    original_unlink = project_init.os.unlink
    interrupted = False
    before = _open_fd_count()

    def interrupt_first_temp_unlink(path, *arguments, **keywords):
        nonlocal interrupted
        if isinstance(path, str) and path.startswith(".agent-cassette-") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_unlink(path, *arguments, **keywords)

    monkeypatch.setattr(project_init.os, "unlink", interrupt_first_temp_unlink)
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: True)

    with pytest.raises(KeyboardInterrupt):
        initialize_project(tmp_path)

    assert list(tmp_path.iterdir()) == []
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


@pytest.mark.parametrize("interruption", ["stat", "open"])
def test_keyboard_interrupt_after_mkdir_uses_only_proven_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    from agent_cassette import project_init

    original_open_components = project_init._open_or_create_components
    original_open_directory = project_init._open_directory_at
    original_stat = project_init.os.stat
    active = False
    interrupted = False
    before = _open_fd_count()

    def activate_interrupt(*arguments, **keywords):
        nonlocal active
        active = True
        try:
            return original_open_components(*arguments, **keywords)
        finally:
            active = False

    def interrupt_stat(path, *arguments, **keywords):
        nonlocal interrupted
        if active and interruption == "stat" and path == "tests" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_stat(path, *arguments, **keywords)

    def interrupt_open(parent_fd: int, name: str) -> int:
        nonlocal interrupted
        if active and interruption == "open" and name == "tests" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_open_directory(parent_fd, name)

    monkeypatch.setattr(project_init, "_open_or_create_components", activate_interrupt)
    monkeypatch.setattr(project_init, "_open_directory_at", interrupt_open)
    monkeypatch.setattr(project_init.os, "stat", interrupt_stat)
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: True)

    with pytest.raises(KeyboardInterrupt):
        initialize_project(tmp_path)

    if interruption == "stat":
        assert [path.name for path in tmp_path.iterdir()] == ["tests"]
        assert list((tmp_path / "tests").iterdir()) == []
    else:
        assert list(tmp_path.iterdir()) == []
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


def test_pre_identity_directory_replacement_is_never_removed_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    original_open_components = project_init._open_or_create_components
    original_stat = project_init.os.stat
    active = False
    interrupted = False
    identities: dict[str, tuple[int, int]] = {}
    before = _open_fd_count()

    def activate_interrupt(*arguments, **keywords):
        nonlocal active
        active = True
        try:
            return original_open_components(*arguments, **keywords)
        finally:
            active = False

    def replace_then_interrupt(path, *arguments, **keywords):
        nonlocal interrupted
        if active and path == "tests" and not interrupted:
            interrupted = True
            intended = tmp_path / "tests"
            renamed = tmp_path / "tests-created"
            intended.rename(renamed)
            intended.mkdir()
            created_stat = os.lstat(renamed)
            replacement_stat = os.lstat(intended)
            identities["created"] = (created_stat.st_dev, created_stat.st_ino)
            identities["replacement"] = (
                replacement_stat.st_dev,
                replacement_stat.st_ino,
            )
            raise KeyboardInterrupt
        return original_stat(path, *arguments, **keywords)

    monkeypatch.setattr(project_init, "_open_or_create_components", activate_interrupt)
    monkeypatch.setattr(project_init.os, "stat", replace_then_interrupt)
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: True)

    with pytest.raises(KeyboardInterrupt):
        initialize_project(tmp_path)

    created_after = os.lstat(tmp_path / "tests-created")
    replacement_after = os.lstat(tmp_path / "tests")
    assert (created_after.st_dev, created_after.st_ino) == identities["created"]
    assert (replacement_after.st_dev, replacement_after.st_ino) == identities["replacement"]
    assert identities["created"] != identities["replacement"]
    assert list((tmp_path / "tests-created").iterdir()) == []
    assert list((tmp_path / "tests").iterdir()) == []
    assert not (tmp_path / ".agent-cassette.toml").exists()
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


def test_dup_failure_after_directory_first_dup_removes_untracked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    original_dup = project_init.os.dup
    original_open_components = project_init._open_or_create_components
    active = False
    calls = 0
    before = _open_fd_count()

    def fail_fourth_dup(descriptor: int) -> int:
        nonlocal calls
        if active:
            calls += 1
        if active and calls == 3:
            raise OSError("dup failed after directory creation")
        return original_dup(descriptor)

    def activate_failure(*arguments, **keywords):
        nonlocal active
        active = True
        try:
            return original_open_components(*arguments, **keywords)
        finally:
            active = False

    monkeypatch.setattr(project_init.os, "dup", fail_fourth_dup)
    monkeypatch.setattr(project_init, "_open_or_create_components", activate_failure)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "dup failed after directory creation" in report["error"]
    assert list(tmp_path.iterdir()) == []
    after = _open_fd_count()
    if before is not None and after is not None:
        assert after <= before


def test_target_created_between_preflight_and_publish_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    real_publish = project_init._publish_file
    raced = False

    def race(parent_fd: int, name: str, content: bytes) -> tuple[tuple[int, int], int]:
        nonlocal raced
        if not raced:
            raced = True
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
                dir_fd=parent_fd,
            )
            os.write(descriptor, b"user-owned\n")
            os.close(descriptor)
        return real_publish(parent_fd, name, content)

    monkeypatch.setattr(project_init, "_publish_file", race)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "target appeared during initialization" in report["error"]
    assert (tmp_path / ".agent-cassette.toml").read_text() == "user-owned\n"
    assert not (tmp_path / "tests").exists()


def test_rollback_uses_open_identity_and_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    real_publish = project_init._publish_file
    calls = 0

    def replace_then_fail(parent_fd: int, name: str, content: bytes) -> tuple[tuple[int, int], int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            config_path = tmp_path / ".agent-cassette.toml"
            config_path.unlink()
            config_path.write_text("replacement owned by another process\n")
            raise OSError("force identity-aware rollback")
        return real_publish(parent_fd, name, content)

    monkeypatch.setattr(project_init, "_publish_file", replace_then_fail)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "force identity-aware rollback" in report["error"]
    assert (tmp_path / ".agent-cassette.toml").read_text() == (
        "replacement owned by another process\n"
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_parent_swapped_to_symlink_during_apply_never_writes_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    outside = tmp_path / "outside"
    outside.mkdir()
    real_publish = project_init._publish_file
    swapped = False

    def swap(parent_fd: int, name: str, content: bytes) -> tuple[tuple[int, int], int]:
        nonlocal swapped
        if name == ".gitkeep" and not swapped:
            swapped = True
            (tmp_path / "tests").rename(tmp_path / "tests-held")
            (tmp_path / "tests").symlink_to(outside, target_is_directory=True)
        return real_publish(parent_fd, name, content)

    monkeypatch.setattr(project_init, "_publish_file", swap)

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "unsafe scaffold parent" in report["error"]
    assert list(outside.iterdir()) == []
    assert not (tmp_path / ".agent-cassette.toml").exists()
    assert not (tmp_path / "tests-held").exists()


def test_configured_cassette_dir_controls_gitkeep_location(tmp_path: Path) -> None:
    _write(
        tmp_path / ".agent-cassette.toml",
        """schema_version = 1
cassette_dir = "fixtures/agent-cassettes"
match = "exact"
strict = true
providers = []
frameworks = []
""",
    )

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 0, report
    assert (tmp_path / "fixtures/agent-cassettes/.gitkeep").is_file()
    assert not (tmp_path / "tests/cassettes").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_config_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / ".agent-cassette.toml")

    report, exit_code = initialize_project(tmp_path)

    assert exit_code == 2
    assert "not a regular file" in report["error"]
    assert not (tmp_path / "tests").exists()


def test_portable_read_only_config_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    _write(
        tmp_path / ".agent-cassette.toml",
        """schema_version = 1
cassette_dir = "portable/cassettes"
match = "subset"
strict = false
providers = []
frameworks = []
test_frameworks = ["pytest"]
""",
    )
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: False)

    loaded = load_project_config_from_root(tmp_path)

    assert loaded is not None
    assert loaded[0].cassette_dir == "portable/cassettes"
    assert loaded[0].match == "subset"
    assert loaded[0].strict is False


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_portable_config_fallback_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_cassette import project_init

    outside = tmp_path / "outside.toml"
    _write(outside, "schema_version = 1\n")
    (tmp_path / ".agent-cassette.toml").symlink_to(outside)
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: False)

    with pytest.raises(ProjectInitError, match="not a regular file"):
        load_project_config_from_root(tmp_path)


@pytest.mark.parametrize("codepoint", [0, 1, 9, 10, 31, 127])
def test_cassette_dir_rejects_ascii_control_characters(tmp_path: Path, codepoint: int) -> None:
    escaped = f"\\u{codepoint:04x}"
    _write(
        tmp_path / ".agent-cassette.toml",
        f"""schema_version = 1
cassette_dir = "tests{escaped}cassettes"
match = "exact"
strict = true
providers = []
frameworks = []
test_frameworks = []
""",
    )

    report, exit_code = initialize_project(tmp_path, mode="dry-run")

    assert exit_code == 2
    assert report["status"] == "invalid"
    assert "ASCII control characters" in report["error"]


def test_nul_in_project_path_returns_invalid_report(tmp_path: Path) -> None:
    report, exit_code = initialize_project(f"{tmp_path}/bad\0project", mode="dry-run")

    assert exit_code == 2
    assert report["status"] == "invalid"
    assert report["error"]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (
            "pyproject.toml",
            """[project]
dependencies = ["OpenAI>=1", "anthropic", "mcp>=1", "openai_agents"]
[project.optional-dependencies]
agent = ["langchain-community"]
""",
        ),
        (
            "requirements-dev.txt",
            "openai==1\nanthropic[bedrock]>=0.34\nmcp-sdk\nopenai-agents\nlangchain-core\n",
        ),
        (
            "setup.cfg",
            """[options]
install_requires =
    openai
    anthropic
[options.extras_require]
agents =
    openai-agents
    langchain
protocol = mcp
""",
        ),
        (
            "setup.py",
            """from setuptools import setup
setup(
    install_requires=["openai", "anthropic", "mcp"],
    extras_require={"all": ["openai-agents", "langchain-core"]},
)
raise RuntimeError("must never execute")
""",
        ),
    ],
)
def test_static_detection_supports_manifest_forms(
    tmp_path: Path, filename: str, content: str
) -> None:
    _write(tmp_path / filename, content)

    detected = detect_integrations(tmp_path)

    assert detected == {
        "providers": ["anthropic", "mcp", "openai", "openai-agents"],
        "frameworks": ["langchain"],
        "test_frameworks": [],
    }


def test_setup_py_dynamic_expressions_are_not_evaluated(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    _write(
        tmp_path / "setup.py",
        f"""from setuptools import setup
from pathlib import Path
Path({str(marker)!r}).write_text("executed")
setup(install_requires=get_requirements())
""",
    )

    assert detect_integrations(tmp_path) == {
        "providers": [],
        "frameworks": [],
        "test_frameworks": [],
    }
    assert not marker.exists()


def test_setup_py_literal_assignment_is_detected_without_execution(tmp_path: Path) -> None:
    _write(
        tmp_path / "setup.py",
        """from setuptools import setup
requirements = ["openai", "langchain-core"]
setup(install_requires=requirements)
""",
    )

    assert detect_integrations(tmp_path) == {
        "providers": ["openai"],
        "frameworks": ["langchain"],
        "test_frameworks": [],
    }


def test_detection_is_sorted_and_deduplicated(tmp_path: Path) -> None:
    _write(tmp_path / "requirements.txt", "openai\nAnthropic\nopenai\nlangchain\n")
    _write(tmp_path / "requirements-dev.txt", "MCP\nopenai-agents\nlangchain-core\n")

    assert detect_integrations(tmp_path) == {
        "providers": ["anthropic", "mcp", "openai", "openai-agents"],
        "frameworks": ["langchain"],
        "test_frameworks": [],
    }


def test_test_framework_detection_uses_dependencies_and_static_imports(tmp_path: Path) -> None:
    _write(tmp_path / "requirements-dev.txt", "pytest>=8\n")
    _write(
        tmp_path / "tests/test_stdlib.py",
        """import unittest

class Example(unittest.TestCase):
    pass
""",
    )

    detected = detect_integrations(tmp_path)

    assert detected["test_frameworks"] == ["pytest", "unittest"]


def test_detected_test_frameworks_are_written_to_new_config(tmp_path: Path) -> None:
    _write(tmp_path / "requirements-dev.txt", "pytest>=8\n")

    report, exit_code = initialize_project(tmp_path, detect=True)
    config, _ = load_project_config(tmp_path / ".agent-cassette.toml")

    assert exit_code == 0, report
    assert config.test_frameworks == ("pytest",)
    assert report["detected"]["test_frameworks"] == ["pytest"]


@pytest.mark.parametrize(
    ("filename", "content", "warning"),
    [
        ("pyproject.toml", b"[project\n", "cannot parse pyproject.toml"),
        ("setup.cfg", b"[options\n", "cannot parse setup.cfg"),
        ("setup.py", b"setup(\n", "cannot parse setup.py"),
        ("requirements.txt", b"\xff", "cannot decode requirements.txt"),
    ],
)
def test_malformed_manifests_produce_deterministic_init_warnings(
    tmp_path: Path, filename: str, content: bytes, warning: str
) -> None:
    (tmp_path / filename).write_bytes(content)

    report, exit_code = initialize_project(tmp_path, detect=True, mode="dry-run")

    assert exit_code == 0
    assert report["status"] == "would-change"
    assert len(report["warnings"]) == 1
    assert report["warnings"][0].startswith(f"manifest {filename} ignored: {warning}")
    assert detect_integrations(tmp_path) == {
        "providers": [],
        "frameworks": [],
        "test_frameworks": [],
    }


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlinked_manifest_is_reported_and_never_followed(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-manifest"
    outside.write_text('[project]\ndependencies = ["openai"]\n')
    (tmp_path / "pyproject.toml").symlink_to(outside)

    report, exit_code = initialize_project(tmp_path, detect=True, mode="dry-run")

    assert exit_code == 0
    assert report["detected"]["providers"] == []
    assert report["warnings"] == [
        "manifest pyproject.toml ignored: pyproject.toml is not a regular file"
    ]


def test_load_config_returns_sorted_deduplicated_integrations(tmp_path: Path) -> None:
    path = tmp_path / ".agent-cassette.toml"
    _write(
        path,
        """schema_version = 1
cassette_dir = "tests/cassettes"
match = "normalized"
strict = true
providers = ["openai", "anthropic", "openai"]
frameworks = ["langchain", "langchain"]
test_frameworks = ["unittest", "pytest", "pytest"]
""",
    )

    config, warnings = load_project_config(path)

    assert config.providers == ("anthropic", "openai")
    assert config.frameworks == ("langchain",)
    assert config.test_frameworks == ("pytest", "unittest")
    assert warnings == ()


def test_cli_json_report_has_only_json_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["init", str(tmp_path), "--detect", "--dry-run", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert report["project"] == str(tmp_path)
    assert report["mode"] == "dry-run"


def test_cli_dry_run_conflict_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path / "tests/test_agent_cassette_smoke.py", "# user-owned\n")

    exit_code = main(["init", str(tmp_path), "--dry-run", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "conflict"


def test_cli_project_defaults_to_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init", "--dry-run", "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["project"] == str(tmp_path)


def test_cli_rejects_check_with_dry_run() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["init", "--check", "--dry-run"])

    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("configured", "arguments", "expected_strict", "expected_match"),
    [
        (True, [], False, "subset"),
        (True, ["--strict", "--match", "exact"], True, "exact"),
        (False, [], True, "exact"),
    ],
)
def test_replay_cli_uses_config_defaults_and_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
    arguments: list[str],
    expected_strict: bool,
    expected_match: str,
) -> None:
    from agent_cassette import cli as cli_module
    from agent_cassette import project_init

    if configured:
        _write(
            tmp_path / ".agent-cassette.toml",
            """schema_version = 1
cassette_dir = "tests/cassettes"
match = "subset"
strict = false
providers = []
frameworks = []
test_frameworks = []
""",
        )
    captured: dict[str, object] = {}

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *values):
            return None

    def replay(path, *, strict, match):
        captured.update(path=path, strict=strict, match=match)
        return Context()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: False)
    monkeypatch.setattr(cli_module.Cassette, "replay", replay)
    monkeypatch.setattr(cli_module, "validate_python_command", lambda command: command)
    monkeypatch.setattr(cli_module, "run_python", lambda command, cassette: 0)

    exit_code = main(["replay", *arguments, str(tmp_path / "run.jsonl"), "--", "agent.py"])

    assert exit_code == 0
    assert captured["strict"] is expected_strict
    assert captured["match"] == expected_match


def test_replay_cli_rejects_invalid_project_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / ".agent-cassette.toml", "schema_version = 999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_cassette.cli.validate_python_command", lambda command: command)

    exit_code = main(["replay", str(tmp_path / "run.jsonl"), "--", "agent.py"])

    assert exit_code == 2
    assert "Invalid project configuration" in capsys.readouterr().err


def test_replay_cli_keeps_cassette_path_required() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["replay"])

    assert raised.value.code == 2


def test_generated_smoke_test_records_and_replays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, exit_code = initialize_project(tmp_path)
    assert exit_code == 0, report
    generated = tmp_path / "tests/test_agent_cassette_smoke.py"
    namespace: dict[str, object] = {}
    exec(compile(generated.read_text(), str(generated), "exec"), namespace)
    smoke = namespace["test_agent_cassette_offline_round_trip"]
    assert callable(smoke)

    class TmpPathFactory:
        pass

    smoke_dir = tmp_path / "smoke-runtime"
    smoke_dir.mkdir()
    smoke(smoke_dir)
    assert _snapshot(smoke_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_init_never_modifies_dependency_manifests_or_gitignore(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\ndependencies = ["openai"]\n')
    _write(tmp_path / ".gitignore", "dist/\n")
    before = {
        "pyproject.toml": (tmp_path / "pyproject.toml").read_bytes(),
        ".gitignore": (tmp_path / ".gitignore").read_bytes(),
    }

    report, exit_code = initialize_project(tmp_path, detect=True)

    assert exit_code == 0, report
    assert (tmp_path / "pyproject.toml").read_bytes() == before["pyproject.toml"]
    assert (tmp_path / ".gitignore").read_bytes() == before[".gitignore"]


def test_all_match_modes_are_valid(tmp_path: Path) -> None:
    for match in ("exact", "fuzzy", "normalized", "subset"):
        path = tmp_path / f"{match}.toml"
        _write(
            path,
            f"""schema_version = 1
cassette_dir = "tests/cassettes"
match = "{match}"
strict = true
providers = []
frameworks = []
""",
        )
        config, _ = load_project_config(path)
        assert config.match == match


def test_config_requires_every_known_key(tmp_path: Path) -> None:
    path = tmp_path / ".agent-cassette.toml"
    _write(path, "schema_version = 1\n")

    with pytest.raises(ProjectInitError, match="cassette_dir"):
        load_project_config(path)
