from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, cast


def _pytest_command(test_file):
    return [
        sys.executable,
        "-I",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:agent_cassette",
        "-p",
        "agent_cassette.pytest_plugin",
        str(test_file),
    ]


def _run(command, root, *arguments):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [*command, *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pytest_fixture_records_then_replays(tmp_path):
    test_file = tmp_path / "test_agent.py"
    test_file.write_text(
        """
import pytest
from agent_cassette import EventType

@pytest.mark.cassette("agent.jsonl")
def test_agent(cassette):
    if hasattr(cassette, "add"):
        cassette.add(EventType.MODEL_CALL, "answer", input={"prompt": "hi"}, output="hello")
        result = "hello"
    else:
        result = cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "hi"})
    assert result == "hello"
""",
        encoding="utf-8",
    )
    command = _pytest_command(test_file)
    recorded = _run(command, tmp_path, "--cassette-mode=record")
    replayed = _run(command, tmp_path, "--cassette-mode=replay")

    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert (tmp_path / "tests" / "cassettes" / "agent.jsonl").exists()


def test_project_config_sets_fixture_path_strictness_and_matching(tmp_path):
    (tmp_path / ".agent-cassette.toml").write_text(
        """schema_version = 1
cassette_dir = "fixtures/cassettes"
match = "subset"
strict = false
providers = []
frameworks = []
""",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_configured.py"
    test_file.write_text(
        """
import pytest
from agent_cassette import EventType

@pytest.mark.cassette
def test_configured(cassette):
    if hasattr(cassette, "add"):
        cassette.add(EventType.MODEL_CALL, "answer", input={"prompt": "hi"}, output="ok")
        cassette.add(EventType.CUSTOM, "left-unconsumed", output=True)
        result = "ok"
    else:
        assert cassette.strict is False
        assert cassette.match == "subset"
        result = cassette.call(
            EventType.MODEL_CALL,
            "answer",
            {"prompt": "hi", "extra": "accepted"},
        )
    assert result == "ok"
""",
        encoding="utf-8",
    )
    command = _pytest_command(test_file)

    recorded = _run(command, tmp_path, "--cassette-mode=record")
    replayed = _run(command, tmp_path, "--cassette-mode=replay")

    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert (tmp_path / "fixtures/cassettes/test_configured.jsonl").is_file()


def test_marker_values_override_project_defaults(tmp_path):
    (tmp_path / ".agent-cassette.toml").write_text(
        """schema_version = 1
cassette_dir = "configured"
match = "subset"
strict = false
providers = []
frameworks = []
""",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_override.py"
    test_file.write_text(
        """
import pytest
from agent_cassette import EventType

@pytest.mark.cassette(path="override.jsonl", strict=True, match="exact")
def test_override(cassette):
    if hasattr(cassette, "add"):
        cassette.add(EventType.MODEL_CALL, "answer", input={"prompt": "hi"}, output="ok")
        result = "ok"
    else:
        assert cassette.strict is True
        assert cassette.match == "exact"
        result = cassette.call(EventType.MODEL_CALL, "answer", {"prompt": "hi"})
    assert result == "ok"
""",
        encoding="utf-8",
    )
    command = _pytest_command(test_file)

    recorded = _run(command, tmp_path, "--cassette-mode=record")
    replayed = _run(command, tmp_path, "--cassette-mode=replay")

    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert (tmp_path / "configured/override.jsonl").is_file()


def test_invalid_project_config_is_a_clear_pytest_usage_error(tmp_path):
    (tmp_path / ".agent-cassette.toml").write_text(
        "schema_version = 999\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_invalid.py"
    test_file.write_text("def test_never_runs():\n    assert True\n", encoding="utf-8")

    result = _run(_pytest_command(test_file), tmp_path)

    assert result.returncode == 4
    assert "invalid .agent-cassette.toml" in result.stderr
    assert "newer than supported" in result.stderr


def test_pytest_configure_uses_portable_read_only_config_fallback(tmp_path, monkeypatch):
    from agent_cassette import project_init, pytest_plugin

    (tmp_path / ".agent-cassette.toml").write_text(
        """schema_version = 1
cassette_dir = "portable/cassettes"
match = "normalized"
strict = false
providers = []
frameworks = []
test_frameworks = ["pytest"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(project_init, "_secure_primitives_available", lambda: False)

    class FakeConfig:
        def __init__(self):
            self.rootpath = tmp_path
            self.stash = {}

        def addinivalue_line(self, name, value):
            return None

    config = FakeConfig()
    pytest_plugin.pytest_configure(cast(Any, config))

    loaded = config.stash[pytest_plugin._PROJECT_CONFIG_KEY]
    assert loaded.cassette_dir == "portable/cassettes"
    assert loaded.match == "normalized"
    assert loaded.strict is False
