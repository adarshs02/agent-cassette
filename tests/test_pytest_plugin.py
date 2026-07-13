from __future__ import annotations

import os
import subprocess
import sys


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
    environment = os.environ.copy()
    source_path = os.path.abspath("src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "agent_cassette.pytest_plugin",
        str(test_file),
    ]

    recorded = subprocess.run(
        [*command, "--cassette-mode=record"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    replayed = subprocess.run(
        [*command, "--cassette-mode=replay"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert (tmp_path / "tests" / "cassettes" / "agent.jsonl").exists()
