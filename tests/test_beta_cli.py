from __future__ import annotations

import json
import sys
import types

from agent_cassette import Cassette, EventType
from agent_cassette.cli import main


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def model_dump(self, mode=None):
        return {"output_text": self.output_text}


class _Responses:
    live_calls = 0

    def create(self, **kwargs):
        type(self).live_calls += 1
        return _Response(f"answer:{kwargs['input']}")


class _OpenAI:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1
        self.responses = _Responses()


class _AsyncOpenAI:
    pass


def test_cli_records_and_replays_python_script_automatically(tmp_path, monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _OpenAI
    fake_openai.AsyncOpenAI = _AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    _Responses.live_calls = 0
    _OpenAI.constructions = 0

    script = tmp_path / "agent.py"
    output = tmp_path / "output.txt"
    cassette = tmp_path / "run.jsonl"
    script.write_text(
        """
import sys
from pathlib import Path
from openai import OpenAI
response = OpenAI().responses.create(model="test", input="hello")
Path(sys.argv[1]).write_text(response.output_text)
""",
        encoding="utf-8",
    )

    assert main(["record", str(cassette), "--", str(script), str(output)]) == 0
    assert output.read_text() == "answer:hello"
    output.unlink()
    assert main(["replay", str(cassette), "--", str(script), str(output)]) == 0

    assert output.read_text() == "answer:hello"
    assert _Responses.live_calls == 1
    assert _OpenAI.constructions == 1


def test_invalid_record_command_does_not_destroy_existing_cassette(tmp_path):
    cassette = tmp_path / "existing.jsonl"
    cassette.write_text("preserve me\n", encoding="utf-8")

    try:
        main(["record", str(cassette)])
    except ValueError:
        pass

    assert cassette.read_text(encoding="utf-8") == "preserve me\n"


def test_cli_check_view_reports_and_otlp_round_trip(tmp_path):
    cassette = tmp_path / "run.jsonl"
    with Cassette.record(cassette) as session:
        session.add(EventType.TOOL_CALL, "search", input={"query": "agents"}, output=[])

    assertion_report = tmp_path / "assertions.json"
    assert (
        main(
            [
                "check",
                str(cassette),
                "--no-errors",
                "--require",
                "tool_call:search",
                "--report-json",
                str(assertion_report),
            ]
        )
        == 0
    )
    assert json.loads(assertion_report.read_text())["passed"] is True

    html = tmp_path / "viewer.html"
    assert main(["view", str(cassette), "--output", str(html)]) == 0
    assert "Agent Cassette Viewer" in html.read_text()

    otlp = tmp_path / "trace.json"
    restored = tmp_path / "restored.jsonl"
    assert main(["export-otlp", str(cassette), str(otlp)]) == 0
    assert main(["import-otlp", str(otlp), str(restored)]) == 0
    assert main(["inspect", str(restored), "--json"]) == 0


def test_cli_diff_writes_deterministic_ci_report(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "diff.json"
    for path, output in ((baseline, "old"), (candidate, "new")):
        with Cassette.record(path) as cassette:
            cassette.add(EventType.MODEL_CALL, "answer", output=output)

    assert main(["diff", str(baseline), str(candidate), "--report-json", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["passed"] is False
    assert data["diffs"]["trajectory"]["first_divergence"] == 1
