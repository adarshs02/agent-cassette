import json

from agent_cassette import Cassette, EventType
from agent_cassette.cli import main


def test_inspect_json(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.MODEL_CALL, "answer", cost=0.01)

    assert main(["inspect", str(path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["events"] == 1
    assert output["types"] == {"model_call": 1}


def test_diff_exit_code_marks_divergence(tmp_path, capsys):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    with Cassette.record(baseline) as cassette:
        cassette.add(EventType.CUSTOM, "event", output="a")
    with Cassette.record(candidate) as cassette:
        cassette.add(EventType.CUSTOM, "event", output="b")

    assert main(["diff", str(baseline), str(candidate)]) == 1
    assert "First divergence: step 1" in capsys.readouterr().out
