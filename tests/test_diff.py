from agent_cassette import Cassette, EventType, compare_cassettes


def test_diff_finds_first_changed_output(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    with Cassette.record(baseline) as cassette:
        cassette.add(EventType.TOOL_CALL, "search", input="query", output=["a"])
        cassette.add(EventType.MODEL_CALL, "answer", input=["a"], output="old")
    with Cassette.record(candidate) as cassette:
        cassette.add(EventType.TOOL_CALL, "search", input="query", output=["a"])
        cassette.add(EventType.MODEL_CALL, "answer", input=["a"], output="new")

    report = compare_cassettes(baseline, candidate)

    assert not report.identical
    assert report.first_divergence == 2
    assert report.differences[0].fields["output"] == {
        "baseline": "old",
        "candidate": "new",
    }
    assert report.to_dict()["candidate_steps"] == 2


def test_diff_detects_changed_trajectory_length(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    with Cassette.record(baseline) as cassette:
        cassette.add(EventType.CUSTOM, "one")
    with Cassette.record(candidate) as cassette:
        cassette.add(EventType.CUSTOM, "one")
        cassette.add(EventType.CUSTOM, "two")

    report = compare_cassettes(baseline, candidate)

    assert report.first_divergence == 2
    assert "trajectory length changed" in report.to_text()
