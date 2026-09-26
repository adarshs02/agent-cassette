import json

from retrace.config import Settings
from retrace.datahub.fake import FakeDataHubSession
from retrace.datahub.mcp_client import DataHubConnection
from retrace.evals.runner import MANIFEST, ScenarioResult, run_eval, run_scenario
from retrace.evals.scenarios import SCENARIOS, get_scenario
from retrace.evals.scorecard import write_results
from retrace.testing import ScriptedModel, unit_cents_script

FACTORIES = {
    "model_factory": lambda: ScriptedModel(unit_cents_script()),
    "datahub_factory": lambda c: DataHubConnection.from_session(FakeDataHubSession(), c),
}


def test_scenario_catalog():
    names = [s.name for s in SCENARIOS]
    assert names[:8] == [
        "unit_cents",
        "schema_rename",
        "join_fanout",
        "tz_shift",
        "stale_feed",
        "null_surge",
        "control_healthy",
        "control_distractor",
    ]
    assert set(names[8:]) == {"bad_repair_rejected", "datahub_timeout", "rate_limit_midrun"}
    for s in SCENARIOS:
        if s.base:
            assert names.index(s.base) < names.index(s.name)


def test_gate_scenario_needs_nothing(tmp_path):
    result = run_scenario(
        get_scenario("bad_repair_rejected"),
        "replay",
        settings=Settings(),
        cassette_dir=tmp_path / "c",
        work_root=tmp_path / "w",
    )
    assert result.status == "passed", result.grades


def test_live_then_replay_unit_cents(tmp_path):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    live = run_scenario(get_scenario("unit_cents"), "live", ingest=False, **kw, **FACTORIES)
    assert live.status == "passed", live.grades
    assert (tmp_path / "c" / "unit_cents.jsonl").exists()
    replay = run_scenario(get_scenario("unit_cents"), "replay", **kw)
    assert replay.status == "passed", replay.grades
    assert replay.stats == live.stats


def test_wrong_answer_is_graded_failed(tmp_path):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    result = run_scenario(
        get_scenario("control_healthy"),
        "live",
        ingest=False,
        **kw,
        model_factory=lambda: ScriptedModel(unit_cents_script()),
        datahub_factory=FACTORIES["datahub_factory"],
    )
    assert result.status in ("failed", "error")


def test_replay_missing_cassette(tmp_path):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    assert run_scenario(get_scenario("unit_cents"), "replay", **kw).status == "skipped"
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / MANIFEST).write_text(json.dumps({"scenarios": ["unit_cents"]}))
    assert run_scenario(get_scenario("unit_cents"), "replay", **kw).status == "error"


def test_replay_mismatch_is_error_not_crash(tmp_path, monkeypatch):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    run_scenario(get_scenario("unit_cents"), "live", ingest=False, **kw, **FACTORIES)
    # A prompt change alters every model request, so replay must diverge on the first call.
    monkeypatch.setattr("retrace.agent.loop.SYSTEM_PROMPT", "a changed prompt")
    result = run_scenario(get_scenario("unit_cents"), "replay", **kw)
    assert result.status == "error" and "diverged" in result.error


def test_scorecard(tmp_path):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    results = run_eval("replay", ["bad_repair_rejected"], **kw)
    path = write_results(results, "replay", tmp_path / "results")
    text = path.read_text()
    assert "| bad_repair_rejected | passed |" in text
    assert (tmp_path / "results" / "run_0001.json").exists()


def test_scorecard_only_counts_failed_controls_as_false_positives(tmp_path):
    results = [
        ScenarioResult("control_healthy", "skipped", error="no cassette recorded yet"),
        ScenarioResult("control_distractor", "passed"),
        ScenarioResult("unit_cents", "error", error="boom"),
    ]
    text = write_results(results, "replay", tmp_path / "results").read_text()
    assert "Control false positives: 0/2" in text
    not_graded_line = next(line for line in text.splitlines() if "Not graded" in line)
    assert "control_healthy" in not_graded_line
    assert "unit_cents" in not_graded_line


def test_scorecard_counts_a_failed_control_as_false_positive(tmp_path):
    results = [
        ScenarioResult("control_healthy", "failed"),
        ScenarioResult("control_distractor", "passed"),
    ]
    text = write_results(results, "replay", tmp_path / "results").read_text()
    assert "Control false positives: 1/2" in text
    not_graded_line = next(line for line in text.splitlines() if "Not graded" in line)
    assert "none" in not_graded_line


def test_scorecard_table_row_stays_single_line_for_multiline_error(tmp_path):
    results = [
        ScenarioResult("unit_cents", "error", error="RuntimeError: a|b\nTraceback...\n  File x"),
    ]
    text = write_results(results, "replay", tmp_path / "results").read_text()
    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    header_pipes = table_lines[0].replace("\\|", "").count("|")
    row_lines = [line for line in table_lines if "unit_cents" in line]
    assert len(row_lines) == 1
    assert "RuntimeError: a\\|b" in row_lines[0]
    for line in table_lines:
        assert line.replace("\\|", "").count("|") == header_pipes


def test_errored_scenario_keeps_traceback(tmp_path, monkeypatch):
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("retrace.evals.runner.run_agent", boom)
    result = run_scenario(get_scenario("unit_cents"), "live", ingest=False, **kw, **FACTORIES)
    assert result.status == "error"
    assert "RuntimeError: boom" in result.error
    assert "Traceback" in result.error or 'File "' in result.error
