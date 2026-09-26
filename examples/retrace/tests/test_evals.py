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


class _DocUrnSession(FakeDataHubSession):
    async def call_tool(self, name, arguments=None):
        if name == "save_document":
            from retrace.datahub.fake import _ok

            self.calls.append((name, dict(arguments or {})))
            return _ok({"success": True, "urn": "urn:li:document:retrace-1"})
        return await super().call_tool(name, arguments)


DOC_FACTORIES = {
    "model_factory": lambda: ScriptedModel(unit_cents_script()),
    "datahub_factory": lambda c: DataHubConnection.from_session(_DocUrnSession(), c),
}


def test_live_mode_soft_deletes_writeback_docs(tmp_path, monkeypatch):
    deleted = []
    monkeypatch.setattr(
        "retrace.datahub.ingest.soft_delete", lambda urns, settings: deleted.append(list(urns))
    )
    kw = {"settings": Settings(), "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    live = run_scenario(get_scenario("unit_cents"), "live", ingest=False, **kw, **DOC_FACTORIES)
    assert live.status == "passed", live.grades
    assert deleted == [["urn:li:document:retrace-1"]]
    replay = run_scenario(get_scenario("unit_cents"), "replay", **kw)
    assert replay.status == "passed", replay.error
    assert deleted == [["urn:li:document:retrace-1"]]  # replay never deletes


def test_live_ingest_settles_but_replay_never_sleeps(tmp_path, monkeypatch):
    import dataclasses

    from retrace.evals import runner

    slept = []
    monkeypatch.setattr("retrace.datahub.ingest.ingest_workspace", lambda ws, s: 0)
    monkeypatch.setattr(runner, "_settle", lambda seconds: slept.append(seconds))
    settings = dataclasses.replace(Settings(), ingest_settle_s=2.5)
    kw = {"settings": settings, "cassette_dir": tmp_path / "c", "work_root": tmp_path / "w"}
    live = run_scenario(get_scenario("unit_cents"), "live", **kw, **FACTORIES)
    assert live.status == "passed", live.grades
    assert slept == [2.5]
    run_scenario(get_scenario("unit_cents"), "replay", **kw)
    assert slept == [2.5]


def test_settle_setting_from_env(monkeypatch):
    monkeypatch.setenv("RETRACE_INGEST_SETTLE_S", "0.5")
    assert Settings.from_env().ingest_settle_s == 0.5
    monkeypatch.delenv("RETRACE_INGEST_SETTLE_S")
    assert Settings.from_env().ingest_settle_s == 5.0


def test_errored_live_scenario_drops_cassette_and_manifest_entry(tmp_path, monkeypatch):
    from retrace.evals import runner

    real = runner.run_agent

    def flaky(client, executor, state, **kwargs):
        if state.scenario == "schema_rename":
            client.messages.create(model="m", max_tokens=1, messages=[])  # partial recording
            raise RuntimeError("boom")
        return real(client, executor, state, **kwargs)

    monkeypatch.setattr(runner, "run_agent", flaky)
    selected = [get_scenario("unit_cents"), get_scenario("schema_rename")]
    monkeypatch.setattr(runner, "SCENARIOS", selected)
    cassettes = tmp_path / "c"
    results = run_eval(
        "live",
        None,
        settings=Settings(),
        cassette_dir=cassettes,
        work_root=tmp_path / "w",
        ingest=False,
        **FACTORIES,
    )
    status = {r.scenario: r.status for r in results}
    assert status["schema_rename"] == "error"
    assert not (cassettes / "schema_rename.jsonl").exists()
    assert (cassettes / "unit_cents.jsonl").exists()
    manifest = json.loads((cassettes / MANIFEST).read_text())["scenarios"]
    assert "schema_rename" not in manifest and "unit_cents" in manifest


def _robustness_grades(stage):
    from retrace.agent.state import IncidentState, Stage
    from retrace.evals.graders import grade_robustness
    from retrace.faults import get_fault

    fault = get_fault(get_scenario("datahub_timeout").fault)
    state = IncidentState(scenario="datahub_timeout", report="r", stage=Stage(stage))
    return {g.name: g.passed for g in grade_robustness(fault, state)}


def test_datahub_timeout_grader_requires_escalated_or_failed():
    verified = _robustness_grades("VERIFIED")
    assert verified["ends_escalated_or_failed"] is False
    assert verified["no_false_all_clear"] is True
    assert _robustness_grades("FAILED")["ends_escalated_or_failed"] is True
    assert _robustness_grades("ESCALATED")["ends_escalated_or_failed"] is True
    no_incident = _robustness_grades("NO_INCIDENT")
    assert no_incident["no_false_all_clear"] is False
    assert no_incident["ends_escalated_or_failed"] is False
