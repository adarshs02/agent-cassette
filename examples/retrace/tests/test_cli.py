import json

from retrace import cli


def test_eval_replay_gate_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    code = cli.main(["eval", "--replay", "--scenarios", "bad_repair_rejected"])
    assert code == 0
    assert "bad_repair_rejected" in capsys.readouterr().out
    assert (tmp_path / "results" / "replay" / "scorecard.md").exists()
    assert not (tmp_path / "results" / "scorecard.md").exists()


def test_eval_live_writes_top_level_results(tmp_path, monkeypatch):
    from retrace.evals import runner

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(runner, "run_eval", lambda *a, **k: [])
    assert cli.main(["eval", "--live"]) == 0
    assert (tmp_path / "results" / "scorecard.md").exists()
    assert not (tmp_path / "results" / "replay").exists()
    assert not (tmp_path / "results" / "partial").exists()


def test_eval_live_scoped_writes_partial_results(tmp_path, monkeypatch):
    from retrace.evals import runner

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(runner, "run_eval", lambda *a, **k: [])
    # A scoped live run is scratch too -- it must not clobber the committed,
    # full-suite scorecard at evals/results/scorecard.md.
    assert cli.main(["eval", "--live", "--scenarios", "bad_repair_rejected"]) == 0
    assert (tmp_path / "results" / "partial" / "scorecard.md").exists()
    assert not (tmp_path / "results" / "scorecard.md").exists()


def test_replay_and_partial_results_are_gitignored():
    from retrace.config import APP_ROOT

    lines = (APP_ROOT / ".gitignore").read_text().splitlines()
    assert "evals/results/replay/" in lines
    assert "evals/results/partial/" in lines


def test_eval_requires_mode():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["eval"])


def test_unknown_scenario_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    assert cli.main(["eval", "--replay", "--scenarios", "nope"]) == 2


def _manifest_with_expected(tmp_path, expected):
    cassettes = tmp_path / "cassettes"
    cassettes.mkdir()
    (cassettes / "MANIFEST.json").write_text(
        json.dumps({"scenarios": sorted(expected), "expected": expected})
    )
    return cassettes


def test_eval_replay_recorded_failure_replayed_as_same_failure_exits_0(tmp_path, monkeypatch):
    from retrace.evals import runner
    from retrace.evals.runner import ScenarioResult

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    cassette_dir = _manifest_with_expected(tmp_path, {"tz_shift": "failed"})
    monkeypatch.setattr(cli, "CASSETTE_DIR", cassette_dir)
    monkeypatch.setattr(runner, "run_eval", lambda *a, **k: [ScenarioResult("tz_shift", "failed")])
    assert cli.main(["eval", "--replay", "--scenarios", "tz_shift"]) == 0


def test_eval_replay_mismatch_against_manifest_exits_1(tmp_path, monkeypatch, capsys):
    from retrace.evals import runner
    from retrace.evals.runner import ScenarioResult

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    cassette_dir = _manifest_with_expected(tmp_path, {"tz_shift": "failed"})
    monkeypatch.setattr(cli, "CASSETTE_DIR", cassette_dir)
    monkeypatch.setattr(runner, "run_eval", lambda *a, **k: [ScenarioResult("tz_shift", "passed")])
    code = cli.main(["eval", "--replay", "--scenarios", "tz_shift"])
    assert code == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "tz_shift" in out


def test_eval_replay_unrecorded_scenario_failure_exits_1(tmp_path, monkeypatch, capsys):
    from retrace.evals import runner
    from retrace.evals.runner import ScenarioResult

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    # "expected" only covers unit_cents; bad_repair_rejected (the gate, no cassette)
    # has no recording to compare against, but a failure there still has to fail CI.
    cassette_dir = _manifest_with_expected(tmp_path, {"unit_cents": "passed"})
    monkeypatch.setattr(cli, "CASSETTE_DIR", cassette_dir)
    monkeypatch.setattr(
        runner,
        "run_eval",
        lambda *a, **k: [
            ScenarioResult("unit_cents", "passed"),
            ScenarioResult("bad_repair_rejected", "failed"),
        ],
    )
    code = cli.main(["eval", "--replay", "--scenarios", "unit_cents,bad_repair_rejected"])
    assert code == 1
    out = capsys.readouterr().out
    assert "bad_repair_rejected" in out


def test_internal_keyerror_propagates(tmp_path, monkeypatch):
    import pytest
    from retrace.evals import runner

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")

    def mock_run_eval(*args, **kwargs):
        raise KeyError("internal")

    monkeypatch.setattr(runner, "run_eval", mock_run_eval)
    with pytest.raises(KeyError, match="internal"):
        cli.main(["eval", "--replay", "--scenarios", "bad_repair_rejected"])
