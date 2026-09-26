from retrace import cli


def test_eval_replay_gate_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    code = cli.main(["eval", "--replay", "--scenarios", "bad_repair_rejected"])
    assert code == 0
    assert "bad_repair_rejected" in capsys.readouterr().out
    assert (tmp_path / "results" / "scorecard.md").exists()


def test_eval_requires_mode():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["eval"])


def test_unknown_scenario_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    assert cli.main(["eval", "--replay", "--scenarios", "nope"]) == 2
