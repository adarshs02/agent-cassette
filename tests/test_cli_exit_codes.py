from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_cassette.cli as cli
from agent_cassette import Cassette, EventType


def _cassette(path: Path, *, output: str = "ok") -> None:
    with Cassette.record(path) as recorder:
        recorder.add(EventType.CUSTOM, "step", output=output)


def _otlp_span(identifier: str, **updates: object) -> dict[str, object]:
    span: dict[str, object] = {
        "spanId": identifier,
        "name": "external",
        "startTimeUnixNano": "1767225600000000000",
        "attributes": [],
    }
    span.update(updates)
    return span


def _otlp_document(*spans: object) -> dict[str, object]:
    return {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}


def test_missing_cassette_is_human_error_with_exit_two(tmp_path: Path, capsys) -> None:
    status = cli.main(["inspect", str(tmp_path / "missing.jsonl")])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err.startswith("Error: ")
    assert "Traceback" not in captured.err


def test_json_error_is_one_versioned_object_on_stdout(tmp_path: Path, capsys) -> None:
    path = tmp_path / "corrupt.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    assert cli.main(["inspect", str(path), "--json"]) == 2

    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["command"] == "inspect"
    assert isinstance(report["error"], str)
    assert report["schema_version"] == 1
    assert report["status"] == "error"


def test_json_command_does_not_print_success_before_late_io_error(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _cassette(baseline)
    _cassette(candidate)

    def fail_write(self: object, path: Path) -> None:
        raise OSError("report destination is unavailable")

    monkeypatch.setattr(cli.CIReport, "write_json", fail_write)
    status = cli.main(
        [
            "diff",
            str(baseline),
            str(candidate),
            "--json",
            "--report-json",
            str(tmp_path / "report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "error"


def test_replay_mismatch_is_deterministic_negative(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    _cassette(cassette)
    script.write_text("pass\n", encoding="utf-8")

    assert cli.main(["replay", str(cassette), "--", str(script)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Error: ")
    assert "Traceback" not in captured.err


def test_replay_mismatch_during_execution_stays_exit_one(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    _cassette(cassette)
    script.write_text("pass\n", encoding="utf-8")

    def mismatch(command: object, session: object) -> int:
        raise cli.ReplayMismatchError("execution diverged")

    monkeypatch.setattr(cli, "run_python", mismatch)
    assert cli.main(["replay", "--no-strict", str(cassette), "--", str(script)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "execution diverged" in captured.err


def test_executed_child_nonzero_is_normalized_to_one(tmp_path: Path) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    script.write_text("raise SystemExit(17)\n", encoding="utf-8")

    assert cli.main(["record", str(cassette), "--", str(script)]) == 1


def test_child_programmer_value_error_is_not_swallowed(tmp_path: Path) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    script.write_text("raise ValueError('child bug')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="child bug"):
        cli.main(["record", str(cassette), "--", str(script)])


def test_child_os_error_is_not_mapped_to_cli_io_error(tmp_path: Path) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    script.write_text("raise OSError('child io failure')\n", encoding="utf-8")

    with pytest.raises(OSError, match="child io failure"):
        cli.main(["record", str(cassette), "--", str(script)])


def test_child_project_config_error_is_not_mapped_to_cli_config_error(tmp_path: Path) -> None:
    cassette = tmp_path / "run.jsonl"
    script = tmp_path / "agent.py"
    script.write_text(
        "from agent_cassette.project_init import ProjectInitError\n"
        "raise ProjectInitError('child config-like failure')\n",
        encoding="utf-8",
    )

    with pytest.raises(cli.ProjectInitError, match="child config-like failure"):
        cli.main(["record", str(cassette), "--", str(script)])


def test_unexpected_programmer_error_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_path: Path) -> list[object]:
        raise RuntimeError("programmer bug")

    monkeypatch.setattr(cli, "load_events", fail)
    with pytest.raises(RuntimeError, match="programmer bug"):
        cli.main(["inspect", str(tmp_path / "unused.jsonl")])


@pytest.mark.parametrize(
    "document",
    [
        _otlp_document(_otlp_span("1" * 16, status=[])),
        _otlp_document(
            _otlp_span(
                "1" * 16,
                attributes=[{"key": "nested", "value": {"arrayValue": []}}],
            )
        ),
        _otlp_document(_otlp_span("1" * 16, startTimeUnixNano="9" * 400)),
        _otlp_document(_otlp_span("1" * 16, endTimeUnixNano="9" * 400)),
    ],
)
def test_import_otlp_malformed_input_is_exit_two_without_traceback(
    tmp_path: Path, capsys, document: dict[str, object]
) -> None:
    source = tmp_path / "malformed.json"
    destination = tmp_path / "imported.jsonl"
    source.write_text(json.dumps(document), encoding="utf-8")

    status = cli.main(["import-otlp", str(source), str(destination)])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err.startswith("Error: ")
    assert "Traceback" not in captured.err
    assert not destination.exists()


def test_import_otlp_permissive_skips_bad_span_and_saves_valid_span(tmp_path: Path, capsys) -> None:
    source = tmp_path / "mixed.json"
    destination = tmp_path / "imported.jsonl"
    source.write_text(
        json.dumps(
            _otlp_document(
                _otlp_span("1" * 16),
                _otlp_span("2" * 16, status=[]),
            )
        ),
        encoding="utf-8",
    )

    status = cli.main(["import-otlp", str(source), str(destination), "--permissive"])

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert captured.out == f"{destination}\n"
    assert [event.id for event in cli.load_events(destination)] == [f"otlp-{'1' * 16}"]


def test_recover_incomplete_tail_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "recovered.jsonl"
    _cassette(source)
    with source.open("ab") as stream:
        stream.write(b'{"id":')

    assert cli.main(["recover", str(source), str(destination), "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["command"] == "recover"
    assert report["status"] == "recovered"
    assert report["events"] == 1
    assert report["discarded"]["bytes"] == len(b'{"id":')
    assert len(cli.load_events(destination)) == 1


def test_recover_rejects_same_source_and_output_as_json_error(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    _cassette(source)

    assert cli.main(["recover", str(source), str(source), "--json"]) == 2

    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["command"] == "recover"
    assert report["status"] == "error"


def test_recover_human_output_for_unchanged_cassette(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"
    _cassette(source)

    assert cli.main(["recover", str(source), str(destination)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "no incomplete tail found" in captured.out
    assert destination.exists()


def test_migrate_requires_output_flag(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _cassette(source)

    with pytest.raises(SystemExit) as raised:
        cli.main(["migrate", str(source)])
    assert raised.value.code == 2


def test_migrate_rejects_same_source_and_output(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _cassette(source)

    assert cli.main(["migrate", str(source), "--output", str(source)]) == 2


def test_migrate_to_separate_output_succeeds(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "upgraded.jsonl"
    _cassette(source)

    assert cli.main(["migrate", str(source), "--output", str(destination)]) == 0
    assert destination.exists()


def test_argparse_usage_remains_exit_two() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code == 2
