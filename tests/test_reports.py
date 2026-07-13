from __future__ import annotations

import json

import pytest

from agent_cassette.assertions import AssertionReport, AssertionResult
from agent_cassette.diff import DiffReport
from agent_cassette.reports import CIReport


def assertion_report(passed: bool) -> AssertionReport:
    return AssertionReport((AssertionResult("check", passed, "result"),))


def test_ci_report_is_name_sorted_and_deterministic():
    report = CIReport(metadata={"z": 1, "a": 2})
    report.add_diff("z-diff", DiffReport(1, 1, None))
    report.add_assertions("b-check", assertion_report(True))
    report.add_assertions("a-check", assertion_report(True))

    first = report.to_json()
    second = report.to_json()

    assert first == second
    assert first.index('"a-check"') < first.index('"b-check"') < first.index('"z-diff"')
    assert json.loads(first)["passed"] is True


def test_ci_report_aggregates_failures_and_rejects_duplicate_names():
    report = CIReport()
    report.add_assertions("quality", assertion_report(False))

    assert not report.passed
    with pytest.raises(ValueError, match="duplicate"):
        report.add_diff("quality", DiffReport(0, 0, None))


def test_write_json_atomically_replaces_destination(tmp_path, monkeypatch):
    destination = tmp_path / "artifacts" / "report.json"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")
    replacements = []

    from agent_cassette import reports

    real_replace = reports.os.replace

    def recording_replace(source, target):
        replacements.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(reports.os, "replace", recording_replace)
    report = CIReport()
    report.add_assertions("checks", assertion_report(True))

    report.write_json(destination)

    assert json.loads(destination.read_text(encoding="utf-8"))["passed"] is True
    assert len(replacements) == 1
    assert replacements[0][1] == destination
    assert not list(destination.parent.glob("*.tmp"))
