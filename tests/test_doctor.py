from __future__ import annotations

import builtins
import json

import pytest

from agent_cassette import diagnostics
from agent_cassette.cli import build_parser, main


def _patch_environment(monkeypatch, *, modules=(), versions=None):
    module_names = set(modules)
    distribution_versions = versions or {}
    probes = []

    def fake_find_spec(name):
        probes.append(name)
        return object() if name in module_names else None

    def fake_version(name):
        if name not in distribution_versions:
            raise diagnostics.metadata.PackageNotFoundError(name)
        return distribution_versions[name]

    monkeypatch.setattr(diagnostics, "find_spec", fake_find_spec)
    monkeypatch.setattr(diagnostics.metadata, "version", fake_version)
    return probes


def test_collect_diagnostics_has_stable_schema_and_does_not_import_optional_packages(monkeypatch):
    probes = _patch_environment(
        monkeypatch,
        modules={"agent_cassette", "mcp", "openai", "agents"},
        versions={
            "agent-cassette": "0.11.1b1",
            "mcp": "1.9.0",
            "openai": "1.2.3",
            "openai-agents": "0.4.0",
        },
    )
    optional_modules = {"agents", "anthropic", "langchain_core", "mcp", "openai"}
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.partition(".")[0] not in optional_modules
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = diagnostics.collect_diagnostics()

    assert list(report) == ["healthy", "integrations", "package", "python", "schema_version"]
    assert report["schema_version"] == 1
    assert report["healthy"] is True
    assert report["package"] == {
        "installed": True,
        "module_available": True,
        "name": "agent-cassette",
        "version": "0.11.1b1",
    }
    assert [item["name"] for item in report["integrations"]] == [
        "Anthropic",
        "LangChain",
        "MCP",
        "OpenAI",
        "OpenAI Agents",
    ]
    assert all(
        {"supported", "distribution", "installed"} <= item.keys() for item in report["integrations"]
    )
    assert report["integrations"][2] == {
        "available": True,
        "distribution": "mcp",
        "installed": True,
        "kind": "built-in",
        "module": "mcp",
        "name": "MCP",
        "supported": True,
        "version": "1.9.0",
    }
    assert report["integrations"][0]["available"] is False
    assert report["integrations"][3]["version"] == "1.2.3"
    assert probes == ["agent_cassette", "anthropic", "langchain_core", "mcp", "openai", "agents"]


def test_missing_optional_packages_do_not_make_doctor_unhealthy(monkeypatch, capsys):
    _patch_environment(monkeypatch, modules={"agent_cassette"})

    assert main(["doctor"]) == 0

    captured = capsys.readouterr()
    assert "Healthy: yes" in captured.out
    assert "Anthropic: not installed (optional)" in captured.out
    assert "MCP: built-in support; SDK not installed (optional)" in captured.out
    assert captured.err == ""


def test_mcp_sdk_discovery_is_separate_from_builtin_support(monkeypatch):
    _patch_environment(
        monkeypatch,
        modules={"agent_cassette", "mcp"},
    )

    report = diagnostics.collect_diagnostics()
    mcp = next(item for item in report["integrations"] if item["name"] == "MCP")

    assert mcp["supported"] is True
    assert mcp["kind"] == "built-in"
    assert mcp["installed"] is False
    assert mcp["available"] is True
    assert mcp["version"] is None
    assert (
        "MCP: built-in support; SDK module available; "
        "distribution metadata missing (version unknown)"
    ) in diagnostics.render_human(report)


@pytest.mark.parametrize(
    ("modules", "versions", "expected"),
    [
        ({"anthropic"}, {"anthropic": "1.0"}, "Anthropic: 1.0"),
        (set(), {"anthropic": "1.0"}, "Anthropic: installed 1.0; module unavailable"),
        (
            {"anthropic"},
            {},
            "Anthropic: module available; distribution metadata missing (version unknown)",
        ),
        (set(), {}, "Anthropic: not installed (optional)"),
    ],
)
def test_human_output_distinguishes_optional_sdk_states(monkeypatch, modules, versions, expected):
    _patch_environment(
        monkeypatch,
        modules={"agent_cassette", *modules},
        versions=versions,
    )

    assert expected in diagnostics.render_human(diagnostics.collect_diagnostics())


def test_doctor_json_is_sorted_and_machine_readable(monkeypatch, capsys):
    _patch_environment(monkeypatch, modules={"agent_cassette"})

    assert main(["doctor", "--json"]) == 0

    output = capsys.readouterr().out
    assert output == json.dumps(json.loads(output), indent=2, sort_keys=True) + "\n"
    parsed = json.loads(output)
    assert set(parsed) == {
        "schema_version",
        "package",
        "python",
        "integrations",
        "healthy",
    }
    names = [item["name"] for item in parsed["integrations"]]
    assert names == sorted(names)


def test_doctor_returns_one_when_core_module_is_unavailable(monkeypatch, capsys):
    _patch_environment(monkeypatch)

    assert main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert "Package: agent-cassette source checkout / unknown (unavailable)" in captured.out
    assert captured.err == "Error: Agent Cassette core module is unavailable.\n"


def test_cli_help_uses_provider_neutral_descriptions(capsys):
    parser = build_parser()

    parser.print_help()

    help_text = capsys.readouterr().out
    assert "run Python and record supported agent calls" in help_text
    assert "run Python with offline agent-call replay" in help_text
    assert "record OpenAI calls" not in help_text
