"""1.0 contract freeze snapshots.

These tests pin the public CLI tree and the public callable signatures. They are
intentionally brittle: a failure means a frozen contract changed, so update the
expected snapshot only as a deliberate, documented decision.
"""

from __future__ import annotations

import argparse
import inspect

import agent_cassette
import agent_cassette.cli as cli

EXPECTED_CLI_TREE: dict[str, dict[str, list[str]]] = {
    "check": {
        "positionals": ["cassette"],
        "options": [
            "--help",
            "--max-cost",
            "--max-duration-ms",
            "--no-errors",
            "--report-json",
            "--require",
            "--tool-called",
            "--tool-not-called",
            "-h",
        ],
    },
    "diff": {
        "positionals": ["baseline", "candidate"],
        "options": ["--help", "--json", "--report-json", "-h"],
    },
    "doctor": {"positionals": [], "options": ["--help", "--json", "-h"]},
    "export-otlp": {"positionals": ["cassette", "output"], "options": ["--help", "-h"]},
    "fork": {
        "positionals": ["source", "output", "python_command"],
        "options": ["--at", "--help", "--inject", "--mismatch", "-h"],
    },
    "import-otlp": {
        "positionals": ["input", "cassette"],
        "options": ["--help", "--permissive", "-h"],
    },
    "init": {
        "positionals": ["project"],
        "options": ["--check", "--detect", "--dry-run", "--help", "--json", "-h"],
    },
    "inspect": {"positionals": ["cassette"], "options": ["--help", "--json", "-h"]},
    "migrate": {
        "positionals": ["cassette"],
        "options": ["--help", "--output", "-h", "-o"],
    },
    "record": {
        "positionals": ["cassette", "python_command"],
        "options": ["--help", "--name", "--project", "--report-json", "-h"],
    },
    "recover": {"positionals": ["source", "output"], "options": ["--help", "--json", "-h"]},
    "replay": {
        "positionals": ["cassette", "python_command"],
        "options": [
            "--help",
            "--match",
            "--name",
            "--no-strict",
            "--project",
            "--report-json",
            "--strict",
            "-h",
        ],
    },
    "rerecord": {
        "positionals": ["python_command"],
        "options": ["--help", "--name", "--project", "--report-json", "-h"],
    },
    "setup": {
        "positionals": ["project"],
        "options": [
            "--apply",
            "--cassette-dir",
            "--check",
            "--dry-run",
            "--framework",
            "--github-ci",
            "--help",
            "--json",
            "--match",
            "--no-strict",
            "--provider",
            "--strict",
            "--test-framework",
            "-h",
        ],
    },
    "status": {"positionals": ["project"], "options": ["--help", "--json", "-h"]},
    "agent-manifest": {"positionals": ["project"], "options": ["--help", "--json", "-h"]},
    "ci": {
        "positionals": ["project"],
        "options": ["--apply", "--check", "--dry-run", "--github", "--help", "--json", "-h"],
    },
    "view": {"positionals": ["cassette"], "options": ["--help", "--output", "-h", "-o"]},
}

EXPECTED_PUBLIC_SIGNATURES: dict[str, str] = {
    "assert_trajectory": "(trajectory: 'Trajectory', *checks: 'Check') -> 'AssertionReport'",
    "automatic_openai_from_env": "(environ: 'Mapping[str, str] | None' = None) -> 'Iterator[Any]'",
    "check_trajectory": "(trajectory: 'Trajectory', *checks: 'Check') -> 'AssertionReport'",
    "compare_cassettes": "(baseline_path: 'str | Path', candidate_path: 'str | Path') -> 'DiffReport'",
    "contains_event": "(event_type: 'EventType | str | None' = None, *, name: 'str | None' = None) -> 'Check'",
    "event_count": "(expected: 'int | None' = None, *, minimum: 'int | None' = None, maximum: 'int | None' = None, event_type: 'EventType | str | None' = None) -> 'Check'",
    "event_sequence": "(*expected: 'EventType | str | tuple[EventType | str, str]') -> 'Check'",
    "export_otlp": "(events: 'Iterable[Event]', destination: 'str | Path | None' = None) -> 'dict[str, Any]'",
    "import_otlp": "(source: 'Mapping[str, Any] | str | bytes | Path', *, strict: 'bool' = True) -> 'list[Event]'",
    "langchain_callback_handler": "(cassette)",
    "max_total_cost": "(limit: 'float') -> 'Check'",
    "max_total_duration_ms": "(limit: 'float') -> 'Check'",
    "migrate_cassette": "(source: 'str | Path', destination: 'str | Path') -> 'Path'",
    "migrate_event_dict": "(data: 'dict[str, Any]') -> 'dict[str, Any]'",
    "no_errors": "() -> 'Check'",
    "patch_anthropic": "(cassette: 'Any') -> 'Iterator[None]'",
    "patch_gemini": "(cassette: 'Any') -> 'Iterator[None]'",
    "patch_mistral": "(cassette: 'Any') -> 'Iterator[None]'",
    "patch_openai": "(cassette: 'Any') -> 'Iterator[None]'",
    "patch_openai_agents": "(cassette: 'Any') -> 'Iterator[None]'",
    "recover_cassette": "(source: 'str | Path', destination: 'str | Path') -> 'RecoveryReport'",
    "register_migration": "(from_version: 'int', migration: 'EventMigration') -> 'None'",
    "render_viewer": "(trajectory: 'Iterable[Event] | str | Path', *, redact_secrets: 'bool' = True, max_events: 'int' = 1000, max_payload_chars: 'int' = 20000, title: 'str' = 'Agent Cassette Viewer') -> 'str'",
    "tool_called": "(name: 'str', *, with_input: 'Any' = Ellipsis, times: 'int | None' = None, minimum: 'int | None' = None, maximum: 'int | None' = None, match: 'MatchMode' = 'exact', ignore_paths: 'tuple[str, ...]' = (), fuzzy_threshold: 'float' = 0.9) -> 'Check'",
    "tool_not_called": "(name: 'str', *, with_input: 'Any' = Ellipsis, match: 'MatchMode' = 'exact', ignore_paths: 'tuple[str, ...]' = (), fuzzy_threshold: 'float' = 0.9) -> 'Check'",
    "unregister_migration": "(from_version: 'int') -> 'EventMigration'",
    "wrap_anthropic": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
    "wrap_gemini": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
    "wrap_langchain": "(runnable, cassette, *, name='langchain.runnable')",
    "wrap_langchain_tools": "(tools, cassette, *, name_prefix='langchain.tool')",
    "wrap_mcp": "(session: 'Session | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Session'",
    "wrap_mistral": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
    "wrap_openai": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
    "wrap_tool": "(function: 'Callable[P, R]', cassette: 'Any', *, name: 'str | None' = None) -> 'Callable[P, R]'",
    "write_viewer": "(path: 'str | Path', trajectory: 'Iterable[Event] | str | Path', **options: 'Any') -> 'None'",
    "Cassette.record": "(path: 'str | Path', *, redact_secrets: 'bool' = True) -> 'Recorder'",
    "Cassette.replay": "(path: 'str | Path', *, strict: 'bool' = True, match: 'MatchMode' = 'exact', ignore_paths: 'tuple[str, ...]' = (), matcher: 'InputMatcher | None' = None, fuzzy_threshold: 'float' = 0.9) -> 'Replayer'",
    "Cassette.fork": "(source: 'str | Path', output: 'str | Path', *, at: 'int | None' = None, mismatch: 'MismatchPolicy' = 'raise', injections: 'Sequence[InjectionRule]' = (), match: 'MatchMode' = 'exact', ignore_paths: 'tuple[str, ...]' = (), matcher: 'InputMatcher | None' = None, fuzzy_threshold: 'float' = 0.9, redact_secrets: 'bool' = True) -> 'Hybrid'",
}


def _command_tree() -> dict[str, dict[str, list[str]]]:
    parser = cli.build_parser()
    tree: dict[str, dict[str, list[str]]] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                options = sorted(
                    {option for act in subparser._actions for option in act.option_strings}
                )
                positionals = [
                    act.dest
                    for act in subparser._actions
                    if not act.option_strings and act.dest != "help"
                ]
                tree[name] = {"positionals": positionals, "options": options}
    return tree


def _public_callable_signatures() -> dict[str, str]:
    signatures: dict[str, str] = {}
    for name in agent_cassette.__all__:
        obj = getattr(agent_cassette, name)
        if inspect.isfunction(obj):
            signatures[name] = str(inspect.signature(obj))
    for method in ("record", "replay", "fork"):
        signatures[f"Cassette.{method}"] = str(
            inspect.signature(getattr(agent_cassette.Cassette, method))
        )
    return signatures


def test_cli_tree_is_frozen() -> None:
    assert _command_tree() == EXPECTED_CLI_TREE


def test_public_callable_signatures_are_frozen() -> None:
    assert _public_callable_signatures() == EXPECTED_PUBLIC_SIGNATURES
