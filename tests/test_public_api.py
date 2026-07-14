from __future__ import annotations

import os
import subprocess
import sys
from textwrap import dedent

import agent_cassette

EXPECTED_PUBLIC_API = {
    "Adapter",
    "AdapterRegistry",
    "AgentCassetteDeprecationWarning",
    "AgentCassetteRunHooks",
    "AssertionReport",
    "AssertionResult",
    "CIReport",
    "Cassette",
    "CassetteCorruptionError",
    "Delay",
    "DiffReport",
    "Event",
    "EventType",
    "Hybrid",
    "InjectionRule",
    "Raise",
    "RateLimitError",
    "RecordedCallError",
    "RecoveryReport",
    "RedactionError",
    "ReplayMismatchError",
    "Return",
    "assert_trajectory",
    "automatic_openai_from_env",
    "check_trajectory",
    "compare_cassettes",
    "contains_event",
    "event_count",
    "event_sequence",
    "export_otlp",
    "import_otlp",
    "langchain_callback_handler",
    "max_total_cost",
    "max_total_duration_ms",
    "migrate_cassette",
    "migrate_event_dict",
    "no_errors",
    "patch_anthropic",
    "patch_openai",
    "patch_openai_agents",
    "recover_cassette",
    "register_migration",
    "render_viewer",
    "unregister_migration",
    "wrap_anthropic",
    "wrap_langchain",
    "wrap_mcp",
    "wrap_openai",
    "write_viewer",
}


def test_public_api_candidate_snapshot() -> None:
    assert len(agent_cassette.__all__) == len(set(agent_cassette.__all__))
    assert set(agent_cassette.__all__) == EXPECTED_PUBLIC_API
    assert all(getattr(agent_cassette, name) is not None for name in EXPECTED_PUBLIC_API)


def test_core_import_does_not_load_optional_dependencies(tmp_path) -> None:
    script = dedent(
        """
        import sys

        blocked = {"agents", "anthropic", "langchain_core", "openai"}

        class BlockOptionalDependencies:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked:
                    raise AssertionError(f"optional dependency imported: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptionalDependencies())
        import agent_cassette

        assert not blocked.intersection(sys.modules)
        assert set(agent_cassette.__all__)
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
