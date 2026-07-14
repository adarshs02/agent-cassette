"""Public API for Agent Cassette."""

from agent_cassette.adapters import Adapter, AdapterRegistry
from agent_cassette.assertions import (
    AssertionReport,
    AssertionResult,
    assert_trajectory,
    check_trajectory,
    contains_event,
    event_count,
    event_sequence,
    max_total_cost,
    max_total_duration_ms,
    no_errors,
)
from agent_cassette.automatic import automatic_openai_from_env, patch_anthropic, patch_openai
from agent_cassette.cassette import Cassette
from agent_cassette.deprecations import AgentCassetteDeprecationWarning
from agent_cassette.diff import DiffReport, compare_cassettes
from agent_cassette.events import Event, EventType, register_migration, unregister_migration
from agent_cassette.hybrid import Delay, Hybrid, InjectionRule, Raise, Return
from agent_cassette.integrations.anthropic import wrap_anthropic
from agent_cassette.integrations.mcp import wrap_mcp
from agent_cassette.integrations.openai import wrap_openai
from agent_cassette.integrations.openai_agents import AgentCassetteRunHooks, patch_openai_agents
from agent_cassette.interop import export_otlp, import_otlp
from agent_cassette.migration import migrate_cassette, migrate_event_dict
from agent_cassette.redaction import RedactionError
from agent_cassette.replay import RateLimitError, RecordedCallError, ReplayMismatchError
from agent_cassette.reports import CIReport
from agent_cassette.storage import CassetteCorruptionError, RecoveryReport, recover_cassette
from agent_cassette.viewer import render_viewer, write_viewer


def wrap_langchain(runnable, cassette, *, name="langchain.runnable"):
    """Lazily wrap a LangChain Runnable without making LangChain a core dependency."""
    from agent_cassette.integrations.langchain import wrap_langchain as _wrap_langchain

    return _wrap_langchain(runnable, cassette, name=name)


def langchain_callback_handler(cassette):
    """Create lifecycle tracing callbacks without making LangChain a core dependency."""
    from agent_cassette.integrations.langchain_callbacks import (
        langchain_callback_handler as _langchain_callback_handler,
    )

    return _langchain_callback_handler(cassette)


__all__ = [
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
    "RecoveryReport",
    "RecordedCallError",
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
    "register_migration",
    "recover_cassette",
    "render_viewer",
    "unregister_migration",
    "wrap_anthropic",
    "wrap_langchain",
    "wrap_mcp",
    "wrap_openai",
    "write_viewer",
]
