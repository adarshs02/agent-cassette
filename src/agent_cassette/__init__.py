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
from agent_cassette.automatic import automatic_openai_from_env, patch_openai
from agent_cassette.cassette import Cassette
from agent_cassette.diff import DiffReport, compare_cassettes
from agent_cassette.events import Event, EventType
from agent_cassette.hybrid import Hybrid, InjectionRule, Raise, Return
from agent_cassette.integrations.mcp import wrap_mcp
from agent_cassette.integrations.openai import wrap_openai
from agent_cassette.integrations.openai_agents import AgentCassetteRunHooks, patch_openai_agents
from agent_cassette.interop import export_otlp, import_otlp
from agent_cassette.migration import migrate_cassette
from agent_cassette.replay import RecordedCallError, ReplayMismatchError
from agent_cassette.reports import CIReport
from agent_cassette.viewer import render_viewer, write_viewer

__all__ = [
    "Adapter",
    "AdapterRegistry",
    "AgentCassetteRunHooks",
    "AssertionReport",
    "AssertionResult",
    "CIReport",
    "Cassette",
    "DiffReport",
    "Event",
    "EventType",
    "Hybrid",
    "InjectionRule",
    "Raise",
    "RecordedCallError",
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
    "max_total_cost",
    "max_total_duration_ms",
    "migrate_cassette",
    "no_errors",
    "patch_openai",
    "patch_openai_agents",
    "render_viewer",
    "wrap_mcp",
    "wrap_openai",
    "write_viewer",
]
