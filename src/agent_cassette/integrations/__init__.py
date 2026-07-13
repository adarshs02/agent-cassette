"""Optional third-party integrations."""

from agent_cassette.integrations.anthropic import wrap_anthropic
from agent_cassette.integrations.mcp import wrap_mcp
from agent_cassette.integrations.openai import wrap_openai
from agent_cassette.integrations.openai_agents import AgentCassetteRunHooks, patch_openai_agents

__all__ = [
    "AgentCassetteRunHooks",
    "patch_openai_agents",
    "wrap_anthropic",
    "wrap_mcp",
    "wrap_openai",
]
