"""Deprecation policy shared by the Python API and CLI."""

from __future__ import annotations


class AgentCassetteDeprecationWarning(DeprecationWarning):
    """Warning for an Agent Cassette API scheduled for removal after 1.x."""


__all__ = ["AgentCassetteDeprecationWarning"]
