from __future__ import annotations

import importlib.metadata

import agent_cassette


def test_version_attribute_is_nonempty_string() -> None:
    assert isinstance(agent_cassette.__version__, str)
    assert agent_cassette.__version__


def test_version_matches_installed_metadata() -> None:
    assert agent_cassette.__version__ == importlib.metadata.version("agent-cassette")
