"""Pytest fixture for cassette-backed agent tests."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from agent_cassette.cassette import Cassette
from agent_cassette.matching import MatchMode
from agent_cassette.project_init import (
    CONFIG_NAME,
    ProjectConfig,
    ProjectInitError,
    load_project_config_from_root,
)

if TYPE_CHECKING:
    from agent_cassette.recorder import Recorder
    from agent_cassette.replay import Replayer


_PROJECT_CONFIG_KEY = pytest.StashKey[ProjectConfig]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register global cassette controls."""
    group = parser.getgroup("agent-cassette")
    group.addoption(
        "--cassette-mode",
        choices=("record", "replay"),
        default="replay",
        help="record new cassette events or replay existing events",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the cassette marker."""
    config.addinivalue_line(
        "markers",
        "cassette(path, strict=True, match='exact'): override cassette project defaults",
    )
    project_config = ProjectConfig()
    try:
        loaded = load_project_config_from_root(Path(str(config.rootpath)))
    except ProjectInitError as error:
        raise pytest.UsageError(f"invalid {CONFIG_NAME}: {error}") from error
    if loaded is not None:
        project_config, config_warnings = loaded
        for warning in config_warnings:
            warnings.warn(pytest.PytestConfigWarning(f"{CONFIG_NAME}: {warning}"), stacklevel=1)
    config.stash[_PROJECT_CONFIG_KEY] = project_config


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Iterator[Recorder | Replayer]:
    """Yield a recorder or replayer selected by the cassette marker and CLI mode."""
    marker = request.node.get_closest_marker("cassette")
    project_config = request.config.stash[_PROJECT_CONFIG_KEY]
    if marker and marker.args:
        relative_path = marker.args[0]
    elif marker and "path" in marker.kwargs:
        relative_path = marker.kwargs["path"]
    else:
        relative_path = f"{request.node.name}.jsonl"
    strict = marker.kwargs.get("strict", project_config.strict) if marker else project_config.strict
    match_value = (
        marker.kwargs.get("match", project_config.match) if marker else project_config.match
    )
    match = cast(MatchMode, match_value)
    root_path = Path(str(request.config.rootpath))
    path = Path(relative_path)
    if not path.is_absolute():
        path = root_path / project_config.cassette_dir / path

    mode = request.config.getoption("--cassette-mode")
    context = (
        Cassette.record(path)
        if mode == "record"
        else Cassette.replay(path, strict=strict, match=match)
    )
    with context as session:
        yield session
