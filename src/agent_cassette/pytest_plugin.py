"""Pytest fixture for cassette-backed agent tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_cassette.cassette import Cassette

if TYPE_CHECKING:
    from agent_cassette.recorder import Recorder
    from agent_cassette.replay import Replayer


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
        "cassette(path, strict=True): select the cassette used by the cassette fixture",
    )


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Iterator[Recorder | Replayer]:
    """Yield a recorder or replayer selected by the cassette marker and CLI mode."""
    marker = request.node.get_closest_marker("cassette")
    relative_path = marker.args[0] if marker and marker.args else f"{request.node.name}.jsonl"
    strict = marker.kwargs.get("strict", True) if marker else True
    root_path = Path(str(request.config.rootpath))
    path = Path(relative_path)
    if not path.is_absolute():
        path = root_path / "tests" / "cassettes" / path

    mode = request.config.getoption("--cassette-mode")
    context = Cassette.record(path) if mode == "record" else Cassette.replay(path, strict=strict)
    with context as session:
        yield session
