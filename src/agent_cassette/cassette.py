"""Context-manager facade for recording and replaying cassettes."""

from __future__ import annotations

from pathlib import Path

from agent_cassette.recorder import Recorder
from agent_cassette.replay import Replayer


class Cassette:
    """Create recording and replay sessions."""

    @staticmethod
    def record(path: str | Path, *, redact_secrets: bool = True) -> Recorder:
        """Create a recording context for a cassette path."""
        return Recorder(path, redact_secrets=redact_secrets)

    @staticmethod
    def replay(path: str | Path, *, strict: bool = True) -> Replayer:
        """Create a replay context for a cassette path."""
        return Replayer(path, strict=strict)
