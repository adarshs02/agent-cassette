"""Context-manager facade for recording and replaying cassettes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent_cassette.hybrid import Hybrid, InjectionRule, MismatchPolicy
from agent_cassette.matching import InputMatcher, MatchMode
from agent_cassette.recorder import Recorder
from agent_cassette.replay import Replayer


class Cassette:
    """Create recording and replay sessions."""

    @staticmethod
    def record(path: str | Path, *, redact_secrets: bool = True) -> Recorder:
        """Create a recording context for a cassette path."""
        return Recorder(path, redact_secrets=redact_secrets)

    @staticmethod
    def replay(
        path: str | Path,
        *,
        strict: bool = True,
        match: MatchMode = "exact",
        ignore_paths: tuple[str, ...] = (),
        matcher: InputMatcher | None = None,
    ) -> Replayer:
        """Create a replay context with configurable request matching."""
        return Replayer(
            path,
            strict=strict,
            match=match,
            ignore_paths=ignore_paths,
            matcher=matcher,
        )

    @staticmethod
    def fork(
        source: str | Path,
        output: str | Path,
        *,
        at: int | None = None,
        mismatch: MismatchPolicy = "raise",
        injections: Sequence[InjectionRule] = (),
        match: MatchMode = "exact",
        ignore_paths: tuple[str, ...] = (),
        matcher: InputMatcher | None = None,
        redact_secrets: bool = True,
    ) -> Hybrid:
        """Replay a prefix, then record a live or fault-injected branch."""
        return Hybrid(
            source,
            output,
            prefix=at,
            mismatch=mismatch,
            injections=injections,
            match=match,
            ignore_paths=ignore_paths,
            matcher=matcher,
            redact_secrets=redact_secrets,
        )
