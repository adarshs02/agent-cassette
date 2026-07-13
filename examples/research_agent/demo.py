"""Keyless research-agent demo for record, replay, and diff."""

from __future__ import annotations

import sys
from pathlib import Path

from agent_cassette import Cassette, EventType, compare_cassettes

HERE = Path(__file__).parent
BASELINE = HERE / "baseline.jsonl"
CANDIDATE = HERE / "candidate.jsonl"


def fake_search(query: str) -> list[str]:
    return [f"Result about {query}", "Agent testing improves reliability"]


def fake_model(prompt: str, *, revised: bool = False) -> str:
    prefix = "Evidence indicates" if revised else "Research suggests"
    return f"{prefix}: {prompt[:42]}..."


def record_run(path: Path, *, revised: bool = False) -> str:
    with Cassette.record(path) as cassette:
        results = cassette.call(
            EventType.TOOL_CALL,
            "search",
            {"query": "deterministic AI agent testing"},
            lambda: fake_search("deterministic AI agent testing"),
        )
        return cassette.call(
            EventType.MODEL_CALL,
            "summarize",
            {"results": results},
            lambda: fake_model(" ".join(results), revised=revised),
            metadata={"model": "fake-model-v1", "tokens": 24},
        )


def replay_run(path: Path) -> str:
    with Cassette.replay(path) as cassette:
        results = cassette.call(
            EventType.TOOL_CALL, "search", {"query": "deterministic AI agent testing"}
        )
        return cassette.call(EventType.MODEL_CALL, "summarize", {"results": results})


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "record":
        print(record_run(BASELINE))
    elif mode == "replay":
        print(replay_run(BASELINE))
    else:
        record_run(BASELINE)
        record_run(CANDIDATE, revised=True)
        print(compare_cassettes(BASELINE, CANDIDATE).to_text())


if __name__ == "__main__":
    main()
