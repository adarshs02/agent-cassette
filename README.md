# Agent Cassette

**VCR.py for AI agents.** Record complete agent executions, replay them offline, and find the first behavioral divergence after a model, prompt, or code change.

Observability platforms such as Langfuse and Phoenix show what happened. Agent Cassette reproduces what happened, compares the trajectory, and turns real runs into deterministic regression tests.

## Features

- Framework-independent, versioned JSONL event format
- Offline replay with strict or permissive matching
- First-divergence trajectory diffing in text or JSON
- Secret redaction before cassette data reaches disk
- Pytest fixture for record/replay regression tests
- No API keys or hosted service required

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from agent_cassette import Cassette, EventType

with Cassette.record("tests/cassettes/research.jsonl") as cassette:
    results = cassette.call(
        EventType.TOOL_CALL,
        "search",
        {"query": "agent testing"},
        lambda: live_search("agent testing"),
    )

with Cassette.replay("tests/cassettes/research.jsonl") as cassette:
    results = cassette.call(
        EventType.TOOL_CALL,
        "search",
        {"query": "agent testing"},
    )
```

Replay returns the recorded result without invoking `live_search`. A changed event type, name, or input raises a `ReplayMismatchError` at the exact divergent step.

## CLI

```bash
agent-cassette inspect tests/cassettes/research.jsonl
agent-cassette inspect tests/cassettes/research.jsonl --json
agent-cassette diff baseline.jsonl candidate.jsonl
agent-cassette diff baseline.jsonl candidate.jsonl --json
```

`diff` exits with status 1 when trajectories diverge, making it suitable for CI.

## Pytest

```python
import pytest
from agent_cassette import EventType

@pytest.mark.cassette("research_agent.jsonl")
def test_agent(cassette):
    result = cassette.call(EventType.MODEL_CALL, "answer", {"question": "Why?"})
    assert result["completed"]
```

Replay committed cassettes by default, or refresh them explicitly:

```bash
pytest --cassette-mode=record
pytest --cassette-mode=replay
```

## Keyless Demo

```bash
python examples/research_agent/demo.py record
python examples/research_agent/demo.py replay
python examples/research_agent/demo.py
```

The final command records baseline and revised fake-agent runs, then reports their first divergence.

## Architecture

Each JSONL event carries a schema version, ID, timestamp, type, name, input, output, metadata, timing, optional cost, and parent/span relationships. The core package does not depend on an agent framework: adapters can map model, tool, and custom calls onto this event contract.

## Roadmap

- Partial replay and execution forks from selected events
- Tool timeout, malformed response, and API failure injection
- OpenTelemetry and OpenInference import/export
- OpenAI Agents SDK, PydanticAI, LangGraph, and MCP adapters
- GitHub Action for pull-request trajectory diffs
- Interactive local trajectory viewer

Agent Cassette is alpha software. Review cassettes before committing sensitive production data even though common credential fields are redacted by default.
