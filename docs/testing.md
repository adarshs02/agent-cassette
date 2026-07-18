# Testing

## Pytest

```python
import pytest
from agent_cassette import EventType

@pytest.mark.cassette("research_agent.jsonl")
def test_agent(cassette):
    result = cassette.call(EventType.MODEL_CALL, "answer", {"question": "Why?"})
    assert result["completed"]
```

```bash
pytest --cassette-mode=record
pytest --cassette-mode=replay
```

The `cassette` fixture reads `cassette_dir`, `match`, and `strict` from
`.agent-cassette.toml`; a `@pytest.mark.cassette(...)` marker overrides per test.

## Trajectory tests

Assert on events, cost, duration, and errors across a whole run:

```python
from agent_cassette import assert_trajectory, contains_event, max_total_cost, no_errors

assert_trajectory(
    "run.jsonl",
    no_errors(),
    contains_event("tool_call", name="search"),
    max_total_cost(0.05),
)
```

## CI reports

Deterministic reports for CI artifacts:

```bash
agent-cassette check run.jsonl --no-errors --require tool_call:search --max-cost 0.05 --report-json checks.json
agent-cassette diff baseline.jsonl candidate.jsonl --report-json trajectory-diff.json
```

`check` and a divergent `diff` exit `1`. See [CLI exit codes](cli-exit-codes.md).

## GitHub Action

A reusable action ships in this repo:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: adarshs02/agent-cassette@main
    with:
      baseline: tests/cassettes/baseline.jsonl
      candidate: tests/cassettes/candidate.jsonl
      report: agent-cassette-report.json
```
