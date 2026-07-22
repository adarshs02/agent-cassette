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

Predicates: `no_errors`, `contains_event`, `event_count`, `event_sequence`,
`max_total_cost`, `max_total_duration_ms`, `tool_called`, `tool_not_called`.

### Verifying tool replay

`tool_called(name, *, with_input=..., times=None, minimum=None, maximum=None, match="exact",
ignore_paths=(), fuzzy_threshold=0.9)` and `tool_not_called(name, *, with_input=..., match=...)`
assert on recorded logical tool-call boundaries — a `TOOL_CALL` event (or an ERROR whose logical
`call_type` is `tool_call`, i.e. a failed call), matched by exact name, one per invocation across
`wrap_tool`, MCP, OpenAI Agents, and LangChain (`TOOL_RESULT` is never counted). Omitting
`with_input` matches any input; passing a value matches it with the same `normalize_input`/
`inputs_match` machinery the Replayer uses (`exact`/`subset`/`normalized`/`fuzzy`).

`Replayer.consumed_events` returns detached copies (in cassette order) of the events this session
actually consumed. Combine it with a full-consumption check to verify a replay reproduced the tool
trajectory with zero live tool execution — replay never runs the real tool body:

```python
from agent_cassette import Cassette, assert_trajectory, tool_called, tool_not_called, wrap_tool

with Cassette.replay("run.jsonl", strict=False) as replayer:
    search = wrap_tool(real_search, replayer)      # or wrap_langchain_tools / patch_openai_agents
    ...  # drive your agent; tool bodies never execute
    assert replayer.remaining == 0                 # every recorded tool boundary consumed
    assert_trajectory(
        replayer.consumed_events,
        tool_called("search", with_input={"args": ["agents"], "kwargs": {}}, match="subset"),
        tool_not_called("send_email"),
    )
```

The CLI mirrors the name-only checks: `agent-cassette check run.jsonl --tool-called search
--tool-not-called send_email` (repeatable). Structured-input and count assertions are Python-only.

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
