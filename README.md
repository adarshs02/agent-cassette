# Agent Cassette

**VCR.py for AI agents.** Record complete agent executions, replay them offline, fork from a known trajectory, inject deterministic failures, and turn real runs into regression tests.

Observability platforms show what happened. Agent Cassette reproduces what happened and makes the execution testable.

> `0.12.0b1` is a public beta. Cassette schema compatibility is maintained within the beta line, but integration APIs may still evolve before `1.0`.

## Why Agent Cassette

- Near-zero-instrumentation recording through the CLI
- Offline replay for OpenAI Responses and Chat Completions
- Offline replay for Anthropic Messages
- OpenAI Agents lifecycle capture for agents, LLM boundaries, tools, and handoffs
- Sync and async streaming capture
- Durable JSONL writes, nested spans, and parallel-call replay matching
- Exact, subset, normalized, and fuzzy request matching
- Time-travel forks with replayed prefixes and live suffixes
- Deterministic returned-value, exception, latency, and rate-limit injection
- Versioned schema migrations for older cassettes
- Trajectory assertions, stable CI reports, and a reusable GitHub Action
- Secure standalone HTML viewer with no scripts or network requests
- OpenTelemetry/OpenInference JSON import and export
- MCP client tool-call recording and replay
- Secret redaction before cassette data reaches disk

## Install

For a deterministic contributor or coding-agent setup from a fresh checkout:

```bash
uv sync --frozen --all-extras
uv run --frozen pytest
```

The committed `uv.lock` is the dependency source of truth. See `AGENTS.md` for the
complete validation and package-smoke-test commands.

For library use, install the core package or only the integrations you need:

```bash
pip install agent-cassette
```

The core package has no runtime dependencies. Provider and framework integrations
are optional:

```bash
pip install "agent-cassette[openai]"
pip install "agent-cassette[anthropic]"
pip install "agent-cassette[agents]"
pip install "agent-cassette[langchain]"
```

## LangChain Runnables

Wrap an LCEL composition, `@chain` Runnable, or other LangChain Runnable at its
execution boundary:

```python
from agent_cassette import Cassette, wrap_langchain

chain = prompt | model | parser

with Cassette.record("chain.jsonl") as cassette:
    recorded = wrap_langchain(chain, cassette, name="research.chain")
    result = recorded.invoke({"topic": "agent testing"})

with Cassette.replay("chain.jsonl") as cassette:
    replayed = wrap_langchain(None, cassette, name="research.chain")
    result = replayed.invoke({"topic": "agent testing"})
```

`invoke`, `ainvoke`, `stream`, `astream`, `batch`, and `abatch` are supported.
Replay with `None` never constructs or calls a live Runnable. LangChain values use
safe, code-owned serialization envelopes within cassette schema v1; replay never
imports cassette-named types. Independent concurrent calls through wrapped
Runnables are not supported yet. Exhaust or explicitly close streams so their
recordings become replayable.

## Automatic Record and Replay

Run an ordinary Python script without changing supported provider client construction:

```bash
agent-cassette record tests/cassettes/research.jsonl -- python research_agent.py
agent-cassette replay tests/cassettes/research.jsonl -- python research_agent.py
```

Module execution is also supported:

```bash
agent-cassette record run.jsonl -- -m my_agent --topic "agent testing"
```

Within the executed program, standard clients are patched for the lifetime of the run:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(model="gpt-4.1-mini", input="Research agent testing")
```

Replay validates the request and returns an inert, attribute-compatible response object without constructing a live client, using an API key, dynamically importing cassette-specified code, or making a network request. `AsyncOpenAI`, `chat.completions.create`, derived clients, and streaming iterators are supported.

If the Anthropic SDK is installed, `Anthropic` and `AsyncAnthropic` clients are patched the same way. `messages.create` is recorded and replayed, including `stream=True` iterators:

```python
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Research agent testing"}],
)
```

Python callers can also wrap clients explicitly with `wrap_anthropic` or patch constructors with `patch_anthropic`, mirroring `wrap_openai` and `patch_openai`.

If the OpenAI Agents SDK is installed, `Runner.run`, `Runner.run_sync`, and `Runner.run_streamed` automatically receive lifecycle hooks. Agent starts and outputs, LLM boundaries, local tools, and handoffs become cassette events while existing user hooks continue to run.

## Time-Travel Forks

Replay a known prefix, then continue against live dependencies:

```bash
agent-cassette fork baseline.jsonl experiment.jsonl \
  --at 4 \
  -- python research_agent.py
```

The output cassette contains the replayed prefix and live suffix with lineage metadata. A fork permanently switches to live execution after its selected boundary.

Use `--mismatch live` to fork on the first request divergence rather than rejecting it:

```bash
agent-cassette fork baseline.jsonl prompt-change.jsonl \
  --mismatch live \
  -- python research_agent.py
```

## Deterministic Failure Injection

Pass JSON rules to a fork:

```json
[
  {
    "type": "tool_call",
    "name": "mcp.search",
    "occurrence": 2,
    "action": "raise",
    "error": "TimeoutError",
    "message": "search timed out"
  },
  {
    "type": "model_call",
    "name": "openai.responses.create",
    "action": "return",
    "value": {"output_text": "malformed fallback"}
  },
  {
    "type": "model_call",
    "action": "raise",
    "error": "RateLimitError",
    "message": "too many requests",
    "retry_after": 30
  },
  {
    "type": "tool_call",
    "action": "delay",
    "seconds": 2.5,
    "then": {"action": "raise", "error": "TimeoutError", "message": "slow backend"}
  }
]
```

```bash
agent-cassette fork baseline.jsonl chaos.jsonl \
  --inject failures.json \
  -- python research_agent.py
```

Rules are ordered and occurrence-based. Supported CLI exception types are `TimeoutError`, `ConnectionError`, `RateLimitError`, `ValueError`, and `RuntimeError`. A `delay` action sleeps deterministically and then applies its optional `then` action or continues with the live call. Python callers can use `InjectionRule`, `Return`, `Raise`, `Delay`, and `RateLimitError` directly; `RateLimitError` subclasses `ConnectionError` and carries an optional `retry_after`.

## Matching Dynamic Requests

Exact ordered matching is the default. Python callers can ignore dynamic fields, require only a recorded subset, or use a custom matcher:

```python
from agent_cassette import Cassette

with Cassette.replay(
    "run.jsonl",
    match="subset",
    ignore_paths=("kwargs.metadata.request_id",),
) as cassette:
    ...
```

Set `strict=False` to match any remaining event. This supports parallel calls that complete in a different order while still requiring every event to be consumed.

Two tolerant match modes survive cosmetic prompt edits:

```python
with Cassette.replay("run.jsonl", match="normalized") as cassette:
    ...  # whitespace and case differences in strings are ignored

with Cassette.replay("run.jsonl", match="fuzzy", fuzzy_threshold=0.9) as cassette:
    ...  # strings match when their similarity ratio meets the threshold
```

Both modes still require identical structure, keys, names, and non-string values; only string leaves are compared tolerantly.

## Trajectory Tests

```python
from agent_cassette import (
    assert_trajectory,
    contains_event,
    max_total_cost,
    no_errors,
)

assert_trajectory(
    "run.jsonl",
    no_errors(),
    contains_event("tool_call", name="search"),
    max_total_cost(0.05),
)
```

Available checks include event counts, required events, ordered event sequences, total cost, total duration, and error absence.

The CLI writes deterministic reports suitable for CI artifacts:

```bash
agent-cassette check run.jsonl \
  --no-errors \
  --require tool_call:search \
  --max-cost 0.05 \
  --report-json checks.json

agent-cassette diff baseline.jsonl candidate.jsonl \
  --report-json trajectory-diff.json
```

`check` and divergent `diff` commands exit with status `1`.

## GitHub Action

This repository includes a composite action:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: adarshs02/agent-cassette@main
    with:
      baseline: tests/cassettes/baseline.jsonl
      candidate: tests/cassettes/candidate.jsonl
      report: agent-cassette-report.json
```

The action installs Agent Cassette from its checked-out action directory, compares trajectories, and exposes the report path as an output.

## Secure Local Viewer

Generate a standalone, script-free HTML report:

```bash
agent-cassette view run.jsonl --output run.html
```

The viewer escapes all cassette content, applies redaction again, includes a restrictive Content Security Policy, performs no network requests, and bounds rendered event and payload sizes.

## OpenTelemetry and OpenInference

```bash
agent-cassette export-otlp run.jsonl trace.json
agent-cassette import-otlp trace.json restored.jsonl
```

Conversion is dependency-free and uses OTLP JSON trace spans with OpenInference semantic attributes. Event type, input, output, timing, cost, IDs, metadata, and parent relationships are preserved where representable.

## MCP

Wrap an MCP client session once:

```python
from agent_cassette import Cassette, wrap_mcp

async with Cassette.record("mcp.jsonl") as cassette:
    session = wrap_mcp(live_session, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})

async with Cassette.replay("mcp.jsonl") as cassette:
    session = wrap_mcp(None, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})
```

Sync and async `call_tool` methods are supported.

## Manual API

Framework-independent model and tool calls remain available:

```python
from agent_cassette import Cassette, EventType

with Cassette.record("run.jsonl") as cassette:
    result = cassette.call(
        EventType.TOOL_CALL,
        "search",
        {"query": "agent testing"},
        lambda: live_search("agent testing"),
    )

with Cassette.replay("run.jsonl") as cassette:
    result = cassette.call(
        EventType.TOOL_CALL,
        "search",
        {"query": "agent testing"},
    )
```

Use `call`/`acall` for sync/async execution and `recorder.span()` as a sync or async context to establish parent/span relationships. Completed events are flushed and fsynced immediately, then atomically normalized on context exit.

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
uv run --frozen pytest --cassette-mode=record
uv run --frozen pytest --cassette-mode=replay
```

## Other Commands

```bash
agent-cassette doctor
agent-cassette doctor --json
agent-cassette inspect run.jsonl
agent-cassette inspect run.jsonl --json
agent-cassette migrate old.jsonl
agent-cassette migrate old.jsonl --output upgraded.jsonl
```

Missing optional integrations are reported but do not make the environment
unhealthy. `doctor` exits with status `1` only when the core package or Python
environment is unusable.

## Schema Migrations

Cassette events carry a schema version. When a future release raises the schema version, registered migrations upgrade older events transparently on load, and `agent-cassette migrate` rewrites files in place:

```python
from agent_cassette import register_migration

register_migration(1, upgrade_v1_to_v2)  # each migration advances exactly one version
```

Events newer than the installed release are rejected rather than silently misread.

## Architecture

Each JSONL event includes a schema version, ID, timestamp, type, name, input, output, metadata, duration, optional cost, and parent/span relationships. The core remains provider-independent:

- `Recorder` durably captures successful calls and failures.
- `Replayer` validates requests and returns or raises recorded outcomes.
- `Hybrid` composes replayed history with live or injected execution.
- Integrations translate provider calls and lifecycle events onto this contract.
- Assertions, reports, the viewer, and OTLP conversion consume the same event model.

`AdapterRegistry` provides an explicit caller-owned extension point without import-time plugin discovery or process-global state.

## Security and Current Limits

- Common authorization, API-key, token, password, and secret fields are redacted recursively.
- Replayed failures restore only allowlisted built-in exceptions; unknown types become `RecordedCallError` rather than being dynamically imported.
- Review cassettes before committing sensitive production data.
- Provider streaming events are recorded when a stream is exhausted, fails, or is explicitly closed; partial chunks replay before a recorded stream failure is raised.
- `with_raw_response` and `with_streaming_response` transport helpers fail explicitly because their raw HTTP semantics cannot yet be replayed faithfully. The Anthropic `messages.stream` helper is rejected in favor of `messages.create(stream=True)`.
- Automatic CLI execution runs Python in-process so SDK constructor patching reaches user code; treat executed scripts as trusted code.
- Voice, realtime, browser/computer streams, and distributed multi-process capture are not yet supported.

## Toward `1.0`

The remaining stabilization work is broader provider/framework adapters (Gemini, Bedrock, LangChain), multi-process ordering, richer pull-request annotations, and compatibility testing against supported SDK release ranges.
