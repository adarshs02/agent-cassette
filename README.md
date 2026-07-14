# Agent Cassette

**VCR.py for AI agents.** Record a real agent run once, then replay it offline — no API keys, no network, no flakiness. Fork from any point, inject failures, and turn recordings into regression tests.

Observability tools show you what happened. Agent Cassette *reproduces* what happened and makes it testable.

> `0.15.0b1` — public beta. Cassette files stay compatible across the beta; integration APIs may still change before `1.0`.

## Install

```bash
pip install agent-cassette
```

The core has no dependencies beyond the standard library (plus `tomli` on Python 3.10). Add only the integrations you use:

```bash
pip install "agent-cassette[openai]"
pip install "agent-cassette[anthropic]"
pip install "agent-cassette[agents]"      # OpenAI Agents SDK
pip install "agent-cassette[langchain]"
```

Tested against OpenAI `>=1,<3`, Anthropic `>=0.34,<1`, OpenAI Agents `>=0.1,<1`, LangChain Core `>=0.3,<2`. See [compatibility](docs/compatibility.md). Contributing? Setup and test commands live in [AGENTS.md](AGENTS.md).

## Quickstart

Wrap any Python script. On `record`, real calls run and get saved. On `replay`, the same script runs with zero live calls:

```bash
agent-cassette record run.jsonl -- python agent.py    # runs live, saves to run.jsonl
agent-cassette replay run.jsonl -- python agent.py     # offline, no keys, deterministic
```

Your script needs no changes — supported clients are patched for the run:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(model="gpt-4.1-mini", input="Research agent testing")
```

On replay, that call returns an inert, attribute-compatible response from the cassette — no client, no key, no network, no dynamic imports.

## Set up an existing project

```bash
cd your-project
agent-cassette init --detect                       # writes config + an offline smoke test
pytest tests/test_agent_cassette_smoke.py
```

`init --detect` statically inspects your manifests (never imports or runs your code, never touches dependencies) and writes three owned files: `.agent-cassette.toml`, a cassette fixture dir, and an offline record/replay smoke test. It only creates missing files and never overwrites yours. Add `--dry-run`, `--check`, or `--json` for previews and machine-readable CI output.

Configure defaults in `.agent-cassette.toml` (drives the pytest fixture and replay CLI):

```toml
schema_version = 1
cassette_dir = "tests/cassettes"
match = "exact"        # exact | subset | normalized | fuzzy
strict = true
providers = ["openai"]
frameworks = ["langchain"]
test_frameworks = ["pytest"]
```

## What it can do

- **Offline replay** for OpenAI (Responses + Chat Completions), Anthropic Messages, and MCP tool calls — sync, async, and streaming.
- **Framework capture** for the OpenAI Agents SDK (agents, LLM boundaries, tools, handoffs) and LangChain (Runnable boundaries + nested lifecycle spans).
- **Time-travel forks** — replay a known prefix, then go live.
- **Failure injection** — deterministic exceptions, return values, latency, and rate limits.
- **Trajectory tests + CI reports** — assert on events, cost, duration, and errors; diff two runs.
- **Secret redaction** before anything hits disk, plus a script-free HTML viewer.
- **OpenTelemetry/OpenInference** JSON import and export.

## Usage

### Anthropic and OpenAI Agents (automatic)

If installed, `Anthropic`/`AsyncAnthropic` are patched like OpenAI (`messages.create`, including `stream=True`). The OpenAI Agents SDK gets lifecycle hooks on `Runner.run`, `run_sync`, and `run_streamed` automatically. Python callers can also wrap explicitly with `wrap_openai`, `wrap_anthropic`, or patch constructors with `patch_openai` / `patch_anthropic`.

### LangChain

Wrap a Runnable at its execution boundary:

```python
from agent_cassette import Cassette, wrap_langchain

chain = prompt | model | parser

with Cassette.record("chain.jsonl") as cassette:
    recorded = wrap_langchain(chain, cassette, name="research.chain")
    result = recorded.invoke({"topic": "agent testing"})

with Cassette.replay("chain.jsonl") as cassette:
    replayed = wrap_langchain(None, cassette, name="research.chain")   # None = never touches a live Runnable
    result = replayed.invoke({"topic": "agent testing"})
```

`invoke`, `ainvoke`, `stream`, `astream`, `batch`, and `abatch` are supported. For framework-level trace spans, add `langchain_callback_handler(cassette)` to your `config={"callbacks": [...]}`; trace events are marked observational and never disturb replay.

### MCP

```python
from agent_cassette import Cassette, wrap_mcp

async with Cassette.record("mcp.jsonl") as cassette:
    session = wrap_mcp(live_session, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})

async with Cassette.replay("mcp.jsonl") as cassette:
    session = wrap_mcp(None, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})
```

### Manual API

Framework-independent, for any model or tool call:

```python
from agent_cassette import Cassette, EventType

with Cassette.record("run.jsonl") as cassette:
    result = cassette.call(
        EventType.TOOL_CALL, "search", {"query": "agent testing"},
        lambda: live_search("agent testing"),
    )

with Cassette.replay("run.jsonl") as cassette:
    result = cassette.call(EventType.TOOL_CALL, "search", {"query": "agent testing"})
```

Use `call`/`acall` for sync/async and `recorder.span()` to nest parent/child relationships.

### Pytest

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

The `cassette` fixture reads `cassette_dir`, `match`, and `strict` from `.agent-cassette.toml`; a `@pytest.mark.cassette(...)` marker overrides per test.

## Forks, failure injection, and matching

Replay a prefix then continue live, optionally forking on the first request that diverges:

```bash
agent-cassette fork baseline.jsonl experiment.jsonl --at 4 -- python agent.py
agent-cassette fork baseline.jsonl prompt-change.jsonl --mismatch live -- python agent.py
```

Inject deterministic failures with an ordered, occurrence-based rules file (`raise`, `return`, `delay`, rate limits):

```bash
agent-cassette fork baseline.jsonl chaos.jsonl --inject failures.json -- python agent.py
```

Matching is exact and ordered by default. Loosen it when requests carry dynamic fields:

```python
with Cassette.replay("run.jsonl", match="subset", ignore_paths=("kwargs.metadata.request_id",)) as cassette:
    ...
# match="normalized" ignores whitespace/case in strings; match="fuzzy" compares by similarity ratio.
# strict=False allows parallel calls to complete out of order while still consuming every event.
```

## Trajectory tests and CI

```python
from agent_cassette import assert_trajectory, contains_event, max_total_cost, no_errors

assert_trajectory(
    "run.jsonl",
    no_errors(),
    contains_event("tool_call", name="search"),
    max_total_cost(0.05),
)
```

Deterministic reports for CI artifacts:

```bash
agent-cassette check run.jsonl --no-errors --require tool_call:search --max-cost 0.05 --report-json checks.json
agent-cassette diff baseline.jsonl candidate.jsonl --report-json trajectory-diff.json
```

`check` and a divergent `diff` exit `1`. A reusable GitHub Action ships in this repo:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: adarshs02/agent-cassette@main
    with:
      baseline: tests/cassettes/baseline.jsonl
      candidate: tests/cassettes/candidate.jsonl
      report: agent-cassette-report.json
```

## Other commands

```bash
agent-cassette view run.jsonl --output run.html      # standalone, script-free HTML viewer
agent-cassette inspect run.jsonl                     # summarize a cassette
agent-cassette export-otlp run.jsonl trace.json      # OTLP/OpenInference JSON
agent-cassette import-otlp trace.json restored.jsonl
agent-cassette migrate old.jsonl --output upgraded.jsonl
agent-cassette recover interrupted.jsonl recovered.jsonl   # salvage an incomplete final write
agent-cassette doctor                                # environment + integration health
```

Cassettes are JSONL, one strict event per line (schema v1). Loading fails closed on corruption; `recover` only salvages a torn final byte fragment into a new file. See the [schema contract](docs/cassette-schema.md), [CLI exit codes](docs/cli-exit-codes.md), and [migrations](docs/beta-upgrade.md).

## Architecture

Every event carries a schema version, ID, timestamp, type, name, input, output, metadata, duration, optional cost, and parent/span links. The core is provider-independent:

- **`Recorder`** durably captures successful calls and failures.
- **`Replayer`** validates requests and returns or raises recorded outcomes.
- **`Hybrid`** composes replayed history with live or injected execution.
- Integrations translate provider calls onto this contract; assertions, reports, the viewer, and OTLP all consume the same model.

`AdapterRegistry` is an explicit, caller-owned extension point — no import-time plugin discovery, no global state.

## Security

- Authorization, API-key, token, password, and secret fields are redacted recursively before write.
- Replayed failures restore only allowlisted built-in exceptions; unknown types become `RecordedCallError` and are never dynamically imported.
- The viewer escapes all content, re-applies redaction, sets a restrictive CSP, and makes no network requests.
- Automatic CLI execution runs your script in-process — treat executed scripts as trusted code.
- Review cassettes before committing production data.

Full details and current limits (voice/realtime, browser streams, multi-process capture) are in the [security model](docs/security.md). Public API is inventoried in [public-api.md](docs/public-api.md).

## License

See [LICENSE](LICENSE).
