# Agent Cassette

**VCR.py for AI agents.** Record complete agent executions, replay them offline, fork from a known trajectory, inject deterministic failures, and turn real runs into regression tests.

Observability platforms show what happened. Agent Cassette reproduces what happened and makes the execution testable.

> `0.15.0b1` is a public beta. Cassette schema compatibility is maintained within the beta line, but integration APIs may still evolve before `1.0`.

## Why Agent Cassette

- Near-zero-instrumentation recording through the CLI
- Offline replay for OpenAI Responses and Chat Completions
- Offline replay for Anthropic Messages
- OpenAI Agents lifecycle capture for agents, LLM boundaries, tools, and handoffs
- Optional LangChain lifecycle tracing with nested chain, model, retriever, parser, and tool spans
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
uv sync --frozen --all-extras --dev
uv run --frozen pytest
```

The committed `uv.lock` is the dependency source of truth. See `AGENTS.md` for the
complete validation and package-smoke-test commands.

Release validation uses the installed locked environment and never needs a manual
`PYTHONPATH`:

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests examples benchmarks
uv run --frozen ruff format --check src tests examples benchmarks
uv run --frozen mypy src tests
uv build --no-build-isolation
```

When an agent needs a single-file repository snapshot, generate it on demand. The
output is ignored and must not be committed:

```bash
npx repomix --output repomix-output.xml
```

For library use, install the core package or only the integrations you need:

```bash
pip install agent-cassette
```

The core package uses only the standard library on Python 3.11 and newer, and the
`tomli` TOML reader on Python 3.10. Provider and framework integrations are optional:

```bash
pip install "agent-cassette[openai]"
pip install "agent-cassette[anthropic]"
pip install "agent-cassette[agents]"
pip install "agent-cassette[langchain]"
```

Tested provider ranges are OpenAI `>=1,<3`, Anthropic `>=0.34,<1`, OpenAI Agents
`>=0.1,<1`, and LangChain Core `>=0.3,<2`. Clean installed-wheel jobs cover core,
each extra, and `all`; see [the compatibility policy](docs/compatibility.md).

## Agent-first initialization

Initialize an existing Python project without changing its dependency manifests or
running project code:

```bash
cd your-project
agent-cassette init --detect
pytest tests/test_agent_cassette_smoke.py
```

Detection statically inspects `pyproject.toml`, `requirements*.txt`, `setup.cfg`,
and `setup.py`. Supported provider tokens are `anthropic`, `mcp`, `openai`, and
`openai-agents`. Framework detection recognizes `langchain` plus the `pytest` and
`unittest` test frameworks using static project evidence. Results are sorted for
deterministic reviews. Detection does not import the project, discover secrets,
install dependencies, or modify a dependency manifest. A manifest that cannot be
parsed produces a warning instead of executing or guessing from it.

The initializer creates only these owned scaffolds:

- `.agent-cassette.toml`, the project configuration
- `<cassette_dir>/.gitkeep`, preserving the configured cassette fixture directory
- `tests/test_agent_cassette_smoke.py`, an entirely offline record/replay test

Cassette fixtures remain tracked. Initialization does not create or edit
`.gitignore` because it generates no transient viewer output.

Initialization creates only missing scaffold files. Byte-for-byte matches make
repeated initialization a no-op; any different existing scaffold is a conflict and
is never overwritten, even if it contains an Agent Cassette marker from an earlier
version.

Writes use directory-relative, no-follow filesystem operations and revalidate path
components while applying the preflighted plan. Mutating initialization fails closed
when the platform does not provide the required dirfd/no-follow primitives.
Read-only `.agent-cassette.toml` loading and its pytest/replay runtime defaults remain
portable on supported Python platforms. Symlinked roots, parents, and targets are
rejected during initialization.

The schema-version-1 configuration supports these keys:

```toml
schema_version = 1
cassette_dir = "tests/cassettes"
match = "exact"
strict = true
providers = ["openai"]
frameworks = ["langchain"]
test_frameworks = ["pytest"]
```

`schema_version` must be `1`; `cassette_dir` is a project-relative path; `match` is
one of `exact`, `subset`, `normalized`, or `fuzzy`; `strict` is a boolean; and
`providers`, `frameworks`, and `test_frameworks` are sorted arrays of their
supported tokens above. `frameworks` contains integrations such as `langchain`;
test runners remain separate in `test_frameworks`. Future schema versions and
invalid known values are rejected. Unknown keys are preserved and reported with a
warning for forward compatibility.

The `cassette` pytest fixture reads `cassette_dir`, `match`, and `strict` from this
file. A `@pytest.mark.cassette(...)` marker can override the cassette filename,
`match`, or `strict` for one test; relative filenames remain under `cassette_dir`.

The replay CLI also uses `match` and `strict` from `.agent-cassette.toml` when a
command-line override is not supplied. Replay still requires an explicit cassette
path; `cassette_dir` does not silently choose a CLI cassette:

```bash
agent-cassette replay tests/cassettes/research.jsonl -- python research_agent.py
```

Coding agents can preview, apply, and verify setup with machine-readable output:

```bash
agent-cassette init . --detect --dry-run --json
agent-cassette init . --detect --json
agent-cassette init . --detect --check --json
```

The versioned JSON envelope contains `schema_version`, `command`, `mode`, `project`,
`detect`, `detected`, `actions`, `status`, `warnings`, `error`, and `next_steps`.
`detected` separates `providers`, `frameworks`, and `test_frameworks`.
Each action reports its project-relative `path`, `status`, and optional `reason`.
`--dry-run` previews a valid, conflict-free plan and exits 0 without writing; it
exits 2 for invalid or conflicting state. `--check` exits 0 when scaffolding is
current, 1 when changes are needed, and 2 for invalid or conflicting state; it also
never writes. Normal initialization exits 0 on success and 2 on invalid or
conflicting state. `--check` and `--dry-run` are mutually exclusive.

## Strict cassette format and recovery

Cassette schema v1 remains unchanged in 0.15. Each nonblank JSONL record must be a
strict Event object: duplicate object keys, `NaN`/`Infinity`, unsupported Python
objects, invalid fields, cycles, and excessive nesting are rejected. Writes never
fall back to `str` or `repr`. Normal load, replay, migration, and inspection fail
closed and identify the path and line without echoing cassette payloads.

Interrupted writes are never ignored implicitly. Recover only a malformed,
unterminated final byte fragment into a different destination:

```bash
agent-cassette recover interrupted.jsonl recovered.jsonl
agent-cassette recover interrupted.jsonl recovered.jsonl --json
```

Earlier corruption, newline-terminated corruption, and decoded-but-invalid Events
remain hard failures. Recovery never modifies its source. Python callers can use
`recover_cassette(source, output)` and inspect the returned `RecoveryReport`.
See [the cassette schema contract](docs/cassette-schema.md) and
[security model](docs/security.md) for exact boundaries.

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
imports cassette-named types. Recorder instances support independent concurrent
top-level calls; sharing one Replayer across concurrent calls remains unsupported.
Exhaust or explicitly close streams so their recordings become replayable.

### LangChain lifecycle tracing

Add an optional callback handler when you want framework-level trace spans as well
as a replayable Runnable boundary:

```python
from agent_cassette import Cassette, langchain_callback_handler, wrap_langchain

chain = prompt | model | parser

with Cassette.record("chain.jsonl") as cassette:
    handler = langchain_callback_handler(cassette)
    recorded = wrap_langchain(chain, cassette, name="research.chain")
    result = recorded.invoke(
        {"topic": "agent testing"},
        config={"callbacks": [handler]},
    )
```

Put existing application handlers in the same `callbacks` list. Agent Cassette
preserves LangChain run IDs as span IDs and parent run IDs as parent IDs. It emits
one terminal lifecycle event for each chain, model, retriever, parser, or tool run,
including failed runs and sync, async, and streaming execution.

Lifecycle events carry `metadata["_agent_cassette"]["observational"] == true`.
Replay matching skips only events with that exact marker, so traces do not duplicate
provider boundaries or disturb deterministic offline replay. The handler is inert
with a replay cassette; use `wrap_langchain(None, cassette, ...)` to replay results
without constructing or calling a live provider.

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
agent-cassette migrate old.jsonl --output upgraded.jsonl
agent-cassette recover interrupted.jsonl recovered.jsonl --json
```

Missing optional integrations are reported but do not make the environment
unhealthy. `doctor` exits with status `1` only when the core package or Python
environment is unusable.

## Schema Migrations

Cassette events carry a schema version. When a future release raises the schema
version, registered migrations upgrade older events transparently on load. Keep
the source as an upgrade and rollback point by passing a separate output:

```bash
agent-cassette migrate old.jsonl --output upgraded.jsonl
```

In-place `migrate_cassette(path)` and `agent-cassette migrate path` remain available
but emit `AgentCassetteDeprecationWarning`. Valid schema-v1 cassettes need no
migration for 0.15:

```python
from agent_cassette import register_migration

register_migration(1, upgrade_v1_to_v2)  # each migration advances exactly one version
```

Events newer than the installed release are rejected rather than silently misread.

## Deterministic large-cassette benchmark

Run the versioned benchmark report without a timing gate:

```bash
uv run --frozen python benchmarks/large_cassette.py \
  --events 10000 --output /tmp/agent-cassette-large.jsonl
```

The JSON report includes schema version, event count, byte count, SHA-256, and
read/write measurements. CI compares two generated cassettes by count, bytes, and
hash; elapsed time is diagnostic only, so slow shared runners do not cause flakes.

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
- Redaction rejects cyclic or more-than-64-container values with a payload-free
  `RedactionError`; shared acyclic aliases remain supported.
- Replayed failures restore only allowlisted built-in exceptions; unknown types become `RecordedCallError` rather than being dynamically imported.
- Review cassettes before committing sensitive production data.
- Provider streaming events are recorded when a stream is exhausted, fails, or is explicitly closed; partial chunks replay before a recorded stream failure is raised.
- `with_raw_response` and `with_streaming_response` transport helpers fail explicitly because their raw HTTP semantics cannot yet be replayed faithfully. The Anthropic `messages.stream` helper is rejected in favor of `messages.create(stream=True)`.
- Automatic CLI execution runs Python in-process so SDK constructor patching reaches user code; treat executed scripts as trusted code.
- Voice, realtime, browser/computer streams, and distributed multi-process capture are not yet supported.

Candidate public names are snapshotted in tests and inventoried in
[the public API document](docs/public-api.md). 0.15 deprecations and stricter data
handling are summarized in [the beta upgrade guide](docs/beta-upgrade.md). CLI
automation should follow the common [0/1/2 exit-code contract](docs/cli-exit-codes.md).

## Toward `1.0`

Feature expansion is frozen through 1.0. Remaining work is contract snapshots,
real-project offline fixtures, release-candidate validation, and a final decision to
implement cross-process coordination or keep its unsupported status explicit.
