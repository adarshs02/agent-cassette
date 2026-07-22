# Architecture

## What it can do

- **Offline replay** for OpenAI (Responses + Chat Completions), Anthropic Messages,
  Mistral Chat Completions, Gemini `generate_content`, and MCP tool calls — sync, async,
  and streaming.
- **Python tool record/replay** (`wrap_tool`, `cassette.tool`) for any sync or async
  Python callable — including generator and async-generator tools, whose yielded items,
  terminal return/error, and early close replay without running the real generator — with
  strict JSON-native input/output validation.
- **Framework capture** for the OpenAI Agents SDK (agents, LLM boundaries, tools,
  handoffs) and LangChain (Runnable boundaries + nested lifecycle spans). Ordinary OpenAI
  Agents `FunctionTool` callbacks are bridged at `on_invoke_tool`, and registered LangChain
  tools are bridged at the protected `_run`/`_arun` boundary (`wrap_langchain_tools`), so
  their results record and replay deterministically without running the real tool.
- **Time-travel forks** — replay a known prefix, then go live.
- **Failure injection** — deterministic exceptions, return values, latency, and rate
  limits.
- **Trajectory tests + CI reports** — assert on events, cost, duration, and errors; diff
  two runs. `tool_called`/`tool_not_called` assert on recorded/consumed tool boundaries by
  name, input, and count, and `Replayer.consumed_events` exposes what a replay session
  actually consumed (in cassette order); these verify Agent Cassette boundaries, not
  uninstrumented side effects.
- **Secret redaction** before anything hits disk, plus a script-free HTML viewer.
- **OpenTelemetry/OpenInference** JSON import and export.

## Core model

Every event carries a schema version, ID, timestamp, type, name, input, output,
metadata, duration, optional cost, and parent/span links. The core is
provider-independent:

- **`Recorder`** durably captures successful calls and failures.
- **`Replayer`** validates requests and returns or raises recorded outcomes.
- **`Hybrid`** composes replayed history with live or injected execution.
- Integrations translate provider calls onto this contract; assertions, reports, the
  viewer, and OTLP all consume the same model.

`AdapterRegistry` is an explicit, caller-owned extension point — no import-time plugin
discovery, no global state.

## Security

- Authorization, API-key, token, password, and secret fields are redacted recursively
  before write, along with connection-URI userinfo passwords
  (`scheme://user:password@host`) and secret-named URL query values.
- Replayed failures restore only allowlisted built-in exceptions; unknown types become
  `RecordedCallError` and are never dynamically imported.
- The viewer escapes all content, re-applies redaction, sets a restrictive CSP, and
  makes no network requests.
- Automatic CLI execution runs your script in-process — treat executed scripts as
  trusted code.
- Review cassettes before committing production data.

Full details and current limits (voice/realtime, browser streams, multi-process capture)
are in the [security model](security.md). Public API is inventoried in
[public-api.md](public-api.md).
