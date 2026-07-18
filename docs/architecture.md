# Architecture

## What it can do

- **Offline replay** for OpenAI (Responses + Chat Completions), Anthropic Messages, and
  MCP tool calls — sync, async, and streaming.
- **Framework capture** for the OpenAI Agents SDK (agents, LLM boundaries, tools,
  handoffs) and LangChain (Runnable boundaries + nested lifecycle spans).
- **Time-travel forks** — replay a known prefix, then go live.
- **Failure injection** — deterministic exceptions, return values, latency, and rate
  limits.
- **Trajectory tests + CI reports** — assert on events, cost, duration, and errors; diff
  two runs.
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
  before write.
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
