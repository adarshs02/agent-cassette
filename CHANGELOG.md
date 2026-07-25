# Changelog

All notable changes to Agent Cassette are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onward.

## [1.1.0] - Unreleased

Complete tool-replay release. Deterministically replay an agent's Python tool calls — scalar and
streaming — without executing the real tools, across `wrap_tool`, MCP, OpenAI Agents, and LangChain,
plus an agent-native non-interactive closed operational loop. Held unpublished until every
tool-replay phase (A core replay, B streaming, C1/C2 framework bridges, D verification UX) and the
Phase E closed loop passed acceptance, then shipped as one release. No new public Python export,
`EventType`, cassette schema (still v1), or runtime dependency; version is `1.1.0`.

### Added
- Tool record/replay (Phase A): `wrap_tool(function, cassette, *, name=None)` and the
  bound `cassette.tool` decorator (available on `Recorder`, `Replayer`, and `Hybrid`)
  wrap any sync or async Python callable so each call records, replays, or injects
  through a single `TOOL_CALL` boundary event; the session type alone decides record vs.
  replay vs. inject. Inputs are validated before the tool body runs and outputs before
  any event is persisted, using a private exact-type JSON copier that accepts only plain
  builtin JSON by type identity — subclasses (`IntEnum`, `str`/`float`/`list`/`dict`
  subclasses, and non-`str` mapping keys) are rejected, so a recorded value and its
  replay have identical Python types (no `IntEnum` → `int` drift). No
  `model_dump()`/`str()`/`repr()` or other duck-typed conversion is ever called; errors
  name only the value's type.
- Tool stream record/replay (Phase B): `wrap_tool` also wraps generator and
  async-generator tools, returning a streaming proxy that records and replays each
  yielded item, the terminal return value (sync), a terminal error, async cancellation,
  or a deliberate early `close()`/`aclose()` through one versioned tool-stream envelope
  (reusing `TOOL_CALL`/`ERROR`; no new public export, `EventType`, or schema version).
  Yielded items are exact-type validated and detached at yield time; replay never
  constructs or runs the underlying generator. A stream must be exhausted or explicitly
  closed to record (abandoned streams persist nothing); `send(None)`/`asend(None)` act as
  `next()`/`anext()` while non-`None` `send`/`asend` and `throw`/`athrow` raise
  `NotImplementedError`.
- OpenAI Agents `FunctionTool` replay bridge (Phase C1): `patch_openai_agents` now bridges
  ordinary SDK `FunctionTool` callbacks at the public `on_invoke_tool` boundary so their
  results record and replay through a versioned tool-result envelope (reusing `TOOL_RESULT`;
  no new public export, `EventType`, or schema version). On replay the agent loop runs
  against replayed model responses, hooks, guardrails, and handoffs while the original
  Python tool callback is never invoked. JSON-native results and the SDK structured outputs
  (`ToolOutputText`/`ToolOutputImage`/`ToolOutputFileContent`, and homogeneous lists of
  them) round-trip to the exact SDK type; unsupported outputs fail before any success event
  is persisted, without rendering the value. Non-`FunctionTool` local tools keep
  lifecycle-only capture, agent-as-tool values replay their nested loop, and legacy pre-C1
  lifecycle cassettes still replay through an unmarked fallback.
- LangChain registered-tool replay bridge (Phase C2): `wrap_langchain_tools(tools, cassette,
  *, name_prefix="langchain.tool")` returns bridged shallow clones of installed `BaseTool`
  objects (originals untouched) whose results record and replay through a versioned
  tool-result envelope (reusing `TOOL_CALL`; no new `EventType` or schema version). During
  replay the agent loop, args-schema validation, callbacks, output formatting, and
  `handle_tool_error` still run while the real `_run`/`_arun` body is never called. Supports
  `Tool`/`StructuredTool`/custom `BaseTool`, `invoke`/`ainvoke`/`run`/`arun`, and
  `content`/`content_and_artifact` results; matches on strict bounded-JSON
  args/kwargs/cleaned-config and translates a recorded `ToolException` so `handle_tool_error`
  reruns. Complements — does not replace — `wrap_langchain` and `langchain_callback_handler`.
- Tool-replay assertions (Phase D): public `tool_called(name, *, with_input=..., times=None,
  minimum=None, maximum=None, match="exact", ignore_paths=(), fuzzy_threshold=0.9)` and
  `tool_not_called(name, *, with_input=..., match="exact", ignore_paths=(), fuzzy_threshold=0.9)`
  predicates (in `agent_cassette.assertions`, composable with `assert_trajectory`/
  `check_trajectory`) assert on recorded logical tool-call boundaries — a `TOOL_CALL` event, or an
  ERROR whose logical `call_type` is `tool_call` (a failed call, counted once), one per invocation
  across `wrap_tool`, MCP, OpenAI Agents, and LangChain; `TOOL_RESULT` is never counted. Omitting
  `with_input` (the `Ellipsis` sentinel) matches any input; a supplied value is detached and
  strictly validated at creation, then matched with the Replayer's `normalize_input`/`inputs_match`
  machinery (`exact`/`subset`/`normalized`/`fuzzy`). Diagnostics stay secret-safe (JSON-native
  details only, never raw payloads or a value `repr`). A new read-only
  `Replayer.consumed_events -> tuple[Event, ...]` returns detached copies, in cassette order, of the
  events a session actually consumed. `agent-cassette check` gains repeatable name-only
  `--tool-called NAME` / `--tool-not-called NAME`. Combined with a full-consumption
  (`remaining == 0`) replay, these are the public, deterministic way to verify in CI that an agent
  run replayed its expected tool trajectory with zero live tool execution.
- Agent-native closed-loop CLI (Phase E): `agent-cassette setup`, `status`, `agent-manifest`,
  and `ci` plus named `record`/`replay`/`rerecord` give an agent one safe, non-interactive loop —
  preview/apply project scaffolding, know project/cassette state, record and replay cassettes by
  config-owned name, read a structured pass/mismatch report, then fix-and-retry or explicitly
  re-record. Every machine response and report file uses one canonical JSON envelope
  (`schema_version`/`command`/`status`/`ok`/`exit_code`/`project`/`warnings`/`changes`/
  `next_actions`/`data`) with argv-vector next actions and semantic exit codes (`0`/`1`/`2`).
  `setup` builds on the existing no-follow atomic-apply engine, records generated-file SHA-256 in
  `.agent-cassette/manifest.json`, never overwrites a file whose bytes differ (conflict), and never
  runs consumer code, reads env values, or installs dependencies. Named `record` is create-only and
  publishes a temporary cassette to the golden path only after full validation; `rerecord` is the
  sole explicit golden-update path; `replay` stays offline (zero live calls at supported boundaries)
  and writes a structured, secret-safe mismatch report. `ReplayMismatchError` gained backward-
  compatible structured fields and payload-safe messages. `ci --github` scaffolds a replay-only
  GitHub workflow (credential-empty env, no secrets, `permissions: contents: read`). Existing
  `init`, positional `record`/`replay`, `fork`, the pytest fixture, and the public Python API are
  unchanged; no new public Python export, `EventType`, schema, dependency, or version change.

### Security
- Connection-URI credential redaction: the recursive `redact()` path (recorder, hybrid,
  replay input normalization, assertions, and viewer) now scrubs passwords embedded in
  hierarchical connection URIs — the userinfo password in `scheme://user:password@host`
  (using the last `@` so an unescaped `@` inside a password is still removed) and
  secret-named URL query values (`?password=`/`?token=`/`?api_key=`/…) — for any valid
  scheme (`postgres`, `postgresql`, `mysql`, `mariadb`, `redis`, `rediss`, `mongodb`,
  `mongodb+srv`, `amqp`, `amqps`, and custom schemes). Scheme, username, host/port
  (including IPv6), path, non-secret query parameters, fragment, and surrounding prose are
  preserved byte-for-byte; percent-encoded secrets are removed; the scrub is idempotent.
  Ordinary URLs, emails, username-only userinfo, and `@`/`:` in paths/prose are left
  unchanged. A value such as `postgres://user:p@ssw0rd@db.internal/app` previously reached
  cassette/viewer output with the password intact. No public API, `EventType`, schema, or
  dependency change; `redact_secrets=False` remains the exact opt-out.
- Phase E filesystem and report trust: the closed loop's project I/O goes through one
  directory-FD / `O_NOFOLLOW` layer. Owned files (cassette, report, manifest, workflow, config,
  temporary) must be regular and single-link (`st_nlink == 1`); a symlink, hard link, or
  non-regular object fails closed with exit `2` and no symlink-following fallback. Reads are
  revalidated by a post-read `fstat` of the same descriptor (type, single-link, identity, and
  `st_size`/`st_mtime_ns`/`st_ctime_ns` stable), so an inode that gains a link or is rewritten
  mid-read is rejected. Child recordings stage in a private `mkdtemp(0o700)` outside the consumer
  tree and are copied in via FD; `record` is create-only, `rerecord` revalidates identity before a
  directory-relative atomic replace and preserves the old bytes on any pre-publish failure, and
  staging is removed on every exit including `KeyboardInterrupt`/`SystemExit`. Machine envelopes,
  blockers, and mismatch reports carry only code-owned, payload-free data (no recorded value,
  child string, environment value, credential, or `repr`).

## [1.0.1] - 2026-07-21

Trust-repair release. Also ships the Mistral and Gemini provider integrations that
landed after 1.0.0.

### Added
- Mistral provider support (`wrap_mistral`, `patch_mistral`, `agent-cassette[mistral]`):
  sync `chat.complete`, async `chat.complete_async`, and streaming `chat.stream` /
  `chat.stream_async`.
- Gemini (`google-genai`) provider support (`wrap_gemini`, `patch_gemini`,
  `agent-cassette[gemini]`): sync/async `generate_content` and streaming
  `generate_content_stream`, with `response.text` preserved on replay.
- Provider foundation: per-operation async and stream-operation triggering, and
  context-manager-tolerant record/replay streams.
- Provider foundation: `response_attributes` capture for computed response
  properties.
- `agent-cassette init --detect` recognizes Mistral (`mistralai`) and Gemini
  (`google-genai`) as providers.

### Changed
- Trust repair: the shared provider, MCP, and OpenAI Agents serialization no longer
  falls back to `str(value)` or coerces non-string mapping keys. Recorded values must
  be JSON-native, or come from a trusted integration SDK root (the provider's own
  package, e.g. `openai` or `mcp`) that is dumped through its `model_dump`; the
  `model_dump` of a value defined outside the trusted roots is never read or called.
  Unsupported values and non-string keys are rejected before persistence with a
  type-name-only error (never the value's `repr`). Values previously persisted via lossy
  stringification now raise `StrictJSONError` — a correction of behavior that violated
  the stable security contract.

### Security
- Hostile `__str__`/`__repr__` methods on recorded values are never invoked during
  serialization (enforced by adversarial tests).

## [1.0.0] - 2026-07-18

First stable release. The public API, CLI tree, adapter protocol, and cassette
schema v1 are frozen and enforced by contract-snapshot tests.

### Added
- `agent_cassette.__version__` exposes the installed distribution version.
- `[project.urls]` metadata (Homepage, Repository, Documentation, Issues) for PyPI.
- This changelog.
- Contract-freeze snapshot tests pinning the public CLI tree and callable signatures.

### Removed
- In-place cassette migration. `migrate_cassette` now requires a `destination`
  argument and `agent-cassette migrate` requires `--output`; the destination must
  differ from the source (raises `ValueError`, CLI exit 2).

## [0.15.0b1]

### Added
- Compatibility matrices for Python, OpenAI, Anthropic, OpenAI Agents, and LangChain,
  validated by isolated installed-wheel CI jobs.
- Public API inventory (`docs/public-api.md`) enforced by a snapshot test.
- Standardized CLI exit codes and a `AgentCassetteDeprecationWarning` deprecation path.
- Explicit cassette recovery (`recover_cassette`) for a torn, unterminated final write.
- Large-cassette benchmark with deterministic output.

### Changed
- Cassette loading stays fail-closed; serialization rejects unbounded or cyclic values
  and never falls back to arbitrary object stringification.

### Deprecated
- In-place cassette migration; pass a separate destination path instead.

## [0.14.0b1]

### Added
- Secure consumer-project bootstrap: `agent-cassette init` with `--detect`, `--dry-run`,
  and `--check`, a schema-versioned `.agent-cassette.toml`, cassette-directory
  scaffolding, and an offline smoke test.
- Static provider, framework, and test-framework detection without importing or
  executing consumer code.

## [0.13.0b2]

### Added
- LangChain lifecycle tracing: chain, model, retriever, parser, and tool spans with
  run-ID correlation and parent/child relationships, marked observational so they never
  disturb replay.

## [0.13.0b1]

### Added
- LangChain Runnable replay via `wrap_langchain`, supporting `invoke`, `ainvoke`,
  `stream`, `astream`, `batch`, and `abatch`.

## [0.12.0b1]

### Added
- Agent-friendly bootstrap: canonical `uv` workflow, `agent-cassette doctor`, and
  dependency/provider detection with actionable diagnostics.
