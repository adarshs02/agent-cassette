# Changelog

All notable changes to Agent Cassette are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onward.

## [Unreleased]

## [1.1.0] - Unreleased

Tool record/replay: deterministically replay an agent's Python tool calls — scalar and
streaming — without executing the real tools. Held unpublished until all tool-replay
phases (A core replay, B streaming, C1/C2 framework bridges, D verification UX) pass
acceptance, then shipped as one release.

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
