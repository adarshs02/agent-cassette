# Changelog

All notable changes to Agent Cassette are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onward.

## [Unreleased]

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
