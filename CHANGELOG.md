# Changelog

All notable changes to Agent Cassette are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onward.

## [Unreleased]

### Added
- `agent_cassette.__version__` exposes the installed distribution version.
- `[project.urls]` metadata (Homepage, Repository, Documentation, Issues) for PyPI.
- This changelog.

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
