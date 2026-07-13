# Agent Cassette Release Plan

Target: `1.0.0`

This file is durable release state. Versions execute sequentially. A version may be
marked complete only after its implementation, complete validation suite, Sol-level
review, review fixes, version synchronization, and Git checkpoint all finish.

## Global invariants

- Preserve deterministic offline replay and recursive secret redaction.
- Never import a type named by cassette data. Replay reconstruction uses only a
  code-owned allowlist.
- Preserve schema compatibility. Raise `SCHEMA_VERSION` only when existing JSONL
  event structure cannot represent a feature.
- Preserve unrelated user changes and never push, publish, merge, or create a
  remote release.
- Use `uv.lock` as the dependency lock and `uv sync --frozen --all-extras` as the
  canonical development setup. All release commands use `uv run --frozen`.
- Required release gates: full pytest suite, Ruff lint and formatting, configured
  mypy type checking, wheel and sdist builds, installed-wheel CLI smoke tests, and
  clean Git status. Builds use the locked Hatchling environment with
  `uv build --no-build-isolation`.
- Each release gets one coherent checkpoint commit before the next starts.
- Artifact smoke tests run outside the checkout with `PYTHONPATH` removed.

## 0.11.1b1 — Seamless setup

Status: **complete**

Architecture decision: diagnostics live in a dependency-free `diagnostics.py` and
use `importlib.metadata` / `importlib.util.find_spec` only; CLI renders schema-v1
data. Build/setup uses locked uv commands, an installed-package subprocess test,
and artifact smoke tests outside the checkout. No schema or runtime API changes.

Assigned routes/subtasks: Terra-equivalent `doctor_builder` owns diagnostics, CLI,
and focused tests. Terra-equivalent `setup_builder` owns dependency/build metadata,
lockfile, AGENTS/README/CI, and fresh-checkout tests. Root Sol integrates and reviews.

Review attempts: 3/3 — final Sol-equivalent review found no material issues

Validation evidence: `uv lock --check`; frozen all-extras sync; 117 pytest tests;
Ruff lint and format across `src tests examples`; mypy across 52 configured files;
wheel and sdist build; clean Python 3.13 wheel install outside checkout with
`PYTHONPATH` removed; CLI help and doctor JSON smoke tests. All passed.

Checkpoint commit: this version checkpoint commit; hash recorded in following
durable status update and final report.

### Goal

Make a fresh checkout deterministic and self-describing for humans and coding
agents, and expose actionable installation diagnostics.

### Included work

- Add and commit `uv.lock`.
- Define deterministic development, lint, format, type-check, test, and build
  commands.
- Add root `AGENTS.md` with exact commands and repository-specific safety rules.
- Ensure tests run after fresh setup without manual `PYTHONPATH` changes.
- Add `agent-cassette doctor` with stable human output and deterministic JSON.
- Report Python/package state and availability of OpenAI, Anthropic, OpenAI Agents,
  MCP, and LangChain-related modules without importing optional packages.
- Make CLI descriptions provider-neutral.
- Add clean-install and package-build CI.
- Keep vendored `.agents/` tools outside project lint/type-check scopes.
- Add mypy as the repository type checker and define its exact checked paths.
- Remove subprocess-test `PYTHONPATH` injection; exercise installed package state
  from a working directory outside the checkout.

### Explicit exclusions

- No project scaffolding or configuration file; those belong to `0.13.0b1`.
- No LangChain execution integration.
- No cassette schema change.
- No automated package installation from `doctor`.

### Dependencies

- Existing Hatchling build backend.
- `uv` for lock and deterministic contributor setup.
- Standard-library discovery APIs for diagnostics.

### Compatibility and migration risks

- Lock generation must cover supported Python 3.10–3.13 without narrowing current
  runtime compatibility.
- Add Python 3.13 classifier and CI coverage to match the supported matrix.
- Doctor JSON schema v1 is `{schema_version, package, python, integrations,
  healthy}` with sorted keys. `package` reports distribution version/install state;
  `python` reports version/supported; each integration reports `supported`,
  `distribution`, and `installed`. OpenAI Agents maps distribution
  `openai-agents` to import `agents`; MCP reports built-in support separately from
  optional SDK discovery. Human and JSON modes write data only to stdout,
  diagnostics only to stderr, and return 0 when core environment is healthy or 1
  when Python/package state is unusable. Missing optional integrations are healthy.
- CI must install the built wheel, not accidentally import from the checkout.

### Acceptance criteria

- `uv sync --frozen --all-extras` produces a working environment from a clean
  checkout without changing `uv.lock`.
- `uv run --frozen pytest` passes without `PYTHONPATH` manipulation.
- `agent-cassette doctor` and `agent-cassette doctor --json` return success and
  correctly report optional integrations.
- CLI record/replay/fork help is provider-neutral.
- CI builds wheel and sdist, installs the wheel into a clean environment, runs CLI
  smoke tests, and executes the supported Python matrix.
- README and AGENTS commands agree.

### Required validation

- Targeted doctor and CLI tests.
- `uv run --frozen pytest`
- `uv run --frozen ruff check src tests examples`
- `uv run --frozen ruff format --check src tests examples`
- `uv run --frozen mypy src tests`
- `uv build --no-build-isolation`
- Clean virtual-environment wheel install and `agent-cassette --help` / `doctor`
  smoke tests.

## 0.12.0b1 — LangChain Runnable support

Status: **planned**

Architecture decision: pending

Assigned routes/subtasks: pending

Review attempts: 0/3

Validation evidence: pending

Checkpoint commit: pending

### Goal

Record and replay LangChain Runnable boundaries with offline-safe behavior across
sync, async, streaming, and batch APIs.

### Included work

- Add optional `langchain` dependency backed by `langchain-core` and include it in
  development/all dependency sets.
- Add a Runnable-compatible wrapper and public `wrap_langchain` API.
- Support `invoke`, `ainvoke`, `stream`, `astream`, `batch`, and `abatch`.
- Support LCEL compositions and `@chain` Runnables.
- Add code-owned, versioned serialization envelopes for safe LangChain values.
- Reconstruct only explicitly allowlisted LangChain message, message-chunk,
  document, and generation value types.
- Replay without constructing or calling a live Runnable/provider.
- Preserve partial-stream completion/error semantics consistent with provider
  integrations.
- Document direct Python usage and installation.
- Contract: `wrap_langchain(runnable: Runnable | None, cassette, *,
  name="langchain.runnable") -> Runnable`. The wrapper subclasses Runnable,
  delegates input/output schemas, config specs, graph/introspection attributes, and
  unknown safe attributes only when a live Runnable exists; replay with `None`
  retains execution methods but unavailable introspection fails explicitly.
- Each top-level method records one replayable boundary event. Batch/async batch are
  atomic boundaries with stable input/result ordering; `return_exceptions` is
  serialized using the existing safe exception allowlist. Independent concurrent
  top-level calls are explicitly unsupported until Recorder/Replayer locking lands.
- Matching input contains value plus output-affecting `RunnableConfig` fields
  (`configurable`, `tags`, `metadata`, `recursion_limit`, `max_concurrency`) and
  method kwargs. Ephemeral callbacks and run IDs are excluded. Unknown config keys
  are retained as serialized values unless explicitly documented ephemeral.
- Envelopes use fixed keys `__agent_cassette_langchain__`, `version`, `type`, and
  `data`, currently version 1. Decoder dispatch comes only from a fixed code mapping;
  cassette module/class strings are never imported. Values have bounded recursion
  depth and cycle detection; unsupported/cyclic values raise a precise serialization
  error before disk write.
- Streams finalize only on exhaustion, explicit `close`/`aclose`, or iteration
  failure/cancellation. Abandoned streams without close are incomplete and not
  replayable; document this. Replayed errors use existing allowlisting.

### Explicit exclusions

- No lifecycle callback tracing; `0.12.1b1` owns internal spans.
- No automatic global monkey-patching of LangChain.
- No LangGraph-specific state interception beyond its Runnable-compatible surface.
- No arbitrary object or cassette-directed type reconstruction.

### Dependencies

- `0.11.1b1` deterministic setup and validation.
- Supported `langchain-core>=0.3,<2` Runnable and serializable value contracts, with
  minimum and current-compatible CI coverage.
- Existing Recorder/Replayer call and stream semantics.

### Compatibility and migration risks

- LangChain has multiple major API lines; the supported version range and tested
  matrix must be explicit.
- Batch concurrency may complete out of order; recorded return order must still
  follow Runnable API guarantees.
- Stream recordings must not be finalized before exhaustion, explicit close, or
  failure.
- Existing event JSON can contain versioned payload envelopes, so schema v1 is
  expected to remain sufficient. If implementation disproves this, Sol must design
  a one-step migration and update this plan before code changes.
- Existing Recorder/Replayer/storage mutation is unsynchronized. This version uses
  one atomic event per top-level batch and documents independent concurrent wrapper
  calls as unsupported; it does not claim general parallel replay safety.

### Acceptance criteria

- Wrapped LCEL and `@chain` Runnables retain Runnable compatibility.
- All six required execution methods record and replay equal supported values.
- Replay with `None` as live Runnable makes no provider/network call.
- Unsupported values degrade to safe JSON data or fail with a precise error; no
  dynamic imports occur.
- Sync/async failures and partial streams replay deterministically.
- Wrapper preserves `**kwargs`, supported `RunnableConfig`, and batch
  `return_exceptions` semantics.
- Schema-v1 cassettes recorded before LangChain support remain readable.

### Required validation

- Unit tests for serialization allowlist and hostile type metadata.
- Runnable, LCEL, `@chain`, stream, async stream, batch, async batch, and failure
  tests.
- Full repository gates from global invariants.
- Installed-wheel LangChain record/replay smoke test.

## 0.12.1b1 — LangChain lifecycle tracing

Status: **planned**

Architecture decision: pending

Assigned routes/subtasks: pending

Review attempts: 0/3

Validation evidence: pending

Checkpoint commit: pending

### Goal

Offer optional internal LangChain lifecycle traces with correct hierarchy and no
duplicate replay/provider events.

### Included work

- Add public LangChain callback handler.
- Trace chain, model, retriever, parser, and tool lifecycle spans.
- Preserve `run_id` / `parent_run_id` relationships under concurrent callbacks.
- Mark tracing events observational so strict replay does not consume them.
- Avoid duplicate provider model/tool events while retaining lifecycle context.
- Cover sync, async, streaming, cancellation, and failure callbacks.
- Document callback composition through `RunnableConfig`.
- Represent each lifecycle run as one terminal observational event with
  `metadata._agent_cassette.observational=true`, `span_id=<run_id>`, and
  `parent_id=<parent_run_id>`. Event names identify lifecycle category and terminal
  outcome. Model/tool lifecycle uses `CUSTOM`; replayable provider events remain
  `MODEL_CALL`/`TOOL_CALL`.
- Add synchronized Recorder/storage mutation boundaries so concurrent callback
  events cannot race file appends or `events` updates.

### Explicit exclusions

- No replay short-circuiting through callbacks; wrapper remains replay boundary.
- No LangSmith transport or remote tracing integration.
- No process-global callback installation.
- No project scaffolding changes.

### Dependencies

- `0.12.0b1` serialization and LangChain dependency.
- Existing event span/parent fields.

### Compatibility and migration risks

- Callback sequences differ across Runnable types and LangChain versions.
- Callback methods can arrive from multiple threads/tasks; mapping updates require
  synchronization.
- Observational metadata must be ignored by replay without hiding actual calls.
- Parser lifecycle may surface as chain callbacks; classification is best-effort
  using documented LangChain metadata and otherwise remains `chain`.

### Acceptance criteria

- Nested lifecycle events preserve parent/child relationships.
- Strict replay ignores observational events and consumes every execution boundary.
- Provider calls remain single replayable events when callbacks are active.
- Existing user callbacks compose normally.
- Failure events retain original run hierarchy and sanitized payloads.
- Replay ignores exactly events marked observational and does not hide ordinary
  custom/error events.
- Concurrent and late/out-of-order children, duplicate terminal callbacks,
  cancellation, and handler use during replay have defined tested behavior.

### Required validation

- Callback unit tests for every required lifecycle category.
- Concurrent sync/async hierarchy tests.
- Provider-plus-callback deduplication tests.
- Stream and failure tests.
- Full repository gates from global invariants.

## 0.13.0b1 — Agent-first initialization

Status: **planned**

Architecture decision: pending

Assigned routes/subtasks: pending

Review attempts: 0/3

Validation evidence: pending

Checkpoint commit: pending

### Goal

Let a coding agent initialize Agent Cassette in a consumer repository safely,
idempotently, and without guessing project structure.

### Included work

- Add `agent-cassette init --detect`.
- Add idempotent `--check` and non-mutating `--dry-run` modes.
- Define and load `agent-cassette.toml` schema version 1. Supported keys:
  `schema_version`, `cassette_dir`, `match`, `strict`, `providers`, and `frameworks`.
  Reject future schema versions and invalid known values; preserve/ignore unknown
  keys with a warning for forward compatibility. Use `tomli>=2` only on Python
  `<3.11` and standard-library `tomllib` otherwise.
- Detect supported providers/frameworks from `pyproject.toml`, `requirements*.txt`,
  `setup.cfg`, and `setup.py` using static text/config parsing without executing code.
- Scaffold `agent-cassette.toml`, `tests/cassettes/.gitkeep`, and
  `tests/test_agent_cassette_smoke.py`. Do not ignore cassette fixtures; they are
  regression artifacts. Add only a marked Agent Cassette block for transient viewer
  outputs to `.gitignore` when needed.
- Print concise next commands tailored to detected integrations.
- Return deterministic exit codes and optional JSON-friendly diagnostics where
  appropriate.
- Document generated files, conflict behavior, and automation usage.

### Explicit exclusions

- No dependency installation or modification of dependency manifests.
- No secret discovery, API-key writing, or execution of consumer code.
- No overwrite of non-generated user files.
- No support for non-Python project detection.

### Dependencies

- `0.11.1b1` doctor and setup language.
- `0.12.x` framework inventory and LangChain examples.
- Standard-library TOML reader on Python 3.11+, with a safe Python 3.10 strategy.

### Compatibility and migration risks

- Initialization touches user repositories; path traversal, symlinks, encoding, and
  partial-write behavior require security review.
- Configuration must tolerate unknown future keys while rejecting invalid known
  values.
- Repeated runs must not duplicate `.gitignore` or alter accepted scaffolds.
- Modes are mutually exclusive: `--dry-run` previews and returns 0 when applicable;
  `--check` performs no writes and returns 0 when current, 1 when changes are needed,
  2 on invalid/conflicting state; normal mode returns 0 on success, 2 on conflict.
- Perform full preflight before writes, resolve every target under project root,
  reject symlinked targets/parents, write atomically, and roll back files created by
  the current run after partial failure. Existing non-generated conflicts are never
  overwritten.

### Acceptance criteria

- `init --detect --dry-run` changes nothing and reports planned files.
- First `init --detect` creates only documented files inside the project root.
- Second run makes no changes.
- `init --check` returns success only when required generated state is present and
  compatible.
- Provider/framework detection has fixture coverage for supported manifest forms.
- Generated smoke test records and replays without a live provider.

### Required validation

- Filesystem, symlink, conflict, idempotency, dry-run, check, and detection tests.
- Generated-project smoke test.
- Full repository gates from global invariants.
- Installed-wheel initialization smoke test in an empty temporary project.

## 1.0.0 — Stable public release

Status: **planned**

Architecture decision: pending

Assigned routes/subtasks: pending

Review attempts: 0/3

Validation evidence: pending

Checkpoint commit: pending

### Goal

Freeze the documented public contract and prove fresh-install/offline-replay behavior
for the first stable release.

### Included work

- Audit and explicitly stabilize supported public APIs.
- Add public `__version__` and typed-package marker if absent.
- Publish Python, provider SDK, LangChain/LangGraph, and cassette schema compatibility
  matrix.
- Document schema migration guarantees and safe replay reconstruction policy.
- Add complete beta-to-1.0 upgrade guide.
- Add fresh-install and offline-replay integration tests using built artifacts.
- Explicitly support single-process recording only. Add a cross-process exclusive
  writer lock per cassette path; reject concurrent/inherited writers with a precise
  error instead of risking truncation or lost events. Define lock acquisition,
  release, process ownership, and stale-lock recovery behavior without silently
  deleting a live writer lock.
- Synchronize metadata, classifiers, README, examples, and CLI version output.
- Perform security, compatibility, determinism, documentation, and scope audits.
- Publish an explicit stable API table covering `__all__`, CLI/config contracts,
  compatibility aliases/deprecations, optional-dependency failures, and a decision
  on migration registry APIs.
- Audit every version surface including package metadata, README beta text, CLI
  `--version`, action metadata, built wheel metadata, and OTLP scope version
  (currently hardcoded to an older beta label).

### Explicit exclusions

- No new provider family solely for release breadth.
- No remote service, hosted viewer, package publication, or GitHub release.
- No silent best-effort behavior for unsupported multi-process recording.

### Dependencies

- Every preceding version complete and committed.
- Stable cassette schema and configuration semantics from prior releases.

### Compatibility and migration risks

- Removing or renaming beta APIs requires explicit compatibility aliases or upgrade
  instructions.
- Stable version promises make accidental exports and undocumented behavior costly.
- Multi-process claims must match actual file-locking and event-order semantics.

### Acceptance criteria

- Documented public imports, CLI commands, configuration, and schema guarantees match
  implementation and tests.
- Compatibility matrix reflects tested ranges.
- Fresh wheel install can initialize, diagnose, record, and replay offline.
- Unsupported multi-process use is either deterministic or explicitly rejected and
  documented.
- Sol final review finds no material correctness, security, compatibility, migration,
  replay, test, documentation, or scope issues.
- Git worktree is clean at the target commit.

### Required validation

- Full test, lint, format, and type-check suites.
- Wheel and sdist build plus metadata inspection.
- Fresh-environment install from wheel.
- CLI help/version/doctor/init smoke tests.
- LangChain installed-wheel record/replay smoke test with network/provider calls
  forbidden.
- Migration and older-cassette compatibility tests.
- Final Git status and version-string audit.
