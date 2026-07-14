# Agent Cassette Release Plan

Target: `1.0.0`

This roadmap supersedes the earlier prerelease numbering. Versions execute in the
order below. Historical commits are retained as implementation checkpoints; release
metadata must be synchronized to the new labels before each corresponding version is
declared complete.

## Durable release rules

- Preserve deterministic offline replay and recursive secret redaction.
- Never import or execute a type named by cassette data. Replay reconstruction uses
  only code-owned allowlists.
- Preserve schema compatibility and provide explicit migrations when a schema change
  is unavoidable.
- Preserve unrelated user changes. Do not push, publish, merge, or create remote
  releases.
- Use `uv.lock`. The contributor bootstrap is `uv sync --all-extras --dev`; release
  validation additionally verifies frozen lock use.
- Required gates are the full pytest suite, Ruff lint and format checks, mypy, wheel
  and source-distribution builds, clean installed-wheel smoke tests outside the
  checkout with `PYTHONPATH` removed, and Sol-level review.
- Commit each version as a coherent local Git checkpoint before starting the next.
- Generate Repomix snapshots on demand. Do not commit generated Repomix snapshots.

## Renumbering state

The following implementation checkpoints were created under the superseded working
labels and are preserved without rewriting Git history:

| New roadmap version | Implemented work | Historical checkpoint |
| --- | --- | --- |
| `0.12.0b1` | Agent-friendly setup and diagnostics | `2d345e7` (formerly `0.11.1b1`) |
| `0.13.0b1` | LangChain Runnable replay | `c1e4db8` (formerly `0.12.0b1`) |
| `0.13.0b2` | LangChain lifecycle tracing | `7680e74` (formerly `0.12.1b1`) |

The roadmap status below distinguishes implemented functionality from final release
label synchronization.

## 0.12.0b1 — Agent-friendly setup

Status: **complete via mapped historical checkpoint `2d345e7`**

Goal: make a fresh checkout immediately usable by a human or coding agent.

Included work:

- Dependency lockfile and canonical `uv` workflow.
- Development build dependencies.
- Root `AGENTS.md` with exact setup, test, lint, type-check, and build commands.
- Tests that work without manual `PYTHONPATH` changes.
- `agent-cassette doctor` and `agent-cassette doctor --json`.
- Dependency and provider detection with actionable diagnostics.
- Provider-neutral CLI descriptions.
- Clean-install CI, wheel validation, and source-distribution validation.
- Improved installation documentation.
- On-demand Repomix generation instead of committed snapshots.

Explicit exclusions:

- No LangChain execution integration.
- No consumer-project initializer.
- No automatic dependency installation by `doctor`.

Dependencies:

- Hatchling build backend.
- `uv` for environment and lock management.
- Standard-library package discovery for diagnostics.

Compatibility and migration risks:

- The lock must resolve across supported Python versions.
- Clean-install tests must import the installed artifact, not the checkout.
- Doctor output and exit codes must stay deterministic and provider-neutral.
- Removing committed Repomix output must not remove the on-demand generation command
  or documentation.

Acceptance criteria:

- `uv sync --all-extras --dev` prepares a fresh checkout without guessing.
- `uv run agent-cassette doctor` reports actionable failures.
- `uv run pytest` succeeds without `PYTHONPATH` changes.
- An agent can discover every required command from root documentation.
- Wheel and source-distribution artifacts build and install successfully.
- Generated Repomix snapshots are ignored or otherwise kept out of commits.

Required validation:

- Fresh-checkout bootstrap and frozen-lock check.
- Full tests, Ruff lint/format, and mypy.
- Wheel and source-distribution build.
- Installed-wheel CLI help and doctor human/JSON smoke tests.
- Repomix on-demand workflow and clean-Git snapshot check.

Historical validation: 117 tests, Ruff, mypy, lock, wheel/sdist, and installed-wheel
doctor smokes passed at checkpoint `2d345e7`.

## 0.13.0b1 — LangChain Runnable replay

Status: **complete via mapped historical checkpoint `c1e4db8`**

Goal: make LangChain chains deterministic replay boundaries.

Included work:

- Optional `langchain` extra and lazy `wrap_langchain` public API.
- Runnable-compatible proxy supporting `invoke`, `ainvoke`, `stream`, `astream`,
  `batch`, and `abatch`.
- LCEL composition and `@chain` compatibility.
- Classic `Chain` compatibility where practical without weakening Runnable behavior.
- Safe message, message-chunk, document, structured-output, and Pydantic-value
  serialization.
- Failure recording/replay, offline chain replay, batch concurrency handling, and
  streaming failure handling.
- Provider and chain-wrapper composition.
- A chain event type and schema migration only if the existing schema cannot safely
  represent the boundary.

Explicit exclusions:

- No lifecycle callback tracing; `0.13.0b2` owns internal spans.
- No arbitrary object reconstruction or cassette-directed dynamic imports.
- No global LangChain monkey-patching.

Dependencies:

- `0.12.0b1` setup and artifact validation.
- Supported `langchain-core` Runnable contracts.
- Existing Recorder/Replayer boundary semantics.

Compatibility and migration risks:

- LangChain values and Runnable behavior vary across supported versions.
- Batch results must preserve API ordering under concurrency.
- Streams must finalize consistently on exhaustion, close, failure, or cancellation.
- Schema migration must remain one-step and backward compatible if introduced.

Acceptance criteria:

- Chains replay without live model/provider calls.
- Reconstructed supported values remain LangChain-compatible.
- Sync, async, batch, and streaming paths are covered.
- Wrapped provider and chain boundaries compose without duplicate execution.
- Replay never imports a cassette-named type.

Required validation:

- Runnable, LCEL, `@chain`, practical classic Chain, serialization, failure, batch,
  and streaming tests.
- Minimum/current supported LangChain installed-wheel smokes.
- Full repository release gates.

Historical validation: 152 tests plus LangChain Core 0.3.0 and 1.4.9 installed-wheel
record/replay smokes passed at checkpoint `c1e4db8`.

## 0.13.0b2 — LangChain lifecycle tracing

Status: **complete via mapped historical checkpoint `7680e74`**

Goal: complete the LangChain feature set before the next minor version.

Included work:

- Public `AgentCassetteCallback`-compatible callback surface.
- Chain, model, retriever, parser, and tool lifecycle spans/events.
- Nested Runnable spans, run-ID correlation, and parent/child relationships.
- Sync, async, streaming, cancellation, and failure events.
- Provider-event deduplication.
- Boundary-replay and trace-only instrumentation modes.
- Recorder synchronization needed for concurrent callback delivery.

Explicit exclusions:

- No LangSmith transport.
- No process-global callback installation.
- No consumer-project initialization.

Dependencies:

- `0.13.0b1` Runnable replay and serialization contracts.
- LangChain callback and run hierarchy contracts.

Compatibility and migration risks:

- Callback ordering and parser classification vary across LangChain releases.
- Observational trace events must not be consumed as replay boundaries.
- Callback instrumentation must not alter user callback behavior or provider calls.

Acceptance criteria:

- Full internal trajectories are visible with correct parent/child relationships.
- Provider and lifecycle events are not duplicated.
- Callback instrumentation does not change chain behavior.
- Boundary replay ignores observational trace events deterministically.
- Sync, async, and streaming lifecycle suites pass.

Required validation:

- Every lifecycle category, nesting, correlation, streaming, failure, cancellation,
  deduplication, trace-only, and boundary-replay tests.
- Concurrent sync/async callback tests.
- Full repository release gates and minimum/current LangChain smoke tests.

Historical validation: 162 tests, Ruff, mypy, artifact builds, and LangChain Core
0.3.0/1.4.9 callback replay smokes passed at checkpoint `7680e74`.

## 0.14.0b1 — Agent-first initialization

Status: **complete**

Architecture decision: initialization uses a schema-v1 `.agent-cassette.toml`,
static manifest/source inspection, a versioned deterministic report, and fail-closed
directory-FD/no-follow filesystem operations with atomic create-without-replace.
Configuration drives cassette directory and replay matching/strictness defaults;
explicit CLI/pytest options override it. Existing compatible configuration is
accepted, while non-exact scaffold files are never overwritten.

Assigned routes: root Sol owns filesystem security, config/CLI semantics, and final
integration. Terra-equivalent `init_builder` owns implementation/tests;
Terra-equivalent `init_packaging` owns version, dependency, docs, lock, and CI.
Sol-equivalent review follows implementation, with at most three fix cycles.

Review attempts: **5/5 complete (cycles 4 and 5 explicitly authorized by the user)**.
Attempt 1 closed the original filesystem,
configuration, FIFO, warning, and dry-run findings. Attempt 2 closed portable
read-only config loading, control-character path rejection, frozen bootstrap docs,
and identity-held rollback; attempt 3 closes a post-publication descriptor-failure
bookkeeping window. Final review found two remaining helper-internal
`BaseException` windows immediately after `os.link` and `os.mkdir`, before rollback
records exist. Cycle 4 addresses only those two helper-internal rollback windows.
Its final review verified file cleanup but found that post-`mkdir` cleanup without a
captured identity can delete an unrelated replacement directory. Cycle 5 applied
the conservative rule that unknown-identity directories are never removed by name,
added the replacement-race test, and received final Sol-equivalent acceptance with
no material findings.

Validation evidence: 240 tests passed; Ruff lint and format passed across 59 files;
mypy passed across 58 files; frozen lock resolved 77 packages; wheel and sdist
built; an exact 0.14.0b1 wheel in a fresh Python 3.13 consumer outside the checkout
passed doctor, dry-run/check/apply/idempotence, custom cassette directory,
configuration-driven normalized/non-strict record/replay, and generated offline
smokes. Workflow YAML, wheel metadata, and diff checks passed.

Checkpoint commit: `758bcb4`

Goal: let Agent Cassette configure itself safely inside another Python repository.

Included work:

- `agent-cassette init`, `init --detect`, `init --check`, and `init --dry-run`.
- Schema-versioned `.agent-cassette.toml` project configuration.
- Static provider, framework, and test-framework detection without importing or
  executing consumer code.
- Cassette-directory scaffolding driven by configuration.
- Offline smoke-test generation and pytest configuration.
- Suggested matching defaults and configuration-driven CLI behavior.
- Idempotent configuration updates that preserve existing files.
- Clear, integration-aware next-step commands.

Explicit exclusions:

- No dependency installation or dependency-manifest modification.
- No API-key discovery/writes or consumer-code execution.
- No overwrite of unowned user files.
- No non-Python project detection.

Dependencies:

- Completed `0.12.0b1` setup/doctor behavior.
- Completed `0.13.0b1` and `0.13.0b2` framework surfaces.
- Safe TOML loading on every supported Python version.

Compatibility and migration risks:

- Initialization writes to consumer repositories; path traversal, symlink races,
  atomic publication, rollback, FIFOs/devices, and partial failure require strict
  security review.
- Existing configuration and files must remain preserved and idempotent.
- Detection errors must be reported rather than silently becoming empty detection.
- Configuration defaults must actually drive CLI and pytest behavior.
- Earlier uncommitted work used `agent-cassette.toml`; final implementation and
  migration behavior must align with the new `.agent-cassette.toml` name.

Acceptance criteria:

- Initialization is idempotent and existing files are preserved.
- `--dry-run` and `--check` never write and have deterministic exit codes.
- Detected providers, frameworks, and test frameworks are accurate and explain
  skipped/malformed manifests.
- Generated projects pass `doctor` and their offline smoke test.
- Users can immediately record and replay a smoke cassette.
- Configuration changes cassette paths, matching, strictness, CLI, and pytest
  defaults as documented.

Required validation:

- Filesystem race, symlink, FIFO/device, traversal, conflict, rollback, idempotency,
  dry-run/check, and detection tests.
- Configuration-driven CLI and pytest tests with explicit override coverage.
- Empty-project installed-wheel initialization and generated-project smoke tests.
- Full repository release gates and Sol security review.

## 0.15.0b1 — Compatibility and security

Status: **in progress — externally blocked at the minimum-SDK validation gate**

Architecture decision: cassette parsing remains fail-closed by default. Recovery is
an explicit source-to-output operation limited to discarding the bytes after the
last line feed when that unterminated fragment cannot be decoded as strict UTF-8
and JSON. A valid final event without a newline is retained; complete malformed
records, duplicate keys, non-finite numbers, and schema-invalid events are never
skipped. Blank lines remain accepted for beta compatibility. Serialization accepts
only bounded, acyclic JSON values with string mapping keys and never falls back to
arbitrary object stringification; values are fully serialized before any write.
CLI exit codes are `0` for success, `1` for deterministic negative outcomes, and
`2` for usage, configuration, corruption, or recovery errors. Compatibility claims
are tied to isolated built-wheel CI jobs, not dependency declarations alone. Disk
shape remains schema v1, so no schema bump is required.

Assigned routes: root Sol owns corruption/recovery semantics, public API and CLI
contracts, security decisions, integration, and final review. Terra-equivalent
workers own bounded core-hardening, CLI/test, and packaging/documentation changes
without overlapping file ownership. Luna-equivalent work is limited to mechanical
version/document synchronization.

Review attempts: **3/3 complete; final Sol-equivalent review accepted**. Attempt 1 found
incomplete-token recovery classification gaps and malformed-OTLP exception leaks;
both bounded fixes and adversarial tests pass. Attempt 2 accepted recovery but found
one remaining extreme end-timestamp overflow in OTLP duration conversion; its
bounded fix and regressions pass. Attempt 3 accepted correctness, security,
compatibility, schema, replay, and documentation with no material findings.

Validation evidence: 388 tests passed; Ruff lint and formatting passed across 67
files; mypy passed across 65 files; the frozen lock resolves 77 packages; the
deterministic 1,000-event benchmark produced byte-identical cassettes; exact
0.15.0b1 wheel and sdist builds passed; an isolated installed core wheel passed
doctor, init, redaction, torn-tail recovery, malformed-OTLP exit handling, and
offline replay; isolated current OpenAI, Anthropic, OpenAI Agents, LangChain, and
all-extras smokes passed. Minimum/current CI jobs are configured for every extra.

External blocker: exact minimum OpenAI 1.0.0, Anthropic 0.34.0, OpenAI Agents
0.1.0, and LangChain Core 0.3.0 package downloads are unavailable locally. The
required network escalation was explicitly rejected because the platform account
hit its usage limit, and the repository rules prohibit pushing merely to trigger
CI. Those minimum-boundary jobs therefore remain unexecuted. This version is not
marked complete, not committed, and the release candidate has not started.

Checkpoint commit: pending.

Goal: freeze feature expansion and prepare a safe stable contract.

Included work:

- Python, OpenAI, Anthropic, LangChain, and OpenAI Agents compatibility matrices.
- Optional-extra installation tests.
- Corrupted-cassette handling and partial-write recovery.
- Large-cassette benchmarks.
- Serialization security review, redaction audit, and replay safety audit.
- Public API inventory.
- CLI exit-code standardization and deprecation warnings.
- Beta upgrade documentation and schema compatibility documentation.

Explicit exclusions:

- No new providers, frameworks, or major feature families.
- No breaking public API removal without a documented deprecation path.
- No publication or remote release operations.

Dependencies:

- Every beta feature version through `0.14.0b1` complete.
- Representative supported SDK versions available to CI.

Compatibility and migration risks:

- Matrix claims must reflect tested versions, not broad dependency specifiers alone.
- Recovery must never reinterpret corrupted data as trusted executable/type metadata.
- Standardized exit codes may require beta compatibility aliases or warnings.

Acceptance criteria:

- Supported dependency ranges are explicit and tested.
- Public APIs intended for 1.0 are identified.
- No known unsafe replay paths remain.
- Fresh-install and offline-replay tests pass across the supported matrix.
- Corruption and partial writes fail safely or recover deterministically.

Required validation:

- Full compatibility matrix and optional-extra installation jobs.
- Corruption, recovery, security, redaction, and replay-adversary suites.
- Reproducible large-cassette benchmark report.
- Full repository release gates and Sol security review.

## 1.0.0rc1 — Contract freeze

Status: **planned**

Goal: test the final stable API, CLI, adapter, schema, and migration contract.

Included work:

- Freeze Python API and CLI structure.
- Freeze adapter protocol v1.
- Freeze cassette schema policy and migration guarantees.
- Run real-project compatibility tests.
- Publish final migration documentation in the repository.
- Accept only release-blocking fixes and issue-report feedback.

Explicit exclusions:

- No new providers, frameworks, or major features.
- No opportunistic refactors unrelated to release blockers.
- No remote publication from this orchestration task.

Dependencies:

- `0.15.0b1` complete with no unresolved material security findings.
- Complete stable public API inventory and compatibility matrices.

Compatibility and migration risks:

- Any RC contract change must be treated as release-blocking and fully documented.
- Real-project fixtures must not require secrets for offline replay validation.

Acceptance criteria:

- Frozen contracts are documented and enforced by compatibility tests.
- Real-project record fixtures replay offline without live calls.
- Only release-blocking defects remain eligible for changes.
- Final migration documentation is complete.

Required validation:

- Full repository and compatibility-matrix gates.
- Public API/CLI/schema snapshots.
- Real-project fresh-install, upgrade, and offline-replay suites.
- Sol contract-freeze review.

## 1.0.0 — Stable release

Status: **planned**

Goal: ship Agent Cassette as the open, provider-neutral record and replay system for
testing AI agents.

Stable guarantees:

- Stable public API and CLI contract.
- Documented cassette compatibility and safe migrations.
- Offline replay and secret-redaction guarantees.
- Provider and framework conformance tests.
- Tested installation and upgrade paths.
- Maintainer-controlled architecture and releases.

Explicit exclusions:

- No new feature work after RC beyond release-blocking fixes.
- No push, publication, GitHub release, or external-system modification in this
  local implementation task.

Dependencies:

- `1.0.0rc1` complete and all release blockers resolved.
- Every prior version complete with coherent Git checkpoints.

Compatibility and migration risks:

- Stable promises must exactly match implementation and tested compatibility ranges.
- Version, metadata, documentation, schema, and built artifacts must agree.

Acceptance criteria:

- Every stable guarantee is backed by tests and documentation.
- Full tests, lint, formatting, typing, builds, installed-wheel smokes, compatibility
  matrices, migrations, and offline replay pass.
- Sol final review finds no material correctness, security, compatibility, replay,
  migration, documentation, or scope issue.
- The working tree is clean and every version has a local Git checkpoint.

Required validation:

- Full repository, compatibility, conformance, migration, corruption, redaction,
  fresh-install, upgrade, and offline-replay suites.
- Wheel/sdist metadata and version-string audit.
- CLI help/version/doctor/init smoke tests from the installed wheel.
- Final public API/CLI/schema snapshot review and clean Git status.
