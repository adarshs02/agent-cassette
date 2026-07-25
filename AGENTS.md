# Agent Cassette contributor guide

This repository is designed to be bootstrapped without guessing commands or adding
the source tree to `PYTHONPATH`.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## Bootstrap

Install the exact locked development environment, including all optional provider
and framework integrations:

```bash
uv sync --frozen --all-extras --dev
```

Do not set `PYTHONPATH`. The project is installed into the uv environment.

## Consumer project initialization

From an installed wheel, a coding agent can safely preview, apply, and verify
Agent Cassette scaffolding in a consumer repository:

```bash
agent-cassette init . --detect --dry-run --json
agent-cassette init . --detect --json
agent-cassette init . --detect --check --json
pytest tests/test_agent_cassette_smoke.py
```

These commands never install dependencies, modify dependency manifests, execute
consumer code, discover secrets, or write API keys. A repeated normal init is
idempotent only when every generated file matches exactly; init never updates a
different existing scaffold. Treat exit 1 from `--check` as "changes needed" and
exit 2 from any init mode as invalid or conflicting state. A successful dry run
returns 0 and writes nothing.

`.agent-cassette.toml` supplies the default cassette directory, match mode, and
strictness for the `cassette` pytest fixture. Per-test `cassette` marker arguments
override those defaults. The replay CLI uses configured match and strict defaults
when flags do not override them, but always requires an explicit cassette path.
Static detection recognizes `pytest` and `unittest` as test frameworks.
It stores them in `test_frameworks`, separate from integration `frameworks` such as
`langchain`. Read-only config loading and pytest/replay runtime defaults are portable
on supported Python platforms. Mutating initialization uses required
directory-relative, no-follow filesystem primitives and fails closed when the
platform lacks them.
Static manifest parse failures are warnings; do not execute project code to infer
the missing dependency information.

## Agent-native closed loop (machine flow)

For a fully non-interactive, machine-parseable flow, an agent can drive one closed loop.
Every command emits the same JSON envelope (`--json`) with `status`, `exit_code`, and
argv-vector `next_actions`:

```bash
agent-cassette setup .  --apply --json                    # create-only scaffold + manifest
agent-cassette status . --json                            # readiness, managed files, cassettes
agent-cassette agent-manifest . --json                    # static command/exit/safety surface
agent-cassette record  --name smoke -- python agent.py    # live; create-only golden cassette
agent-cassette replay  --name smoke -- python agent.py    # offline; structured pass/mismatch report
agent-cassette rerecord --name smoke -- python agent.py   # explicit golden update (atomic)
agent-cassette ci . --github --apply --json               # replay-only workflow scaffold
```

`setup`/`status`/`agent-manifest`/`ci` never run consumer code, read environment values, or
install dependencies; `setup` is dry-run unless `--apply`. Named `record` is **live** and
create-only (an existing golden is exit 2); `replay` is **offline** at supported boundaries and
writes its report to `.agent-cassette/reports/<command>-<NAME>.json` — the machine channel,
separate from the child's untouched stdout. On a replay mismatch, read the report's
`data.failure` (event index, kind, value-free changed paths) and either fix the code and retry or
run the explicit, approval-marked `rerecord`. Never relax matching or delete a cassette to make a
mismatch pass. `status.data.capture_coverage` distinguishes automatic capture (OpenAI, Anthropic,
OpenAI Agents) from providers/frameworks needing an explicit wrapper.

## "Set up agent cassette"

When a user asks a coding agent to "set up agent cassette" in their project, drive the
non-interactive closed loop above — it writes the config, scaffolds, and records a manifest
in one machine-parseable pass, with no TTY. Prefer this over the legacy `init` flow.

1. **Preview (dry run).** Run `agent-cassette setup . --dry-run --json`. Detection is
   static — it executes no project code, reads no environment values, and installs nothing.
   Read `config` (configured + `detected` providers/frameworks/test-frameworks) and the
   planned `files`.
2. **Choose overrides (optional).** Detection fills sensible defaults; override only what the
   user wants, via flags (repeatable where noted) rather than hand-editing config:
   - `--provider NAME` (repeatable) — `openai`, `anthropic`, `mistral`, `gemini`.
   - `--framework NAME` (repeatable) — `openai-agents`, `langchain`, `mcp`.
   - `--test-framework NAME` — `pytest`.
   - `--match {exact,subset,normalized,fuzzy}` (default `exact`), `--strict`/`--no-strict`.
   - `--cassette-dir DIR` (default `tests/cassettes`).
   - `--github-ci` to also own a replay-only CI workflow.
   An override that disagrees with an existing `.agent-cassette.toml` is a **conflict**
   (exit 2), never a silent overwrite.
3. **Apply.** Run `agent-cassette setup . --apply --json` (add the same override flags). This
   is create-only: it writes `.agent-cassette.toml`, the cassette-dir `.gitkeep`, the offline
   smoke test, and (with `--github-ci`) the replay workflow, and records every generated
   file's SHA-256 in `.agent-cassette/manifest.json`. A file whose bytes differ from the
   generated content is a conflict, never overwritten.
4. **Confirm state.** Run `agent-cassette status . --json`: read `readiness`
   (`setup_ready`/`record_ready`/`replay_ready`/`ci_ready`), `managed_files`, `cassettes`,
   and any `blockers`. `agent-cassette agent-manifest . --json` describes the full
   command/exit/safety surface for planning the next action.
5. **Verify.** Run `pytest tests/test_agent_cassette_smoke.py`, confirm offline replay
   passes, and report the files created.

Treat `--check` exit 1 as "changes needed" and exit 2 as invalid or conflicting state; follow
each command's `next_actions` argv vector.

### Legacy `init` (lower-level alternative)

The interactive `init` flow (`agent-cassette init . --detect [--dry-run|--check] --json`, see
"Consumer project initialization" above) remains supported for callers that want to write the
config themselves and scaffold without a manifest. It never overwrites an existing config or
runs consumer code. Prefer `setup` for new projects; reach for `init` only when you need the
lower-level, manifest-free path.

## Validation

Run the complete repository gate before a release checkpoint:

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests examples benchmarks
uv run --frozen ruff format --check src tests examples benchmarks
uv run --frozen mypy src tests
uv build --no-build-isolation
```

Run one focused test file with the same environment:

```bash
uv run --frozen pytest tests/test_record_replay.py
```

Record and replay pytest cassettes explicitly when live credentials are available:

```bash
uv run --frozen pytest --cassette-mode=record
uv run --frozen pytest --cassette-mode=replay
```

## Installed-artifact smoke test

Build both distributions, install only the wheel into a clean environment, and run
the CLI outside this checkout with `PYTHONPATH` removed:

```bash
uv build --no-build-isolation
SMOKE_DIR="$(mktemp -d)"
uv venv "$SMOKE_DIR/venv"
uv pip install --python "$SMOKE_DIR/venv/bin/python" dist/*.whl
(cd "$SMOKE_DIR" && env -u PYTHONPATH "$SMOKE_DIR/venv/bin/agent-cassette" --help)
(cd "$SMOKE_DIR" && env -u PYTHONPATH "$SMOKE_DIR/venv/bin/agent-cassette" doctor --json)
```

CI repeats installed-wheel smokes in isolated environments for core, `openai`,
`anthropic`, `agents`, `langchain`, and `all`. Tested ranges are OpenAI `>=1,<3`,
Anthropic `>=0.34,<1`, OpenAI Agents `>=0.1,<1`, and LangChain Core `>=0.3,<2`.
Minimum-boundary jobs use Python 3.10; current-boundary jobs use Python 3.13.
Replay smokes unset provider credentials and must not access the network.

## Cassette validation and recovery

Schema-v1 JSONL is strict: reject duplicate object keys, non-finite numbers,
unsupported values, invalid Event fields, cycles, and excessive nesting. Never use
`default=str`, stringify unknown keys, or print payload representations in errors.
Redaction runs before persistence validation and fails safely on cycles or depth.

Normal reads, replay, migration, and inspection stay fail-closed. Recovery is
explicit, source-to-different-output, and may discard only a malformed,
unterminated final byte fragment:

```bash
agent-cassette recover interrupted.jsonl recovered.jsonl
agent-cassette recover interrupted.jsonl recovered.jsonl --json
```

Never silently recover earlier corruption, newline-terminated corruption, or a
decoded but invalid Event. Use `agent-cassette migrate SOURCE --output OUTPUT`; the
destination must differ from the source, so the source stays an upgrade and rollback
point.

## Benchmark smoke

Generate a versioned large-cassette report with deterministic contents:

```bash
uv run --frozen python benchmarks/large_cassette.py \
  --events 1000 --output /tmp/agent-cassette-benchmark.jsonl
```

Validate event count, byte count, and SHA-256. Timing fields are diagnostic only;
never add a wall-clock CI threshold.

## Publishing

Releases publish to PyPI via `.github/workflows/publish.yml` using OIDC Trusted
Publishing (no stored tokens): PyPI on a published GitHub Release, TestPyPI on manual
dispatch. Process and one-time setup are in `docs/releasing.md`.

## Repository invariants

- Preserve deterministic offline replay and recursive secret redaction.
- Keep normal cassette loading fail-closed; recovery must remain explicit and
  source-preserving.
- Never dynamically import a type named by cassette data.
- Preserve cassette schema compatibility; add an explicit one-version migration
  when a schema change is unavoidable.
- Keep optional integrations optional. Core uses only the standard library on
  Python 3.11+ and the `tomli` compatibility reader on Python 3.10.
- Keep `agent_cassette.__all__` synchronized with `tests/test_public_api.py` and
  `docs/public-api.md`; optional provider imports must remain lazy.
- Keep generated cassettes, credentials, build products, and virtual environments
  out of commits unless a reviewed test fixture intentionally requires them.
- Treat `.agents/` as vendored tooling; do not include it in project lint or type
  checking and do not edit it for product changes.
