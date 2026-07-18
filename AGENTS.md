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

## Interactive setup ("set up agent cassette")

When a user asks a coding agent to "set up agent cassette" in their project, follow
this flow. It needs no new tooling — use your own question tool plus the `init`
commands above.

1. **Detect.** Run `agent-cassette init . --detect --dry-run --json`. Read
   `detected.providers`, `detected.frameworks`, `detected.test_frameworks`, and the
   planned files. This executes no project code and touches no dependencies.
2. **Confirm with the user.** Ask one short multiple-choice round, prefilled from
   detection:
   - **Providers** to capture — `openai`, `anthropic`.
   - **Frameworks** — `openai-agents`, `langchain`, `mcp`.
   - **Test framework** — `pytest` (or `unittest`).
   - **Match strictness** — `exact` (default), `subset`, `normalized`, `fuzzy`.
   - **Cassette directory** — default `tests/cassettes`.
3. **Write config.** Create `.agent-cassette.toml` from the answers before scaffolding
   (`init` never overwrites an existing config; if one exists, edit it instead):
   ```toml
   schema_version = 1
   cassette_dir = "tests/cassettes"
   match = "exact"
   strict = true
   providers = ["openai"]
   frameworks = ["langchain"]
   test_frameworks = ["pytest"]
   ```
4. **Scaffold.** Run `agent-cassette init . --json`. The existing config is left
   untouched; only the missing cassette directory and offline smoke test are created.
5. **Verify.** Run `pytest tests/test_agent_cassette_smoke.py`, confirm offline replay
   passes, and report the files created.

Treat `--check` exit 1 as "changes needed" and exit 2 from any init mode as invalid or
conflicting state.

## On-demand repository snapshot

Generate a single-file Repomix snapshot only when an agent needs one:

```bash
npx repomix --output repomix-output.xml
```

`repomix-output.xml` is ignored. Never commit the generated snapshot.

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
decoded but invalid Event. Prefer `agent-cassette migrate SOURCE --output OUTPUT`;
in-place migration is deprecated and warns.

## Benchmark smoke

Generate a versioned large-cassette report with deterministic contents:

```bash
uv run --frozen python benchmarks/large_cassette.py \
  --events 1000 --output /tmp/agent-cassette-benchmark.jsonl
```

Validate event count, byte count, and SHA-256. Timing fields are diagnostic only;
never add a wall-clock CI threshold.

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
