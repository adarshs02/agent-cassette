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
uv run --frozen ruff check src tests examples
uv run --frozen ruff format --check src tests examples
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

## Repository invariants

- Preserve deterministic offline replay and recursive secret redaction.
- Never dynamically import a type named by cassette data.
- Preserve cassette schema compatibility; add an explicit one-version migration
  when a schema change is unavoidable.
- Keep optional integrations optional. Core uses only the standard library on
  Python 3.11+ and the `tomli` compatibility reader on Python 3.10.
- Keep generated cassettes, credentials, build products, and virtual environments
  out of commits unless a reviewed test fixture intentionally requires them.
- Treat `.agents/` as vendored tooling; do not include it in project lint or type
  checking and do not edit it for product changes.
