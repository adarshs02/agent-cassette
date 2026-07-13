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
integrations:

```bash
uv sync --frozen --all-extras
```

Do not set `PYTHONPATH`. The project is installed into the uv environment.

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
- Keep optional integrations optional and the core dependency-free.
- Keep generated cassettes, credentials, build products, and virtual environments
  out of commits unless a reviewed test fixture intentionally requires them.
- Treat `.agents/` as vendored tooling; do not include it in project lint or type
  checking and do not edit it for product changes.
