# CLI reference

## record / replay

```bash
agent-cassette record run.jsonl -- python agent.py    # runs live, saves to run.jsonl
agent-cassette replay run.jsonl -- python agent.py     # offline, no keys, deterministic
```

On `record`, real calls run and get saved. On `replay`, the same script runs with zero
live calls. Your script needs no changes — supported clients are patched for the run.

> Automatic CLI execution runs your script in-process. Treat executed scripts as trusted code.

## init

```bash
agent-cassette init --detect
```

`init --detect` statically inspects your manifests (never imports or runs your code,
never touches dependencies) and writes three owned files: `.agent-cassette.toml`, a
cassette fixture dir, and an offline record/replay smoke test. It only creates missing
files and never overwrites yours. Add `--dry-run`, `--check`, or `--json` for previews
and machine-readable CI output.

Configure defaults in `.agent-cassette.toml` (drives the pytest fixture and replay CLI):

```toml
schema_version = 1
cassette_dir = "tests/cassettes"
match = "exact"        # exact | subset | normalized | fuzzy
strict = true
providers = ["openai"]
frameworks = ["langchain"]
test_frameworks = ["pytest"]
```

## fork

Time-travel replay, live continuation, and failure injection — see
[Forks & failure injection](forks.md).

## check / diff

Trajectory reports for CI — see [Testing](testing.md#ci-reports).

```bash
agent-cassette check run.jsonl \
  --require tool_call:search \
  --tool-called search --tool-not-called send_email \
  --max-cost 0.05 --report-json checks.json
```

`--tool-called NAME` / `--tool-not-called NAME` are repeatable name-only tool-boundary
checks. Checks run in a fixed order — every `--require`, then each `--tool-called` /
`--tool-not-called` in command-line order, then `--max-cost` / `--max-duration-ms`. An
empty tool name is a usage error (exit `2`). Supplying any check makes the command
explicit, so `no_errors()` is added only when you pass `--no-errors`. Structured-input
and count assertions are Python-only (`tool_called(..., with_input=..., times=...)`).
These checks inspect recorded/consumed Agent Cassette tool boundaries; they do not
observe uninstrumented side effects.

## Other commands

```bash
agent-cassette view run.jsonl --output run.html      # standalone, script-free HTML viewer
agent-cassette inspect run.jsonl                     # summarize a cassette
agent-cassette export-otlp run.jsonl trace.json      # OTLP/OpenInference JSON
agent-cassette import-otlp trace.json restored.jsonl
agent-cassette migrate old.jsonl --output upgraded.jsonl
agent-cassette recover interrupted.jsonl recovered.jsonl   # salvage an incomplete final write
agent-cassette doctor                                # environment + integration health
```

Cassettes are JSONL, one strict event per line (schema v1). Loading fails closed on
corruption; `recover` only salvages a torn final byte fragment into a new file. See the
[schema contract](cassette-schema.md), [CLI exit codes](cli-exit-codes.md), and
[migrations](beta-upgrade.md).
