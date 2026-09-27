# Retrace

Autonomous data-incident agent. Given a vague on-call report, Retrace traces the executive
revenue KPI through DataHub lineage, profiles the warehouse, confirms a root cause with cited
evidence, and either repairs the SQL transform (verified by 19 pipeline invariants) or
escalates an upstream data problem. It records every incident back to DataHub.

The whole eval suite replays offline in CI through [Agent Cassette](../../README.md):
no API key and no DataHub.

## How it works

- **Evidence gates.** Every fact tool returns an `evidence_id`. A root cause must cite
  DataHub metadata and warehouse data, and the asset must appear in DataHub lineage.
- **Verified repairs.** Fixes apply to a scratch copy of the transforms; the pipeline is
  rebuilt and must pass every invariant. Raw-level checks can't be fixed by SQL, so upstream
  problems must be escalated.
- **Deterministic graders.** Scores come from the final state, the diff, and a real
  invariant run. No LLM judge.
- **Cassettes.** Live runs record Claude and DataHub MCP calls. Replay runs the same agent
  with the warehouse, repair, and check tools executing for real.
- **Found a library bug.** Building Retrace surfaced and fixed an Agent Cassette redaction bug: integer token-count fields (`input_tokens`, `max_tokens`, Gemini `*_token_count`) were being scrubbed as secrets.

## Scenarios

| scenario | injected problem | expected |
|---|---|---|
| unit_cents | new processor sends cents | scoped SQL fix |
| schema_rename | upstream renames `currency` | SQL fix |
| join_fanout | duplicate customer keys | dedupe fix |
| tz_shift | new processor sends local time | UTC fix |
| stale_feed | FX vendor feed stops | escalate |
| null_surge | 30% null amounts for one processor | escalate |
| control_healthy | none | no incident |
| control_distractor | FX one day stale, negligible | no incident |
| bad_repair_rejected | naive blanket `/100` fix | rejected by gates |
| datahub_timeout | DataHub down from the first call | never a false all-clear |
| rate_limit_midrun | Anthropic 429 mid-run | retries, same outcome |

## Run it

```bash
uv sync
uv run retrace eval --replay            # offline, from committed cassettes; agent scenarios report "skipped" until cassettes are recorded
```

Live (records cassettes; needs Docker, DataHub, and an Anthropic key). Pin the DataHub MCP
server so recordings stay reproducible, e.g. `export RETRACE_MCP_SERVER=mcp-server-datahub==<version>`;
Retrace checks the server's tool list at startup and refuses to run against a mismatched one:

```bash
uv sync --extra datahub
uvx --from acryl-datahub datahub docker quickstart
export ANTHROPIC_API_KEY=...  DATAHUB_GMS_URL=http://localhost:8080
uv run retrace eval --live --trials 3
uv run agent-cassette view evals/cassettes/unit_cents.jsonl --output unit_cents.html
```

## Results

Pending the first live recording (`uv run retrace eval --live`). Once recorded, the scorecard is written to `evals/results/scorecard.md` and CI replays every scenario from `evals/cassettes/`.

## Credits

Inspired by [Project Blackbox](https://github.com/alejandro-publius/blackbox-datahub)
(Apache-2.0), grand-prize winner of Build with DataHub: The Agent Hackathon (2026).
Adapted files are listed in [NOTICE](NOTICE).
