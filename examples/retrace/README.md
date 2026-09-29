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
uv run retrace eval --replay            # offline, from committed cassettes
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

Live run 2 against a real DataHub v1.7 quickstart with `claude-sonnet-5`, 3 trials per agent
scenario, after token-cost improvements (conversation caching, compacted results, turn cap,
early stop on outage) and the cross-platform determinism fix. Full details in
[`evals/results/scorecard.md`](evals/results/scorecard.md).

| scenario | passed |
|---|---|
| unit_cents | 3/3 |
| schema_rename | 3/3 |
| join_fanout | 3/3 |
| tz_shift | 2/3 |
| stale_feed | 3/3 |
| null_surge | 3/3 |

- Controls: 0 false positives in 6 runs (`control_healthy`, `control_distractor` × 3 trials).
- Extra scenarios (`bad_repair_rejected`, `datahub_timeout`, `rate_limit_midrun`): all passed.
- 23/24 agent trials overall. `tz_shift` trial 3 hit the turn cap; trials 1–2 passed.
- Wall time: 2543.3s live. Keyless replay from `evals/cassettes/` (the recorded trial 1 of each
  scenario): < 1 min, all 10/10 recorded outcomes match (macOS and Linux verified).
- Tokens: 692 uncached input, 401,378 cache-write, 4,314,329 cache-read, 172,721 output.
  Estimated cost ≈ $3.59 at Sonnet 5 list prices ($2/$10 per MTok, cache write 1.25×, cache read 0.1×),
  vs ≈ $14.38 for run 1 (6.0M uncached input / 237k output) — about **75% cheaper** with same accuracy.

CI runs `retrace eval --replay` keylessly, offline, from the committed cassettes. It replays
every scenario and checks that each outcome matches the recording. A replay that reproduces a
recorded failure is a pass; a replay whose outcome diverges from the recording is not.

## Credits

Inspired by [Project Blackbox](https://github.com/alejandro-publius/blackbox-datahub)
(Apache-2.0), grand-prize winner of Build with DataHub: The Agent Hackathon (2026).
Adapted files are listed in [NOTICE](NOTICE).
