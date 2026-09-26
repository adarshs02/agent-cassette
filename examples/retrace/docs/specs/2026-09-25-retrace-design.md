# Retrace — Design Spec

- **Date:** 2026-09-25
- **Status:** Draft, pending review
- **Location:** `examples/retrace/` inside the agent-cassette repository

## 1. Purpose

Retrace is an autonomous data-incident agent. Given a vague on-call report
("exec revenue looks wrong today"), it investigates a data pipeline using
DataHub metadata and warehouse queries, confirms a root cause with cited
evidence, and either repairs the offending SQL transform (verified by an
invariant suite) or escalates an upstream data problem. It writes the incident
back to DataHub.

Retrace doubles as the flagship consumer of Agent Cassette: every live eval run
is recorded as a cassette, and CI replays the full eval suite without an
Anthropic API key or a running DataHub.

### Success criteria

1. Live eval: the agent reaches the correct terminal stage and root cause on the
   6 fault scenarios, with zero false positives on the 2 controls. Results are
   reported as measured, per scenario, not asserted in advance.
2. Replay eval runs in CI with no secrets and no DataHub, using committed
   cassettes, and grades against real DuckDB and invariant results.
3. A change to Agent Cassette's `src/` that breaks Retrace replay fails CI.
4. README headline numbers (accuracy, false positives, replay vs live wall time,
   cost per live run) come from a generated scorecard.

### Non-goals

- No web UI. Runs are inspected through the CLI, markdown reports, and
  `agent-cassette view`.
- No git pushes, PR creation, or edits outside a scratch directory.
- No local metadata-catalog substitute for DataHub; DataHub is required for live runs.
- No randomized fault generator in v1.
- No changes to `~/myProjects/oncall-agent`.
- No contributions to the upstream Blackbox project.

## 2. Provenance and licensing

Retrace is inspired by Project Blackbox
(https://github.com/alejandro-publius/blackbox-datahub, Apache-2.0), grand-prize
winner of "Build with DataHub: The Agent Hackathon" (2026).

- Files adapted from Blackbox (data fixture generator, SQL transforms, invariant
  suite patterns) carry a header noting the origin and that they were modified.
- `examples/retrace/NOTICE` credits Blackbox and includes the Apache-2.0 notice;
  `examples/retrace/LICENSE-APACHE` contains the Apache-2.0 text.
- Written new for Retrace: agent loop, stage machine, evidence gates, tool layer,
  DataHub ingest, fault library, graders, eval runner, cassette integration, CLI.

## 3. Architecture

```
examples/retrace/
  pyproject.toml            # agent-cassette[anthropic] (path dep), anthropic, acryl-datahub, mcp, duckdb, sqlglot, pytest
  NOTICE, LICENSE-APACHE
  README.md
  pipeline/
    generate.py             # seeded synthetic retail data; --fault <name> | --healthy
    faults/                 # one module per fault: inject(), ground_truth(), repair_rules()
    transforms/*.sql        # stg_orders, stg_customers, stg_fx_rates, fct_revenue, exec_metric
    checks/                 # invariant suite (pytest), ~30 checks
    baselines/              # committed healthy-run baselines
    build.py                # CSV -> DuckDB -> transforms -> metric snapshot
  src/retrace/
    agent/loop.py           # Anthropic tool loop, budgets, nudges
    agent/stages.py         # stage machine and transition rules
    agent/evidence.py       # evidence store and citation gates
    agent/prompts.py
    tools/warehouse.py      # read-only SQL, profile_column, metric history, baseline compare
    tools/datahub.py        # DataHub via official MCP server (uvx mcp-server-datahub)
    tools/repair.py         # patch scratch transforms, rebuild, run checks
    datahub/ingest.py       # push schemas, column docs, ownership, lineage
    datahub/writeback.py    # incident note + tags on affected asset
    cassette.py             # record/replay wiring (patch_anthropic, wrap_mcp)
    cli.py                  # retrace run | eval | ingest
  evals/
    scenarios.py
    graders.py
    runner.py
    cassettes/              # committed, one per scenario
    results/                # scorecard.md + run JSON
  tests/
```

### Unit boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `pipeline/*` | Deterministic data, transforms, invariants | DuckDB only |
| `faults/*` | Inject one fault; define ground truth and valid-repair rules | `pipeline/generate.py` |
| `tools/warehouse` | Read-only fact tools | DuckDB (read-only connection) |
| `tools/datahub` | Metadata fact tools over MCP | DataHub MCP server |
| `tools/repair` | Only writer; scratch-dir patch + rebuild + checks | `pipeline/build.py`, `pipeline/checks` |
| `agent/evidence` | Store every fact result with an id; validate citations | none |
| `agent/stages` | Legal transitions and terminal states | `agent/evidence` |
| `agent/loop` | Drive the model, dispatch tools, enforce budgets | Anthropic SDK, tools, stages |
| `evals/graders` | Deterministic grading from final state, diff, check results | `faults/*` ground truth |

The agent loop never touches DuckDB or DataHub directly; it only dispatches tools.

## 4. Data fixture and fault library

The fixture models 90 days of retail orders (~400/day), customers, and daily FX
rates, flowing through staging into a daily revenue fact table and a single-row
executive KPI. All data is seeded (fixed RNG seed) and anchored to a fixed day;
no wall clock is consulted. The same inputs always produce byte-identical CSVs.

Lineage:

```
raw_orders    -> stg_orders    --+
                                 +--> fct_revenue --> exec_metric
raw_fx_rates  -> stg_fx_rates  --+
raw_customers -> stg_customers --+   (joined into fct_revenue for segment revenue)
```

`raw_customers` is joined into `fct_revenue` so that `join_fanout` affects the KPI.

### Faults

Each fault module exposes:

- `inject(frames) -> frames` — mutates generated data (or the upstream schema).
- `ground_truth() -> GroundTruth` — expected terminal stage, root-cause asset and
  field, and expected fix kind (`sql_repair` or `escalate`).
- `repair_rules(diff) -> list[RuleResult]` — checks that a SQL repair is targeted.

| Fault | Injection | Root cause | Expected outcome |
|---|---|---|---|
| `unit_cents` | One processor sends integer cents after a cutover date | `raw_orders.amount` | `sql_repair`: conditional divide scoped to that processor and cutover |
| `schema_rename` | Upstream renames `currency` to `currency_code` | `raw_orders.currency` | `sql_repair`: update column reference in `stg_orders` |
| `join_fanout` | Duplicate keys in customers dimension | `raw_customers.customer_id` | `sql_repair`: dedupe before join |
| `stale_feed` | FX feed stops at day N-4 and non-USD share rises to 60% | `raw_fx_rates` (freshness) | `escalate`: no SQL patch |
| `tz_shift` | Order timestamps arrive as local time without offset | `raw_orders.order_ts` | `sql_repair`: normalize to UTC in staging |
| `null_surge` | 30% of `amount` null for one processor | `raw_orders.amount` | `escalate`: no imputation |
| `control_healthy` | None | none | `NO_INCIDENT`, no patch |
| `control_distractor` | Stale FX with negligible (<2%) effect | none | `NO_INCIDENT`, no patch |

Ground truth lives only in fault modules. It never appears in prompts, DataHub
metadata, or anything the agent can read. The agent receives only the vague
incident report.

## 5. Data flow (live run)

1. `generate.py --fault <name>` writes CSVs.
2. `build.py` loads DuckDB, runs transforms, writes the metric snapshot.
3. `retrace ingest` pushes schemas, column docs (data contract), ownership, and
   lineage into DataHub. Idempotent.
4. The agent receives the incident report and investigates.
5. On a confirmed SQL root cause, `propose_repair` patches a scratch copy of
   `transforms/`, rebuilds, and runs the invariant suite.
6. On success, the agent writes back to DataHub and calls `finish`. Retrace emits
   `incident.json` and `report.md`.

## 6. Agent loop and gates

### Stages

```
INVESTIGATING -> ROOT_CAUSE_CONFIRMED -> REPAIRING -> VERIFIED -> WRITTEN_BACK
INVESTIGATING -> NO_INCIDENT
ROOT_CAUSE_CONFIRMED -> ESCALATED
any -> FAILED
```

Terminal: `WRITTEN_BACK`, `NO_INCIDENT`, `ESCALATED`, `FAILED`.

### Fact tools

Every fact tool stores its raw result in the evidence store and returns an
`evidence_id` to the model.

- Warehouse: `run_sql` (read-only, 5 s timeout, result truncated to ~8k chars
  with row count), `profile_column`, `get_metric_history`, `compare_to_baseline`,
  `list_transforms`, `read_transform`.
- DataHub: `datahub_search`, `datahub_get_dataset`, `datahub_lineage`.

### Workflow tools and gates

| Tool | Gate |
|---|---|
| `confirm_root_cause(asset, field, summary, evidence_ids)` | All ids exist; at least one DataHub-sourced and one data-sourced item; asset is upstream of the KPI in DataHub lineage |
| `declare_no_incident(evidence_ids)` | Cites a `compare_to_baseline` result showing no significant divergence |
| `escalate_upstream(asset, reason, evidence_ids)` | Same evidence rules as `confirm_root_cause`; forbids later repair |
| `propose_repair(file, patch)` | Stage is `ROOT_CAUSE_CONFIRMED` or `REPAIRING`; file reads the root-cause asset; runs invariant suite; failures returned to model; max 3 attempts |
| `finish(report)` | Only from a terminal-eligible stage |

Gate rejections are returned to the model as tool errors with the reason, so it
can continue investigating.

### Budgets

- 40 model turns, 3 repair attempts.
- Identical repeated tool calls trigger a nudge message; after 2 nudges the next
  repeat ends the run as `FAILED`.
- Budget exhaustion yields `FAILED` with a reason.

### Model

Anthropic SDK, model configurable, default `claude-sonnet-5`
(`--model claude-opus-5-5` for harder runs). Prompt caching on the system prompt.
SDK retries with backoff for transient errors.

## 7. Error handling

- **DataHub unreachable:** tool returns a structured error. The citation gate
  still requires DataHub evidence, so the agent cannot confirm a root cause and
  must escalate or the run ends `FAILED`. It must never reach `NO_INCIDENT`
  without a baseline comparison.
- **Warehouse:** read-only connection; timeouts and truncation as above.
- **Repair isolation:** patches apply to a temporary copy of `transforms/`;
  originals are never modified. The report includes the diff.
- **Transcript:** the cassette is the transcript; `agent-cassette view` renders it.

## 8. Agent Cassette integration

- Model calls: `patch_anthropic` records/replays `messages.create`.
- DataHub: `wrap_mcp` around the MCP client session.
- Warehouse, repair, and invariant tools are **not** replayed; they run live in
  both modes because they are local and deterministic. Grading therefore uses
  real DuckDB results and real diffs.
- Any change that alters tool output makes replay mismatch at the exact
  divergence point. Resolution is a code fix or an explicit live re-record.

## 9. Evals

### Modes

- `retrace eval --live [--trials N] [--scenarios ...]` — requires
  `ANTHROPIC_API_KEY` and DataHub. Grades each scenario and records one cassette
  per scenario to `evals/cassettes/`.
- `retrace eval --replay` — no key, no DataHub. Replays cassettes; grades with
  real warehouse and invariant runs. Exit code non-zero on any failure.

### Graders (deterministic, no LLM judge)

| Check | Source |
|---|---|
| Terminal stage matches ground truth | final state |
| Root-cause asset and field match | final state |
| Repair is targeted | diff vs `repair_rules` |
| Invariants 100% green and KPI within baseline band after repair | real invariant run |
| Controls: `NO_INCIDENT` and empty diff | final state + diff |
| Efficiency: turns, tool calls, tokens, cost | cassette |

### Extra scenarios

- `bad_repair_rejected` — a naive blanket `/100` patch is applied through
  `tools/repair` without the LLM; it must fail historical-immutability checks.
  Always runs; needs nothing.
- `datahub_timeout` — DataHub call #3 raises (Agent Cassette injection). Must end
  `ESCALATED` or `FAILED`, never `NO_INCIDENT`.
- `rate_limit_midrun` — one Anthropic call returns 429. Must retry and reach the
  same terminal stage as the uninjected run.

Robustness cassettes are recorded live with injection active, so replay
reproduces them deterministically.

### Output

`evals/results/scorecard.md`: per-scenario pass/fail, totals, mean turns, tool
calls, cost, and replay vs live wall time. `evals/results/run_<n>.json` holds full
results.

## 10. Testing and CI

- **Unit tests:** evidence gates (bad or missing citations rejected, stage
  transitions), each fault's injection and ground truth, repair isolation to the
  scratch dir, SQL tool read-only enforcement, DataHub ingest payload shape.
- **Replay tests:** pytest parametrized over committed cassettes, running
  `retrace eval --replay`.
- **Workflow** `.github/workflows/retrace.yml` in agent-cassette: triggers on
  changes to `examples/retrace/**` or `src/**`; no secrets, no DataHub; runs ruff,
  unit tests, and the replay eval.

## 11. Open questions

None blocking. Pipeline scale (~400 orders/day, 90 days) and invariant count
(~30) may be tuned during implementation if replay runtime in CI exceeds 2 minutes.
