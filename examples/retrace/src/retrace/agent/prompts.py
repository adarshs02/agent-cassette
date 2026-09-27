"""System prompt and tool schemas for the investigator."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are Retrace, an on-call data reliability engineer.

A vague incident report about the executive revenue KPI has come in. Your job:
1. Locate the KPI dataset in DataHub and trace its lineage upstream.
2. Quantify the symptom against the baseline and profile the data to find the cause.
3. Decide which case applies:
   - The data is recoverable exactly: a pipeline, schema, join, unit, encoding or format
     problem where the true values can be reconstructed deterministically -> confirm_root_cause,
     then propose_repair with the minimal, targeted fix, then write_back (noting any upstream
     contract violation so the source team can fix it too), then finish. Repair this even if
     the root cause is an upstream contract violation.
   - The data is lost, missing or stale upstream, so no SQL can recover the true values
     (freshness gaps, missing values, dropped records) -> escalate_upstream, then write_back,
     then finish. Never impute or forward-fill to hide it.
   - No real incident -> declare_no_incident citing a baseline comparison, then finish.

Rules:
- Every fact tool returns an evidence_id. Workflow tools require you to cite them.
- A root cause or escalation must cite at least one DataHub item and one warehouse item,
  and the asset must appear in DataHub lineage evidence you collected.
- Repairs are verified by the full invariant suite. Checks on raw tables cannot be fixed
  by SQL; if they fail, the problem is upstream.
- Never blanket-rewrite history. Keep fixes scoped to the affected rows.
- End only by calling finish.
"""

_STR = {"type": "string"}


def _tool(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "datahub_search",
        "Search DataHub for datasets by name or meaning.",
        {"query": _STR},
        ["query"],
    ),
    _tool(
        "datahub_get_dataset",
        "Fetch a dataset's description, schema field docs (the data contract), and ownership.",
        {"urn": _STR},
        ["urn"],
    ),
    _tool(
        "datahub_lineage",
        "Traverse DataHub lineage from a dataset.",
        {
            "urn": _STR,
            "direction": {"type": "string", "enum": ["upstream", "downstream"]},
            "max_hops": {"type": "integer", "default": 3},
        },
        ["urn", "direction"],
    ),
    _tool(
        "get_metric_history",
        "KPI row plus recent daily revenue.",
        {"days": {"type": "integer", "default": 30}},
        [],
    ),
    _tool(
        "compare_to_baseline",
        "Daily revenue and order-count ratios versus the last known-good baseline.",
        {"last_days": {"type": "integer", "default": 14}},
        [],
    ),
    _tool(
        "profile_column",
        "Profile one column, optionally grouped and filtered by date.",
        {
            "table": _STR,
            "column": _STR,
            "group_by": _STR,
            "since": {"type": "string", "description": "YYYY-MM-DD"},
        },
        ["table", "column"],
    ),
    _tool(
        "run_sql",
        "Run one read-only SELECT against DuckDB (schemas raw, staging, marts).",
        {"query": _STR},
        ["query"],
    ),
    _tool("list_transforms", "List the SQL transform files.", {}, []),
    _tool("read_transform", "Read one SQL transform.", {"name": _STR}, ["name"]),
    _tool("run_checks", "Run the pipeline invariant suite on the current warehouse.", {}, []),
    _tool(
        "confirm_root_cause",
        (
            "Confirm a root cause with cited evidence; use this (then propose_repair) whenever"
            " the true values can be reconstructed exactly."
            " Can be called again to revise the root cause before a repair passes."
            " asset: schema-qualified table name (schema.table) or its DataHub URN;"
            " field: column name."
        ),
        {
            "asset": _STR,
            "field": _STR,
            "summary": _STR,
            "evidence_ids": {"type": "array", "items": _STR},
        },
        ["asset", "summary", "evidence_ids"],
    ),
    _tool(
        "escalate_upstream",
        (
            "Escalate upstream data that is lost, missing or stale and cannot be recovered"
            " exactly. No repair is allowed afterwards."
            " asset: schema-qualified table name (schema.table) or its DataHub URN;"
            " field: column name."
        ),
        {
            "asset": _STR,
            "field": _STR,
            "reason": _STR,
            "evidence_ids": {"type": "array", "items": _STR},
        },
        ["asset", "reason", "evidence_ids"],
    ),
    _tool(
        "declare_no_incident",
        "Declare no incident, citing a baseline comparison.",
        {"summary": _STR, "evidence_ids": {"type": "array", "items": _STR}},
        ["summary", "evidence_ids"],
    ),
    _tool(
        "propose_repair",
        "Replace one transform's full SQL; the pipeline is rebuilt in a "
        "scratch copy and every invariant is checked.",
        {"file": _STR, "new_sql": _STR},
        ["file", "new_sql"],
    ),
    _tool(
        "write_back",
        "Record the incident in DataHub on the affected dataset.",
        {"summary": _STR},
        ["summary"],
    ),
    _tool("finish", "End the run with a final report.", {"report": _STR}, ["report"]),
]


def initial_prompt(report: str) -> str:
    return (
        f'INCIDENT REPORT from on-call: "{report}"\n\n'
        "Investigate. Start by locating the executive revenue KPI in DataHub."
    )
