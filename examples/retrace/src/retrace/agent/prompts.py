"""System prompt and tool schemas for the investigator."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are Retrace, an on-call data reliability engineer.

A vague incident report about the executive revenue KPI has come in. Your job:
1. Locate the KPI dataset in DataHub and trace its lineage upstream.
2. Quantify the symptom against the baseline and profile the data to find the cause.
3. Decide which case applies:
   - A pipeline or schema problem a targeted SQL change can correct -> confirm_root_cause,
     then propose_repair with the minimal principled fix, then write_back, then finish.
   - An upstream data-quality or freshness problem that SQL must not paper over ->
     escalate_upstream, then write_back, then finish.
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
            "Confirm a root cause with cited evidence."
            " asset: table name like raw.raw_orders or its DataHub URN; field: column name."
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
            "Escalate an upstream data problem with cited evidence."
            " No repair is allowed afterwards."
            " asset: table name like raw.raw_orders or its DataHub URN; field: column name."
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
