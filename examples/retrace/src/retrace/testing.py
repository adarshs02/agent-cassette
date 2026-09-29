"""Keyless test doubles: a scripted Anthropic-shaped model and canned scripts."""

from __future__ import annotations

from typing import Any

from anthropic.types import Message

from retrace.datahub.catalog import dataset_urn
from retrace.faults import get_fault

Step = list[tuple[str, dict[str, Any]]]


class ScriptedModel:
    """Returns pre-scripted tool calls in order, ignoring the conversation."""

    def __init__(self, script: list[Step]) -> None:
        self._script = script
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        index = len(self.requests)
        step = self._script[index - 1] if index <= len(self._script) else []
        content: list[dict[str, Any]] = [
            {"type": "tool_use", "id": f"toolu_{index:03d}_{j}", "name": name, "input": args}
            for j, (name, args) in enumerate(step)
        ] or [{"type": "text", "text": "Thinking."}]
        return Message.model_validate(
            {
                "id": f"msg_{index:03d}",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": content,
                "stop_reason": "tool_use" if step else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }
        )


def unit_cents_script() -> list[Step]:
    fix = get_fault("unit_cents").reference_patch
    assert fix is not None
    return [
        [("datahub_search", {"query": "executive revenue"})],
        [
            (
                "datahub_lineage",
                {"urn": dataset_urn("marts.exec_metric"), "direction": "upstream", "max_hops": 3},
            )
        ],
        [("compare_to_baseline", {"last_days": 14})],
        [
            (
                "profile_column",
                {
                    "table": "raw.raw_orders",
                    "column": "amount",
                    "group_by": "payment_processor",
                    "since": "2026-08-07",
                },
            )
        ],
        [
            (
                "confirm_root_cause",
                {
                    "asset": "raw.raw_orders",
                    "field": "amount",
                    "summary": "cloudpay_v2 reports integer cents",
                    "evidence_ids": ["ev_002", "ev_004"],
                },
            )
        ],
        [("read_transform", {"name": "stg_orders"})],
        [("propose_repair", {"file": "stg_orders.sql", "new_sql": fix["stg_orders"]})],
        [("write_back", {"summary": "Scoped cents-to-dollars fix for cloudpay_v2."})],
        [("finish", {"report": "Root cause: cloudpay_v2 amounts in cents. Fixed in staging."})],
    ]
