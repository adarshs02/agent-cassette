"""Faults that must be escalated upstream, plus healthy controls."""

from __future__ import annotations

import random
from datetime import timedelta

from retrace.faults import Fault, GroundTruth, register
from retrace.pipeline.generate import ANCHOR_DAY, SEED, Frames


def _drop_vendor_rates_after(frames: Frames, last_day: str) -> None:
    frames.fx_rates[:] = [
        r for r in frames.fx_rates if r["currency"] == "USD" or r["rate_day"] <= last_day
    ]


def _inject_stale_feed(frames: Frames) -> None:
    _drop_vendor_rates_after(frames, (ANCHOR_DAY - timedelta(days=4)).isoformat())


register(
    Fault(
        name="stale_feed",
        report="Exec revenue dipped over the last two days. Is the data OK?",
        inject=_inject_stale_feed,
        ground_truth=GroundTruth(
            "escalate", "raw.raw_fx_rates", "rate_day", alt_fields=("usd_rate",)
        ),
        must_fail=("fx_feed_fresh", "revenue_within_baseline_band"),
    )
)


def _inject_null_surge(frames: Frames) -> None:
    rng = random.Random(SEED + 6)
    for row in frames.orders:
        if row["payment_processor"] == "cloudpay_v2" and rng.random() < 0.30:
            row["amount"] = ""


register(
    Fault(
        name="null_surge",
        report="Exec revenue has been down since the middle of last week. Real or data?",
        inject=_inject_null_surge,
        ground_truth=GroundTruth("escalate", "raw.raw_orders", "amount"),
        must_fail=("raw_no_null_amount", "revenue_within_baseline_band"),
    )
)


register(
    Fault(
        name="control_healthy",
        report="Revenue felt slightly off this morning. Can you sanity check the exec KPI?",
        inject=lambda frames: None,
        ground_truth=GroundTruth("no_incident", None, None),
    )
)


def _inject_distractor(frames: Frames) -> None:
    _drop_vendor_rates_after(frames, (ANCHOR_DAY - timedelta(days=1)).isoformat())


register(
    Fault(
        name="control_distractor",
        report="An FX freshness alert fired overnight. Is the exec revenue KPI affected?",
        inject=_inject_distractor,
        ground_truth=GroundTruth("no_incident", None, None),
    )
)
