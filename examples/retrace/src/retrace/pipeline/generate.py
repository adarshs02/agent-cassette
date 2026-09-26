# Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub),
# Apache-2.0. Modified for Retrace.
"""Deterministic synthetic retail data for the Retrace pipeline."""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

ANCHOR_DAY = date(2026, 8, 9)
DAYS = 90
START_DAY = ANCHOR_DAY - timedelta(days=DAYS - 1)
CUTOVER_DAY = date(2026, 8, 7)
SEED = 42
ORDERS_PER_DAY = 300
N_CUSTOMERS = 2000
CURRENCIES = ("USD", "EUR", "GBP", "CAD")
CURRENCY_WEIGHTS = (0.80, 0.10, 0.06, 0.04)
BASE_RATES = {"USD": 1.0, "EUR": 1.09, "GBP": 1.27, "CAD": 0.73}
SEGMENTS = ("consumer", "smb", "enterprise")
COUNTRIES = ("US", "DE", "GB", "CA")
# Orders per UTC hour; demand peaks in the US afternoon/evening (15:00-03:00 UTC).
HOUR_WEIGHTS = (3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 5, 6, 6, 6, 6, 5, 4)
PRE_CUTOVER_PROCESSORS = (("legacy_pos", 0.6), ("shopgate", 0.4))
POST_CUTOVER_PROCESSORS = (("legacy_pos", 0.35), ("shopgate", 0.25), ("cloudpay_v2", 0.40))
STATUSES = (("completed", 0.92), ("cancelled", 0.05), ("refunded", 0.03))
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

Row = dict[str, str]


@dataclass
class Frames:
    orders: list[Row] = field(default_factory=list)
    customers: list[Row] = field(default_factory=list)
    fx_rates: list[Row] = field(default_factory=list)


def _pick(rng: random.Random, options: Sequence[tuple[str, float]]) -> str:
    values = [value for value, _ in options]
    weights = [weight for _, weight in options]
    return rng.choices(values, weights=weights, k=1)[0]


def generate_frames() -> Frames:
    rng = random.Random(SEED)
    frames = Frames()
    for index in range(N_CUSTOMERS):
        created = START_DAY - timedelta(days=rng.randrange(1, 720))
        frames.customers.append(
            {
                "customer_id": f"c{index:05d}",
                "segment": rng.choice(SEGMENTS),
                "country": rng.choice(COUNTRIES),
                "created_at": created.isoformat(),
            }
        )
    rates = dict(BASE_RATES)
    order_seq = 0
    for offset in range(DAYS):
        day = START_DAY + timedelta(days=offset)
        for currency in CURRENCIES:
            if currency != "USD":
                rates[currency] *= 1 + rng.gauss(0, 0.002)
            frames.fx_rates.append(
                {
                    "rate_day": day.isoformat(),
                    "currency": currency,
                    "usd_rate": f"{rates[currency]:.6f}",
                }
            )
        weekend = day.weekday() >= 5
        count = round(ORDERS_PER_DAY * (0.85 if weekend else 1.0) * (1 + rng.uniform(-0.05, 0.05)))
        processors = POST_CUTOVER_PROCESSORS if day >= CUTOVER_DAY else PRE_CUTOVER_PROCESSORS
        for _ in range(count):
            order_seq += 1
            hour = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
            ts = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                rng.randrange(60),
                rng.randrange(60),
            )
            amount = min(max(rng.lognormvariate(3.6, 0.6), 2.0), 2000.0)
            frames.orders.append(
                {
                    "order_id": f"o{order_seq:07d}",
                    "order_ts": ts.strftime(TS_FORMAT),
                    "customer_id": f"c{rng.randrange(N_CUSTOMERS):05d}",
                    "currency": rng.choices(CURRENCIES, weights=CURRENCY_WEIGHTS, k=1)[0],
                    "amount": f"{amount:.2f}",
                    "payment_processor": _pick(rng, processors),
                    "status": _pick(rng, STATUSES),
                }
            )
    return frames


def _write_csv(path: Path, rows: list[Row]) -> None:
    header = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sources(frames: Frames, sources: Path) -> None:
    sources.mkdir(parents=True, exist_ok=True)
    cutover = CUTOVER_DAY.isoformat()
    _write_csv(
        sources / "raw_orders_part1.csv",
        [r for r in frames.orders if r["order_ts"][:10] < cutover],
    )
    _write_csv(
        sources / "raw_orders_part2.csv",
        [r for r in frames.orders if r["order_ts"][:10] >= cutover],
    )
    _write_csv(sources / "raw_customers.csv", frames.customers)
    _write_csv(sources / "raw_fx_rates.csv", frames.fx_rates)


def generate(sources: Path, fault: str | None = None) -> None:
    frames = generate_frames()
    if fault is not None:
        from retrace.faults import get_fault

        get_fault(fault).inject(frames)
    write_sources(frames, sources)
