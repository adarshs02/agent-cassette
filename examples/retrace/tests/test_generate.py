import csv
from collections import Counter

from retrace.pipeline.generate import (
    ANCHOR_DAY,
    CUTOVER_DAY,
    START_DAY,
    generate,
    generate_frames,
)


def test_generation_is_byte_identical(tmp_path):
    generate(tmp_path / "a")
    generate(tmp_path / "b")
    for name in (
        "raw_orders_part1.csv",
        "raw_orders_part2.csv",
        "raw_customers.csv",
        "raw_fx_rates.csv",
    ):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_orders_split_at_cutover(tmp_path):
    generate(tmp_path)
    with (tmp_path / "raw_orders_part1.csv").open() as f:
        part1 = list(csv.DictReader(f))
    with (tmp_path / "raw_orders_part2.csv").open() as f:
        part2 = list(csv.DictReader(f))
    assert all(r["order_ts"][:10] < CUTOVER_DAY.isoformat() for r in part1)
    assert all(r["order_ts"][:10] >= CUTOVER_DAY.isoformat() for r in part2)
    assert {r["payment_processor"] for r in part1} == {"legacy_pos", "shopgate"}
    assert "cloudpay_v2" in {r["payment_processor"] for r in part2}


def test_fixture_shape():
    frames = generate_frames()
    days = {r["order_ts"][:10] for r in frames.orders}
    assert min(days) == START_DAY.isoformat()
    assert max(days) == ANCHOR_DAY.isoformat()
    assert len(days) == 90
    assert len({r["order_id"] for r in frames.orders}) == len(frames.orders)
    fx_days = Counter(r["currency"] for r in frames.fx_rates)
    assert set(fx_days.values()) == {90}
    assert all(r["usd_rate"] == "1.000000" for r in frames.fx_rates if r["currency"] == "USD")
