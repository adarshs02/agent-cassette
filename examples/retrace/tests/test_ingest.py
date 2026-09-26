import pytest

pytest.importorskip("datahub")

from retrace.datahub.catalog import TABLES, dataset_urn  # noqa: E402
from retrace.datahub.ingest import build_proposals  # noqa: E402


def test_proposals_cover_every_table(healthy_ws):
    proposals = build_proposals(healthy_ws.warehouse)
    urns = {p.entityUrn for p in proposals}
    for table in TABLES:
        assert dataset_urn(table) in urns
    aspects = {type(p.aspect).__name__ for p in proposals}
    assert {
        "DatasetPropertiesClass",
        "SchemaMetadataClass",
        "OwnershipClass",
        "UpstreamLineageClass",
        "CorpUserInfoClass",
    } <= aspects


def test_schema_is_introspected(tmp_path):
    from retrace.pipeline.workspace import prepare

    ws = prepare(tmp_path / "rename", fault="schema_rename")
    proposals = build_proposals(ws.warehouse)
    raw = next(
        p
        for p in proposals
        if p.entityUrn == dataset_urn("raw.raw_orders")
        and type(p.aspect).__name__ == "SchemaMetadataClass"
    )
    assert "currency_code" in {f.fieldPath for f in raw.aspect.fields}


def test_docs_never_mention_faults(healthy_ws):
    text = " ".join(
        str(getattr(p.aspect, "description", "")) for p in build_proposals(healthy_ws.warehouse)
    ).lower()
    for token in ("cents", "duplicate", "stale", "null surge", "timezone shift"):
        assert token not in text
