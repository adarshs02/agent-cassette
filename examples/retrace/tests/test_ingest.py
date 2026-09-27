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


def test_every_table_gets_global_tags(healthy_ws):
    from retrace.datahub.catalog import TAGS

    proposals = build_proposals(healthy_ws.warehouse)
    tags = {
        p.entityUrn: [t.tag for t in p.aspect.tags]
        for p in proposals
        if type(p.aspect).__name__ == "GlobalTagsClass"
    }
    assert len(TABLES) == 8
    assert set(tags) == {dataset_urn(t) for t in TABLES}
    for table in TABLES:
        expected = [f"urn:li:tag:{tag}" for tag in TAGS.get(table, [])]
        assert tags[dataset_urn(table)] == expected


def test_tag_entities_are_created_before_dataset_proposals(healthy_ws):
    from retrace.datahub.catalog import INCIDENT_TAG

    proposals = build_proposals(healthy_ws.warehouse)
    tag_props = {
        p.entityUrn: p.aspect for p in proposals if type(p.aspect).__name__ == "TagPropertiesClass"
    }
    for tag in (INCIDENT_TAG, "kpi", "executive-reporting", "revenue"):
        urn = f"urn:li:tag:{tag}"
        assert urn in tag_props
        assert tag_props[urn].name == tag

    first_dataset_index = next(
        i for i, p in enumerate(proposals) if type(p.aspect).__name__ == "DatasetPropertiesClass"
    )
    tag_indices = [
        i for i, p in enumerate(proposals) if type(p.aspect).__name__ == "TagPropertiesClass"
    ]
    assert tag_indices and all(i < first_dataset_index for i in tag_indices)


def test_soft_delete_emits_status_removed(monkeypatch):
    from retrace.config import Settings
    from retrace.datahub import ingest

    emitted = []

    class FakeEmitter:
        def __init__(self, **kwargs):
            pass

        def emit_mcp(self, proposal):
            emitted.append(proposal)

    monkeypatch.setattr("datahub.emitter.rest_emitter.DatahubRestEmitter", FakeEmitter)
    assert ingest.soft_delete(["urn:li:document:a", "urn:li:document:b"], Settings()) == 2
    assert [p.entityUrn for p in emitted] == ["urn:li:document:a", "urn:li:document:b"]
    assert all(type(p.aspect).__name__ == "StatusClass" and p.aspect.removed for p in emitted)
