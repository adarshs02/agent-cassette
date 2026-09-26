import pytest
from retrace.agent.evidence import (
    EvidenceStore,
    GateError,
    check_claim_evidence,
    check_no_incident_evidence,
)
from retrace.agent.stages import advance, fail
from retrace.agent.state import IncidentState, Stage

URN = "urn:li:dataset:(urn:li:dataPlatform:duckdb,raw.raw_orders,PROD)"


def _store():
    state = IncidentState(scenario="t", report="r")
    store = EvidenceStore(state)
    lineage = store.add(
        "datahub",
        "lineage",
        "upstream of KPI",
        {"upstreams": [{"urn": URN, "name": "raw.raw_orders"}]},
    )
    profile = store.add("warehouse", "profile", "amount by processor", {"rows": []})
    transform = store.add("pipeline", "transform", "stg_orders.sql", "SELECT 1")
    return state, store, lineage, profile, transform


def test_ids_are_sequential():
    _, _, lineage, profile, transform = _store()
    assert [lineage.id, profile.id, transform.id] == ["ev_001", "ev_002", "ev_003"]


def test_valid_claim_passes_with_urn_or_table():
    _, store, lineage, profile, _ = _store()
    check_claim_evidence(store, URN, [lineage.id, profile.id])
    check_claim_evidence(store, "raw.raw_orders", [lineage.id, profile.id])


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["ev_999"], "unknown evidence ids"),
        (["ev_002"], "DataHub"),
        (["ev_001"], "warehouse"),
        (["ev_001", "ev_003"], "warehouse"),
    ],
)
def test_claim_gate_rejections(ids, message):
    _, store, *_ = _store()
    with pytest.raises(GateError, match=message):
        check_claim_evidence(store, "raw.raw_orders", ids)


def test_claim_gate_requires_asset_in_lineage():
    _, store, lineage, profile, _ = _store()
    with pytest.raises(GateError, match="lineage"):
        check_claim_evidence(store, "raw.raw_customers", [lineage.id, profile.id])


def test_claim_gate_requires_cited_lineage():
    state = IncidentState(scenario="t", report="r")
    store = EvidenceStore(state)
    store.add(
        "datahub",
        "lineage",
        "upstream of KPI",
        {"upstreams": [{"urn": URN, "name": "raw.raw_orders"}]},
    )
    metadata = store.add("datahub", "metadata", "table metadata", {"owner": "analytics"})
    profile = store.add("warehouse", "profile", "amount by processor", {"rows": []})
    with pytest.raises(GateError, match="lineage"):
        check_claim_evidence(store, "raw.raw_orders", [metadata.id, profile.id])


def test_no_incident_gate():
    state = IncidentState(scenario="t", report="r")
    store = EvidenceStore(state)
    calm = store.add("warehouse", "baseline_comparison", "ok", {"max_abs_deviation": 0.004})
    wild = store.add("warehouse", "baseline_comparison", "bad", {"max_abs_deviation": 0.4})
    check_no_incident_evidence(store, [calm.id])
    with pytest.raises(GateError, match="baseline"):
        check_no_incident_evidence(store, [wild.id])
    with pytest.raises(GateError, match="baseline"):
        check_no_incident_evidence(store, [])


def test_stage_transitions():
    state = IncidentState(scenario="t", report="r")
    with pytest.raises(GateError, match="cannot move"):
        advance(state, Stage.VERIFIED)
    advance(state, Stage.ROOT_CAUSE_CONFIRMED)
    advance(state, Stage.REPAIRING)
    advance(state, Stage.REPAIRING)  # idempotent
    advance(state, Stage.VERIFIED)
    advance(state, Stage.WRITTEN_BACK)
    fail(state, "boom")
    assert state.stage is Stage.FAILED and state.failure_reason == "boom"


def test_to_dict_is_json_ready():
    import json

    state, *_ = _store()
    data = state.to_dict()
    assert data["stage"] == "INVESTIGATING"
    json.dumps(data)
