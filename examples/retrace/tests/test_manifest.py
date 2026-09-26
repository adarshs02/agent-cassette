import json

import pytest
from retrace.config import CASSETTE_DIR
from retrace.evals.runner import MANIFEST
from retrace.evals.scenarios import SCENARIOS

manifest = CASSETTE_DIR / MANIFEST


@pytest.mark.skipif(not manifest.exists(), reason="live cassettes not recorded yet")
def test_every_agent_scenario_has_a_cassette():
    recorded = set(json.loads(manifest.read_text())["scenarios"])
    expected = {s.name for s in SCENARIOS if s.kind != "gate"}
    assert expected <= recorded
    for name in expected:
        path = CASSETTE_DIR / f"{name}.jsonl"
        assert path.exists()
        assert path.stat().st_size < 10 * 1024 * 1024, f"{name} cassette over 10 MB"
