import json

from agent_cassette import Cassette, EventType
from agent_cassette.migration import migrate_cassette


def test_migrate_normalizes_legacy_schema_in_place(tmp_path):
    path = tmp_path / "legacy.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.CUSTOM, "legacy", output=True)
    data = json.loads(path.read_text())
    data.pop("schema_version")
    path.write_text(json.dumps(data) + "\n")

    assert migrate_cassette(path) == path
    migrated = json.loads(path.read_text())
    assert migrated["schema_version"] == 1
