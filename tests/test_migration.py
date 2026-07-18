import json

import pytest

from agent_cassette import Cassette, EventType
from agent_cassette.events import Event, register_migration, unregister_migration
from agent_cassette.migration import migrate_cassette, migrate_event_dict


def test_migrate_normalizes_legacy_schema_to_destination(tmp_path):
    source = tmp_path / "legacy.jsonl"
    with Cassette.record(source) as cassette:
        cassette.add(EventType.CUSTOM, "legacy", output=True)
    data = json.loads(source.read_text())
    data.pop("schema_version")
    source.write_text(json.dumps(data) + "\n")

    destination = tmp_path / "upgraded.jsonl"
    assert migrate_cassette(source, destination) == destination
    migrated = json.loads(destination.read_text())
    assert migrated["schema_version"] == 1


def test_migrate_requires_a_separate_destination(tmp_path):
    source = tmp_path / "run.jsonl"
    with Cassette.record(source) as cassette:
        cassette.add(EventType.CUSTOM, "legacy", output=True)
    with pytest.raises(ValueError, match="must differ from source"):
        migrate_cassette(source, source)


@pytest.fixture
def future_schema(monkeypatch):
    """Pretend the current release uses schema version 2 with a 1 -> 2 migration."""
    monkeypatch.setattr("agent_cassette.events.SCHEMA_VERSION", 2)

    def upgrade(data):
        data.setdefault("metadata", {})["upgraded"] = True
        data["schema_version"] = 2
        return data

    register_migration(1, upgrade)
    yield
    unregister_migration(1)


def test_registered_migration_upgrades_old_events(future_schema):
    event = Event.from_dict(
        {
            "id": "1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "custom",
            "name": "legacy",
            "schema_version": 1,
        }
    )
    assert event.schema_version == 2
    assert event.metadata == {"upgraded": True}


def test_migrate_cassette_rewrites_old_events_through_chain(tmp_path, future_schema):
    source = tmp_path / "old.jsonl"
    with Cassette.record(source) as cassette:
        cassette.add(EventType.CUSTOM, "legacy", output=True)
    lines = [json.loads(line) for line in source.read_text().splitlines()]
    for line in lines:
        line["schema_version"] = 1
    source.write_text("".join(json.dumps(line) + "\n" for line in lines))

    destination = tmp_path / "upgraded.jsonl"
    migrate_cassette(source, destination)
    migrated = json.loads(destination.read_text().splitlines()[0])
    assert migrated["schema_version"] == 2
    assert migrated["metadata"]["upgraded"] is True


def test_newer_schema_versions_are_rejected():
    with pytest.raises(ValueError, match="newer than this release"):
        migrate_event_dict({"schema_version": 99})


def test_missing_migration_path_is_rejected(monkeypatch):
    monkeypatch.setattr("agent_cassette.events.SCHEMA_VERSION", 2)
    with pytest.raises(ValueError, match="No migration registered"):
        migrate_event_dict({"schema_version": 1})


def test_migration_must_advance_exactly_one_version(monkeypatch):
    monkeypatch.setattr("agent_cassette.events.SCHEMA_VERSION", 3)
    register_migration(1, lambda data: {**data, "schema_version": 3})
    try:
        with pytest.raises(ValueError, match="instead of 2"):
            migrate_event_dict({"schema_version": 1})
    finally:
        unregister_migration(1)


def test_duplicate_migration_registration_is_rejected():
    register_migration(1, lambda data: data)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_migration(1, lambda data: data)
    finally:
        unregister_migration(1)
