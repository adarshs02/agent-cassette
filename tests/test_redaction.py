import json

from agent_cassette import Cassette, EventType
from agent_cassette.redaction import REDACTED, redact


def test_redacts_nested_secrets_and_bearer_tokens():
    value = {
        "headers": {"Authorization": "Bearer super-secret"},
        "api_key": "sk-live",
        "safe": ["Bearer abc123", "visible"],
    }

    assert redact(value) == {
        "headers": {"Authorization": REDACTED},
        "api_key": REDACTED,
        "safe": [f"Bearer {REDACTED}", "visible"],
    }


def test_recording_redacts_before_writing(tmp_path):
    path = tmp_path / "safe.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.TOOL_CALL, "request", input={"access_token": "secret"})

    raw = path.read_text()
    assert "secret" not in raw
    assert json.loads(raw)["input"]["access_token"] == REDACTED
