"""v1.0 cassettes (token counts redacted) must still match live requests with int counts."""

from __future__ import annotations

import json

import pytest

from agent_cassette import Cassette, EventType, ReplayMismatchError
from agent_cassette.matching import inputs_match, normalize_input
from agent_cassette.redaction import REDACTED, is_token_count_field

REQUEST = {"model": "m", "max_tokens": 1024, "messages": [{"role": "user", "content": "hi"}]}


def _record(path, request):
    with Cassette.record(path) as cassette:
        cassette.call(EventType.MODEL_CALL, "messages.create", request, lambda: {"ok": 1})


def _downgrade_to_v1_0(path):
    """Rewrite a recording the way v1.0.0 persisted it: token counts as the marker."""
    lines = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if isinstance(event.get("input"), dict) and "max_tokens" in event["input"]:
            event["input"]["max_tokens"] = REDACTED
        lines.append(json.dumps(event))
    path.write_text("\n".join(lines) + "\n")


def test_is_token_count_field_predicate():
    assert is_token_count_field("max_tokens", 1024)
    assert is_token_count_field("promptTokenCount", 3)
    assert not is_token_count_field("max_tokens", "1024")
    assert not is_token_count_field("max_tokens", True)
    assert not is_token_count_field("token", 5)
    assert not is_token_count_field("temperature", 1)


def test_normalize_matches_redacted_and_int_counts():
    old = normalize_input({"max_tokens": REDACTED})
    new = normalize_input({"max_tokens": 1024})
    assert inputs_match(old, new)


def test_v1_0_cassette_with_redacted_max_tokens_replays(tmp_path):
    path = tmp_path / "old.jsonl"
    _record(path, REQUEST)
    _downgrade_to_v1_0(path)
    assert json.loads(path.read_text().splitlines()[0])["input"]["max_tokens"] == REDACTED
    with Cassette.replay(path) as cassette:
        assert cassette.call(EventType.MODEL_CALL, "messages.create", REQUEST) == {"ok": 1}


def test_new_cassette_keeps_int_and_replays(tmp_path):
    path = tmp_path / "new.jsonl"
    _record(path, REQUEST)
    assert json.loads(path.read_text().splitlines()[0])["input"]["max_tokens"] == 1024
    with Cassette.replay(path) as cassette:
        assert cassette.call(EventType.MODEL_CALL, "messages.create", REQUEST) == {"ok": 1}


def test_non_count_field_still_mismatches(tmp_path):
    path = tmp_path / "old.jsonl"
    _record(path, REQUEST)
    _downgrade_to_v1_0(path)
    changed = {**REQUEST, "model": "other"}
    with pytest.raises(ReplayMismatchError):
        with Cassette.replay(path) as cassette:
            cassette.call(EventType.MODEL_CALL, "messages.create", changed)
