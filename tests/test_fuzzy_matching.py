import pytest

from agent_cassette import Cassette, EventType, ReplayMismatchError
from agent_cassette.matching import inputs_match, normalize_text


def _record(path):
    with Cassette.record(path) as cassette:
        cassette.call(
            EventType.MODEL_CALL,
            "answer",
            {"prompt": "Summarize the  meeting notes."},
            lambda: "summary",
        )


def test_normalized_matching_tolerates_whitespace_and_case(tmp_path):
    path = tmp_path / "normalized.jsonl"
    _record(path)

    with Cassette.replay(path, match="normalized") as cassette:
        result = cassette.call(
            EventType.MODEL_CALL,
            "answer",
            {"prompt": "  summarize the meeting notes. "},
        )
    assert result == "summary"


def test_normalized_matching_still_rejects_different_text(tmp_path):
    path = tmp_path / "normalized-reject.jsonl"
    _record(path)

    with pytest.raises(ReplayMismatchError):
        with Cassette.replay(path, match="normalized") as cassette:
            cassette.call(
                EventType.MODEL_CALL,
                "answer",
                {"prompt": "Summarize the budget spreadsheet."},
            )


def test_fuzzy_matching_tolerates_small_edits(tmp_path):
    path = tmp_path / "fuzzy.jsonl"
    _record(path)

    with Cassette.replay(path, match="fuzzy") as cassette:
        result = cassette.call(
            EventType.MODEL_CALL,
            "answer",
            {"prompt": "Summarize the meeting notes"},
        )
    assert result == "summary"


def test_fuzzy_matching_rejects_low_similarity(tmp_path):
    path = tmp_path / "fuzzy-reject.jsonl"
    _record(path)

    with pytest.raises(ReplayMismatchError):
        with Cassette.replay(path, match="fuzzy") as cassette:
            cassette.call(
                EventType.MODEL_CALL,
                "answer",
                {"prompt": "Write a haiku about databases."},
            )


def test_fuzzy_threshold_is_configurable(tmp_path):
    path = tmp_path / "fuzzy-threshold.jsonl"
    _record(path)

    with Cassette.replay(path, match="fuzzy", fuzzy_threshold=0.3) as cassette:
        result = cassette.call(
            EventType.MODEL_CALL,
            "answer",
            {"prompt": "Summarize meeting notes and the action items."},
        )
    assert result == "summary"


def test_invalid_fuzzy_threshold_is_rejected(tmp_path):
    path = tmp_path / "invalid-threshold.jsonl"
    _record(path)

    with pytest.raises(ValueError, match="fuzzy_threshold"):
        Cassette.replay(path, match="fuzzy", fuzzy_threshold=0)
    with pytest.raises(ValueError, match="fuzzy_threshold"):
        Cassette.replay(path, match="fuzzy", fuzzy_threshold=1.5)


def test_tolerant_matching_requires_matching_structure():
    assert inputs_match({"a": "x  y"}, {"a": "X Y"}, mode="normalized")
    assert not inputs_match({"a": "x"}, {"a": "x", "b": "y"}, mode="normalized")
    assert not inputs_match(["x"], ["x", "y"], mode="fuzzy")
    assert not inputs_match({"a": 1}, {"a": 2}, mode="fuzzy")


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Hello   World\n") == "hello world"
