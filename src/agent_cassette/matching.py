"""Input normalization and matching policies for deterministic replay."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Literal

from agent_cassette.redaction import redact

MatchMode = Literal["exact", "subset", "normalized", "fuzzy"]
InputMatcher = Callable[[Any, Any], bool]

DEFAULT_FUZZY_THRESHOLD = 0.9


def normalize_input(value: Any, ignore_paths: tuple[str, ...] = ()) -> Any:
    """Redact an input and remove explicitly ignored dotted paths."""
    normalized = deepcopy(redact(value))
    for path in ignore_paths:
        _remove_path(normalized, path.split("."))
    return normalized


def inputs_match(
    expected: Any,
    actual: Any,
    *,
    mode: MatchMode = "exact",
    matcher: InputMatcher | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> bool:
    """Compare normalized inputs using a built-in or custom policy."""
    if matcher is not None:
        return matcher(expected, actual)
    if mode == "subset":
        return _is_subset(expected, actual)
    if mode == "normalized":
        return _tolerant_equal(expected, actual, threshold=None)
    if mode == "fuzzy":
        validate_fuzzy_threshold(fuzzy_threshold)
        return _tolerant_equal(expected, actual, threshold=fuzzy_threshold)
    return expected == actual


def validate_fuzzy_threshold(threshold: float) -> None:
    """Reject fuzzy similarity thresholds outside (0, 1]."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("fuzzy_threshold must be a number in (0, 1]")
    if not 0 < threshold <= 1:
        raise ValueError("fuzzy_threshold must be in (0, 1]")


def normalize_text(value: str) -> str:
    """Collapse whitespace and casefold a string for tolerant comparison."""
    return " ".join(value.split()).casefold()


def _tolerant_equal(expected: Any, actual: Any, *, threshold: float | None) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        expected_text = normalize_text(expected)
        actual_text = normalize_text(actual)
        if expected_text == actual_text:
            return True
        if threshold is None:
            return False
        return SequenceMatcher(None, expected_text, actual_text).ratio() >= threshold
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _tolerant_equal(value, actual[key], threshold=threshold)
            for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _tolerant_equal(expected_item, actual_item, threshold=threshold)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    return expected == actual


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _is_subset(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    return expected == actual


def _remove_path(value: Any, parts: list[str]) -> None:
    if not parts:
        return
    part = parts[0]
    if isinstance(value, dict):
        if part == "*":
            for child in value.values():
                _remove_path(child, parts[1:])
        elif len(parts) == 1:
            value.pop(part, None)
        elif part in value:
            _remove_path(value[part], parts[1:])
    elif isinstance(value, list):
        if part == "*":
            for child in value:
                _remove_path(child, parts[1:])
        elif part.isdigit() and int(part) < len(value):
            if len(parts) == 1:
                value.pop(int(part))
            else:
                _remove_path(value[int(part)], parts[1:])
