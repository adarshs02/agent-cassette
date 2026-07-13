"""Input normalization and matching policies for deterministic replay."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from agent_cassette.redaction import redact

MatchMode = Literal["exact", "subset"]
InputMatcher = Callable[[Any, Any], bool]


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
) -> bool:
    """Compare normalized inputs using a built-in or custom policy."""
    if matcher is not None:
        return matcher(expected, actual)
    if mode == "subset":
        return _is_subset(expected, actual)
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
