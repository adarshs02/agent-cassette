"""Strict, bounded JSON validation used by cassette storage."""

from __future__ import annotations

import json
import math
from typing import Any

MAX_JSON_DEPTH = 64


class StrictJSONError(ValueError):
    """Raised when JSON violates the cassette's strict encoding profile."""


def validate_json_value(value: Any) -> None:
    """Validate that ``value`` is finite, acyclic, bounded JSON data."""
    _validate_json_value(value, depth=0, active=set())


def copy_json_value(value: Any) -> Any:
    """Return a detached copy of validated JSON data."""
    validate_json_value(value)
    return _copy_json_value(value)


def strict_json_loads(value: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_object_from_unique_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
        validate_json_value(decoded)
        return decoded
    except json.JSONDecodeError:
        raise
    except StrictJSONError:
        raise
    except RecursionError as error:
        raise StrictJSONError("maximum cassette JSON depth exceeded while decoding") from error


def strict_json_dumps(value: Any) -> str:
    """Encode validated JSON without implicit object stringification."""
    validate_json_value(value)
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _validate_json_value(value: Any, *, depth: int, active: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError("non-finite floats are not valid cassette JSON")
        return
    if not isinstance(value, (dict, list)):
        raise StrictJSONError(
            f"unsupported cassette JSON value type: {type(value).__module__}."
            f"{type(value).__qualname__}"
        )
    if depth >= MAX_JSON_DEPTH:
        raise StrictJSONError(f"maximum cassette JSON depth {MAX_JSON_DEPTH} exceeded")

    value_id = id(value)
    if value_id in active:
        raise StrictJSONError("cyclic values are not valid cassette JSON")
    active.add(value_id)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise StrictJSONError("cassette JSON object keys must be strings")
                _validate_json_value(item, depth=depth + 1, active=active)
        else:
            for item in value:
                _validate_json_value(item, depth=depth + 1, active=active)
    finally:
        active.remove(value_id)


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _object_from_unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError("duplicate JSON object keys are not allowed")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise StrictJSONError(f"non-finite JSON number {value} is not allowed")


__all__ = [
    "MAX_JSON_DEPTH",
    "StrictJSONError",
    "copy_json_value",
    "strict_json_dumps",
    "strict_json_loads",
    "validate_json_value",
]
