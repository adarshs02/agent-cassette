"""Recursive secret redaction for cassette payloads."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
_MAX_DEPTH = 64
_SECRET_KEY = re.compile(
    r"(?:authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")


class RedactionError(ValueError):
    """Raised when a value cannot be redacted safely."""


def redact(value: Any) -> Any:
    """Return a copy with common credentials removed."""
    return _redact_at(value, depth=0, active=set())


def _redact_at(value: Any, *, depth: int, active: set[int]) -> Any:
    if depth >= _MAX_DEPTH and isinstance(value, (dict, list, tuple)):
        raise RedactionError(f"maximum redaction depth {_MAX_DEPTH} exceeded")

    if isinstance(value, dict):
        value_id = _enter_container(value, active)
        try:
            return {
                key: REDACTED
                if isinstance(key, str) and _SECRET_KEY.search(key)
                else _redact_at(item, depth=depth + 1, active=active)
                for key, item in value.items()
            }
        finally:
            active.remove(value_id)
    if isinstance(value, list):
        value_id = _enter_container(value, active)
        try:
            return [_redact_at(item, depth=depth + 1, active=active) for item in value]
        finally:
            active.remove(value_id)
    if isinstance(value, tuple):
        value_id = _enter_container(value, active)
        try:
            return tuple(_redact_at(item, depth=depth + 1, active=active) for item in value)
        finally:
            active.remove(value_id)
    if isinstance(value, str):
        return _BEARER.sub(f"Bearer {REDACTED}", value)
    return value


def _enter_container(value: object, active: set[int]) -> int:
    value_id = id(value)
    if value_id in active:
        raise RedactionError("cyclic value cannot be redacted")
    active.add(value_id)
    return value_id
