"""Recursive secret redaction for cassette payloads."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(
    r"(?:authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|secret|password)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")


def redact(value: Any) -> Any:
    """Return a copy with common credentials removed."""
    if isinstance(value, dict):
        return {
            key: REDACTED if _SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return _BEARER.sub(f"Bearer {REDACTED}", value)
    return value
