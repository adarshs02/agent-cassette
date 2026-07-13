"""Versioned event types used by cassette files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


class EventType(str, Enum):
    """Built-in event categories."""

    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass(slots=True)
class Event:
    """One observable step in an agent execution."""

    id: str
    timestamp: str
    type: EventType
    name: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    cost: float | None = None
    parent_id: str | None = None
    span_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event dictionary."""
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Build an event while validating its schema version."""
        version = data.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported cassette schema version: {version}")
        values = dict(data)
        values["type"] = EventType(values["type"])
        return cls(**values)
