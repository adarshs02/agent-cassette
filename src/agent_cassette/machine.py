"""Machine-operable response envelope for the Phase E agent-native CLI loop.

Every Phase E command (``setup``, ``status``, ``agent-manifest``, named ``record``/
``replay``/``rerecord``, and ``ci``) and every runner report file emits the single
envelope defined here as canonical JSON (``indent=2``, ``sort_keys=True``, trailing
newline). The envelope carries only code-owned, bounded data: it never contains a child
command, environment value, payload, exception message, or credential.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_cassette.json_codec import validate_json_value

ENVELOPE_SCHEMA_VERSION = 1

# Semantic process exit codes, shared by every Phase E command.
EXIT_OK = 0
EXIT_CHANGES = 1  # check needs changes, replay mismatch, or child returned nonzero
EXIT_INVALID = 2  # invalid input/config/cassette/report path, conflict, or unsafe state

Network = Literal["forbidden", "allowed", "required"]

# A named cassette is an exact ``str`` in this shape: a leading alphanumeric then up to
# 127 more of ``[A-Za-z0-9._-]``. Path separators, ``.``/``..``, control characters, and
# normalization tricks are rejected by exact-type + exact-regex matching.
_CASSETTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class MachineInputError(ValueError):
    """Invalid machine-command input that is safe to surface without a traceback."""


def validate_cassette_name(name: Any) -> str:
    """Return ``name`` if it is an exact, safe cassette name, else raise."""
    if type(name) is not str or _CASSETTE_NAME.fullmatch(name) is None:
        raise MachineInputError("cassette name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    if name in (".", "..") or "/" in name or "\\" in name:  # defence in depth
        raise MachineInputError("cassette name must not contain path separators")
    return name


@dataclass(frozen=True, slots=True)
class NextAction:
    """One deterministic follow-up an agent may take. ``argv`` is an argument vector,
    never a shell string; a command needing the caller's Python command ends ``argv`` at
    ``"--"`` and sets ``requires_child_command=True``."""

    id: str
    argv: tuple[str, ...]
    mutates: tuple[str, ...] = ()
    network: Network = "forbidden"
    approval_required: bool = False
    requires_child_command: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "argv": list(self.argv),
            "mutates": list(self.mutates),
            "network": self.network,
            "approval_required": self.approval_required,
            "requires_child_command": self.requires_child_command,
        }


@dataclass(frozen=True, slots=True)
class Envelope:
    """The single machine response shape. ``ok`` is true only for an exit-0 outcome and
    ``exit_code`` always equals the process result."""

    command: str
    status: str
    exit_code: int
    project: str
    warnings: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "command": self.command,
            "status": self.status,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "project": self.project,
            "warnings": list(self.warnings),
            "changes": list(self.changes),
            "next_actions": [action.to_dict() for action in self.next_actions],
            "data": self.data,
        }

    def to_json(self) -> str:
        payload = self.to_dict()
        validate_json_value(payload)  # code-owned, bounded, JSON-native only
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
