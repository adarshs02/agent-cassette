"""Read and write JSONL cassette files."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from agent_cassette.events import Event
from agent_cassette.json_codec import StrictJSONError, strict_json_dumps, strict_json_loads

_LITERAL_PREFIX = re.compile(r"(?:^|[\s\[:,])(?P<token>t|tr|tru|f|fa|fal|fals|n|nu|nul)$")
_LITERAL_COMPLETIONS = {"t": "true", "tr": "true", "tru": "true"}
_LITERAL_COMPLETIONS.update({prefix: "false" for prefix in ("f", "fa", "fal", "fals")})
_LITERAL_COMPLETIONS.update({prefix: "null" for prefix in ("n", "nu", "nul")})
_INCOMPLETE_NUMBER = re.compile(
    r"(?:^|[\s\[:,])(?P<token>-(?:$)|-?(?:0|[1-9]\d*)\.$|"
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?$)"
)
_PARTIAL_UNICODE_ESCAPE = re.compile(r"\\u(?P<digits>[0-9a-fA-F]{0,3})$")


class CassetteCorruptionError(ValueError):
    """Raised when a cassette line is not valid, supported event JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        line_number: int,
        byte_offset: int,
        reason: str,
    ) -> None:
        self.path = Path(path)
        self.line_number = line_number
        self.byte_offset = byte_offset
        self.reason = _safe_diagnostic(reason)
        super().__init__(
            f"Invalid cassette event in {str(self.path)!r} at line {line_number}, "
            f"byte {byte_offset}: {self.reason}"
        )


class _RecoverableTailError(CassetteCorruptionError):
    """Internal marker for malformed bytes consistent with a torn final write."""


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Result of validating and normalizing a cassette into a new file."""

    source: Path
    destination: Path
    events: int
    discarded_offset: int | None
    discarded_bytes: int

    @property
    def recovered(self) -> bool:
        """Whether an unterminated malformed tail was discarded."""
        return self.discarded_offset is not None

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-safe recovery output for CLI consumers."""
        discarded = (
            None
            if self.discarded_offset is None
            else {"offset": self.discarded_offset, "bytes": self.discarded_bytes}
        )
        return {
            "schema_version": 1,
            "command": "recover",
            "source": str(self.source),
            "destination": str(self.destination),
            "status": "recovered" if self.recovered else "unchanged",
            "events": self.events,
            "discarded": discarded,
        }


def load_events(path: str | Path) -> list[Event]:
    """Load all events from a cassette file."""
    cassette_path = Path(path)
    if not cassette_path.exists():
        raise FileNotFoundError(f"Cassette does not exist: {cassette_path}")

    events: list[Event] = []
    with cassette_path.open("rb") as cassette_file:
        byte_offset = 0
        for line_number, raw_line in enumerate(cassette_file, start=1):
            event = _parse_event_line(cassette_path, raw_line, line_number, byte_offset)
            if event is not None:
                events.append(event)
            byte_offset += len(raw_line)
    return events


def append_event(path: str | Path, event: Event) -> None:
    """Append and flush one event so completed calls survive process failure."""
    cassette_path = Path(path)
    serialized = _event_line(event)
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    with cassette_path.open("a", encoding="utf-8") as cassette_file:
        cassette_file.write(serialized)
        cassette_file.flush()
        os.fsync(cassette_file.fileno())


def save_events(path: str | Path, events: Iterable[Event]) -> None:
    """Atomically replace a cassette with the supplied events."""
    cassette_path = Path(path)
    serialized_events = [_event_line(event) for event in events]
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cassette_path.parent,
            prefix=f".{cassette_path.name}.",
            delete=False,
        ) as cassette_file:
            temporary_path = Path(cassette_file.name)
            cassette_file.writelines(serialized_events)
            cassette_file.flush()
            os.fsync(cassette_file.fileno())
        temporary_path.replace(cassette_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _event_line(event: Event) -> str:
    return strict_json_dumps(event.to_dict()) + "\n"


def recover_cassette(source: str | Path, destination: str | Path) -> RecoveryReport:
    """Validate ``source`` and atomically normalize it into another file.

    Only a malformed, unterminated final line can be discarded. A syntactically
    valid event that fails schema or field validation always fails recovery.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(f"Cassette does not exist: {source_path}")
    if _same_file(source_path, destination_path):
        raise ValueError("recovery destination must differ from source cassette")

    events: list[Event] = []
    discarded_offset: int | None = None
    discarded_bytes = 0
    source_size = source_path.stat().st_size
    with source_path.open("rb") as cassette_file:
        byte_offset = 0
        for line_number, raw_line in enumerate(cassette_file, start=1):
            is_unterminated_tail = (
                not raw_line.endswith(b"\n") and byte_offset + len(raw_line) == source_size
            )
            try:
                decoded = _decode_line(source_path, raw_line, line_number, byte_offset)
                if not decoded.strip():
                    byte_offset += len(raw_line)
                    continue
                data = _decode_json(source_path, decoded, line_number, byte_offset)
            except _RecoverableTailError:
                if not is_unterminated_tail:
                    raise
                discarded_offset = byte_offset
                discarded_bytes = len(raw_line)
                break

            try:
                event = Event.from_dict(data)
            except (TypeError, ValueError) as error:
                raise CassetteCorruptionError(
                    source_path,
                    line_number=line_number,
                    byte_offset=byte_offset,
                    reason=str(error),
                ) from error
            events.append(event)
            byte_offset += len(raw_line)

    save_events(destination_path, events)
    return RecoveryReport(
        source=source_path,
        destination=destination_path,
        events=len(events),
        discarded_offset=discarded_offset,
        discarded_bytes=discarded_bytes,
    )


def _parse_event_line(
    path: Path, raw_line: bytes, line_number: int, byte_offset: int
) -> Event | None:
    decoded = _decode_line(path, raw_line, line_number, byte_offset)
    if not decoded.strip():
        return None
    data = _decode_json(path, decoded, line_number, byte_offset)
    try:
        return Event.from_dict(data)
    except (TypeError, ValueError) as error:
        raise CassetteCorruptionError(
            path,
            line_number=line_number,
            byte_offset=byte_offset,
            reason=str(error),
        ) from error


def _decode_line(path: Path, raw_line: bytes, line_number: int, byte_offset: int) -> str:
    try:
        return raw_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _RecoverableTailError(
            path,
            line_number=line_number,
            byte_offset=byte_offset + error.start,
            reason="line is not valid UTF-8",
        ) from error


def _decode_json(path: Path, line: str, line_number: int, byte_offset: int) -> Any:
    try:
        return strict_json_loads(line)
    except json.JSONDecodeError as error:
        error_offset = len(line[: error.pos].encode("utf-8"))
        error_type = (
            _RecoverableTailError if _is_truncated_json(line, error) else CassetteCorruptionError
        )
        raise error_type(
            path,
            line_number=line_number,
            byte_offset=byte_offset + error_offset,
            reason=error.msg,
        ) from error
    except StrictJSONError as error:
        raise CassetteCorruptionError(
            path,
            line_number=line_number,
            byte_offset=byte_offset,
            reason=str(error),
        ) from error


def _is_truncated_json(line: str, error: json.JSONDecodeError) -> bool:
    if _is_ordinary_eof_truncation(line, error):
        return True
    if error.msg == "Expecting value":
        literal = _LITERAL_PREFIX.search(line)
        if literal is not None:
            token = literal.group("token")
            completed = line[: literal.start("token")] + _LITERAL_COMPLETIONS[token]
            return _candidate_is_json_prefix(completed)
    number = _INCOMPLETE_NUMBER.search(line)
    if number is not None:
        completed = line + "0"
        return _candidate_is_json_prefix(completed)
    if error.msg == "Invalid \\uXXXX escape":
        unicode_escape = _PARTIAL_UNICODE_ESCAPE.search(line)
        if unicode_escape is not None:
            completed = line + ("0" * (4 - len(unicode_escape.group("digits")))) + '"'
            return _candidate_is_json_prefix(completed)
    return False


def _is_ordinary_eof_truncation(line: str, error: json.JSONDecodeError) -> bool:
    content = line.rstrip()
    return error.pos >= len(content) or error.msg.startswith("Unterminated string")


def _candidate_is_json_prefix(candidate: str) -> bool:
    try:
        strict_json_loads(candidate)
    except json.JSONDecodeError as error:
        return _is_ordinary_eof_truncation(candidate, error)
    except StrictJSONError:
        return False
    return True


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return source.samefile(destination)
    except FileNotFoundError:
        return source.resolve(strict=False) == destination.resolve(strict=False)


def _safe_diagnostic(value: str) -> str:
    escaped = value.encode("unicode_escape", errors="backslashreplace").decode("ascii")
    return escaped[:500]


__all__ = [
    "CassetteCorruptionError",
    "RecoveryReport",
    "append_event",
    "load_events",
    "recover_cassette",
    "save_events",
]
