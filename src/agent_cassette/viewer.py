"""Secure, dependency-free standalone trajectory viewer."""

from __future__ import annotations

import html
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_cassette.events import Event
from agent_cassette.redaction import redact
from agent_cassette.storage import load_events

_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"


def _truncate(value: Any, limit: int) -> tuple[Any, bool]:
    serialized = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    if len(serialized) <= limit:
        return value, False
    return serialized[:limit] + f"… [truncated {len(serialized) - limit} chars]", True


def render_viewer(
    trajectory: Iterable[Event] | str | Path,
    *,
    redact_secrets: bool = True,
    max_events: int = 1_000,
    max_payload_chars: int = 20_000,
    title: str = "Agent Cassette Viewer",
) -> str:
    """Render events as a script-free HTML document with safe bounded payloads."""
    if max_events < 0 or max_payload_chars < 1:
        raise ValueError("viewer limits must be positive")
    events = load_events(trajectory) if isinstance(trajectory, (str, Path)) else list(trajectory)
    visible = events[:max_events]
    rows: list[str] = []
    for index, event in enumerate(visible, start=1):
        data: Any = event.to_dict()
        if redact_secrets:
            data = redact(data)
        data, truncated = _truncate(data, max_payload_chars)
        payload = json.dumps(data, indent=2, sort_keys=True, default=str, ensure_ascii=False)
        badge = " <strong>(payload truncated)</strong>" if truncated else ""
        summary = f"{index}. {event.type.value}: {event.name}"
        rows.append(
            f"<details><summary>{html.escape(summary)}{badge}</summary>"
            f"<pre>{html.escape(payload)}</pre></details>"
        )
    omitted = len(events) - len(visible)
    notice = f"<p><strong>{omitted} event(s) omitted by limit.</strong></p>" if omitted else ""
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{html.escape(_CSP, quote=True)}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#202124}}
details{{border:1px solid #ccc;border-radius:6px;margin:.75rem 0;padding:.75rem}}
summary{{cursor:pointer;font-weight:600}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem}}
strong{{color:#9c2f20}}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<p>{len(events)} event(s); redaction {"enabled" if redact_secrets else "disabled"}.</p>
{notice}{"".join(rows)}
</body>
</html>
"""


def write_viewer(
    path: str | Path, trajectory: Iterable[Event] | str | Path, **options: Any
) -> None:
    """Atomically write a rendered standalone viewer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = render_viewer(trajectory, **options)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
