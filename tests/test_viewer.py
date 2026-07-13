from __future__ import annotations

from agent_cassette.events import Event, EventType
from agent_cassette.viewer import render_viewer, write_viewer


def event(identifier: str, payload) -> Event:
    return Event(
        id=identifier,
        timestamp="2026-01-01T00:00:00Z",
        type=EventType.CUSTOM,
        name='unsafe </summary><script>alert("x")</script>',
        input=payload,
    )


def test_viewer_is_standalone_script_free_and_escapes_payloads():
    rendered = render_viewer([event("1", {"text": "</pre><script>bad()</script>"})])

    assert "Content-Security-Policy" in rendered
    assert "default-src &#x27;none&#x27;" in rendered
    assert "<script" not in rendered.lower()
    assert "https://" not in rendered
    assert "http://" not in rendered
    assert "&lt;/pre&gt;&lt;script&gt;" in rendered
    assert "&lt;/summary&gt;&lt;script&gt;" in rendered


def test_viewer_redacts_secrets_by_default_and_allows_explicit_opt_out():
    sensitive = event("1", {"api_key": "sk-secret", "safe": "visible"})

    safe = render_viewer([sensitive])
    unsafe = render_viewer([sensitive], redact_secrets=False)

    assert "sk-secret" not in safe
    assert "[REDACTED]" in safe
    assert "sk-secret" in unsafe


def test_viewer_enforces_event_and_payload_limits():
    rendered = render_viewer(
        [event("1", {"value": "x" * 100}), event("2", {"value": "visible"})],
        max_events=1,
        max_payload_chars=30,
    )

    assert "1 event(s) omitted by limit" in rendered
    assert "payload truncated" in rendered
    assert "visible" not in rendered


def test_write_viewer_creates_atomic_standalone_file(tmp_path):
    destination = tmp_path / "nested" / "viewer.html"

    write_viewer(destination, [event("1", {"safe": True})], title="A < B")

    contents = destination.read_text(encoding="utf-8")
    assert contents.startswith("<!doctype html>")
    assert "A &lt; B" in contents
    assert not list(destination.parent.glob("*.tmp"))
