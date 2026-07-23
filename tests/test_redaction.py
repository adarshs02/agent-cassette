import html
import json

import pytest

import agent_cassette.redaction as redaction_module
from agent_cassette import (
    Cassette,
    EventType,
    InjectionRule,
    Return,
    check_trajectory,
    tool_called,
    wrap_tool,
)
from agent_cassette.events import Event
from agent_cassette.redaction import REDACTED, RedactionError, redact
from agent_cassette.viewer import render_viewer, write_viewer


def test_redacts_nested_secrets_and_bearer_tokens():
    value = {
        "headers": {"Authorization": "Bearer super-secret"},
        "api_key": "sk-live",
        "safe": ["Bearer abc123", "visible"],
    }

    assert redact(value) == {
        "headers": {"Authorization": REDACTED},
        "api_key": REDACTED,
        "safe": [f"Bearer {REDACTED}", "visible"],
    }


def test_recording_redacts_before_writing(tmp_path):
    path = tmp_path / "safe.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(EventType.TOOL_CALL, "request", input={"access_token": "secret"})

    raw = path.read_text()
    assert "secret" not in raw
    assert json.loads(raw)["input"]["access_token"] == REDACTED


def test_redacts_bare_token_field():
    assert redact({"token": "credential"}) == {"token": REDACTED}


@pytest.mark.parametrize("container_type", [dict, list, tuple])
def test_redaction_rejects_cycles_without_exposing_values(container_type):
    secret = "never-print-this-secret"
    if container_type is dict:
        dictionary: dict[str, object] = {"safe": secret}
        dictionary["cycle"] = dictionary
        value: object = dictionary
    elif container_type is list:
        items: list[object] = [secret]
        items.append(items)
        value = items
    else:
        child: list[object] = [secret]
        tuple_value: tuple[object, ...] = (child,)
        child.append(tuple_value)
        value = tuple_value

    with pytest.raises(RedactionError) as raised:
        redact(value)

    assert str(raised.value) == "cyclic value cannot be redacted"
    assert secret not in str(raised.value)


def test_redaction_depth_bound_accepts_64_levels_and_rejects_65():
    accepted: object = {"password": "secret"}
    for _ in range(63):
        accepted = [accepted]

    result = redact(accepted)
    for _ in range(63):
        result = result[0]
    assert result == {"password": REDACTED}

    rejected: object = {"password": "secret"}
    for _ in range(64):
        rejected = [rejected]

    with pytest.raises(RedactionError, match=r"^maximum redaction depth 64 exceeded$"):
        redact(rejected)


def test_redaction_allows_shared_acyclic_aliases():
    shared = {"password": "secret", "safe": "Bearer token"}

    result = redact({"left": shared, "right": shared})

    expected = {"password": REDACTED, "safe": f"Bearer {REDACTED}"}
    assert result == {"left": expected, "right": expected}


def test_non_string_keys_do_not_invoke_user_string_methods():
    class HostileKey:
        def __str__(self):
            raise AssertionError("__str__ must not run")

        def __repr__(self):
            raise AssertionError("__repr__ must not run")

    key = HostileKey()

    result = redact({key: {"password": "secret"}})

    assert result[key] == {"password": REDACTED}


def test_redaction_error_does_not_render_hostile_values():
    class HostileValue:
        def __str__(self):
            raise AssertionError("__str__ must not run")

        def __repr__(self):
            raise AssertionError("__repr__ must not run")

    value: list[object] = [HostileValue()]
    value.append(value)

    with pytest.raises(RedactionError, match=r"^cyclic value cannot be redacted$"):
        redact(value)


# --------------------------------------------------------------------------- #
# Connection-URI userinfo passwords
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        # every listed database/broker scheme
        ("postgres://user:pw@db/app", "postgres://user:[REDACTED]@db/app"),
        ("postgresql://user:pw@db:5432/app", "postgresql://user:[REDACTED]@db:5432/app"),
        ("mysql://root:pw@localhost/app", "mysql://root:[REDACTED]@localhost/app"),
        ("mariadb://root:pw@localhost/app", "mariadb://root:[REDACTED]@localhost/app"),
        ("redis://:pw@cache:6379/0", "redis://:[REDACTED]@cache:6379/0"),
        ("rediss://user:pw@cache:6380/0", "rediss://user:[REDACTED]@cache:6380/0"),
        ("mongodb://admin:pw@m1/db", "mongodb://admin:[REDACTED]@m1/db"),
        ("mongodb+srv://admin:pw@cluster/db", "mongodb+srv://admin:[REDACTED]@cluster/db"),
        ("amqp://guest:pw@broker/vhost", "amqp://guest:[REDACTED]@broker/vhost"),
        ("amqps://guest:pw@broker/vhost", "amqps://guest:[REDACTED]@broker/vhost"),
        # a custom valid scheme
        ("myproto://u:pw@host/x", "myproto://u:[REDACTED]@host/x"),
        # last-@ so an unescaped '@' inside the password is still removed
        ("postgres://user:p@ssw0rd@db.internal/app", "postgres://user:[REDACTED]@db.internal/app"),
        # percent-encoded password
        ("postgres://u:p%40ss%3Aword@db/app", "postgres://u:[REDACTED]@db/app"),
        # IPv6 host with port
        ("redis://:pw@[::1]:6379/0", "redis://:[REDACTED]@[::1]:6379/0"),
        # empty username preserved
        ("redis://:pw@host/0", "redis://:[REDACTED]@host/0"),
    ],
)
def test_uri_userinfo_password_is_redacted(raw, expected):
    assert redact(raw) == expected
    assert "pw" not in redact(raw).replace("[REDACTED]", "")


def test_uri_embedded_in_prose_and_multiple_urls():
    text = "primary postgres://a:secret1@h1/x, backup mysql://b:secret2@h2/y."
    result = redact(text)
    assert result == "primary postgres://a:[REDACTED]@h1/x, backup mysql://b:[REDACTED]@h2/y."
    assert "secret1" not in result and "secret2" not in result


def test_uri_followed_by_bracket_and_period():
    assert redact("(see amqps://guest:guestpw@broker/vhost).") == (
        "(see amqps://guest:[REDACTED]@broker/vhost)."
    )


# --------------------------------------------------------------------------- #
# Secret query parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://h/p?password=pw", "https://h/p?password=[REDACTED]"),
        ("https://h/p?secret=pw", "https://h/p?secret=[REDACTED]"),
        ("https://h/p?token=pw", "https://h/p?token=[REDACTED]"),
        ("https://h/p?access_token=pw", "https://h/p?access_token=[REDACTED]"),
        ("https://h/p?refresh-token=pw", "https://h/p?refresh-token=[REDACTED]"),
        ("https://h/p?api_key=pw", "https://h/p?api_key=[REDACTED]"),
        ("https://h/p?api-key=pw", "https://h/p?api-key=[REDACTED]"),
        ("https://h/p?API_KEY=pw", "https://h/p?API_KEY=[REDACTED]"),
        # mixed secret + non-secret, order and separators preserved
        (
            "https://h/p?page=2&token=pw&sort=asc",
            "https://h/p?page=2&token=[REDACTED]&sort=asc",
        ),
        ("https://h/p?a=1;token=pw", "https://h/p?a=1;token=[REDACTED]"),
        # blank value preserved (nothing to leak)
        ("https://h/p?token=", "https://h/p?token="),
        # fragment preserved
        ("https://h/p?token=pw#section", "https://h/p?token=[REDACTED]#section"),
        # userinfo password AND query secret together
        (
            "postgres://u:dbpw@h/db?password=qpw",
            "postgres://u:[REDACTED]@h/db?password=[REDACTED]",
        ),
        # percent-encoded secret value entirely replaced
        ("https://h/p?token=ab%26cd", "https://h/p?token=[REDACTED]"),
        # percent-encoded secret KEY: decode to classify, preserve the encoded spelling
        ("https://h/p?api%5Fkey=encoded-secret", "https://h/p?api%5Fkey=[REDACTED]"),
    ],
)
def test_secret_query_values_are_redacted(raw, expected):
    result = redact(raw)
    assert result == expected
    assert "pw" not in result.replace("[REDACTED]", "")


def test_uri_redaction_is_idempotent():
    raw = "postgres://user:p@ssw0rd@db/app?password=x&token=y#f"
    once = redact(raw)
    assert redact(once) == once
    assert "p@ssw0rd" not in once and "=x" not in once and "=y" not in once


# --------------------------------------------------------------------------- #
# Every URI in one non-whitespace run is scanned
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "postgres://a:s1@h/x,mysql://b:s2@j/y",
            "postgres://a:[REDACTED]@h/x,mysql://b:[REDACTED]@j/y",
        ),
        (
            "postgres://a:s1@h,mysql://b:s2@j",
            "postgres://a:[REDACTED]@h,mysql://b:[REDACTED]@j",
        ),
        (
            "https://h/p?next=postgres://u:pw@db",
            "https://h/p?next=postgres://u:[REDACTED]@db",
        ),
        (
            "https://h/p?next=postgres://u:pw@db,redis://:rpw@cache/0",
            "https://h/p?next=postgres://u:[REDACTED]@db,redis://:[REDACTED]@cache/0",
        ),
        # semicolon adjacency and three URLs in one run
        (
            "postgres://a:s1@h;mysql://b:s2@j;redis://:s3@k/0",
            "postgres://a:[REDACTED]@h;mysql://b:[REDACTED]@j;redis://:[REDACTED]@k/0",
        ),
    ],
)
def test_every_uri_occurrence_is_redacted(raw, expected):
    result = redact(raw)
    assert result == expected
    for leaked in ("s1", "s2", "s3", ":pw@", ":rpw@"):
        assert leaked not in result
    assert redact(result) == result  # idempotent


def test_whitespace_separated_prose_still_works():
    assert redact("primary postgres://a:s1@h and backup mysql://b:s2@j done") == (
        "primary postgres://a:[REDACTED]@h and backup mysql://b:[REDACTED]@j done"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        # a secret query value that itself contains '://' is replaced whole
        ("https://h/p?password=abc://def", "https://h/p?password=[REDACTED]"),
        ("https://h/p?token=http://evil.com/cb", "https://h/p?token=[REDACTED]"),
        # an outer secret query param after a nested URI is still redacted, and the
        # nested URI's userinfo is scrubbed too
        (
            "https://h/p?a=1&next=redis://:pw@c&secret=xyz",
            "https://h/p?a=1&next=redis://:[REDACTED]@c&secret=[REDACTED]",
        ),
    ],
)
def test_nested_and_secret_query_values(raw, expected):
    result = redact(raw)
    assert result == expected
    assert redact(result) == result
    for leaked in ("abc://def", "http://evil.com/cb", ":pw@", "xyz"):
        assert leaked not in result


@pytest.mark.parametrize(
    "hostile",
    [
        "postgres://" + "a" * 500_000,  # long scheme-char run, no '://'
        "a" * 60_000 + ":" + "a" * 60_000 + "://x",  # colon-split token (no regex backtracking)
    ],
)
def test_uri_scan_is_linear_on_hostile_input(hostile):
    import time

    start = time.perf_counter()
    redact(hostile)
    assert time.perf_counter() - start < 2.0


def test_deeply_nested_query_uris_fail_closed_with_depth_error():
    # A pathologically deep chain of URIs nested through query values must fail closed
    # with the same bounded, secret-free depth error as deep containers — never a
    # RecursionError or a leak.
    deep = "a://h?k=" * 150 + "b://:SEKRET@h"
    with pytest.raises(RedactionError, match=r"^maximum redaction depth 64 exceeded$"):
        redact(deep)


def test_many_adjacent_uris_scrub_iteratively_without_recursion_error():
    # Hundreds of comma-adjacent URLs must all redact (walked iteratively, not recursively,
    # so the Python recursion limit is never reached).
    raw = ",".join(f"postgres://u{i}:s{i}@h{i}/x" for i in range(400))
    result = redact(raw)
    assert result.count(REDACTED) == 400
    assert ":s0@" not in result and ":s399@" not in result


# --------------------------------------------------------------------------- #
# Unmatched surrounding brackets are preserved
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        # unmatched trailing ']' from prose is kept
        ("[https://h/p?token=secret]", "[https://h/p?token=[REDACTED]]"),
        # bracketed userinfo password with IPv6 host: balanced ']' stays inside
        ("[redis://:pw@[::1]:6379/0]", "[redis://:[REDACTED]@[::1]:6379/0]"),
        # adjacent bracketed URLs each scrubbed
        (
            "[postgres://a:s1@h][mysql://b:s2@j]",
            "[postgres://a:[REDACTED]@h][mysql://b:[REDACTED]@j]",
        ),
        # angle/paren/quote prose already terminates the token; still preserved
        ("<https://h?token=secret>", "<https://h?token=[REDACTED]>"),
        ("(redis://:pw@cache/0)", "(redis://:[REDACTED]@cache/0)"),
    ],
)
def test_unmatched_brackets_preserved(raw, expected):
    result = redact(raw)
    assert result == expected
    assert redact(result) == result  # idempotence, including already-bracketed


# --------------------------------------------------------------------------- #
# Hybrid injection + opt-out + fail-safe
# --------------------------------------------------------------------------- #


def test_hybrid_injection_scrubs_uri(tmp_path):
    source = tmp_path / "src.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass  # empty baseline; the call is injected

    rule = InjectionRule(Return({"dsn": _DSN}), type="tool_call", name="connect")
    with Cassette.fork(source, output, injections=(rule,)) as hybrid:
        hybrid.call(EventType.TOOL_CALL, "connect", {"dsn": _DSN}, lambda: {"unused": True})
    # the injected output and the recorded input are both scrubbed before persistence
    _artifact_has_no_secret(output.read_text(encoding="utf-8"))


def test_hybrid_redact_secrets_false_preserves_uri_secret(tmp_path):
    source = tmp_path / "src.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source):
        pass

    rule = InjectionRule(Return({"dsn": _DSN}), type="tool_call", name="connect")
    with Cassette.fork(source, output, injections=(rule,), redact_secrets=False) as hybrid:
        hybrid.call(EventType.TOOL_CALL, "connect", {"dsn": _DSN}, lambda: {"unused": True})
    assert "p@ssw0rd" in output.read_text(encoding="utf-8")


def test_uri_helper_failure_raises_generic_error_and_persists_nothing(tmp_path, monkeypatch):
    def boom(_authority):
        raise RuntimeError("SECRET-BEARING-INTERNAL-DETAIL")

    monkeypatch.setattr(redaction_module, "_redact_authority", boom)

    with pytest.raises(RedactionError) as raised:
        redact("postgres://user:p@ssw0rd@db/app")
    message = str(raised.value)
    assert "p@ssw0rd" not in message
    assert "SECRET-BEARING-INTERNAL-DETAIL" not in message
    assert message == "malformed connection URI could not be redacted"

    path = tmp_path / "run.jsonl"
    with pytest.raises(RedactionError):
        with Cassette.record(path) as cassette:
            cassette.add(EventType.TOOL_CALL, "connect", input={"dsn": "postgres://u:p@ssw0rd@h/x"})
    if path.exists():
        assert "p@ssw0rd" not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# False positives — must NOT be altered
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "unchanged",
    [
        "https://example.com/path?page=2&sort=asc",  # ordinary URL, no secret
        "alice@example.com",  # bare email
        "https://user@host/path",  # username-only userinfo
        "https://h/p/user:pass@thing",  # ':@' only in the path
        "https://h/p?redirect=user@example.com",  # '@' in a non-secret query value
        "https://h/p#user:tok@frag",  # ':@' only in the fragment
        "redis://cache/db#note?token=kept",  # '?' inside the fragment is not a query
        "/home/user:group/a@b/file.txt",  # filesystem path, no scheme
        "mailto:alice@example.com",  # non-hierarchical scheme
        "run at 12:30 and see user@host later",  # prose colons/at-signs
    ],
)
def test_false_positives_are_not_redacted(unchanged):
    assert redact(unchanged) == unchanged


def test_uri_redaction_recurses_and_preserves_shapes_without_rendering():
    class Hostile:
        def __str__(self):
            raise AssertionError("__str__ must not run")

        def __repr__(self):
            raise AssertionError("__repr__ must not run")

    hostile = Hostile()
    shared = {"dsn": "postgres://u:pw@h/db"}
    value = {
        "a": shared,
        "b": shared,  # acyclic alias
        "list": ["mysql://x:pw@h/y", 1, hostile],
        "tuple": ("redis://:pw@h/0",),
    }
    result = redact(value)
    assert result["a"] == {"dsn": "postgres://u:[REDACTED]@h/db"}
    assert result["b"] == {"dsn": "postgres://u:[REDACTED]@h/db"}
    assert result["list"][0] == "mysql://x:[REDACTED]@h/y"
    assert result["list"][2] is hostile  # hostile object passed through untouched
    assert isinstance(result["tuple"], tuple)
    assert result["tuple"][0] == "redis://:[REDACTED]@h/0"


# --------------------------------------------------------------------------- #
# End-to-end: Recorder, replay matching, viewer, opt-out
# --------------------------------------------------------------------------- #

_DSN = "postgres://svc:p@ssw0rd@db.internal:5432/app?token=qT0ken%26ABC"
_ENCODED_SECRETS = ("p@ssw0rd", "qT0ken%26ABC", "qT0ken&ABC")


def _artifact_has_no_secret(text: str) -> None:
    for secret in _ENCODED_SECRETS:
        assert secret not in text, secret
        assert html.escape(secret) not in text, secret


def test_recorder_scrubs_uri_in_input_output_metadata(tmp_path):
    path = tmp_path / "run.jsonl"
    with Cassette.record(path) as cassette:
        cassette.add(
            EventType.TOOL_CALL,
            "connect",
            input={"dsn": _DSN},
            output={"echo": f"connected to {_DSN}"},
            metadata={"note": _DSN},
        )
    raw = path.read_text(encoding="utf-8")
    _artifact_has_no_secret(raw)
    stored = json.loads(raw)
    assert (
        stored["input"]["dsn"] == "postgres://svc:[REDACTED]@db.internal:5432/app?token=[REDACTED]"
    )


def test_wrapped_tool_replay_matches_original_live_uri(tmp_path):
    path = tmp_path / "run.jsonl"

    def connect(dsn):
        return {"ok": True}

    with Cassette.record(path) as cassette:
        wrap_tool(connect, cassette, name="connect")(_DSN)
    _artifact_has_no_secret(path.read_text(encoding="utf-8"))

    ran: list[str] = []

    def forbidden(dsn):
        ran.append("ran")
        raise AssertionError("live tool ran during replay")

    with Cassette.replay(path) as replayer:
        # replay is driven with the ORIGINAL live URI; normalize_input redacts it to the
        # same stored form, so the match succeeds without ever storing the secret.
        assert wrap_tool(forbidden, replayer, name="connect")(_DSN) == {"ok": True}
        assert replayer.remaining == 0
        assert check_trajectory(
            replayer.consumed_events,
            tool_called("connect", with_input={"args": [_DSN]}, match="subset"),
        ).passed
    assert ran == []


def test_hybrid_redacts_replayed_prefix_and_live_suffix(tmp_path):
    source = tmp_path / "src.jsonl"
    output = tmp_path / "fork.jsonl"
    with Cassette.record(source) as cassette:
        cassette.add(EventType.TOOL_CALL, "prefix", input={"dsn": _DSN}, output={"ok": 1})

    with Cassette.fork(source, output, at=1) as hybrid:
        # prefix replays by matching the original live URI (normalized to the stored form)
        assert hybrid.call(EventType.TOOL_CALL, "prefix", {"dsn": _DSN}, lambda: {"ok": 1}) == {
            "ok": 1
        }
        # live suffix records a fresh URI secret, which is redacted before append
        hybrid.call(
            EventType.TOOL_CALL,
            "suffix",
            {"dsn": _DSN},
            lambda: {"echo": f"live {_DSN}"},
        )
    _artifact_has_no_secret(output.read_text(encoding="utf-8"))


def test_viewer_scrubs_unredacted_event(tmp_path):
    event = Event(
        id="e1",
        timestamp="2026-01-01T00:00:00Z",
        type=EventType.TOOL_CALL,
        name="connect",
        input={"dsn": _DSN},
    )
    html_text = render_viewer([event])  # default redaction on
    _artifact_has_no_secret(html_text)

    destination = tmp_path / "viewer.html"
    write_viewer(destination, [event])
    _artifact_has_no_secret(destination.read_text(encoding="utf-8"))


def test_redact_secrets_false_preserves_uri_secret(tmp_path):
    path = tmp_path / "run.jsonl"
    with Cassette.record(path, redact_secrets=False) as cassette:
        cassette.add(EventType.TOOL_CALL, "connect", input={"dsn": _DSN})
    assert "p@ssw0rd" in path.read_text(encoding="utf-8")

    event = Event(
        id="e1",
        timestamp="2026-01-01T00:00:00Z",
        type=EventType.TOOL_CALL,
        name="connect",
        input={"dsn": _DSN},
    )
    assert "p@ssw0rd" in render_viewer([event], redact_secrets=False)


def test_existing_secret_key_and_bearer_behavior_stable():
    assert redact({"Authorization": "Bearer abc123", "api_key": "sk-live"}) == {
        "Authorization": REDACTED,
        "api_key": REDACTED,
    }
    assert redact("prefix Bearer tok.en-value suffix") == f"prefix Bearer {REDACTED} suffix"
