import json

import pytest

from agent_cassette import Cassette, EventType
from agent_cassette.redaction import REDACTED, RedactionError, redact


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
