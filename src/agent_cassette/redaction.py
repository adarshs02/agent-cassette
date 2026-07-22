"""Recursive secret redaction for cassette payloads."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

REDACTED = "[REDACTED]"
_MAX_DEPTH = 64
_SECRET_KEY = re.compile(
    r"(?:authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")

# A hierarchical URI token: a valid scheme, ``://``, then any run of characters that
# are not whitespace or a delimiter that ends a URL in prose/markup. Sentence
# punctuation and a closing paren stay in the token and are peeled back off afterwards,
# so trailing ``)`` or ``.`` in surrounding prose is preserved byte-for-byte. ``]``/``}``
# are not peeled: they are structural in IPv6 authorities and in the ``[REDACTED]``
# marker, so peeling them would corrupt IPv6 hosts and break idempotence.
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>`\\{}|^]*")
_TRAILER_CHARS = ".,;:!?)"
_QUERY_SEPARATORS = re.compile(r"([&;])")


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
        return _scrub_uris(_BEARER.sub(f"Bearer {REDACTED}", value))
    return value


def _scrub_uris(text: str) -> str:
    """Redact userinfo passwords and secret query values in every hierarchical URI."""
    if "://" not in text:
        return text
    return _URI.sub(_scrub_uri_token, text)


def _scrub_uri_token(match: re.Match[str]) -> str:
    token = match.group(0)
    try:
        core, trailer = _split_trailer(token)
        scheme, _, rest = core.partition("://")
        cut = _first_delimiter(rest)
        authority, remainder = (rest, "") if cut is None else (rest[:cut], rest[cut:])
        return f"{scheme}://{_redact_authority(authority)}{_redact_remainder(remainder)}{trailer}"
    except Exception:
        # Fail safe: never leak the value through an exception, never render it.
        raise RedactionError("malformed connection URI could not be redacted") from None


def _split_trailer(token: str) -> tuple[str, str]:
    end = len(token)
    while end > 0 and token[end - 1] in _TRAILER_CHARS:
        end -= 1
    return token[:end], token[end:]


def _first_delimiter(rest: str) -> int | None:
    indexes = [rest.index(char) for char in "/?#" if char in rest]
    return min(indexes) if indexes else None


def _redact_authority(authority: str) -> str:
    if "@" not in authority:
        return authority  # no userinfo (or username-only handled below)
    userinfo, _, host = authority.rpartition("@")  # last '@' separates userinfo from host
    if ":" not in userinfo:
        return authority  # username-only userinfo is not a password
    username, _, _password = userinfo.partition(":")
    return f"{username}:{REDACTED}@{host}"


def _redact_remainder(remainder: str) -> str:
    # Split off the fragment first: a '?' after the '#' belongs to the fragment and is
    # preserved verbatim; only a query before any '#' is scrubbed.
    path_query, hash_sep, fragment = remainder.partition("#")
    if "?" not in path_query:
        return remainder
    head, _, query = path_query.partition("?")
    return f"{head}?{_redact_query(query)}{hash_sep}{fragment}"


def _redact_query(query: str) -> str:
    return "".join(
        part if index % 2 else _redact_query_param(part)
        for index, part in enumerate(_QUERY_SEPARATORS.split(query))
    )


def _redact_query_param(part: str) -> str:
    key, sep, value = part.partition("=")
    if not sep or not value:
        return part  # a flag or an already-blank value cannot leak a secret
    return f"{key}={REDACTED}" if _SECRET_KEY.search(unquote(key)) else part


def _enter_container(value: object, active: set[int]) -> int:
    value_id = id(value)
    if value_id in active:
        raise RedactionError("cyclic value cannot be redacted")
    active.add(value_id)
    return value_id
