"""Recursive secret redaction for cassette payloads."""

from __future__ import annotations

import re
import string
from typing import Any
from urllib.parse import unquote

REDACTED = "[REDACTED]"
_MAX_DEPTH = 64
_SECRET_KEY = re.compile(
    r"(?:authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")

# Hierarchical URIs are located by a linear ``str.find("://")`` scan (no regex, so no
# backtracking on hostile input) and each URI's extent is found by a single bracket- and
# adjacency-aware forward pass (:func:`_uri_end`): a URI ends at whitespace, a hard
# delimiter, an *unmatched* closing ``)``/``]`` (surrounding prose), or a ``,``/``;`` that
# — after optional prose brackets — precedes the next scheme (an adjacent URL). Balanced
# ``[...]`` (IPv6 authority) and ``(...)`` stay inside the URI. Every scheme occurrence in
# a run is therefore visited without swallowing a neighbour or a closing bracket.
_SCHEME_CHARS = frozenset(string.ascii_letters + string.digits + "+.-")
_SCHEME_START_CHARS = frozenset(string.ascii_letters)
_URI_DELIMS = frozenset("\"'<>`\\{}|^")  # hard delimiters; whitespace also ends a URI
_ADJACENCY_DELIMS = frozenset(",;&")  # separate one URI from an adjacent one before a scheme
_ADJACENCY_PROSE = frozenset(",;&()[]")  # skipped (with brackets) when testing for an adjacent URI
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


def _scrub_uris(text: str, depth: int = 0) -> str:
    """Redact userinfo passwords and secret query values in every hierarchical URI."""
    if "://" not in text:
        return text
    if depth >= _MAX_DEPTH:
        raise RedactionError(f"maximum redaction depth {_MAX_DEPTH} exceeded")
    try:
        out: list[str] = []
        pos = 0
        while True:
            marker = text.find("://", pos)
            if marker == -1:
                out.append(text[pos:])
                break
            start = _scheme_start_for_marker(text, marker, pos)
            if start is None:  # a '://' with no valid scheme in front of it
                out.append(text[pos : marker + 3])
                pos = marker + 3
                continue
            end = _uri_end(text, start)  # bracket/adjacency-aware forward scan
            out.append(text[pos:start])  # prose/separator before this URI
            out.append(_redact_uri(text[start:end], depth))
            pos = end
        return "".join(out)
    except RedactionError:
        raise
    except Exception:
        # Fail safe: never leak the value through an exception, never render it.
        raise RedactionError("malformed connection URI could not be redacted") from None


def _uri_end(text: str, start: int) -> int:
    # One linear pass: a URI ends at whitespace, a hard delimiter, an unmatched closing
    # ')'/']' (surrounding prose), or a ','/';'/'&' that (after optional prose brackets)
    # precedes the next scheme. Balanced '[...]'/'(...)' stay inside the URI. Each
    # adjacency-prose run is consumed once, so a long delimiter run is linear, not O(n^2).
    index = start
    length = len(text)
    square = 0
    paren = 0
    while index < length:
        char = text[index]
        if char.isspace() or char in _URI_DELIMS:
            break
        if char in _ADJACENCY_PROSE:
            # Consume this whole adjacency-prose run (',' ';' '&' '(' ')' '[' ']') exactly
            # once, recording bracket balance, the first ','/';'/'&', and the first
            # unmatched closer. Each character is touched a constant number of times and
            # _scheme_at is called at most once per run -- so a long delimiter run stays
            # linear. Do NOT return early on the closer: the URI breaks at whichever comes
            # first, an adjacency delimiter (when a scheme follows the run) or the closer.
            cursor = index
            first_delim = -1
            first_unmatched = -1
            run_square, run_paren = square, paren
            while cursor < length and text[cursor] in _ADJACENCY_PROSE:
                run_char = text[cursor]
                if run_char in _ADJACENCY_DELIMS:
                    if first_delim < 0:
                        first_delim = cursor
                elif run_char == "[":
                    run_square += 1
                elif run_char == "]":
                    if run_square == 0:
                        if first_unmatched < 0:
                            first_unmatched = cursor
                    else:
                        run_square -= 1
                elif run_char == "(":
                    run_paren += 1
                elif run_char == ")":
                    if run_paren == 0:
                        if first_unmatched < 0:
                            first_unmatched = cursor
                    else:
                        run_paren -= 1
                cursor += 1
            # ``cursor`` is the run end; a scheme reached by skipping the whole prose run
            # (including any unmatched closer) makes an earlier delimiter an adjacency split.
            breaks = []
            if first_unmatched >= 0:
                breaks.append(first_unmatched)
            if first_delim >= 0 and _scheme_at(text, cursor):
                breaks.append(first_delim)
            if breaks:
                return min(breaks)
            square, paren = run_square, run_paren
            index = cursor  # the whole run is URI body; jump past it
            continue
        index += 1
    return index


def _scheme_at(text: str, index: int) -> bool:
    if index >= len(text) or text[index] not in _SCHEME_START_CHARS:
        return False
    cursor = index + 1
    length = len(text)
    while cursor < length and text[cursor] in _SCHEME_CHARS:
        cursor += 1
    return text[cursor : cursor + 3] == "://"


def _redact_uri(uri: str, depth: int) -> str:
    scheme, _, rest = uri.partition("://")
    # The authority ends at the first '/', '?', '#', or the start of the next scheme, so a
    # scheme embedded in the path region (e.g. '...&db2=postgres://…' with no query) is
    # never severed across its '://'; the remainder is scrubbed recursively.
    cut = _first_of(rest, "/?#")
    next_scheme = _next_scheme_start(rest)
    if next_scheme is not None:
        cut = next_scheme if cut is None else min(cut, next_scheme)
    authority = rest if cut is None else rest[:cut]
    remainder = "" if cut is None else rest[cut:]
    return f"{scheme}://{_redact_authority(authority)}{_redact_remainder(remainder, depth)}"


def _next_scheme_start(rest: str) -> int | None:
    # Linear ``str.find("://")`` then walk back to the leading scheme letter (no regex).
    search = 0
    while True:
        marker = rest.find("://", search)
        if marker == -1:
            return None
        begin = marker
        while begin > 0 and rest[begin - 1] in _SCHEME_CHARS:
            begin -= 1
        while begin < marker and rest[begin] not in _SCHEME_START_CHARS:
            begin += 1
        if begin < marker:
            return begin
        search = marker + 3


def _first_of(text: str, chars: str) -> int | None:
    indexes = [text.index(char) for char in chars if char in text]
    return min(indexes) if indexes else None


def _scheme_start_for_marker(text: str, marker: int, floor: int) -> int | None:
    # Walk back from a ``://`` over scheme characters to the leading letter, not before
    # ``floor`` (already-emitted text). Returns None if no valid scheme precedes ``://``.
    begin = marker
    while begin > floor and text[begin - 1] in _SCHEME_CHARS:
        begin -= 1
    while begin < marker and text[begin] not in _SCHEME_START_CHARS:
        begin += 1
    return begin if begin < marker else None


def _redact_authority(authority: str) -> str:
    if "@" not in authority:
        return authority  # no userinfo (or username-only handled below)
    userinfo, _, host = authority.rpartition("@")  # last '@' separates userinfo from host
    if ":" not in userinfo:
        return authority  # username-only userinfo is not a password
    username, _, _password = userinfo.partition(":")
    return f"{username}:{REDACTED}@{host}"


def _redact_remainder(remainder: str, depth: int) -> str:
    # Split off the fragment first (a '?' after the '#' belongs to the fragment). The path
    # and the fragment are scrubbed recursively for nested URIs but otherwise preserved
    # byte-for-byte; only a query before any '#' has its secret parameters redacted.
    path_query, hash_sep, fragment = remainder.partition("#")
    scrubbed_fragment = _scrub_uris(fragment, depth + 1) if hash_sep else fragment
    if "?" in path_query:
        head, _, query = path_query.partition("?")
        return (
            f"{_scrub_uris(head, depth + 1)}?{_redact_query(query, depth)}"
            f"{hash_sep}{scrubbed_fragment}"
        )
    return f"{_scrub_uris(path_query, depth + 1)}{hash_sep}{scrubbed_fragment}"


def _redact_query(query: str, depth: int) -> str:
    return "".join(
        part if index % 2 else _redact_query_param(part, depth)
        for index, part in enumerate(_QUERY_SEPARATORS.split(query))
    )


def _redact_query_param(part: str, depth: int) -> str:
    key, sep, value = part.partition("=")
    if not sep:
        return _scrub_uris(part, depth + 1)  # a flag segment may still hold a nested URI
    if not value:
        return part  # an already-blank value cannot leak a secret
    if _SECRET_KEY.search(unquote(key)):
        return f"{key}={REDACTED}"  # secret value replaced whole, even if it contains '://'
    # non-secret value: scrub any URI nested inside it (depth-bounded)
    return f"{key}={_scrub_uris(value, depth + 1)}"


def _enter_container(value: object, active: set[int]) -> int:
    value_id = id(value)
    if value_id in active:
        raise RedactionError("cyclic value cannot be redacted")
    active.add(value_id)
    return value_id
