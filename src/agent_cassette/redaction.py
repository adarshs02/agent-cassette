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

# A hierarchical URI token is a valid scheme (letter then ``[A-Za-z0-9+.-]``), ``://``,
# then a run of characters that are not whitespace or a delimiter that ends a URL in
# prose/markup. Tokens are located by a linear ``str.find("://")`` scan (no regex, so no
# backtracking on hostile input); a run holding several adjacent URIs is walked
# iteratively, and only a URI nested inside a non-secret query value recurses.
_SCHEME_CHARS = frozenset(string.ascii_letters + string.digits + "+.-")
_SCHEME_START_CHARS = frozenset(string.ascii_letters)
_URI_DELIMS = frozenset("\"'<>`\\{}|^")  # end a URI token; whitespace also ends it
# Sentence punctuation and a closing paren are peeled off the token and re-appended, so
# trailing prose is preserved byte-for-byte. A ``]`` is peeled only when *unmatched* (a
# surrounding prose bracket): a balanced ``]`` belonging to an IPv6 authority (``[::1]``)
# or the ``[REDACTED]`` marker stays inside the token, so IPv6 hosts and idempotence hold.
# (``{``/``}`` never reach here — they terminate the token at the match level.)
_TRAILER_CHARS = ".,;:!?)"
_UNMATCHED_TRAILERS = {"]": "["}
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
            end = _run_end(text, start)  # linear forward scan to the URI's end
            out.append(text[pos:start])  # prose before this URI run
            out.append(_scrub_run(text[start:end], depth))
            pos = end
        return "".join(out)
    except RedactionError:
        raise
    except Exception:
        # Fail safe: never leak the value through an exception, never render it.
        raise RedactionError("malformed connection URI could not be redacted") from None


def _run_end(text: str, start: int) -> int:
    end = start
    length = len(text)
    while end < length and not (text[end].isspace() or text[end] in _URI_DELIMS):
        end += 1
    return end


def _scrub_run(run: str, depth: int) -> str:
    """Redact one maximal non-whitespace run, which may hold several adjacent URIs.

    Adjacent URIs are walked iteratively (no per-URI recursion), so a run with hundreds
    of comma/semicolon-separated URLs stays shallow; only a URI genuinely nested inside a
    non-secret query value recurses, and that nesting is bounded by ``_MAX_DEPTH``.
    """
    core, trailer = _split_trailer(run)
    out: list[str] = []
    pos = 0
    while pos < len(core):
        marker = core.find("://", pos)
        if marker == -1:
            out.append(core[pos:])
            break
        start = _scheme_start_for_marker(core, marker, pos)
        if start is None:  # a '://' with no valid scheme in front of it
            out.append(core[pos : marker + 3])
            pos = marker + 3
            continue
        out.append(core[pos:start])  # prose/separator before this URI
        piece, pos = _scrub_single(core, start, marker, depth)
        out.append(piece)
    return "".join(out) + trailer


def _scrub_single(core: str, start: int, marker: int, depth: int) -> tuple[str, int]:
    """Redact one URI starting at ``start`` (its ``://`` at ``marker``); return the
    redacted text and the index in ``core`` where scanning should resume."""
    scheme = core[start:marker]
    rest = core[marker + 3 :]
    first_delim = _first_of(rest, "/?#")
    query_at = rest.find("?")
    next_scheme = _next_scheme_start(rest)
    # A next scheme reached before this URI's query is a separate adjacent URI (or one
    # embedded in the path): end this URI at it and let the run loop pick the next up.
    if next_scheme is not None and (query_at == -1 or next_scheme < query_at):
        authority_end = next_scheme if first_delim is None else min(first_delim, next_scheme)
        authority = rest[:authority_end]
        between = rest[authority_end:next_scheme]  # path text before the next scheme
        piece = f"{scheme}://{_redact_authority(authority)}{between}"
        return piece, marker + 3 + next_scheme
    # Otherwise this URI owns its whole remainder (path/query/fragment); secret query
    # values are replaced whole and non-secret values are scrubbed recursively.
    authority = rest if first_delim is None else rest[:first_delim]
    remainder = "" if first_delim is None else rest[first_delim:]
    piece = f"{scheme}://{_redact_authority(authority)}{_redact_remainder(remainder, depth)}"
    return piece, len(core)


def _first_of(text: str, chars: str) -> int | None:
    indexes = [text.index(char) for char in chars if char in text]
    return min(indexes) if indexes else None


def _scheme_start_for_marker(core: str, marker: int, floor: int) -> int | None:
    # Walk back from a ``://`` over scheme characters to the leading letter, not before
    # ``floor`` (already-emitted text). Returns None if no valid scheme precedes ``://``.
    begin = marker
    while begin > floor and core[begin - 1] in _SCHEME_CHARS:
        begin -= 1
    while begin < marker and core[begin] not in _SCHEME_START_CHARS:
        begin += 1
    return begin if begin < marker else None


def _split_trailer(token: str) -> tuple[str, str]:
    end = len(token)
    while end > 0:
        char = token[end - 1]
        if char in _TRAILER_CHARS:
            end -= 1
            continue
        opener = _UNMATCHED_TRAILERS.get(char)
        if opener is not None and token.count(char, 0, end) > token.count(opener, 0, end):
            end -= 1  # unmatched closing bracket from surrounding prose
            continue
        break
    return token[:end], token[end:]


def _next_scheme_start(rest: str) -> int | None:
    # Find the next ``scheme://`` using a linear ``str.find`` for ``://`` (avoiding any
    # regex backtracking), then walk back over scheme characters to the leading letter.
    search = 0
    while True:
        marker = rest.find("://", search)
        if marker == -1:
            return None
        begin = marker
        while begin > 0 and rest[begin - 1] in _SCHEME_CHARS:
            begin -= 1
        while begin < marker and rest[begin] not in _SCHEME_START_CHARS:
            begin += 1  # a scheme must start with a letter; skip leading digits/+.-
        if begin < marker:
            return begin
        search = marker + 3  # no valid scheme preceded this '://'; keep scanning


def _redact_authority(authority: str) -> str:
    if "@" not in authority:
        return authority  # no userinfo (or username-only handled below)
    userinfo, _, host = authority.rpartition("@")  # last '@' separates userinfo from host
    if ":" not in userinfo:
        return authority  # username-only userinfo is not a password
    username, _, _password = userinfo.partition(":")
    return f"{username}:{REDACTED}@{host}"


def _redact_remainder(remainder: str, depth: int) -> str:
    # Split off the fragment first: a '?' after the '#' belongs to the fragment and is
    # preserved verbatim; only a query before any '#' is scrubbed. The path (and any URI
    # nested in it, e.g. a comma-adjacent second URL) is scrubbed recursively.
    path_query, hash_sep, fragment = remainder.partition("#")
    if "?" in path_query:
        head, _, query = path_query.partition("?")
        return f"{_scrub_uris(head, depth + 1)}?{_redact_query(query, depth)}{hash_sep}{fragment}"
    return f"{_scrub_uris(path_query, depth + 1)}{hash_sep}{fragment}"


def _redact_query(query: str, depth: int) -> str:
    return "".join(
        part if index % 2 else _redact_query_param(part, depth)
        for index, part in enumerate(_QUERY_SEPARATORS.split(query))
    )


def _redact_query_param(part: str, depth: int) -> str:
    key, sep, value = part.partition("=")
    if not sep or not value:
        return part  # a flag or an already-blank value cannot leak a secret
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
