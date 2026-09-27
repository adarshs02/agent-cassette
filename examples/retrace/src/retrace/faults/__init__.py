"""Fault library: each fault injects one data problem and carries its ground truth."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import sqlglot
import sqlglot.errors

from retrace.pipeline.generate import Frames
from retrace.pipeline.workspace import TRANSFORMS_DIR

Outcome = Literal["sql_repair", "escalate", "no_incident"]


@dataclass(frozen=True)
class GroundTruth:
    outcome: Outcome
    asset: str | None
    field: str | None
    alt_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    detail: str


RuleFn = Callable[[dict[str, str]], RuleResult]


@dataclass(frozen=True)
class Fault:
    name: str
    report: str
    inject: Callable[[Frames], None]
    ground_truth: GroundTruth
    must_fail: tuple[str, ...] = ()
    reference_patch: dict[str, str] | None = None
    repair_rules: tuple[RuleFn, ...] = field(default_factory=tuple)
    variant: Callable[[Frames], None] | None = None


_REGISTRY: dict[str, Fault] = {}


def register(fault: Fault) -> Fault:
    _REGISTRY[fault.name] = fault
    return fault


def get_fault(name: str) -> Fault:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown fault {name!r}; known: {sorted(_REGISTRY)}") from None


def fault_names() -> list[str]:
    return list(_REGISTRY)


def patched_transform(name: str, old: str, new: str) -> dict[str, str]:
    original = (TRANSFORMS_DIR / f"{name}.sql").read_text()
    if original.count(old) != 1:
        raise ValueError(f"expected exactly one {old!r} in {name}.sql")
    return {name: original.replace(old, new)}


# A comment-stripping regex that also recognizes quoted strings/identifiers, so it
# can be applied even where a literal containing "--" or "/*" might be present
# without mangling it (the quoted branches match first and are copied through as-is).
_SQL_TOKEN_OR_COMMENT_RE = re.compile(
    r"'(?:[^']|'')*'" r'|"(?:[^"]|"")*"' r"|--[^\n]*" r"|/\*.*?\*/",
    re.DOTALL,
)


def _strip_sql_comments_regex(sql: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        text = match.group()
        return "" if text.startswith(("--", "/*")) else text

    return _SQL_TOKEN_OR_COMMENT_RE.sub(_replace, sql)


def strip_sql_comments(sql: str) -> str:
    """Remove ``-- ...`` line comments and ``/* ... */`` block comments from SQL.

    Repair rules must judge the *code*, not comments -- a rule like
    ``mentions_all("cloudpay_v2", "100")`` should not pass just because a patch's
    comment happens to namedrop the right processor while the logic itself is an
    unscoped format heuristic.

    Preferred path: sqlglot's tokenizer (duckdb dialect) gives the exact source
    span of every real token. By construction, whatever lies *between* two
    adjacent spans (or before the first one) is nothing but whitespace and/or
    comments -- a tokenizer can't emit an in-between span that contains actual
    code, since that code would itself be a token. That makes it safe to run the
    comment-stripping regex on just those gaps: it can never mistake a string
    literal for a comment there, because a literal is always its own complete
    token and is copied through unmodified via its span, never via a gap. This
    sidesteps needing to know which side of a gap sqlglot happened to attach a
    given comment to (it's inconsistent: a same-line trailing comment attaches
    to the token before it, one on its own line attaches to the token after).

    Falls back to running that same regex over the *whole* string only if the
    tokenizer cannot lex the SQL at all (``SqlglotError``, e.g. an unterminated
    string/comment in an in-progress patch); there, a literal containing ``--``
    is still protected because the regex's quoted-string/identifier branches are
    tried first and copied through as-is, but this path is unvalidated against
    every SQL corner case, so it is a best-effort fallback only.
    """
    try:
        tokens = sqlglot.tokenize(sql, read="duckdb")
    except sqlglot.errors.SqlglotError:
        return _strip_sql_comments_regex(sql)
    if not tokens:
        return sql
    parts: list[str] = [_strip_sql_comments_regex(sql[: tokens[0].start])]
    for prev, cur in zip(tokens, tokens[1:]):
        parts.append(sql[prev.start : prev.end + 1])
        parts.append(_strip_sql_comments_regex(sql[prev.end + 1 : cur.start]))
    parts.append(sql[tokens[-1].start : tokens[-1].end + 1])
    return "".join(parts)


def files_within(*names: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        extra = sorted(set(patched) - set(names))
        return RuleResult(
            "files_within",
            bool(patched) and not extra,
            f"patched={sorted(patched)} allowed={sorted(names)}",
        )

    return rule


def _patched_code_text(patched: dict[str, str]) -> str:
    """Join patched SQL with comments stripped, so rules judge code, not comments."""
    return "\n".join(strip_sql_comments(sql) for sql in patched.values()).lower()


def mentions_all(*needles: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        text = _patched_code_text(patched)
        missing = [n for n in needles if n.lower() not in text]
        return RuleResult(f"mentions_all({', '.join(needles)})", not missing, f"missing={missing}")

    return rule


def mentions_any(*needles: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        text = _patched_code_text(patched)
        hit = [n for n in needles if n.lower() in text]
        return RuleResult(f"mentions_any({', '.join(needles)})", bool(hit), f"found={hit}")

    return rule


def evaluate_rules(fault: Fault, patched: dict[str, str]) -> list[RuleResult]:
    return [rule(patched) for rule in fault.repair_rules]


from retrace.faults import repairable, upstream  # noqa: E402,F401  (registers faults)
