"""Fault library: each fault injects one data problem and carries its ground truth."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from retrace.pipeline.generate import Frames
from retrace.pipeline.workspace import TRANSFORMS_DIR

Outcome = Literal["sql_repair", "escalate", "no_incident"]


@dataclass(frozen=True)
class GroundTruth:
    outcome: Outcome
    asset: str | None
    field: str | None


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


def files_within(*names: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        extra = sorted(set(patched) - set(names))
        return RuleResult(
            "files_within",
            bool(patched) and not extra,
            f"patched={sorted(patched)} allowed={sorted(names)}",
        )

    return rule


def mentions_all(*needles: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        text = "\n".join(patched.values()).lower()
        missing = [n for n in needles if n.lower() not in text]
        return RuleResult(f"mentions_all({', '.join(needles)})", not missing, f"missing={missing}")

    return rule


def mentions_any(*needles: str) -> RuleFn:
    def rule(patched: dict[str, str]) -> RuleResult:
        text = "\n".join(patched.values()).lower()
        hit = [n for n in needles if n.lower() in text]
        return RuleResult(f"mentions_any({', '.join(needles)})", bool(hit), f"found={hit}")

    return rule


def evaluate_rules(fault: Fault, patched: dict[str, str]) -> list[RuleResult]:
    return [rule(patched) for rule in fault.repair_rules]


from retrace.faults import repairable, upstream  # noqa: E402,F401  (registers faults)
