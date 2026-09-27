"""The only writer: patch a scratch copy of the transforms, rebuild, verify."""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrace.pipeline.build import TRANSFORM_ORDER, BuildError, build
from retrace.pipeline.checks import failed_names, run_checks
from retrace.pipeline.lineage import dependency_closure, table_name
from retrace.pipeline.workspace import Workspace


class RepairRejected(ValueError):
    """The proposed repair target is not allowed."""


ATTRIBUTION_PREFIX = "-- Adapted from Project Blackbox"


def _first_nonempty_line(text: str) -> str | None:
    return next((line for line in text.splitlines() if line.strip()), None)


@dataclass
class RepairOutcome:
    passed: bool
    failed_checks: list[str]
    results: list[dict[str, Any]]
    diff: str
    error: str | None
    patched: dict[str, str] = field(default_factory=dict)


class Transforms:
    def __init__(self, ws: Workspace) -> None:
        self._dir = ws.transforms

    def list(self) -> list[str]:
        return [f"{name}.sql" for name in TRANSFORM_ORDER]

    def read(self, name: str) -> str:
        stem = name.removesuffix(".sql")
        if stem not in TRANSFORM_ORDER:
            raise ValueError(f"unknown transform {name!r}; known: {self.list()}")
        return (self._dir / f"{stem}.sql").read_text()


def allowed_repair_targets(asset: str) -> set[str]:
    table = table_name(asset)
    return {name for name, deps in dependency_closure().items() if table in deps}


def unified_diff(original_dir: Path, patched: dict[str, str]) -> str:
    chunks: list[str] = []
    for name in sorted(patched):
        before = (original_dir / f"{name}.sql").read_text().splitlines(keepends=True)
        after = patched[name].splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(before, after, fromfile=f"a/{name}.sql", tofile=f"b/{name}.sql")
        )
    return "".join(chunks)


def _stem(value: str, *, label: str = "file") -> str:
    stem = value.removesuffix(".sql")
    if not value or stem not in TRANSFORM_ORDER or value not in (stem, f"{stem}.sql"):
        raise RepairRejected(
            f"{label} must be one of {[f'{n}.sql' for n in TRANSFORM_ORDER]}, got {value!r}"
        )
    return stem


class Repairer:
    def __init__(self, ws: Workspace, baseline: dict, scratch_root: Path) -> None:
        self.ws = ws
        self.baseline = baseline
        self.scratch_root = scratch_root
        self.attempts = 0

    def _keep_attribution_header(self, stem: str, new_sql: str) -> str:
        """Deterministically restore a dropped Project Blackbox attribution line.

        If the transform being patched originally started with the attribution
        comment and ``new_sql``'s own first *non-empty* line isn't that exact
        line, prepend it. Checking the leading position (rather than a substring
        search anywhere in the text) avoids being fooled by the header text
        merely appearing later in the file, e.g. buried in a mid-file comment.
        The repair is otherwise untouched.
        """
        original = (self.ws.transforms / f"{stem}.sql").read_text()
        first_line = original.splitlines()[0] if original else ""
        kept = _first_nonempty_line(new_sql) == first_line
        if first_line.startswith(ATTRIBUTION_PREFIX) and not kept:
            return f"{first_line}\n{new_sql}"
        return new_sql

    def propose(
        self, file: str, new_sql: str, root_asset: str, accepted: dict[str, str]
    ) -> RepairOutcome:
        stem = _stem(file)
        accepted_stems = {_stem(key, label="accepted key"): sql for key, sql in accepted.items()}
        allowed = allowed_repair_targets(root_asset)
        if stem not in allowed:
            raise RepairRejected(
                f"{stem}.sql is not downstream of {table_name(root_asset)}; "
                f"allowed: {sorted(allowed)}"
            )
        new_sql = self._keep_attribution_header(stem, new_sql)
        self.attempts += 1
        scratch = self.scratch_root / f"attempt_{self.attempts}"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        shutil.copytree(self.ws.transforms, scratch / "transforms")
        patched = {**accepted_stems, stem: new_sql}
        for name, sql in patched.items():
            (scratch / "transforms" / f"{name}.sql").write_text(sql)
        diff = unified_diff(self.ws.transforms, patched)
        try:
            build(self.ws.sources, scratch / "transforms", scratch / "warehouse.duckdb")
        except BuildError as error:
            message = str(error).replace(str(scratch), "<scratch>")
            return RepairOutcome(False, [], [], diff, message, patched)
        results = run_checks(scratch / "warehouse.duckdb", self.baseline)
        failed = failed_names(results)
        return RepairOutcome(
            not failed, failed, [r.to_dict() for r in results], diff, None, patched
        )
