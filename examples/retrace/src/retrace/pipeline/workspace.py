"""A self-contained directory holding sources, transforms, and a warehouse."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from retrace.pipeline.build import build
from retrace.pipeline.generate import generate

TRANSFORMS_DIR = Path(__file__).parent / "transforms"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def transforms(self) -> Path:
        return self.root / "transforms"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse.duckdb"

    @classmethod
    def create(cls, root: Path) -> Workspace:
        root.mkdir(parents=True, exist_ok=True)
        target = root / "transforms"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(TRANSFORMS_DIR, target)
        return cls(root)


def prepare(root: Path, fault: str | None, *, variant: bool = False) -> Workspace:
    ws = Workspace.create(root)
    generate(ws.sources, fault, variant=variant)
    build(ws.sources, ws.transforms, ws.warehouse)
    return ws
