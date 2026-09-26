"""Install-hint coverage for retrace.datahub.ingest, independent of the datahub extra.

Unlike test_ingest.py, this module has no module-level `pytest.importorskip("datahub")`: it
must run in the default env (no extra installed) as well as when the extra IS installed, by
forcing the "not installed" path via sys.modules so the ImportError hint stays covered either
way.
"""

from __future__ import annotations

import sys

import pytest
from retrace.config import Settings
from retrace.datahub.ingest import build_proposals, ingest_workspace
from retrace.pipeline.workspace import Workspace

_DATAHUB_MODULES = [
    "datahub",
    "datahub.metadata",
    "datahub.metadata.schema_classes",
    "datahub.emitter",
    "datahub.emitter.mcp",
    "datahub.emitter.rest_emitter",
]


def _block_datahub(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DATAHUB_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def test_build_proposals_raises_install_hint_without_extra(monkeypatch, tmp_path):
    _block_datahub(monkeypatch)
    with pytest.raises(ImportError, match="uv sync --extra datahub"):
        build_proposals(tmp_path / "missing" / "warehouse.duckdb")


def test_ingest_workspace_raises_install_hint_without_extra(monkeypatch, tmp_path):
    _block_datahub(monkeypatch)
    ws = Workspace(root=tmp_path / "missing")
    with pytest.raises(ImportError, match="uv sync --extra datahub"):
        ingest_workspace(ws, Settings())
