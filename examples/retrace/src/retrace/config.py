"""Runtime settings and well-known paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
CASSETTE_DIR = APP_ROOT / "evals" / "cassettes"
RESULTS_DIR = APP_ROOT / "evals" / "results"
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    datahub_gms_url: str = "http://localhost:8080"
    datahub_gms_token: str | None = None
    mcp_server_spec: str = "mcp-server-datahub"
    max_turns: int = 40
    max_repair_attempts: int = 3
    max_nudges: int = 2
    ingest_settle_s: float = 5.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.environ.get("RETRACE_MODEL", DEFAULT_MODEL),
            datahub_gms_url=os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080"),
            datahub_gms_token=os.environ.get("DATAHUB_GMS_TOKEN") or None,
            mcp_server_spec=os.environ.get("RETRACE_MCP_SERVER", "mcp-server-datahub"),
            ingest_settle_s=float(os.environ.get("RETRACE_INGEST_SETTLE_S", "5.0")),
        )
