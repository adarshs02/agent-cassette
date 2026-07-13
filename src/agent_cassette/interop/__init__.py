"""Interoperability helpers for external telemetry formats."""

from agent_cassette.interop.otlp import export_otlp, import_otlp

__all__ = ["export_otlp", "import_otlp"]
