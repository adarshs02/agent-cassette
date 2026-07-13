"""Dependency-free environment diagnostics for Agent Cassette."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from importlib.util import find_spec
from typing import Any, TextIO

SCHEMA_VERSION = 1
_PACKAGE_NAME = "agent-cassette"
_CORE_MODULE = "agent_cassette"
_MINIMUM_PYTHON = (3, 10)

_INTEGRATIONS = (
    ("Anthropic", "anthropic", "anthropic", "optional"),
    ("LangChain", "langchain-core", "langchain_core", "optional"),
    ("MCP", "mcp", "mcp", "built-in"),
    ("OpenAI", "openai", "openai", "optional"),
    ("OpenAI Agents", "openai-agents", "agents", "optional"),
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def collect_diagnostics() -> dict[str, Any]:
    """Collect deterministic diagnostics without importing optional integrations."""
    package_version = _version(_PACKAGE_NAME)
    core_available = _module_available(_CORE_MODULE)
    python_supported = sys.version_info >= _MINIMUM_PYTHON

    integrations: list[dict[str, Any]] = []
    for name, distribution, module, kind in _INTEGRATIONS:
        integration_version = _version(distribution)
        integrations.append(
            {
                "available": _module_available(module),
                "distribution": distribution,
                "installed": integration_version is not None,
                "kind": kind,
                "module": module,
                "name": name,
                "supported": True,
                "version": integration_version,
            }
        )

    healthy = core_available and python_supported
    return {
        "healthy": healthy,
        "integrations": integrations,
        "package": {
            "installed": package_version is not None,
            "module_available": core_available,
            "name": _PACKAGE_NAME,
            "version": package_version,
        },
        "python": {
            "minimum": ".".join(str(part) for part in _MINIMUM_PYTHON),
            "supported": python_supported,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "schema_version": SCHEMA_VERSION,
    }


def render_human(report: dict[str, Any]) -> str:
    """Render a diagnostics report for terminal users."""
    package = report["package"]
    python = report["python"]
    package_version = package["version"] or "source checkout / unknown"
    lines = [
        "Agent Cassette doctor",
        f"Package: {package['name']} {package_version} ({_status(package['module_available'])})",
        (
            f"Python: {python['version']} ({_status(python['supported'])}; "
            f"requires >= {python['minimum']})"
        ),
        "Integrations:",
    ]
    for integration in report["integrations"]:
        detail = _integration_status(integration)
        if integration["kind"] == "built-in":
            detail = f"built-in support; SDK {detail}"
        lines.append(f"  {integration['name']}: {detail}")
    lines.append(f"Healthy: {'yes' if report['healthy'] else 'no'}")
    return "\n".join(lines)


def _status(value: bool) -> str:
    return "ok" if value else "unavailable"


def _integration_status(integration: dict[str, Any]) -> str:
    installed = integration["installed"]
    available = integration["available"]
    version = integration["version"]
    if installed and available:
        return version or "installed; version unknown"
    if installed:
        version_detail = f" {version}" if version else ""
        return f"installed{version_detail}; module unavailable"
    if available:
        return "module available; distribution metadata missing (version unknown)"
    return "not installed (optional)"


def doctor(
    *,
    as_json: bool = False,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Print environment diagnostics and return a process-compatible status."""
    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    report = collect_diagnostics()
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    else:
        print(render_human(report), file=output)
    if not report["package"]["module_available"]:
        print("Error: Agent Cassette core module is unavailable.", file=error_output)
    if not report["python"]["supported"]:
        print(
            f"Error: Python {report['python']['version']} is unsupported; "
            f"requires >= {report['python']['minimum']}.",
            file=error_output,
        )
    return 0 if report["healthy"] else 1


__all__ = ["SCHEMA_VERSION", "collect_diagnostics", "doctor", "render_human"]
