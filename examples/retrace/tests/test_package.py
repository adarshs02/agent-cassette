from retrace import __version__
from retrace.config import DEFAULT_MCP_LOG_PATH, DEFAULT_MODEL, Settings


def test_version():
    assert __version__ == "0.1.0"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RETRACE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("DATAHUB_GMS_URL", "http://dh:8080")
    monkeypatch.delenv("DATAHUB_GMS_TOKEN", raising=False)
    settings = Settings.from_env()
    assert settings.model == "claude-opus-5-5"
    assert settings.datahub_gms_url == "http://dh:8080"
    assert settings.datahub_gms_token is None
    assert settings.max_turns == 40


def test_default_model():
    assert Settings().model == DEFAULT_MODEL == "claude-sonnet-5"


def test_mcp_log_path_from_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom-mcp.log"
    monkeypatch.setenv("RETRACE_MCP_LOG", str(custom))
    assert Settings.from_env().mcp_log_path == custom
    monkeypatch.delenv("RETRACE_MCP_LOG")
    assert Settings.from_env().mcp_log_path == DEFAULT_MCP_LOG_PATH
