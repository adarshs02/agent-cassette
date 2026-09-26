from retrace import __version__
from retrace.config import DEFAULT_MODEL, Settings


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
