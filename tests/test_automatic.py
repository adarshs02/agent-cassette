from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_cassette import Cassette
from agent_cassette.automatic import (
    OpenAIAlreadyPatchedError,
    OpenAIUnavailableError,
    automatic_openai_from_env,
    patch_openai,
)


class FakeResponse:
    def __init__(self, identifier: str, output_text: str) -> None:
        self.id = identifier
        self.output_text = output_text

    def model_dump(self, mode=None):
        return {"id": self.id, "output_text": self.output_text}


class FakeResponses:
    live_calls = 0

    def create(self, **kwargs):
        type(self).live_calls += 1
        return FakeResponse("sync-response", f"sync: {kwargs['input']}")


class FakeAsyncResponses:
    live_calls = 0

    async def create(self, **kwargs):
        type(self).live_calls += 1
        return FakeResponse("async-response", f"async: {kwargs['input']}")


class FakeOpenAI:
    constructed = []

    def __init__(self, api_key=None):
        type(self).constructed.append(api_key)
        self.responses = FakeResponses()


class FakeAsyncOpenAI:
    constructed = []

    def __init__(self, api_key=None):
        type(self).constructed.append(api_key)
        self.responses = FakeAsyncResponses()


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeResponses.live_calls = 0
    FakeAsyncResponses.live_calls = 0
    FakeOpenAI.constructed = []
    FakeAsyncOpenAI.constructed = []


@pytest.fixture
def fake_openai(monkeypatch):
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    module.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def test_patch_openai_wraps_sync_and_async_constructors(tmp_path, fake_openai):
    path = tmp_path / "automatic.jsonl"
    original_sync = fake_openai.OpenAI
    original_async = fake_openai.AsyncOpenAI

    with Cassette.record(path) as cassette, patch_openai(cassette):
        sync_client = fake_openai.OpenAI(api_key="sync-key")
        async_client = fake_openai.AsyncOpenAI(api_key="async-key")
        sync_response = sync_client.responses.create(model="test", input="hello")
        async_response = asyncio.run(async_client.responses.create(model="test", input="hello"))

    assert sync_response.output_text == "sync: hello"
    assert async_response.output_text == "async: hello"
    assert FakeOpenAI.constructed == ["sync-key"]
    assert FakeAsyncOpenAI.constructed == ["async-key"]
    assert fake_openai.OpenAI is original_sync
    assert fake_openai.AsyncOpenAI is original_async


def test_patch_openai_restores_constructors_after_exception(tmp_path, fake_openai):
    original_sync = fake_openai.OpenAI
    original_async = fake_openai.AsyncOpenAI

    with pytest.raises(RuntimeError, match="client failure"):
        with Cassette.record(tmp_path / "exception.jsonl") as cassette, patch_openai(cassette):
            raise RuntimeError("client failure")

    assert fake_openai.OpenAI is original_sync
    assert fake_openai.AsyncOpenAI is original_async


def test_nested_patch_openai_has_clear_error(tmp_path, fake_openai):
    with Cassette.record(tmp_path / "nested.jsonl") as cassette, patch_openai(cassette):
        with pytest.raises(OpenAIAlreadyPatchedError, match="nested"):
            with patch_openai(cassette):
                pass


def test_automatic_environment_records_and_replays_without_live_call(tmp_path, fake_openai):
    path = tmp_path / "environment.jsonl"
    request = {"model": "test", "input": "environment"}

    with automatic_openai_from_env(
        {"AGENT_CASSETTE_MODE": "record", "AGENT_CASSETTE_PATH": str(path)}
    ):
        recorded = fake_openai.OpenAI().responses.create(**request)

    with automatic_openai_from_env(
        {"AGENT_CASSETTE_MODE": "replay", "AGENT_CASSETTE_PATH": str(path)}
    ):
        replayed = fake_openai.OpenAI().responses.create(**request)

    assert recorded.output_text == replayed.output_text == "sync: environment"
    assert FakeResponses.live_calls == 1
    assert FakeOpenAI.constructed == [None]


def test_automatic_environment_replays_async_without_live_call(tmp_path, fake_openai):
    path = tmp_path / "async-environment.jsonl"
    environment = {"AGENT_CASSETTE_PATH": str(path)}

    async def scenario():
        with automatic_openai_from_env({**environment, "AGENT_CASSETTE_MODE": "record"}):
            recorded = await fake_openai.AsyncOpenAI().responses.create(
                model="test", input="environment"
            )
        with automatic_openai_from_env({**environment, "AGENT_CASSETTE_MODE": "replay"}):
            replayed = await fake_openai.AsyncOpenAI().responses.create(
                model="test", input="environment"
            )
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded.output_text == replayed.output_text == "async: environment"
    assert FakeAsyncResponses.live_calls == 1
    assert FakeAsyncOpenAI.constructed == [None]


def test_missing_openai_dependency_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    def missing_openai(name):
        assert name == "openai"
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr("agent_cassette.automatic.importlib.import_module", missing_openai)

    with Cassette.record(tmp_path / "missing.jsonl") as cassette:
        with pytest.raises(OpenAIUnavailableError, match=r"agent-cassette\[openai\]"):
            with patch_openai(cassette):
                pass


def test_automatic_environment_validates_configuration():
    with pytest.raises(ValueError, match="AGENT_CASSETTE_MODE"):
        with automatic_openai_from_env({}):
            pass

    with pytest.raises(ValueError, match="AGENT_CASSETTE_PATH"):
        with automatic_openai_from_env({"AGENT_CASSETTE_MODE": "record"}):
            pass
