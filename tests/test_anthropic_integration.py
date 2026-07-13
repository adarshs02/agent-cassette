from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass

import pytest

from agent_cassette import Cassette, wrap_anthropic
from agent_cassette.automatic import (
    AnthropicAlreadyPatchedError,
    AnthropicUnavailableError,
    patch_anthropic,
)
from agent_cassette.integrations.anthropic import (
    AnthropicRawResponseUnsupportedError,
    AnthropicStreamingUnsupportedError,
)


@dataclass
class FakeMessage:
    id: str
    text: str

    def model_dump(self, mode=None):
        return {"id": self.id, "text": self.text}


class FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("stream"):
            return iter(
                [FakeMessage(id="chunk_1", text="Hel"), FakeMessage(id="chunk_2", text="lo")]
            )
        return FakeMessage(id="msg_1", text=f"Claude: {kwargs['messages'][0]['content']}")

    def stream(self, **kwargs):
        raise AssertionError("stream helper should be intercepted before reaching the client")


class FakeAnthropic:
    def __init__(self, api_key=None):
        self.messages = FakeMessages()

    def with_options(self, **kwargs):
        return self


class FakeAsyncMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return FakeMessage(id="msg_async", text=f"Async: {kwargs['messages'][0]['content']}")


class FakeAsyncAnthropic:
    def __init__(self, api_key=None):
        self.messages = FakeAsyncMessages()


REQUEST = {
    "model": "claude-test",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hello"}],
}


def test_anthropic_messages_record_and_replay_without_live_client(tmp_path):
    path = tmp_path / "anthropic.jsonl"
    live_client = FakeAnthropic()

    with Cassette.record(path) as cassette:
        response = wrap_anthropic(live_client, cassette).messages.create(**REQUEST)

    with Cassette.replay(path) as cassette:
        replayed = wrap_anthropic(None, cassette).messages.create(**REQUEST)

    assert response.text == replayed.text == "Claude: hello"
    assert replayed.id == "msg_1"
    assert live_client.messages.calls == 1


def test_anthropic_async_messages_record_and_replay(tmp_path):
    path = tmp_path / "anthropic-async.jsonl"
    live_client = FakeAsyncAnthropic()

    async def scenario():
        with Cassette.record(path) as cassette:
            recorded = await wrap_anthropic(live_client, cassette).messages.create(**REQUEST)
        with Cassette.replay(path) as cassette:
            replayed = await wrap_anthropic(
                None, cassette, asynchronous=True
            ).messages.create(**REQUEST)
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded.text == replayed.text == "Async: hello"
    assert live_client.messages.calls == 1


def test_anthropic_streaming_record_and_replay(tmp_path):
    path = tmp_path / "anthropic-stream.jsonl"
    live_client = FakeAnthropic()
    request = {**REQUEST, "stream": True}

    with Cassette.record(path) as cassette:
        chunks = list(wrap_anthropic(live_client, cassette).messages.create(**request))

    with Cassette.replay(path) as cassette:
        replayed = list(wrap_anthropic(None, cassette).messages.create(**request))

    assert [chunk.text for chunk in chunks] == ["Hel", "lo"]
    assert [chunk.text for chunk in replayed] == ["Hel", "lo"]
    assert live_client.messages.calls == 1


def test_anthropic_stream_helper_is_rejected(tmp_path):
    with Cassette.record(tmp_path / "helper.jsonl") as cassette:
        client = wrap_anthropic(FakeAnthropic(), cassette)
        with pytest.raises(AnthropicStreamingUnsupportedError, match="stream=True"):
            client.messages.stream(**REQUEST)


def test_anthropic_raw_response_helpers_are_rejected(tmp_path):
    with Cassette.record(tmp_path / "raw.jsonl") as cassette:
        client = wrap_anthropic(FakeAnthropic(), cassette)
        with pytest.raises(AnthropicRawResponseUnsupportedError):
            client.messages.with_raw_response


@pytest.fixture
def fake_anthropic(monkeypatch):
    module = types.ModuleType("anthropic")
    module.Anthropic = FakeAnthropic
    module.AsyncAnthropic = FakeAsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return module


def test_patch_anthropic_records_and_replays_without_live_call(tmp_path, fake_anthropic):
    path = tmp_path / "automatic.jsonl"

    with Cassette.record(path) as cassette, patch_anthropic(cassette):
        recorded = fake_anthropic.Anthropic(api_key="key").messages.create(**REQUEST)

    with Cassette.replay(path) as cassette, patch_anthropic(cassette):
        replayed = fake_anthropic.Anthropic().messages.create(**REQUEST)

    assert recorded.text == replayed.text == "Claude: hello"
    assert fake_anthropic.Anthropic is FakeAnthropic


def test_nested_patch_anthropic_has_clear_error(tmp_path, fake_anthropic):
    with Cassette.record(tmp_path / "nested.jsonl") as cassette, patch_anthropic(cassette):
        with pytest.raises(AnthropicAlreadyPatchedError, match="nested"):
            with patch_anthropic(cassette):
                pass


def test_missing_anthropic_dependency_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    def missing_anthropic(name):
        assert name == "anthropic"
        raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")

    monkeypatch.setattr("agent_cassette.automatic.importlib.import_module", missing_anthropic)

    with Cassette.record(tmp_path / "missing.jsonl") as cassette:
        with pytest.raises(AnthropicUnavailableError, match=r"agent-cassette\[anthropic\]"):
            with patch_anthropic(cassette):
                pass
