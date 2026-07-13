from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from agent_cassette import Cassette, wrap_openai
from agent_cassette.integrations.openai import (
    OpenAIRawResponseUnsupportedError,
    OpenAIStreamingUnsupportedError,
)


@dataclass
class FakeResponse:
    id: str
    output_text: str

    def model_dump(self, mode=None):
        return {"id": self.id, "output_text": self.output_text}


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(id="resp_1", output_text=f"Answer: {kwargs['input']}")


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(id="chat_1", output_text=kwargs["messages"][0]["content"])


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self):
        self.responses = FakeResponses()
        self.chat = FakeChat()

    def with_options(self, **kwargs):
        return self


class FakeAsyncResponses:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(id="resp_async", output_text=f"Async: {kwargs['input']}")


class FakeAsyncOpenAI:
    def __init__(self):
        self.responses = FakeAsyncResponses()


def test_openai_responses_record_and_replay_without_live_client(tmp_path):
    path = tmp_path / "openai.jsonl"
    live_client = FakeOpenAI()

    with Cassette.record(path) as cassette:
        client = wrap_openai(live_client, cassette)
        response = client.responses.create(model="gpt-test", input="hello")

    with Cassette.replay(path) as cassette:
        offline_client = wrap_openai(None, cassette)
        replayed = offline_client.responses.create(model="gpt-test", input="hello")

    assert response.output_text == replayed.output_text == "Answer: hello"
    assert replayed.id == "resp_1"
    assert live_client.responses.calls == 1


def test_derived_openai_clients_remain_wrapped_during_offline_replay(tmp_path):
    path = tmp_path / "derived.jsonl"
    live_client = FakeOpenAI()

    with Cassette.record(path) as cassette:
        recorded = (
            wrap_openai(live_client, cassette)
            .with_options(timeout=10)
            .responses.create(model="gpt-test", input="hello")
        )
    with Cassette.replay(path) as cassette:
        replayed = (
            wrap_openai(None, cassette)
            .with_options(timeout=10)
            .responses.create(model="gpt-test", input="hello")
        )

    assert recorded.output_text == replayed.output_text == "Answer: hello"
    assert live_client.responses.calls == 1


def test_openai_chat_completions_are_intercepted(tmp_path):
    path = tmp_path / "chat.jsonl"
    live_client = FakeOpenAI()
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}

    with Cassette.record(path) as cassette:
        response = wrap_openai(live_client, cassette).chat.completions.create(**request)
    with Cassette.replay(path) as cassette:
        replayed = wrap_openai(None, cassette).chat.completions.create(**request)

    assert response.output_text == replayed.output_text == "hello"
    assert live_client.chat.completions.calls == 1


def test_async_openai_responses_record_and_replay(tmp_path):
    path = tmp_path / "async-openai.jsonl"
    live_client = FakeAsyncOpenAI()

    async def scenario():
        async with Cassette.record(path) as cassette:
            client = wrap_openai(live_client, cassette)
            recorded = await client.responses.create(model="gpt-test", input="hello")
        async with Cassette.replay(path) as cassette:
            client = wrap_openai(None, cassette, asynchronous=True)
            replayed = await client.responses.create(model="gpt-test", input="hello")
        return recorded, replayed

    recorded, replayed = asyncio.run(scenario())

    assert recorded.output_text == replayed.output_text == "Async: hello"
    assert live_client.responses.calls == 1


def test_openai_raw_response_helpers_fail_explicitly(tmp_path):
    with Cassette.record(tmp_path / "raw.jsonl") as cassette:
        client = wrap_openai(FakeOpenAI(), cassette)
        with pytest.raises(OpenAIRawResponseUnsupportedError, match="not replay-safe"):
            _ = client.responses.with_raw_response


def test_openai_streaming_fails_explicitly(tmp_path):
    with Cassette.record(tmp_path / "stream.jsonl") as cassette:
        client = wrap_openai(FakeOpenAI(), cassette)
        with pytest.raises(OpenAIStreamingUnsupportedError, match="stream=False"):
            client.responses.create(model="gpt-test", input="hello", stream=True)
