import asyncio
from collections.abc import Iterator  # noqa: F401

from agent_cassette import Cassette, EventType  # noqa: F401
from agent_cassette.integrations._provider import ProviderSpec, wrap_provider
from agent_cassette.integrations.openai import OPENAI_SPEC


def test_provider_spec_has_empty_generalization_defaults():
    spec = ProviderSpec(provider="x", operations=frozenset(), prefixes=frozenset())
    assert spec.stream_operations == frozenset()
    assert spec.async_operations == frozenset()


def test_existing_specs_do_not_use_new_fields():
    assert OPENAI_SPEC.stream_operations == frozenset()
    assert OPENAI_SPEC.async_operations == frozenset()


class _Resp:
    def __init__(self, text):
        self.text = text

    def model_dump(self, mode=None):
        return {"text": self.text}


class _Chat:
    def go(self, **kw):
        return _Resp("SYNC")

    async def go_async(self, **kw):
        return _Resp("ASYNC")


class _Client:
    def __init__(self):
        self.chat = _Chat()


SPEC = ProviderSpec(
    provider="demo",
    operations=frozenset({"chat.go", "chat.go_async"}),
    prefixes=frozenset({"chat"}),
    async_operations=frozenset({"chat.go_async"}),
)


def test_sync_and_async_ops_coexist_on_one_client(tmp_path):
    path = tmp_path / "demo.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_Client(), cassette, SPEC)
        sync_result = client.chat.go(model="m", messages=[])
        async_result = asyncio.run(client.chat.go_async(model="m", messages=[]))
    assert sync_result.text == "SYNC"
    assert async_result.text == "ASYNC"

    with Cassette.replay(path) as cassette:
        client = wrap_provider(None, cassette, SPEC)
        assert client.chat.go(model="m", messages=[]).text == "SYNC"
        assert asyncio.run(client.chat.go_async(model="m", messages=[])).text == "ASYNC"


class _EventCM:
    """Mimics mistralai: a context manager whose entered value is the iterator."""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *a):
        return False


class _StreamChat:
    def stream(self, **kw):
        return _EventCM([_Resp("A"), _Resp("B")])


class _StreamClient:
    def __init__(self):
        self.chat = _StreamChat()


STREAM_SPEC = ProviderSpec(
    provider="demo",
    operations=frozenset({"chat.stream"}),
    prefixes=frozenset({"chat"}),
    stream_operations=frozenset({"chat.stream"}),
)


def test_context_manager_stream_records_and_replays(tmp_path):
    path = tmp_path / "stream.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_StreamClient(), cassette, STREAM_SPEC)
        with client.chat.stream(model="m", messages=[]) as events:
            recorded = [e.text for e in events]
    assert recorded == ["A", "B"]

    with Cassette.replay(path) as cassette:
        client = wrap_provider(None, cassette, STREAM_SPEC)
        with client.chat.stream(model="m", messages=[]) as events:  # type: Iterator[_Resp]
            replayed = [e.text for e in events]
    assert replayed == ["A", "B"]
