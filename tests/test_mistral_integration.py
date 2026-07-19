import asyncio

from agent_cassette import Cassette
from agent_cassette.integrations.mistral import MISTRAL_SPEC, wrap_mistral


class _Resp:
    def __init__(self, text): self.text = text
    def model_dump(self, mode=None): return {"text": self.text}


class _EventCM:
    def __init__(self, events): self._events = events
    def __enter__(self): return iter(self._events)
    def __exit__(self, *a): return False


class _AsyncEventCM:
    def __init__(self, events): self._events = events
    async def __aenter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()
    async def __aexit__(self, *a): return False


class _Chat:
    def complete(self, **kw): return _Resp(kw["messages"][0]["content"].upper())
    async def complete_async(self, **kw): return _Resp(kw["messages"][0]["content"].upper())
    def stream(self, **kw): return _EventCM([_Resp("A"), _Resp("B")])
    def stream_async(self, **kw): return _AsyncEventCM([_Resp("A"), _Resp("B")])


class _Mistral:
    def __init__(self, api_key=None): self.chat = _Chat()


def test_mistral_spec_shape():
    assert MISTRAL_SPEC.provider == "mistral"
    assert "chat.complete" in MISTRAL_SPEC.operations
    assert "chat.complete_async" in MISTRAL_SPEC.async_operations
    assert "chat.stream" in MISTRAL_SPEC.stream_operations
    assert MISTRAL_SPEC.async_probe_path == ()


def test_mistral_sync_complete(tmp_path):
    path = tmp_path / "m.jsonl"
    with Cassette.record(path) as c:
        recorded = wrap_mistral(_Mistral(), c).chat.complete(
            model="mistral-small", messages=[{"role": "user", "content": "works"}]
        )
    with Cassette.replay(path) as c:
        replayed = wrap_mistral(None, c).chat.complete(  # type: ignore[var-annotated]
            model="mistral-small", messages=[{"role": "user", "content": "works"}]
        )
    assert recorded.text == replayed.text == "WORKS"


def test_mistral_async_complete(tmp_path):
    path = tmp_path / "m.jsonl"
    request = {"model": "mistral-small", "messages": [{"role": "user", "content": "works"}]}
    with Cassette.record(path) as c:
        recorded = asyncio.run(wrap_mistral(_Mistral(), c).chat.complete_async(**request))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(  # type: ignore[var-annotated]
            wrap_mistral(None, c).chat.complete_async(**request)
        )
    assert recorded.text == replayed.text == "WORKS"


def test_mistral_sync_stream(tmp_path):
    path = tmp_path / "m.jsonl"
    with Cassette.record(path) as c:
        with wrap_mistral(_Mistral(), c).chat.stream(model="m", messages=[]) as events:
            recorded = [e.text for e in events]
    with Cassette.replay(path) as c:
        with wrap_mistral(None, c).chat.stream(  # type: ignore[var-annotated]
            model="m", messages=[]
        ) as events:
            replayed = [e.text for e in events]
    assert recorded == replayed == ["A", "B"]


def test_mistral_async_stream(tmp_path):
    path = tmp_path / "m.jsonl"

    async def drive(client):
        async with client.chat.stream_async(model="m", messages=[]) as events:
            return [e.text async for e in events]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_mistral(_Mistral(), c)))
    with Cassette.replay(path) as c:  # type: ignore[assignment]
        replayed = asyncio.run(drive(wrap_mistral(None, c)))
    assert recorded == replayed == ["A", "B"]
