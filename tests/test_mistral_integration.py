import asyncio
import sys
import types

import pytest

import agent_cassette.automatic as automatic
from agent_cassette import Cassette
from agent_cassette.automatic import MistralUnavailableError
from agent_cassette.integrations.mistral import MISTRAL_SPEC, wrap_mistral


class _Resp:
    def __init__(self, text):
        self.text = text

    def model_dump(self, mode=None):
        return {"text": self.text}


class _EventCM:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *a):
        return False


class _AsyncEventStream:
    """Mirror mistralai's ``EventStreamAsync``: an async iterator that is also
    usable as an async context manager."""

    def __init__(self, events):
        self._events = list(events)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Chat:
    def complete(self, **kw):
        return _Resp(kw["messages"][0]["content"].upper())

    async def complete_async(self, **kw):
        return _Resp(kw["messages"][0]["content"].upper())

    def stream(self, **kw):
        return _EventCM([_Resp("A"), _Resp("B")])

    async def stream_async(self, **kw):
        # Real mistralai ``chat.stream_async`` is ``async def``: callers do
        # ``response = await client.chat.stream_async(...)`` and then iterate.
        return _AsyncEventStream([_Resp("A"), _Resp("B")])


class _Mistral:
    def __init__(self, api_key=None):
        self.chat = _Chat()


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
        # Real mistralai convention: await the coroutine, then async-iterate.
        response = await client.chat.stream_async(model="m", messages=[])
        return [e.text async for e in response]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_mistral(_Mistral(), c)))
    with Cassette.replay(path) as c:  # type: ignore[assignment]
        replayed = asyncio.run(drive(wrap_mistral(None, c)))
    assert recorded == replayed == ["A", "B"]


def test_mistral_async_stream_context_manager(tmp_path):
    path = tmp_path / "m.jsonl"

    async def drive(client):
        # ``EventStreamAsync`` is also usable as ``async with``.
        response = await client.chat.stream_async(model="m", messages=[])
        async with response as events:
            return [e.text async for e in events]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_mistral(_Mistral(), c)))
    with Cassette.replay(path) as c:  # type: ignore[assignment]
        replayed = asyncio.run(drive(wrap_mistral(None, c)))
    assert recorded == replayed == ["A", "B"]


def _install_fake_mistralai(monkeypatch):
    module = types.ModuleType("mistralai")
    module.Mistral = _Mistral  # type: ignore[attr-defined]  # from earlier in this file
    monkeypatch.setitem(sys.modules, "mistralai", module)
    return module


def test_patch_mistral_wraps_constructor(tmp_path, monkeypatch):
    module = _install_fake_mistralai(monkeypatch)
    path = tmp_path / "m.jsonl"
    request = {"model": "m", "messages": [{"role": "user", "content": "works"}]}
    with Cassette.record(path) as c, automatic.patch_mistral(c):
        recorded = module.Mistral(api_key="x").chat.complete(**request)
    with Cassette.replay(path) as c, automatic.patch_mistral(c):  # type: ignore[assignment]
        replayed = module.Mistral(api_key="x").chat.complete(**request)
    assert recorded.text == replayed.text == "WORKS"


def test_missing_mistral_dependency_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "mistralai", raising=False)

    def missing_mistralai(name, *args, **kwargs):
        assert name == "mistralai"
        raise ModuleNotFoundError("No module named 'mistralai'", name="mistralai")

    monkeypatch.setattr("agent_cassette.automatic.importlib.import_module", missing_mistralai)

    with Cassette.record(tmp_path / "missing.jsonl") as cassette:
        with pytest.raises(MistralUnavailableError, match=r"agent-cassette\[mistral\]") as excinfo:
            with automatic.patch_mistral(cassette):
                pass

    assert "agent-cassette[mistral]" in str(excinfo.value)
    assert "agent-cassette[mistralai]" not in str(excinfo.value)
