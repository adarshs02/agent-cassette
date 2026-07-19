import asyncio
import sys
import types

import agent_cassette.automatic as automatic
from agent_cassette import Cassette
from agent_cassette.integrations.gemini import GEMINI_SPEC, wrap_gemini


class _Resp:
    def __init__(self, value): self._value = value
    def model_dump(self, mode=None): return {"raw": self._value}
    @property
    def text(self): return self._value.upper()


class _AsyncEventStream:
    def __init__(self, values): self._values = values
    def __aiter__(self):
        async def gen():
            for v in self._values:
                yield _Resp(v)
        return gen()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class _Models:
    def generate_content(self, **kw): return _Resp(kw["contents"])
    def generate_content_stream(self, **kw): return [_Resp("a"), _Resp("b")]


class _AioModels:
    async def generate_content(self, **kw): return _Resp(kw["contents"])
    async def generate_content_stream(self, **kw): return _AsyncEventStream(["a", "b"])


class _Aio:
    def __init__(self): self.models = _AioModels()


class _Client:
    def __init__(self, api_key=None):
        self.models = _Models()
        self.aio = _Aio()


def test_gemini_spec_shape():
    assert GEMINI_SPEC.provider == "gemini"
    assert "models.generate_content" in GEMINI_SPEC.operations
    assert "aio.models.generate_content" in GEMINI_SPEC.async_operations
    assert "models.generate_content_stream" in GEMINI_SPEC.stream_operations
    assert "text" in GEMINI_SPEC.response_attributes
    assert GEMINI_SPEC.async_probe_path == ()


def test_gemini_sync_generate(tmp_path):
    path = tmp_path / "g.jsonl"
    with Cassette.record(path) as c:
        recorded = wrap_gemini(_Client(), c).models.generate_content(
            model="gemini-2.0-flash", contents="works"
        )
    with Cassette.replay(path) as c:
        replayed = wrap_gemini(None, c).models.generate_content(  # type: ignore[var-annotated]
            model="gemini-2.0-flash", contents="works"
        )
    assert recorded.text == replayed.text == "WORKS"


def test_gemini_async_generate(tmp_path):
    path = tmp_path / "g.jsonl"
    req = {"model": "gemini-2.0-flash", "contents": "works"}
    with Cassette.record(path) as c:
        recorded = asyncio.run(wrap_gemini(_Client(), c).aio.models.generate_content(**req))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(  # type: ignore[var-annotated]
            wrap_gemini(None, c).aio.models.generate_content(**req)
        )
    assert recorded.text == replayed.text == "WORKS"


def test_gemini_sync_stream(tmp_path):
    path = tmp_path / "g.jsonl"
    with Cassette.record(path) as c:
        recorded = [ch.text for ch in wrap_gemini(_Client(), c).models.generate_content_stream(
            model="m", contents="x")]
    with Cassette.replay(path) as c:
        replayed = [  # type: ignore[var-annotated]
            ch.text
            for ch in wrap_gemini(None, c).models.generate_content_stream(model="m", contents="x")
        ]
    assert recorded == replayed == ["A", "B"]


def test_gemini_async_stream(tmp_path):
    path = tmp_path / "g.jsonl"

    async def drive(client):
        return [ch.text async for ch in await client.aio.models.generate_content_stream(
            model="m", contents="x")]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_gemini(_Client(), c)))
    with Cassette.replay(path) as c:  # type: ignore[assignment]
        replayed = asyncio.run(drive(wrap_gemini(None, c)))
    assert recorded == replayed == ["A", "B"]


def _install_fake_google_genai(monkeypatch):
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # mark as package
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    return genai_mod


def test_patch_gemini_wraps_constructor(tmp_path, monkeypatch):
    module = _install_fake_google_genai(monkeypatch)
    path = tmp_path / "g.jsonl"
    req = {"model": "gemini-2.0-flash", "contents": "works"}
    with Cassette.record(path) as c, automatic.patch_gemini(c):
        recorded = module.Client(api_key="x").models.generate_content(**req)  # type: ignore[attr-defined]
    with Cassette.replay(path) as c, automatic.patch_gemini(c):  # type: ignore[assignment]
        replayed = module.Client(api_key="x").models.generate_content(**req)  # type: ignore[attr-defined]
    assert recorded.text == replayed.text == "WORKS"
