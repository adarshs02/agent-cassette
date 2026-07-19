# Gemini Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add record/replay for the `google-genai` SDK — sync `models.generate_content`, async `aio.models.generate_content`, and streaming `models.generate_content_stream` / `aio.models.generate_content_stream` — as Phase 2 of the multi-provider foundation, and make `response.text` (a computed property) survive replay.

**Architecture:** The async `.aio` subtree and `*_stream` methods are already handled by the shipped foundation (G5 `async_operations`, G2 `stream_operations`, the awaiting async-stream path, and context-manager-tolerant streams) — the `aio.*` operations are simply enumerated in the spec. The one new foundation piece is **G6**: a `ProviderSpec.response_attributes` field that captures declared computed attributes (e.g. `text`) at record time so the replay object exposes them, preserving the offline / no-SDK-import guarantee. Then define `GEMINI_SPEC`, `wrap_gemini`, and a single-constructor `patch_gemini`.

**Tech Stack:** Python ≥3.10, `uv`, pytest, ruff, mypy, hatchling. Optional dep `google-genai>=1,<2`.

**Design source:** `docs/superpowers/specs/2026-07-18-multi-provider-foundation-design.md` and `docs/adding-a-provider.md`.

## Verified SDK facts (google-genai)

- Client: `from google.genai import Client; client = Client(api_key=...)` — a single class; async is exposed under `client.aio` (a subtree, not a separate class).
- Sync: `client.models.generate_content(model=, contents=, config=)` → `GenerateContentResponse`.
- Sync stream: `client.models.generate_content_stream(...)` → sync iterator of `GenerateContentResponse` chunks.
- Async: `await client.aio.models.generate_content(...)`.
- Async stream: `async for chunk in await client.aio.models.generate_content_stream(...)` — the stream method is a coroutine (`await`) returning an async iterator (identical shape to Mistral `stream_async`; the foundation already routes async∩stream ops through the awaiting path).
- `GenerateContentResponse.text` is a computed `@property` (derived from `candidates`), NOT a stored field — `model_dump()` excludes it, so it is lost by the generic serializer unless captured explicitly (this is what G6 solves).

## Global Constraints

- OpenAI, Anthropic, and Mistral behavior, recorded shapes, and the frozen contract tests stay unchanged. All new `ProviderSpec` fields default to empty; existing specs untouched. With `response_attributes=frozenset()` the serializer is byte-for-byte identical to today.
- Core imports only the standard library; `google-genai` is imported lazily (never at `import agent_cassette`). `integrations/gemini.py` must not import `google.genai` at module top level; `patch_gemini` imports the module lazily on entry.
- Public API additions mirrored in `agent_cassette.__all__`, `docs/public-api.md`, `tests/test_public_api.py` (`EXPECTED_PUBLIC_API` + the lazy-import blocked set), and `tests/test_contract_snapshots.py` (`EXPECTED_PUBLIC_SIGNATURES`). Additive → SemVer minor.
- Optional extra pin: `google-genai>=1,<2`; CI minimum `google-genai==1.0.0`. Refresh `uv.lock` with `uv lock`.
- Run everything with `uv run --frozen`. New/changed code passes ruff (`E,F,I,UP,B`, line 100) and `uv run --frozen mypy src tests` (Success, 0 errors).

---

### Task 1: Add `response_attributes` to `ProviderSpec` (G6)

**Files:**
- Modify: `src/agent_cassette/integrations/_provider.py` (the `ProviderSpec` dataclass, ~line 25-40)
- Test: `tests/test_provider_generalizations.py`

**Interfaces:**
- Produces: `ProviderSpec(..., response_attributes: frozenset[str] = frozenset())`.

- [ ] **Step 1: Write the failing test**

```python
def test_provider_spec_has_response_attributes_default():
    from agent_cassette.integrations._provider import ProviderSpec
    spec = ProviderSpec(provider="x", operations=frozenset(), prefixes=frozenset())
    assert spec.response_attributes == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py::test_provider_spec_has_response_attributes_default -v`
Expected: FAIL (`TypeError`/`AttributeError`).

- [ ] **Step 3: Add the field**

Add after `async_operations` in the dataclass:

```python
    async_operations: frozenset[str] = frozenset()
    response_attributes: frozenset[str] = frozenset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_cassette/integrations/_provider.py tests/test_provider_generalizations.py
git commit -m "feat(provider): add response_attributes spec field"
```

---

### Task 2: Capture declared response attributes on record and expose on replay (G6)

**Files:**
- Modify: `src/agent_cassette/integrations/_provider.py` — `_serialize_response`, `_serialize_stream`, `_serialize_stream_error`, the `_RecordingStream`/`_AsyncRecordingStream` `_finish`/`_fail` calls, and the `_wrap_create` `serializer=` arguments.
- Test: `tests/test_provider_generalizations.py`

**Interfaces:**
- Consumes: `ProviderSpec.response_attributes` (Task 1).
- Produces: recorded response `data` includes each declared attribute (captured via `getattr(response, attr, None)` when not already a serialized key); replay object exposes them via attribute access.

- [ ] **Step 1: Write the failing test**

```python
def test_response_attributes_are_captured_and_replayed(tmp_path):
    from agent_cassette import Cassette
    from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

    class _Resp:
        def __init__(self, value): self._value = value
        def model_dump(self, mode=None): return {"raw": self._value}
        @property
        def text(self): return self._value.upper()

    class _Models:
        def generate(self, **kw): return _Resp(kw["contents"])

    class _Client:
        def __init__(self): self.models = _Models()

    spec = ProviderSpec(
        provider="demo",
        operations=frozenset({"models.generate"}),
        prefixes=frozenset({"models"}),
        response_attributes=frozenset({"text"}),
    )
    path = tmp_path / "demo.jsonl"
    with Cassette.record(path) as c:
        recorded = wrap_provider(_Client(), c, spec).models.generate(contents="works")
    assert recorded.text == "WORKS"
    with Cassette.replay(path) as c:
        replayed = wrap_provider(None, c, spec).models.generate(contents="works")
    assert replayed.text == "WORKS"      # computed property survives replay
    assert replayed.raw == "works"       # serialized field still present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py::test_response_attributes_are_captured_and_replayed -v`
Expected: FAIL — `replayed.text` raises `AttributeError` (property not serialized).

- [ ] **Step 3: Make serialization capture declared attributes**

Change `_serialize_response` to accept the attribute set and merge captured values into `data`:

```python
def _serialize_response(
    response: Any, response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    response_type = type(response)
    data = _to_data(response)
    if response_attributes and isinstance(data, dict):
        for attribute in response_attributes:
            if attribute not in data:
                data[attribute] = _to_data(getattr(response, attribute, None))
    return {
        "__agent_cassette_response__": True,
        "module": response_type.__module__,
        "class": response_type.__qualname__,
        "data": data,
    }
```

Thread the attribute set through the stream serializers:

```python
def _serialize_stream(
    chunks: list[Any], response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk, response_attributes) for chunk in chunks],
    }


def _serialize_stream_error(
    chunks: list[Any], error: Exception, response_attributes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    return {
        "__agent_cassette_stream__": True,
        "chunks": [_serialize_response(chunk, response_attributes) for chunk in chunks],
        "error": {"type": type(error).__name__, "message": str(error)},
    }
```

In `_RecordingStream._finish`/`_fail` and `_AsyncRecordingStream._finish`/`_fail`, pass `self._spec.response_attributes` to `_serialize_stream(...)` / `_serialize_stream_error(...)`. In `_record_stream_start_error`, pass `spec.response_attributes` too.

In `_wrap_create`, both `sync_create` and `async_create` non-stream branches pass a spec-bound serializer to `cassette.call`/`cassette.acall`:

```python
                serializer=lambda response: _serialize_response(
                    response, spec.response_attributes
                ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — existing providers unchanged**

Run: `uv run --frozen pytest tests/test_openai_integration.py tests/test_anthropic_integration.py tests/test_mistral_integration.py tests/test_streaming_spans.py -q && uv run --frozen mypy src tests`
Expected: PASS; mypy clean. (Empty `response_attributes` ⇒ `_serialize_response` output identical to before.)

- [ ] **Step 6: Commit**

```bash
git add src/agent_cassette/integrations/_provider.py tests/test_provider_generalizations.py
git commit -m "feat(provider): capture declared response attributes for replay"
```

---

### Task 3: `GEMINI_SPEC` and `wrap_gemini`

**Files:**
- Create: `src/agent_cassette/integrations/gemini.py`
- Test: `tests/test_gemini_integration.py` (create)

**Interfaces:**
- Consumes: `ProviderSpec`, `wrap_provider`.
- Produces: `GEMINI_SPEC`; `wrap_gemini(client, cassette, *, asynchronous=None) -> Client`; `GeminiStreamingUnsupportedError`, `GeminiRawResponseUnsupportedError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gemini_integration.py
import asyncio

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
        replayed = wrap_gemini(None, c).models.generate_content(
            model="gemini-2.0-flash", contents="works"
        )
    assert recorded.text == replayed.text == "WORKS"


def test_gemini_async_generate(tmp_path):
    path = tmp_path / "g.jsonl"
    req = {"model": "gemini-2.0-flash", "contents": "works"}
    with Cassette.record(path) as c:
        recorded = asyncio.run(wrap_gemini(_Client(), c).aio.models.generate_content(**req))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(wrap_gemini(None, c).aio.models.generate_content(**req))
    assert recorded.text == replayed.text == "WORKS"


def test_gemini_sync_stream(tmp_path):
    path = tmp_path / "g.jsonl"
    with Cassette.record(path) as c:
        recorded = [ch.text for ch in wrap_gemini(_Client(), c).models.generate_content_stream(
            model="m", contents="x")]
    with Cassette.replay(path) as c:
        replayed = [ch.text for ch in wrap_gemini(None, c).models.generate_content_stream(
            model="m", contents="x")]
    assert recorded == replayed == ["A", "B"]


def test_gemini_async_stream(tmp_path):
    path = tmp_path / "g.jsonl"

    async def drive(client):
        return [ch.text async for ch in await client.aio.models.generate_content_stream(
            model="m", contents="x")]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_gemini(_Client(), c)))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(drive(wrap_gemini(None, c)))
    assert recorded == replayed == ["A", "B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --frozen pytest tests/test_gemini_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: agent_cassette.integrations.gemini`.

- [ ] **Step 3: Create `integrations/gemini.py`**

```python
"""Transparent record and replay wrapper for the google-genai client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class GeminiStreamingUnsupportedError(NotImplementedError):
    """Raised when a Gemini stream does not expose the expected iterator API."""


class GeminiRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


GEMINI_SPEC = ProviderSpec(
    provider="gemini",
    operations=frozenset(
        {
            "models.generate_content",
            "models.generate_content_stream",
            "aio.models.generate_content",
            "aio.models.generate_content_stream",
        }
    ),
    prefixes=frozenset({"models", "aio", "aio.models"}),
    stream_operations=frozenset(
        {"models.generate_content_stream", "aio.models.generate_content_stream"}
    ),
    async_operations=frozenset(
        {"aio.models.generate_content", "aio.models.generate_content_stream"}
    ),
    response_attributes=frozenset({"text"}),
    async_probe_path=(),
    streaming_error=GeminiStreamingUnsupportedError,
    raw_response_error=GeminiRawResponseUnsupportedError,
)


def wrap_gemini(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a google-genai client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. Sync operations live under
    ``client.models`` and async ones under ``client.aio.models``; async routing is
    per operation via ``GEMINI_SPEC.async_operations``. ``response.text`` is captured
    at record time (``response_attributes``) so it survives offline replay.
    """
    return wrap_provider(client, cassette, GEMINI_SPEC, asynchronous=asynchronous)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_gemini_integration.py -v`
Expected: PASS (all five). If mypy on the test needs `# type: ignore` for the `wrap_gemini(None, ...)` generic-`None` pattern, add narrowly-scoped `# type: ignore[...]` matching the convention in `tests/test_mistral_integration.py`.

- [ ] **Step 5: Full gate + commit**

Run: `uv run --frozen pytest -q && uv run --frozen ruff check src tests && uv run --frozen mypy src tests`
Expected: all pass.

```bash
git add src/agent_cassette/integrations/gemini.py tests/test_gemini_integration.py
git commit -m "feat(gemini): add GEMINI_SPEC and wrap_gemini"
```

---

### Task 4: `patch_gemini` (single-constructor auto-patch)

**Files:**
- Modify: `src/agent_cassette/automatic.py`
- Test: `tests/test_gemini_integration.py`

**Interfaces:**
- Consumes: `wrap_gemini` (Task 3), the existing `_patch_single_constructor`, `_load_module`, `_PATCH_LOCK`, `_PATCHED_MODULES`.
- Produces: `patch_gemini(cassette) -> ContextManager[None]`; `GeminiUnavailableError`, `GeminiAlreadyPatchedError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemini_integration.py (append)
import sys
import types

import agent_cassette.automatic as automatic


def _install_fake_google_genai(monkeypatch):
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # mark as package
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _Client
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    return genai_mod


def test_patch_gemini_wraps_constructor(tmp_path, monkeypatch):
    module = _install_fake_google_genai(monkeypatch)
    path = tmp_path / "g.jsonl"
    req = {"model": "gemini-2.0-flash", "contents": "works"}
    with Cassette.record(path) as c, automatic.patch_gemini(c):
        recorded = module.Client(api_key="x").models.generate_content(**req)
    with Cassette.replay(path) as c, automatic.patch_gemini(c):
        replayed = module.Client(api_key="x").models.generate_content(**req)
    assert recorded.text == replayed.text == "WORKS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_gemini_integration.py::test_patch_gemini_wraps_constructor -v`
Expected: FAIL — `AttributeError: module 'agent_cassette.automatic' has no attribute 'patch_gemini'`.

- [ ] **Step 3: Add `patch_gemini`**

In `automatic.py`, add the error classes and the context manager (reusing `_patch_single_constructor`, imported `wrap_gemini` lazily):

```python
class GeminiUnavailableError(ImportError):
    """Raised when automatic Gemini support is requested without google-genai installed."""


class GeminiAlreadyPatchedError(RuntimeError):
    """Raised when the same google-genai module is patched by a nested context."""


@contextmanager
def patch_gemini(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``google.genai.Client``.

    The single client carries both sync and async (``client.aio``) operations; the
    constructor is wrapped with ``asynchronous=False`` and per-operation async routing
    is handled by ``GEMINI_SPEC``.
    """
    from agent_cassette.integrations.gemini import wrap_gemini

    with _patch_single_constructor(
        cassette,
        "google.genai",
        "Client",
        wrap_gemini,
        GeminiUnavailableError,
        GeminiAlreadyPatchedError,
    ):
        yield
```

Add `"GeminiAlreadyPatchedError"`, `"GeminiUnavailableError"`, `"patch_gemini"` to `automatic.py`'s `__all__`. Add `google-genai` to the module docstring's list of lazily-imported optional SDKs.

- [ ] **Step 4: Run test + gate**

Run: `uv run --frozen pytest tests/test_gemini_integration.py -q && uv run --frozen ruff check src tests && uv run --frozen mypy src tests`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/agent_cassette/automatic.py tests/test_gemini_integration.py
git commit -m "feat(gemini): add patch_gemini single-constructor auto-patch"
```

---

### Task 5: Export the public API, update contract snapshots and the lazy-import guard

**Files:**
- Modify: `src/agent_cassette/__init__.py`, `docs/public-api.md`, `tests/test_public_api.py`, `tests/test_contract_snapshots.py`

**Interfaces:**
- Consumes: `wrap_gemini` (Task 3), `patch_gemini` (Task 4).
- Produces: `agent_cassette.wrap_gemini`, `agent_cassette.patch_gemini` in `__all__`.

- [ ] **Step 1: Update expected sets first (red)**

In `tests/test_public_api.py`: add `"patch_gemini"` and `"wrap_gemini"` to `EXPECTED_PUBLIC_API`, and add `"google"` to the `blocked` set in `test_core_import_does_not_load_optional_dependencies` (the finder matches the top-level package; `google.genai` imports under the `google` namespace — importing `agent_cassette` must not pull it).

In `tests/test_contract_snapshots.py`, add to `EXPECTED_PUBLIC_SIGNATURES`:

```python
    "wrap_gemini": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
    "patch_gemini": "(cassette: 'Any') -> 'Iterator[None]'",
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --frozen pytest tests/test_public_api.py tests/test_contract_snapshots.py -q`
Expected: FAIL — names/signatures not yet exported.

- [ ] **Step 3: Export from `__init__.py`**

Add `patch_gemini` to the `agent_cassette.automatic` import, add `from agent_cassette.integrations.gemini import wrap_gemini`, and add `"patch_gemini"` / `"wrap_gemini"` to `__all__` (sorted within their local neighborhood). Update the "Providers/frameworks" row of `docs/public-api.md`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_public_api.py tests/test_contract_snapshots.py -q`
Expected: PASS, including `test_core_import_does_not_load_optional_dependencies` (proves `import agent_cassette` does not import anything under `google`).

- [ ] **Step 5: Full gate + commit**

Run: `uv run --frozen pytest -q && uv run --frozen ruff check src tests && uv run --frozen mypy src tests`

```bash
git add src/agent_cassette/__init__.py docs/public-api.md tests/test_public_api.py tests/test_contract_snapshots.py
git commit -m "feat(gemini): export wrap_gemini and patch_gemini"
```

---

### Task 6: Packaging — optional extra and lockfile

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Add the extra**

In `[project.optional-dependencies]`: `gemini = ["google-genai>=1,<2"]`, and add `"google-genai>=1,<2"` to the `all` list.

- [ ] **Step 2: Refresh the lockfile**

Run: `uv lock`
Expected: resolves; `uv.lock` gains `google-genai` + transitive deps, no version changes to pre-existing pins. If `uv lock`/`uv sync` cannot reach PyPI, report BLOCKED with the exact error (the `pyproject.toml` edit itself still succeeds); do not commit a partial lockfile.

- [ ] **Step 3: Install + verify**

Run: `uv sync --frozen --all-extras --dev && uv run --frozen python -c "import google.genai; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(gemini): add optional google-genai extra"
```

---

### Task 7: CI matrix rows and installed-wheel smoke

**Files:**
- Modify: `.github/workflows/ci.yml` (`installed-wheel-extras` matrix + inline smoke)

- [ ] **Step 1: Add matrix rows**

In `matrix.include`:

```yaml
          - compatibility: gemini-minimum
            python-version: "3.10"
            extra: gemini
            dependency-spec: "google-genai==1.0.0"
          - compatibility: gemini-current
            python-version: "3.13"
            extra: gemini
            dependency-spec: ""
```

- [ ] **Step 2: Add the smoke branch**

In the inline `PY` smoke, before the final `else:`, add a `gemini` branch mirroring the mistral one — a fake client with `models.generate_content`, record → `disable_network()` → replay, assert `recorded.text == replayed.text`, and `import google.genai` / `assert google.genai`:

```python
          elif case == "gemini":
              import google.genai
              from agent_cassette import wrap_gemini

              class Resp:
                  def __init__(self, value): self._value = value
                  def model_dump(self, mode=None): return {"raw": self._value}
                  @property
                  def text(self): return self._value.upper()

              class Models:
                  def generate_content(self, **kw): return Resp(kw["contents"])

              class Client:
                  def __init__(self): self.models = Models()

              request = {"model": "offline", "contents": "works"}
              with Cassette.record(path) as cassette:
                  recorded = wrap_gemini(Client(), cassette).models.generate_content(**request)
              disable_network()
              with Cassette.replay(path) as cassette:
                  replayed = wrap_gemini(None, cassette).models.generate_content(**request)
              assert recorded.text == replayed.text == "WORKS"
              assert google.genai
```

- [ ] **Step 3: Validate YAML**

Run: `uvx --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(gemini): add min/current matrix rows and offline smoke"
```

---

### Task 8: User docs, adding-a-provider note, and changelog

**Files:**
- Modify: `docs/integrations.md`, `docs/compatibility.md`, `docs/adding-a-provider.md`, `CHANGELOG.md`

- [ ] **Step 1: Integrations**

In `docs/integrations.md`, add a "Gemini" section: install `agent-cassette[gemini]`, automatic patching via `patch_gemini` (or explicit `wrap_gemini`), supported ops (sync/async `generate_content`, sync/async `generate_content_stream`), and that `response.text` is preserved on replay.

- [ ] **Step 2: Compatibility**

In `docs/compatibility.md`, add Gemini with tested range `google-genai >=1,<2` (minimum `1.0.0`, current), matching the other rows.

- [ ] **Step 3: adding-a-provider note**

In `docs/adding-a-provider.md`, add a short note documenting `response_attributes` (capture computed response properties like `text` so they survive offline replay) alongside the other `ProviderSpec` fields, with Gemini as the example.

- [ ] **Step 4: Changelog**

Under `## [Unreleased]` `### Added`:

```markdown
- Gemini (`google-genai`) provider support (`wrap_gemini`, `patch_gemini`,
  `agent-cassette[gemini]`): sync/async `generate_content` and streaming
  `generate_content_stream`, with `response.text` preserved on replay.
- Provider foundation: `response_attributes` capture for computed response
  properties.
```

- [ ] **Step 5: Full gate**

Run:
```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests examples benchmarks
uv run --frozen ruff format --check src tests examples benchmarks
uv run --frozen mypy src tests
uv build --no-build-isolation
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add docs/integrations.md docs/compatibility.md docs/adding-a-provider.md CHANGELOG.md
git commit -m "docs(gemini): integration guide, compatibility, and changelog"
```

---

## Self-Review

**Spec coverage:** G6 (`response_attributes`) → Tasks 1,2; Gemini spec/wrap (sync+async+streaming via existing G2/G5 + async-stream path + aio subtree enumeration) → Task 3; single-constructor patch → Task 4; public-API contract + lazy-import guard → Task 5; packaging → Task 6; CI → Task 7; docs → Task 8. Async `.aio` subtree needs no new proxy field (reuses G5), as designed. Covered.

**Placeholder scan:** every code step contains real code; no TBD/TODO.

**Type consistency:** `wrap_gemini(client, cassette, *, asynchronous=None)` matches across Task 3, the contract snapshot (Task 5), and `patch_gemini(cassette)`. `response_attributes` field name is consistent across Tasks 1, 2, 3. `GEMINI_SPEC` operation/prefix/async/stream/response-attr names are internally consistent and match the SDK facts. `_patch_single_constructor` reuses the existing helper from the Mistral phase.

## Execution Handoff

Not started — subagent-driven execution to follow.
