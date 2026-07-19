# Mistral Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add record/replay support for the `mistralai` SDK — sync (`chat.complete`), async (`chat.complete_async`), and streaming (`chat.stream` / `chat.stream_async`) — as the first phase of the multi-provider foundation.

**Architecture:** Extend the provider-agnostic proxy in `_provider.py` with three additive, default-off generalizations — per-operation async (G5), stream-operation triggering (G2), and context-manager-tolerant streams — then define a `MISTRAL_SPEC`, `wrap_mistral`, and a single-constructor `patch_mistral`. OpenAI/Anthropic behavior stays byte-for-byte identical.

**Tech Stack:** Python ≥3.10, `uv`, pytest, ruff, mypy, hatchling. Optional dep `mistralai>=1,<2`.

**Design source:** `docs/superpowers/specs/2026-07-18-multi-provider-foundation-design.md`.

## Global Constraints

- OpenAI and Anthropic behavior, recorded shapes, and their frozen contract tests stay unchanged. All new `ProviderSpec` fields default to empty.
- Core imports only the standard library; `mistralai` is imported lazily and only inside `patch_mistral` / when a live client is wrapped. `tests/test_public_api.py::test_core_import_does_not_load_optional_dependencies` must stay green.
- New source (`integrations/mistral.py`, `automatic.py`, `_provider.py`) is type-checked by mypy (not in the override exclusion list) and linted by ruff (`E,F,I,UP,B`, line length 100).
- Public API additions must be mirrored in `agent_cassette.__all__`, `docs/public-api.md`, `tests/test_public_api.py`, and `tests/test_contract_snapshots.py`. Additive only → SemVer minor.
- Optional extra pin: `mistralai>=1,<2`. Refresh `uv.lock` with `uv lock` after editing `pyproject.toml`.
- Run everything with `uv run --frozen`.

---

### Task 1: Add `stream_operations` and `async_operations` to `ProviderSpec`

**Files:**
- Modify: `src/agent_cassette/integrations/_provider.py:25-40`
- Test: `tests/test_provider_generalizations.py` (create)

**Interfaces:**
- Produces: `ProviderSpec(..., stream_operations: frozenset[str] = frozenset(), async_operations: frozenset[str] = frozenset())`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_generalizations.py
from agent_cassette.integrations._provider import ProviderSpec
from agent_cassette.integrations.openai import OPENAI_SPEC


def test_provider_spec_has_empty_generalization_defaults():
    spec = ProviderSpec(provider="x", operations=frozenset(), prefixes=frozenset())
    assert spec.stream_operations == frozenset()
    assert spec.async_operations == frozenset()


def test_existing_specs_do_not_use_new_fields():
    assert OPENAI_SPEC.stream_operations == frozenset()
    assert OPENAI_SPEC.async_operations == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: FAIL with `TypeError` / `AttributeError` (fields do not exist).

- [ ] **Step 3: Add the fields**

In the `ProviderSpec` dataclass, after `unsupported_operations`:

```python
    unsupported_operations: dict[str, str] = field(default_factory=dict)
    stream_operations: frozenset[str] = frozenset()
    async_operations: frozenset[str] = frozenset()
    derive_methods: frozenset[str] = frozenset({"with_options", "copy"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_cassette/integrations/_provider.py tests/test_provider_generalizations.py
git commit -m "feat(provider): add stream_operations and async_operations spec fields"
```

---

### Task 2: Per-operation async (G5) and stream-operation triggering (G2)

**Files:**
- Modify: `src/agent_cassette/integrations/_provider.py:310-421` (`_ResourceProxy._wrap_create`)
- Test: `tests/test_provider_generalizations.py`

**Interfaces:**
- Consumes: `ProviderSpec.async_operations`, `ProviderSpec.stream_operations` (Task 1).
- Produces: an operation wraps async when `self._asynchronous or operation in spec.async_operations`; treats the call as streaming when `operation in spec.stream_operations or kwargs.get("stream")`.

- [ ] **Step 1: Write the failing test**

```python
import asyncio

from agent_cassette import Cassette, EventType  # noqa: F401
from agent_cassette.integrations._provider import ProviderSpec, wrap_provider


class _Resp:
    def __init__(self, text): self.text = text
    def model_dump(self, mode=None): return {"text": self.text}


class _Chat:
    def go(self, **kw): return _Resp("SYNC")
    async def go_async(self, **kw): return _Resp("ASYNC")


class _Client:
    def __init__(self): self.chat = _Chat()


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py::test_sync_and_async_ops_coexist_on_one_client -v`
Expected: FAIL — `chat.go_async` is wrapped synchronously (returns a coroutine that is never awaited / replay routes wrong), raising an assertion or runtime error.

- [ ] **Step 3: Implement per-op async + stream trigger**

In `_ResourceProxy._wrap_create`, replace the branch selector and both stream guards:

```python
    def _wrap_create(self, create: Callable[..., Any] | None, operation: str) -> Callable[..., Any]:
        spec = self._spec
        event_name = spec.event_name(operation)
        is_async = self._asynchronous or operation in spec.async_operations
        stream_operation = operation in spec.stream_operations
        if is_async:

            async def async_create(*args: Any, **kwargs: Any) -> Any:
                request = _serialize_request(args, kwargs)
                if stream_operation or kwargs.get("stream"):
                    # ... existing async streaming body unchanged ...
```

And in `sync_create`:

```python
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            request = _serialize_request(args, kwargs)
            if stream_operation or kwargs.get("stream"):
                # ... existing sync streaming body unchanged ...
```

Only the three conditions change (`if is_async:`, and the two `if stream_operation or kwargs.get("stream"):`). Leave the recording/replay bodies as-is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — existing providers unchanged**

Run: `uv run --frozen pytest tests/test_openai_integration.py tests/test_anthropic_integration.py tests/test_streaming_spans.py -q`
Expected: PASS (no behavior change for whole-client-async providers).

- [ ] **Step 6: Commit**

```bash
git add src/agent_cassette/integrations/_provider.py tests/test_provider_generalizations.py
git commit -m "feat(provider): per-operation async and stream-operation triggering"
```

---

### Task 3: Context-manager-tolerant streams (record + replay)

**Files:**
- Modify: `src/agent_cassette/integrations/_provider.py` — `_RecordingStream.__init__` (53-71), `_AsyncRecordingStream.__init__` (141-159), `_ReplayStream` (237-252), `_AsyncReplayStream` (255-270)
- Test: `tests/test_provider_generalizations.py`

**Interfaces:**
- Produces: recording streams accept a stream object that is only iterable via its `__enter__`/`__aenter__` result; `_ReplayStream`/`_AsyncReplayStream` support the (a)sync context-manager protocol returning `self`.

- [ ] **Step 1: Write the failing test**

```python
class _EventCM:
    """Mimics mistralai: a context manager whose entered value is the iterator."""
    def __init__(self, events): self._events = events
    def __enter__(self): return iter(self._events)
    def __exit__(self, *a): return False


class _StreamChat:
    def stream(self, **kw): return _EventCM([_Resp("A"), _Resp("B")])


class _StreamClient:
    def __init__(self): self.chat = _StreamChat()


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
        with client.chat.stream(model="m", messages=[]) as events:
            replayed = [e.text for e in events]
    assert replayed == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py::test_context_manager_stream_records_and_replays -v`
Expected: FAIL — `_RecordingStream.__init__` calls `iter(stream)` on the CM (TypeError → streaming_error), and `_ReplayStream` has no `__enter__`.

- [ ] **Step 3: Defer iteration in recording streams**

`_RecordingStream.__init__`: replace `self._iterator = iter(stream)` with tolerant init:

```python
        self._stream = stream
        try:
            self._iterator: Any = iter(stream)
        except TypeError:
            self._iterator = None  # entered lazily via __enter__
```

`_RecordingStream.__next__`: guard the un-entered case:

```python
    def __next__(self) -> Any:
        if self._iterator is None:
            raise TypeError("stream must be entered as a context manager before iteration")
        try:
            chunk = next(self._iterator)
        ...
```

Apply the analogous change to `_AsyncRecordingStream.__init__` (wrap `stream.__aiter__()` in `try/except (TypeError, AttributeError)` → `self._iterator = None`) and `__anext__` guard.

- [ ] **Step 4: Add context-manager protocol to replay streams**

```python
class _ReplayStream(Iterator[Any]):
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __iter__(self) -> _ReplayStream:
        return self

    def __enter__(self) -> _ReplayStream:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        return None

    def __next__(self) -> Any:
        ...  # unchanged
```

```python
class _AsyncReplayStream(AsyncIterator[Any]):
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __aiter__(self) -> _AsyncReplayStream:
        return self

    async def __aenter__(self) -> _AsyncReplayStream:
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        return None

    async def __anext__(self) -> Any:
        ...  # unchanged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_provider_generalizations.py -v`
Expected: PASS.

- [ ] **Step 6: Regression + types**

Run: `uv run --frozen pytest tests/test_streaming_spans.py -q && uv run --frozen mypy src`
Expected: PASS; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/agent_cassette/integrations/_provider.py tests/test_provider_generalizations.py
git commit -m "feat(provider): tolerate context-manager-only streams on record and replay"
```

---

### Task 4: `MISTRAL_SPEC` and `wrap_mistral`

**Files:**
- Create: `src/agent_cassette/integrations/mistral.py`
- Test: `tests/test_mistral_integration.py` (create)

**Interfaces:**
- Consumes: `ProviderSpec`, `wrap_provider` (existing).
- Produces: `MISTRAL_SPEC: ProviderSpec`; `wrap_mistral(client, cassette, *, asynchronous=None) -> Client`; `MistralStreamingUnsupportedError`, `MistralRawResponseUnsupportedError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mistral_integration.py
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
        replayed = wrap_mistral(None, c).chat.complete(
            model="mistral-small", messages=[{"role": "user", "content": "works"}]
        )
    assert recorded.text == replayed.text == "WORKS"


def test_mistral_async_complete(tmp_path):
    path = tmp_path / "m.jsonl"
    request = {"model": "mistral-small", "messages": [{"role": "user", "content": "works"}]}
    with Cassette.record(path) as c:
        recorded = asyncio.run(wrap_mistral(_Mistral(), c).chat.complete_async(**request))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(wrap_mistral(None, c).chat.complete_async(**request))
    assert recorded.text == replayed.text == "WORKS"


def test_mistral_sync_stream(tmp_path):
    path = tmp_path / "m.jsonl"
    with Cassette.record(path) as c:
        with wrap_mistral(_Mistral(), c).chat.stream(model="m", messages=[]) as events:
            recorded = [e.text for e in events]
    with Cassette.replay(path) as c:
        with wrap_mistral(None, c).chat.stream(model="m", messages=[]) as events:
            replayed = [e.text for e in events]
    assert recorded == replayed == ["A", "B"]


def test_mistral_async_stream(tmp_path):
    path = tmp_path / "m.jsonl"

    async def drive(client):
        async with client.chat.stream_async(model="m", messages=[]) as events:
            return [e.text async for e in events]

    with Cassette.record(path) as c:
        recorded = asyncio.run(drive(wrap_mistral(_Mistral(), c)))
    with Cassette.replay(path) as c:
        replayed = asyncio.run(drive(wrap_mistral(None, c)))
    assert recorded == replayed == ["A", "B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --frozen pytest tests/test_mistral_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: agent_cassette.integrations.mistral`.

- [ ] **Step 3: Create `integrations/mistral.py`**

```python
"""Transparent record and replay wrapper for the Mistral Python client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class MistralStreamingUnsupportedError(NotImplementedError):
    """Raised when a Mistral stream does not expose the expected iterator API."""


class MistralRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


MISTRAL_SPEC = ProviderSpec(
    provider="mistral",
    operations=frozenset(
        {"chat.complete", "chat.complete_async", "chat.stream", "chat.stream_async"}
    ),
    prefixes=frozenset({"chat"}),
    stream_operations=frozenset({"chat.stream", "chat.stream_async"}),
    async_operations=frozenset({"chat.complete_async", "chat.stream_async"}),
    async_probe_path=(),
    streaming_error=MistralStreamingUnsupportedError,
    raw_response_error=MistralRawResponseUnsupportedError,
)


def wrap_mistral(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a Mistral client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. Sync and async operations
    (``chat.complete`` / ``chat.complete_async``) coexist on one client; async
    routing is driven per operation by ``MISTRAL_SPEC.async_operations``.
    """
    return wrap_provider(client, cassette, MISTRAL_SPEC, asynchronous=asynchronous)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_mistral_integration.py -v`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add src/agent_cassette/integrations/mistral.py tests/test_mistral_integration.py
git commit -m "feat(mistral): add MISTRAL_SPEC and wrap_mistral"
```

---

### Task 5: `patch_mistral` with single-constructor auto-patch

**Files:**
- Modify: `src/agent_cassette/automatic.py`
- Test: `tests/test_automatic.py` (add a Mistral case) or `tests/test_mistral_integration.py`

**Interfaces:**
- Consumes: `wrap_mistral` (Task 4), `_load_module`, `_wrapped_constructor`, `_PATCH_LOCK`, `_PATCHED_MODULES` (existing in `automatic.py`).
- Produces: `patch_mistral(cassette) -> ContextManager[None]`; `MistralUnavailableError`, `MistralAlreadyPatchedError`; helper `_patch_single_constructor(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mistral_integration.py (append)
import sys
import types

import agent_cassette.automatic as automatic
from agent_cassette import Cassette


def _install_fake_mistralai(monkeypatch):
    module = types.ModuleType("mistralai")
    module.Mistral = _Mistral  # from earlier in this file
    monkeypatch.setitem(sys.modules, "mistralai", module)
    return module


def test_patch_mistral_wraps_constructor(tmp_path, monkeypatch):
    module = _install_fake_mistralai(monkeypatch)
    path = tmp_path / "m.jsonl"
    request = {"model": "m", "messages": [{"role": "user", "content": "works"}]}
    with Cassette.record(path) as c, automatic.patch_mistral(c):
        recorded = module.Mistral(api_key="x").chat.complete(**request)
    with Cassette.replay(path) as c, automatic.patch_mistral(c):
        replayed = module.Mistral(api_key="x").chat.complete(**request)
    assert recorded.text == replayed.text == "WORKS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_mistral_integration.py::test_patch_mistral_wraps_constructor -v`
Expected: FAIL with `AttributeError: module 'agent_cassette.automatic' has no attribute 'patch_mistral'`.

- [ ] **Step 3: Add single-constructor patch + `patch_mistral`**

In `automatic.py`, add error classes and a single-constructor context manager modeled on `_patch_constructors`:

```python
class MistralUnavailableError(ImportError):
    """Raised when automatic Mistral support is requested without Mistral installed."""


class MistralAlreadyPatchedError(RuntimeError):
    """Raised when the same Mistral module is patched by a nested context."""


@contextmanager
def _patch_single_constructor(
    cassette: Any,
    module_name: str,
    constructor_name: str,
    wrapper: Callable[..., Any],
    unavailable_error: type[ImportError],
    already_patched_error: type[RuntimeError],
) -> Iterator[None]:
    module = _load_module(module_name, unavailable_error)
    module_identity = (module_name, id(module))
    try:
        original = getattr(module, constructor_name)
    except AttributeError as error:
        raise unavailable_error(
            f"The installed '{module_name}' package does not expose {constructor_name}."
        ) from error
    with _PATCH_LOCK:
        if module_identity in _PATCHED_MODULES:
            raise already_patched_error(
                f"{module_name} constructors are already patched; nested patch contexts "
                "for the same module are not supported."
            )
        _PATCHED_MODULES.add(module_identity)
        setattr(
            module,
            constructor_name,
            _wrapped_constructor(original, cassette, wrapper, asynchronous=False),
        )
    try:
        yield
    finally:
        with _PATCH_LOCK:
            setattr(module, constructor_name, original)
            _PATCHED_MODULES.remove(module_identity)


@contextmanager
def patch_mistral(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``mistralai.Mistral``.

    The single Mistral client carries both sync and async operations; the
    constructor is wrapped with ``asynchronous=False`` and per-operation async
    routing is handled by ``MISTRAL_SPEC``.
    """
    from agent_cassette.integrations.mistral import wrap_mistral

    with _patch_single_constructor(
        cassette,
        "mistralai",
        "Mistral",
        wrap_mistral,
        MistralUnavailableError,
        MistralAlreadyPatchedError,
    ):
        yield
```

Add `"MistralAlreadyPatchedError"`, `"MistralUnavailableError"`, `"patch_mistral"` to `automatic.py`'s `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_mistral_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_cassette/automatic.py tests/test_mistral_integration.py
git commit -m "feat(mistral): add patch_mistral single-constructor auto-patch"
```

---

### Task 6: Export the public API and update contract snapshots

**Files:**
- Modify: `src/agent_cassette/__init__.py`
- Modify: `docs/public-api.md`
- Modify: `tests/test_public_api.py:10-60` (`EXPECTED_PUBLIC_API`)
- Modify: `tests/test_contract_snapshots.py` (`EXPECTED_PUBLIC_SIGNATURES`)

**Interfaces:**
- Consumes: `wrap_mistral` (Task 4), `patch_mistral` (Task 5).
- Produces: `agent_cassette.wrap_mistral`, `agent_cassette.patch_mistral` in `__all__`.

- [ ] **Step 1: Update the public-API test first (red)**

Add `"patch_mistral"` and `"wrap_mistral"` to `EXPECTED_PUBLIC_API` in `tests/test_public_api.py`. Add to `EXPECTED_PUBLIC_SIGNATURES` in `tests/test_contract_snapshots.py`:

```python
    "wrap_mistral": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
```

(`patch_mistral` is a `@contextmanager` function; add it too:)

```python
    "patch_mistral": "(cassette: 'Any') -> 'Iterator[None]'",
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --frozen pytest tests/test_public_api.py tests/test_contract_snapshots.py -q`
Expected: FAIL — names/signatures not yet exported.

- [ ] **Step 3: Export from `__init__.py`**

Add imports near the other integration imports:

```python
from agent_cassette.automatic import (
    automatic_openai_from_env,
    patch_anthropic,
    patch_mistral,
    patch_openai,
)
from agent_cassette.integrations.mistral import wrap_mistral
```

(Adjust the existing `automatic` import line to include `patch_mistral`.) Add `"patch_mistral"` and `"wrap_mistral"` to `__all__` (keep it sorted).

- [ ] **Step 4: Update `docs/public-api.md`**

In the "Providers/frameworks" row, add `wrap_mistral` and `patch_mistral`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_public_api.py tests/test_contract_snapshots.py -q`
Expected: PASS.

- [ ] **Step 6: Verify lazy-import invariant still holds**

Run: `uv run --frozen pytest tests/test_public_api.py::test_core_import_does_not_load_optional_dependencies -q`
Expected: PASS (importing `agent_cassette` must not import `mistralai`; the `wrap_mistral` import is a light module that only imports `mistralai` when a live client is used, and `patch_mistral` imports it lazily inside the function).

Note: if `integrations/mistral.py` ever imports `mistralai` at module top level, this test fails — it must not.

- [ ] **Step 7: Commit**

```bash
git add src/agent_cassette/__init__.py docs/public-api.md tests/test_public_api.py tests/test_contract_snapshots.py
git commit -m "feat(mistral): export wrap_mistral and patch_mistral"
```

---

### Task 7: Packaging — optional extra and lockfile

**Files:**
- Modify: `pyproject.toml:26-43`
- Modify: `uv.lock` (generated)

**Interfaces:**
- Produces: `pip install "agent-cassette[mistral]"`.

- [ ] **Step 1: Add the extra**

In `[project.optional-dependencies]`:

```toml
mistral = ["mistralai>=1,<2"]
```

And add `"mistralai>=1,<2"` to the `all` list.

- [ ] **Step 2: Refresh the lockfile**

Run: `uv lock`
Expected: `Resolved N packages`; `uv.lock` updated with `mistralai`.

- [ ] **Step 3: Install and verify import**

Run: `uv sync --frozen --all-extras --dev && uv run --frozen python -c "import mistralai; print(mistralai.__name__)"`
Expected: prints `mistralai`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(mistral): add optional mistralai extra"
```

---

### Task 8: CI matrix rows and installed-wheel smoke

**Files:**
- Modify: `.github/workflows/ci.yml:128-359` (`installed-wheel-extras` matrix + smoke script)

**Interfaces:**
- Consumes: the installed wheel + `wrap_mistral`.

- [ ] **Step 1: Add matrix rows**

In the `installed-wheel-extras` `matrix.include` list add:

```yaml
          - compatibility: mistral-minimum
            python-version: "3.10"
            extra: mistral
            dependency-spec: "mistralai==1.0.0"
          - compatibility: mistral-current
            python-version: "3.13"
            extra: mistral
            dependency-spec: ""
```

- [ ] **Step 2: Add the smoke branch**

In the inline `PY` smoke (the `if case == ...` chain), add before the `else:` a `mistral` branch that records + replays against a fake client (the CI smoke may not use live keys). Because the installed wheel test imports the real `mistralai` only to assert its presence, mirror the OpenAI branch shape:

```python
          elif case == "mistral":
              import mistralai
              from agent_cassette import wrap_mistral

              class Resp:
                  def __init__(self, text): self.text = text
                  def model_dump(self, mode=None): return {"text": self.text}

              class Chat:
                  def complete(self, **kw): return Resp(kw["messages"][0]["content"].upper())

              class Client:
                  def __init__(self): self.chat = Chat()

              request = {"model": "offline", "messages": [{"role": "user", "content": "works"}]}
              with Cassette.record(path) as cassette:
                  recorded = wrap_mistral(Client(), cassette).chat.complete(**request)
              disable_network()
              with Cassette.replay(path) as cassette:
                  replayed = wrap_mistral(None, cassette).chat.complete(**request)
              assert recorded.text == replayed.text == "WORKS"
              assert mistralai.__name__ == "mistralai"
```

- [ ] **Step 3: Validate the workflow YAML**

Run: `uvx --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(mistral): add min/current matrix rows and offline smoke"
```

---

### Task 9: `docs/adding-a-provider.md` (the reusable recipe)

**Files:**
- Create: `docs/adding-a-provider.md`
- Modify: `README.md` (Docs table row)

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the recipe**

Create `docs/adding-a-provider.md` documenting the 8 steps from the design spec, using Mistral as the worked example, including: choosing `operations`/`prefixes`/`async_operations`/`stream_operations`, `wrap_<p>`, `patch_<p>` (constructor vs single-constructor vs factory), the four files that must be updated for the public-API contract (`__init__.py`, `docs/public-api.md`, `tests/test_public_api.py`, `tests/test_contract_snapshots.py`), the extra + `uv lock`, CI rows, and an offline record→replay test template.

- [ ] **Step 2: Link it**

Add a row to the README Docs table: `| [Adding a provider](docs/adding-a-provider.md) | Extend record/replay to a new SDK |`.

- [ ] **Step 3: Commit**

```bash
git add docs/adding-a-provider.md README.md
git commit -m "docs: add the adding-a-provider recipe"
```

---

### Task 10: User docs and changelog

**Files:**
- Modify: `docs/integrations.md`
- Modify: `docs/compatibility.md`
- Modify: `CHANGELOG.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Document Mistral usage**

In `docs/integrations.md`, add a "Mistral" section mirroring the Anthropic one: install `agent-cassette[mistral]`, note automatic patching via `patch_mistral`, and that `chat.complete`, `chat.complete_async`, `chat.stream`, and `chat.stream_async` are supported.

- [ ] **Step 2: Record supported versions**

In `docs/compatibility.md`, add Mistral with the tested range `mistralai >=1,<2` (minimum `1.0.0`, current) matching the CI matrix.

- [ ] **Step 3: Changelog**

Under `## [Unreleased]` add:

```markdown
### Added
- Mistral provider support (`wrap_mistral`, `patch_mistral`, `agent-cassette[mistral]`):
  sync `chat.complete`, async `chat.complete_async`, and streaming `chat.stream` /
  `chat.stream_async`.
- Provider foundation: per-operation async and stream-operation triggering, and
  context-manager-tolerant record/replay streams.
```

- [ ] **Step 4: Full gate**

Run:
```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests examples benchmarks
uv run --frozen ruff format --check src tests examples benchmarks
uv run --frozen mypy src tests
uv build --no-build-isolation
```
Expected: all green; builds `1.1.0`-line artifacts (version bump is a separate release step, not in this plan).

- [ ] **Step 5: Commit**

```bash
git add docs/integrations.md docs/compatibility.md CHANGELOG.md
git commit -m "docs(mistral): integration guide, compatibility, and changelog"
```

---

## Self-Review

**Spec coverage:** G5 → Tasks 1,2; G2 → Tasks 1,2; context-manager streams → Task 3; per-provider recipe → Tasks 4–7,9; testing strategy (synthetic clients, generalization unit tests, offline record→replay) → Tasks 2,3,4,5,8; public-API contract updates → Task 6; CI matrix → Task 8; docs → Tasks 9,10. G1/G3/G4 are out of scope for Mistral (later phases). Covered.

**Placeholder scan:** every code step contains real code; no TBD/TODO. Streaming bodies in Task 2 reference the existing unchanged blocks explicitly (only the three conditions change), which is precise, not a placeholder.

**Type consistency:** `wrap_mistral(client, cassette, *, asynchronous=None)` signature is identical in Task 4, the contract snapshot (Task 6), and `patch_mistral(cassette)`. `async_operations`/`stream_operations` field names match across Tasks 1, 2, and 4. `_patch_single_constructor` reuses existing `_wrapped_constructor`/`_load_module`/`_PATCH_LOCK` names from `automatic.py`.

## Execution Handoff

Not started — awaiting the execution-mode choice below.
