# Adding a provider

Extend agent-cassette to a new LLM SDK by implementing a transparent record/replay wrapper. This recipe documents the 8 steps using **Mistral** as the worked example.

## Overview

A provider integration has three parts:

- **ProviderSpec** — declarative schema matching your SDK's API shape (operations, prefixes, async routing, streaming)
- **Wrapper** — wraps SDK clients so they record/replay transparently; your code sees no difference
- **Patch context** — temporarily wraps constructors so clients created inside the context are automatically wrapped
- **Public API** — exports the wrapper and patch functions, updates 4 contract files
- **Dependency & CI** — optional extra in `pyproject.toml`, matrix rows in CI
- **Tests & docs** — offline record→replay test (no keys, no network) and integration guides

The full specification framework (`ProviderSpec` generalizations for flat operations, async subtrees, per-method async, factory clients, and streaming variants) is documented in [the design spec](superpowers/specs/2026-07-18-multi-provider-foundation-design.md). This recipe focuses on adding a provider following one of these patterns.

## Step 1: Define the ProviderSpec

Create `src/agent_cassette/integrations/<provider>.py`. Declare a `ProviderSpec` that describes your SDK's call structure:

```python
from agent_cassette.integrations._provider import ProviderSpec

class MyProviderStreamingUnsupportedError(NotImplementedError):
    """Raised when a stream does not expose the expected iterator API."""

class MyProviderRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""

MY_PROVIDER_SPEC = ProviderSpec(
    provider="my_provider",
    operations=frozenset({
        "resource.operation",           # dotted operation names, e.g. "chat.complete"
        # ... include all supported operations
    }),
    prefixes=frozenset({
        "resource",                     # attribute nodes to descend, e.g. "chat"
        # ... intermediate paths
    }),
    stream_operations=frozenset({
        # (optional) operations that are streaming variants, e.g. "chat.stream"
        # instead of "chat.complete(..., stream=True)"
    }),
    async_operations=frozenset({
        # (optional) operations that are async on a single client, e.g. 
        # "chat.complete_async" on the same client as "chat.complete"
    }),
    async_probe_path=(),                # (optional) path to introspect async client type
    response_attributes=frozenset({     # (optional) computed response properties to preserve
        # e.g., "text" for Gemini, which is computed from underlying model response
    }),
    streaming_error=MyProviderStreamingUnsupportedError,
    raw_response_error=MyProviderRawResponseUnsupportedError,
)
```

**Mistral example** ([src/agent_cassette/integrations/mistral.py](../src/agent_cassette/integrations/mistral.py)):
- `operations`: `{chat.complete, chat.complete_async, chat.stream, chat.stream_async}`
- `prefixes`: `{chat}` (client descends to `client.chat`, then to methods)
- `stream_operations`: `{chat.stream, chat.stream_async}` (separate methods, not `stream=True`)
- `async_operations`: `{chat.complete_async, chat.stream_async}` (per-method async on one client)
- `async_probe_path`: `()` (single-client Mistral, not a sync/async class pair)
- `response_attributes`: `{}` (no computed properties to preserve)

**Gemini example** ([src/agent_cassette/integrations/gemini.py](../src/agent_cassette/integrations/gemini.py)):
- `operations`: `{models.generate_content, models.generate_content_stream, aio.models.generate_content, aio.models.generate_content_stream}`
- `prefixes`: `{models, aio, aio.models}` (sync under `models`, async under `aio.models`)
- `stream_operations`: `{models.generate_content_stream, aio.models.generate_content_stream}` (separate methods)
- `async_operations`: `{aio.models.generate_content, aio.models.generate_content_stream}` (async lives under `aio.models` subtree, not `_async` suffix)
- `async_probe_path`: `()` (single-client Gemini, not a sync/async class pair)
- `response_attributes`: `{"text"}` (computed from underlying model response, preserved on replay)

## Step 2: Implement the wrapper function

Add a `wrap_<provider>` function in the same file:

```python
def wrap_my_provider(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a provider client so supported calls record or replay automatically.
    
    Pass ``client=None`` during offline replay. Async routing is spec-driven 
    when the SDK carries both sync and async on one client.
    """
    return wrap_provider(client, cassette, MY_PROVIDER_SPEC, asynchronous=asynchronous)
```

The `wrap_provider` helper (from `_provider`) handles all routing, recording, replaying, and streaming based on the spec.

**Mistral example** ([wrap_mistral](../src/agent_cassette/integrations/mistral.py)):
```python
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

## Step 3: Create the patch context manager

Add exception classes and a patch function to `src/agent_cassette/automatic.py`:

```python
class MyProviderUnavailableError(ImportError):
    """Raised when automatic MyProvider support is requested without SDK installed."""

class MyProviderAlreadyPatchedError(RuntimeError):
    """Raised when the same module is patched by a nested context."""

@contextmanager
def patch_my_provider(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by the MyProvider constructor.
    
    Choose your patch mode based on SDK structure:
    
    - **Two constructors (sync/async pair)**: Use _patch_constructors
      (e.g., OpenAI has OpenAI and AsyncOpenAI classes)
      
    - **Single constructor (both sync and async on one client)**: Use _patch_single_constructor
      (e.g., Mistral has one Mistral class, with per-method async)
    """
    from agent_cassette.integrations.my_provider import wrap_my_provider
    
    # Two constructors pattern:
    with _patch_constructors(
        cassette,
        "my_provider_module",
        ("SyncClient", "AsyncClient"),  # or similar names
        wrap_my_provider,
        MyProviderUnavailableError,
        MyProviderAlreadyPatchedError,
    ):
        yield
    
    # OR single constructor pattern:
    # with _patch_single_constructor(
    #     cassette,
    #     "my_provider_module",
    #     "Client",
    #     wrap_my_provider,
    #     MyProviderUnavailableError,
    #     MyProviderAlreadyPatchedError,
    # ):
    #     yield
```

Add the function to `__all__` in `automatic.py`.

**Mistral example** ([patch_mistral](../src/agent_cassette/automatic.py)):
```python
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

## Step 4: Update the public-API contract (4 files)

The wrapper and patch functions must be exported and added to 4 contract snapshots:

### 4a. `src/agent_cassette/__init__.py`

Import the wrapper and patch function:

```python
from agent_cassette.integrations.my_provider import wrap_my_provider
from agent_cassette.automatic import patch_my_provider
```

Add both to `__all__`:

```python
__all__ = [
    # ... existing names ...
    "patch_my_provider",
    "wrap_my_provider",
]
```

**Mistral example**: [wrap_mistral and patch_mistral](../src/agent_cassette/__init__.py) are imported and exported.

### 4b. `docs/public-api.md`

Add a row to the "Providers/frameworks" table:

```markdown
| Providers/frameworks | `wrap_openai`, `wrap_anthropic`, `wrap_my_provider`, `wrap_mcp`, `wrap_langchain`, `patch_openai`, `patch_anthropic`, `patch_my_provider`, … |
```

**Mistral example**: Added to the table under "Providers/frameworks" as `wrap_mistral` and `patch_mistral`.

### 4c. `tests/test_public_api.py`

Update the `EXPECTED_PUBLIC_API` set:

```python
EXPECTED_PUBLIC_API = {
    # ... existing names ...
    "patch_my_provider",
    "wrap_my_provider",
}
```

Also update the blocked optional dependencies if needed. For Mistral, `"mistralai"` was added to the blocked set:

```python
blocked = {"agents", "anthropic", "langchain_core", "mistralai", "openai"}
```

**Mistral example**: Both functions added to `EXPECTED_PUBLIC_API`, and `"mistralai"` added to the blocked imports.

### 4d. `tests/test_contract_snapshots.py`

Update the `EXPECTED_PUBLIC_SIGNATURES` dict with your function signatures:

```python
EXPECTED_PUBLIC_SIGNATURES: dict[str, str] = {
    # ... existing signatures ...
    "patch_my_provider": "(cassette: 'Any') -> 'Iterator[None]'",
    "wrap_my_provider": "(client: 'Client | None', cassette: 'Any', *, asynchronous: 'bool | None' = None) -> 'Client'",
}
```

**Mistral example**: Both signatures added to the snapshot.

## Step 5: Add the optional dependency extra

Edit `pyproject.toml` under `[project.optional-dependencies]`:

```toml
my_provider = ["my-provider-sdk>=x,<y"]
```

Add it to the `all` extra:

```toml
all = [
    "openai>=1,<3",
    "anthropic>=0.34,<1",
    "my-provider-sdk>=x,<y",
    # ... others ...
]
```

Run `uv lock` to update the lock file.

**Mistral example**: 
```toml
[project.optional-dependencies]
mistral = ["mistralai>=1,<2"]

all = [
    "openai>=1,<3",
    "anthropic>=0.34,<1",
    "openai-agents>=0.1,<1",
    "langchain-core>=0.3,<2",
    "mistralai>=1,<2",
]
```

## Step 6: Add CI matrix rows

In `.github/workflows/ci.yml`, under the `installed-wheel-extras` job, add matrix rows for your provider:

```yaml
- compatibility: my-provider-minimum
  python-version: "3.10"
  extra: my_provider
  dependency-spec: "my-provider-sdk==x.0.0"
- compatibility: my-provider-current
  python-version: "3.13"
  extra: my_provider
  dependency-spec: ""
```

These rows run the full test suite (including your integration test) with the minimum and current supported versions of your SDK.

**Mistral example**: Two rows added—`mistral-minimum` with Python 3.10 and mistralai 1.0.0, and `mistral-current` with Python 3.13 and the latest mistralai.

## Step 7: Write an offline integration test

Create `tests/test_<provider>_integration.py`. Build a **synthetic client** that mirrors the SDK's shape but uses no keys, no network, and has deterministic behavior:

```python
import asyncio
from agent_cassette import Cassette
from agent_cassette.integrations.my_provider import wrap_my_provider

# Synthetic resource class mimicking your SDK
class _Resource:
    def operation(self, **kwargs):
        # Return a response-like object with the right structure
        return Response(data=kwargs)
    
    async def operation_async(self, **kwargs):
        return Response(data=kwargs)

# Synthetic client mimicking your SDK
class _SyntheticClient:
    def __init__(self, api_key=None):
        self.resource = _Resource()

def test_record_and_replay_sync(tmp_path):
    path = tmp_path / "cassette.jsonl"
    
    # Record
    with Cassette.record(path) as cassette:
        client = wrap_my_provider(_SyntheticClient(), cassette)
        result = client.resource.operation(input="test")
    
    # Replay (no network, no SDK needed)
    with Cassette.replay(path) as cassette:
        client = wrap_my_provider(None, cassette)  # None = offline replay
        replayed = client.resource.operation(input="test")
    
    assert result.data == replayed.data

async def test_record_and_replay_async(tmp_path):
    path = tmp_path / "cassette.jsonl"
    
    # Record
    with Cassette.record(path) as cassette:
        client = wrap_my_provider(_SyntheticClient(), cassette)
        result = await client.resource.operation_async(input="test")
    
    # Replay
    with Cassette.replay(path) as cassette:
        client = wrap_my_provider(None, cassette)
        replayed = await client.resource.operation_async(input="test")
    
    assert result.data == replayed.data

def test_patch_context_manager(tmp_path):
    """Verify patch_my_provider wraps constructors."""
    path = tmp_path / "cassette.jsonl"
    
    # This test imports the SDK module and verifies the patch works,
    # but uses a synthetic client to avoid network/keys.
    # See test_mistral_integration.py for the full pattern.
```

**Mistral example**: [tests/test_mistral_integration.py](../tests/test_mistral_integration.py) 
- Synthetic `_Chat` and `_Mistral` classes with deterministic behavior
- Tests for sync/async operations and streaming (with context manager support)
- Both record and replay paths exercised offline

## Step 8: Update integration and compatibility documentation

### 8a. `docs/integrations.md`

Add a section for your provider describing:
- Supported versions
- Sync and async support
- Streaming support
- How to use the wrapper and patch

### 8b. `docs/compatibility.md`

Add rows to the provider compatibility table with your SDK's versions.

### 8c. `CHANGELOG.md` (if following semver)

Add an entry documenting the new provider support.

**Mistral example**: 
- [docs/integrations.md](../docs/integrations.md) documents Mistral sync/async/streaming
- [docs/compatibility.md](../docs/compatibility.md) lists Mistral versions
- Changelog entry for the release

## Generalizations reference

The `ProviderSpec` supports the following generalizations (all optional, defaulted to off):

| Generalization | Field | Use case |
|---|---|---|
| **Streaming variants** (G2) | `stream_operations: frozenset[str]` | SDK has separate stream methods (`client.chat.stream`) instead of `stream=True` |
| **Async subtrees** (G3) | `async_subtrees: frozenset[str]` | SDK exposes async under a subtree (`client.aio.*`), not a parallel class |
| **Per-method async** (G5) | `async_operations: frozenset[str]` | SDK carries both sync and async on one client (`client.chat.complete` + `client.chat.complete_async`); auto-patch with `_patch_single_constructor` |
| **Flat operations** (G1) | Dotless entries in `operations` | SDK exposes operations directly on client (`client.chat`), not under a resource |
| **Factory clients** (G4) | Factory function pattern in `automatic.py` | SDK client comes from factory (`boto3.client(...)`) rather than class constructor |

See the [design spec](superpowers/specs/2026-07-18-multi-provider-foundation-design.md#non-goals) for details on future generalizations.

## Verification checklist

After implementing these 8 steps:

1. [ ] `ProviderSpec` accurately reflects your SDK's operation paths, prefixes, and async/streaming structure
2. [ ] Wrapper function handles both live client and offline replay (`client=None`)
3. [ ] Patch context manager uses the correct constructor mode (dual vs. single)
4. [ ] Public API: all 4 contract files updated and tests pass
5. [ ] Optional extra added to `pyproject.toml` and `uv lock` refreshed
6. [ ] CI matrix rows cover minimum and current SDK versions
7. [ ] Offline integration test exercises record → replay with synthetic client
8. [ ] Integration and compatibility docs updated

Run `pytest tests/test_public_api.py tests/test_contract_snapshots.py tests/test_<provider>_integration.py` to verify.
