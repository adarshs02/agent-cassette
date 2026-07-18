# Multi-provider extension foundation — design

Status: approved design (2026-07-18). Target: `1.1.0` and following minors.

## Goal

Establish one documented, tested recipe plus the proxy/auto-patch generalizations so
each new LLM provider lands as a small, self-contained unit. Adding a provider should
mean: define a `ProviderSpec`, a `wrap_<p>`, and a `patch_<p>`; update the public-API
snapshots; add an extra, CI matrix rows, tests, and docs — nothing deeper.

## Non-goals

- No change to OpenAI or Anthropic behavior. Their specs, wrappers, and recorded
  cassette shapes stay byte-for-byte identical (they are contract-frozen at 1.0).
- No HTTP/wire-level capture. Recording stays at the semantic SDK-call level.
- No TypeScript/JavaScript providers or frameworks (e.g. Vercel AI SDK). Those belong
  to the separate language-agnostic effort.
- OpenAI-compatible endpoints (Groq, Together, Fireworks, DeepSeek, xAI, Perplexity,
  OpenRouter, Azure OpenAI, Ollama, …) are already supported through `wrap_openai`
  against the `openai` SDK. They need verification + docs, not new specs, and are
  tracked separately from this foundation.

## Current architecture (baseline)

- `ProviderSpec` (frozen dataclass) declares `provider`, `operations` (dotted names,
  e.g. `chat.completions.create`), `prefixes` (resource nodes), `raw_response_attrs`,
  `unsupported_operations`, `derive_methods`, `async_probe_path`, and streaming/raw
  error types.
- `wrap_provider(client, cassette, spec, asynchronous=None)` returns a
  `ProviderClientProxy`.
- `ProviderClientProxy.__getattr__` routes `prefixes` to `_ResourceProxy`,
  `derive_methods` to a re-wrap, and otherwise passes through. It does **not** wrap
  top-level operations.
- `_ResourceProxy.__getattr__` descends nested prefixes and, for a name in
  `operations`, calls `_wrap_create`, which records/replays and handles the
  `stream=True` keyword.
- Async detection (`_is_async_client`) assumes a parallel `Async*` client class whose
  attribute tree mirrors the sync client.
- Streaming is supported only via `create(..., stream=True)`.
- Auto-patching (`automatic._patch_constructors`) patches `module.<SyncClass>` and
  `module.<AsyncClass>`.

This fits OpenAI, Anthropic, and any OpenAI-compatible SDK.

## Gaps and generalizations

The five target SDKs (Mistral, Gemini, Cohere, Vertex AI, Bedrock) do not all follow
the `client.<resource>.<method>` + `stream=True` + `Async*Client` shape. Four additive
generalizations close the gaps. Each is default-off, so existing specs are unaffected.

### G1 — flat top-level operations
Cohere (`client.chat`) and Bedrock (`client.converse`) expose operations directly on
the client, not under a resource prefix. Change: factor the create-wrapping logic out
of `_ResourceProxy` into a shared helper, and have `ProviderClientProxy.__getattr__`
wrap a bare method name when it is in `operations`. No spec field needed — a flat
operation is simply a dotless entry in `operations`.

### G2 — sibling `*_stream` methods
Mistral (`chat.stream`) and Gemini (`generate_content_stream`) stream through a
separate method rather than a `stream=True` keyword. Add a spec field
`stream_operations: frozenset[str]` naming operations that are streaming variants; the
proxy captures/replays their chunks via the existing stream machinery. A phase may
instead mark the stream method in `unsupported_operations` (as Anthropic's
`messages.stream` is) and support only non-streaming calls in that provider's first
release. The umbrella provides the mechanism; each phase chooses.

### G3 — async subtrees
google-genai exposes async under `client.aio.models.generate_content` — an attribute
subtree on the same client, not an `Async*` class. Add a spec field
`async_subtrees: frozenset[str]` (e.g. `{"aio"}`). When traversal enters a listed
subtree, operations beneath it wrap as asynchronous regardless of the top-level client
type.

### G4 — factory-created clients
Bedrock's client comes from `boto3.client("bedrock-runtime", ...)`, not a class
constructor. Add a factory patch mode to `automatic.py`: wrap the factory function so
it returns a wrapped client when `service_name` matches the target service and passes
through otherwise. Bedrock also uses G1 for its flat operations.

## `ProviderSpec` additions

Backward-compatible, all defaulted:

- `stream_operations: frozenset[str] = frozenset()` (G2)
- `async_subtrees: frozenset[str] = frozenset()` (G3)

G1 needs no field. G4 lives in `automatic.py`, not the spec. The OpenAI and Anthropic
specs are not modified, so their behavior and recorded shapes are unchanged.

## Public API impact

Each provider adds `wrap_<p>` and `patch_<p>` (and any provider-specific error types)
to `agent_cassette.__all__`. This intentionally fails `tests/test_public_api.py` and
`tests/test_contract_snapshots.py`; the snapshots are updated as part of the provider's
PR. Additions are backward-compatible (SemVer minor). Existing public signatures do not
change, so their frozen snapshots stay intact.

## Per-provider recipe (`docs/adding-a-provider.md`, written with the first phase)

1. `integrations/<p>.py`: `<P>_SPEC = ProviderSpec(...)` + `wrap_<p>(client, cassette, *, asynchronous=None)`.
2. `automatic.py`: `patch_<p>` (constructor mode, or factory mode for G4) + `Unavailable`/`AlreadyPatched` errors.
3. `__init__.py`: export `wrap_<p>`/`patch_<p>`; update `__all__`, `docs/public-api.md`, and the contract snapshots.
4. `pyproject.toml`: optional extra `<p> = ["<sdk>>=x,<y"]`; add to `all`. Refresh `uv.lock`.
5. `.github/workflows/ci.yml`: add `<p>-minimum` and `<p>-current` matrix rows with an offline smoke.
6. `tests/test_<p>_integration.py`: offline record→replay against a synthetic client mirroring the SDK shape (no keys, no network).
7. `docs/integrations.md` + `docs/compatibility.md` + `CHANGELOG.md`.

## Testing strategy

- Generalization unit tests (provider-agnostic, synthetic clients): flat-op wrapping
  (G1), async-subtree wrapping (G3), stream-method capture (G2), factory-client patch
  (G4).
- Per-provider offline record→replay tests using fake clients, matching the inline CI
  smoke pattern; replay disables the network.
- Keep `test_public_api` and `test_contract_snapshots` in sync as public names grow.
- OpenAI/Anthropic regression tests must stay green unchanged.

## Phasing

Each phase is its own spec → plan → implementation → PR.

1. **Mistral** — closest to the current pattern; decides G2 (stream method vs
   unsupported); proves the end-to-end recipe and writes `docs/adding-a-provider.md`.
2. **Gemini** — implements G2 + G3 (`.aio` subtree, `generate_content_stream`).
3. **Cohere** — implements G1 (flat `chat`).
4. **Vertex AI** — model-instance shape (`GenerativeModel(name).generate_content`);
   validate feasibility first, possibly via google-genai's Vertex mode.
5. **Bedrock** — implements G4 + G1 (botocore factory client, `converse`).

`1.1.0` is cut at Mistral + Gemini. Cohere/Vertex/Bedrock follow in later minors and
carry their respective generalizations.

## Risks

- **Vertex AI** creates a `GenerativeModel` object per call, which may not fit the
  client-proxy model at all. Its phase spec must confirm a workable interception point
  (client proxy, an adapter, or the google-genai unified Vertex path) before
  committing to implementation.
- **Bedrock** uses a dynamically generated botocore client; the factory wrapper must
  not disturb other AWS services or client features (pagination, waiters).
- **Async + streaming interactions** under G2/G3 need explicit coverage.
- Exact SDK symbols (Gemini `.aio`, Mistral `chat.stream`, Cohere flat `chat`, Bedrock
  `converse`) are design inputs here and are confirmed against the SDK/current docs in
  each phase's spec before implementation.
