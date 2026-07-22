# Integrations

Agent Cassette records and replays through the clients you already use. Install the
matching extra (see the [README](../README.md)) and the client is patched for each
`record`/`replay` run — no code changes.

## OpenAI

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(model="gpt-4.1-mini", input="Research agent testing")
```

Responses and Chat Completions are supported — sync, async, and streaming. On replay
that call returns an inert, attribute-compatible response from the cassette: no client,
no key, no network, no dynamic imports.

## Anthropic and OpenAI Agents (automatic)

If installed, `Anthropic`/`AsyncAnthropic` are patched like OpenAI (`messages.create`,
including `stream=True`). The OpenAI Agents SDK gets lifecycle hooks on `Runner.run`,
`run_sync`, and `run_streamed` automatically — capturing agents, LLM boundaries, tools,
and handoffs.

Python callers can also wrap explicitly with `wrap_openai`, `wrap_anthropic`, or patch
constructors with `patch_openai` / `patch_anthropic`.

### OpenAI Agents tool replay (`FunctionTool`)

`patch_openai_agents(cassette)` (used automatically by `agent-cassette run` and available
directly as a context manager) bridges ordinary SDK `FunctionTool` callbacks so their
results record and replay:

```python
from agent_cassette import Cassette
from agent_cassette.integrations.openai_agents import patch_openai_agents

with Cassette.record("agent.jsonl") as cassette, patch_openai_agents(cassette):
    result = await Runner.run(agent, "research agents")

with Cassette.replay("agent.jsonl") as cassette, patch_openai_agents(cassette):
    result = await Runner.run(agent, "research agents")  # tool callbacks never run
```

Agent Cassette is **not** a replacement for the SDK tracer: it replays the provider and
local tool boundaries so agent loops become deterministic tests. On replay the SDK agent
loop still runs against replayed model responses, hooks, guardrails, and handoffs — but the
original Python tool callback is never invoked. The successful event sequence is unchanged
from earlier cassettes: a lifecycle `TOOL_CALL` (from `on_tool_start`) immediately followed
by a `TOOL_RESULT` recorded at the `FunctionTool.on_invoke_tool` boundary.

Supported and unsupported in C1:

- Ordinary `FunctionTool` callbacks are bridged. JSON-native results, and the SDK
  structured outputs `ToolOutputText` / `ToolOutputImage` / `ToolOutputFileContent` (and a
  homogeneous list of them), round-trip exactly; the replayed value is the same SDK type
  downstream/user `on_tool_end` hooks saw. Unsupported outputs (tuples, subclasses,
  arbitrary objects, mixed structured/JSON lists, cycles, over-depth, NaN/Inf, non-string
  keys) fail before any success event is persisted, without rendering the value.
- Other local tool families (non-`FunctionTool`) keep lifecycle-only capture
  (`on_tool_start` → `TOOL_CALL`, `on_tool_end` → `TOOL_RESULT`).
- Agent-as-tool `FunctionTool` values are **not** bridged: their orchestration callback
  runs on replay so the nested loop's model/lifecycle events are consumed in order; the
  nested ordinary tools are bridged when their agent starts, so no real nested tool runs.
- A streamed run (`Runner.run_streamed`) must be created and fully consumed while both the
  cassette session and `patch_openai_agents` contexts are open — patched callbacks are
  restored on context exit (there is no GC/finalizer persistence).

Legacy pre-C1 lifecycle cassettes replay through an unmarked fallback. A legacy
`on_tool_end` value that had already been transformed by an output guardrail should be
re-recorded, because C1 treats the recorded value as the callback result and runs the
current guardrail again on replay.

## Mistral

Install `agent-cassette[mistral]` and the Mistral client is patched for each `record`/`replay`
run. Python callers can also wrap explicitly with `wrap_mistral` or patch the constructor with
`patch_mistral`.

Supported operations: sync `chat.complete`, async `chat.complete_async`, and streaming
`chat.stream` / `chat.stream_async`. On replay, these calls return inert, attribute-compatible
responses from the cassette: no client, no key, no network, no dynamic imports.

## Gemini

Install `agent-cassette[gemini]` and the Gemini client is patched for each `record`/`replay`
run. Python callers can also wrap explicitly with `wrap_gemini` or patch the constructor with
`patch_gemini`.

Supported operations: sync `models.generate_content`, async `aio.models.generate_content`, and streaming
`models.generate_content_stream` / `aio.models.generate_content_stream`. On replay, these calls return
inert, attribute-compatible responses from the cassette with `response.text` preserved: no
client, no key, no network, no dynamic imports.

## LangChain

Wrap a Runnable at its execution boundary:

```python
from agent_cassette import Cassette, wrap_langchain

chain = prompt | model | parser

with Cassette.record("chain.jsonl") as cassette:
    recorded = wrap_langchain(chain, cassette, name="research.chain")
    result = recorded.invoke({"topic": "agent testing"})

with Cassette.replay("chain.jsonl") as cassette:
    replayed = wrap_langchain(None, cassette, name="research.chain")   # None = never touches a live Runnable
    result = replayed.invoke({"topic": "agent testing"})
```

`invoke`, `ainvoke`, `stream`, `astream`, `batch`, and `abatch` are supported. For
framework-level trace spans, add `langchain_callback_handler(cassette)` to your
`config={"callbacks": [...]}`; trace events are marked observational and never disturb
replay.

## MCP

```python
from agent_cassette import Cassette, wrap_mcp

async with Cassette.record("mcp.jsonl") as cassette:
    session = wrap_mcp(live_session, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})

async with Cassette.replay("mcp.jsonl") as cassette:
    session = wrap_mcp(None, cassette)
    result = await session.call_tool("search", {"query": "agent testing"})
```

## Python tools

Wrap a plain Python function (sync or async) at its call boundary:

```python
from agent_cassette import Cassette, wrap_tool

def search(query: str) -> dict:
    return live_http_get(query)

with Cassette.record("tools.jsonl") as cassette:
    recorded_search = wrap_tool(search, cassette)
    result = recorded_search("agents")          # runs live, records the return value

with Cassette.replay("tools.jsonl") as cassette:
    replayed_search = wrap_tool(search, cassette)
    result = replayed_search("agents")          # returns the recorded value; search() never runs
```

`@cassette.tool` and `@cassette.tool(name="web.search")` are equivalent sugar for
`wrap_tool` bound to that session. The wrapper must be invoked while the session's `with`
block is open.

Inputs **and** outputs must be **plain builtin** JSON types only (`None`, `str`, `bool`,
`int`, finite `float`, `list`, and string-keyed `dict`) — subclasses are rejected. An
`IntEnum`, a `str`/`float`/`list`/`dict` subclass, or a `str`-subclass mapping key fails
validation, so a recorded value and its replay always have the identical Python type (no
`IntEnum` → `int` drift). The wrapper validates and copies these values with exact-type
checks; it never calls `model_dump()`, `str()`, `repr()`, or any other duck-typed
conversion. Pydantic models, dataclasses, `tuple`/`set`/`bytes`, and other rich objects are
rejected as both input and output; passing or returning one fails before the tool's live body runs (for
inputs) or before the event is persisted (for outputs), with no value representation
leaked into the error. Rich-object support is deferred to a future explicit, opt-in codec.

Sync and async functions, bound methods, and named `functools.partial`/`lambda` callables
are supported. Generator and async-generator tools are also supported: `wrap_tool` returns
a streaming proxy that records and replays each yielded item, the terminal return value
(sync only), a terminal error, or a deliberate early `close()`/`aclose()`. Every yielded
item is exact-type validated and detached before it reaches the caller, so a recorded item
and its replay have identical Python types. Replay never constructs or runs the underlying
generator.

Streaming limits (Phase B): a stream must be fully exhausted **or** explicitly
closed for the cassette to record it — a stream that is abandoned part-way records nothing
(no `__del__`/GC persistence). The proxy implements the iterator protocol plus
`close()`/`aclose()`; `send(None)`/`asend(None)` behave as `next()`/`anext()`, while a
non-`None` `send`/`asend` and any `throw`/`athrow` raise `NotImplementedError` (bidirectional
streaming is a later phase). An async stream cancelled mid-iteration records its partial
prefix plus a terminal cancellation and replays a fresh `CancelledError` after the prefix.

Matching is by exact call shape: `wrapped(1)` and `wrapped(value=1)` record different
inputs (`{"args": [1], "kwargs": {}}` vs `{"args": [], "kwargs": {"value": 1}}`) and do not
match each other on replay. Call the wrapped tool the same way every time.

For concurrent async tools with distinct inputs, replay with `strict=False` so calls match
by first-unconsumed input rather than strict arrival order; assert
`replayer.remaining == 0` yourself afterward, since a non-strict `Replayer.__exit__` does
not enforce full consumption. Identical concurrent inputs with different recorded outputs
are an accepted ambiguity — `wrap_tool` cannot know which concurrent task should receive
which recorded output.

## Manual API

Framework-independent, for any model or tool call:

```python
from agent_cassette import Cassette, EventType

with Cassette.record("run.jsonl") as cassette:
    result = cassette.call(
        EventType.TOOL_CALL, "search", {"query": "agent testing"},
        lambda: live_search("agent testing"),
    )

with Cassette.replay("run.jsonl") as cassette:
    result = cassette.call(EventType.TOOL_CALL, "search", {"query": "agent testing"})
```

Use `call`/`acall` for sync/async and `recorder.span()` to nest parent/child
relationships.
