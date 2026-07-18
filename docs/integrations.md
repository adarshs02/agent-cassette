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
