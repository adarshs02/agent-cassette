# Tool record/replay — Phase C2 design (LangChain registered-tool bridge)

Status: **implemented-pending-review** (2026-07-22). Scope: **Phase C2 specification,
implemented.** Base: accepted C1 head `f6191003` on `release/1.1.0`. Implementation is
committed on `release/1.1.0`; the exact base/head SHAs are recorded in the coding-agent
handoff report (the spec is committed with the implementation, so its own commit hash
cannot be embedded without invalidating it).

Feature goal (all phases): replay a full agent loop without executing real tools. C2 bridges
registered LangChain tools. `1.1.0` stays unpublished until A, B, C1, C2, and D all pass.
Agent Cassette is a deterministic test/replay layer, not a tracer — tracing callbacks can
coexist but are not the replay mechanism.

## 1. Public API

One lazy public function (importing `agent_cassette` still never imports `langchain_core`):

```python
wrap_langchain_tools(tools, cassette, *, name_prefix="langchain.tool")
```

Accepts a finite `list`/`tuple` of installed LangChain `BaseTool` objects and returns a new
`list` of bridged shallow clones in the same order, to register with a LangChain agent.
During record each tool's real `_run`/`_arun` executes once; during replay the agent loop,
args-schema validation, callback lifecycle, output formatting, and error handler still run,
but the real body is never called. Added to `agent_cassette.__all__`, the public API and
signature snapshots, and the docs.

This complements the Runnable boundary (`wrap_langchain`) and the observational
`langchain_callback_handler`; it does not replace either. The three LangChain features are
distinct: Runnable-boundary replay, registered leaf-tool replay (this), and observational
lifecycle callbacks.

## 2. Cloning and wrap-time validation

Implemented in `src/agent_cassette/integrations/langchain_tools.py` (lazy imports). `BaseTool`
and `ToolException` resolve only after the wrapper is called. Requirements:

- `type(tools)` must be `list` or `tuple`; generators, strings, and other iterables are
  rejected before iteration. Each entry must be an `isinstance(entry, BaseTool)` with an
  exact nonempty `str` `name`; errors name only the rejected type/index.
- `name_prefix` must be a nonempty `str`; the event name is `f"{name_prefix}.{tool.name}"`.
- `Tool`, `StructuredTool`, and custom `BaseTool` subclasses that customize `_run`/`_arun`
  are supported. A class that overrides any of the four public methods `invoke`/`ainvoke`/
  `run`/`arun` **in user code** (defining module outside `langchain_core`) is rejected;
  the SDK's own `ainvoke` override on `Tool`/`StructuredTool` is allowed.
- Cloning uses the fixed `BaseTool.model_copy(tool, deep=False)` (never a user override) and
  verifies a distinct, same-concrete-type result. Originals and class/global descriptors are
  never mutated.
- Instance-bound `_run`/`_arun` wrappers are installed with `MethodType` and
  `functools.wraps` on the original bound method, so `inspect.signature` still exposes
  `config`/`run_manager` for LangChain's injection.
- Duplicate identity is preserved (the same original yields the same clone). A bridged clone
  is privately marked; same-cassette rewrap is idempotent, different-cassette raises
  `ValueError`. `response_format` must be exactly `content` or `content_and_artifact`.

No context manager or restoration is needed — originals are untouched and the clones remain
bound to the supplied cassette session (execute only while it is open).

## 3. Boundary and request matching

The bridge intercepts the protected `_run`/`_arun`, not `invoke`/`run`, so SDK input
validation and callbacks run before it and output formatting / `handle_tool_error` after it.
The request is built and strict-copied before any live thunk:

```python
{"args": [<positional>], "kwargs": {<user kwargs>}, "config": <clean dict or None>}
```

SDK-injected `run_manager` is removed from matching; the `config` parameter is removed from
`kwargs` and its cleaned value stored separately. Config cleaning accepts only an exact plain
`dict`/`None` and drops only the volatile top-level keys `callbacks`, `run_id`, `run_name`.
The completed request passes through `serialize_recorded_value`, so only exact bounded JSON
is accepted (subclasses, tuples, cycles, depth, non-finite, hostile containers, unsupported
injected runtime objects fail before live execution). The broad Runnable `_encode`/`_decode`
codec is not reused, and no `model_dump`/`str`/`repr` is called.

One replayable `TOOL_CALL` is recorded with metadata
`{"integration": "langchain", "operation": "tool", "tool_bridge": True, "tool": tool.name}`.
`cassette.call` handles `_run`; `cassette.acall` handles a genuinely async `_arun`; a
sync-only tool invoked through LangChain's async fallback enters the sync `_run` bridge in a
worker (single call supported; thread-concurrent replay is not promised). The live sentinel
is set immediately before calling/awaiting the saved original bound method;
Replayer/Hybrid-replay consume the event before any live call or coroutine construction.

## 4. Result codec

A C2-specific envelope (not the Phase B, C1, provider/MCP, or Runnable markers):

```python
{"__agent_cassette_langchain_tool_result__": True, "version": 1,
 "kind": "json" | "content_and_artifact", "value": <strict JSON>}
```

Encoding: `content` accepts exact bounded JSON via `serialize_recorded_value` (`kind="json"`);
`content_and_artifact` requires an exact length-2 tuple, strict-copied into a two-item list
(`kind="content_and_artifact"`). The completed envelope is strict-copied/validated before
persistence; invalid output raises `StrictJSONError` after the live body but before any
success event. A live record path returns the original raw result/tuple unchanged.

Decoding: strict-copy the whole envelope; require exact container/key shape, `True` marker,
integer version `1` (not `True`), exact kind, and no extra keys. `json` is valid only for a
`content` tool (returns a detached JSON value); `content_and_artifact` is valid only for that
response format and restores an exact tuple so LangChain formats its `ToolMessage`. A
marker-bearing malformed dict fails closed; no recorded type name is imported. For Hybrid
`Return`, a raw exact JSON value (content) or raw exact two-tuple (artifact) is also accepted,
detached, and returned in the correct live shape, while the serializer persists the normal
envelope. There is no legacy unmarked C2 format; raw unmarked values arise only from
Hybrid/custom-session injection.

Direct Pydantic/dataclass, `Document`, `BaseMessage`/`ToolMessage`, and other
`ToolOutputMixin` objects returned by `_run` are out of C2 scope; normal `ToolMessage`
formatting still works from a JSON content value or the restored artifact tuple.

## 5. Errors, callbacks, and Hybrid

Ordinary exceptions use the existing Recorder/Replayer ERROR path with
`_agent_cassette.call_type == "tool_call"`; the safe allowlist replays directly, others become
`RecordedCallError`, and no error class is imported by name. The fixed installed
`ToolException` resolves at wrap time; on replay a `RecordedCallError` with
`recorded_type == "ToolException"` is translated to `ToolException(message) from None` so
`handle_tool_error` runs again. `CancelledError`/`KeyboardInterrupt`/`SystemExit` are not
replayable (the Phase A limit). User callbacks receive one normal lifecycle on record and
replay; the bridge synthesizes none. Hybrid replay-prefix-then-live and `Return`/`Raise`/
`Delay` target `event_type="tool_call"` and the bridged event name.

Orchestration tools whose body invokes another cassette-instrumented tool/agent are not
bridged (their inner event precedes the outer result; skipping the outer body would consume
events out of order). Users keep orchestration tools live and bridge their leaf tools; the
bridge fails closed on such an ordering mismatch rather than scanning past events.

## 6. Tests

`tests/test_langchain_tool_replay.py` uses the real `@tool`/`Tool`/`StructuredTool`/custom
`BaseTool` APIs with no network/credential, skipping without `langchain_core`. It covers
sync/async record+replay with zero-live sentinel and `remaining == 0`, `run`/`arun`, sync-only
async fallback, truly-async concurrent unique calls (`strict=False`), args-schema validation
running on replay with no event consumed, `content` detachment and `content_and_artifact`
`ToolMessage`/tuple round-trip, allowlisted / unknown / `ToolException`-with-handler errors,
Hybrid `Return`/`Raise`/`Delay` and injected artifact tuple, invalid request and result
matrices, malformed-envelope fail-closed, clone type/schema/field/signature preservation with
untouched originals, duplicate identity / idempotent same-cassette / different-cassette
rejection / bad sequence-tool-name-prefix / custom public-method-override rejection, and the
blocked optional-import test. Public export/signature snapshots are updated. The existing
LangChain Runnable/callback, provider, MCP, OpenAI Agents, Phase A/B, Hybrid, and trust suites
remain green; the broad Runnable serializer is unchanged.
