# Tool record/replay — Phase C1 design (OpenAI Agents FunctionTool bridge)

Status: **implemented-pending-review** (2026-07-22). Scope: **Phase C1 specification,
implemented.** Base: accepted Phase B head `b1f1437` on `release/1.1.0`. Implementation is
committed on `release/1.1.0`; the exact base/head SHAs are recorded in the coding-agent
handoff report (the spec is committed with the implementation, so its own commit hash
cannot be embedded without invalidating it).

Feature goal (all phases): replay a full agent loop without executing real tools. C1 bridges
ordinary OpenAI Agents SDK `FunctionTool` callbacks. `1.1.0` stays unpublished until A, B,
C1, C2, and D all pass. Agent Cassette is **not** replacing the SDK tracer: it replays the
provider and local tool boundaries so agent loops become deterministic tests.

## 1. Outcome

`patch_openai_agents(cassette)` bridges each ordinary registered `FunctionTool` so its
result records and replays at the public `FunctionTool.on_invoke_tool(context, arguments)`
callback. During replay the SDK agent loop still runs against replayed model responses,
hooks, guardrails, and handoffs, but the original Python tool callback is never invoked.

All `agents`/`openai` imports stay lazy — importing `agent_cassette` imports neither. SDK
classes resolve only inside `patch_openai_agents` after the guarded
`importlib.import_module("agents")`. Only the public callback is patched; no private SDK
runner/execution functions. No new public export, `EventType`, schema version, or
dependency; package version stays `1.1.0`.

## 2. Event sequence

1. `AgentCassetteRunHooks.on_tool_start` records/consumes the existing `TOOL_CALL` event
   (name, agent, raw arguments, tool-call ID).
2. The bridged `on_invoke_tool` records/consumes the existing `TOOL_RESULT` event with the
   same input shape `on_tool_end` used: `{"agent": <name>, "tool_call_id": <id>}`. Bridged
   result metadata is `{"provider": "openai-agents", "tool_bridge": True, "stage":
   "invoke"}`; error metadata keeps `_agent_cassette.call_type == "tool_result"`.
3. `on_tool_end` records nothing for a bridged tool; user hooks still receive it through
   `_CompositeRunHooks`.

Non-`FunctionTool` local tools keep lifecycle-only capture (`on_tool_start` → `TOOL_CALL`,
`on_tool_end` → `TOOL_RESULT`).

## 3. Bridge state

`_AgentsToolBridgeState` owns the cassette, the lazily resolved `FunctionTool` class, the
optional structured-output classes, strong references mapping each patched tool identity to
its original callback and installed wrapper, and active tool-call metadata keyed by tool
identity plus tool-call ID.

- `patch_agent(agent)` wraps each ordinary `FunctionTool` in `agent.tools` at most once and
  leaves other families untouched. Called before a runner invocation (starting agent from
  the first positional / `starting_agent`) and again from `on_agent_start` (which runs after
  user hooks, so tools they add are bridged before the first tool executes).
- `on_handoff` patches the destination agent before recording; no private handoff traversal.
- A shared `FunctionTool` is wrapped once. `is_bridged(tool)` verifies the tool still points
  at the installed wrapper; if user code replaced the callback, the lifecycle `TOOL_RESULT`
  is not suppressed.
- All callbacks and `Runner` descriptors are restored on context exit, including exceptional
  exits; a user-replaced callback is never overwritten. No module-global cassette or context
  variable.
- Agent-as-tool `FunctionTool` values are excluded (`_is_agent_tool is True`, with
  `_agent_instance` as a corroborating fallback), so their nested run replays in order.

## 4. Bridged callback

The installed callback is async and uses `functools.wraps(original)`. Per invocation it
builds and strictly serializes the result-match input, then calls
`cassette.acall(EventType.TOOL_RESULT, tool_name, input, live_thunk, metadata=...,
serializer=_encode_agents_tool_result)`. The live thunk sets an `executed` sentinel
immediately before awaiting the original callback. If live executed, the original live
object is returned (identity preserved); otherwise the recorded/injected value is decoded and
returned. `Replayer.acall` never creates or awaits the original coroutine. The Phase A safe
exception allowlist and `RecordedCallError` fallback are reused through the existing
Recorder/Replayer machinery; no recorded exception type is dynamically imported.

## 5. Tool-result envelope

C1-specific, versioned (not the provider/stream/LangChain markers):

```python
{
    "__agent_cassette_openai_agents_tool_result__": True,
    "version": 1,
    "kind": "json" | "structured" | "structured_list",
    "value": <strict JSON>,
}
```

Encoding: `json` via `serialize_recorded_value` (exact JSON-native only); `structured` /
`structured_list` accept only exact instances of the lazily captured SDK `ToolOutputText` /
`ToolOutputImage` / `ToolOutputFileContent` classes, serialized once through
`serialize_sdk_value(..., trusted_roots=("agents", "openai"))`; an empty list uses
`kind="json"`. Mixed, subclassed, tuple, or otherwise unsupported outputs are rejected — no
`str`/`repr`/arbitrary `model_dump` fallback. The completed envelope is strict-copied and
validated before persistence.

Decoding: strict-copy the whole envelope with `serialize_recorded_value`, require exact
marker/version/kind/key shape. `json` returns a detached JSON value. Structured kinds
reconstruct only the fixed captured class references, selected by the exact SDK `type`
discriminator (`text`/`image`/`file`); a class is never imported by name from cassette data,
and an SDK validation failure raises a type/path-only `StrictJSONError`. Exact JSON without
the marker is treated as a legacy pre-C1 `TOOL_RESULT` (strict-copied and returned); a
present-but-malformed marker fails closed. A Hybrid `Return` injection may supply exact JSON
or an exact structured SDK object directly (returned safely on the injected run; the
serializer stores the versioned envelope for later replay); `Raise`/`Delay` retain existing
Hybrid behavior.

## 6. Runner and streaming lifecycle

`Runner.run`, `run_sync`, and `run_streamed` keep hook injection and descriptor restoration.
A streamed run must be created and fully consumed while both the cassette session and
`patch_openai_agents` contexts are open. No GC/finalizer persistence; no patched callback is
left installed after context exit.

## 7. Tests

`tests/test_openai_agents_tool_replay.py` uses the installed SDK's real
`FunctionTool`/`function_tool` with a deterministic fake `Runner` (no network, no
credential), and skips cleanly without the extra. It covers the envelope codec
(json/structured/structured_list/legacy/malformed/unknown-discriminator/unsupported), the
two-turn zero-live loop, event order and single result event, structured round-trips,
allowlisted exceptions, non-`FunctionTool` lifecycle capture, late-added and handoff-target
tools, wrap-once, callback and `Runner` restoration after normal and exceptional exit,
user-replaced callbacks, concurrent unique-ID calls, `run_sync`/`run_streamed`, Hybrid
`Return`/`Raise` on `tool_result`, and agent-as-tool exclusion. The existing lifecycle and
serialization-trust suites remain green.
