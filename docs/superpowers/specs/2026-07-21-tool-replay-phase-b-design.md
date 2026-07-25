# Tool record/replay — Phase B design (generator streaming)

Status: **implemented-approved** (2026-07-21; replay-envelope validation correction
2026-07-22; accepted at `b1f1437`). Scope: **Phase B specification, implemented.** Base: accepted
Phase A checkpoint `c7b4523` on `release/1.1.0`. Implementation accepted at corrected head
`b1f1437` on `release/1.1.0`.

Feature goal (all phases): replay a full agent loop without executing real tools. Phase B
adds streaming tools. `1.1.0` stays unpublished until A, B, C1, C2, and D all pass.

## 1. Outcome

`wrap_tool` and the bound `.tool` decorator support ordinary generator functions and
async-generator functions in addition to the Phase A scalar functions. A recorded stream
replays the same yielded JSON values, terminal return/error behavior, and a deliberate
early close, **without constructing or executing the real generator during replay**.

No new public export, policy object, `EventType`, schema version, or dependency. Tool
streams reuse `TOOL_CALL` and `ERROR`; error events keep
`metadata._agent_cassette.call_type == "tool_call"` so they remain replayable.

## 2. Public behavior

1. `wrap_tool(generator_fn, cassette, name=...)` returns a synchronous iterable proxy;
   `wrap_tool(async_generator_fn, ...)` returns an asynchronous iterable proxy. Scalar
   sync/async functions, partials, lambdas, signatures, and the decorator are unchanged.
2. Calling a replay wrapper consumes the matching cassette event but never calls the
   wrapped function, constructs its generator, or touches credentials/network/filesystem.
3. Live recording constructs the generator only after the tool input passes the exact-type
   `_copy_tool_json` validation.
4. Each yielded item is exact-type validated and detached with `_copy_tool_json` before it
   is returned. Accepted types are exact `None`, `str`, `bool`, finite `int`/`float`,
   `list`, and string-keyed `dict`; subclasses, tuples, cycles, over-depth, and hostile
   containers are rejected.
5. Unsupported input fails before generator construction. An unsupported yielded item or
   sync-generator return value raises `StrictJSONError`, closes the live generator, and
   persists no event (already-yielded items may have reached the caller, but no unusable
   cassette is written).
6. Items stored in the cassette are detached at yield time; mutating an item after it was
   yielded does not mutate the recorded event.
7. Natural exhaustion records once. A sync generator's `StopIteration.value` is recorded
   and restored; async exhaustion has return value `None`.
8. Explicit `close()`/`aclose()` closes the underlying live generator (running its
   `finally`) and records the consumed prefix once with completion `closed`. Repeated close
   is idempotent. Replay yields the prefix then stops normally.
9. A stream abandoned without exhaustion or explicit close records nothing (no `__del__`/GC
   persistence). Callers must exhaust or close a stream for a reusable cassette.
10. Exceptions before the first item are phase `start`; after at least one item, phase
    `iteration`; from close, phase `close`. The prefix plus terminal error is recorded once,
    the live error re-raised, and replay yields the prefix then the restored terminal error.
11. Allowlist: `ValueError`, `RuntimeError`, `TimeoutError`, `ConnectionError`, and
    `RateLimitError` restore as themselves; every other type restores as
    `RecordedCallError`. No exception type named by cassette data is dynamically imported.
12. Async `CancelledError` during `__anext__` is the one Phase B extension to the Phase A
    cancellation limit: the live async generator is closed, the partial stream recorded as a
    terminal `cancellation` error, cancellation re-raised, and replay yields the prefix then
    a fresh `asyncio.CancelledError`. `KeyboardInterrupt`/`SystemExit` are never treated as
    replayable outcomes.
13. Iterator protocol plus `close`/`aclose` is the contract. `send(None)`/`asend(None)` act
    as `next()`/`anext()`; non-`None` `send`/`asend` and all `throw`/`athrow` raise
    `NotImplementedError`.

## 3. Internal payload

One versioned tool-specific envelope (not the provider or LangChain markers). Successful
exhaustion or explicit close:

```python
{
    "__agent_cassette_tool_stream__": True,
    "version": 1,
    "items": [...],
    "completion": "exhausted" | "closed",
    "return": <strict JSON value>,  # sync exhaustion return; otherwise None
}
```

Terminal failure:

```python
{
    "__agent_cassette_tool_stream__": True,
    "version": 1,
    "items": [...],
    "completion": "error",
    "phase": "start" | "iteration" | "close" | "cancellation",
    "error": {"type": "ExceptionName", "message": "safe message"},
}
```

Replay strictly validates the envelope. `_restore_tool_stream(recorded, *, asynchronous)`
first drives the **complete** envelope through the exact-type `_copy_tool_json` codec, so
nested tuples, subclasses, hostile containers, cycles, over-depth values, non-finite floats,
and non-`str` keys are rejected — without invoking any user `__str__`/`__repr__`/`model_dump`
— and every returned item and return value is a detached copy that never aliases
`Replayer.events`. It then checks exact marker/version scalar types, exact `str`
`completion`/`phase` **before** set membership (a list/dict there fails closed with
`StrictJSONError`, never a raw unhashable `TypeError`), exact key sets, exact string error
fields, and start errors with no items. Semantic shape is enforced, not only key shape:
`completion == "closed"` requires `return is None`; an async replay envelope requires
`return is None` even when exhausted; `phase == "cancellation"` requires error type exactly
`CancelledError`; and `CancelledError` is rejected for every non-cancellation phase.
Malformed envelopes fail closed. Error type names come from `type(error).__name__`; the
exception and payload are never `repr`'d.

## 4. Implementation structure

Internal to `src/agent_cassette/tools.py`: four private proxies
(`_RecordingToolStream`, `_AsyncRecordingToolStream`, `_ReplayToolStream`,
`_AsyncReplayToolStream`), private encode/validate/restore helpers for the envelope, and a
capability probe (`_dispatch_tool_stream`) that mirrors the provider stream pattern without
importing `Recorder`/`Replayer`/`Hybrid` (all import `tools.py`): `Hybrid` exposes
`prepare_stream`, `Replayer` exposes `consume`+`position`, and a plain `Recorder` records
live. Recording success calls `cassette.add(EventType.TOOL_CALL, ...)` once with
`metadata={"streaming": True}`; failure calls `cassette.add(EventType.ERROR, ...)` once with
`streaming` plus the tool-call `call_type`. Normal recorder span/parent behavior is
preserved when consumption happens inside a span.

`Hybrid.prepare_stream` gains an additive `defer_errors: bool = False`. Existing
provider/LangChain callers keep raise-at-preparation behavior; tool streams pass
`defer_errors=True`, so an injected `Raise` is serialized into a replay proxy that raises on
first iteration on both the injected run and later replay. Hybrid tool-stream `Return`
requires an exact plain `list` of valid items (encoded as natural exhaustion with return
`None`); a caller-supplied internal envelope is not accepted as `Return` data.

## 5. Tests

Focused coverage in `tests/test_tool_streaming.py`; the Phase A generator-rejection tests in
`tests/test_tools.py` are replaced with dispatch coverage. The provider, LangChain, MCP,
OpenAI Agents, Phase A tools, public API, and contract-snapshot suites are re-run to prove
the additive `Hybrid` change and unchanged public surface do not regress.
