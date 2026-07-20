# Tool record/replay — Phase A design

Status: **implemented-approved** (revision 2, 2026-07-19; approved-pending-review → revision-requested → ready-for-review → revision-requested (security: no duck-typed `model_dump`) → ready-for-review → approved-for-planning → implemented-pending-review → implemented-approved). Scope: **Phase A specification, implemented.** Accepted base: `1.0.1` at `8ff84d9` (tag `v1.0.1`). Accepted Phase A implementation candidate: `f0159cb`, integrated into the `1.1.0` release branch.

Feature goal (all phases): replay a full agent loop without executing real tools.
Phases: **A** core sync/async tool replay · **B** generator/async-generator streaming ·
**C1** OpenAI Agents bridge · **C2** LangChain bridge · **D** public assertions and
verification UX. `1.1.0` closes only after A, B, C1, C2, and D pass acceptance.
**Zero-live-execution sentinel tests belong in every phase** (safety proof cannot wait
for D).

**Build dependency:** Phase A implementation **cannot begin until `1.0.1` trust repair is
accepted.** Phase A serialization binds to the `1.0.1` strict-JSON contract (§6); it must
not ship on top of the pre-1.0.1 `_to_data` behavior.

## Revision map (six blockers → sections/tests)

| Blocker | Resolved in |
| --- | --- |
| 1. Safe serialization (no `_to_data`) | §6 — strict `validate_json_value`/`copy_json_value`; JSON-native input **and** output; **no `model_dump`/duck-typing** (no user code in the codec path); tests T-SER-1/2 |
| 2. Output type contract | §6 "Output type contract" (JSON-native only); tests T-OUT-1/2 |
| 3. Generator rejection | §5 "Generators and async generators"; tests T-GEN-1/2 |
| 4. Cancellation semantics | §4 cancellation matrix; tests T-CANCEL-1/2 |
| 5. Callable and name validation | §1 "Callable support matrix" + "Name validation" + "Typing"; tests T-CALL-* |
| 6. Concurrency claim | §5 "Concurrency"; tests T-CONC-1/2 |
| Non-blocking: post-context invocation | §2 + test T-LIFECYCLE-1 (documents current behavior) |

## 0. Phase A boundaries (fixed by the maintainer)

- Canonical API: `wrap_tool(fn, cassette, *, name=...)`. Sugar: bound `@cassette.tool(...)`.
- No active-cassette global or contextvar. No wrapper `policy` argument.
- Sync and async only. Generators/streaming → Phase B. Frameworks → C1/C2.
- Record return values and exceptions. Replay never invokes the wrapped callable.
- Preserve callable metadata/signature. Compose with existing parent/span handling.
- One replay-boundary event. No schema change. `Hybrid`/`InjectionRule` remain the
  injection authority.

## 1. Public signatures, callable support, and examples

### `wrap_tool`
```python
def wrap_tool(fn, cassette, *, name: str | None = None):
    """Wrap a tool callable so each invocation records or replays through `cassette`.

    `cassette` exposes `.call`/`.acall` (Recorder | Replayer | Hybrid). A coroutine
    function yields an async wrapper; a plain function yields a sync wrapper. `name`
    defaults to a usable `__name__`.
    """
```
Returns a `functools.wraps`-wrapped callable (`__name__`, `__qualname__`, `__doc__`,
`__wrapped__`, `__dict__`, `inspect.signature` preserved).

### Callable support matrix (Phase A)

| Callable kind | Supported | Async detection | Name source |
| --- | --- | --- | --- |
| Plain function (`def` / `async def`) | ✅ | `inspect.iscoroutinefunction(fn)` | `fn.__name__` |
| Bound method | ✅ | `inspect.iscoroutinefunction(fn)` | `fn.__name__` |
| `functools.partial` | ✅ | unwrap `.func` recursively, then `inspect.iscoroutinefunction` on the root | no `__name__` → `name=` **required** |
| `lambda` | ✅ | `inspect.iscoroutinefunction` | `__name__ == "<lambda>"` is rejected → `name=` **required** |
| Callable object (instance with `__call__`) | ❌ rejected | — | raise `ValueError` with "callable objects are not supported in Phase A; wrap the underlying function or pass `name=` with a supported callable" |
| Builtin / C callable with no reliable async introspection | ❌ rejected | — | same rejection message; revisit in a later phase |

Rationale: functions, bound methods, and partials cover real tool usage with reliable
async detection and are testable; callable objects have ambiguous async semantics and are
deferred rather than guessed.

### Name validation
- `fn` must be callable → else `TypeError`.
- Explicit `name` must be a non-empty `str` → else `ValueError`.
- A callable without a usable `__name__` (partial) or with `__name__ == "<lambda>"`
  → `name=` is required; otherwise `ValueError`.
- The resolved name is the tool's event `name` (matching + injection key).

### Typing
- `wrap_tool` is generic over `P = ParamSpec("P")`, `R = TypeVar("R")`:
  `def wrap_tool(fn: Callable[P, R], cassette: _Session, *, name: str | None = None) -> Callable[P, R]`.
  (For async `fn: Callable[P, Awaitable[R]]` the returned wrapper is `Callable[P, Awaitable[R]]`.)
- `cassette.tool` is `@overload`ed for both decorator forms:
  ```python
  @overload
  def tool(self, fn: Callable[P, R]) -> Callable[P, R]: ...
  @overload
  def tool(self, *, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
  ```
- mypy must pass on `src` and `tests` (project gate); no `Any`-typed public signature.

### Examples
```python
def search(query: str) -> dict:
    return live_http_get(query)

with Cassette.record("run.jsonl") as c:
    recorded = wrap_tool(search, c)
    recorded("agents")                    # runs live, records return

with Cassette.replay("run.jsonl") as c:
    replayed = wrap_tool(search, c)
    replayed("agents")                    # returns recorded value; search() NEVER runs
```
```python
with Cassette.record("run.jsonl") as c:
    @c.tool(name="fetch")
    async def fetch(url): return await live_fetch(url)
    await fetch("https://x")
```

## 2. Bound-decorator lifecycle and closed-session behavior

- `cassette.tool` binds to that session instance; the wrapper holds a strong reference and
  calls `session.call/acall` on every invocation.
- Phase A adds **no** session-lifecycle/"closed" state (no such flag exists today; adding
  it to three classes is out of scope for a thin wrapper). **Contract: invoke wrapped tools
  only while the session's `with` block is open.**
- Non-blocking (accepted): post-context invocation is inherited session behavior
  (Recorder still appends+writes; Replayer still consumes/raises). This is dangerous but
  accurately documented. **T-LIFECYCLE-1** records the current behavior so any future
  change is a conscious decision. Implementation must **not** silently introduce
  lifecycle-state changes.

## 3. Recorder / Replayer / Hybrid behavior matrix

`wrap_tool` is session-agnostic: it always calls
`cassette.call(EventType.TOOL_CALL, name, input, live_thunk, serializer=...)` (sync) or
`.acall(...)` (async), after validating input (§6). The session decides:

| Session | Tool executes? | Outcome |
| --- | --- | --- |
| `record` (Recorder) | **Yes** | Runs `fn`; records one `TOOL_CALL` event (input+return) or, on raise, an `ERROR` event (`call_type=tool_call`) and re-raises the original exception. |
| `replay` (Replayer) | **No** | Returns recorded output; recorded error → raises restored exception. `fn` never called. Mismatch/exhaustion → `ReplayMismatchError`. |
| `fork` (Hybrid) | prefix: **No**; after prefix / live-mismatch: **Yes** | Replays prefix, then executes+records live per `prefix`/`mismatch`. Injections short-circuit selected calls without executing `fn`. |

Four requested behaviors map onto sessions with **no wrapper policy**: `replay`=Replayer;
`live`=Recorder / Hybrid-past-prefix; `deny`=Replayer's inherent refusal (§8);
`inject`=Hybrid+`InjectionRule` (§9).

## 4. Exception and cancellation semantics

**Exceptions (`Exception` subclasses):**
- Record: `Recorder.call` catches `Exception`, records `ERROR` (`output={"type": <ClassName>,
  "message": str(e)}`, `call_type=tool_call`), re-raises the original.
- Replay: raises `_restore_error(event)` — allowlisted types round-trip
  (`ConnectionError`, `RateLimitError`, `RuntimeError`, `TimeoutError`, `ValueError`);
  every other type → `RecordedCallError(recorded_type, message)`. **Documented limitation:**
  custom tool exceptions do not replay as their own class; branch on an allowlisted type or
  catch `RecordedCallError`. No dynamic import of cassette-named types.

**Cancellation matrix (`asyncio.CancelledError`; also `KeyboardInterrupt`/`SystemExit` —
all `BaseException`):**

| Scenario | Phase A behavior |
| --- | --- |
| Cancellation caught inside the session (tool coroutine cancelled, handled before the `with` exits) | `Recorder.acall`'s `except Exception` does not catch it → **no tool event** is recorded for that call. |
| Cancellation escapes the session `with` block | `Recorder.__exit__` receives the exception and records a terminal `uncaught_exception` `ERROR` event with **no `_agent_cassette.call_type` metadata at all** (empty `metadata`); it is filtered from replay solely because its event `type` is `ERROR` (per `_is_replayable`). This is the existing generic behavior — **not** a tool event and **not** a replayable tool outcome. |
| Replay | No live coroutine is awaited; a tool call is never a "cancelled" outcome. |

**Cancellation is not a replayable tool outcome.** This is an **accepted limitation for
1.1** (cancellation is a live-execution concern). Tests: **T-CANCEL-1** (caught → no tool
event) and **T-CANCEL-2** (uncaught → generic `uncaught_exception`, non-replayable).

## 5. Sync/async detection, generators, and concurrency

**Kind detection:** `wrap_tool` inspects `fn` (after unwrapping `functools.partial`):
- async generator function (`inspect.isasyncgenfunction`) → **reject** (see below).
- generator function (`inspect.isgeneratorfunction`) → **reject**.
- coroutine function (`inspect.iscoroutinefunction`) → async wrapper (`acall`).
- otherwise → sync wrapper (`call`).

**Generators and async generators (blocker 3):** `wrap_tool` detects generator and
async-generator functions **at wrap time** and raises immediately — **before any
execution** — e.g. `ValueError("generator/async-generator tools are not supported in
Phase A; streaming tool support is Phase B")`. It must never execute the function and then
try to serialize an iterator. Tests: **T-GEN-1** (sync generator rejected at wrap time),
**T-GEN-2** (async generator rejected at wrap time), each asserting the body never ran.

**Concurrency (blocker 6):**
- **Phase A guarantee: async-task concurrency only.** `Replayer.acall` never awaits
  anything (it delegates straight to the synchronous `Replayer.call` match-and-return, so
  the event loop cannot interleave a partial consume during replay). `Recorder.acall`
  tasks **may interleave while awaiting live tools** — that suspension is exactly what lets
  concurrent record-mode tool calls overlap their I/O; what stays safe is persistence:
  `Recorder.add` is synchronous and guarded by `Recorder`'s `RLock`, so concurrent
  `Recorder.acall` tasks never interleave mid-write.
- **Threaded replay is explicitly excluded** — `Replayer` holds no lock. Threaded record is
  serialized by `Recorder`'s `RLock`, but threaded *replay* is unsafe and out of scope
  unless/until `Replayer` locking is added (future work, not Phase A).
- Duplicate / concurrent matching rides the session `strict` flag (not `wrap_tool`):
  `strict=True` = sequential position match; `strict=False` = first-unconsumed match
  (`_find_match`). Recommend `strict=False` for concurrent tools.
- **`strict=False` consumption is not enforced at context exit** — `Replayer.__exit__` only
  raises when `strict` is true. A non-strict concurrency test **must explicitly assert
  `replayer.remaining == 0`** after the loop; the `with` exit will not.
- **Identical concurrent calls with different recorded outputs are ambiguous** under
  `strict=False` (`_find_match` selects the first unconsumed event matching (type, name,
  input); if inputs are identical it cannot know which task should get which output).
  Documented as a known ambiguity; deterministic mapping of identical-input calls is out of
  scope for Phase A. Tests: **T-CONC-1** (async gather, `strict=False`, assert
  `remaining == 0`), **T-CONC-2** (documents identical-input ambiguity).

## 6. Serialization rules (strict; binds to 1.0.1 — blockers 1 & 2)

Phase A **does not use `_to_data`, does not use `copy_json_value`, and never calls
`model_dump()` or any other method on the value.** Tool input and output go through a
private **exact-type** copier `_copy_tool_json` (in `tools.py`) that reuses the `1.0.1`
limits (`MAX_JSON_DEPTH`, `StrictJSONError`) but accepts only **plain builtin** JSON types
by exact `type(value)` identity: `None`, `str`, `bool`, `int`, finite `float`, `list`, and
`dict` with **plain-`str`** keys. **Subclasses are rejected** — an `IntEnum`, a
`str`/`float`/`list`/`dict` subclass, or a `str`-subclass mapping key raises
`StrictJSONError`, so a recorded value and its replay always have the identical Python type
(no `IntEnum` → `int` drift). (`json_codec.copy_json_value` uses `isinstance` and would
accept subclasses, causing exactly that drift; Phase A must not use it.) Every other type
raises `StrictJSONError` naming the offending **type** (`module.qualname`) — never the
value's `repr` — and duplicate keys, non-finite numbers, cycles, and depth > 64 are
rejected.

**Security — no duck-typed conversion.** The wrapper must **not** probe for or call
`model_dump()` (or `dict()`, `__iter__`, etc.) on inputs or outputs. Duck-typing
`model_dump` would execute arbitrary user code inside the trusted serialization path, which
is exactly the trust hole `1.0.1` closes. Generic tool values are therefore required to be
**already JSON-native**; the wrapper only validates and detaches them, it never coerces or
converts.

**Input validation (before execution):**
- Build `input = {"args": [...], "kwargs": {...}}` from the call and run it through
  `_copy_tool_json` (exact-type validate + detached copy) **before executing `fn`**.
- Any non-plain-builtin input value — a Pydantic model, dataclass, `tuple`/`set`/`bytes`,
  **any subclass of a JSON type** (`IntEnum`, `str`/`float`/`list`/`dict` subclass),
  arbitrary object, non-`str` (or `str`-subclass) mapping key, non-finite number, or cyclic
  structure — raises `StrictJSONError` **before the tool runs**. The error names the offending type/path
  only; it does not embed the value's `repr`.
- Rich input support (e.g. passing Pydantic models as tool args) is **out of scope for
  Phase A**; it requires a future *explicit, trusted codec* (opt-in, not duck-typed), the
  same way rich outputs are deferred. Phase A neither converts nor guesses.

**Output type contract (blocker 2) — transparent replay is guaranteed ONLY for JSON-native
outputs:**

| Output value | Supported (same type + value on replay)? |
| --- | --- |
| `None`, `str`, `bool` | ✅ |
| finite `int` / `float` (no `NaN`/`Inf`) | ✅ |
| `list` of supported values | ✅ |
| `dict` with **string keys** and supported values | ✅ |
| `tuple`, `set`, `bytes` | ❌ rejected as output |
| Subclass of a JSON type (`IntEnum`, `str`/`float`/`list`/`dict` subclass) | ❌ rejected (exact-type check) |
| Pydantic model / dataclass / arbitrary rich object | ❌ rejected as output |

- Unsupported output **fails persistence** (raises before/at write) with a type/path-named
  error and **no representation leakage** (no `repr`/`str` of the value).
- No silent projection: a live call and its replay return the **same Python type and
  value** for supported outputs. (Contrast: the provider proxy intentionally projects
  responses; tools do not, to avoid agent-loop divergence.)
- Rich/framework-owned return reconstruction is **out of scope**; it belongs to C1/C2, or
  to a future explicitly-designed trusted codec — not a silent Phase A projection.

**Redaction:** applied by the session (`Recorder.add` → `redact`) before write, on the
already-validated JSON-native structure.

Tests: **T-SER-1** unsupported input (non-string dict key / arbitrary object) fails before
execution and does not leak `repr`; **T-SER-2** an object exposing a `model_dump`/`dict`
method passed as a tool arg is **rejected** (`StrictJSONError`) and its `model_dump` is
**never called** (assert via a sentinel that flags invocation) — proving no duck-typed code
execution; **T-OUT-1** every supported output type round-trips with identical type+value
(record vs replay); **T-OUT-2** `tuple`/rich-object output fails persistence without leaking
the value.

## 7. Event shape, name, ID, parent, and span semantics

- **One event per call.** Success → `type = TOOL_CALL`; failure → `type = ERROR` with
  `metadata._agent_cassette.call_type = "tool_call"`. Single boundary event (like the
  provider proxy's single `MODEL_CALL`), not the observational `TOOL_CALL`+`TOOL_RESULT`
  pair.
- `name` = resolved tool name (§1). `id` = recorder UUID; `timestamp` = recorder UTC ISO.
- `input`/`output` = strict-validated JSON (§6). `duration_ms` measured by `Recorder.call`.
- `parent_id`/`span_id` = recorder span contextvar stack, so wrapping inside
  `with recorder.span():` nests correctly, identical to manual `cassette.call`.
- **No new `EventType`; schema_version stays `1`; no migration.**

## 8. Replay mismatch and "deny-live" behavior

- Divergence (wrong `name`/`input`/type) or exhausted cassette → `ReplayMismatchError`
  (subject to `match`/`ignore_paths`/`strict`).
- "Deny live execution" is **inherent** to a `replay` session: `Replayer.call` ignores the
  live thunk, so a wrapped tool can never run under replay; an unmatched call fails closed
  with `ReplayMismatchError`. **No separate `deny` policy/error in Phase A.**

## 9. Interaction with `InjectionRule`

- Injection stays with `Cassette.fork(..., injections=[...])` (`Hybrid`). Because
  `wrap_tool` emits `EventType.TOOL_CALL` carrying `name`, an `InjectionRule` selects tool
  calls exactly as it selects model calls:
  ```python
  Cassette.fork("base.jsonl", "exp.jsonl", injections=[
      InjectionRule(Return({"ok": True}), event_type="tool_call", name="search", occurrence=1),
      InjectionRule(Raise(TimeoutError("slow")), event_type="tool_call", name="fetch"),
  ])
  ```
- `Hybrid.call/acall` fire the matched rule and record the injected outcome **without
  executing `fn`**. `wrap_tool` adds no injection logic; it only makes the event selectable.
  Injected `Return`/`Raise` values are subject to the same strict serialization (§6).

## 10. Test matrix and explicit Phase A exclusions

All tests offline (no network/keys). Named tests referenced by the revision map:
- Sync/async return equality; sync/async raised allowlisted exception; non-allowlisted →
  `RecordedCallError`.
- **Zero-live-execution sentinel (required, sync + async):** wrapped tool body sets a flag /
  raises `AssertionError` if entered; on **replay** assert recorded value returned AND
  sentinel untouched.
- Metadata/signature preservation (`__name__`, `__doc__`, `__wrapped__ is fn`,
  `inspect.signature` equal).
- Decorator forms `@c.tool`, `@c.tool(name=...)`; delegation equivalence to `wrap_tool`.
- Parent/span composition inside `recorder.span()`.
- **T-SER-1/2, T-OUT-1/2** (§6). **T-GEN-1/2** (§5). **T-CANCEL-1/2** (§4).
  **T-CALL-*** (callable matrix: function, bound method, partial+`name=`, lambda+`name=`,
  callable-object rejection, missing-name rejection). **T-CONC-1/2** (§5).
  **T-LIFECYCLE-1** (§2). Redaction of a secret arg. Hybrid replay-prefix-then-live +
  `InjectionRule` (`Return`/`Raise`) on a `tool_call` asserting `fn` did not execute.

Explicit exclusions (later phases / rejected): generators + async generators (B);
streaming outputs (B); OpenAI Agents bridge (C1); LangChain bridge (C2); public
assertion/verification UX (D); wrapper `policy` arg; active-cassette global/contextvar; new
`EventType`/schema change; custom `serializer` parameter; rich/tuple outputs; threaded
replay; callable-object/builtin tools.

## 11. Public API and schema compatibility analysis

- **New public callable `wrap_tool`** → add to `agent_cassette.__all__`, `docs/public-api.md`,
  `tests/test_public_api.py::EXPECTED_PUBLIC_API`, `tests/test_contract_snapshots.py::
  EXPECTED_PUBLIC_SIGNATURES` (exact signature string). Additive → SemVer **minor** (1.1.0).
- **New method `tool`** on `Recorder`, `Replayer`, `Hybrid` (additive instance method; not
  in the `__all__` callable snapshot). Add via a tiny shared mixin/helper to avoid
  triplication (implementation detail).
- **Schema unchanged**: reuse `TOOL_CALL`/`ERROR` + `call_type` metadata; `schema_version`
  stays `1`; no migration.
- **Serialization** binds to the `1.0.1` strict contract (`strict_json_dumps`); **Phase A
  build is gated on 1.0.1 acceptance.**
- **Lazy imports**: `wrap_tool` is core (stdlib + internal only); preserves
  `test_core_import_does_not_load_optional_dependencies`.

## 12. Alternatives considered and rejected

- **Active-cassette global / contextvar** for `@cassette.tool`: rejected — implicit global
  state, cross-test leakage, ambiguous under nested/parallel sessions. Bound
  `cassette.tool` + explicit `wrap_tool` keep the session explicit. (Maintainer-mandated.)
- **Wrapper-level `policy=` argument**: rejected — §3/§8/§9 prove all four behaviors are
  reachable via session type + `InjectionRule`; a wrapper policy would create a second,
  conflicting source of truth (e.g. `policy="live"` on a `Replayer`).
- **Reusing `_to_data` for tool serialization**: rejected — its `str()`/`repr` fallback and
  non-string key coercion violate the `1.0.1` trust contract; Phase A uses strict
  validation with fail-closed input/output (§6).
- **Duck-typed `model_dump()` (or `dict()`/`__iter__`) conversion of tool inputs**:
  rejected — Phase A cannot identify a "Pydantic value" without importing Pydantic, and
  probing/calling `model_dump` executes arbitrary user code inside the trusted codec path
  (the exact hole `1.0.1` closes). Generic tool input must be already JSON-native; rich
  input awaits a future explicit, opt-in trusted codec (never duck-typed).
- **Silent rich-object output projection**: rejected — record returns a rich object while
  replay returns a dict/list, diverging agent behavior. Phase A guarantees same-type replay
  for JSON-native outputs and rejects the rest (§6).
- **Two events per call** (`TOOL_CALL`+`TOOL_RESULT`): rejected for the replay boundary
  (one event is simpler to match/replay/inject; mirrors the provider proxy).
- **Signature-bound named input** (`inspect.signature(fn).bind`): rejected in favor of
  `{args, kwargs}` (robust to `*args`/`**kwargs`/partials).

## Next step

Return for maintainer review. On approval: writing-plans → Phase A plan (TDD, bite-sized),
then subagent-driven implementation — **after `1.0.1` acceptance**. No build before then.
