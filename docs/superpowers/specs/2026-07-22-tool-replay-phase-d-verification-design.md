# Tool record/replay — Phase D design (offline tool-call assertions and verification UX)

Status: **implemented-pending-review** (2026-07-22). Scope: **Phase D specification,
implemented.** Base: accepted C2 head `2859726` on `release/1.1.0`, plus a correction commit
enforcing exact-type validation and command-line CLI occurrence order. Implementation is committed
on `release/1.1.0`; the exact base/head SHAs are in the coding-agent handoff report.

Feature goal (all phases): replay a full agent loop without executing real tools. Phase D closes
the feature with public, deterministic, offline tool-call assertions. `1.1.0` stays unpublished
until A–D pass, the queued connection-URI credential-redaction blocker is fixed, and a final
release gate passes. Agent Cassette is a deterministic test/replay layer, not a tracer: these
checks inspect a cassette or a Replayer's consumed transcript offline; they do not intercept
arbitrary OS/database/browser side effects.

## 1. Public API

Two additive core exports from `agent_cassette.assertions` (re-exported from root, added to
`__all__`, `docs/public-api.md`, and both snapshots), returning `AssertionResult` so they compose
with `check_trajectory`/`assert_trajectory`/`AssertionReport`/`CIReport`:

```python
tool_called(name, *, with_input=..., times=None, minimum=None, maximum=None,
            match="exact", ignore_paths=(), fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD) -> Check
tool_not_called(name, *, with_input=..., match="exact", ignore_paths=(),
                fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD) -> Check
```

`Ellipsis` is the stable sentinel: an omitted `with_input` matches any input; `None` matches an
actual JSON-null input. No process-global cassette or mutable matcher.

Plus a read-only `Replayer.consumed_events -> tuple[Event, ...]`: detached copies (round-tripped
through `Event.to_dict`/`Event.from_dict`, never references into `Replayer.events`) of the events
this session consumed, ordered by cassette index (not scheduler order), valid inside or after the
context, under strict or non-strict/concurrent consumption. It does not change `remaining`,
strict-exit, matching, or consumption order and adds no iteration.

## 2. What counts as one tool call

A matching boundary is `event.type is EventType.TOOL_CALL`, or `event.type is EventType.ERROR`
with exact metadata `metadata["_agent_cassette"]["call_type"] == "tool_call"` (a failed
Phase A/B/LangChain/MCP call, counted once). `TOOL_RESULT`, a `tool_result` ERROR, provider calls,
observational LangChain `CUSTOM` events, and malformed lookalike metadata are not counted; OpenAI
Agents ordinary tools count by their lifecycle `TOOL_CALL`, not the bridged `TOOL_RESULT`.
Candidates are filtered by exact `event.name == name` (no substring/regex/case-folding).

## 3. Input matching and counts

When `with_input is not Ellipsis`, it is detached and validated at factory creation through the
exact strict copier (`serialize_recorded_value`) — tuples, subclasses, cycles, depth overflow,
NaN/Inf, and non-`str`/key-subclass keys are rejected without rendering values. Expected and
recorded inputs are normalized with `normalize_input(..., ignore_paths)` and compared with
`inputs_match` using `exact`/`subset`/`normalized`/`fuzzy` and the threshold; `subset` means the
expected value is a recursive subset of the recorded tool envelope. Factory validation is
exact-type (`type(x) is ...`, never `isinstance`, so a hostile `str` subclass name never reaches
`name!r`): nonempty `str` name; `match` an exact `str` in the four modes; `fuzzy_threshold` via the
existing validator; `ignore_paths` an exact tuple of exact nonempty `str`; counts exact
non-negative `int` (never `bool`, `IntEnum`, or another `int` subclass); `times` mutually exclusive
with `minimum`/`maximum`; `minimum <= maximum`. The replayable-ERROR boundary likewise requires an
exact `dict` internal metadata object and an exact `str` `call_type == "tool_call"`, so mutated
in-memory lookalike events are not counted. For `tool_called`, no count
argument means minimum 1; `times` is exact; otherwise inclusive bounds (zero permitted).
`tool_not_called` passes only when the match count is exactly zero.

Result names are exactly `tool_called`/`tool_not_called`. Messages name only the tool, count
requirement, and observed count; `details` are JSON-native only (`tool_name`, `matching_indexes`,
`actual`, count bounds, `input_filtered`, `match`, `ignore_paths` as a list) — never raw
inputs/outputs, matcher objects, or a secret payload. A caller mutating its `with_input` after
creation, or a returned `consumed_events` payload, cannot alter the check or Replayer state.

## 4. CLI

`agent-cassette check` gains repeatable name-only `--tool-called NAME` / `--tool-not-called NAME`.
Both families feed one ordered argparse destination (a small `argparse.Action` appending
`(kind, name)`), so checks append in a deterministic order — every `--require` first, then each
`--tool-called` / `--tool-not-called` occurrence in exact command-line order (interleaving is
preserved), then cost/duration. Empty names fail as CLI input (exit `2`), not a traceback.
Including either option makes the command explicit, so `no_errors()` is not added implicitly unless
`--no-errors` is also supplied. Existing exit codes and `--report-json` are preserved (failures
exit `1`, invalid flags exit `2`). Structured-input and count assertions remain Python-only.

## 5. Tests

`tests/test_tool_assertions.py` and CLI tests cover: successful/failed logical calls (incl. ERROR
`call_type=tool_call`); exclusion of `TOOL_RESULT`, tool-result ERROR, provider/observational
`CUSTOM`, and no double-counting of the OpenAI Agents call/result pair; any-input sentinel vs JSON
null; exact/subset/normalized/fuzzy/threshold/ignore-path matching; exact/min/max/default counts,
duplicates, zero, validation conflicts and bool rejection; `tool_not_called` by name and by input
(same-name different-input); the strict input-copier matrix (tuple, IntEnum/primitive/container
subclasses, key subclass, NaN/Inf, cycle, depth, hostile list/dict) with no rendering;
with_input detached at creation and no secret in messages/JSON; path / `Iterable[Event]` /
`consumed_events` inputs; consumed-events order + deep detachment under non-strict out-of-order
consumption; strict + Phase A sync/async/stream zero-live sentinels composed with the checks and a
C1/C2-shaped fixture counted once; CLI repeated checks, order, JSON report, default-no_errors
interaction, exit `0`/`1`/`2`, and help; public API/signature snapshots and the lazy-import
invariant. All assertion/report/CLI and Phase A/B/C1/C2/Hybrid/provider/trust regressions remain
green. Isolated installed-wheel smoke: record a wrapped tool, replay with its body forbidden and
network/credentials disabled, assert `remaining == 0` and `tool_called(..., match="subset")` +
`tool_not_called("dangerous")` over `consumed_events`, and scan the failure text/JSON report for a
seeded secret.
