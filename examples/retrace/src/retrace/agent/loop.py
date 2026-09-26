"""The investigation loop: the model drives strategy, tools produce facts, gates decide."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import anthropic

from retrace.agent.executor import ToolExecutor
from retrace.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS, initial_prompt
from retrace.agent.stages import fail
from retrace.agent.state import IncidentState

# OverloadedError (HTTP 529) only exists in newer SDKs; include it when present.
_OVERLOADED: type[BaseException] | None = getattr(anthropic, "OverloadedError", None)
RETRYABLE: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    *((_OVERLOADED,) if _OVERLOADED is not None else ()),
    ConnectionError,
    TimeoutError,
)
MAX_ATTEMPTS = 4
MAX_TOKENS = 16000
TRUNCATED_TOOL_USE = "response was truncated at max_tokens; retry with a shorter reply"


@dataclass
class LoopStats:
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _plain_blocks(response: Any) -> list[dict[str, Any]]:
    blocks = []
    for block in _get(response, "content"):
        kind = _get(block, "type")
        if kind == "thinking":
            # Thinking blocks must go back verbatim (with their signature) and in order.
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": _get(block, "thinking"),
                    "signature": _get(block, "signature"),
                }
            )
        elif kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": _get(block, "data")})
        elif kind == "text":
            blocks.append({"type": "text", "text": _get(block, "text")})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": _get(block, "id"),
                    "name": _get(block, "name"),
                    "input": json.loads(json.dumps(_get(block, "input"))),
                }
            )
    return blocks


def _create(
    client: Any, model: str, messages: list[dict[str, Any]], sleep: Callable[[float], None]
) -> Any:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
                ],
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except RETRYABLE:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            sleep(2.0**attempt)
    raise AssertionError("unreachable")


def run_agent(
    client: Any,
    executor: ToolExecutor,
    state: IncidentState,
    *,
    model: str,
    max_turns: int = 40,
    max_nudges: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> LoopStats:
    stats = LoopStats()
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt(state.report)}]
    nudges = 0
    last_call: tuple[str, str] | None = None
    for _ in range(max_turns):
        response = _create(client, model, messages, sleep)
        stats.turns += 1
        usage = _get(response, "usage")
        stats.input_tokens += int(_get(usage, "input_tokens") or 0)
        stats.output_tokens += int(_get(usage, "output_tokens") or 0)
        blocks = _plain_blocks(response) or [{"type": "text", "text": "(no content)"}]
        messages.append({"role": "assistant", "content": blocks})
        stop_reason = _get(response, "stop_reason")
        if stop_reason == "refusal":
            fail(state, "model refused")
            return stats
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        if tool_uses and stop_reason == "max_tokens":
            # A truncated reply may carry incomplete tool input: never dispatch it.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": use["id"],
                            "content": TRUNCATED_TOOL_USE,
                            "is_error": True,
                        }
                        for use in tool_uses
                    ],
                }
            )
            continue
        if not tool_uses:
            if nudges >= max_nudges:
                fail(state, "model stopped calling tools without finishing")
                return stats
            nudges += 1
            messages.append(
                {
                    "role": "user",
                    "content": "Continue by calling a tool. End only by calling finish.",
                }
            )
            continue
        results: list[dict[str, Any]] = []
        for use in tool_uses:
            if executor.finished:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use["id"],
                        "content": "run already finished",
                        "is_error": True,
                    }
                )
                continue
            stats.tool_calls += 1
            key = (use["name"], json.dumps(use["input"], sort_keys=True))
            if key == last_call:
                nudges += 1
                if nudges > max_nudges:
                    fail(state, "repeated identical tool calls")
                    return stats
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use["id"],
                        "content": "Identical to your previous call; use that result.",
                        "is_error": True,
                    }
                )
                continue
            last_call = key
            output = executor.dispatch(use["name"], use["input"])
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": use["id"],
                "content": json.dumps(output, sort_keys=True, default=str),
            }
            if "error" in output:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
        if executor.finished:
            return stats
    fail(state, f"turn budget of {max_turns} exhausted")
    return stats
