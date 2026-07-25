"""Adversarial and regression tests for the cassette value serializers.

``agent_cassette.integrations._serialization`` exposes two internal entry
points:

- ``serialize_recorded_value`` is STRICT: it never inspects or invokes
  ``model_dump``, accepts only exact JSON builtin types (by ``type(x) is ...``
  identity, so tuples, ``IntEnum`` members, and every subclass are rejected),
  requires exact ``str`` keys, and validates the detached copy it returns.
- ``serialize_sdk_value`` trusts SDK types whose defining module falls under a
  caller-supplied ``trusted_roots`` package; it dumps those via ``model_dump``
  and rejects everything else -- naming only the type -- before ``model_dump``
  is ever read.

These tests prove neither serializer calls ``str()``/``repr()`` on an
unsupported value, neither coerces a non-``str`` key, the untrusted path never
touches ``model_dump``, ``model_dump`` on a trusted value is called exactly once
with no ``TypeError`` retry, and the provider/MCP/OpenAI-Agents call sites reject
hostile values before persistence.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import pytest

from agent_cassette import Cassette, EventType
from agent_cassette.integrations._provider import ProviderSpec, wrap_provider
from agent_cassette.integrations._serialization import (
    serialize_recorded_value,
    serialize_sdk_value,
)
from agent_cassette.integrations.mcp import wrap_mcp
from agent_cassette.integrations.openai_agents import patch_openai_agents
from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError, validate_json_value
from agent_cassette.storage import load_events


class _Hostile:
    """Rejected value whose __str__/__repr__ raise -- proves they are unused."""

    def __str__(self) -> str:
        raise AssertionError("str() must never be called on an unsupported recorded value")

    def __repr__(self) -> str:
        raise AssertionError("repr() must never be called on an unsupported recorded value")


class _SecretLeak:
    """Rejected value whose str()/repr() reveal a secret.

    Unlike ``_Hostile``, calling str()/repr() on this value does not raise --
    it is used to prove the raised error message never contains the value's
    data, independent of whether str()/repr() happen to be safe to call.
    """

    def __str__(self) -> str:
        return "super-secret-token-should-never-appear"

    def __repr__(self) -> str:
        return "_SecretLeak(token='super-secret-token-should-never-appear')"


class _HostileKey:
    """A hashable mapping key whose __str__ raises -- keys must not be str()'d."""

    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return self is other

    def __str__(self) -> str:
        raise AssertionError("str() must never be called on a rejected mapping key")


class _PropertyModelDump:
    """``model_dump`` is a property whose access raises.

    Proves neither the strict path nor the untrusted SDK path ever *reads* the
    ``model_dump`` attribute: touching the property (even without calling it)
    would raise ``AssertionError`` rather than the expected ``StrictJSONError``.
    """

    @property
    def model_dump(self) -> Any:
        raise AssertionError("model_dump must never be accessed on this value")


class _OpenAIEvil:
    """Object under module ``openai_evil`` -- must NOT be treated as ``openai``."""

    def model_dump(self, mode: str | None = None) -> dict[str, object]:
        raise AssertionError("model_dump must never run for an untrusted module")


_OpenAIEvil.__module__ = "openai_evil"


class _TrustedJsonMode:
    """Trusted SDK fake requiring ``model_dump(mode='json')``; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def model_dump(self, mode: str | None = None) -> dict[str, object]:
        self.calls += 1
        assert mode == "json"
        return {"ok": True}


_TrustedJsonMode.__module__ = "openai.types.responses.response"


class _TrustedNoMode:
    """Trusted SDK fake whose ``model_dump()`` takes no ``mode``; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def model_dump(self) -> dict[str, object]:
        self.calls += 1
        return {"legacy": True}


_TrustedNoMode.__module__ = "openai"


class _TrustedRaises:
    """Trusted SDK fake whose ``model_dump`` raises an internal ``TypeError``."""

    def __init__(self) -> None:
        self.calls = 0

    def model_dump(self, mode: str | None = None) -> dict[str, object]:
        self.calls += 1
        raise TypeError("boom from inside model_dump")


_TrustedRaises.__module__ = "openai"


class _HostileModelDump:
    """Nested value whose ``model_dump`` must never be invoked by strict path."""

    def model_dump(self, mode: str | None = None) -> dict[str, object]:
        raise AssertionError("nested model_dump must never be invoked by the strict path")


class _NestedHostileDump:
    """Trusted SDK fake whose dump smuggles in another model_dump-bearing value."""

    def model_dump(self, mode: str | None = None) -> dict[str, object]:
        return {"child": _HostileModelDump()}


_NestedHostileDump.__module__ = "openai"


class _Color(IntEnum):
    RED = 1


class _StrSub(str):
    pass


class _IntSub(int):
    pass


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


class _FloatSub(float):
    pass


# --- Strict serializer: hostile values and non-string keys ------------------


def test_hostile_object_is_rejected_without_invoking_str_or_repr() -> None:
    with pytest.raises(StrictJSONError):
        serialize_recorded_value(_Hostile())


def test_hostile_object_nested_in_containers_is_rejected_without_str_or_repr() -> None:
    with pytest.raises(StrictJSONError):
        serialize_recorded_value({"a": [1, 2, {"b": _Hostile()}]})


def test_non_string_dict_key_is_rejected_not_coerced() -> None:
    with pytest.raises(StrictJSONError, match="object keys must be strings"):
        serialize_recorded_value({1: "value"})


def test_hostile_dict_key_is_rejected_without_str_call() -> None:
    with pytest.raises(StrictJSONError, match="object keys must be strings"):
        serialize_recorded_value({_HostileKey(): "value"})


def test_unsupported_value_message_names_type_only() -> None:
    with pytest.raises(StrictJSONError) as excinfo:
        serialize_recorded_value(_SecretLeak())
    message = str(excinfo.value)
    assert "super-secret-token-should-never-appear" not in message
    assert f"{_SecretLeak.__module__}.{_SecretLeak.__qualname__}" in message


# --- Strict serializer: model_dump is never accessed at all -----------------


def test_strict_path_never_accesses_model_dump_property() -> None:
    with pytest.raises(StrictJSONError):
        serialize_recorded_value(_PropertyModelDump())


# --- Strict serializer: primitives, cycles, depth, non-finite ---------------


def test_primitives_and_none_pass_through_as_is() -> None:
    assert serialize_recorded_value(None) is None
    assert serialize_recorded_value(True) is True
    assert serialize_recorded_value(3) == 3
    assert serialize_recorded_value(3.5) == 3.5
    assert serialize_recorded_value("text") == "text"


def test_lists_and_dicts_recurse_and_are_detached() -> None:
    source = [1, "a", {"b": [2, 3]}]
    result = serialize_recorded_value(source)
    assert result == [1, "a", {"b": [2, 3]}]
    assert result is not source
    assert result[2] is not source[2]


def test_cyclic_list_is_rejected_without_recursion_error() -> None:
    cyclic: list[Any] = [1]
    cyclic.append(cyclic)
    with pytest.raises(StrictJSONError, match="cyclic"):
        serialize_recorded_value(cyclic)


def test_cyclic_dict_is_rejected_without_recursion_error() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(StrictJSONError, match="cyclic"):
        serialize_recorded_value(cyclic)


def test_indirect_cycle_through_containers_is_rejected() -> None:
    inner: dict[str, Any] = {}
    outer: list[Any] = [inner]
    inner["back"] = outer
    with pytest.raises(StrictJSONError, match="cyclic"):
        serialize_recorded_value({"root": outer})


def test_excessive_depth_is_rejected_with_strict_error() -> None:
    value: Any = 1
    for _ in range(MAX_JSON_DEPTH + 2):
        value = [value]
    with pytest.raises(StrictJSONError, match="depth"):
        serialize_recorded_value(value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_is_rejected(bad: float) -> None:
    with pytest.raises(StrictJSONError, match="non-finite"):
        serialize_recorded_value(bad)


def test_non_finite_float_nested_in_container_is_rejected() -> None:
    with pytest.raises(StrictJSONError, match="non-finite"):
        serialize_recorded_value({"x": [1.0, float("inf")]})


# --- tuple / IntEnum / subclasses / str-subclass keys are rejected ----------

_SUBCLASS_AND_TUPLE_CASES: list[Any] = [
    (1, 2),
    _Color.RED,
    _StrSub("x"),
    _IntSub(3),
    _FloatSub(1.5),
    _ListSub([1]),
    _DictSub({"a": 1}),
]


@pytest.mark.parametrize("value", _SUBCLASS_AND_TUPLE_CASES)
def test_strict_rejects_tuple_intenum_and_subclasses(value: Any) -> None:
    with pytest.raises(StrictJSONError):
        serialize_recorded_value(value)


@pytest.mark.parametrize("value", _SUBCLASS_AND_TUPLE_CASES)
def test_sdk_path_rejects_tuple_intenum_and_subclasses(value: Any) -> None:
    with pytest.raises(StrictJSONError):
        serialize_sdk_value(value, trusted_roots=("openai",))


def test_str_subclass_key_is_rejected_in_both_paths() -> None:
    with pytest.raises(StrictJSONError, match="object keys must be strings"):
        serialize_recorded_value({_StrSub("k"): "v"})
    with pytest.raises(StrictJSONError, match="object keys must be strings"):
        serialize_sdk_value({_StrSub("k"): "v"}, trusted_roots=("openai",))


# --- Trusted SDK serializer -------------------------------------------------


def test_untrusted_sdk_path_rejects_before_accessing_model_dump() -> None:
    with pytest.raises(StrictJSONError):
        serialize_sdk_value(_PropertyModelDump(), trusted_roots=("openai",))


def test_openai_evil_module_is_rejected_by_prefix_confusion() -> None:
    # "openai_evil" neither equals "openai" nor starts with "openai." so it is
    # untrusted; its (callable) model_dump must never run.
    with pytest.raises(StrictJSONError):
        serialize_sdk_value(_OpenAIEvil(), trusted_roots=("openai",))


def test_trusted_openai_types_fake_serialized_via_model_dump_once() -> None:
    value = _TrustedJsonMode()
    result = serialize_sdk_value(value, trusted_roots=("openai",))
    assert result == {"ok": True}
    assert value.calls == 1
    validate_json_value(result)


def test_trusted_no_mode_fake_calls_model_dump_once() -> None:
    value = _TrustedNoMode()
    result = serialize_sdk_value(value, trusted_roots=("openai",))
    assert result == {"legacy": True}
    assert value.calls == 1


def test_trusted_model_dump_typeerror_propagates_without_retry() -> None:
    value = _TrustedRaises()
    with pytest.raises(TypeError, match="boom from inside model_dump"):
        serialize_sdk_value(value, trusted_roots=("openai",))
    assert value.calls == 1


def test_trusted_dump_returning_nested_model_dump_object_is_rejected() -> None:
    # The trusted dump succeeds, but its result carries an object the strict
    # serializer rejects without ever invoking the nested model_dump.
    with pytest.raises(StrictJSONError):
        serialize_sdk_value(_NestedHostileDump(), trusted_roots=("openai",))


def test_trusted_multi_root_prefix_and_exact_match() -> None:
    # Exact-root match ("agents") and nested-root match ("openai.types...").
    class _Exact:
        def model_dump(self, mode: str | None = None) -> dict[str, object]:
            return {"root": "agents"}

    _Exact.__module__ = "agents"
    assert serialize_sdk_value(_Exact(), trusted_roots=("agents", "openai")) == {"root": "agents"}
    assert serialize_sdk_value(_TrustedJsonMode(), trusted_roots=("agents", "openai")) == {
        "ok": True
    }


def test_sdk_path_recurses_plain_json_and_rejects_non_finite_and_cycles() -> None:
    assert serialize_sdk_value({"a": [1, "x", True, None]}, trusted_roots=("openai",)) == {
        "a": [1, "x", True, None]
    }
    with pytest.raises(StrictJSONError, match="non-finite"):
        serialize_sdk_value(float("inf"), trusted_roots=("openai",))
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(StrictJSONError, match="cyclic"):
        serialize_sdk_value(cyclic, trusted_roots=("openai",))


# --- Through one provider path ----------------------------------------------

_DEMO_SPEC = ProviderSpec(
    provider="demo",
    operations=frozenset({"chat.go"}),
    prefixes=frozenset({"chat"}),
)


def test_provider_path_rejects_hostile_response_without_str_or_repr(tmp_path) -> None:
    class _HostileChat:
        def go(self, **kwargs: object) -> _Hostile:
            return _Hostile()

    class _HostileClient:
        def __init__(self) -> None:
            self.chat = _HostileChat()

    path = tmp_path / "demo-hostile-response.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_HostileClient(), cassette, _DEMO_SPEC)
        with pytest.raises(StrictJSONError):
            client.chat.go(model="m")
    # An unrepresentable output writes no success event.
    assert [event for event in load_events(path) if event.type == EventType.MODEL_CALL] == []


def test_provider_path_rejects_hostile_request_before_live_call(tmp_path) -> None:
    calls = {"count": 0}

    class _EchoChat:
        def go(self, **kwargs: object) -> dict[str, bool]:
            calls["count"] += 1
            return {"ok": True}

    class _EchoClient:
        def __init__(self) -> None:
            self.chat = _EchoChat()

    path = tmp_path / "demo-hostile-request.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_EchoClient(), cassette, _DEMO_SPEC)
        with pytest.raises(StrictJSONError):
            client.chat.go(model="m", hostile=_Hostile())
    assert calls["count"] == 0


# --- Through MCP ------------------------------------------------------------


def test_mcp_path_rejects_hostile_tool_result_without_str_or_repr(tmp_path) -> None:
    class _HostileSession:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, name: str, arguments: object, **kwargs: object) -> _Hostile:
            self.calls += 1
            return _Hostile()

    path = tmp_path / "mcp-hostile-response.jsonl"
    session = _HostileSession()

    async def scenario() -> None:
        async with Cassette.record(path) as cassette:
            with pytest.raises(StrictJSONError):
                await wrap_mcp(session, cassette).call_tool("search", {"query": "x"})

    asyncio.run(scenario())
    assert session.calls == 1


def test_mcp_path_rejects_hostile_argument_before_live_call(tmp_path) -> None:
    calls = {"count": 0}

    class _Session:
        async def call_tool(self, name: str, arguments: object, **kwargs: object) -> dict:
            calls["count"] += 1
            return {"ok": True}

    path = tmp_path / "mcp-hostile-request.jsonl"

    async def scenario() -> None:
        async with Cassette.record(path) as cassette:
            with pytest.raises(StrictJSONError):
                await wrap_mcp(_Session(), cassette).call_tool("search", {"query": _Hostile()})

    asyncio.run(scenario())
    assert calls["count"] == 0


# --- Through OpenAI Agents lifecycle hooks ----------------------------------


@dataclass
class _Named:
    name: str


def test_openai_agents_path_rejects_hostile_output_without_str_or_repr(
    tmp_path, monkeypatch
) -> None:
    class _HostileRunner:
        @staticmethod
        async def run(agent: object, prompt: str, *, hooks: Any = None) -> str:
            await hooks.on_agent_end(None, agent, _Hostile())
            return "done"

    agents = types.ModuleType("agents")
    agents.Runner = _HostileRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", agents)
    path = tmp_path / "agents-hostile.jsonl"

    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        with pytest.raises(StrictJSONError):
            asyncio.run(agents.Runner.run(_Named("researcher"), "find facts"))


# --- Combined-root depth: the dump boundary must not reset total depth --------


def _nested_list(depth: int, leaf: Any) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = [value]
    return value


class _DeepDumpOpenAI:
    """Trusted SDK fake whose dump is itself deeply nested."""

    def model_dump(self, mode: str | None = None) -> Any:
        return _nested_list(40, {"deep": True})


_DeepDumpOpenAI.__module__ = "openai.types.deep"


class _DeepDumpDemo:
    def model_dump(self, mode: str | None = None) -> Any:
        return _nested_list(40, {"deep": True})


_DeepDumpDemo.__module__ = "demo"


_DEEP_DEMO_SPEC = ProviderSpec(
    provider="demo",
    operations=frozenset({"chat.go"}),
    prefixes=frozenset({"chat"}),
    trusted_roots=("demo",),
)


def test_sdk_combined_depth_across_dump_boundary_is_rejected() -> None:
    # 40 outer list levels + a trusted object whose dump nests 40 more levels
    # exceeds MAX_JSON_DEPTH once the dump boundary no longer resets the count.
    outer = _nested_list(40, _DeepDumpOpenAI())
    with pytest.raises(StrictJSONError, match="depth"):
        serialize_sdk_value(outer, trusted_roots=("openai",))


def test_provider_request_split_depth_rejected_before_live_call(tmp_path) -> None:
    calls = {"count": 0}

    class _Chat:
        def go(self, **kwargs: object) -> dict[str, bool]:
            calls["count"] += 1
            return {"ok": True}

    class _Client:
        def __init__(self) -> None:
            self.chat = _Chat()

    path = tmp_path / "demo-split-depth-request.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_Client(), cassette, _DEEP_DEMO_SPEC)
        with pytest.raises(StrictJSONError, match="depth"):
            client.chat.go(payload=_nested_list(40, _DeepDumpDemo()))
    assert calls["count"] == 0


def test_provider_response_envelope_crossing_max_depth_persists_no_event(tmp_path) -> None:
    class _DeepChat:
        # data alone is at the limit; the response envelope wrapper pushes it over.
        def go(self, **kwargs: object) -> Any:
            return _nested_list(MAX_JSON_DEPTH, "leaf")

    class _Client:
        def __init__(self) -> None:
            self.chat = _DeepChat()

    path = tmp_path / "demo-deep-response.jsonl"
    with Cassette.record(path) as cassette:
        client = wrap_provider(_Client(), cassette, _DEMO_SPEC)
        with pytest.raises(StrictJSONError, match="depth"):
            client.chat.go(model="m")
    assert [event for event in load_events(path) if event.type == EventType.MODEL_CALL] == []


def test_mcp_request_envelope_crossing_max_depth_rejected_before_live_call(tmp_path) -> None:
    calls = {"count": 0}

    class _Session:
        async def call_tool(
            self, name: str, arguments: object, **kwargs: object
        ) -> dict[str, bool]:
            calls["count"] += 1
            return {"ok": True}

    path = tmp_path / "mcp-deep-request.jsonl"

    async def scenario() -> None:
        async with Cassette.record(path) as cassette:
            with pytest.raises(StrictJSONError, match="depth"):
                await wrap_mcp(_Session(), cassette).call_tool(
                    "search", {"deep": _nested_list(MAX_JSON_DEPTH + 5, "leaf")}
                )

    asyncio.run(scenario())
    assert calls["count"] == 0


def test_openai_agents_trusted_payload_dumps_exactly_once(tmp_path, monkeypatch) -> None:
    class _TrustedResponse:
        calls = 0

        def model_dump(self, mode: str | None = None) -> dict[str, bool]:
            type(self).calls += 1
            return {"ok": True}

    _TrustedResponse.__module__ = "openai"

    class _Runner:
        @staticmethod
        async def run(agent: object, prompt: str, *, hooks: Any = None) -> str:
            await hooks.on_llm_end(None, agent, _TrustedResponse())
            return "done"

    agents = types.ModuleType("agents")
    agents.Runner = _Runner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", agents)
    path = tmp_path / "agents-dump-once.jsonl"

    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        asyncio.run(agents.Runner.run(_Named("r"), "p"))
    assert _TrustedResponse.calls == 1


def test_openai_agents_deep_input_payload_rejected(tmp_path, monkeypatch) -> None:
    class _Runner:
        @staticmethod
        async def run(agent: object, prompt: str, *, hooks: Any = None) -> str:
            await hooks.on_llm_start(None, agent, None, _nested_list(MAX_JSON_DEPTH + 2, "x"))
            return "done"

    agents = types.ModuleType("agents")
    agents.Runner = _Runner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", agents)
    path = tmp_path / "agents-deep-input.jsonl"

    with Cassette.record(path) as cassette, patch_openai_agents(cassette):
        with pytest.raises(StrictJSONError, match="depth"):
            asyncio.run(agents.Runner.run(_Named("r"), "p"))
    assert load_events(path) == []
