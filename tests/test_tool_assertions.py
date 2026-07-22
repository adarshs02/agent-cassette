"""Phase D — public tool-replay assertion predicates and CLI checks."""

from __future__ import annotations

import enum
import json

import pytest

import agent_cassette.cli as cli
from agent_cassette import (
    Cassette,
    EventType,
    assert_trajectory,
    check_trajectory,
    tool_called,
    tool_not_called,
    wrap_tool,
)
from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError
from agent_cassette.storage import load_events


def _seed(path, entries):
    """entries: list of (event_type, name, input, metadata)."""
    with Cassette.record(path) as cassette:
        for event_type, name, input_value, metadata in entries:
            cassette.add(event_type, name, input=input_value, output="ok", metadata=metadata)


def _tool_call(name, input_value=None, metadata=None):
    return (EventType.TOOL_CALL, name, input_value, metadata or {})


def _tool_call_error(name, input_value=None):
    """A failed logical tool call, shaped exactly like Recorder._error_metadata."""
    return (EventType.ERROR, name, input_value, {"_agent_cassette": {"call_type": "tool_call"}})


# --------------------------------------------------------------------------- #
# tool_called — presence and counts
# --------------------------------------------------------------------------- #


def test_tool_called_presence_default_minimum_one(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    assert check_trajectory(path, tool_called("search")).passed
    assert not check_trajectory(path, tool_called("missing")).passed


@pytest.mark.parametrize("times,expected", [(1, True), (0, False), (2, False)])
def test_tool_called_exact_times(tmp_path, times, expected):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    assert check_trajectory(path, tool_called("search", times=times)).passed is expected


@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [
        (2, None, True),  # exactly at the floor
        (3, None, False),  # floor above observed
        (None, 2, True),  # at the ceiling
        (None, 1, False),  # over the ceiling
        (1, 3, True),  # inside inclusive bounds
        (0, 0, False),  # zero window but two present
    ],
)
def test_tool_called_inclusive_bounds(tmp_path, minimum, maximum, expected):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search"), _tool_call("search")])
    result = check_trajectory(path, tool_called("search", minimum=minimum, maximum=maximum))
    assert result.passed is expected


def test_tool_called_zero_window_passes_when_absent(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("other")])
    assert check_trajectory(path, tool_called("search", minimum=0, maximum=0)).passed


def test_tool_called_default_minimum_one_fails_on_zero(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("other")])
    result = check_trajectory(path, tool_called("search")).results[0]
    assert result.passed is False
    assert result.details["actual"] == 0


# --------------------------------------------------------------------------- #
# What counts as one logical tool call
# --------------------------------------------------------------------------- #


def test_counts_one_boundary_per_integration_shape(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(
        path,
        [
            _tool_call("search", metadata={"streaming": True}),  # wrap_tool (Phase A/B)
            _tool_call("mcp.lookup", metadata={"provider": "mcp"}),  # MCP
            _tool_call("langchain.tool.calc", metadata={"tool_bridge": True}),  # C2 bridge
            _tool_call("fetch", metadata={"provider": "openai-agents"}),  # C1 lifecycle
            # C1 bridged TOOL_RESULT for the same call — must NOT be counted:
            (EventType.TOOL_RESULT, "fetch", None, {"stage": "invoke"}),
        ],
    )
    for name in ("search", "mcp.lookup", "langchain.tool.calc", "fetch"):
        assert check_trajectory(path, tool_called(name, times=1)).passed, name
    # the OpenAI-Agents call/result pair for "fetch" is one call, not two:
    assert check_trajectory(path, tool_called("fetch", times=2)).passed is False


def test_failed_logical_call_is_counted_once(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call_error("search")])
    assert check_trajectory(path, tool_called("search", times=1)).passed


def test_real_raising_wrapped_tool_records_a_counted_boundary(tmp_path):
    path = tmp_path / "c.jsonl"

    def boom(query):
        raise RuntimeError("tool exploded")

    with Cassette.record(path) as cassette:
        with pytest.raises(RuntimeError):
            wrap_tool(boom, cassette, name="boom")("x")

    assert check_trajectory(path, tool_called("boom", times=1)).passed


@pytest.mark.parametrize(
    "entry",
    [
        (EventType.TOOL_RESULT, "search", None, {}),  # a bare result
        (EventType.ERROR, "search", None, {"_agent_cassette": {"call_type": "tool_result"}}),
        (EventType.ERROR, "search", None, {"_agent_cassette": {"call_type": "provider"}}),
        (EventType.ERROR, "search", None, {"_agent_cassette": "tool_call"}),  # lookalike, not dict
        (EventType.ERROR, "search", None, {}),  # uncaught, no marker
        (EventType.MODEL_CALL, "search", None, {}),  # provider call
        (EventType.CUSTOM, "search", None, {"observational": True}),  # LangChain trace
    ],
)
def test_non_tool_call_boundaries_are_never_counted(tmp_path, entry):
    path = tmp_path / "c.jsonl"
    _seed(path, [entry])
    assert check_trajectory(path, tool_called("search", minimum=0, maximum=0)).passed
    assert check_trajectory(path, tool_not_called("search")).passed


def test_exact_name_match_only(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search_web")])
    assert check_trajectory(path, tool_called("search")).passed is False  # no substring
    assert check_trajectory(path, tool_called("SEARCH_WEB")).passed is False  # no case-fold
    assert check_trajectory(path, tool_called("search_web")).passed


# --------------------------------------------------------------------------- #
# Input matching
# --------------------------------------------------------------------------- #


def test_omitted_input_matches_any_but_none_matches_only_json_null(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(
        path,
        [
            _tool_call("a", input_value={"args": ["x"], "kwargs": {}}),
            _tool_call("b", input_value=None),
        ],
    )
    # Ellipsis sentinel: any input counts.
    assert check_trajectory(path, tool_called("a")).passed
    assert check_trajectory(path, tool_called("b")).passed
    # with_input=None matches only the recorded JSON-null input.
    assert check_trajectory(path, tool_called("b", with_input=None)).passed
    assert check_trajectory(path, tool_called("a", with_input=None)).passed is False


def test_exact_input_match(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("s", input_value={"args": ["agents"], "kwargs": {}})])
    assert check_trajectory(
        path, tool_called("s", with_input={"args": ["agents"], "kwargs": {}})
    ).passed
    assert (
        check_trajectory(
            path, tool_called("s", with_input={"args": ["other"], "kwargs": {}})
        ).passed
        is False
    )


def test_subset_input_match(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("s", input_value={"args": ["agents"], "kwargs": {"n": 1}})])
    assert check_trajectory(
        path, tool_called("s", with_input={"args": ["agents"]}, match="subset")
    ).passed
    assert (
        check_trajectory(
            path, tool_called("s", with_input={"kwargs": {"n": 2}}, match="subset")
        ).passed
        is False
    )


def test_normalized_and_ignore_paths_input_match(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("s", input_value={"query": "hi", "trace_id": "abc"})])
    # ignore_paths drops the volatile field before comparison.
    assert check_trajectory(
        path,
        tool_called(
            "s",
            with_input={"query": "hi", "trace_id": "zzz"},
            match="normalized",
            ignore_paths=("trace_id",),
        ),
    ).passed


def test_fuzzy_threshold_input_match(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("s", input_value={"q": "the quick brown fox jumps"})])
    assert check_trajectory(
        path,
        tool_called(
            "s", with_input={"q": "the quick brown fox jump"}, match="fuzzy", fuzzy_threshold=0.8
        ),
    ).passed
    assert (
        check_trajectory(
            path,
            tool_called(
                "s", with_input={"q": "utterly different text"}, match="fuzzy", fuzzy_threshold=0.9
            ),
        ).passed
        is False
    )


# --------------------------------------------------------------------------- #
# tool_not_called
# --------------------------------------------------------------------------- #


def test_tool_not_called_by_name(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    assert check_trajectory(path, tool_not_called("send_email")).passed
    assert check_trajectory(path, tool_not_called("search")).passed is False


def test_tool_not_called_by_input_same_name(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("email", input_value={"args": ["safe@example.com"], "kwargs": {}})])
    # the tool ran, but never with this dangerous input:
    assert check_trajectory(
        path, tool_not_called("email", with_input={"args": ["attacker@evil.test"], "kwargs": {}})
    ).passed
    # the input it actually ran with is correctly detected:
    assert (
        check_trajectory(
            path, tool_not_called("email", with_input={"args": ["safe@example.com"], "kwargs": {}})
        ).passed
        is False
    )


# --------------------------------------------------------------------------- #
# Factory validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", None, 0])
def test_rejects_bad_name(bad):
    with pytest.raises(ValueError):
        tool_called(bad)
    with pytest.raises(ValueError):
        tool_not_called(bad)


@pytest.mark.parametrize("bad_count", [True, -1, 1.0, "1"])
def test_rejects_bad_count(bad_count):
    with pytest.raises(ValueError):
        tool_called("s", times=bad_count)
    with pytest.raises(ValueError):
        tool_called("s", minimum=bad_count)
    with pytest.raises(ValueError):
        tool_called("s", maximum=bad_count)


def test_rejects_conflicting_count_arguments():
    with pytest.raises(ValueError):
        tool_called("s", times=1, minimum=1)
    with pytest.raises(ValueError):
        tool_called("s", times=1, maximum=1)


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        tool_called("s", minimum=3, maximum=2)


@pytest.mark.parametrize("factory", [tool_called, tool_not_called])
def test_rejects_bad_match_and_threshold_and_ignore_paths(factory):
    with pytest.raises(ValueError):
        factory("s", match="regex")
    with pytest.raises(ValueError):
        factory("s", fuzzy_threshold=1.5)
    with pytest.raises(ValueError):
        factory("s", ignore_paths=["a"])  # list, not tuple
    with pytest.raises(ValueError):
        factory("s", ignore_paths=("",))  # empty element

    class _PathSub(str):
        pass

    with pytest.raises(ValueError):
        factory("s", ignore_paths=(_PathSub("a"),))  # str subclass, exact-type only


# --------------------------------------------------------------------------- #
# Strict input copier — rejection matrix, no rendering
# --------------------------------------------------------------------------- #


class _StrSub(str):
    pass


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


class _IntColor(enum.IntEnum):
    RED = 1


class _Hostile:
    def __repr__(self):  # pragma: no cover - must never run
        raise RuntimeError("repr must not be called")

    def __str__(self):  # pragma: no cover - must never run
        raise RuntimeError("str must not be called")


def _cyclic():
    d: dict = {}
    d["self"] = d
    return d


def _too_deep():
    root: dict = {}
    current = root
    for _ in range(MAX_JSON_DEPTH + 5):
        child: dict = {}
        current["x"] = child
        current = child
    return root


@pytest.mark.parametrize(
    "bad",
    [
        ("x",),  # tuple
        _IntColor.RED,  # IntEnum
        _StrSub("x"),  # str subclass
        _ListSub([1]),  # list subclass
        _DictSub({"a": 1}),  # dict subclass
        {1: "a"},  # non-str key
        {_StrSub("a"): 1},  # str-subclass key
        float("nan"),
        float("inf"),
        _cyclic(),
        _too_deep(),
        _Hostile(),
        {"nested": _Hostile()},
    ],
)
@pytest.mark.parametrize("factory", [tool_called, tool_not_called])
def test_with_input_strictly_rejected_without_rendering(factory, bad):
    with pytest.raises(StrictJSONError):
        factory("s", with_input=bad)


# --------------------------------------------------------------------------- #
# Detachment and secret safety
# --------------------------------------------------------------------------- #


def test_with_input_is_detached_at_creation(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("s", input_value={"args": ["x"], "kwargs": {}})])
    args: list[str] = ["x"]
    check = tool_called("s", with_input={"args": args, "kwargs": {}})
    args.append("MUTATED")  # must not affect the already-detached expectation
    assert check_trajectory(path, check).passed


def test_diagnostics_never_leak_input_values(tmp_path):
    secret = "hunter2-SECRET-TOKEN"
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search", input_value={"args": [secret], "kwargs": {}})])
    # A failing count so a message is produced; expected input also carries the secret.
    result = check_trajectory(
        path, tool_called("search", with_input={"args": [secret], "kwargs": {}}, times=5)
    ).results[0]
    assert result.passed is False
    assert secret not in result.message
    assert secret not in json.dumps(result.details)
    # details are JSON-native only
    json.dumps(result.details)
    assert result.details["input_filtered"] is True
    assert result.details["matching_indexes"] == [0]


# --------------------------------------------------------------------------- #
# Trajectory sources, composition, consumed_events
# --------------------------------------------------------------------------- #


def test_accepts_iterable_path_and_composes(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search"), _tool_call("summarize")])
    events = load_events(path)  # Iterable[Event]
    report = check_trajectory(
        events,
        tool_called("search", times=1),
        tool_not_called("send_email"),
    )
    assert report.passed
    assert {r.name for r in report.results} == {"tool_called", "tool_not_called"}
    assert "tool_called" in report.to_text() and "tool_not_called" in report.to_text()
    assert report.to_dict()["passed"] is True
    assert check_trajectory(path, tool_called("summarize")).passed  # also from a path


def test_assert_trajectory_raises_on_failure(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    with pytest.raises(AssertionError):
        assert_trajectory(path, tool_called("search", times=2))


def test_consumed_events_order_and_deep_detachment(tmp_path):
    path = tmp_path / "c.jsonl"

    def a(x):
        return {"a": x}

    def b(x):
        return {"b": x}

    with Cassette.record(path) as cassette:
        wrap_tool(a, cassette, name="a")("1")
        wrap_tool(b, cassette, name="b")("2")

    with Cassette.replay(path, strict=False) as replayer:
        # consume in the opposite order of the cassette
        wrap_tool(a, replayer, name="b")("2")
        wrap_tool(a, replayer, name="a")("1")
        assert replayer.remaining == 0
        consumed = replayer.consumed_events
        # ordered by cassette index, not consumption order
        assert [event.name for event in consumed] == ["a", "b"]
        # deep detachment: mutating a returned copy cannot alter Replayer state
        consumed[0].input["args"].append("MUTATED")
        assert replayer.events[0].input["args"] == ["1"]
        assert [e.input["args"] for e in replayer.consumed_events] == [["1"], ["2"]]


# --------------------------------------------------------------------------- #
# End-to-end record -> replay, zero live tool execution
# --------------------------------------------------------------------------- #


def test_end_to_end_zero_live_replay_with_consumed_events(tmp_path):
    path = tmp_path / "c.jsonl"

    def search(query):
        return {"query": query}

    def summarize(text):
        return {"summary": text}

    with Cassette.record(path) as cassette:
        wrap_tool(search, cassette)("agents")
        wrap_tool(summarize, cassette)("hi")

    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("ran")
        raise AssertionError("live tool ran during replay")

    with Cassette.replay(path) as replayer:
        wrap_tool(forbidden, replayer, name="search")("agents")
        wrap_tool(forbidden, replayer, name="summarize")("hi")
        assert replayer.remaining == 0
        report = check_trajectory(
            replayer.consumed_events,
            tool_called("search", with_input={"args": ["agents"]}, match="subset"),
            tool_called("summarize", times=1),
            tool_not_called("dangerous"),
        )
    assert report.passed
    assert calls == []  # zero live tool execution on replay


def test_public_exports_present():
    import agent_cassette

    assert "tool_called" in agent_cassette.__all__
    assert "tool_not_called" in agent_cassette.__all__
    assert agent_cassette.tool_called is tool_called
    assert agent_cassette.tool_not_called is tool_not_called


# --------------------------------------------------------------------------- #
# CLI: agent-cassette check --tool-called / --tool-not-called
# --------------------------------------------------------------------------- #


def test_cli_tool_called_pass_and_fail(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    assert cli.main(["check", str(path), "--tool-called", "search"]) == 0
    capsys.readouterr()
    assert cli.main(["check", str(path), "--tool-called", "missing"]) == 1


def test_cli_tool_not_called_pass_and_fail(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    assert cli.main(["check", str(path), "--tool-not-called", "send_email"]) == 0
    capsys.readouterr()
    assert cli.main(["check", str(path), "--tool-not-called", "search"]) == 1


def test_cli_repeatable_and_deterministic_order(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search"), _tool_call("summarize")])
    status = cli.main(
        [
            "check",
            str(path),
            "--tool-called",
            "search",
            "--tool-called",
            "summarize",
            "--tool-not-called",
            "delete",
        ]
    )
    assert status == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PASS")]
    # documented order: tool-called (in order), then tool-not-called
    assert lines == [
        "PASS tool_called: tool 'search': expected at least 1 call(s); observed 1",
        "PASS tool_called: tool 'summarize': expected at least 1 call(s); observed 1",
        "PASS tool_not_called: tool 'delete': expected 0 call(s); observed 0",
    ]


def test_cli_tool_checks_do_not_add_implicit_no_errors(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    # an error event is present; without --no-errors the tool check alone must still pass
    _seed(path, [_tool_call("search"), _tool_call_error("other")])
    assert cli.main(["check", str(path), "--tool-called", "search"]) == 0
    names = capsys.readouterr().out
    assert "no_errors" not in names
    # opting in re-enables the error gate and now fails
    assert cli.main(["check", str(path), "--tool-called", "search", "--no-errors"]) == 1


def test_cli_empty_tool_name_is_exit_two(tmp_path):
    path = tmp_path / "c.jsonl"
    _seed(path, [_tool_call("search")])
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["check", str(path), "--tool-called", ""])
    assert excinfo.value.code == 2


def test_cli_report_json_includes_tool_checks(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    report_path = tmp_path / "report.json"
    _seed(path, [_tool_call("search")])
    status = cli.main(
        ["check", str(path), "--tool-called", "search", "--report-json", str(report_path)]
    )
    assert status == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    names = json.dumps(report)
    assert "tool_called" in names
