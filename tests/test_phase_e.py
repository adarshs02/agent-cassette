"""Phase E — agent-native closed operational loop (setup/status/named runs/ci)."""

from __future__ import annotations

import contextlib
import io
import json
import os as _os
import sys
import types
from pathlib import Path

import pytest

import agent_cassette.cli as cli
from agent_cassette import Cassette as _Cassette
from agent_cassette.events import EventType as _ET
from agent_cassette.machine import Envelope, MachineInputError, NextAction, validate_cassette_name
from agent_cassette.named_runs import (
    run_named_record,
    run_named_replay,
    run_named_rerecord,
)
from agent_cassette.project_loop import (
    MANIFEST_PATH,
    WORKFLOW_PATH,
    Overrides,
    run_agent_manifest,
    run_ci,
    run_setup,
    run_status,
)
from agent_cassette.replay import ReplayMismatchError, _diff_paths, _safe_name

# --------------------------------------------------------------------------- #
# Fake automatic provider (records one MODEL_CALL through the runner's patch)
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def model_dump(self, mode=None):
        return {"output_text": self.output_text}


_Response.__module__ = "openai"


class _Responses:
    live_calls = 0

    def create(self, **kwargs):
        type(self).live_calls += 1
        return _Response(f"answer:{kwargs['input']}")


class _OpenAI:
    def __init__(self) -> None:
        self.responses = _Responses()


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI  # type: ignore[attr-defined]
    module.AsyncOpenAI = type("AsyncOpenAI", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _Responses.live_calls = 0
    return module


def _agent_script(tmp_path: Path, secret: str = "hello") -> Path:
    script = tmp_path / "agent.py"
    script.write_text(
        f"from openai import OpenAI\nOpenAI().responses.create(model='test', input={secret!r})\n",
        encoding="utf-8",
    )
    return script


def _project(tmp_path: Path, **overrides) -> Path:
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    env = run_setup(root, mode="apply", overrides=Overrides(**overrides), include_ci=False)
    assert env.exit_code == 0, env.to_json()
    return root


def _cli(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(args)
        except SystemExit as exit_error:  # argparse usage errors
            code = int(exit_error.code or 0)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Envelope + name validation
# --------------------------------------------------------------------------- #


def test_envelope_canonical_shape_and_serialization():
    action = NextAction(id="a", argv=("agent-cassette", "status"), mutates=("x",))
    env = Envelope(
        command="status", status="current", exit_code=0, project="/p", next_actions=(action,)
    )
    data = env.to_dict()
    assert set(data) == {
        "schema_version",
        "command",
        "status",
        "ok",
        "exit_code",
        "project",
        "warnings",
        "changes",
        "next_actions",
        "data",
    }
    assert data["ok"] is True and data["schema_version"] == 1
    assert data["next_actions"][0] == {
        "id": "a",
        "argv": ["agent-cassette", "status"],
        "mutates": ["x"],
        "network": "forbidden",
        "approval_required": False,
        "requires_child_command": False,
    }
    text = env.to_json()
    assert text.endswith("\n") and json.loads(text)["status"] == "current"
    # canonical: sorted keys, 2-space indent
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"


def test_envelope_ok_only_for_exit_zero():
    assert Envelope("status", "x", 0, "/p").ok is True
    assert Envelope("status", "x", 1, "/p").ok is False
    assert Envelope("status", "x", 2, "/p").ok is False


@pytest.mark.parametrize("good", ["smoke", "a", "A1._-", "x" * 128])
def test_valid_cassette_names(good):
    assert validate_cassette_name(good) == good


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "a/b", "a\\b", "-lead", "_lead", "x" * 129, "a b", "a\nb", 5, None]
)
def test_invalid_cassette_names(bad):
    with pytest.raises(MachineInputError):
        validate_cassette_name(bad)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def test_setup_dry_run_writes_nothing(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_setup(root, mode="dry-run", overrides=Overrides(), include_ci=False)
    assert env.status == "would-change" and env.exit_code == 0
    assert not (root / ".agent-cassette.toml").exists()
    assert not (root / MANIFEST_PATH).exists()


def test_setup_apply_then_idempotent_and_check(tmp_path):
    root = _project(tmp_path)
    assert (root / MANIFEST_PATH).exists()
    again = run_setup(root, mode="apply", overrides=Overrides(), include_ci=False)
    assert again.status == "current" and again.changes == ()
    checked = run_setup(root, mode="check", overrides=Overrides(), include_ci=False)
    assert checked.status == "current" and checked.exit_code == 0


def test_setup_check_reports_missing_changes(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_setup(root, mode="check", overrides=Overrides(), include_ci=False)
    assert env.status == "changes-needed" and env.exit_code == 1


def test_setup_explicit_override_and_detection_fallback(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_setup(
        root,
        mode="apply",
        overrides=Overrides(providers=("openai",), match="subset"),
        include_ci=False,
    )
    assert env.exit_code == 0
    config = (root / ".agent-cassette.toml").read_text()
    assert '"openai"' in config and 'match = "subset"' in config


def test_setup_conflicting_override_is_conflict(tmp_path):
    root = _project(tmp_path)  # match defaults to exact
    env = run_setup(root, mode="apply", overrides=Overrides(match="fuzzy"), include_ci=False)
    assert env.status == "conflict" and env.exit_code == 2
    assert any("match" in message for message in env.data["conflicts"])
    # golden config unchanged
    assert 'match = "exact"' in (root / ".agent-cassette.toml").read_text()


def test_setup_modified_managed_file_is_conflict(tmp_path):
    root = _project(tmp_path)
    smoke = root / "tests/test_agent_cassette_smoke.py"
    smoke.write_text(smoke.read_text() + "\n# hand edit\n")
    env = run_setup(root, mode="apply", overrides=Overrides(), include_ci=False)
    assert env.status == "conflict" and env.exit_code == 2


def test_setup_creates_missing_owned_file(tmp_path):
    root = _project(tmp_path)
    (root / MANIFEST_PATH).unlink()
    env = run_setup(root, mode="apply", overrides=Overrides(), include_ci=False)
    assert env.exit_code == 0 and (root / MANIFEST_PATH).exists()


# --------------------------------------------------------------------------- #
# status + agent-manifest
# --------------------------------------------------------------------------- #


def test_status_before_setup(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_status(root)
    assert env.status == "changes-needed" and env.exit_code == 1
    assert env.data["readiness"]["setup_ready"] is False


def test_status_ready_and_capture_coverage(tmp_path):
    root = _project(tmp_path, providers=("openai", "mistral"))
    env = run_status(root)
    assert env.status == "current" and env.data["readiness"]["setup_ready"] is True
    coverage = env.data["capture_coverage"]
    assert coverage["automatic"] == ["openai"]
    assert "mistral" in coverage["requires_explicit_integration"]


def test_status_reports_modified_and_cassette_inventory(tmp_path):
    root = _project(tmp_path)
    # a valid cassette
    from agent_cassette import Cassette, EventType

    cassette = root / "tests/cassettes/smoke.jsonl"
    with Cassette.record(cassette) as recorder:
        recorder.add(EventType.MODEL_CALL, "openai", input={"q": 1}, output={"a": 2})
    # a corrupt cassette
    (root / "tests/cassettes/broken.jsonl").write_text("not-json\n")
    env = run_status(root)
    names = [item["name"] for item in env.data["cassettes"]]
    assert names == ["broken", "smoke"]  # sorted
    by_name = {item["name"]: item for item in env.data["cassettes"]}
    assert by_name["smoke"]["valid"] is True and by_name["smoke"]["replayable_events"] == 1
    assert by_name["broken"]["valid"] is False
    assert env.data["readiness"]["replay_ready"] is True


def test_status_corrupt_config_is_invalid(tmp_path):
    root = _project(tmp_path)
    (root / ".agent-cassette.toml").write_text("this is = not valid = toml\n")
    env = run_status(root)
    assert env.exit_code == 2 and env.status == "invalid"


def test_agent_manifest_is_useful_uninitialized_and_initialized(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_agent_manifest(root)
    assert env.exit_code == 0 and "commands" in env.data and "safety" in env.data
    assert "project_status" not in env.data
    _project(tmp_path)  # different subdir; make one with config
    initialized = run_agent_manifest((tmp_path / "proj").resolve())
    assert "project_status" in initialized.data


# --------------------------------------------------------------------------- #
# Named record / replay / rerecord (with the automatic-provider fake)
# --------------------------------------------------------------------------- #


def test_named_record_then_replay_zero_live(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = _agent_script(tmp_path)
    rec = run_named_record(root, "smoke", [str(script)], None)
    assert rec.envelope.exit_code == 0 and rec.envelope.status == "current"
    golden = root / "tests/cassettes/smoke.jsonl"
    assert golden.exists()
    assert rec.envelope.data["replayable_events"] >= 1
    assert _Responses.live_calls == 1

    _Responses.live_calls = 0
    rep = run_named_replay(root, "smoke", [str(script)], None, match=None, strict=None)
    assert rep.envelope.exit_code == 0 and rep.envelope.status == "current"
    assert rep.envelope.data["remaining_events"] == 0
    assert _Responses.live_calls == 0  # zero live during replay


def test_named_record_is_create_only(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = _agent_script(tmp_path)
    assert run_named_record(root, "smoke", [str(script)], None).envelope.exit_code == 0
    before = (root / "tests/cassettes/smoke.jsonl").read_bytes()
    second = run_named_record(root, "smoke", [str(script)], None)
    assert second.envelope.exit_code == 2 and second.envelope.status == "invalid"
    assert (root / "tests/cassettes/smoke.jsonl").read_bytes() == before  # untouched


def test_named_record_child_failure_leaves_no_golden(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)\n")
    result = run_named_record(root, "smoke", [str(script)], None)
    assert result.envelope.exit_code == 1 and result.envelope.status == "child-failed"
    assert not (root / "tests/cassettes/smoke.jsonl").exists()
    assert not list((root / "tests/cassettes").glob(".smoke.*.tmp"))


def test_named_record_empty_recording_is_invalid(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = tmp_path / "noop.py"
    script.write_text("pass\n")  # no provider call -> zero replayable events
    result = run_named_record(root, "smoke", [str(script)], None)
    assert result.envelope.exit_code == 2 and result.envelope.status == "invalid"
    assert not (root / "tests/cassettes/smoke.jsonl").exists()


def test_named_rerecord_preserves_golden_on_failure_and_changes_hash_on_success(
    tmp_path, fake_openai
):
    root = _project(tmp_path)
    script = _agent_script(tmp_path, secret="one")
    run_named_record(root, "smoke", [str(script)], None)
    golden = root / "tests/cassettes/smoke.jsonl"
    original = golden.read_bytes()

    boom = tmp_path / "boom.py"
    boom.write_text("import sys; sys.exit(1)\n")
    failed = run_named_rerecord(root, "smoke", [str(boom)], None)
    assert failed.envelope.exit_code == 1
    assert golden.read_bytes() == original  # untouched

    changed_script = _agent_script(tmp_path, secret="two")
    ok = run_named_rerecord(root, "smoke", [str(changed_script)], None)
    assert ok.envelope.exit_code == 0 and ok.envelope.status == "changed"
    assert golden.read_bytes() != original
    assert ok.envelope.data["previous_sha256"] != ok.envelope.data["cassette"]["sha256"]


def test_named_replay_structured_mismatch(tmp_path, fake_openai):
    root = _project(tmp_path)
    run_named_record(root, "smoke", [str(_agent_script(tmp_path, secret="hello"))], None)
    # replay a diverging child (different input)
    diverging = _agent_script(tmp_path, secret="different")
    result = run_named_replay(root, "smoke", [str(diverging)], None, match=None, strict=None)
    assert result.envelope.exit_code == 1 and result.envelope.status == "mismatch"
    failure = result.envelope.data["failure"]
    assert failure["kind"] == "input"
    assert "changed_paths" in failure and failure["changed_paths_truncated"] is False


def test_named_replay_missing_cassette_is_invalid(tmp_path, fake_openai):
    root = _project(tmp_path)
    result = run_named_replay(
        root, "smoke", [str(_agent_script(tmp_path))], None, match=None, strict=None
    )
    assert result.envelope.exit_code == 2 and result.envelope.status == "invalid"


# --------------------------------------------------------------------------- #
# Report file: separate channel, atomic, default/override, secret-free
# --------------------------------------------------------------------------- #


def test_report_default_destination_and_child_stdout_untouched(tmp_path, fake_openai, capsys):
    root = _project(tmp_path)
    script = tmp_path / "noisy.py"
    script.write_text(
        "from openai import OpenAI\n"
        "print('CHILD-STDOUT-NOISE {not json')\n"
        "OpenAI().responses.create(model='m', input='hi')\n",
        encoding="utf-8",
    )
    result = run_named_record(root, "smoke", [str(script)], None)
    report = root / ".agent-cassette/reports/record-smoke.json"
    assert result.report_path == report and report.exists()
    payload = json.loads(report.read_text())  # standalone valid JSON despite child noise
    assert payload["command"] == "record" and payload["ok"] is True


def test_report_override_must_be_within_project(tmp_path, fake_openai):
    root = _project(tmp_path)
    outside = tmp_path / "outside.json"
    result = run_named_record(root, "smoke", [str(_agent_script(tmp_path))], str(outside))
    assert result.envelope.exit_code == 2  # escapes project root


def test_reports_and_mismatch_never_contain_seeded_secret(tmp_path, fake_openai):
    root = _project(tmp_path)
    secret = "sk-PHASE-E-SECRET-9999"
    run_named_record(root, "smoke", [str(_agent_script(tmp_path, secret=secret))], None)
    diverging = _agent_script(tmp_path, secret="different-value")
    result = run_named_replay(root, "smoke", [str(diverging)], None, match=None, strict=None)
    assert result.report_path is not None
    blob = result.report_path.read_text()
    assert secret not in blob and "different-value" not in blob


# --------------------------------------------------------------------------- #
# CI scaffold
# --------------------------------------------------------------------------- #


def test_ci_requires_pytest(tmp_path):
    root = _project(tmp_path)  # no test framework detected/configured
    env = run_ci(root, mode="dry-run")
    assert env.exit_code == 1 and env.data["blockers"][0]["code"] == "pytest-not-configured"


def test_ci_apply_check_idempotent_and_replay_only(tmp_path):
    root = _project(tmp_path, test_frameworks=("pytest",))
    assert run_ci(root, mode="dry-run").status == "would-change"
    applied = run_ci(root, mode="apply")
    assert applied.exit_code == 0 and (root / WORKFLOW_PATH).exists()
    workflow = (root / WORKFLOW_PATH).read_text()
    assert "permissions:\n  contents: read" in workflow
    assert "pull_request" in workflow and "workflow_dispatch" in workflow
    assert "--cassette-mode=replay" in workflow
    assert 'OPENAI_API_KEY: ""' in workflow and "secrets." not in workflow
    assert run_ci(root, mode="check").exit_code == 0  # idempotent
    # modified workflow is never overwritten
    (root / WORKFLOW_PATH).write_text(workflow + "\n# edit\n")
    assert run_ci(root, mode="apply").exit_code == 2


def test_setup_github_ci_parity(tmp_path):
    # Fresh project with --github-ci owns the workflow (and manifest recording it) from
    # creation, so it matches the standalone `ci` command's generated plan.
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_setup(
        root, mode="apply", overrides=Overrides(test_frameworks=("pytest",)), include_ci=True
    )
    assert env.exit_code == 0 and (root / WORKFLOW_PATH).exists()
    standalone = run_ci(root, mode="check")  # already owned -> current
    assert standalone.exit_code == 0 and standalone.status == "current"


def test_setup_github_ci_without_pytest_is_blocked(tmp_path):
    root = (tmp_path / "p").resolve()
    root.mkdir()
    env = run_setup(root, mode="apply", overrides=Overrides(), include_ci=True)
    assert env.exit_code == 1 and env.data["blockers"][0]["code"] == "pytest-not-configured"


# --------------------------------------------------------------------------- #
# CLI integration: exit codes, JSON, no TTY, legacy unchanged
# --------------------------------------------------------------------------- #


def test_cli_setup_status_manifest_json(tmp_path):
    root = str((tmp_path / "p").resolve())
    Path(root).mkdir()
    code, out, _ = _cli(["setup", root, "--apply", "--json"])
    assert code == 0 and json.loads(out)["command"] == "setup"
    code, out, _ = _cli(["status", root, "--json"])
    assert code == 0 and json.loads(out)["command"] == "status"
    code, out, _ = _cli(["agent-manifest", root, "--json"])
    assert code == 0 and json.loads(out)["command"] == "agent-manifest"


def test_cli_named_run_prints_only_report_path(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = _agent_script(tmp_path)
    code, out, _ = _cli(["record", "--name", "smoke", "--project", str(root), "--", str(script)])
    assert code == 0
    assert "current" in out and "record-smoke.json" in out
    assert "{" not in out  # no JSON mixed into stdout


def test_cli_record_rejects_both_path_and_name(tmp_path, fake_openai):
    root = _project(tmp_path)
    # A real path alongside --name folds into the child command and fails validation:
    # exit 2, no golden ever created.
    code, out, _err = _cli(
        ["record", "--name", "smoke", "x.jsonl", "--", str(_agent_script(tmp_path))]
    )
    assert code == 2 and "invalid" in out
    assert not (root / "tests/cassettes/smoke.jsonl").exists()


def test_cli_legacy_positional_record_replay_unchanged(tmp_path, fake_openai):
    script = _agent_script(tmp_path)
    cassette = tmp_path / "legacy.jsonl"
    assert _cli(["record", str(cassette), "--", str(script)])[0] == 0
    assert cassette.exists()
    assert _cli(["replay", str(cassette), "--", str(script)])[0] == 0


# --------------------------------------------------------------------------- #
# Full consumer acceptance loop
# --------------------------------------------------------------------------- #


def test_consumer_acceptance_loop(tmp_path, fake_openai):
    """setup preview -> apply -> status ready -> record -> replay -> mismatch ->
    fix/retry -> rerecord -> ci preview/apply/check -> final setup/status check,
    with a noisy child, all via the CLI."""
    root = str((tmp_path / "consumer").resolve())
    Path(root).mkdir()

    # setup preview then apply (with pytest so CI is available later)
    assert _cli(["setup", root, "--test-framework", "pytest", "--dry-run", "--json"])[0] == 0
    assert not (Path(root) / MANIFEST_PATH).exists()
    assert _cli(["setup", root, "--test-framework", "pytest", "--apply", "--json"])[0] == 0
    code, out, _ = _cli(["status", root, "--json"])
    assert code == 0 and json.loads(out)["data"]["readiness"]["setup_ready"] is True

    # a noisy child that also makes one automatic provider call
    noisy = Path(root) / "agent.py"
    noisy.write_text(
        "import sys\n"
        "print('CHILD NOISE not-json {')\n"
        "print('warn', file=sys.stderr)\n"
        "from openai import OpenAI\n"
        "OpenAI().responses.create(model='m', input='hello')\n",
        encoding="utf-8",
    )

    # named live record -> golden created
    assert _cli(["record", "--name", "smoke", "--project", root, "--", str(noisy)])[0] == 0
    record_report = json.loads(
        (Path(root) / ".agent-cassette/reports/record-smoke.json").read_text()
    )
    assert record_report["status"] == "current" and record_report["data"]["replayable_events"] >= 1

    # offline replay of the same child -> pass, zero live
    _Responses.live_calls = 0
    assert _cli(["replay", "--name", "smoke", "--project", root, "--", str(noisy)])[0] == 0
    assert _Responses.live_calls == 0

    # a divergent child -> structured mismatch, exit 1
    diverged = Path(root) / "diverged.py"
    diverged.write_text(
        "from openai import OpenAI\nOpenAI().responses.create(model='m', input='changed')\n",
        encoding="utf-8",
    )
    code, _out, _ = _cli(["replay", "--name", "smoke", "--project", root, "--", str(diverged)])
    assert code == 1
    mismatch = json.loads((Path(root) / ".agent-cassette/reports/replay-smoke.json").read_text())
    assert mismatch["status"] == "mismatch" and mismatch["data"]["failure"]["kind"] == "input"

    # fix and retry (original child) -> pass again
    assert _cli(["replay", "--name", "smoke", "--project", root, "--", str(noisy)])[0] == 0

    # explicit rerecord updates the golden
    assert _cli(["rerecord", "--name", "smoke", "--project", root, "--", str(diverged)])[0] == 0

    # CI preview/apply/check
    assert _cli(["ci", root, "--github", "--dry-run", "--json"])[0] == 0
    assert _cli(["ci", root, "--github", "--apply", "--json"])[0] == 0
    assert (Path(root) / WORKFLOW_PATH).exists()
    assert _cli(["ci", root, "--github", "--check", "--json"])[0] == 0

    # final setup/status check remains stable
    assert _cli(["setup", root, "--check", "--json"])[0] == 0
    assert _cli(["status", root, "--json"])[0] == 0


# --------------------------------------------------------------------------- #
# Adversarial-review regressions
# --------------------------------------------------------------------------- #


def test_report_override_rejects_dotdot_escape(tmp_path, fake_openai):
    root = _project(tmp_path)
    result = run_named_record(root, "smoke", [str(_agent_script(tmp_path))], "../../pwned.json")
    assert result.envelope.exit_code == 2  # '..' cannot escape the project root
    assert not (tmp_path.parent / "pwned.json").exists()


def test_named_record_child_exception_is_exit_one_and_leaves_no_golden(tmp_path, fake_openai):
    root = _project(tmp_path)
    script = tmp_path / "raises.py"
    script.write_text("raise RuntimeError('boom')\n")
    result = run_named_record(root, "smoke", [str(script)], None)
    # exit_code must equal the process result (a re-raised child exception exits 1)
    assert result.envelope.exit_code == 1 and result.envelope.status == "child-failed"
    assert result.envelope.data["exception_type"] == "RuntimeError"
    assert result.child_exception is not None
    assert not (root / "tests/cassettes/smoke.jsonl").exists()
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text())
    assert report["exit_code"] == 1 and report["ok"] is False
    assert "boom" not in report["data"].get("error_code", "") and "boom" not in json.dumps(report)


def test_ci_requires_pytest_specifically_not_any_framework(tmp_path):
    root = _project(tmp_path, test_frameworks=("unittest",))
    env = run_ci(root, mode="dry-run")
    assert env.exit_code == 1 and env.data["blockers"][0]["code"] == "pytest-not-configured"


def test_cli_invalid_name_emits_envelope_json(tmp_path, fake_openai):
    root = _project(tmp_path)
    code, out, _ = _cli(
        ["record", "--name", "bad/name", "--project", str(root), "--", str(_agent_script(tmp_path))]
    )
    assert code == 2
    payload = json.loads(out)  # invalid input emits the canonical envelope (no child ran)
    assert payload["command"] == "record" and payload["ok"] is False
    assert payload["data"]["error_code"] == "invalid-name"


# --------------------------------------------------------------------------- #
# Secure-filesystem correction regressions (fail closed on symlink/TOCTOU)
# --------------------------------------------------------------------------- #


def _symlink_dir_outside(tmp_path, project_relative):
    """Replace <project>/<relative> with a symlink to an outside directory."""
    outside = (tmp_path / "outside").resolve()
    outside.mkdir(exist_ok=True)
    target = Path(project_relative)
    if target.exists():
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    _os.symlink(outside, target)
    return outside


def test_symlinked_cassette_dir_fails_closed(tmp_path, fake_openai):
    root = _project(tmp_path)
    outside = _symlink_dir_outside(tmp_path, root / "tests/cassettes")
    result = run_named_record(root, "smoke", [str(_agent_script(tmp_path))], None)
    assert result.envelope.exit_code == 2
    assert not any(outside.iterdir())  # nothing written outside
    assert _Responses.live_calls == 0  # child never ran (create-only pre-check failed)


def test_symlinked_report_dir_fails_closed(tmp_path, fake_openai):
    root = _project(tmp_path)
    (root / ".agent-cassette").mkdir(exist_ok=True)
    outside = (tmp_path / "reports_out").resolve()
    outside.mkdir()
    _os.symlink(outside, root / ".agent-cassette/reports")
    result = run_named_record(root, "smoke", [str(_agent_script(tmp_path))], None)
    assert result.envelope.exit_code == 2 and result.report_path is None
    assert not any(outside.iterdir())  # no report written outside
    assert _Responses.live_calls == 0  # child never ran (report preflight failed)


def test_symlink_and_hardlink_golden_rejected(tmp_path, fake_openai):
    root = _project(tmp_path)
    run_named_record(root, "good", [str(_agent_script(tmp_path))], None)
    golden = root / "tests/cassettes/good.jsonl"
    # symlinked golden
    link = root / "tests/cassettes/linked.jsonl"
    _os.symlink(golden, link)
    assert (
        run_named_replay(
            root, "linked", [str(_agent_script(tmp_path))], None, match=None, strict=None
        ).envelope.exit_code
        == 2
    )
    # hard-linked golden (st_nlink > 1)
    hard = root / "tests/cassettes/hard.jsonl"
    _os.link(golden, hard)
    assert (
        run_named_replay(
            root, "hard", [str(_agent_script(tmp_path))], None, match=None, strict=None
        ).envelope.exit_code
        == 2
    )


def test_mismatch_indexes_are_one_based(tmp_path):
    path = tmp_path / "c.jsonl"
    with _Cassette.record(path) as rec:
        rec.add(_ET.TOOL_CALL, "a", input={"x": 1}, output="1")
        rec.add(_ET.TOOL_CALL, "b", input={"x": 2}, output="2")

    # strict divergence at step 1 -> event_index 1
    with pytest.raises(ReplayMismatchError) as diverge:
        with _Cassette.replay(path) as rep:
            rep.call(_ET.TOOL_CALL, "a", {"x": 999})
    assert diverge.value.kind == "input" and diverge.value.event_index == 1

    # exhausted -> event_index len(events)+1 = 3
    with pytest.raises(ReplayMismatchError) as exhausted:
        with _Cassette.replay(path) as rep:
            rep.call(_ET.TOOL_CALL, "a", {"x": 1})
            rep.call(_ET.TOOL_CALL, "b", {"x": 2})
            rep.call(_ET.TOOL_CALL, "c", {})
    assert exhausted.value.kind == "exhausted" and exhausted.value.event_index == 3

    # unconsumed strict exit -> first unconsumed index +1 = 1, with expected summary
    with pytest.raises(ReplayMismatchError) as unconsumed:
        with _Cassette.replay(path):
            pass
    assert unconsumed.value.kind == "unconsumed" and unconsumed.value.event_index == 1
    assert unconsumed.value.expected == {"type": "tool_call", "name": "a"}

    # non-strict no-match -> event_index null
    with pytest.raises(ReplayMismatchError) as nomatch:
        with _Cassette.replay(path, strict=False) as rep:
            rep.call(_ET.TOOL_CALL, "missing", {})
    assert nomatch.value.kind == "no-match" and nomatch.value.event_index is None


def test_diff_paths_hostile_container_stays_structured():
    class Hostile:
        def __iter__(self):  # pragma: no cover - must never run
            raise AssertionError("iterated hostile container")

        def __repr__(self):  # pragma: no cover - must never run
            raise AssertionError("repr of hostile container")

    # an actual value that is not strict JSON -> value-free root diff, no exception
    paths, truncated = _diff_paths({"a": 1}, {"a": Hostile()})
    assert paths == (".",) and truncated is False


def test_safe_name_redacts_secrets():
    name = "postgres://user:sk-DSNPW@db/app?token=sk-QUERY Bearer sk-BEARER"
    safe = _safe_name(name)
    assert safe is not None
    for secret in ("sk-DSNPW", "sk-QUERY", "sk-BEARER"):
        assert secret not in safe


def test_mismatch_next_actions_have_no_placeholder(tmp_path, fake_openai):
    root = _project(tmp_path)
    run_named_record(root, "smoke", [str(_agent_script(tmp_path, secret="one"))], None)
    result = run_named_replay(
        root, "smoke", [str(_agent_script(tmp_path, secret="two"))], None, match=None, strict=None
    )
    assert result.report_path is not None
    payload = json.loads(result.report_path.read_text())
    for action in payload["next_actions"]:
        assert "<cassette-path>" not in action["argv"]
    inspect = [a for a in payload["next_actions"] if a["id"] == "inspect-cassette"][0]
    assert inspect["argv"][2].endswith("smoke.jsonl")


# --------------------------------------------------------------------------- #
# Second-review regressions
# --------------------------------------------------------------------------- #


def test_named_replay_counts_only_replayable_events(tmp_path, fake_openai):
    from agent_cassette.events import Event

    root = _project(tmp_path)
    run_named_record(root, "smoke", [str(_agent_script(tmp_path))], None)
    golden = root / "tests/cassettes/smoke.jsonl"
    # raw-append an observational (non-replayable) event line to the golden
    observational = Event(
        id="obs",
        timestamp="2026-01-01T00:00:00+00:00",
        type=_ET.CUSTOM,
        name="trace",
        input={"n": 1},
        output="x",
        metadata={"_agent_cassette": {"observational": True}},
    )
    with golden.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(observational.to_dict()) + "\n")
    _Responses.live_calls = 0
    result = run_named_replay(
        root, "smoke", [str(_agent_script(tmp_path))], None, match=None, strict=None
    )
    assert result.envelope.exit_code == 0
    # only the one replayable MODEL_CALL is counted, not the observational event
    assert result.envelope.data["replayable_events"] == 1
    assert result.envelope.data["consumed_events"] == 1
    assert _Responses.live_calls == 0


def test_named_replay_corrupt_golden_is_structured_invalid(tmp_path, fake_openai):
    root = _project(tmp_path)
    (root / "tests/cassettes/broken.jsonl").write_text("not-json\n{also bad\n")
    result = run_named_replay(
        root, "broken", [str(_agent_script(tmp_path))], None, match=None, strict=None
    )
    assert result.envelope.exit_code == 2 and result.envelope.status == "invalid"
    assert result.envelope.data["blockers"][0]["code"] == "cassette-invalid"
    assert _Responses.live_calls == 0  # child never ran on an invalid golden


def test_cli_report_json_root_is_clean_exit_two(tmp_path, fake_openai):
    root = _project(tmp_path)
    code, out, _ = _cli(
        [
            "record",
            "--name",
            "smoke",
            "--project",
            str(root),
            "--report-json",
            str(root),
            "--",
            str(_agent_script(tmp_path)),
        ]
    )
    assert code == 2
    payload = json.loads(out)  # a clean canonical envelope, not an ad-hoc error
    assert payload["command"] == "record" and payload["ok"] is False


def test_safe_name_survives_hostile_name():
    hostile = "z://a/" * 70  # deep enough to make redaction fail closed
    assert _safe_name(hostile) == "[unrenderable-name]"
