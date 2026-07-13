"""In-process Python execution with automatic cassette instrumentation."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_cassette.automatic import (
    AnthropicUnavailableError,
    OpenAIUnavailableError,
    patch_anthropic,
    patch_openai,
)
from agent_cassette.integrations.openai_agents import (
    OpenAIAgentsUnavailableError,
    patch_openai_agents,
)


class RunnerUsageError(ValueError):
    """Raised when an executable Python target is missing or malformed."""


def validate_python_command(command: Sequence[str]) -> list[str]:
    """Normalize and validate a CLI Python target before opening a cassette."""
    arguments = list(command)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if arguments and Path(arguments[0]).name in {"python", "python3", Path(sys.executable).name}:
        arguments.pop(0)
    if not arguments:
        raise RunnerUsageError("expected a Python script or '-m module' after '--'")
    if arguments[0] == "-m":
        if len(arguments) < 2:
            raise RunnerUsageError("'-m' requires a module name")
    elif not Path(arguments[0]).resolve().is_file():
        raise RunnerUsageError(f"Python script does not exist: {Path(arguments[0]).resolve()}")
    return arguments


def run_python(command: Sequence[str], cassette: Any) -> int:
    """Run a Python script or module while automatically patching supported SDKs."""
    arguments = validate_python_command(command)

    original_argv = sys.argv[:]
    original_path = sys.path[:]
    try:
        with (
            _optional_openai_patch(cassette),
            _optional_anthropic_patch(cassette),
            _optional_openai_agents_patch(cassette),
        ):
            if arguments[0] == "-m":
                if len(arguments) < 2:
                    raise RunnerUsageError("'-m' requires a module name")
                module = arguments[1]
                sys.argv = [module, *arguments[2:]]
                return _run(lambda: runpy.run_module(module, run_name="__main__", alter_sys=True))

            script = Path(arguments[0]).resolve()
            if not script.is_file():
                raise RunnerUsageError(f"Python script does not exist: {script}")
            sys.argv = [str(script), *arguments[1:]]
            sys.path.insert(0, str(script.parent))
            return _run(lambda: runpy.run_path(str(script), run_name="__main__"))
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def _run(execute: Any) -> int:
    try:
        execute()
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(error.code, file=sys.stderr)
        return 1
    return 0


@contextmanager
def _optional_openai_patch(cassette: Any) -> Iterator[None]:
    try:
        context = patch_openai(cassette)
        context.__enter__()
    except OpenAIUnavailableError:
        yield
        return
    try:
        yield
    finally:
        context.__exit__(None, None, None)


@contextmanager
def _optional_anthropic_patch(cassette: Any) -> Iterator[None]:
    try:
        context = patch_anthropic(cassette)
        context.__enter__()
    except AnthropicUnavailableError:
        yield
        return
    try:
        yield
    finally:
        context.__exit__(None, None, None)


@contextmanager
def _optional_openai_agents_patch(cassette: Any) -> Iterator[None]:
    try:
        context = patch_openai_agents(cassette)
        context.__enter__()
    except OpenAIAgentsUnavailableError:
        yield
        return
    try:
        yield
    finally:
        context.__exit__(None, None, None)
