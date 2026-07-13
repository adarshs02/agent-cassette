"""Automatic OpenAI recording and replay contexts.

This module deliberately imports the optional ``openai`` dependency only when
``patch_openai`` is entered.
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any

from agent_cassette.cassette import Cassette
from agent_cassette.integrations.openai import wrap_openai

_PATCH_LOCK = threading.Lock()
_PATCHED_MODULES: set[int] = set()


class OpenAIUnavailableError(ImportError):
    """Raised when automatic OpenAI support is requested without OpenAI installed."""


class OpenAIAlreadyPatchedError(RuntimeError):
    """Raised when the same OpenAI module is patched by a nested context."""


def _load_openai() -> Any:
    try:
        return importlib.import_module("openai")
    except ModuleNotFoundError as error:
        if error.name != "openai":
            raise
        raise OpenAIUnavailableError(
            "Automatic OpenAI recording requires the optional 'openai' package; "
            "install it with `pip install agent-cassette[openai]`."
        ) from error


def _wrapped_constructor(constructor: Any, cassette: Any, *, asynchronous: bool) -> Any:
    @wraps(constructor)
    def construct(*args: Any, **kwargs: Any) -> Any:
        if _is_offline_replay(cassette):
            return wrap_openai(None, cassette, asynchronous=asynchronous)
        client = constructor(*args, **kwargs)
        return wrap_openai(client, cassette, asynchronous=asynchronous)

    return construct


def _is_offline_replay(cassette: Any) -> bool:
    return hasattr(cassette, "position") and not hasattr(cassette, "recorder")


@contextmanager
def patch_openai(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``OpenAI`` and ``AsyncOpenAI``.

    The optional dependency is resolved on context entry. Constructor attributes
    are restored even when client code raises. Nesting against the same imported
    OpenAI module is rejected rather than risking double wrapping.
    """
    openai = _load_openai()
    module_identity = id(openai)

    try:
        original_openai = openai.OpenAI
        original_async_openai = openai.AsyncOpenAI
    except AttributeError as error:
        raise OpenAIUnavailableError(
            "The installed 'openai' package does not expose OpenAI and AsyncOpenAI clients."
        ) from error

    with _PATCH_LOCK:
        if module_identity in _PATCHED_MODULES:
            raise OpenAIAlreadyPatchedError(
                "OpenAI constructors are already patched; nested patch_openai contexts "
                "for the same module are not supported."
            )
        _PATCHED_MODULES.add(module_identity)
        openai.OpenAI = _wrapped_constructor(original_openai, cassette, asynchronous=False)
        openai.AsyncOpenAI = _wrapped_constructor(
            original_async_openai, cassette, asynchronous=True
        )

    try:
        yield
    finally:
        with _PATCH_LOCK:
            openai.OpenAI = original_openai
            openai.AsyncOpenAI = original_async_openai
            _PATCHED_MODULES.remove(module_identity)


@contextmanager
def automatic_openai_from_env(
    environ: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Open and automatically patch an OpenAI cassette configured by environment.

    ``AGENT_CASSETTE_MODE`` must be ``record`` or ``replay`` and
    ``AGENT_CASSETTE_PATH`` must identify the cassette file. The yielded value is
    the active recorder or replayer.
    """
    environment = os.environ if environ is None else environ
    mode = environment.get("AGENT_CASSETTE_MODE", "").strip().lower()
    path = environment.get("AGENT_CASSETTE_PATH", "").strip()

    if mode not in {"record", "replay"}:
        raise ValueError("AGENT_CASSETTE_MODE must be set to 'record' or 'replay'")
    if not path:
        raise ValueError("AGENT_CASSETTE_PATH must be set to a cassette file path")

    cassette_context = Cassette.record(path) if mode == "record" else Cassette.replay(path)
    with cassette_context as cassette, patch_openai(cassette):
        yield cassette


patch_openai_from_env = automatic_openai_from_env

__all__ = [
    "OpenAIAlreadyPatchedError",
    "OpenAIUnavailableError",
    "automatic_openai_from_env",
    "patch_openai",
    "patch_openai_from_env",
]
