"""Automatic provider recording and replay contexts.

This module deliberately imports the optional provider SDKs (``openai``,
``anthropic``) only when the matching patch context is entered.
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any

from agent_cassette.cassette import Cassette
from agent_cassette.integrations.anthropic import wrap_anthropic
from agent_cassette.integrations.openai import wrap_openai

_PATCH_LOCK = threading.Lock()
_PATCHED_MODULES: set[tuple[str, int]] = set()


class OpenAIUnavailableError(ImportError):
    """Raised when automatic OpenAI support is requested without OpenAI installed."""


class OpenAIAlreadyPatchedError(RuntimeError):
    """Raised when the same OpenAI module is patched by a nested context."""


class AnthropicUnavailableError(ImportError):
    """Raised when automatic Anthropic support is requested without Anthropic installed."""


class AnthropicAlreadyPatchedError(RuntimeError):
    """Raised when the same Anthropic module is patched by a nested context."""


class MistralUnavailableError(ImportError):
    """Raised when automatic Mistral support is requested without Mistral installed."""


class MistralAlreadyPatchedError(RuntimeError):
    """Raised when the same Mistral module is patched by a nested context."""


def _load_module(name: str, unavailable_error: type[ImportError]) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise unavailable_error(
            f"Automatic {name} recording requires the optional '{name}' package; "
            f"install it with `pip install agent-cassette[{name}]`."
        ) from error


def _wrapped_constructor(
    constructor: Any,
    cassette: Any,
    wrapper: Callable[..., Any],
    *,
    asynchronous: bool,
) -> Any:
    @wraps(constructor)
    def construct(*args: Any, **kwargs: Any) -> Any:
        if _is_offline_replay(cassette):
            return wrapper(None, cassette, asynchronous=asynchronous)
        client = constructor(*args, **kwargs)
        return wrapper(client, cassette, asynchronous=asynchronous)

    return construct


def _is_offline_replay(cassette: Any) -> bool:
    return hasattr(cassette, "position") and not hasattr(cassette, "recorder")


@contextmanager
def _patch_constructors(
    cassette: Any,
    module_name: str,
    constructor_names: tuple[str, str],
    wrapper: Callable[..., Any],
    unavailable_error: type[ImportError],
    already_patched_error: type[RuntimeError],
) -> Iterator[None]:
    module = _load_module(module_name, unavailable_error)
    module_identity = (module_name, id(module))
    sync_name, async_name = constructor_names

    try:
        original_sync = getattr(module, sync_name)
        original_async = getattr(module, async_name)
    except AttributeError as error:
        raise unavailable_error(
            f"The installed '{module_name}' package does not expose "
            f"{sync_name} and {async_name} clients."
        ) from error

    with _PATCH_LOCK:
        if module_identity in _PATCHED_MODULES:
            raise already_patched_error(
                f"{module_name} constructors are already patched; nested patch contexts "
                "for the same module are not supported."
            )
        _PATCHED_MODULES.add(module_identity)
        setattr(
            module,
            sync_name,
            _wrapped_constructor(original_sync, cassette, wrapper, asynchronous=False),
        )
        setattr(
            module,
            async_name,
            _wrapped_constructor(original_async, cassette, wrapper, asynchronous=True),
        )

    try:
        yield
    finally:
        with _PATCH_LOCK:
            setattr(module, sync_name, original_sync)
            setattr(module, async_name, original_async)
            _PATCHED_MODULES.remove(module_identity)


@contextmanager
def _patch_single_constructor(
    cassette: Any,
    module_name: str,
    constructor_name: str,
    wrapper: Callable[..., Any],
    unavailable_error: type[ImportError],
    already_patched_error: type[RuntimeError],
) -> Iterator[None]:
    module = _load_module(module_name, unavailable_error)
    module_identity = (module_name, id(module))
    try:
        original = getattr(module, constructor_name)
    except AttributeError as error:
        raise unavailable_error(
            f"The installed '{module_name}' package does not expose {constructor_name}."
        ) from error
    with _PATCH_LOCK:
        if module_identity in _PATCHED_MODULES:
            raise already_patched_error(
                f"{module_name} constructors are already patched; nested patch contexts "
                "for the same module are not supported."
            )
        _PATCHED_MODULES.add(module_identity)
        setattr(
            module,
            constructor_name,
            _wrapped_constructor(original, cassette, wrapper, asynchronous=False),
        )
    try:
        yield
    finally:
        with _PATCH_LOCK:
            setattr(module, constructor_name, original)
            _PATCHED_MODULES.remove(module_identity)


@contextmanager
def patch_openai(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``OpenAI`` and ``AsyncOpenAI``.

    The optional dependency is resolved on context entry. Constructor attributes
    are restored even when client code raises. Nesting against the same imported
    OpenAI module is rejected rather than risking double wrapping.
    """
    with _patch_constructors(
        cassette,
        "openai",
        ("OpenAI", "AsyncOpenAI"),
        wrap_openai,
        OpenAIUnavailableError,
        OpenAIAlreadyPatchedError,
    ):
        yield


@contextmanager
def patch_anthropic(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``Anthropic`` and ``AsyncAnthropic``.

    The optional dependency is resolved on context entry. Constructor attributes
    are restored even when client code raises. Nesting against the same imported
    Anthropic module is rejected rather than risking double wrapping.
    """
    with _patch_constructors(
        cassette,
        "anthropic",
        ("Anthropic", "AsyncAnthropic"),
        wrap_anthropic,
        AnthropicUnavailableError,
        AnthropicAlreadyPatchedError,
    ):
        yield


@contextmanager
def patch_mistral(cassette: Any) -> Iterator[None]:
    """Temporarily wrap clients created by ``mistralai.Mistral``.

    The single Mistral client carries both sync and async operations; the
    constructor is wrapped with ``asynchronous=False`` and per-operation async
    routing is handled by ``MISTRAL_SPEC``.
    """
    from agent_cassette.integrations.mistral import wrap_mistral

    with _patch_single_constructor(
        cassette,
        "mistralai",
        "Mistral",
        wrap_mistral,
        MistralUnavailableError,
        MistralAlreadyPatchedError,
    ):
        yield


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
    "AnthropicAlreadyPatchedError",
    "AnthropicUnavailableError",
    "MistralAlreadyPatchedError",
    "MistralUnavailableError",
    "OpenAIAlreadyPatchedError",
    "OpenAIUnavailableError",
    "automatic_openai_from_env",
    "patch_anthropic",
    "patch_mistral",
    "patch_openai",
    "patch_openai_from_env",
]
