"""Transparent record and replay wrapper for the OpenAI Python client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class OpenAIStreamingUnsupportedError(NotImplementedError):
    """Raised when a provider stream does not expose the expected iterator API."""


class OpenAIRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


OPENAI_SPEC = ProviderSpec(
    provider="openai",
    operations=frozenset({"responses.create", "chat.completions.create"}),
    prefixes=frozenset({"responses", "chat", "chat.completions"}),
    async_probe_path=("responses", "create"),
    streaming_error=OpenAIStreamingUnsupportedError,
    raw_response_error=OpenAIRawResponseUnsupportedError,
)


def wrap_openai(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap an OpenAI client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. In that case, set ``asynchronous``
    explicitly when replaying calls made by ``AsyncOpenAI``.
    """
    return wrap_provider(client, cassette, OPENAI_SPEC, asynchronous=asynchronous)
