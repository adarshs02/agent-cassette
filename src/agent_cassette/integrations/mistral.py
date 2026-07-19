"""Transparent record and replay wrapper for the Mistral Python client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class MistralStreamingUnsupportedError(NotImplementedError):
    """Raised when a Mistral stream does not expose the expected iterator API."""


class MistralRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


MISTRAL_SPEC = ProviderSpec(
    provider="mistral",
    operations=frozenset(
        {"chat.complete", "chat.complete_async", "chat.stream", "chat.stream_async"}
    ),
    prefixes=frozenset({"chat"}),
    stream_operations=frozenset({"chat.stream", "chat.stream_async"}),
    async_operations=frozenset({"chat.complete_async", "chat.stream_async"}),
    async_probe_path=(),
    streaming_error=MistralStreamingUnsupportedError,
    raw_response_error=MistralRawResponseUnsupportedError,
)


def wrap_mistral(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a Mistral client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. Sync and async operations
    (``chat.complete`` / ``chat.complete_async``) coexist on one client; async
    routing is driven per operation by ``MISTRAL_SPEC.async_operations``.
    """
    return wrap_provider(client, cassette, MISTRAL_SPEC, asynchronous=asynchronous)
