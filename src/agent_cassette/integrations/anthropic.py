"""Transparent record and replay wrapper for the Anthropic Python client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class AnthropicStreamingUnsupportedError(NotImplementedError):
    """Raised when a provider stream does not expose the expected iterator API."""


class AnthropicRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


ANTHROPIC_SPEC = ProviderSpec(
    provider="anthropic",
    operations=frozenset({"messages.create"}),
    prefixes=frozenset({"messages"}),
    unsupported_operations={
        "messages.stream": (
            "The anthropic messages.stream helper is not replay-safe; "
            "call messages.create(stream=True) instead"
        ),
    },
    async_probe_path=("messages", "create"),
    streaming_error=AnthropicStreamingUnsupportedError,
    raw_response_error=AnthropicRawResponseUnsupportedError,
)


def wrap_anthropic(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap an Anthropic client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. In that case, set ``asynchronous``
    explicitly when replaying calls made by ``AsyncAnthropic``.
    """
    return wrap_provider(client, cassette, ANTHROPIC_SPEC, asynchronous=asynchronous)
