"""Transparent record and replay wrapper for the google-genai client."""

from __future__ import annotations

from typing import Any, TypeVar

from agent_cassette.integrations._provider import ProviderSpec, wrap_provider

Client = TypeVar("Client")


class GeminiStreamingUnsupportedError(NotImplementedError):
    """Raised when a Gemini stream does not expose the expected iterator API."""


class GeminiRawResponseUnsupportedError(NotImplementedError):
    """Raised for helper APIs whose raw transport semantics cannot be replayed safely."""


GEMINI_SPEC = ProviderSpec(
    provider="gemini",
    operations=frozenset(
        {
            "models.generate_content",
            "models.generate_content_stream",
            "aio.models.generate_content",
            "aio.models.generate_content_stream",
        }
    ),
    prefixes=frozenset({"models", "aio", "aio.models"}),
    stream_operations=frozenset(
        {"models.generate_content_stream", "aio.models.generate_content_stream"}
    ),
    async_operations=frozenset(
        {"aio.models.generate_content", "aio.models.generate_content_stream"}
    ),
    response_attributes=frozenset({"text"}),
    async_probe_path=(),
    streaming_error=GeminiStreamingUnsupportedError,
    raw_response_error=GeminiRawResponseUnsupportedError,
)


def wrap_gemini(
    client: Client | None,
    cassette: Any,
    *,
    asynchronous: bool | None = None,
) -> Client:
    """Wrap a google-genai client so supported calls record or replay automatically.

    Pass ``client=None`` during offline replay. Sync operations live under
    ``client.models`` and async ones under ``client.aio.models``; async routing is
    per operation via ``GEMINI_SPEC.async_operations``. ``response.text`` is captured
    at record time (``response_attributes``) so it survives offline replay.
    """
    return wrap_provider(client, cassette, GEMINI_SPEC, asynchronous=asynchronous)
