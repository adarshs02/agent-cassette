"""Wire Anthropic and DataHub clients to Agent Cassette for record, replay, and injection."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_cassette import Cassette, InjectionRule, wrap_anthropic
from retrace.config import Settings
from retrace.datahub.mcp_client import DataHubConnection

Mode = Literal["live", "record", "replay"]


@dataclass
class Clients:
    anthropic: Any
    datahub: DataHubConnection


def _default_model() -> Any:
    import anthropic

    return anthropic.Anthropic(max_retries=4)


@contextmanager
def open_clients(
    mode: Mode,
    *,
    settings: Settings,
    cassette_path: Path | None = None,
    injections: Sequence[InjectionRule] = (),
    source: Path | None = None,
    model_factory: Callable[[], Any] | None = None,
    datahub_factory: Callable[[Any | None], DataHubConnection] | None = None,
) -> Iterator[Clients]:
    make_model = model_factory or _default_model
    make_datahub = datahub_factory or (lambda c: DataHubConnection.live(settings, c))

    if mode == "live":
        datahub = make_datahub(None)
        try:
            yield Clients(make_model(), datahub)
        finally:
            datahub.close()
        return

    if cassette_path is None:
        raise ValueError(f"{mode} mode needs a cassette_path")

    if mode == "replay":
        with Cassette.replay(cassette_path) as cassette:
            datahub = DataHubConnection.replay(cassette)
            try:
                yield Clients(wrap_anthropic(None, cassette), datahub)
            finally:
                datahub.close()
        return

    if injections and (source is None or not source.exists()):
        raise FileNotFoundError(f"fault injection needs an existing source cassette: {source}")
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    if injections:
        session = Cassette.fork(source, cassette_path, at=0, injections=injections)
    else:
        session = Cassette.record(cassette_path)
    with session as cassette:
        datahub = make_datahub(cassette)
        try:
            yield Clients(wrap_anthropic(make_model(), cassette), datahub)
        finally:
            datahub.close()
