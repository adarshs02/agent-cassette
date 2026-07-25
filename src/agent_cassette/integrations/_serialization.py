"""Serializers for values captured into cassette records.

Two internal entry points turn live integration values into cassette-safe,
JSON-native data before recording.

``serialize_recorded_value`` is the STRICT serializer. It never inspects or
invokes ``model_dump`` (or any other object method): only the exact JSON
builtin types -- ``None``, ``str``, ``bool``, ``int``, finite ``float``,
``list``, ``dict`` -- accepted by ``type(value) is ...`` identity (so tuples,
``IntEnum`` members, and every subclass of a JSON type are rejected), with
exact ``str`` keys. It keeps cycle and depth guards, returns detached plain
builtins, and validates the finished copy.

``serialize_sdk_value`` is the TRUSTED integration SDK serializer. Plain exact
JSON values recurse exactly as above; any other object is dumped through
``model_dump`` *only* when the module that defines its type falls under one of
the caller's ``trusted_roots`` (an integration SDK package such as ``openai``
or ``mcp``). An untrusted object is rejected by type name before ``model_dump``
is ever read, and the dumped result is handed to the strict serializer -- so a
dump that smuggles in another ``model_dump``-bearing object is rejected without
invoking it.

Recorded values must therefore be JSON-native OR come from a trusted
integration SDK root -- never merely "any value exposing ``model_dump``".
Rejections name only the value's type: this module never calls
``str()``/``repr()`` on an unsupported value and never coerces a non-``str``
mapping key.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

from agent_cassette.json_codec import MAX_JSON_DEPTH, StrictJSONError, validate_json_value


def serialize_recorded_value(
    value: Any,
    *,
    _depth: int = 0,
    _active: set[int] | None = None,
) -> Any:
    """Strictly serialize ``value`` into detached, JSON-native cassette data.

    Only exact JSON builtin types are accepted, tested by ``type(value) is
    ...`` identity: ``None``, ``str``, ``bool``, ``int``, finite ``float``,
    ``list``, and ``dict`` with exact ``str`` keys. ``model_dump`` is never
    inspected or called. Anything else -- a tuple, an ``IntEnum`` member, any
    subclass of a JSON type, or an arbitrary object -- raises
    ``StrictJSONError`` naming only the value's type.

    Raises:
        StrictJSONError: ``value`` (or a nested value) is not exact JSON, a
            container is cyclic or exceeds ``MAX_JSON_DEPTH``, a ``float`` is
            non-finite, or a ``dict`` has a non-``str`` key.
    """
    result = _serialize_strict(value, _depth=_depth, _active=_active)
    if _depth == 0:
        validate_json_value(result)
    return result


def _serialize_strict(value: Any, *, _depth: int, _active: set[int] | None) -> Any:
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise StrictJSONError("non-finite floats are not valid cassette JSON")
        return value
    if value_type is list or value_type is dict:
        return _serialize_strict_container(value, _depth=_depth, _active=_active)
    raise StrictJSONError(
        f"unsupported recorded value type: {value_type.__module__}.{value_type.__qualname__}"
    )


def _serialize_strict_container(value: Any, *, _depth: int, _active: set[int] | None) -> Any:
    if _depth >= MAX_JSON_DEPTH:
        raise StrictJSONError(f"maximum cassette JSON depth {MAX_JSON_DEPTH} exceeded")
    if _active is None:
        _active = set()
    value_id = id(value)
    if value_id in _active:
        raise StrictJSONError("cyclic values are not valid cassette JSON")
    _active.add(value_id)
    try:
        if type(value) is list:
            return [
                serialize_recorded_value(item, _depth=_depth + 1, _active=_active) for item in value
            ]
        serialized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise StrictJSONError("recorded value object keys must be strings")
            serialized[key] = serialize_recorded_value(item, _depth=_depth + 1, _active=_active)
        return serialized
    finally:
        _active.remove(value_id)


def serialize_sdk_value(
    value: Any,
    *,
    trusted_roots: tuple[str, ...],
    _depth: int = 0,
    _active: set[int] | None = None,
) -> Any:
    """Serialize ``value``, trusting SDK types defined under ``trusted_roots``.

    Plain exact JSON values recurse exactly as :func:`serialize_recorded_value`.
    Any other object is dumped through ``model_dump`` only when the module that
    defines its type is a trusted root (equal to a root, or nested beneath
    ``root + "."``); an untrusted object is rejected -- naming only its type --
    before ``model_dump`` is ever read. The dumped payload is then handed to the
    strict serializer, so a nested ``model_dump``-bearing value is rejected
    without being invoked.

    Raises:
        StrictJSONError: as :func:`serialize_recorded_value`, plus an untrusted
            object, or a trusted object whose ``model_dump`` is absent,
            non-callable, or uninspectable.
    """
    result = _serialize_sdk(value, trusted_roots=trusted_roots, _depth=_depth, _active=_active)
    if _depth == 0:
        validate_json_value(result)
    return result


def _serialize_sdk(
    value: Any,
    *,
    trusted_roots: tuple[str, ...],
    _depth: int,
    _active: set[int] | None,
) -> Any:
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise StrictJSONError("non-finite floats are not valid cassette JSON")
        return value
    if value_type is list or value_type is dict:
        return _serialize_sdk_container(
            value, trusted_roots=trusted_roots, _depth=_depth, _active=_active
        )
    return _serialize_trusted_object(
        value, trusted_roots=trusted_roots, _depth=_depth, _active=_active
    )


def _serialize_sdk_container(
    value: Any,
    *,
    trusted_roots: tuple[str, ...],
    _depth: int,
    _active: set[int] | None,
) -> Any:
    if _depth >= MAX_JSON_DEPTH:
        raise StrictJSONError(f"maximum cassette JSON depth {MAX_JSON_DEPTH} exceeded")
    if _active is None:
        _active = set()
    value_id = id(value)
    if value_id in _active:
        raise StrictJSONError("cyclic values are not valid cassette JSON")
    _active.add(value_id)
    try:
        if type(value) is list:
            return [
                _serialize_sdk(
                    item, trusted_roots=trusted_roots, _depth=_depth + 1, _active=_active
                )
                for item in value
            ]
        serialized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise StrictJSONError("recorded value object keys must be strings")
            serialized[key] = _serialize_sdk(
                item, trusted_roots=trusted_roots, _depth=_depth + 1, _active=_active
            )
        return serialized
    finally:
        _active.remove(value_id)


def _serialize_trusted_object(
    value: Any,
    *,
    trusted_roots: tuple[str, ...],
    _depth: int,
    _active: set[int] | None,
) -> Any:
    value_type = type(value)
    module = getattr(value_type, "__module__", "") or ""
    if not _module_is_trusted(module, trusted_roots):
        raise StrictJSONError(
            f"unsupported recorded value type: {module}.{value_type.__qualname__}"
        )
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        raise StrictJSONError(
            f"trusted recorded value has no callable model_dump: {module}.{value_type.__qualname__}"
        )
    try:
        signature = inspect.signature(dump)
    except (TypeError, ValueError) as error:
        raise StrictJSONError(
            f"trusted recorded value model_dump is not inspectable: "
            f"{module}.{value_type.__qualname__}"
        ) from error
    dumped = dump(mode="json") if _accepts_mode(signature) else dump()
    # Continue at the CURRENT depth -- the dump boundary must not reset the
    # effective total depth -- and never dump objects returned by a dump.
    return _serialize_strict(dumped, _depth=_depth, _active=_active)


def _module_is_trusted(module: str, trusted_roots: tuple[str, ...]) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in trusted_roots)


def _accepts_mode(signature: inspect.Signature) -> bool:
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "mode" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


__all__ = ["serialize_recorded_value", "serialize_sdk_value"]
