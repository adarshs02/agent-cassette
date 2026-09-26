"""The static DataHub catalog must never hint at the injected faults (no datahub dependency)."""

from typing import Any

from retrace.datahub.catalog import DATASET_DOCS, FIELD_DOCS, OWNERS, TAGS

FAULT_VOCABULARY = ("cents", "duplicate", "stale", "null surge", "timezone shift", "currency_code")


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for item in value for s in _strings(item)]
    return []


def test_catalog_strings_are_scanned():
    for source in (DATASET_DOCS, FIELD_DOCS, OWNERS, TAGS):
        assert _strings(source), "nothing to scan"


def test_catalog_never_mentions_fault_vocabulary():
    leaks = [
        (term, text)
        for source in (DATASET_DOCS, FIELD_DOCS, OWNERS, TAGS)
        for text in _strings(source)
        for term in FAULT_VOCABULARY
        if term in text.lower()
    ]
    assert leaks == []
