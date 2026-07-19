from agent_cassette.integrations._provider import ProviderSpec
from agent_cassette.integrations.openai import OPENAI_SPEC


def test_provider_spec_has_empty_generalization_defaults():
    spec = ProviderSpec(provider="x", operations=frozenset(), prefixes=frozenset())
    assert spec.stream_operations == frozenset()
    assert spec.async_operations == frozenset()


def test_existing_specs_do_not_use_new_fields():
    assert OPENAI_SPEC.stream_operations == frozenset()
    assert OPENAI_SPEC.async_operations == frozenset()
