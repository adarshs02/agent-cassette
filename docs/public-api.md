# Public API inventory

The names exported by `agent_cassette.__all__` are the stable public Python API for
the 1.x release series. Additive names may arrive in minor releases, but no name is
removed and no signature is broken without a documented deprecation period.

| Area | Public names |
| --- | --- |
| Sessions | `Cassette`, `Hybrid`, `InjectionRule`, `Return`, `Raise`, `Delay` |
| Events and replay | `Event`, `EventType`, `ReplayMismatchError`, `RecordedCallError`, `RateLimitError` |
| Assertions | `AssertionReport`, `AssertionResult`, `assert_trajectory`, `check_trajectory`, `contains_event`, `event_count`, `event_sequence`, `max_total_cost`, `max_total_duration_ms`, `no_errors` |
| Comparison/reporting | `DiffReport`, `CIReport`, `compare_cassettes` |
| Providers/frameworks | `wrap_openai`, `wrap_anthropic`, `wrap_mcp`, `wrap_mistral`, `wrap_gemini`, `wrap_langchain`, `wrap_langchain_tools`, `wrap_tool`, `patch_openai`, `patch_anthropic`, `patch_mistral`, `patch_gemini`, `automatic_openai_from_env`, `AgentCassetteRunHooks`, `patch_openai_agents`, `langchain_callback_handler` |
| Interchange/viewer | `export_otlp`, `import_otlp`, `render_viewer`, `write_viewer` |
| Extensions/migrations | `Adapter`, `AdapterRegistry`, `register_migration`, `unregister_migration`, `migrate_event_dict`, `migrate_cassette` |
| Hardening | `AgentCassetteDeprecationWarning`, `CassetteCorruptionError`, `RecoveryReport`, `RedactionError`, `recover_cassette` |

The package also exposes `agent_cassette.__version__`, the installed distribution
version.

Submodules, underscored names, dataclass field layouts, report JSON not explicitly
documented as versioned, and test helpers are internal. The 1.x contract freezes
signatures, the adapter protocol, CLI tree, report envelopes, and schema guarantees
with snapshot tests.

`register_migration` executes caller-provided trusted Python and is not a cassette
plugin mechanism. Optional integration imports stay lazy so importing the core API
does not require provider or framework packages.
