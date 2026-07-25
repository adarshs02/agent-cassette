# Compatibility policy

Agent Cassette tests the lowest declared integration version and the current locked
version from an installed wheel. A range is supported only while both boundaries
and the repository's full current-dependency suite pass.

| Surface | Supported range | Minimum gate | Current gate |
| --- | --- | --- | --- |
| Python | 3.10–3.13 | full suite on 3.10 | full suite on 3.13 |
| OpenAI Python | `>=1,<3` | installed-wheel import and adapter smoke | locked SDK and full suite |
| Anthropic Python | `>=0.34,<1` | installed-wheel import and adapter smoke | locked SDK and full suite |
| OpenAI Agents | `>=0.1,<1` | installed-wheel import and hooks smoke | locked SDK and full suite |
| Mistral Python | `>=1,<2` | installed-wheel import and adapter smoke | locked SDK and full suite |
| Gemini Python | `>=1,<2` | installed-wheel import and adapter smoke | locked SDK and full suite |
| LangChain Core | `>=0.3,<2` | Runnable/callback replay with 0.3.0 | locked Runnable/callback replay |

Core, each optional extra, and the `all` extra are installed in separate clean
environments. Smokes run outside the checkout with `PYTHONPATH` removed. Provider
and framework replay tests remove credentials and do not require a network call.

Versions outside these ranges may work, but are not part of the compatibility
contract. Upper bounds prevent a new major SDK release from silently entering a
previously validated environment. A release may narrow a range if a boundary cannot
pass the full conformance gate; it must not broaden one without a new boundary test.

## Phase D tool-replay assertions (1.1.0)

The additive `tool_called` / `tool_not_called` predicates and the read-only
`Replayer.consumed_events` property operate purely on schema-v1 core events. They add
no dependency, change no provider or framework version range in the table above, and
require no schema or `EventType` change. Their zero-live-execution substitution is
guaranteed only at supported wrapped/bridged boundaries (`wrap_tool`,
`wrap_langchain_tools`, `patch_openai_agents`, and MCP); uninstrumented side effects
(arbitrary filesystem, subprocess, HTTP, database, or browser activity) are outside the
contract and are neither replayed nor asserted on.

## Phase E agent-native loop (1.1.0)

The `setup`/`status`/`agent-manifest`/`ci` commands and named `record`/`replay`/`rerecord` are
additive CLI behavior on schema-v1 core events with no new dependency and no change to the version
ranges above. The machine envelope and manifest are schema `1`. `status` reports automatic capture
coverage (OpenAI, Anthropic, OpenAI Agents) distinctly from providers/frameworks that require their
documented explicit wrapper (Mistral, Gemini, MCP, LangChain), so it never claims automatic capture
it cannot deliver. The generated CI workflow installs dependencies (which may use the network) but
runs the test phase with provider credentials unset; supported replay boundaries are zero-live.

