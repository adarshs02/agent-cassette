# Compatibility policy

Agent Cassette tests the lowest declared integration version and the current locked
version from an installed wheel. A range is supported only while both boundaries
and the repository's full current-dependency suite pass.

| Surface | Supported range | Minimum gate | Current gate |
| --- | --- | --- | --- |
| Python | 3.10–3.13 | full suite on 3.10 | full suite on 3.13 |
| OpenAI Python | `>=1.0,<3` | installed-wheel import and adapter smoke | locked SDK and full suite |
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
previously validated environment. The release candidate may narrow a range if a
boundary cannot pass the full conformance gate; it must not broaden one without a
new boundary test.

