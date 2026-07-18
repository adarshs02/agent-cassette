# Agent Cassette

Record your AI agent's real run once. Replay it offline forever — no API keys, no network, no flaky tests.

Then fork from any point, inject failures, and turn any run into a regression test.

## Install

```bash
pip install agent-cassette
```

The core is pure Python standard library. Add only the integrations you use:

```bash
pip install "agent-cassette[openai]"      # or [anthropic], [agents], [langchain]
```

## Setup

Tell your coding agent:

> **set up agent cassette**

It detects your providers and test setup, asks you a few questions, then writes the config and an offline smoke test. (The flow agents follow lives in [AGENTS.md](AGENTS.md).)

Prefer to do it yourself:

```bash
agent-cassette init --detect
pytest tests/test_agent_cassette_smoke.py
```

## Try it

`record` runs your script live and saves every call. `replay` reruns it offline with zero live calls:

```bash
agent-cassette record run.jsonl -- python agent.py
agent-cassette replay run.jsonl -- python agent.py
```

Your code needs no changes — supported clients (OpenAI, Anthropic) are patched for the run. On replay each call returns an inert, attribute-compatible response straight from the cassette.

## Docs

| Guide | What's in it |
|---|---|
| [Integrations](docs/integrations.md) | OpenAI, Anthropic, OpenAI Agents, LangChain, MCP, manual API |
| [Testing](docs/testing.md) | Pytest fixture, trajectory assertions, CI reports, GitHub Action |
| [Forks & failure injection](docs/forks.md) | Time-travel forks, deterministic failures, request matching |
| [CLI reference](docs/cli.md) | Every `agent-cassette` command |
| [Architecture](docs/architecture.md) | Core model, capabilities, security |
| [Compatibility](docs/compatibility.md) | Supported provider and framework versions |
| [Cassette schema](docs/cassette-schema.md) | JSONL event contract (v1) |
| [CLI exit codes](docs/cli-exit-codes.md) | Exit-code contract for CI |
| [Beta & upgrades](docs/beta-upgrade.md) | Beta status and migrations |
| [Security model](docs/security.md) | Redaction, replay safety, current limits |
| [Public API](docs/public-api.md) | Exported surface |

Contributing? Setup and test commands live in [AGENTS.md](AGENTS.md).

## License

See [LICENSE](LICENSE).
