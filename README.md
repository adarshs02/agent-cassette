# Agent Cassette

Record your AI agent's real run once. Replay it offline forever — no API keys, no network, no flaky tests.

Then fork from any point, inject failures, and turn any run into a regression test.

## Install

```bash
pip install agent-cassette
```

The core is pure Python standard library. Add only the integrations you use:

```bash
pip install "agent-cassette[openai]"      # or [anthropic], [agents], [langchain], [mistral], [gemini]
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

Tool calls replay too: wrap any Python tool with `wrap_tool`, and bridge OpenAI Agents `FunctionTool`s (`patch_openai_agents`) or registered LangChain tools (`wrap_langchain_tools`) so the agent loop replays deterministically without running the real tool body.

Then verify the trajectory offline: `tool_called`/`tool_not_called` assert which recorded tool boundaries appeared (by name, input, and count), `Replayer.consumed_events` exposes exactly what a replay session consumed, and `agent-cassette check --tool-called/--tool-not-called` runs the name-only checks in CI. These inspect recorded/consumed Agent Cassette boundaries — the zero-live guarantee covers supported wrapped/bridged tools, not arbitrary uninstrumented side effects.

For agents, there's a non-interactive machine loop: `agent-cassette setup/status/agent-manifest/ci` and named `record`/`replay`/`rerecord` emit one JSON envelope (with argv-vector next actions and semantic exit codes) and write structured pass/mismatch reports — so an agent can scaffold a project, record by name, replay offline, and fix-and-retry without a TTY. `setup`/`status`/`ci` never run your code. See the [CLI reference](docs/cli.md).

## Docs

| Guide | What's in it |
|---|---|
| [Integrations](docs/integrations.md) | OpenAI, Anthropic, OpenAI Agents, Mistral, Gemini, LangChain, MCP, manual API |
| [Testing](docs/testing.md) | Pytest fixture, trajectory assertions, CI reports, GitHub Action |
| [Forks & failure injection](docs/forks.md) | Time-travel forks, deterministic failures, request matching |
| [CLI reference](docs/cli.md) | Every `agent-cassette` command |
| [Architecture](docs/architecture.md) | Core model, capabilities, security |
| [Compatibility](docs/compatibility.md) | Supported provider and framework versions |
| [Cassette schema](docs/cassette-schema.md) | JSONL event contract (v1) |
| [CLI exit codes](docs/cli-exit-codes.md) | Exit-code contract for CI |
| [Changelog](CHANGELOG.md) | Version history and release notes (1.0.0 → 1.1.0) |
| [Upgrades & migrations](docs/beta-upgrade.md) | Beta upgrade steps and cassette migrations |
| [Security model](docs/security.md) | Secret redaction (fields, bearer tokens, and connection-URI credentials), replay safety, current limits |
| [Public API](docs/public-api.md) | Exported surface |
| [Adding a provider](docs/adding-a-provider.md) | Extend record/replay to a new SDK |

Contributing? Setup and test commands live in [AGENTS.md](AGENTS.md).

## License

See [LICENSE](LICENSE).
