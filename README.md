# Agent Cassette

**Offline regression testing for tool-using Python agents.**

Record one real model-and-tool run. Replay the same agent path in CI without API
keys, network access, token spend, or real tool side effects. When behavior
changes, get a structured mismatch an AI coding agent can act on.

```text
real agent run -> committed cassette -> offline replay -> fix or explicit rerecord
```

## Get to your first offline replay

Install the integrations you use:

```bash
pip install "agent-cassette[openai]"
# Also available: anthropic, agents, langchain, mistral, gemini, all
```

Let your coding agent operate the closed loop:

> **Set up Agent Cassette, record the smoke path, and add replay-only CI.**

Or run the same non-interactive flow yourself:

```bash
# Preview, then explicitly apply project scaffolding and replay-only CI.
agent-cassette setup . --dry-run --json
agent-cassette setup . --apply --github-ci \
  --provider openai --test-framework pytest --json

# Record a real run once, then replay it by name.
agent-cassette record --name smoke -- python agent.py
env -u OPENAI_API_KEY agent-cassette replay --name smoke -- python agent.py

# Inspect machine-readable readiness at any time.
agent-cassette status . --json
```

`setup`, `status`, and `ci` never execute your project code, install
dependencies, or read secrets. Their versioned JSON envelopes include semantic
exit codes and argv-vector `next_actions`, so a coding agent can continue
without scraping prose or using a TTY.

The lower-level `init` and path-based `record`/`replay` commands remain
available for existing workflows.

## Prove a dangerous tool stays offline

`wrap_tool` records a normal Python tool at its execution boundary. During
replay, the saved result is returned and the real function body is not called:

```python
from agent_cassette import Cassette, wrap_tool

live_calls = 0


def charge_card(amount: int) -> dict[str, object]:
    global live_calls
    live_calls += 1
    return {"charged": True, "amount": amount}


with Cassette.record("tests/cassettes/refund.jsonl") as cassette:
    charge = wrap_tool(charge_card, cassette)
    assert charge(25) == {"charged": True, "amount": 25}

with Cassette.replay("tests/cassettes/refund.jsonl") as cassette:
    charge = wrap_tool(charge_card, cassette)
    assert charge(25) == {"charged": True, "amount": 25}

assert live_calls == 1  # recording ran it; replay did not
```

The same boundary supports sync functions, async functions, generators, and
async generators. OpenAI Agents `FunctionTool`s, registered LangChain tools,
and MCP tools have dedicated bridges so the surrounding agent loop can run
while real tool execution stays disabled.

## What a cassette regression test catches

Agent Cassette is useful when you already have a meaningful run and want to
protect the code around it:

- provider request or tool input changes;
- different tool selection, order, count, or error handling;
- streaming, cancellation, and retry-path regressions;
- code that no longer consumes the recorded interaction trajectory;
- accidental live model or supported tool execution during replay.

Use `tool_called` and `tool_not_called` with `assert_trajectory`, or run
`agent-cassette check --tool-called/--tool-not-called` in CI. Fork a recording
and inject failures to exercise rate limits, timeouts, and tool errors without
waiting for them to happen live.

Replay does **not** tell you whether a model will answer a new input correctly,
whether a newer model is better, or whether production behavior has drifted.
Use live evaluations for new behavior, then deliberately rerecord the regression
fixture you want to keep.

## Where it fits

| Need | Best fit |
|---|---|
| Observe live production runs | Tracing / observability |
| Score new inputs or model quality | Live evaluation |
| Replay raw HTTP traffic | VCR-style HTTP recording |
| Re-run a known model-and-tool path offline | **Agent Cassette** |

Agent Cassette is a test package, not a hosted tracing service. Cassettes are
local JSONL fixtures you choose to commit, inspect, diff, migrate, or delete.

## Supported boundaries

- **Automatic CLI capture:** OpenAI, Anthropic, OpenAI Agents
- **Provider wrappers:** OpenAI, Anthropic, Mistral, Gemini
- **Tool/framework bridges:** Python tools, MCP, OpenAI Agents
  `FunctionTool`, LangChain registered tools and runnables
- **Testing:** pytest fixtures, trajectory assertions, structured reports,
  deterministic forks, and failure injection

The core has no runtime dependency on provider SDKs. Optional integrations stay
lazy, and replay needs no provider credentials.

## Safety model

- Replay never invokes the live callable at a supported recorded boundary.
- Recorded values use bounded strict JSON validation; SDK object conversion is
  limited to the integration's trusted module root.
- Secret-named fields, bearer tokens, and connection-URI credentials are
  redacted before persistence.
- Normal reads fail closed on corruption. Recovery and migration are explicit
  and source-preserving.
- Project setup and named-run writes use confined, no-follow, atomic filesystem
  operations.

The zero-live guarantee covers supported wrapped or bridged boundaries, not
arbitrary uninstrumented side effects.

## Documentation

| Guide | What's in it |
|---|---|
| [Integrations](docs/integrations.md) | OpenAI, Anthropic, OpenAI Agents, Mistral, Gemini, LangChain, MCP, manual API |
| [Testing](docs/testing.md) | Pytest fixture, trajectory assertions, CI reports, GitHub Action |
| [Forks and failure injection](docs/forks.md) | Time-travel forks, deterministic failures, request matching |
| [CLI reference](docs/cli.md) | Human and machine-operable commands |
| [CLI exit codes](docs/cli-exit-codes.md) | Stable exit-code contract for agents and CI |
| [Security model](docs/security.md) | Redaction, replay safety, trust boundaries, current limits |
| [Compatibility](docs/compatibility.md) | Supported Python, provider, and framework versions |
| [Architecture](docs/architecture.md) | Core model and capability boundaries |
| [Cassette schema](docs/cassette-schema.md) | Strict JSONL event contract |
| [Public API](docs/public-api.md) | Exported Python surface |
| [Changelog](CHANGELOG.md) | Version history and release notes |
| [Adding a provider](docs/adding-a-provider.md) | Extend record/replay to another SDK |

Contributing and coding-agent instructions live in [AGENTS.md](AGENTS.md).

## License

See [LICENSE](LICENSE).
