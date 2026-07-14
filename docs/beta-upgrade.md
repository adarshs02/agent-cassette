# Beta upgrade guide

## Upgrade to 0.15.0b1

Rebuild the locked environment and validate the project:

```bash
uv sync --frozen --all-extras --dev
uv run --frozen agent-cassette doctor
uv run --frozen pytest
```

Cassette disk shape remains schema v1; no data migration is required for valid v1
files. The loader now enforces the documented format. Earlier releases could write
unsupported values by calling `str`, and Python's JSON implementation accepted
duplicate keys and non-finite numbers. Such files now fail with a corruption error.
Convert intentional custom values in recorder serializers to JSON values before
recording.

An interrupted final write is not ignored during replay. Recover it explicitly to
a different output and review the reported discarded range:

```bash
agent-cassette recover old.jsonl recovered.jsonl --json
```

In-place `migrate_cassette(path)` and `agent-cassette migrate path` remain available
but emit `AgentCassetteDeprecationWarning`; use a separate destination. They will
not be removed during 1.x without another documented compatibility decision.

CLI outcomes now use the common 0/1/2 table in `docs/cli-exit-codes.md`. Automation
that depended on an executed child's exact nonzero code must treat any child
failure as 1. Expected data/configuration errors are 2.

Provider extras now have tested upper major-version bounds. If a resolver previously
selected an untested future major, pin a supported version from
`docs/compatibility.md` or wait for a compatibility release.

