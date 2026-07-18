# Forks, failure injection, and matching

Replay a known prefix then continue live, optionally forking on the first request that
diverges:

```bash
agent-cassette fork baseline.jsonl experiment.jsonl --at 4 -- python agent.py
agent-cassette fork baseline.jsonl prompt-change.jsonl --mismatch live -- python agent.py
```

Inject deterministic failures with an ordered, occurrence-based rules file (`raise`,
`return`, `delay`, rate limits):

```bash
agent-cassette fork baseline.jsonl chaos.jsonl --inject failures.json -- python agent.py
```

## Request matching

Matching is exact and ordered by default. Loosen it when requests carry dynamic fields:

```python
with Cassette.replay("run.jsonl", match="subset", ignore_paths=("kwargs.metadata.request_id",)) as cassette:
    ...
# match="normalized" ignores whitespace/case in strings; match="fuzzy" compares by similarity ratio.
# strict=False allows parallel calls to complete out of order while still consuming every event.
```
