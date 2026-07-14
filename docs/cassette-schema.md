# Cassette schema and recovery

Cassette schema v1 is UTF-8 JSON Lines. Blank lines are ignored for beta
compatibility. Each nonblank line is one object with exactly these fields:

`schema_version`, `id`, `timestamp`, `type`, `name`, `input`, `output`, `metadata`,
`duration_ms`, `cost`, `parent_id`, and `span_id`.

IDs and names are nonempty strings. Timestamps are timezone-aware ISO 8601 values.
`type` is a built-in event type. Metadata is an object. Parent and span IDs are
strings or null. Duration and cost are finite, nonnegative numbers or null. Event
values are bounded, acyclic JSON values; object keys are strings. Duplicate object
keys and `NaN`/`Infinity` are invalid.

Missing `schema_version` is read as legacy schema v1. A newer version is rejected.
Migrations advance exactly one version at a time and are registered by trusted
application code; cassette data never chooses a Python import or migration.

Loading is fail-closed. Invalid UTF-8, malformed JSON, duplicate keys, invalid
fields, and unsupported versions identify the path and line without printing the
payload. A valid final record does not need a trailing newline.

## Partial-write recovery

Recovery is an explicit source-to-different-output operation:

```bash
agent-cassette recover interrupted.jsonl recovered.jsonl
```

Only an unterminated final byte fragment with invalid UTF-8 or syntax that clearly
ends mid-JSON value may be discarded. Duplicate keys, non-finite tokens, a BOM,
excessive nesting, corruption on an earlier line, a newline-terminated bad final
line, and JSON that decodes but is not a valid Event are never skipped.
The source is not modified. The report gives the discarded byte offset and count;
the output is a normalized, atomically replaced cassette.

Normal replay and migration never enable recovery implicitly. In-place migration
is deprecated; pass an explicit output path so the source remains an upgrade and
rollback point.
