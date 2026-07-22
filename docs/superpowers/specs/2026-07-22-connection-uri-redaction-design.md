# Connection-URI credential redaction — design

Status: **implemented-pending-review** (2026-07-22). Base: accepted Phase D head `f4bcad6` on
`release/1.1.0`. Implementation is committed on `release/1.1.0`; the exact base/head SHAs are in
the coding-agent handoff report. This closes the last pre-publish security blocker for the
unpublished `1.1.0`; publication also requires the final release gate.

## Threat model

`redact()` scrubbed secret-named dict keys and `Bearer` tokens, but a plain string carrying a
connection URI — `postgres://user:p@ssw0rd@db.internal/app`, `redis://:pw@cache:6379/0`,
`https://api/x?token=…` — reached raw JSONL, the viewer, and reports with its password or token
intact. Connection strings routinely appear in tool inputs, error messages, config echoes, and log
lines, so this is a real credential leak, not a documented limitation. Documentation cannot make
persisted plaintext credentials safe; the value must be scrubbed before persistence.

## Boundary

Implemented centrally in `src/agent_cassette/redaction.py`, inside the one recursive `redact()`
string leaf (after the existing `Bearer` substitution). Recorder, Hybrid, replay input
normalization (`normalize_input`), assertions, and the viewer all flow through `redact()`, so they
inherit the behavior — no integration-specific or persistence-only redactor is added. Standard
library only (`re`, `urllib.parse.unquote`); no stringify/`repr`/duck-conversion of user values; no
broad regex that erases whole URLs or ordinary `@` text.

## Algorithm

A URI token is matched by `[A-Za-z][A-Za-z0-9+.-]*://` followed by a run of non-whitespace,
non-quote/bracket characters (`"`, `'`, `<`, `>`, backtick, `\`, `{`, `}`, `|`, `^` terminate it).
Trailing sentence punctuation and a closing paren (`.,;:!?)`) are peeled off and re-appended
verbatim so surrounding prose survives byte-for-byte; `]`/`}` are **not** peeled because they are
structural in IPv6 authorities and in the `[REDACTED]` marker (peeling them would corrupt IPv6
hosts and break idempotence). For each token:

1. Split scheme, then split the remainder into authority (up to the first `/`, `?`, or `#`) and the
   rest.
2. **Authority userinfo**: if the authority contains `@`, split at the **last** `@` (so an
   unescaped `@` inside a password is still removed). If the userinfo contains `:`, keep the
   username and replace everything from its first `:` through the last `@` with `[REDACTED]`.
   Username-only userinfo (`scheme://user@host`) is left unchanged.
3. **Query**: only the part after `?` (before any `#`). Split on `&`/`;` preserving separators and
   order; for each `key=value`, decode the key for classification only and, if it matches the
   existing `_SECRET_KEY` vocabulary (authorization, password, secret, token/access-token/
   refresh-token, api-key forms; case-insensitive) and the value is non-empty, replace the value
   with `[REDACTED]`. Percent-encoded secret values are replaced whole; the key spelling/encoding,
   blank values, non-secret parameters, separators, path, and fragment are preserved.

The transform is idempotent (`[REDACTED]` re-redacts to itself) and preserves recursive
list/dict/tuple shape, acyclic aliases, and the existing cycle/depth `RedactionError` behavior. A
malformed token fails safe: parsing never renders the value, and any unexpected error raises a
generic `RedactionError` whose message names only the structural problem — never the input,
password, URI, `str(value)`, or `repr(value)`.

## False positives (left unchanged)

Ordinary URLs without password userinfo or secret query keys; bare email addresses; username-only
userinfo; `@`/`:` appearing only in a path, non-secret query value, or fragment (including a `?`
that appears after the `#`, which belongs to the fragment and is preserved verbatim); filesystem
paths and prose with colons/at-signs; non-hierarchical schemes such as `mailto:`.

## Out of scope (defense-in-depth caveats)

Consistent with "authority ends at the first `/`, `?`, or `#`" and "secret scrubbing applies to the
`?`-query only," two RFC-3986-invalid or fragment-only forms are deliberately not scrubbed and
should use percent-encoding or field-level secrets instead: a password containing an
un-percent-encoded `/`, `?`, or `#` (which prematurely ends the authority, e.g.
`postgres://user:p/w@host/db`), and a secret carried in the URL fragment with no query
(`https://app/cb#access_token=…`). These remain part of redaction's defense-in-depth posture, not a
complete DLP guarantee; percent-encode credentials (`%2F`, `%3F`, `%23`) and prefer secret-named
query keys or structured fields.

## Opt-out

`redact_secrets=False` on `Recorder`/`Hybrid` and on the viewer remains the exact, unchanged
opt-out; viewer default redaction stays on.

## Tests

`tests/test_redaction.py` adds table-driven unit matrices (every listed database/broker scheme plus
a custom scheme; last-`@`, percent-encoded credentials, IPv6/port, embedded and multiple URLs; all
secret query-key forms, mixed secret/non-secret parameters, blanks, fragment preservation, userinfo
password + query secret together; idempotence; false-positive matrix; recursive list/dict/tuple
placement with a hostile object passed through untouched) and end-to-end tests proving: `Recorder`
scrubs URI secrets from input/output/metadata with raw and percent-encoded forms absent from the
JSONL bytes; a wrapped-tool record→replay matches the original live URI through `normalize_input`
with full consumption and zero live work; `Hybrid` redacts a replayed prefix and a live suffix;
`render_viewer`/`write_viewer` scrub an in-memory unredacted event (literal and HTML-escaped forms
absent); `redact_secrets=False` preserves the raw value; and existing secret-key/`Bearer` behavior
plus `tool_called(..., with_input=…)` normalized matching stay stable. Tests scan complete generated
artifacts for seeded raw and encoded secrets, not only parsed objects.

No public API, `EventType`, schema (v1), dependency, or version (`1.1.0`) change.
