# Connection-URI credential redaction — design

Status: **implemented-approved** (2026-07-22; accepted at `b1ba599`). Base: accepted Phase D head `f4bcad6` on
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

URIs are located by a linear `str.find("://")` scan (no regex, so no backtracking on hostile
input); at each `://` the scheme start is found by walking back over `[A-Za-z0-9+.-]` to the leading
letter. Each URI's extent is then found by a **single bracket- and adjacency-aware forward pass**
(`_uri_end`): a URI ends at whitespace, a hard delimiter (`"`, `'`, `<`, `>`, backtick, `\`, `{`,
`}`, `|`, `^`), an **unmatched** closing `)` or `]` (a surrounding prose bracket), or a `,`/`;`/`&`
that — after optional prose `()[]` — precedes the next scheme (a delimiter-separated adjacent URL).
As a second guard, `_redact_uri` ends the authority at the first `/`, `?`, `#`, **or the next scheme
start**, so a scheme embedded in the path region with no query (e.g. `…&db2=postgres://…`) is never
severed across its `://`; the remainder is scrubbed recursively.
Balanced `[...]` (an IPv6 authority `[::1]`, or the `[REDACTED]` marker) and `(...)` stay inside the
URI. Each contiguous adjacency-prose run (`,` `;` `&` `(` `)` `[` `]`) is consumed **once** — its
bracket balance, first `,`/`;`/`&`, and first unmatched closer are found in a single forward scan
and the scheme lookahead is invoked **at most once per run**, so the whole scan is O(n) even on a
long delimiter run (a pathological `scheme://h` + `"&" * N` is linear, not the earlier O(n²) that
rescanned the suffix from every delimiter). This one pass replaces the old separate trailer-peel and
guarantees every scheme occurrence in a run is visited without swallowing a neighbour, a closing
bracket, or a `,`/`;`/`&`-separated next URL. Prose between URIs (including `],[` or `),(`) is
emitted verbatim. For each URI:

1. Split scheme, then split the remainder into authority (up to the first `/`, `?`, `#`) and the
   rest.
2. **Authority userinfo**: if the authority contains `@`, split at the **last** `@` (so an
   unescaped `@` inside a password is still removed). If the userinfo contains `:`, keep the
   username and replace everything from its first `:` through the last `@` with `[REDACTED]`.
   Username-only userinfo (`scheme://user@host`) is left unchanged.
3. **Path**: the remainder before any `?` is scrubbed recursively, so a slash-embedded nested URI
   has its own userinfo/query redacted.
4. **Query**: the part after `?` and before any `#`. Split on `&`/`;` preserving separators and
   order. For a `key=value` param, decode the key for classification only; if it matches the
   `_SECRET_KEY` vocabulary (authorization, password, secret, token/access-token/refresh-token,
   api-key forms; case-insensitive) and the value is non-empty, replace the whole value with
   `[REDACTED]` — even when that value itself begins with `://`. A non-secret value and a **flag**
   segment with no `=` are scrubbed recursively, so a `scheme://…` reached via `&`/`;` (or as a
   non-secret value) still has its userinfo removed while the outer key/value text, separators,
   blank values, and path are preserved byte-for-byte. Because `_uri_end` already split a
   `,`/`;`-adjacent next URL out of the value, a secret value can be replaced whole without
   swallowing a neighbouring URL.
5. **Fragment**: recursively scrubbed for nested `scheme://…` occurrences but otherwise preserved
   byte-for-byte; a `?` after the `#` stays in the fragment, and a fragment-only `#access_token=…`
   is **not** treated as a query key.

The transform is idempotent (`[REDACTED]` re-redacts to itself) and preserves recursive
list/dict/tuple shape, acyclic aliases, and the existing cycle/depth `RedactionError` behavior.
Adjacent URIs in one run are walked iteratively (no per-URI recursion), so hundreds of
comma/semicolon-separated URLs stay shallow; only a URI genuinely nested inside a non-secret query
value recurses, and that recursion is bounded by the same `MAX_DEPTH` (64) as container redaction —
a pathologically deep chain fails closed with the identical `maximum redaction depth 64 exceeded`
error rather than a `RecursionError`. A malformed token fails safe: parsing never renders the value,
and any unexpected error raises a generic `RedactionError` whose message names only the structural
problem — never the input, password, URI, `str(value)`, or `repr(value)`.

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
