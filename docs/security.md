# Security model

Cassettes are untrusted data. Loading or replaying one parses data and reconstructs
only code-owned value envelopes and exception classes. It never imports a module,
resolves a class, registers a migration, or executes code named by cassette data.

The v1 parser rejects duplicate keys, non-finite numbers, invalid fields, cycles,
and excessive nesting. Writes serialize completely before modifying a destination;
atomic replacement is used for full saves and recovery. Recorded values must be
JSON-native or come from a trusted integration SDK root (the provider's own package,
e.g. `openai` or `mcp`), which is dumped through its `model_dump`; the `model_dump` of
a value defined outside those roots is never read or called. Unsupported Python objects
are rejected instead of invoking their `str` or `repr` methods.

Redaction runs before persistence when enabled. It recursively covers common
authorization, API-key, token, secret, and password fields and bearer values. It also
scrubs credentials embedded in hierarchical connection URIs: the password in
`scheme://user:password@host` userinfo (using the last `@` so an unescaped `@` inside a
password is still removed) and secret-named URL query values
(`?password=`/`?token=`/`?api_key=`/…), for any valid scheme — `postgres`, `mysql`,
`redis`, `mongodb`/`mongodb+srv`, `amqp`, and custom schemes alike. Scheme, username,
host/port (including IPv6), path, non-secret query parameters, fragment, and surrounding
prose are preserved byte-for-byte; the scrub is idempotent and runs in the one recursive
`redact()` path, so it reaches recorder, hybrid, replay input normalization, assertions,
and the viewer uniformly. Cycles and excessive depth fail deterministically; shared
acyclic values are safe.

Redaction is defense in depth, not a complete data-loss-prevention boundary: opaque
secret formats, ordinary URLs without recognized userinfo/query secrets, and secrets
embedded in otherwise unrecognized free text may still remain. Review fixtures before
committing them and use synthetic credentials in tests.

Recorded failures replay through a fixed allowlist of built-in exception types.
Unknown recorded types become `RecordedCallError`. Provider and LangChain envelopes
use fixed decoder mappings. Registered migrations, adapters, executed CLI scripts,
and live provider clients are trusted application code.

Tool-boundary assertions (`tool_called`, `tool_not_called`) verify recorded/consumed
Agent Cassette boundaries only; they do not prove the absence of uninstrumented side
effects, and the zero-live-execution guarantee applies solely to supported
wrapped/bridged tools (`wrap_tool`, `wrap_langchain_tools`, `patch_openai_agents`, MCP).
An expected `with_input` is detached and validated through the same exact-type JSON
copier as recorded values — subclasses, tuples, cycles, depth overflow, non-finite
floats, and non-`str` keys are rejected without calling `str`/`repr`/conversion methods —
and assertion messages and `details` carry only JSON-native diagnostics (tool name,
counts, indexes), never a raw input/output payload.

Project initialization statically inspects manifests and source evidence. It does
not execute consumer code or discover credentials. Mutating setup uses no-follow,
directory-relative operations and fail-closed rollback rules on supported POSIX
filesystems; portable runtime config reads do not weaken mutation rules.

Concurrent threads and async tasks are covered by their documented recorder and
adapter synchronization. Cross-process writers are not coordinated in the current
implementation; do not have multiple processes append to the same cassette. This is a
documented current limitation, not a defect.

