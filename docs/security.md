# Security model

Cassettes are untrusted data. Loading or replaying one parses data and reconstructs
only code-owned value envelopes and exception classes. It never imports a module,
resolves a class, registers a migration, or executes code named by cassette data.

The v1 parser rejects duplicate keys, non-finite numbers, invalid fields, cycles,
and excessive nesting. Writes serialize completely before modifying a destination;
atomic replacement is used for full saves and recovery. Unsupported Python objects
are rejected instead of invoking their `str` or `repr` methods.

Redaction runs before persistence when enabled. It recursively covers common
authorization, API-key, token, secret, and password fields and bearer values.
Cycles and excessive depth fail deterministically; shared acyclic values are safe.
Redaction is defense in depth, not a data-loss-prevention boundary: opaque secret
formats and secrets embedded in unrecognized free text may remain. Review fixtures
before committing them and use synthetic credentials in tests.

Recorded failures replay through a fixed allowlist of built-in exception types.
Unknown recorded types become `RecordedCallError`. Provider and LangChain envelopes
use fixed decoder mappings. Registered migrations, adapters, executed CLI scripts,
and live provider clients are trusted application code.

Project initialization statically inspects manifests and source evidence. It does
not execute consumer code or discover credentials. Mutating setup uses no-follow,
directory-relative operations and fail-closed rollback rules on supported POSIX
filesystems; portable runtime config reads do not weaken mutation rules.

Concurrent threads and async tasks are covered by their documented recorder and
adapter synchronization. Cross-process writers are not coordinated in 0.15; do not
have multiple processes append to the same cassette. This limitation must be
implemented or remain explicit before the stable release.

