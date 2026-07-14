# CLI exit codes

Every command follows one process-level contract:

| Code | Meaning | Examples |
| --- | --- | --- |
| `0` | operation succeeded | record/replay completed; cassettes identical; doctor healthy; init current/applied |
| `1` | valid deterministic negative result | diff found changes; trajectory check failed; doctor unhealthy; init check needs changes; replay mismatch; executed child returned nonzero |
| `2` | invocation or data could not be processed | argparse usage; invalid config; missing/corrupt cassette; unsafe recovery; expected I/O failure |

Child program nonzero statuses are normalized to `1`; Agent Cassette reserves `2`
for its own usage/data failures. Signals and unhandled Agent Cassette programming
defects are not converted into expected errors.

Human-mode expected errors are concise on stderr with no traceback. For a command
that supports `--json`, success and expected failure each emit one valid JSON object
on stdout and leave stderr empty. Argparse errors retain argparse's standard stderr
and exit code 2.

