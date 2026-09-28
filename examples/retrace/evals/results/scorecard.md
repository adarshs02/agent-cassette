# Retrace eval — live (run 1)

- Faults correct: 5/6
- Control false positives: 0/2
- Not graded (skipped/error): none
- Total wall time: 3141.8s
- Cost: n/a (set RETRACE_PRICE_INPUT_PER_MTOK / RETRACE_PRICE_OUTPUT_PER_MTOK)

| scenario | status | stage | failed grades | turns | tool calls | tokens in | tokens out | wall s |
|---|---|---|---|---|---|---|---|---|
| unit_cents | passed | WRITTEN_BACK |  | 13 | 24 | 178602 | 6402 | 95.47 |
| unit_cents#t2 | passed | WRITTEN_BACK |  | 11 | 22 | 139557 | 5731 | 82.3 |
| unit_cents#t3 | passed | WRITTEN_BACK |  | 13 | 22 | 174220 | 6732 | 104.85 |
| schema_rename | passed | WRITTEN_BACK |  | 18 | 37 | 314092 | 11675 | 150.17 |
| schema_rename#t2 | passed | WRITTEN_BACK |  | 14 | 26 | 190124 | 8555 | 116.13 |
| schema_rename#t3 | passed | WRITTEN_BACK |  | 11 | 19 | 129621 | 5172 | 75.74 |
| join_fanout | passed | WRITTEN_BACK |  | 14 | 27 | 275546 | 10714 | 142.51 |
| join_fanout#t2 | passed | WRITTEN_BACK |  | 12 | 23 | 187962 | 6156 | 91.28 |
| join_fanout#t3 | passed | WRITTEN_BACK |  | 10 | 18 | 132126 | 5970 | 84.86 |
| tz_shift | failed | ESCALATED | terminal_stage, files_within, mentions_all(cloudpay_v2), mentions_any(interval, hour, timezone, at time zone), invariant | 18 | 33 | 385328 | 24495 | 303.74 |
| tz_shift#t2 | passed | WRITTEN_BACK |  | 20 | 37 | 451645 | 24787 | 289.5 |
| tz_shift#t3 | passed | WRITTEN_BACK |  | 29 | 53 | 832968 | 27894 | 329.11 |
| stale_feed | passed | ESCALATED |  | 11 | 21 | 142176 | 8298 | 115.32 |
| stale_feed#t2 | passed | ESCALATED |  | 10 | 18 | 111513 | 7167 | 97.73 |
| stale_feed#t3 | passed | ESCALATED |  | 9 | 17 | 104365 | 7916 | 103.16 |
| null_surge | passed | ESCALATED |  | 14 | 26 | 198237 | 6952 | 91.68 |
| null_surge#t2 | passed | ESCALATED |  | 27 | 44 | 762468 | 18210 | 219.85 |
| null_surge#t3 | passed | ESCALATED |  | 14 | 27 | 201663 | 7422 | 99.5 |
| control_healthy | passed | NO_INCIDENT |  | 5 | 9 | 36204 | 3163 | 50.21 |
| control_healthy#t2 | passed | NO_INCIDENT |  | 5 | 10 | 35519 | 2850 | 47.18 |
| control_healthy#t3 | passed | NO_INCIDENT |  | 5 | 9 | 36185 | 3531 | 53.39 |
| control_distractor | passed | NO_INCIDENT |  | 8 | 15 | 78289 | 4204 | 59.93 |
| control_distractor#t2 | passed | NO_INCIDENT |  | 7 | 14 | 62648 | 3566 | 53.64 |
| control_distractor#t3 | passed | NO_INCIDENT |  | 7 | 12 | 59491 | 3306 | 52.04 |
| bad_repair_rejected | passed |  |  |  |  |  |  | 0.98 |
| datahub_timeout | passed | FAILED |  | 40 | 59 | 638634 | 9484 | 142.79 |
| rate_limit_midrun | passed | WRITTEN_BACK |  | 12 | 19 | 147160 | 6332 | 88.72 |
