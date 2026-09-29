# Retrace eval — live (run 2)

- Faults correct: 6/6
- Control false positives: 0/2
- Not graded (skipped/error): none
- Total wall time: 2543.3s
- Total tokens: 4889120
- Cost: n/a (set RETRACE_PRICE_INPUT_PER_MTOK / RETRACE_PRICE_OUTPUT_PER_MTOK)

| scenario | status | stage | failed grades | turns | tool calls | tokens in | tokens out | cache read | cache write | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| unit_cents | passed | WRITTEN_BACK |  | 14 | 22 | 28 | 5033 | 151631 | 18204 | 85.51 |
| unit_cents#t2 | passed | WRITTEN_BACK |  | 15 | 24 | 30 | 4788 | 160891 | 16003 | 79.47 |
| unit_cents#t3 | passed | WRITTEN_BACK |  | 15 | 24 | 30 | 5369 | 156675 | 15385 | 93.01 |
| schema_rename | passed | WRITTEN_BACK |  | 11 | 20 | 22 | 3827 | 88140 | 10396 | 64.98 |
| schema_rename#t2 | passed | WRITTEN_BACK |  | 11 | 21 | 22 | 3793 | 90770 | 9971 | 64.31 |
| schema_rename#t3 | passed | WRITTEN_BACK |  | 11 | 21 | 22 | 3516 | 89264 | 10008 | 59.82 |
| join_fanout | passed | WRITTEN_BACK |  | 11 | 21 | 22 | 3831 | 90805 | 11439 | 68.98 |
| join_fanout#t2 | passed | WRITTEN_BACK |  | 13 | 24 | 26 | 5425 | 144328 | 14841 | 86.13 |
| join_fanout#t3 | passed | WRITTEN_BACK |  | 12 | 22 | 24 | 4748 | 95672 | 10314 | 75.2 |
| tz_shift | passed | WRITTEN_BACK |  | 21 | 36 | 42 | 16886 | 353482 | 31722 | 214.29 |
| tz_shift#t2 | passed | WRITTEN_BACK |  | 26 | 43 | 52 | 17703 | 539587 | 38776 | 233.64 |
| tz_shift#t3 | failed | FAILED | terminal_stage | 30 | 51 | 60 | 22962 | 713562 | 46128 | 292.84 |
| stale_feed | passed | ESCALATED |  | 10 | 18 | 20 | 4991 | 80575 | 11096 | 73.25 |
| stale_feed#t2 | passed | ESCALATED |  | 9 | 17 | 18 | 4417 | 66393 | 10301 | 66.99 |
| stale_feed#t3 | passed | ESCALATED |  | 11 | 18 | 22 | 4492 | 83693 | 7633 | 67.81 |
| null_surge | passed | ESCALATED |  | 16 | 27 | 32 | 8746 | 236412 | 23104 | 121.97 |
| null_surge#t2 | passed | ESCALATED |  | 20 | 29 | 40 | 9902 | 255681 | 19715 | 144.98 |
| null_surge#t3 | passed | ESCALATED |  | 23 | 45 | 46 | 15519 | 437434 | 30724 | 199.84 |
| control_healthy | passed | NO_INCIDENT |  | 12 | 18 | 24 | 3909 | 91055 | 9731 | 70.97 |
| control_healthy#t2 | passed | NO_INCIDENT |  | 8 | 15 | 16 | 2924 | 52580 | 8915 | 53.78 |
| control_healthy#t3 | passed | NO_INCIDENT |  | 7 | 13 | 14 | 2470 | 40951 | 6753 | 44.64 |
| control_distractor | passed | NO_INCIDENT |  | 10 | 22 | 20 | 4321 | 73071 | 10182 | 62.9 |
| control_distractor#t2 | passed | NO_INCIDENT |  | 7 | 15 | 14 | 4172 | 44929 | 9232 | 65.13 |
| control_distractor#t3 | passed | NO_INCIDENT |  | 8 | 16 | 16 | 3202 | 55196 | 5518 | 53.15 |
| bad_repair_rejected | passed |  |  |  |  |  |  |  |  | 1.01 |
| datahub_timeout | passed | FAILED |  | 3 | 5 | 6 | 461 | 10102 | 2628 | 20.12 |
| rate_limit_midrun | passed | WRITTEN_BACK |  | 12 | 21 | 24 | 5314 | 111450 | 12659 | 78.58 |
