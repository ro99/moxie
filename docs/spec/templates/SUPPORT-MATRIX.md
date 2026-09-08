# Support matrix — claims must link to evidence

Use one row per checkpoint/precision/hardware/context/feature combination; do not put “all” without enumerated passing coverage. Initial status is NOT IMPLEMENTED, not supported.

Statuses: passed / failed / unmeasured / unsupported with reason / owner-deferred with decision ID. A capability probe alone proves kernel availability, not full-model quality/performance.

| Checkpoint + artifact | Hardware / topology | Actual context | Feature / combination | Status | Test / benchmark / quality IDs | Limit or fallback |
|---|---|---|---|---|---|---|
| | | | BF16 / INT8 / INT4 execution (separate rows) | | | |
| | | | First / partial / later chunked prefill | | | |
| | | | Flash/paged attention; host-backed exact state | | | |
| | | | Oversized host/disk weight streaming | | | |
| | | | TP / PP / combined / expert partition (separate rows) | | | |
| | | | Prefix reuse / continuation / cancellation | | | |
| | | | Common sampler pipeline / constraints | | | |
| | | | Future entropy, serial / batched | | | |
| | | | Lookup / draft / native-head speculation | | | |
| | | | Speculation + FE + history/constraints | | | |
| | | | FE/speculation with TP/PP and recurrent state | | | |
| | | | HTTP / CLI / SSE / reasoning / tools / images | | | |

Each automatic fallback reports the actual selected path. A required unsupported capability errors before unsafe execution. Release claims are generated from this matrix and its passing gate IDs, not from the presence of flags.
