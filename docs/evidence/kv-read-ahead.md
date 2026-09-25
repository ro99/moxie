# Host-backed KV read-ahead timing

Measured on **2026-09-25** with `CUDA_DEVICE_ORDER=PCI_BUS_ID` on RTX 3090
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (SM86). The ignored
`host_backed_read_ahead_timing` test used the same one-resident/three-staged
history for `stage_next` and read-ahead, with five warm-up queries and thirty
measured queries per mode. Values are median microseconds per query.

| Run | `stage_next` | Read-ahead |
|---:|---:|---:|
| 1 | 348.022 | 346.241 |
| 2 | 346.940 | 349.557 |

The two runs show no consistent timing advantage: read-ahead was 1.781 µs
faster in the first run and 2.617 µs slower in the second. The harness did
not measure copy-versus-kernel overlap from event intervals, so these results
describe only the paired query times and make no performance claim.
