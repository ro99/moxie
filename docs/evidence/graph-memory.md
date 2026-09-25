# Captured graph memory measurements

Task 0086 measured graph executable memory with `RankContext::memory_info()`
on the 5060 Ti and both 3090s. Each pass synchronized before reading free
memory, captured and instantiated one graph with 16,384 sequential AXPY
launches, synchronized and read memory, then captured 256 one-launch graphs,
synchronized and read memory again. Every graph stayed live through both
readings. The initial assertions used
`B = F = u64::MAX / (1_u64 << 20)` so the measurement was unconstrained.

Both passes were exact repeats. The deltas are bytes; `node` measures the one
large graph, and `graphs` measures the additional 256 small graphs.

| GPU UUID | Pass | Node free before | Node free after | Node delta | Graph free before | Graph free after | Graph delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (RTX 5060 Ti) | 1 | 16,510,025,728 | 16,468,082,688 | 41,943,040 | 16,468,082,688 | 16,440,819,712 | 27,262,976 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (RTX 3090) | 1 | 25,015,222,272 | 24,897,781,760 | 117,440,512 | 24,897,781,760 | 24,872,615,936 | 25,165,824 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` (RTX 3090) | 1 | 25,015,222,272 | 24,897,781,760 | 117,440,512 | 24,897,781,760 | 24,872,615,936 | 25,165,824 |
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (RTX 5060 Ti) | 2 | 16,510,025,728 | 16,468,082,688 | 41,943,040 | 16,468,082,688 | 16,440,819,712 | 27,262,976 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (RTX 3090) | 2 | 25,015,222,272 | 24,897,781,760 | 117,440,512 | 24,897,781,760 | 24,872,615,936 | 25,165,824 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` (RTX 3090) | 2 | 25,015,222,272 | 24,897,781,760 | 117,440,512 | 24,897,781,760 | 24,872,615,936 | 25,165,824 |

Each per-device delta repeated exactly (ratio 1.0), within the required factor
of two. The rule uses the first pass:

- Maximum node delta: 117,440,512 bytes. `ceil(117,440,512 / 16,384) = 7,168`; the next power of two is 8,192. Applying the 256-byte floor gives `CAPTURED_KERNEL_BOUND_BYTES = 8,192`.
- Maximum graph delta: 27,262,976 bytes. `ceil(27,262,976 / 256) = 106,496`; the next power of two is 131,072. Applying the 4,096-byte floor gives `CAPTURED_GRAPH_BOUND_BYTES = 131,072`.

The resulting enforced limits are 134,348,800 bytes for one 16,384-node graph
(`16,384 × 8,192 + 131,072`) and 35,651,584 bytes for 256 one-node graphs
(`256 × (8,192 + 131,072)`). Both exceed the largest measured deltas.
