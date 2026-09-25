fixture-scale, O6 open, not a performance claim

# Dense step timing

This fixed-plan measurement compares the same ignored `dense_step_timing`
harness across three repository snapshots. Each phase has five warmups and 50
measured executions; each table row reports that run's median, minimum and
maximum in microseconds. The timing brackets `execute_dense` plus `finish`.

## Hardware and revisions

- GPU: NVIDIA GeForce RTX 3090, UUID
  `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`.
- Driver: `610.57.04`; `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Candidate harness commit: `105b7263ce486237d854734fce8b164166954532`
  (parent `ebe9767ec010c2f7b5c77967d5ab6132b140c4b7`).
- Paired bases: `6b8239c85e3ebb83b57dc7dbe64d7d83054601b4` and
  `97a72813fdc106be5a84e4e3fcf4a4f3cf6b0ade`.
- On each detached base, only the appended test diff from
  `ebe9767..105b726` was applied. It applied cleanly on both; after removing
  it and returning to `main`, status showed only the carried `.gitignore`,
  `docs/evidence/specification-version.md`, and ADR 0034/0035 paths.

## Unprofiled runs

Command for each of the nine runs:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device dense_step_timing -- --ignored --nocapture
~~~

| Commit | Run | Phase | Median (µs) | Min (µs) | Max (µs) |
|---|---:|---|---:|---:|---:|
| 105b726 candidate | 1 | prefill | 1665.271 | 1617.676 | 1867.433 |
| 105b726 candidate | 1 | decode | 1604.944 | 1564.227 | 1780.171 |
| 105b726 candidate | 2 | prefill | 1674.835 | 1619.606 | 1793.598 |
| 105b726 candidate | 2 | decode | 1596.945 | 1554.432 | 1672.924 |
| 105b726 candidate | 3 | prefill | 1614.220 | 1581.105 | 1699.830 |
| 105b726 candidate | 3 | decode | 1562.666 | 1528.838 | 1612.750 |
| 6b8239c base | 1 | prefill | 1784.220 | 1739.713 | 1874.337 |
| 6b8239c base | 1 | decode | 1727.466 | 1685.051 | 1818.051 |
| 6b8239c base | 2 | prefill | 1766.639 | 1729.920 | 1816.030 |
| 6b8239c base | 2 | decode | 1712.323 | 1673.047 | 1762.830 |
| 6b8239c base | 3 | prefill | 1770.672 | 1729.390 | 1907.105 |
| 6b8239c base | 3 | decode | 1717.768 | 1677.642 | 1794.203 |
| 97a7281 base | 1 | prefill | 1733.818 | 1681.784 | 1873.023 |
| 97a7281 base | 1 | decode | 1664.340 | 1623.753 | 1782.723 |
| 97a7281 base | 2 | prefill | 1718.561 | 1678.950 | 1813.104 |
| 97a7281 base | 2 | decode | 1654.890 | 1610.493 | 1729.970 |
| 97a7281 base | 3 | prefill | 1757.613 | 1708.037 | 1851.222 |
| 97a7281 base | 3 | decode | 1684.613 | 1644.627 | 1770.485 |

## CUDA driver API breakdown

One candidate profile used Nsight Systems `2025.3.2.474-253236389321v0`:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/local/bin/nsys profile -t cuda --stats=true -o /tmp/task-0078-dense-step-timing-candidate-105b726 cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device dense_step_timing -- --ignored --nocapture
~~~

Counts and total API duration are for the whole profiled test process. The
trace has 111 `execute_dense` + `finish` invocations (55 prefill repetitions,
one committed prefill, 55 decode repetitions); the aggregate API report does
not attribute calls separately to those invocations or to prefill/decode.

| CUDA driver API | Calls | Total (ms) | Share |
|---|---:|---:|---:|
| `cuLaunchKernel` | 16539 | 61.948793 | 43.6% |
| `cuModuleLoadData` | 117 | 25.222352 | 17.7% |
| `cuMemcpyDtoDAsync_v2` | 2004 | 12.121661 | 8.5% |
| `cuEventSynchronize` | 2109 | 9.748306 | 6.9% |
| `cuMemcpyHtoDAsync_v2` | 2390 | 9.727668 | 6.8% |
| `cuModuleUnload` | 117 | 8.257984 | 5.8% |
| `cuEventRecord` | 2109 | 5.256609 | 3.7% |
| `cuCtxSetCurrent` | 46398 | 5.007336 | 3.5% |
| `cuEventCreate` | 2109 | 1.979674 | 1.4% |
| `cuMemcpyDtoH_v2` | 111 | 1.653641 | 1.2% |
| `cuEventDestroy_v2` | 2109 | 0.806705 | 0.6% |
| `cuMemFree_v2` | 8 | 0.160158 | 0.1% |
| `cuMemAlloc_v2` | 8 | 0.146625 | 0.1% |
| `cuEventQuery` | 111 | 0.084342 | <0.1% |
| `cuMemGetInfo_v2` | 1 | 0.031920 | <0.1% |
| `cuInit` | 7 | 0.012419 | <0.1% |
| `cuStreamCreate` | 1 | 0.006238 | <0.1% |
| `cuStreamDestroy_v2` | 1 | 0.004667 | <0.1% |
| `cuStreamSynchronize` | 0 | 0 | 0% |

The profiled run printed prefill median/min/max `2137.490/2102.534/2330.102 µs`
and decode `2064.001/2038.766/5394.849 µs`; profiler-run timings are excluded
from the unprofiled comparison above.

Raw trace: `/tmp/task-0078-dense-step-timing-candidate-105b726.nsys-rep`
(2,177,562 bytes), SHA-256
`05c568a6a55e94e291958cec10510a865b3dbcabe0272d963491f99c608e2b78`.
Nsight's derived SQLite report is at
`/tmp/task-0078-dense-step-timing-candidate-105b726.sqlite` (5,087,232 bytes),
SHA-256 `6e82f89fd6c5b19244ce21899f06f1f5526a164965ee85dd0d3d9686470b0bea`.
Both files remain outside Git in `/tmp` for task review and follow the host's
ordinary temporary-file cleanup.

These are synthetic Shape A fixtures measured to rank follow-up work. They do
not measure a checkpoint-backed model and do not close O6 or claim a
performance benefit.

## Module per plan

Task 0079 (`3aea3a4feee224d6421d7464299eab33bc22993c`) moves the resolved
dense module into `SelectedReservedPlan`. The first dense step loads and
resolves the candidate's symbols; later steps reuse that module. Launch reads
the module through the plan retained by the operation lease. The failure path
returns the plan with the field still `None` if initial load/resolve fails.
Work began at requested base `855e0cc`; during the assignment, `main` advanced
through coordinator commit `102e1c8`, which changes only the M6 handover. It
was preserved, so the code commit's parent is `102e1c8` and its code baseline
is unchanged from the requested base.

Three unprofiled runs used the same command and GPU as task 0078: NVIDIA RTX
3090 `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, driver `610.57.04`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device dense_step_timing -- --ignored --nocapture
~~~

| Phase | Run | 0078 candidate median (µs) | 0079 median (µs) | 0079 min (µs) | 0079 max (µs) |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 1665.271 | 1163.653 | 1132.633 | 1192.556 |
| decode | 1 | 1604.944 | 1111.448 | 1090.396 | 1134.621 |
| prefill | 2 | 1674.835 | 1148.774 | 1138.437 | 1193.031 |
| decode | 2 | 1596.945 | 1118.419 | 1089.479 | 1157.108 |
| prefill | 3 | 1614.220 | 1204.262 | 1172.659 | 1323.899 |
| decode | 3 | 1562.666 | 1147.323 | 1117.923 | 1193.731 |

One Nsight Systems `2025.3.2.474-253236389321v0` profile of the same filtered
test reported these CUDA driver API calls for the whole process:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/local/bin/nsys profile -t cuda --stats=true -o /tmp/task-0079-dense-module-per-plan-3aea3a4 cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device dense_step_timing -- --ignored --nocapture
~~~

| CUDA driver API | Calls | Total (ms) |
|---|---:|---:|
| `cuModuleLoadData` | 8 | 2.883046 |
| `cuModuleUnload` | 8 | 0.297095 |

The task 0078 profile recorded 117 calls to each API over the same 111 dense
step invocations. This profile recorded eight of each, a small whole-process
count consistent with module reuse across steps; the totals include other
process setup/teardown modules and are not phase-attributed. Profiled timing
output is not used in the unprofiled comparison above.

Raw trace:
`/tmp/task-0079-dense-module-per-plan-3aea3a4.nsys-rep` (1,848,440 bytes),
SHA-256 `343da84bdf2e8e828fa1d2f5f5ccc4cb96291b4b5c51c2843a2945139ce2c128`.
Derived SQLite report:
`/tmp/task-0079-dense-module-per-plan-3aea3a4.sqlite` (4,050,944 bytes),
SHA-256 `3e00d0bd7985419be5cb461385575f53ad253414a7e4f92ff8216ddf67d42e0a`.
Both remain outside Git in `/tmp` for task review and follow ordinary
temporary-file cleanup. These fixture-scale measurements do not close O6 or
make a performance claim.
