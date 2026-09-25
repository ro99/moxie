# DeepSeek: Strata's speed techniques mapped onto Moxie

Task [0093](../tasks/0093-m7-prep-deepseek-strata-speed-map.md) (M7 preparation), 2026-09-25. This
is a documentation-only record. Nothing was built or run, and no GPU was used.
The Strata checkout (`/home/rodrigo/Developer/strata`, HEAD `2dc566e`) was
treated as read-only.

**Why this exists.** The owner ruled that Moxie must eventually show DeepSeek
speed at least equal to Strata's
([M5→M6 handover](../handovers/2026-09-24-m5-closure-to-m6.md), lines
270–280). The comparison itself is M7 work. This map lists every technique that
made Strata's DeepSeek V4 fast, cheaper in memory, or was tried and rejected,
and says where each one lives in Moxie. The point is to find gaps now rather
than in M7.

**What Strata ran versus what Moxie will run.** Strata ran the original
checkpoint `models/dsv4f`, which has "35,328 native FP4 E2M1/per-32 E8M0
modules and 390 native FP8 E4M3/128x128 E8M0 modules" (Strata
`docs/deepseek-v4-runtime.md:38-39`). Moxie's v1 catalog entry is
`Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound`: INT4 weights with 16-bit
activations. Moxie never quantizes ([ADR 0017](../decisions/adr/0017-v1-catalog-and-no-quantizer.md)
line 14) and has no FP4/FP8 runtime family
([ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md) line 10).
Every Strata kernel built for the FP4/FP8 representation is therefore a gap
row. Its note says what the technique bought, so that the equivalent need
stays visible for Moxie's INT4/INT8 kernels.

## How to read this file

**Sources read.**
- **Every DeepSeek experiment record** in Strata's
  `experiments/docs/experiments/`: numbers 0006–0195, 133 files, including
  non-`dsv4`-named records whose subject is DeepSeek, such as 0013–0015,
  0020 and 0114.
- **Strata docs:** `docs/models/deepseek.md`, `docs/deepseek-v4-runtime.md`,
  `docs/dsv4-rank-local-architecture.md` and `docs/flash-attention.md`.
- **Strata code:** the 16 headers in `include/strata/models/deepseek/`, the
  15 sources and 6 `detail/` files in `src/models/deepseek/`, and the kernel
  files they launch (`kernels/cuda/deepseek_rank_local_layer_executor.cu`,
  and in `kernels/cuda/detail/`: `backend_kernels.cuh`,
  `backend_model_kernels.cuh`, `backend_moe.inc.cuh`, `backend_core.inc.cuh`,
  `backend_prepared_attention.inc.cuh` and `backend_indexing.inc.cuh`).
- **Strata history:** about 235 performance commits in `git log`, read with
  `git log` and `git show` only.
- **Moxie:** the sources the contract names.

Subagents did the first reading pass. The builder then re-opened every number
used in Section 1 and in the top-ten list, and every Moxie line cited, before
writing.

**Citation shorthand.** Every Strata citation is relative to the Strata root.
- **`E0123:75`** is line 75 of `experiments/docs/experiments/0123-*.md`.
  Where two files share a number, the full name is given, as in
  `E0006-…-baseline-handoff`.
- **`ds.md`** is `docs/models/deepseek.md`.
- **`rt.md`** is `docs/deepseek-v4-runtime.md`.
- **`rl.md`** is `docs/dsv4-rank-local-architecture.md`.
- **`fa.md`** is `docs/flash-attention.md`.
- **`c:abc1234`** is a Strata commit; the quoted text is from its message.
- **Other Strata source paths** are abbreviated. A `deepseek_*.hpp` file is in
  `include/strata/models/deepseek/`. A `deepseek_*.cpp` or `runtime_*` file is
  in `src/models/deepseek/`, and the `runtime_*` files are in its `detail/`
  directory. A `backend_*` file is in `kernels/cuda/detail/`.

Moxie citations are repository paths, or `01:14`-style references to
`docs/spec/NN-*.md` line numbers.

**Grouping rule** (coordinator ruling, `DECISION`/`ANSWER` of
`task-0093-round-1`). There is one row per technique.
- **Same technique, several sources.** A technique found in an experiment, a
  doc and a commit is one row citing all three.
- **Tuning campaigns.** A campaign that tuned one kernel or one chain is one
  row. Its evidence cell gives the accepted variant's recorded gain first
  (exact numbers, with units), then cites each rejected variant by
  `file:line`.
- **Split campaigns.** If a campaign produced two genuinely different accepted
  techniques, they are split into two rows.
- **Correctness-only records are not rows.** These are pure correctness,
  contract or measurement records with no speed or memory effect: for example
  0010, 0013, 0062, 0064, 0067, 0079–0081, 0085–0091, 0104, 0108, 0118, 0121,
  0125, 0135, 0157 and 0160–0163. They were read and excluded.
- **Moxie-forbidden representations are gap rows.** Techniques tied to a
  weight or cache representation that Moxie does not admit (FP4/FP8 weights,
  or a KV cache below 16 bits) are **gap** rows that cite the rule. Their Note
  says what the technique bought.

**Moxie home values** (five, per the contract as amended after review round
2):
- **present**: the cited Moxie code does what the Strata technique does, by
  the same or an equivalent mechanism, with the same effect.
- **partial**: Moxie has part of it. The Note names the missing part.
- **not needed**: Moxie's design removes the cost the technique addressed. The
  code that removes it is cited.
- **planned**: a named roadmap item covers this exact technique, not merely
  its area. A DeepSeek-only row maps to the M7 DeepSeek row (`06:120`) only
  when it implements mathematics that row names (compressed/sparse state,
  special routing, mHC, rank-local semantics).
- **gap**: none of the above, including a rule that forbids it.

A rejected Strata technique still gets a home, because a later agent may try
it again. Gap rows do not propose designs.

**Scope column.**
- **Generic** means the technique does not depend on DeepSeek's equations.
- **DeepSeek-only** names the equation it depends on: sparse attention and the
  indexer, compressed KV, mHC, hash routing, or DSpark.
- **Generic (FP4)** and **Generic (FP8)** mark a general engine technique tied
  to a weight format that Moxie does not admit.

## 1. Strata baseline (recorded figures only)

Reference hardware for every row: two RTX 3090 24 GiB cards at 1,605 MHz /
250 W, 251 GiB RAM, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, devices `1,2`, VRAM
fraction 0.95, rank-local TP2, 16K context, and no static expert tier unless
stated (`ds.md:187-189`). The command is the `strata-chat` or `strata-server`
invocation at `ds.md:81-128` with `--decode-topology rank-local-tp2
--prefill-page-tokens 8192`.

| Figure | Prompt / shape | Value | Kind | Source |
|---|---|---:|---|---|
| **Prefill (accepted production result)** | 1,925 prompt tokens | **26.231 tok/s** | median of three interleaved arms (21.592, 27.828, 26.231) | `ds.md:191-192`; `E0195:16-18`; landed in `c:fa67531` |
| Prefill, disabled-path control | 1,925 | 20.882 tok/s | median of 21.581, 20.882, 19.142 | `ds.md:193-194` |
| Prefill | 619 | 17.10 tok/s | single screen | `ds.md:195-196` |
| Prefill | 133 | 7.84 tok/s | single screen | `ds.md:196` |
| **Decode (production)** | 128-token response after a 19-token prompt | **8.627 tok/s** | median, server | `ds.md:199-201` |
| Decode, long context, no tier | ~2,000 words | 7.302 tok/s | earlier experiment | `ds.md:202-203` |
| Strata before device query | ~36 / ~500 / ~1,950-token prefill | 4.332 / 17.431 / 22.757 tok/s | baseline matrix | `ds.md:213` |
| vLLM on the same machine | ~36 / ~500 / ~1,950 prefill; decode | 8.439 / 41.901 / 127.490; 10.346 tok/s | baseline matrix | `ds.md:212` |
| Decode with the 10 GB static tier | short / long context | 8.571 → 9.171; 7.302 → 8.077 tok/s | before device query | `ds.md:221-223`; `E0178:83-84` |
| Long prefill with the tier | long prompt | 18.147 → 16.649 tok/s | before device query | `ds.md:223-224` |
| Rank-local decode step (corrected) | fixture chain | 111.177092 ms, `8.994659 tok/s` | Nsight-corrected | `rl.md:404-408` |
| Rank-local decode at landing | three interleaved reps | "rank-local 110.610722 ms/token = 9.040715 token/s / centralized 129.416257 ms/token" | median | `c:516ef3d` |
| Decode step breakdown | 115.9 ms/token | "MoE 72.81, non-MoE layer 41.54, rest 7.42" | attribution | `c:be720ed` |
| Supported context ceiling | — | 65,536 tokens | stated limit | `c:516ef3d`; `rl.md:598-605` |

**Strata's accepted production result is 26.231 prefill tok/s at 1,925 prompt
tokens, together with 8.627 decode tok/s** (`ds.md:191`, `ds.md:200`). Strata
itself records that this prefill figure "is not vLLM-equivalent"
(`ds.md:217-218`). Its stated bottleneck is decode's "3.449 GB of routed-expert
weight read from host DRAM per token" (`ds.md:154-155`).

## 2. The technique map

### 2.1 Weights and experts (residency, tiers, host MoE)

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 1 | Routed experts live in host DRAM; VRAM holds the dense spine plus an expert cache; no NVMe reads in steady state | both | `rt.md:111-120` "Main routed-expert source extents \| 147,169,738,752", "Steady-state NVMe \| 0"; `ds.md:7-10` ("147 GB against 48 GB of aggregate VRAM", line 8) | accepted | Generic | **partial**: host cache tier in `ResidencyAuthority` (`crates/moxie-memory/src/residency.rs:1282`, task 0020) | Missing: a guarantee that the whole routed set stays in DRAM. `host_cap_bytes` is configurable (`residency.rs:1200-1203`), and a miss re-reads the checkpoint (`crates/moxie-executor/src/residency.rs:335`). |
| 2 | One capacity-bounded CUDA weight arena per GPU instead of per-weight `cudaMalloc`/`cudaFree` | both | `E0011:34` "10.13% decode throughput improvement"; `:45` decode allocations "7,716 \| 0"; `:46` prefill wall "4.5967 s \| 3.2035 s \| -30.31%" | accepted | Generic | **present**: one bounded `DeviceBuffer` per device scope (`crates/moxie-executor/src/residency.rs:471-499`); `DeviceArena` (`arena.rs:298`, task 0010) | |
| 3 | Read checkpoint tensors straight into the final resident buffer, with no temporary buffer or copy | load | `E0008:54-55` "190.97 seconds to 144.12 seconds" | accepted | Generic | **present**: `crates/moxie-executor/src/residency.rs:31-38` `ChunkSource::read_chunk` fills the authority's own slot | Load time is not measured in Moxie. |
| 4 | Parallel checkpoint reads (8 readers); contiguous-block dispatch for cold reads | load | `E0008:49` "Parallel resident reads \| 149.0748 s \| 64.2800 s \| 2.319x"; `include/strata/platform/worker_pool.hpp:46-49` "0.69 GB/s taking single expert rows and 1.96 GB/s taking contiguous blocks"; `c:890439f` (rank-local, not measured) | accepted | Generic | **gap** | No parallel checkpoint reader in `moxie-storage` or in `drain_reads`, and no roadmap item for load time. |
| 5 | Parallel GPU spine warm-up, overlapped with host staging | load | `E0008:51` "1.267x"; `:52` "Overlapped stage/warm-up \| 31.4793 s \| 22.1771 s \| 1.419x"; `:55-56` lineage "from approximately 191 seconds to 22.177 seconds, about 8.6x"; `rt.md:202-207` | accepted | Generic | **gap** | No load-overlap mechanism or roadmap item. |
| 6 | FP8 attention projections widened to BF16 once at load (block-wise E4M3 lookup for `wo_a`; rank-local projections) | load | `E0008:50` "Block-wise `wo_a` lookup \| 60.7659 s \| 36.6405 s \| 1.658x"; `E0076:195-196` "43-layer incremental expansion: 4,148,166,656 B"; `E0143:105-106`; `c:725a68f` | accepted | Generic (FP8) | **gap**: FP8 weights not admitted (ADR 0003 line 10; ADR 0017 line 14) | Bought dequantization-free decode projections for 4,148,166,656 B extra per rank. For INT8 the same trade is a prepared-weight cache under `03:122`. |
| 7 | FP4 experts transformed once into two rank TP shards (32-row output blocks, duplicated group scales), replacing the canonical bytes | load | `deepseek_host_expert.hpp:82-85`; `deepseek_checkpoint.cpp:383-460`; `E0059:302-304` "155,826,782,208 bytes (145.125 GiB) … adds only 8,657,043,456 bytes for duplicated scales" | accepted | Generic (FP4) | **gap**: FP4 weights not admitted (ADR 0003 line 10) | Bought a single host copy in a rank- and kernel-friendly layout. For INT4 this is the repack question that ADR 0027 leaves provisional. |
| 8 | GPU prefill streams experts from the host arena's copy, not from the checkpoint files | prefill | `E0096:76` "69.9 GB of reads becomes 0.57 GB"; `:106-107` "101.05 s" → "93.46 s"; rejected: rebuild canonical bytes on the host `E0096:64` "1.21 GB/s single-threaded, or 58 s of rebuild per page" | accepted | Generic | **partial**: uploads come from the host cache (`crates/moxie-executor/src/residency.rs:381` `drain_reads`) | Missing: freedom from checkpoint re-reads, because a host-cache miss reads the file again (`residency.rs:335` `perform_read`). Strata's transformed layout is FP4-specific (row 7). |
| 9 | NUMA: bind each expert-arena shard to its node before first touch; node-local first touch for CPU reads | load, decode | `E0058:49-50` "bind … 56.7 GB/s", "no-bind … 22.9 GB/s"; `E0050:141` "both nodes, node-local \| 28 \| 76.3"; `deepseek_checkpoint.cpp:377-391`; `E0106:74-75`. Rejected for DMA: `E0026:120-121` "NUMA binding alone is 1.02x" | accepted | Generic | **partial**: `mbind` placement of grouped-run buffers (`crates/moxie-executor/src/grouped.rs:387`, task 0021) | Missing: NUMA binding of the expert source itself. The host cache is an ordinary `HostBuffer` (`crates/moxie-memory/src/residency.rs:1434`), and `moxie-host/src/numa.rs` only reads topology. |
| 10 | Transparent hugepages (`MADV_HUGEPAGE`) on the expert arena | decode | `E0122:135` "Decode 0.956x"; `deepseek_runtime.hpp:116-125` "costs 12-20 s of extra staging"; `E0058:54-55` THP does not materialize | rejected | Generic | **gap** | `03:56` bars global memory flags "on folklore". |
| 11 | Page-lock (`cudaHostRegister`) the resident expert arena | decode | `E0024:82` "Decode steps/s \| 1.913 \| 3.110 \| 1.626x"; `:84` demand wait "3.33x less"; `:85` "Registration (s) \| 0 \| 16.88"; `deepseek_runtime.hpp:76-81`. Rejected under rank-local admission: `E0188:73` VRAM overruns "74,145,792 B"; `E0096:170-175` "about 72 MB of device-side mapping". Chunked registration rejected: `E0024:58-59` | opt-in | Generic | **planned**: M6.3 pinned transfers (handover line 120; `03:51`) | `PinnedHostBuffer` (`crates/moxie-cuda/src/driver.rs:985`) exists; there is no arena registration. |
| 12 | Pinned bounce staging for expert H2D in prefill | prefill | `E0189:56` "6.1207 GB/s \| 6.5649 GB/s"; `:58-61` "1.0726x" | rejected | Generic | **planned**: M6.3 (`03:51` "bounded pinned bounce buffers") | |
| 13 | One MoE command per GPU, with every GPU enqueued before any result is collected | decode | `E0006-…-baseline-handoff:77` "1.8018641620x"; `:81` syncs "161,846 \| 62,285" | accepted | Generic | **partial**: device grouped expert kernels on each GPU, run concurrently by rank threads (tasks 0035, 0062) | Missing: one MoE command per GPU. Moxie launches two kernels per expert group (`expert_mlp.cu:1-7`), and M6.1 does not name the enqueue-all-then-collect protocol. |
| 14 | Launch many routed experts per kernel (a layer's experts together) to fill the waves | decode | `E0142:77-82` 1 expert "393.5 \| 450.7", 32 experts "797.9 \| 793.5" GB/s; `E0155:162-164` "one expert at a time is 0.26 and loses roughly 40% of throughput"; grouped prerequisite `E0028:89-90` "cut MoE launches 97.9% … end-to-end neutral" | accepted | Generic | **planned**: M6.1 batched/grouped expert kernels (`06:100`) | Moxie's grouped kernel (`crates/moxie-kernels/cuda/expert_mlp.cu`) has no wave-occupancy measurement. |
| 15 | Register-fed FP4 expert GEMV decoder (PRMT lookup for E2M1, BF16 `HMUL2` for the E8M0 scale), a campaign on one kernel | decode | Accepted: `E0137:5-6` "810.93 GB/s cold on both production shapes … 1.58x faster, and 97.5% of the measured read floor"; real weights `E0156:73-74` "636.4" / "668.5" GB/s. Starting point `E0057:211` "44.6 GB/s against 936". Rejected: shift/rebias decoder `E0136:5-6` "514.02 GB/s"; conventional shared-memory path `E0134:118-119`; 16-byte broadcast scale load `E0140:108`; activation pre-permute at M=1 `E0140:109` | accepted | Generic (FP4) | **gap**: FP4 weights not admitted (ADR 0003 line 10) | Bought 4-bit expert decode near SM86's read floor. Moxie's INT4 expert kernels (`expert_mlp.cu:182`) have no bandwidth measurement. |
| 16 | Decoded weights fed straight into `mma.sync` registers, with a load-time fragment-order prepack (a permutation, same bytes, one copy) | decode, load | `E0139:52,55` MMA cost "0.63%" / "0.64%"; `E0140:61` "Prepack cost is 4–5 ms per matrix, one-time at load"; `backend_core.inc.cuh:755-759` prepack on the upload stream "0.509 ms/token"; `c:9a4d185` | accepted | Generic | **partial**: fused dequantization into WMMA tiles (`crates/moxie-kernels/cuda/affine_linear.cu:41`, task 0028) | Missing: register-fed MMA operands and the load-time fragment prepack. Moxie dequantizes through shared memory. A prepacked layout would fall under ADR 0027. |
| 17 | Skinny-GEMV shaping: split-K chosen per expert width and M, write only the real column at M=1, fold split-K into the launch, unroll the loop, store only `min(M,8)` fragments, skip partials at split 1 | decode | Accepted: `E0148:20-27` M=1 "742.9"/"749.9" GB/s; `E0155:67-68` split-K 4 "635.4" / "668.4"; `E0141:22-23` "+13% / +22%", "+8% / +8%"; `E0148:45-47` unroll restored "742.9"; `E0159:52-53` "80.23% to 82.92%". Rejected: more split-K `E0140:107`; more loads in flight `E0142:21-22`; separate reduction "Nearly a wash" `E0141:57` | accepted | Generic | **gap** | No split-K or skinny-GEMV shaping exists, and M6.1 names grouped kernels, not this. |
| 18 | Register-fed FP8 shared-expert kernels (fused gate/up, in-place prepack), the default | decode | `ds.md:228-230` "worth about 1.7% here (118.7 -> 116.7 ms/tok)"; `E0164:74-76` "No throughput measurement". Rejected: direct-bit E4M3 decode `E0069:1289` "fail: must be <=80.000 ms". Also `E0168:18-20` cites "2.027 against 2.038 ms", which is absent from 0164 | accepted | Generic (FP8) | **gap**: FP8 weights not admitted (ADR 0003 line 10) | Bought 1.7% of the decode step. The shared expert is "the only per-token CUDA dispatch decode makes" (`ds.md:231-232`). |
| 19 | Static hot-expert VRAM tier, chosen offline from decode route traces | decode | `E0178:83-84` "108.9 ms/tok, 9.171 tok/s", "1.070x", long context "1.106x", prefill "0.917x"; `E0124:58` "38.6% against a 10.4% null"; `c:6a52f9a` "8.424 -> 9.425 tok/s, 1.119x"; saturation `E0178:70-72` "106.4 against 106.1"; 14 GB "VRAM failure" `E0178:43`; `ds.md:131-167` | opt-in | Generic | **planned**: M6.3 optional hot-expert tiers (`06:102`; `03:56`) | |
| 20 | Tier dispatch ordering: the callback writes the selection to a pinned slot and never blocks | decode | Accepted: `c:2c5a880` the first version blocked and "was 30x slow". Rejected: blocking per-layer handoff `E0124:187-188` "still running after 15 minutes"; overlapped tier split `E0178:40` "2 of 7" distinct outputs, `deepseek_rank_local_layer_executor.cu:469-487` "one corrupt"; overlap probe `E0126:75` "1.00x" | accepted (serial) | Generic | **gap** | Area: M6.3 hot-expert tiers (`06:102`) names the area, not this technique. |
| 21 | Advisory trace-driven expert prefetch (past-only predictor, bounded queue, byte budget) | decode | `E0022:36` "reduces total modeled bytes by 49,267,000,000 (7.61%)"; `:34` "84.66%" useful; `deepseek_runtime.hpp:102-106`; `c:bdf332c` "keep prefetch H2D at zero"; never run end to end (`E0026:180-182`) | opt-in (default 0 predictions) | Generic | **planned**: M6.3 route prediction (`06:102`) | The `Urgency::Prefetch` class exists (`residency.rs:348`). |
| 22 | Expert prefetch driven by DSpark draft routes | decode | `E0029:187-189` "upload wait was 4.840 seconds, while the observed draft alone cost 4.912 seconds" | rejected | DeepSeek-only (DSpark) | **gap** | Area: M6.3 route prediction (`06:102`) names the area, not this technique. |
| 23 | Dynamic LRU expert cache / frequency-aware placement (projected only) | decode | `E0124:112` "3090s LRU 17.7 GB only … 13.26" (simulation); `E0050:368-369` "83% → 90% hit would cut host bytes from 586 MB to 345 MB per token" (projection) | proposed | Generic | **partial**: demand LRU for device and host scopes (`residency.rs:2952-2967`, task 0020) | Missing: frequency-aware placement. Both halves were only projected in Strata. |
| 24 | O(1) eviction through intrusive recency lists (was a full scan per miss) | decode | `E0057:48` "14.3 ms/step"; `deepseek_runtime.cpp:463-468` "walked roughly 670,000 hash nodes a step"; `E0120:502-503` | accepted | Generic | **gap** | `eviction_candidates` collects and sorts every evictable placement on each call (`crates/moxie-memory/src/residency.rs:2952-2967`). There is no O(1) recency list. |
| 25 | Concurrent routed-expert uploads across devices (one wait per device) | decode, prefill | `E0052:87` "1.90x, 49.95 ms/step" (probe); `:170-171` "Re-priced at layer granularity the mechanism is 1.25x / 12.9 ms"; `deepseek_runtime.cpp:585-592`; `c:7f3394e` "roughly 6%"; proposed resident/arriving split `E0057:207-209` | accepted (default) | Generic | **planned**: M6.3 measured transfer overlap (`06:102`) | |
| 26 | Per-device upload copy stream, ordered by an event instead of a host block | decode | `E0057:106` "host wait 64.8 → 0.07 ms/step"; `backend_core.inc.cuh:836-846` "64.5 ms of a 235 ms decode step"; probe `E0031-dsv4-…:84` "1.89x–1.96x". Earlier rejection `E0026:43-44` "+0.9%, +0.7% and +5.0%", withdrawn by 0031 | accepted | Generic | **partial**: a copy stream ordered by events exists for KV read-ahead (`crates/moxie-executor/src/paged_attention.rs:1368`, task 0092) | Missing: an upload copy stream for weight and expert residency. |
| 27 | Bandwidth-weighted layer/expert placement across GPUs | decode | `E0051:147` "Bandwidth weighting is worth 1.04x. Rejected." | rejected | Generic | **planned**: M6.5 plan selection from measured costs (`06:104`) | |
| 28 | Experts spread across devices round-robin (centralized mode) | both | `deepseek_runtime.cpp:1864-1866`; not measured | accepted | Generic | **partial**: expert-partitioned plans (task 0066) | Moxie assigns contiguous expert ranges (`crates/moxie-plan/src/tensor_parallel.rs:893-894` at HEAD `60fc509`), not `expert % devices` round-robin. |
| 29 | Every routed expert statically on the CPU (centralized three-GPU design) | decode | `E0054:105` "Best case 0.939x against a gate of 1.15x"; `E0058:4-6` "449.2 ms/step … a 0.49x decode"; `deepseek_host_expert.hpp:16-20` | rejected | Generic | **present**: `Candidate::Host` (`crates/moxie-plan/src/expert.rs:84`) | Rejected in the centralized design; row 31 is the later accepted rank-local form. |
| 30 | Divert cache-miss experts to the host kernel | decode | `E0051:344` "0.76x. Rejected."; write-behind `E0051:244` "4.15x regression"; `deepseek_host_expert.hpp:11-15` | rejected | Generic | **gap** | Moxie picks the CPU/GPU split per plan (`crates/moxie-plan/src/expert.rs:84`). Nothing diverts a single miss at run time. Rejected in Strata. |
| 31 | Decode: routed experts on NUMA-local CPU, shared expert on GPU, joined through a stream-ordered host callback and a device join kernel | decode | `E0069:445-447` "Complete MoE is 164.387 ms … routed-GPU arm had spent 396.429 ms … Routed-expert weight H2D falls from 1,377,042,432 bytes to zero"; `E0069:543-545`; `deepseek_executor.cpp:39-50`; `c:bce98dd` | accepted (production bundle) | Generic | **partial**: routed experts on the host (`crates/moxie-plan/src/host_experts.rs:56`, task 0068) | Missing: the concurrent shared expert and the stream-ordered callback with a device join. Host experts run synchronously after `stream.synchronize()` (`crates/moxie-executor/src/dense.rs:1933`). |
| 32 | Each rank's CPU expert half is a TP split of the intermediate dimension on its own NUMA node | decode | `E0059:267` "median: 75.57 ms/token", `:269` "45.64 GB/s"; first attempt rejected `E0077:125-126` "0.973474"; `E0078:70` "faster by 4.039980 ms (4.97458%)" but residual gate failed `:108` | accepted (production) | Generic | **gap** | Tasks 0066 and 0068 partition GPU experts and own host experts. Neither splits a host expert across per-rank NUMA pools. |
| 33 | CPU MoE pool of one worker per physical core (no SMT siblings), disjoint per-rank CPUs | decode | `E0123:3` "Decode 3.066 -> 8.780 tok/s on the server, 2.86x"; `E0123:72,75` "28 logical \| 3.19 \| 222.3", "14 (one per physical core) \| 8.49 \| 72.7"; `deepseek_rank_local_topology.cpp:197-206` | accepted | Generic | **gap** | Moxie's host expert kernel (`crates/moxie-kernels/src/cpu_expert.rs:152`) runs on one thread. There is no host worker pool. |
| 34 | Addressed per-lane dispatch with same-node work stealing in the host MoE pool | decode | `E0059:590` "-25.160 ms (-9.3%)"; `:594` routed CPU "-14.0%"; stealing across the reduction rejected `:558-559`; `deepseek_host_moe_executor.cpp:118-185` | accepted | Generic | **gap** | Same missing pool as row 33. |
| 35 | Expert workers spin for up to 1 ms before sleeping | decode | `E0059:522-523` "improves total by 70.983 ms, 22.4%, against the immediate-sleep arm"; a 10 ms spin was not retained (`:524`) | accepted (1 ms) | Generic | **gap** | Same missing pool as row 33. |
| 36 | Bit-exact AVX2 host FP4 expert kernel (int8 `shuffle_epi8` lookup, 8 accumulator chains) | decode | `E0051:169` "against 6.9 GB/s for the PCIe path it replaces: a 4.1x on the mechanism"; `:166` "2.07 \| 26.23"; `:202-203` interleave "a further 1.09x"; rejected float-LUT variant `E0050:180-181` "16 GB/s" | accepted | Generic (FP4) | **gap**: FP4 weights not admitted (ADR 0003 line 10) | Bought host 4-bit expert reads 4.1x the PCIe path. Moxie's INT4 host kernel (`cpu_expert.rs:179`) has no bandwidth measurement. |
| 37 | Group a page's rows per expert on the CPU (decode each weight tile once for 4 rows) | prefill | `E0094:47` "2.315x"; `E0095:145` "2.180x"; `c:b28219d` "1.204x"; superseded by GPU prefill experts (`c:fe8f9e5`) | superseded | Generic | **partial**: `expert_group_affine` takes one expert's row group (`cpu_expert.rs:179`) | Missing: weight-tile reuse across rows. It loops rows outermost (`:314-339`), so each tile is re-read per row. Superseded in Strata. |
| 38 | Hash-router `tid2eid` table fully resident (no checkpoint I/O in decode) | decode | `rl.md:512-516` | accepted | DeepSeek-only (hash routing) | **planned**: M7 DeepSeek "special routing" (`06:120`) | |

### 2.2 Attention and KV

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 39 | Shared CUDA FlashAttention backend: a campaign covering the crossover, block-parallel softmax and forced dispatch | both | Accepted: block-parallel softmax `E0057:65` "18.2 → 9.7 ms/step"; forced dispatch in the bundle (`deepseek_runtime.hpp:72-75`). Rejected: as default `E0014:28` "`0.520x`"; production flash decode `E0015:103` "1.288 \| 1.148 \| -10.9%"; tiled F32/F64 online softmax on numerics `E0013:82-85`; 256-row hybrid `E0027:89-90` "1.024x" / "1.010x" (superseded by row 40); `fa.md:92-98` | opt-in → forced in the bundle | Generic | **partial**: exact device online-softmax attention with a 128-key tile (`crates/moxie-kernels/cuda/paged_attention.cu:1-60`, task 0037) | Missing: FlashAttention source, tensor cores, tuning and crossover dispatch, which ADR 0032 names as absent. |
| 40 | Batched multi-query attention: pack a page's KV union once, with a per-query visibility mask | both | `E0030:89` "1.308x on prefill and 1.0079x on decode"; `:75` FlashAttention calls "21,973 \| 344"; `:97-98` "forced dispatch beats hybrid by 1.399x on prefill" | accepted | Generic | **partial**: one launch serves every query row of a chunk (`paged_attention.cu:1-9`) | Missing: packing a shared KV union once with a per-query mask. Pages are scanned per row and head. No roadmap item names this. |
| 41 | One pinned H2D and one D2H per attention call | decode | `E0015:59-60` "from about 3.07 ms to 2.00 ms"; `fa.md:50-54` | accepted (opt-in path) | Generic | **not needed**: Moxie attends over device-resident KV with device query and output (`attend_into`, `crates/moxie-executor/src/paged_attention.rs:3070`) | There is no per-call H2D/D2H for this to reduce. |
| 42 | Device-resident KV with incremental append and in-place row patch | decode | `E0069:1083` "KV H2D \| 6,428,672 bytes \| 25,112 bytes"; `:1087` "179.192 ms \| 164.503 ms"; estimate `E0051:411-412` | accepted | Generic | **present**: `append_paged_layer_from_device` (`crates/moxie-executor/src/paged_attention.rs:5400`, task 0076) | |
| 43 | KV pages committed lazily; the context ceiling is never touched | both | `rt.md:81-84`; `rl.md:849-853` "uses only `7,213,568` bytes of KV payload per rank"; `E0012:53-57` (performance not classified) | accepted | Generic | **gap** | `PagedSequence` reserves and allocates its whole admitted capacity at construction (`crates/moxie-state/src/paged.rs:227-240,523-529`). |
| 44 | Tiered typed block KV manager: fixed blocks, bounded host residency, device-promotion leases, fork/COW | both | `E0018:55-56` "50,495,488-byte peak" under a 79,839,232-byte cap; `:62-63` "457.389/1.037 s" vs "456.509/1.024 s" | opt-in | Generic | **partial**: fixed device pages with COW (`crates/moxie-state/src/device.rs:156`); bounded host-backed staging (`paged_attention.rs:1165-1183`, tasks 0041–0045) | Missing: one tiered typed-block manager with promotion and demotion under leases. |
| 45 | Compact block KV: FP8 E4M3 + E8M0 for non-RoPE, BF16 RoPE, FP4 learned index | both | `E0019:51-52` "3,707,490,816 bytes … versus 14,451,884,032 bytes for the F32-backed plan"; `c:0701487`; decode cost `E0032:41` "1.118x" (scalar faster); `rt.md:90-94` | opt-in | DeepSeek-only (compressed KV) | **gap**: cache below 16 bits (`01:14`; `03:129`; ADR 0019 item 2) | Bought 3,707,490,816 vs 14,451,884,032 KV/index bytes at a 1,048,576-token ceiling. ADR 0019 (lines 20-30) already settled the policy; M7 must bring the family evidence it requires. |
| 46 | Physical device paged FP8 DS-MLA KV (256-token block-major pages, 584 B/token) with an FP8 index, the production bundle | decode | `E0063:212` "total cache + persistent state \| 151,228,416"; `:214-215`; `deepseek_kv_cache.hpp:26-27` `PhysicalFp8E4m3Group64Bf16Rope`; `rl.md:880` | accepted | DeepSeek-only (compressed KV) | **gap**: cache below 16 bits (`01:14`; `03:129`; ADR 0019 item 2) | Bought a 151,228,416-byte cache plus state (`E0063:212`). ADR 0019 (lines 20-30) already settled the policy; M7 must bring the family evidence. |
| 47 | 256-token allocator blocks; compressed pages of 256/ratio rows | both | `deepseek_attention_kv.hpp:16`; `deepseek_kv_cache.cpp:159-165` | accepted | DeepSeek-only (compressed KV) | **planned**: M7 DeepSeek "compressed/sparse state" (`06:120`) | |
| 48 | O(1) physical KV block lookup (uniform guess, then binary search, then scan), on host and device | decode | `deepseek_runtime.cpp:1554-1560` "measured 2,889.637 ms per decoded token across the 43 layers" before; `backend_model_kernels.cuh:1167-1182` | accepted | Generic | **not needed**: the device page table is indexed directly (`crates/moxie-kernels/cuda/paged_attention.cu:184`) | Direct indexing is already O(1), so there is no search to speed up. |
| 49 | Reuse block-table buffers (no allocation in the timed path) | decode | `deepseek_runtime.cpp:1782-1787` "cost about 8.1 ms/token" | accepted | Generic | **gap** | Each `PageView` hands a freshly built `Vec<u32>` table to `publish_page_table_deferred` (`paged_attention.rs:2305-2310,5285`). |
| 50 | Positional page numbering with lazy first-touch leasing; cache the positional page prefix | decode | `c:4115d69` "161.338 ms/token" → "159.673 ms/token"; `c:2cd2f20` "it removes 0.061 ms" | accepted | DeepSeek-only (compressed KV) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 51 | Page-width-aware KV admission and an explicit host-KV budget | prefill | `E0119:66-67` "prefill 197.5 s / 19.47 tok/s after" (3,845 tokens; failed before); `E0110:72` host peak "21,729,792 B \| 81,616,576 B"; `:77-78` prior budget "about 5.7x below" | accepted | Generic (sliding window plus compressed stream) | **present**: capacity `min(window + tentative_rows, max_tokens)` (`crates/moxie-state/src/paged.rs:227-240`); `Ledger::admit` (`ledger.rs:355`) | |
| 52 | Incremental KV continuation: a later turn prefills only the uncached suffix | prefill | `E0020:50` "27 / 7.108 \| 14 + 13 / 4.241 \| equal"; `deepseek_runtime.hpp:71`; `c:a5a5899` | accepted | Generic | **present**: continuation via device COW fork (task 0050); `PrefixReuseKey` (`crates/moxie-state/src/prefix_reuse.rs:18`, task 0049) | Task 0049's decision has no consumer yet. |
| 53 | Tiled / multi-pass compressed-KV attention to lift the 640-candidate, 65,536-token ceiling | decode | `E0093:53` "65,536 \| 512 \| 640 \| … \| ok"; the "0.484 ms/layer" at `:95` is labelled not a result | proposed | DeepSeek-only (compressed KV) | **planned**: M7 DeepSeek (`06:120`) | The generic analogue is present: N-block streaming (tasks 0041–0043). |
| 54 | GPU Lightning Indexer v1 and v2 (device key cache, device scores) | both | v1 `E0017:51` "+22.2%", `:54` "+6.1%"; v2 `E0021:36-37` "4.145 s" scalar vs "4.524 s" CUDA | rejected (v1); not promoted (v2) | DeepSeek-only (indexer) | **planned**: M7 DeepSeek sparse selection (`06:120`; `Op::SparseIndexSelect` declared at `crates/moxie-graph/src/lib.rs:82`) | |
| 55 | Physical E4M3 lightning index with radix top-k select (three 16-bit passes) | decode | `c:bfd59ed` "66.3 ms/token at 1M, from 842 ms"; `rl.md:730` "Scoring, `2.15x`"; `rl.md:739` "Radix pivot, `3.01x`" | accepted | DeepSeek-only (indexer) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 56 | Index-query RoPE and E4M3 quantization on device | decode | `c:6bd0059` "rank-local 174.285 -> 158.353 ms/token" | accepted | DeepSeek-only (indexer) | **planned**: M7 DeepSeek (`06:120`) | |
| 57 | In-chain device sparse selection: projection, score, top-k and candidate resolution all enqueued | decode | `c:1224d83` "rank-local 159.673 -> 115.795"; `rl.md:769` "`43.878 ms/token`"; supersedes "chain only below 2,048 tokens" (`rl.md:926-932`); rejected host scalar indexing at 1M `rl.md:885-889` "`192 ms/token` floor" | accepted | DeepSeek-only (indexer) | **planned**: M7 DeepSeek (`06:120`) | |
| 58 | Shard candidate scoring across ranks | decode | `rl.md:813-820` "a strict lower bound of `156.656105 ms/token`" | rejected | DeepSeek-only (indexer) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 59 | Sparse scores: score only each row's attended candidates, not a dense GEMM | prefill | `E0117:43-44` "attention @2,612 \| 78.139 s \| 60.437 s", "score @2,612 \| 51.363 s \| 32.277 s"; `c:ab34b84`; rejected barrier removal `E0190:11-13` "only 4.13 seconds of a 100.42-second forward" | accepted | DeepSeek-only (sparse attention) | **planned**: M7 DeepSeek (`06:120`) | |
| 60 | Row-batched physical prefill attention: gather pages once into a flat workspace, one score op per page | prefill | `E0098:88-92` "paged-attention kernel launches \| 553,109 \| 1,677", "attention score \| 18.324 s \| 13.945 s" (gate failed); promoted inside `E0107:169-171`. Rejected: page-set hoist `E0097:82` "candidate +1.9%"; prepared `index_select` `E0099:125` "14.159 s"; gather-once `E0111:181-184` | accepted (in the 0107 stack) | DeepSeek-only (sparse candidates) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 61 | Page attention bounded by workspace: exact-extent 384 MiB cap, binary-searched query sub-chunks | prefill | `E0099:167-168` "226,492,416 bytes less on each GPU"; `E0109:90` "+1.912258 s" (functional reject; code active per `E0110:62-63`) | accepted | Generic | **not needed**: Moxie's online-softmax kernel materializes no score matrix (`paged_attention.cu:20-51`) | There is no score workspace to cap or sub-chunk. |
| 62 | Fold the value division into the attention finish | prefill | `c:8a6db3f` "Attention at 677 tokens 13.240 s -> 11.979 s" | accepted | Generic | **present**: online-softmax finish inside `paged_attention.cu:85` (task 0037) | |
| 63 | Hold the RoPE-decoded attention result as BF16 | prefill | `c:92b6163` "844.56 -> 681.37 MB … no timing win is claimed" | accepted | Generic | **present**: the kernel rounds its output to BF16 once (`paged_attention.cu:31` `RoundingProfile::FinalBf16Rne`) | |
| 64 | Exact physical-page attention output (materialize, cuBLAS BMM, reference finish); native WMMA page attention rejected | decode | `E0068:269-271` "0.058368", "0.031744", "0.057344" ms; rejected `E0066:87-88` (correctness) | accepted | DeepSeek-only (sparse MLA) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 65 | KV replicated on both ranks rather than sharded | decode | `rl.md:158-162`; `rl.md:870-872` sharding "would reintroduce a cross-rank fetch" | accepted | Generic | **planned**: M7 DeepSeek rank-local semantics (`06:120`) | Moxie's MLA head partition is task 0067. |
| 66 | Rank 0 alone runs the compressor and publishes the row bytes to both ranks | decode | `rl.md:179-186`; `deepseek_rank_local_weights.cpp:286` "would add roughly 0.6 GiB" | accepted | DeepSeek-only (compressor) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |

### 2.3 Multi-GPU (rank-local, TP, transfers)

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 67 | Rank-sharded TP2 ownership with a slice loader that reads only this rank's byte ranges | load | `E0073:328` "6,223,273,412 per rank"; `:327` "peak loader temporary \| 529,530,880 per rank"; `deepseek_rank_local_weights.hpp:82-89`; `rl.md:66-75` | accepted | Generic | **partial**: rank sharding lowered as graph slices (`crates/moxie-plan/src/tensor_parallel.rs:498` at HEAD `60fc509`, tasks 0057, 0060) | Missing: the slice loader. `CanonicalSource::read_chunk` requires a whole-tensor range (`crates/moxie-executor/src/residency.rs:255-264`). |
| 68 | FP32 all-reduce (NCCL over SHM, no P2P), 86 per forward, BF16 publication | decode | `E0074:237` "FP32 NCCL sum → BF16 \| 86 \| … \| 2.845 / 2.817–2.884 ms"; `c:be720ed` "all 86 per token cost 1.36 ms -- 3.3%" | accepted | Generic | **partial**: an exact FP32 declared-order reduction (`moxie_tp_reduce_f32_v1`, `crates/moxie-kernels/cuda/dense_ops.cu:73`; ADR 0036) | Missing: an NCCL-over-SHM all-reduce. Moxie moves partials by peer copy (`dense_tp_workers.rs`), and the 86-per-forward cost is unmeasured. |
| 69 | Host-staged TP reduction (pinned D2H, host sum, H2D) | decode | `E0074:52` "11.275 ms" (serialized); `:145` "25.997 ms" | rejected | Generic | **gap** | Moxie's TP path (`crates/moxie-executor/src/tensor_parallel.rs:281`) moves partials by peer copy. There is no pinned D2H→host→H2D reduction. Rejected in Strata. |
| 70 | BF16 collective sum instead of FP32 | decode | `E0074:235` "2.808 / 2.785–2.814 ms"; `:241-243` "not chosen merely for speed" | rejected | Generic | **gap**: TP reductions are exact in a declared FP32 order (ADR 0036; `AGENTS.md:212`) | Would have saved 0.037 ms against the FP32 arm (row 68) at `E0074:235,237`. |
| 71 | Vocab-parallel embedding all-reduce and LM-head all-gather | decode | `E0074:332` "0.040 / 0.040–0.041 ms"; `:324` "0.088 / 0.081–0.091 ms" | accepted | Generic | **partial**: LM-head all-gather (`crates/moxie-executor/src/tensor_parallel.rs:359`) | Missing: the vocab-parallel embedding all-reduce. |
| 72 | Rank-local TP2 attention, 32 heads per rank | decode | `E0076:227` "22.388704 ms \| 20.874752 ms \| 6.762%"; probe `E0075:181-183` "FAIL" | accepted | Generic | **present**: dense TP2 on the pair (task 0060); MLA head partition (task 0067) | |
| 73 | Callback-free queued 43-layer chain with one completion per token | decode | `E0092:72` "114.944312" ms against centralized "151.155686 ms/forward" (`E0087:9`); `c:516ef3d` "9.040715 token/s". Rejected: driving the executor layer by layer `c:27c67b0` "418 ms/step against the centralized arm's 148" | accepted (production topology) | Generic | **planned**: M6.1 device-resident layer chains (`06:100`) | |
| 74 | The RTX 5060 Ti as a third TP rank | decode | `c:62786da` "0051 measured that at 0.79x"; `ds.md:233` | rejected | Generic | **gap** | Moxie runs TP only on the matched 3090 pair. The 5060 Ti joins through a pipeline (task 0071), not as a TP rank. Rejected in Strata. |
| 75 | Cross-device grouped ownership through fixed pinned staging and one reusable event | decode | `E0069:1441` "synchronization calls \| 197 \| 132 \| -65 (-33.0%)"; `:1449` "-2.516 ms single-arm" | accepted | Generic | **partial**: cross-device expert ownership over peer copies (task 0066; `driver.rs:1378`) | Missing: fixed pinned staging slots and one reusable event. |
| 76 | Cross-request layer-major wavefront batching (expert union across requests) | decode | `E0028:78` "+0.06%"; `:81` "+2.81%" | rejected | Generic | **gap**: one interactive user (`01:12`; `04:7`) | Bought +2.81% at 16 requests, which is out of scope for a single user. |

### 2.4 Kernels and launch (fusion, graphs, streams)

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 77 | FP4 E2M1 decode by exhaustive switch, then branch-free | both | `E0006-…-baseline-handoff:44-45` "a 14.1% single-run increase"; `c:ca87539` "RTX 3090 1.47 -> 0.77 ms 1.91x"; `E0126:105` "0.143 ms against the host's 0.282 ms" | accepted | Generic (FP4) | **gap**: FP4 weights not admitted (ADR 0003 line 10) | Bought 1.91x on the batch-1 4-bit GEMV. Moxie's INT4 nibble decode is unmeasured. |
| 78 | Coalesced GPU expert kernel over transformed shards (a warp owns a 32-row block); row tile 8 → 32 | prefill | `E0096:83-84` "takes the kernel from 21.67 s to 4.02 s" (`:80` records canonical addressing as "2.87x slower"); `:108` row tile 32 "86.25 s"; `:96-97` tile 64 "spills the accumulators"; `backend_kernels.cuh:3620` | accepted | Generic (FP4) | **gap**: FP4 weights not admitted (ADR 0003 line 10) | Bought 21.67 s → 4.02 s on the prefill expert kernel. Moxie's `expert_mlp.cu` has no coalescing study. |
| 79 | SM86 FP8 tensor page tile for multi-row projections (E4M3 → BF16 WMMA, E8M0 per K128); also used on the fused `wo_b` | prefill | Accepted: `E0105:101` "Query matmul device \| 7.158365 s \| 0.338589 s \| 21.14x faster"; `:103` KV "6.25x faster"; `E0116:60` "5.685 s \| 0.020 s"; `:63` total GPU kernel "17.58 s" → "12.47 s". Rejected: plain-BF16 cuBLAS unreachable `E0101:73` / 0102; router and unfused `wo_b` "No effect" `E0116:40-43`; W8A8 incumbent `E0143:222` | accepted (defaults true) | Generic (FP8) | **gap**: FP8 weights and E4M3 activations not admitted (ADR 0003 line 10) | Bought 21.14x on page query projections. The INT8 home to measure is `affine_linear.cu:41` (qualified `max_input` 16,384). |
| 80 | Same-input fusion `wq_b + indexer.wq_b` in the register-fed W8A16 kernel | decode | `E0145:87` "58.368 \| 718.64 \| 84.98%" vs `:86` "682.71 \| 81.11%" | accepted (passes the 82% gate) | DeepSeek-only (indexer), FP8 | **gap**: FP8 weights not admitted (ADR 0003 line 10) | Bought 718.64 vs 682.71 GB/s on the pair. |
| 81 | Persistent layer-resident FP8 projection scheduler (5 CTAs per SM, true dependency barriers only), a campaign | decode | Accepted: `E0150:9-11` "161.792 us", "712.95 GB/s / 84.85%"; unlocked `E0158:43` "83.27%"; composed M=1 `E0159:64` "85.94%". Rejected: 6-CTA exact `E0149:8-9` "678.60 GB/s / 80.77%"; per-shape launches `E0144:86-88` FAIL, `E0159:34-35` "7.5%" / "15.4%"; eight-warp reduction `E0145:68`; two N16 tiles `E0145:69`; scale on B `E0145:70`; `cp.async` `E0145:71`; K128 `E0145:72`; four HMMA chains `E0149:109`; norm placements `E0149:110-112`, `E0150:83-84` | accepted | Generic (FP8) | **gap**: FP8 weights not admitted (ADR 0003 line 10) | Bought 84.85% of the read roofline at M=1. Moxie has no persistent multi-projection kernel. |
| 82 | Warp-vote guarded FP64 replay of rows near a BF16 rounding midpoint | decode | `E0147:15-16` "1.63--1.68% of query rows"; `E0159:101-103` cost "about 1.58 pp"; rejected replay variants `E0149:105-108` | accepted | Generic | **gap** | Moxie's quantized gate is two-clause (ADR 0028), not a byte-exact replay. |
| 83 | Attention preparation on four streams (Q, KV, MLA compressor, indexer compressor) | decode | `E0069:1536` "-5.275423 ms (-13.53%)"; `:1541` decode "-18.906416 ms (-10.71%)" | accepted | Generic | **gap** | Stream primitives exist (`driver.rs:740`). Area: M6.3 overlap and preparation (`06:102`) names the area, not this technique. |
| 84 | Device-resident decode chain: attention output, Q/KV producer, router and FFN boundary on device (0069 checkpoints 9–20) | decode | Accepted: `E0069:746` "201.717 ms \| 170.142 ms"; `:752` syncs "600 \| 471"; `:835-838`; `:1213`; `:1649` "132 \| 89". Gate `E0069:1982` "157.549" ms (rejected against ≤100, kept as baseline). Rejected: fused attention chain `E0053:55` "0.983x"; device `wo_a`→`wo_b` `E0028:99-100` "0.9793x"; output projections on device `E0100:95-98` "no material throughput win" | accepted | Generic | **planned**: M6.1 device-resident layer chains (`06:100`) | Moxie's dense step keeps activations on device (tasks 0076–0083). No MoE model chain exists. |
| 85 | One final wait per token with fixed per-token command ownership | decode | `E0069:1839` "exactly four synchronization calls"; `:1923` "exactly one synchronization"; `:1930` "153.506 ms"; remainder `E0070:270-271` "40.001 ms/token" of final-wait time | accepted structurally | Generic | **planned**: M6.1 (`06:100`) | Deferred completion exists (task 0082: `cuEventSynchronize` 9.7 → 0.93 ms). |
| 86 | Reusable outer decode CUDA graph | decode | `E0070:67-68` "0.388 ms/token (0.25%) slower"; external claim `E0050:85-86` not measured | rejected | Generic | **gap** | Only piecewise capture exists (`crates/moxie-executor/src/dense.rs:379`, task 0085). Attention stays eager, and there is no full-step capture. Area: M6.2 decode graph capture (`06:101`) names the area, not this technique. |
| 87 | CUDA graph with 43 host-function CPU-MoE callbacks (envelope probe) | decode | `E0065:90` envelope medians "2.318336" and "2.182144" ms on the two RTX 3090s (not before/after); `:94` "CPU floor + full envelope \| 90.197334 ms \| 90.061142 ms" | accepted (prerequisite) | Generic | **gap** | Moxie closes a capture segment at every host join (`crates/moxie-executor/src/dense.rs:1133-1134`). M6.2 does not promise host callbacks inside a graph. |
| 88 | Pinned host staging for matmul activations | decode | `E0057:119-120` "8.4 → 4.7 ms/step. On its own this measured flat end to end" | accepted | Generic | **planned**: M6.3 pinned transfers (`03:51`) | |
| 89 | Inline BF16 encode helpers so host attention passes vectorize | prefill | `E0113:58` "27.659 s \| 15.472 s"; `:60` attention total "92.520 s \| 78.139 s"; `c:c7bc93d`; truncate-only proposed `E0113:95-96` | accepted | Generic | **not needed**: attention runs on the device (`paged_attention.cu:85`) | There is no host attention pass to vectorize. |
| 90 | One pool task per row instead of per (row, head) | prefill | `runtime_attention.inc.cpp:1619-1622` "1.86 million tasks … 11.2 s in pool overhead" | accepted | Generic | **not needed**: attention runs on the device (`paged_attention.cu:85`) | There is no host attention task pool. |
| 91 | Host attention heads split across 28 persistent affinity-pinned workers | decode | `E0007:87-88` "3.0227143404 \| 4.3789600899", "1.4486847240x" | accepted, later superseded by CUDA attention | Generic | **not needed**: attention runs on the device (`paged_attention.cu:85`) | There are no host attention workers. CUDA attention superseded this in Strata too. |
| 92 | Persistent, uninitialized page-projection scratch (`make_unique_for_overwrite`) | prefill | `E0111:130-131` "from 8.801146 to 0.000274 s"; `:142-143` "a 3.979124 s or 4.1% reduction"; adverse D2H `:133` | accepted | Generic | **gap** | Device workspaces are admitted once (task 0086), but the host expert path zero-allocates its route, input, slot and partial buffers on every call (`crates/moxie-executor/src/dense.rs:1935-1938`). |
| 93 | Fixed per-command pinned staging slots for queued async H2D | decode | `E0091:164` "43 x 16 KiB = 688 KiB of pinned host memory per device"; `E0069:1805-1806` "5,636,096 pinned bytes"; `rl.md:98-119` | accepted | Generic | **partial**: event-retained leases (`crates/moxie-executor/src/lease.rs:342`, task 0009) | Missing: fixed pinned per-command slots. `PinnedHostBuffer` is not used in the TP or grouped paths. |
| 94 | Widen `<<<1,1>>>` attention-prep norm kernels while keeping the FP64 order | decode | `rl.md:450-454` "`dsv4_query_rank_norm <<<1, 1>>>` 240.6 us … `5.92x`, zero bit mismatches"; `rl.md:404` "111.177092 ms" | accepted | Generic | **gap** | `moxie_dense_grouped_rms_v1` deliberately keeps one thread per row group (`crates/moxie-kernels/cuda/dense_ops.cu:111-113`). |

### 2.5 Prefill specifics

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 95 | Bounded batched prefill pages: multi-row projections and router in one matmul per page | prefill | `E0016:107` "4.232 \| 4.619 \| +9.2%"; `:113` "264,966 \| 4,257 \| -98.4%"; `c:f41126e` "1,501 projection matmul calls … against 122,591" | accepted | Generic | **partial**: row-batched projections per chunk (`crates/moxie-executor/src/dense.rs:540-675`; `selected.rs:170`, task 0087) | Missing: one router matmul per page. `moxie_dense_route_v1` gives each row one thread that recomputes expert logits (`crates/moxie-kernels/cuda/routed_ops.cu:41-50`). |
| 96 | Layer-major prefill tiling (layers outermost over a token tile) | prefill | `E0023:70` "284.16 \| 150.39 \| 1.89x"; `:71` demand H2D "4.17x less"; `runtime_generation.inc.cpp:381-390` "745,172 evictions" before | accepted | Generic | **partial**: each chunk runs layer by layer (task 0087) | Missing: a tile policy sized to amortize expert streaming. The qualified buckets are `[1, 2, 4, 8]` (`crates/moxie-executor/tests/dense_gemma_device.rs:1201`). |
| 97 | Wide prefill pages (8,192 rows) | prefill | `E0119:17-18` "64 \| 58.68 s \| 4.02 tok/s", "8192 \| 23.52 s \| 10.04 tok/s"; `c:04d7932` "a 2.50x"; earlier 512 rejected `E0056:63` "0.96x"; chat wiring `E0192:9-10` | accepted | Generic | **partial**: `prefill_chunks` accepts any bucket list (`crates/moxie-plan/src/selected.rs:170`) | Missing: an executed wide page. Only `[1, 2, 4, 8]` is qualified (task 0087), and `max_input` 16,384 is a shape limit. |
| 98 | Page-major prefill over swappable per-row mHC slots | prefill | `E0095:143` "75.720 s \| 62.886 s \| 1.204x"; `:72` "64 slots are 6.1 MB" | accepted | DeepSeek-only (mHC) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 99 | Prefill routed experts on the GPU once a page has at least N rows | prefill | `E0096:140-143` "89.61 s \| 7.555" → "64.55 s \| 10.488", "1.388x"; `deepseek_runtime.hpp:59-65` | accepted | Generic | **planned**: M6.4 joint prefill/decode placement (`06:103`) | Candidates exist (`expert.rs:84`). The per-phase switch does not. |
| 100 | Expert-major MoE page: each distinct expert read once per page, 8-row tile kernels | prefill | `E0056:97` "1.262"; `:104` "70,483 \| 1,290 \| 54.6x fewer"; `c:ddb7356` "1.262x … MoE 1.889x" | accepted | Generic | **partial**: `GroupedRun` groups a chunk's rows per expert, so each expert is uploaded once (`grouped.rs:1162`) | Missing: the 8-row tile kernel. Each thread reads its own weights (`expert_mlp.cu:182-195`). |
| 101 | Device page-query chain: query projection, RMS norm and RoPE on the GPU, feeding attention (the production result) | prefill | `E0195:16-18` "26.231 prefill tok/s against 20.882 tok/s … 1.256x"; `:21-22` "10.84 GB D2H and 5.42 GB H2D per forward" removed; `:87-88` "VRAM fell by about 226 MB per rank" | accepted | Generic (the q-norm is DeepSeek's) | **present**: device-resident dense step (tasks 0076, 0077, 0082) | Its FP8 query projection is row 79's gap. |
| 102 | Register (pin) the host query-page scratch for D2H | prefill | `E0194:49-50` pageable "5.6062" vs registered "3.5461" GB/s | rejected | Generic | **not needed**: queries stay on the device for attention (`attend_into`, `paged_attention.rs:3070`) | There is no query-page D2H to pin. |
| 103 | Fixed-wave (32 experts) overlap of cold expert upload with page-MoE execution | prefill | `E0193:67-68` "1.269", "0.787"; `:72` demand wait "1.402"; `08:101-105` (R15) | rejected | Generic | **gap** | Area: M6.3 measured transfer overlap (`06:102`) names the area, not this technique. |
| 104 | Parallel host page-MoE row reduction, with the BF16 round fused into the read | prefill | `E0191:59` "0.765"; `:64` demand wait "1.898" | rejected | Generic | **gap** | Moxie's host reduction is single-threaded (`cpu_expert.rs`). |

### 2.6 Decode specifics

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 105 | Host mHC: AVX2 FP64 projection prepacked at admission, blocks tiled 4 at a time, split across host workers | decode | `E0042:144` "1.1134x"; `:146` "saves 21.36 mHC ms/step"; `E0057:89` "27.6 → 18.3 ms/step"; `deepseek_ops.cpp:752-757` "321 us a call and 86 calls". Rejected: exact mHC on host workers `E0007:107-109` "0.9993755894x"; direct row interleave `E0042:36-37` | accepted | DeepSeek-only (mHC) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 106 | Device mHC: fused post/pre/RMSNorm replayed as a graph, persistent across 43 layers with double-buffered residual, widened weighted norm | decode | `E0061:128` "12.895 ms" eager vs "0.956365 ms" graph median; `E0069:47` "adds exactly 135,980,448 bytes"; `rl.md:228-236`; `rl.md:464-485` "Screened at `2.53x`", no end-to-end change claimed. Rejected: serial ascending-double device reductions `E0060:213` "17.12x" | accepted | DeepSeek-only (mHC) | **gap** | M7 may choose it; no roadmap item names it. Area: M7 DeepSeek (`06:120`) names the semantic family, not this algorithm. |
| 107 | Router run on device from the resident output; routing and CPU-MoE input prepared inside the host callback | decode | `E0069:1213` syncs "218 \| 197"; `E0069:1649` "132 \| 89 \| -43 (-32.6%)" | accepted | Generic | **partial**: routing on device (`crates/moxie-kernels/cuda/routed_ops.cu:41`, task 0035) | Missing: in-callback preparation, since there is no stream-ordered host callback (row 31). |
| 108 | DSpark (MTP draft) speculative decoding with exact verification | decode | `E0029:78` "2.566 \| 1.717 \| 0.669x"; `:244-245` "0.785x"; `E0055:10` "DSpark depth 5 is a 0.72x regression on this machine."; `rt.md:96-101` | rejected | DeepSeek-only (DSpark) | **planned**: M9 native MTP head (`06:141`) | |

### 2.7 Memory and admission

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 109 | Explicit per-GPU VRAM byte cap applied before arena allocation | load | `E0082:159` "reduction 2,500,755,456 B/GPU"; `rl.md:363-373` "reserved `23,787,077,632`"; `deepseek_rank_local_topology.hpp:57-65` | accepted | Generic | **present**: `Ledger::admit` (`ledger.rs:355`, task 0006) | |
| 110 | Charge communicator and first-warmup memory into the admission reserve | load | `E0083:129` "communicators initialized \| 391,249,920"; `:130` "first warmup chain complete \| 428,998,656" | accepted | Generic | **gap** | Neither named part exists. `chain.rs:220-223` charges captured-graph pools, a different allocation, and Moxie has no NCCL communicator. |
| 111 | Reserve the shared-expert set in the arena beside a tier | load | `E0178:126` "1.016 GiB per rank"; `c:efd22e3` | accepted | Generic | **planned**: M6.3 hot-expert tiers (`06:102`) | |
| 112 | Cap only the prefill expert cache; the decode window makes zero allocations and zero checkpoint reads | both | `rl.md:327-333`; `rl.md:377-383` | accepted | Generic | **planned**: M6.4 phase transitions (`06:103`) | |
| 113 | Release the centralized spine after prefill | decode | `rl.md:867-870` "returns `9.2 GB`" | proposed | Generic | **planned**: M6.4 "eliminating duplicate resident representations" (`06:103`) | |
| 114 | Compact E4M3 activations plus E8M0 scales instead of an FP32 encoded workspace | prefill | `E0105:122-123` "GPU 1 VRAM … -6,291,456 B" | accepted | Generic (FP8) | **gap**: no W8A8 activations (ADR 0003 line 10) | Bought 6,291,456 B of VRAM on GPU 1. |

### 2.8 Other

| # | Technique | Phase | Strata evidence | Outcome | Scope | Moxie home | Note |
|---|---|---|---|---|---|---|---|
| 115 | SM clock and power operating point: memory-bound kernels are clock-insensitive, ALU-bound ones are not | both | `E0138:62-63` "Removing the clock lock was worth about 20%"; `E0137:146-147` PRMT "−1.0%" under a 250 W cap vs "−8.4%" for the older decoder; production stays locked (`E0138:173-176`) | finding | Generic | **partial**: `docs/evidence/benchmarks/schema.md:57-61` records `clocks_locked`, `power_limit_w` and `thermal_state` | Missing: any characterization or control of clock sensitivity (memory-bound vs ALU-bound). |
| 116 | `CUDA_DEVICE_ORDER=PCI_BUS_ID` pinned in harness and server | other | `c:1da9bab` "13.4 GiB per rank instead of 20.3"; `ds.md:31-55` | accepted | Generic | **partial**: `.cargo/config.toml:16` sets it for Cargo-launched processes | Missing: a server entry point that enforces `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Outside Cargo it must be set by hand (`README.md:187-193`, `AGENTS.md:249`). |
| 117 | Release build for the host MoE executor | decode | `E0078:27` "unoptimized `build-stage5` measured 6.603181 GB/s"; `:32` "routed_gbps=44.769263" | accepted | Generic | **partial**: the timing harness is a release build (`dense-step-timing.md:29`, task 0078) | Missing: a measurement of the host expert path. |

## 3. Summary

**Counts (117 rows, five homes).** Review history is in the task's Result.

| Home | Rows | Of which Strata outcome was accepted or opt-in |
|---|---:|---:|
| present | 11 | 10 |
| partial | 26 | 23 |
| not needed | 7 | 6 |
| planned | 24 | 18 |
| gap | 49 | 38 |

**Present rows (11):** 2, 3, 29, 42, 51, 52, 62, 63, 72, 101, 109.

**Partial rows by number (26):** 1, 8, 9, 13, 16, 23, 26, 28, 31, 37, 39, 40,
44, 67, 68, 71, 75, 93, 95, 96, 97, 100, 107, 115, 116, 117. Each Note names
the missing part. By area:
- **Host expert execution:** 31, 107 (no stream-ordered callback), 37 (no
  tile reuse).
- **Residency:** 1, 8, 9, 23, 26 (DRAM residency not guaranteed; misses
  re-read the checkpoint; expert source not NUMA-bound; no frequency-aware
  placement; no upload copy stream).
- **Expert kernels:** 13, 16, 100 (not one command per GPU; no register feed;
  no row-tile reuse).
- **Attention and KV:** 39, 40, 44.
- **Multi-GPU:** 28, 67, 68, 71, 75.
- **Prefill:** 95, 96, 97.
- **Other:** 93, 115, 116, 117.

**Not-needed rows (7):** 41, 48, 61, 89, 90, 91, 102. In each case Moxie's
device-resident attention, KV or page table removes the cost.

**Gap rows by number (49).**
- **Representation Moxie does not admit** (FP4/FP8 weights, W8A8
  activations, or a KV cache below 16 bits), each citing its rule: 6, 7, 15,
  18, 36, 45, 46, 77, 78, 79, 80, 81, 114.
- **Exact-reduction rule** (ADR 0036): 70.
- **Single-user rule** (`01:12`): 76.
- **Rejected in Strata:** 10, 22, 30, 58, 69, 74, 86, 103, 104.
- **Accepted in Strata with no place in Moxie (25):** 4, 5, 17, 20, 24, 32,
  33, 34, 35, 43, 49, 50, 55, 60, 64, 66, 82, 83, 87, 92, 94, 98, 105, 106,
  110. By area:
  - **Host expert pool:** 32, 33, 34, 35, 92. There is no host worker pool
    (the kernel is single-threaded), no per-rank NUMA split, and buffers are
    allocated per call. Strata's decode rate is set by this host DRAM expert
    read (`ds.md:8-10`, `ds.md:154-155`).
  - **Loading:** 4, 5.
  - **Page bookkeeping:** 24, 43, 49.
  - **Kernels and launch:** 17, 82, 83, 87, 94.
  - **Admission:** 110.
  - **DeepSeek algorithms that M7 may choose but no roadmap item names:** 50,
    55, 60, 64, 66, 98, 105, 106.
  - **Hot-expert tier dispatch ordering:** 20. M6.3 names the tier, not this
    ordering.

**Largest recorded effects (top ten, by the ratio Strata recorded).** Ratios
of call or launch counts, and slowdowns, are excluded.

| Rank | Row | Recorded effect | Scope of the number | Source |
|---:|---:|---|---|---|
| 1 | 60 | "228.7x fewer" physical page bytes, 6.990 GB → 30.564 MB | prefill attention bytes | `E0098:90` |
| 2 | 79 | "21.14x faster" query matmul device time | one prefill term | `E0105:101` |
| 3 | 5 | "about 8.6x" initialization, 191 s → 22.177 s | whole load lineage (rows 3–6) | `E0008:55-56` |
| 4 | 94 | "`5.92x`" | one norm kernel | `rl.md:454` |
| 5 | 96 | "4.17x less" prefill demand H2D bytes (and 1.89x prefill time) | prefill | `E0023:70-71` |
| 6 | 36 | "a 4.1x on the mechanism" | host FP4 expert read vs PCIe path | `E0051:169` |
| 7 | 11 | "3.33x less" demand wait (and "1.626x" decode) | decode | `E0024:82,84` |
| 8 | 55 | "Radix pivot, `3.01x`" | one indexer stage | `rl.md:739` |
| 9 | 33 | "Decode 3.066 -> 8.780 tok/s on the server, 2.86x" | end-to-end decode | `E0123:3` |
| 10 | 97 | "a 2.50x" (4.02 → 10.04 tok/s) | end-to-end served prefill | `c:04d7932`; `E0119:17-18` |

Three figures carry a caveat:
- **Row 60:** the 228.7x byte cut came in an arm that failed its 12 s gate
  (`E0098:92`). It landed later inside the 0107 stack.
- **Row 11:** the 1.626x decode gain was measured before the rank-local
  topology. That topology rejected arena registration at admission
  (`E0188:73`).
- **Row 33:** the 2.86x repairs a pool-width regression; it is not a gain over
  the prior best (`E0123:72-75`).

**Sources whose numbers disagree** (recorded, not resolved):
- **FlashAttention crossover:** `fa.md:92-94` says "below 256 logical KV rows",
  but `deepseek_runtime.hpp:72-75` makes 0 the production crossover.
- **Missing number:** `E0168:18-20` attributes "2.027 against 2.038 ms" to
  experiment 0164, but that number is not in the 0164 file.
- **Pinning:** Laguna's commit `c:4cdffe4` says "pinning is slower",
  contradicting DeepSeek's pinning result (row 11).
