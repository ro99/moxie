# Task 0073 — topology cost probes

Status: **accepted** (coordinator, 2026-09-24, under the owner's auto-mode
delegation), after sol's review rounds R1 and R2. The coordinator verified
the R2 fix directly. Builder Codex `luna`; reviewer Codex `sol`.
- Change 3 was amended after the builder's DECISION: the TOML is written
  and read through `toml::Table`, with no serde (`arch-check` forbids
  `xtask -> serde`).
- R1 (HIGH): the isolated peer samples were taken inside the grant loop,
  which confounded them with setup order. Fixed: all grants, then all
  warm-ups, then the isolated samples, then the concurrent ones.
- R2 (MEDIUM): a typed setup error could be masked by a peer's generic
  cancel error. Fixed with a first-error slot.
- Forward 3090 peer: about 6.58 GB/s isolated and about 2.92 GB/s with both
  directions running, stable over three runs. That is consistent with
  bidirectional contention; the one-sided magnitude is unexplained and
  recorded as such.
- **Size, carried to the milestone-end ponytail audit:**
  `topology_probe.rs` is 406 lines, against an estimate of 200–260.

## Identity and authority

- Task0073, M5 plan slice 6, first task. Slice 6 is:
  - **0073:** this probe;
  - **0074:** the deterministic plan comparison tool;
  - the milestone-end `/ponytail:ponytail-audit`;
  - the acceptance package.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Clauses served:**
  - M5.5: "Build **topology cost probes** …";
  - document 03: "Discover peer access … and measured simultaneous-transfer
    behavior. Recheck links under load";
  - document 03's cost rule, `required_transfer_time >=
    bytes_over_that_link / measured_sustained_bandwidth`;
  - document 08, R12: "New planner costs must come from current topology and
    simultaneous traffic probes".

## Facts established before writing (coordinator, 2026-09-24)

- `cargo xtask-cuda probe` (`xtask/src/probe.rs`, task M0.2) records device
  identity, the `cuDeviceCanAccessPeer` matrix, and a **pageable, one device
  at a time** host-to-device rate. It says itself that it "is not that
  model". Only the 3090 pair has peer access
  (`docs/evidence/topology-p2p.md`).
- **Strata has no bandwidth probe.** `src/engine/placement.cpp` records only
  a boolean `high_speed_peer` matrix plus byte counts. The measured model is
  new work.
- The links M5 plans use are:
  - host-to-device and device-to-host, as synchronous pageable copies (the
    pipeline handoff and host-expert staging);
  - peer copies within the pair (the TP collectives);
  - on-device weight reads (decode is weight-traffic bound).

  **Pinned memory is out of scope.** No M5 path uses it, `moxie-cuda` has
  no pinned allocation, and document 03 calls it "an option, not a
  universal speed claim". Measure it when a pinned path exists (M6.3).
- The building blocks exist in `moxie-cuda`:
  - `RankContext::acquire` and `measure()`;
  - `DeviceBuffer::alloc`, `copy_from_host` and `copy_to_host`;
  - the unsafe `copy_from_peer_async_at`, which covers a same-device source
    and a granted peer, and refuses an ungranted one rather than staging it;
  - `enable_peer_access`, `Stream::new` / `synchronize`, and `Event::new` /
    `record` / `elapsed_ms`.
- `xtask`'s `cuda` feature already depends on `moxie-executor` and
  `moxie-plan`. `toml`'s `serde_derive` is already in the lock graph.

## Bounded deliverable

- A pure `TopologyCosts` record in `moxie-plan`: per-device memory
  bandwidth and usable bytes; per-link small-copy latency, bulk bandwidth and
  **bandwidth under simultaneous traffic**.
- A probe in `moxie-executor` that measures it on the present GPUs.
- `cargo xtask-cuda probe --costs <path>` writes it as TOML.
- A measured evidence file.
- **Non-goals:**
  - pinned or mapped memory;
  - NCCL;
  - plan ranking (task 0074);
  - automatic plan selection (M6.5);
  - any change to execution paths.

## Numbered changes

1. **New `crates/moxie-plan/src/costs.rs`** (pure; declared and re-exported
   in `moxie-plan`'s `lib.rs`):
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
   pub enum Endpoint { Host, Device(DeviceUuid) }
   #[derive(Debug, Clone, PartialEq)]
   pub struct LinkCost {
       pub from: Endpoint, pub to: Endpoint,
       /// Median wall time of one small copy, in microseconds.
       pub latency_us: f64,
       /// Median sustained rate of one bulk copy with the link otherwise idle, GB/s.
       pub bandwidth_gbps: f64,
       /// Median rate of the same bulk copy while every link in its class runs at once, GB/s.
       pub concurrent_gbps: f64,
   }
   #[derive(Debug, Clone, PartialEq)]
   pub struct DeviceCost { pub device: DeviceUuid, pub memory_gbps: f64, pub usable_bytes: u64 }
   #[derive(Debug, Clone, PartialEq, Default)]
   pub struct TopologyCosts { pub devices: Vec<DeviceCost>, pub links: Vec<LinkCost> }
   impl TopologyCosts {
       pub fn link(&self, from: Endpoint, to: Endpoint) -> Option<&LinkCost>;
       pub fn device(&self, device: DeviceUuid) -> Option<&DeviceCost>;
   }
   ```
   - A **missing link means no direct path**. There is never a host
     fallback entry.
   - Size: about 50 lines.

2. **New `crates/moxie-executor/src/topology_probe.rs`**, behind the
   `driver` feature, declared and re-exported in `lib.rs`:
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub struct ProbeConfig { pub small_bytes: usize, pub bulk_bytes: usize, pub reps: usize }
   // Default: 16 KiB, 64 MiB, 5.
   pub fn probe_topology(ordinals: &[u32], config: ProbeConfig) -> Result<TopologyCosts>;
   ```
   - **Setup:** acquire one `RankContext` per ordinal on the calling thread
     (a bounded probe, not an execution rank), and allocate one bulk device
     buffer each. Warm up every measured copy once. Take the **median** over
     `reps` for every figure.
   - **Per device:**
     - `Host → Device(d)` and `Device(d) → Host` use the synchronous
       pageable copies, timed with `Instant`. `latency_us` comes from
       `small_bytes`, and `bandwidth_gbps` from `bulk_bytes`.
     - `memory_gbps` is a same-device `copy_from_peer_async_at` (source and
       destination on `d`) of `bulk_bytes`, timed with events, counting
       2 × bytes (a read and a write).
     - `usable_bytes` is `measure()`'s free bytes.
   - **Peer:** for each ordered pair `(a, b)` whose capability grants peer
     access (`can_access_peer`), `enable_peer_access` both ways, then time
     `Device(a) → Device(b)` with `copy_from_peer_async_at` on `b`'s stream
     and events, for small and bulk. Pairs without access get **no** link.
   - **Simultaneous traffic** (`concurrent_gbps`), one median per link:
     - host-to-device: every device's bulk copy starts together, with one
       scoped thread per device, each acquiring its own context, released by
       a `Barrier`;
     - device-to-host: the same;
     - peer: both directions of every peer pair enqueued together, each on
       its destination stream, then both synchronized.
   - **Every figure must be finite and positive**, or the probe returns
     `InvalidRequest`.
   - Links are sorted by `(from, to)`, and devices by ordinal order.
   - Mark with `ponytail:` that pinned and mapped transfers are unmeasured
     until a pinned path exists (M6.3).
   - Size: about 200–260 lines.

3. **`xtask/src/probe.rs`** and **`xtask/src/main.rs`:**
   - Add an optional `--costs <path>`. When it is given, call
     `probe_topology` on every device, append a "Topology costs" markdown
     table to the existing report, and write the TOML file.
   - The TOML has one `[[device]]` per device (`uuid`, `memory_gbps`,
     `usable_bytes`) and one `[[link]]` per link (`from`, `to` as `"host"` or
     a UUID string, and the three figures).
     - **Amended 2026-09-24, after the builder's DECISION:** no `serde`
       dependency. `arch-check` forbids `xtask -> serde`, and the checker's
       policy is not changed for this.
     - Build a `toml::Table` by hand and write it with `toml::to_string`.
       Read it back with `str::parse::<toml::Table>()` and typed field
       access. A missing or mistyped field is an `InvalidArtifact` or
       `InvalidRequest` naming the field.
     - `xtask/Cargo.toml` and `Cargo.lock` stay unchanged.
   - Add `pub fn read_costs(path) -> Result<TopologyCosts>` next to the
     writer, for task 0074, with a host unit test that a written file reads
     back equal.
   - Replace the closing "this probe does neither and is not that model"
     note with one line naming the `--costs` model.

4. **New `docs/evidence/topology-costs.md`:**
   - the measured tables from one run on this machine (GPUs by UUID);
   - the probe configuration, date and driver version;
   - the TOML file inline;
   - three short observations, each a number from the table:
     - peer bulk rate against host-staged rate (both copies);
     - concurrent against isolated host-to-device;
     - small-copy latency against a decode handoff's size.
   - It supersedes the "inherited, not re-measured" performance note in
     `topology-p2p.md`. Add one line there pointing here, and change nothing
     else in that file.

5. **Test, `crates/moxie-executor/tests/topology_probe_device.rs`**
   (`driver` feature): one test, `probe_measures_every_direct_link_only`.
   - Run `probe_topology` over all present devices with a small config
     (1 MiB bulk, 4 KiB small, 3 reps).
   - Assert:
     - exactly one `DeviceCost` per device;
     - `Host → d` and `d → Host` for every device;
     - a `Device(a) → Device(b)` link **if and only if**
       `can_access_peer` grants it;
     - every figure finite and positive.
   - Print the table.

## Allowed files

- `crates/moxie-plan/src/costs.rs` (new) and `crates/moxie-plan/src/lib.rs`
  (the declaration and re-export)
- `crates/moxie-executor/src/topology_probe.rs` (new) and
  `crates/moxie-executor/src/lib.rs` (the declaration and re-export)
- `crates/moxie-executor/tests/topology_probe_device.rs` (new), plus its
  `[[test]]` entry in `crates/moxie-executor/Cargo.toml` if the crate lists
  its tests
- `xtask/src/probe.rs`, `xtask/src/main.rs`
- `docs/evidence/topology-costs.md` (new), and the one pointer line in
  `docs/evidence/topology-p2p.md`
- This task's Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- `cargo test -p moxie-executor --features driver --test
  topology_probe_device`.
- `cargo xtask-cuda probe --costs <scratch path>`, whose output goes into
  the evidence file.
- `cargo xtask-cuda test-gpu` (no regression).

**Mutations**, each applied, run, shown failing and restored:
1. Record a peer link for every device pair regardless of
   `can_access_peer`. The test's if-and-only-if check fails, or the probe
   refuses the copy.
2. Measure `concurrent_gbps` sequentially (no barrier). Report the figures
   it gives against the barrier run. This mutation is a **measurement
   check**, not a test failure: the Result must show the concurrent host
   figures differ from the isolated ones on this machine, or say that they
   do not.

**Stop conditions:**
- A copy between devices without peer access succeeds (a hidden host
  staging path).
- Any probe step needs pinned memory or a new unsafe FFI entry.
- The TOML support needs any new dependency.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Implemented `TopologyCosts`, the driver-gated probe, the `probe --costs`
  writer/reader and one device test. TOML is built with `toml::Table` and
  `toml::to_string`; the reader parses a table and reports missing or mistyped
  fields by name. No serde dependency was added; `xtask/Cargo.toml` and
  `Cargo.lock` are unchanged. The round-trip test passes.

### R1 fixes

- Built one list of permitted ordered peer pairs. The probe enables every
  grant, warms every pair in both sizes, takes all isolated samples, then
  takes concurrent samples from that same list. The worker setup now uses one
  `Result` closure and one failure store before the pre-sample barrier;
  host and event measurements share the checked median conversion helper.
- Replaced the evidence tables and TOML with the primary R1 run in
  [topology-costs.md](../evidence/topology-costs.md). It reports 5060 Ti
  concurrent H2D of 5.053 GB/s versus 6.586 isolated, host-to-5060 latency of
  9.025 us for 16 KiB, and both direct/staged 3090 peer directions using the
  newly measured host legs. The reverse staged rate derives to 2.82 GB/s from
  this run's D2H/H2D values (5.566/5.734 GB/s); the new table does not support
  the earlier approximate 4.03 GB/s figure.
- Stability check: two more probe runs gave forward peer isolated/concurrent
  bulk of 6.583886/2.924808 and 6.585354/2.923887 GB/s; the primary was
  6.583886/2.924547. Each ratio is about 2.25×. As required, the evidence
  labels this difference unexplained and draws no performance conclusion.
- R1 gates passed: fmt, both clippy gates, `cargo test --workspace --locked`,
  `cargo xtask arch-check` (79 rejected, 21 accepted fixtures; 13 rules),
  `cargo xtask spec-check`, and the focused `topology_probe_device` test with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The primary probe and both stability probes
  completed. The round-trip test passed. The full `cargo xtask-cuda test-gpu`
  passed 63/63 on SM86 and SM120 in the original task round; it was skipped
  for R1 as directed.
- Original task mutations were applied, checked and restored: removing the
  peer-access filter made the device test fail with typed `Unsupported`
  (`3032cfa3` cannot access `97fe4889`); sequential concurrent-H2D values
  versus barrier values (GB/s) were 6.521/4.003 for `97fe`, 5.655/3.009 for
  `3032`, and 6.528/3.757 for `81fe`. The current evidence run also has
  concurrent H2D below isolated on the 5060 Ti. No ungranted peer copy
  succeeded.
- No pinned/mapped path or new unsafe FFI was needed. The revised
  `topology_probe.rs` is 406 lines.

### R2 fixes

- Concurrent host setup stores the first typed setup error before the barrier;
  peers canceled by it return a placeholder. All workers are joined, and the
  original setup error takes precedence over a join error. Reworded the
  forward-peer observation as consistent with bidirectional contention, with
  the one-sided magnitude unexplained. No probe rerun. No new test was added:
  injecting a per-worker setup failure would require new fault-injection
  support, while the stored-error and post-join ordering enforce precedence.
- R2 gates passed: fmt, workspace clippy, driver-feature clippy,
  `cargo test --workspace --locked`, and the PCI-ordered
  `topology_probe_device` test.
