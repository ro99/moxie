# ADR 0007 — Review corrections to measured host capacity and telemetry confinement

- ID / date / author / status: 0007 / 2026-09-09 / implementation agent / proposed with task 0008 review corrections.
- Classification: measured implementation choice (correction of task 0008's accepted contract where review disproved it).
- Scope and owning shared component: `moxie-host` sensor, `moxie-types::HostLimit`, `xtask` telemetry rule.
- Supersedes / superseded by: corrects task 0008's cgroup formula and ADR 0006's confinement claim; supersedes nothing else.

## Problem and mechanism

Task 0008's contract specified, and the implementation was faithful to:

```text
effective_total     = min(MemTotal, limit)
effective_available = min(MemAvailable, limit - current)     (saturating at zero)
```

where `limit` is the smallest `memory.max` over the process's cgroup and every
ancestor. Two review agents reproduced the same counterexample independently:
leaf `session.scope` at 20 GiB limit / 9 GiB current, ancestor `user.slice` at
100 GiB limit / 99 GiB current (90 GiB charged to a sibling scope). A cgroup's
`memory.current` includes its descendants', so this is an ordinary shape on any
machine with more than one scope under a slice.

- Smallest limit = 20 GiB (leaf); headroom off that level = 11 GiB.
- Tightest headroom = 1 GiB (ancestor: 100 - 99).
- The sensor reported 11 GiB available; only 1 GiB was obtainable before the
  kernel refused. The error is in the dangerous direction: the ledger
  over-admits instead of producing the clean refusal document 03 asks for.

The existing suite could not see it: when the ancestor is tighter, min(limit)
and min(headroom) provably coincide, and the leaf-tighter fixture gave its
ancestor 15 GiB of slack. Only a fixture where the leaf has the smaller limit
and the ancestor is under sibling pressure separates the rules.

Two further findings, also reproduced by both reviewers:

- `memory.current` unreadable or malformed defaulted to zero, reporting a
  finite limit as wholly available. Same optimistic direction as the
  `MemAvailable` guess the crate refuses. A `memory.max` that was neither a
  number nor `max` collapsed into "no limit" the same way.
- The telemetry rule matched `starts_with("/proc", "/sys")` only. The sensor
  itself reads `root.join("proc/meminfo")`, `root.join("proc/self/cgroup")`,
  `root.join("sys/fs/cgroup")` — none of which starts with `/`. ADR 0006
  claimed confinement was structural; it was structural against one spelling.

## Options examined

**Cgroup headroom: keep single-binding descriptor.** Rejected. Total and
available can bind at different levels; a single path cannot name both without
lying about one of them.

**Cgroup headroom: minimise both quantities independently (selected).**
Total = `min(MemTotal, min limit)`; available =
`min(MemAvailable, min saturating(limit - current))`, each minimised over all
limited levels. The descriptor records both binding paths. Cost: `HostLimit`
grows three fields; `capacity` reporting and three existing tests name the new
fields. Benefit: the numbers match what the kernel will actually refuse, and
the report can show which level bound which figure.

**Incomplete measurement: keep permissive fallbacks.** Rejected. Treating
missing usage as zero or a malformed limit as `max` turns a partial mount or a
kernel-format surprise into spendable capacity. The crate already refuses a
missing `MemAvailable` for the same reason.

**Incomplete measurement: refuse (selected).** A finite `memory.max` with no
readable numeric `memory.current` is an error; a `memory.max` that is neither
numeric nor `max` is an error. A missing file (`NotFound`) at a level remains
"no limit there", so an unmounted hierarchy and the omitted root files in
fixture trees still yield the machine view. Any other I/O failure refuses.

**Telemetry: prefix match plus relative prefixes.** Rejected in favour of the
stricter version below: prefix matching keeps the `/procfoo` false positive.

**Telemetry: full-segment match (selected).** Split each string literal on `/`
and compare full segments after stripping literal decoration (quotes, `b`
prefix, whitespace, `./` noise). Catches `/proc/meminfo` and `proc/meminfo`,
`sys/fs/cgroup` with or without leading slash, in code position and in macro
arguments; does not catch `artifacts/manifest.toml` or `/procfoo`. Two
rejecting fixtures (relative `proc`, relative `sys`) plus one accepting
artifact-path fixture prove both directions.

## Decision and authority

- `HostLimit::Cgroup` carries both bindings: `{path, limit_bytes,
  current_bytes}` for the smallest limit (binds total) and
  `{avail_path, avail_limit_bytes, avail_current_bytes}` for the smallest
  saturating headroom (binds available). Equal when one level binds both.
- `cgroup_limit` returns `Result`: `Machine` on absent hierarchy (v1, no
  mount, no `0::` line, or `max`/missing at every level); error on incomplete
  measurement as above.
- `read_under` derives total from the limit minimum and available from the
  headroom minimum independently.
- The telemetry rule uses the segment matcher; the one-file vocabulary
  exemption is unchanged.
- No owner gate is reached: read-only introspection, no allocation, no
  checkpoint, no performance claim. Task 0008's stop condition stands.

## Evidence and acceptance

- `moxie-host` 14 tests: the 11 committed plus `headroom_is_minimised…`
  (20 GiB total, 1 GiB available, divergent binding paths asserted),
  `a_level_with_a_limit_but_no_readable_usage_is_refused`, and
  `a_malformed_limit_is_refused_rather_than_treated_as_max`. New fixture
  `cgroup-sibling-pressure` preserves the counterexample; `cgroup-bad-limit`
  proves malformed limits refuse.
- `cargo xtask arch-check`: 51 rejected + 14 accepted, 12 rules. New
  `storage-reads-relative-telemetry`, `storage-reads-relative-sys-telemetry`
  (both rejected) and `storage-reads-artifact-path` (accepted).
- Existing ancestor/leaf fixtures extended to assert the avail binding equals
  the limit binding where they coincide.
- Full lanes: fmt, both clippy lanes, workspace host suite, spec-check,
  no-driver lane, device lane, test-gpu, capacity — reported in task 0008's
  correction section, failed/skipped/unmeasured separately.

## Enforcement and removal

- The three host tests and three arch-check fixtures are the enforcement.
  Nothing temporary is introduced and nothing is deleted beyond the corrected
  formula: the old single-binding derivation is gone, with its counterexample
  preserved as `cgroup-sibling-pressure`.
- Re-evaluate if cgroup v1 limits ever need enforcement (today v1 yields the
  machine view) or if a new telemetry spelling appears that is not a
  `/`-separated `proc`/`sys` segment.
