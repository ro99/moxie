//! `capacity` — measure every visible device and admit a plan against it.
//!
//! This is the composition root for task 0007. Document 02 permits
//! `memory` -> `cuda` and forbids the reverse, so `moxie-cuda` may not name a
//! `CapacitySnapshot`: it takes the reading, `moxie-memory` turns it into a
//! budget, and the wiring happens here, where a composition root is allowed to
//! know both.
//!
//! What it proves is the loop, not a capability. Nothing is allocated, and the
//! numbers it prints are a reading of this machine at one instant.

use moxie_cuda::{RankContext, device_count};
use moxie_memory::{
    BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Scaling, StageSpan, TierReport,
};

const SPILL: Tier = Tier::Host(HostTier::StateSpill);
use moxie_types::{DeviceTier, HostLimit, HostTier, MeasuredDevice, RankId, Scope, Tier};

/// Left unspent on every device, on top of what is already gone.
///
/// A deliberately blunt placeholder, and named so it cannot be mistaken for a
/// derived reserve: document 03 requires the old 48 MiB and two-largest-linears
/// constants to become *derived* reservations, and this is neither. It exists so
/// the command demonstrates a non-zero engine reserve reaching the ledger. The
/// real value comes from a plan's declared buffers (task 0006's `ReserveRule`).
const ENGINE_RESERVE_BYTES: u64 = 64 * 1024 * 1024;

/// Left unspent on the host, on top of what the operating system already holds.
///
/// A placeholder like the device one, and larger for a reason worth stating:
/// document 03 warns against treating all 251 GB as an expert cache, and an
/// interactive machine needs room for the rest of what the person is doing. It
/// is still a constant and still not a derived reservation.
const HOST_RESERVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub fn run() -> i32 {
    let count = match device_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("FAIL  cannot enumerate devices: {e}");
            return 1;
        }
    };
    if count == 0 {
        eprintln!("FAIL  no CUDA device is visible; nothing was measured");
        return 1;
    }

    println!("== measured capacity: this host and every visible device ==");
    println!(
        "reserve left unspent per device: {} MiB\n",
        ENGINE_RESERVE_BYTES / (1024 * 1024)
    );

    // One rank per device, each holding its own context for the measurement.
    let mut contexts = Vec::new();
    for ordinal in 0..count {
        match RankContext::acquire(RankId(ordinal), ordinal) {
            Ok(ctx) => contexts.push(ctx),
            Err(e) => {
                eprintln!("FAIL  rank {ordinal} cannot acquire device {ordinal}: {e}");
                return 1;
            }
        }
    }

    let mut measurements: Vec<MeasuredDevice> = Vec::new();
    for ctx in &contexts {
        match ctx.measure() {
            Ok(m) => measurements.push(m),
            Err(e) => {
                eprintln!("FAIL  rank {} cannot measure: {e}", ctx.rank().get());
                return 1;
            }
        }
    }

    // The host is measured through the sensor that owns telemetry (ADR 0006);
    // this command only joins the reading to the ledger's rule.
    let host = match moxie_host::read() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("FAIL  cannot measure host memory: {e}");
            return 1;
        }
    };
    println!("host:");
    println!(
        "  total {} MiB, available {} MiB",
        host.total_bytes / (1024 * 1024),
        host.available_bytes / (1024 * 1024),
    );
    println!(
        "  available is MemAvailable, not MemFree: free is only {} MiB, and {} MiB of cache is \
         reclaimable",
        host.free_bytes / (1024 * 1024),
        (host.cached_bytes + host.buffers_bytes) / (1024 * 1024),
    );
    println!(
        "  swap {} MiB, reported and in no budget",
        host.swap_total_bytes / (1024 * 1024),
    );
    match &host.limit {
        HostLimit::Machine => println!("  no cgroup limit applies; the machine's figures govern"),
        HostLimit::Cgroup {
            path,
            limit_bytes,
            current_bytes,
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
        } => {
            println!(
                "  cgroup {path} sets the smallest limit: {} MiB total, {} MiB of it in use",
                limit_bytes / (1024 * 1024),
                current_bytes / (1024 * 1024),
            );
            let headroom = avail_limit_bytes.saturating_sub(*avail_current_bytes) / (1024 * 1024);
            if avail_path == path {
                println!("  it also binds headroom: {headroom} MiB for new allocation");
            } else {
                println!(
                    "  headroom binds at {avail_path}: {} MiB limit, {} MiB in use, {headroom} MiB for new allocation",
                    avail_limit_bytes / (1024 * 1024),
                    avail_current_bytes / (1024 * 1024),
                );
            }
            println!(
                "  the machine itself has {} MiB total and {} MiB available",
                host.machine_total_bytes / (1024 * 1024),
                host.machine_available_bytes / (1024 * 1024),
            );
        }
    }

    let mut snapshots = Vec::new();
    match CapacitySnapshot::measured_host(&host, HOST_RESERVE_BYTES) {
        Ok(s) => {
            println!(
                "  admissible {} MiB after an {} MiB reserve\n",
                s.admissible_bytes() / (1024 * 1024),
                HOST_RESERVE_BYTES / (1024 * 1024)
            );
            snapshots.push(s);
        }
        Err(e) => {
            eprintln!("FAIL  the host produced no admissible budget: {e}");
            return 1;
        }
    }
    for m in &measurements {
        println!(
            "rank {} -> {} {} ({})\n  bus {}, {} SMs, ordinal {} (diagnostic only)\n  \
             total {} MiB, free {} MiB",
            contexts
                .iter()
                .find(|c| c.uuid() == m.uuid)
                .map(|c| c.rank().get())
                .unwrap_or(u32::MAX),
            m.name,
            m.sm,
            m.uuid,
            m.pci_bus_id,
            m.multiprocessor_count,
            m.ordinal_label,
            m.total_bytes / (1024 * 1024),
            m.free_bytes / (1024 * 1024)
        );
        match CapacitySnapshot::measured(m, ENGINE_RESERVE_BYTES) {
            Ok(s) => {
                println!(
                    "  admissible {} MiB\n",
                    s.admissible_bytes() / (1024 * 1024)
                );
                snapshots.push(s);
            }
            Err(e) => {
                eprintln!("FAIL  {} produced no admissible budget: {e}", m.uuid);
                return 1;
            }
        }
    }

    let mut ledger = match Ledger::new(snapshots) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("FAIL  cannot build a ledger from the measurements: {e}");
            return 1;
        }
    };

    // A small declared plan, sized from the smallest device so it fits every
    // one of them. Nothing is allocated; this only exercises admission.
    let smallest = measurements
        .iter()
        .map(|m| m.free_bytes.saturating_sub(ENGINE_RESERVE_BYTES))
        .min()
        .unwrap_or(0);
    let weights = smallest / 4;
    let kv = smallest / 8;
    // Host-side work sized from the host's own budget, not the device's.
    let host_admissible = ledger
        .snapshot(Scope::Host)
        .map_or(0, |s| s.admissible_bytes());
    let spill = host_admissible / 16;
    let workspace = host_admissible / 32;

    let mut failed = 0;
    for m in &measurements {
        let scope = Scope::Device(m.uuid);
        let mut plan = PlanRequest::new("capacity-probe", ["prefill", "barrier", "decode"])
            .expect("three distinct stages");
        plan.buffer(BufferRequest::new(
            "weights",
            scope,
            Tier::Device(DeviceTier::PackedResidentWeights),
            weights,
            StageSpan::inclusive(0, 2),
        ))
        .and_then(|p| {
            p.buffer(
                BufferRequest::new(
                    "kv",
                    scope,
                    Tier::Device(DeviceTier::KvStatePages),
                    kv,
                    StageSpan::inclusive(0, 2),
                )
                .scaling(Scaling::Context),
            )
        })
        // The plan spans both tiers, which is the point of measuring both: state
        // spilled to the host and a CPU-side workspace are charged to the host
        // scope in the same envelope as the device buffers.
        .and_then(|p| {
            p.buffer(
                BufferRequest::new(
                    "spill",
                    Scope::Host,
                    SPILL,
                    spill,
                    StageSpan::inclusive(1, 2),
                )
                .scaling(Scaling::Context),
            )
        })
        .and_then(|p| {
            p.buffer(BufferRequest::new(
                "cpu-workspace",
                Scope::Host,
                Tier::Host(HostTier::CpuWorkspace),
                workspace,
                StageSpan::inclusive(0, 2),
            ))
        })
        .expect("declared buffers are well formed");

        // The breakdown is taken *before* admitting. Previewing the same plan
        // afterwards counts it twice -- once as committed, once as this
        // request's peak -- and prints a headroom figure that is arithmetically
        // correct and completely misleading.
        let before = ledger.preview(&plan).expect("the plan is well formed");
        match ledger.admit(&plan) {
            Ok(reservation) => {
                println!(
                    "PASS  {} admits {} MiB weights + {} MiB KV",
                    m.uuid,
                    weights / (1024 * 1024),
                    kv / (1024 * 1024)
                );
                if let Some(s) = before.scope(scope) {
                    println!(
                        "      peak {} MiB of {} MiB admissible at stage {:?}; {} MiB would remain",
                        s.request_peak_bytes / (1024 * 1024),
                        s.admissible_bytes / (1024 * 1024),
                        before.stages[s.peak_stage as usize],
                        s.remaining_headroom_bytes / (1024 * 1024)
                    );
                    for t in s
                        .tiers
                        .iter()
                        .filter(|t: &&TierReport| t.request_peak_bytes > 0)
                    {
                        println!(
                            "      {:<32} {} MiB",
                            t.tier.name(),
                            t.request_peak_bytes / (1024 * 1024)
                        );
                    }
                }
                println!(
                    "      ledger now holds {} MiB on this device and {} MiB on the host",
                    ledger.scope_committed(scope) / (1024 * 1024),
                    ledger.scope_committed(Scope::Host) / (1024 * 1024),
                );
                if let Err(e) = ledger.release(reservation) {
                    eprintln!("FAIL  release refused: {}", e.error);
                    failed += 1;
                }
            }
            Err(e) => {
                eprintln!(
                    "FAIL  {} refused a plan sized from the smallest device:\n{e}",
                    m.uuid
                );
                failed += 1;
            }
        }
    }

    // Every byte given back, on both tiers. A leak here is the R08 shape, and
    // the ledger can see it even though nothing was allocated.
    let scopes: Vec<Scope> = std::iter::once(Scope::Host)
        .chain(measurements.iter().map(|m| Scope::Device(m.uuid)))
        .collect();
    for scope in scopes {
        let held = ledger.scope_committed(scope);
        if held != 0 {
            eprintln!("FAIL  {scope} still holds {held} B after release");
            failed += 1;
        }
    }
    if !ledger.outstanding().is_empty() {
        eprintln!("FAIL  reservations outstanding: {:?}", ledger.outstanding());
        failed += 1;
    }

    if failed > 0 {
        eprintln!("\ncapacity FAILED: {failed} check(s)");
        return 1;
    }
    println!(
        "\ncapacity passed: the host and {} device(s) measured and admitted against. \
         Nothing was allocated; these are readings, not reservations.",
        measurements.len()
    );
    0
}
