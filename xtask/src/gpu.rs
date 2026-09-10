//! The real-hardware lane: `cargo xtask-cuda test-gpu [--profile sm_NN]`.
//!
//! Document 07: "unsupported hardware is skipped/unmeasured, never passed" and
//! "a test that catches a device error then exits successfully is not a passing
//! production test". So this lane reports pass/fail/skip per device explicitly,
//! returns non-zero when any *attempted* case failed, **and** returns non-zero
//! when a required architecture produced no passing case at all.
//!
//! That last rule is the one the M0 review found missing: the previous version
//! printed a NOTE for a missing architecture and exited 0, so a CI wrapper could
//! read an all-skipped run as qualification.

use core::ffi::c_void;
use std::ffi::CString;

use moxie_cuda::{
    DeviceBuffer, Event, Module, ModuleImage, PtxSource, RankContext, Stream, TrustedImage,
    query_device,
};
use moxie_executor::{
    DeviceArena, Lease, OwnedBinding, SelectedAdmitRefused, SelectedReservedPlan, Turn, Upload,
};
use moxie_graph::{
    Bindings, Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry,
    TensorSpec, ValueId, ValueRole,
};
use moxie_interp::{HostTensor, Interpreter, Value};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_plan::{Phase, ResourceWorkload, lower_selected};
use moxie_types::{
    ActivationPrecision, DeviceCapability, DeviceTier, Dim, Error, HostTier, Precision, RankId,
    Scope, SymbolId, TensorLayout, Tier, WeightPrecision,
};

/// Wrap the build's own fatbin as a trusted image.
///
/// This is where the trust assertion belongs: `xtask` is the composition root,
/// and it is the thing that knows these bytes are `include_bytes!` of this
/// build's nvcc output rather than a file someone handed us.
fn smoke_image(bytes: &'static [u8]) -> Result<TrustedImage<'static>, Error> {
    // SAFETY: `bytes` is one of `moxie_kernels`' `include_bytes!` constants --
    // a complete fatbin emitted by the pinned nvcc during this build, embedded
    // in the executable and immutable for its lifetime. It is not read from
    // disk, not truncated, and not attacker-influenced.
    unsafe { TrustedImage::from_build_output(bytes) }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    Failed(String),
    Skipped(String),
}

impl Outcome {
    fn label(&self) -> &'static str {
        match self {
            Outcome::Passed => "PASS",
            Outcome::Failed(_) => "FAIL",
            Outcome::Skipped(_) => "SKIP",
        }
    }
}

#[derive(Debug)]
pub struct CaseResult {
    pub device: u32,
    pub sm: String,
    pub case: &'static str,
    pub outcome: Outcome,
}

const CASES: &[&str] = &[
    "axpy_f32",
    "bf16_round_trip",
    "arch_mismatch_is_typed",
    "stream_event_completion",
    "event_backed_lease",
    "admitted_device_arena",
    "selected_bf16_device_chain",
    "bf16_semantic_numerics",
    "lease_rejects_foreign_completion",
    "non_ptx_text_rejected",
    "rank_context_is_exclusive",
    "concurrent_handoff_is_exclusive",
    "measurement_is_live",
];

/// Run every GPU case on every visible device.
///
/// `profile` restricts the *required* architectures to one `sm_NN`, which is the
/// `--profile sm86` / `sm120` split in document 07's command table. It does not
/// stop other devices from being exercised.
pub fn run(profile: Option<&str>) -> i32 {
    let required: Vec<String> = match profile {
        Some(p) => {
            let want = normalise_profile(p);
            if !moxie_kernels::compiled_sm().contains(&want) {
                eprintln!(
                    "profile {want} is not compiled into this build ({}); nothing to qualify",
                    moxie_kernels::KERNEL_ARCHS
                );
                return 2;
            }
            vec![want]
        }
        None => moxie_kernels::compiled_sm(),
    };

    let count = match moxie_cuda::device_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("cannot enumerate devices: {e}");
            return 2;
        }
    };
    if count == 0 {
        eprintln!("no CUDA devices visible: nothing measured, nothing passed");
        return 2;
    }

    println!("kernel archs compiled: {}", moxie_kernels::KERNEL_ARCHS);
    println!("nvcc: {}", moxie_kernels::NVCC_VERSION);
    println!("host compiler: {}", moxie_kernels::HOST_COMPILER_VERSION);
    println!(
        "image sha256: smoke={} smoke_sm86={}",
        moxie_kernels::SMOKE_FATBIN_SHA256,
        moxie_kernels::SMOKE_FATBIN_SM86_SHA256
    );
    println!("required architectures: {}", required.join(", "));

    let mut results = Vec::new();
    let mut seen_sm = Vec::new();

    for ordinal in 0..count {
        let cap = match query_device(ordinal) {
            Ok(c) => c,
            Err(e) => {
                results.push(CaseResult {
                    device: ordinal,
                    sm: "?".into(),
                    case: "query",
                    outcome: Outcome::Failed(e.to_string()),
                });
                continue;
            }
        };
        println!(
            "\ndevice {ordinal}: {} {} ({}) {} MiB, {} SMs, bus {}",
            cap.name,
            cap.sm(),
            cap.uuid,
            cap.total_memory_bytes / (1024 * 1024),
            cap.multiprocessor_count,
            cap.pci_bus_id
        );
        if !seen_sm.contains(&cap.sm()) {
            seen_sm.push(cap.sm());
        }

        // Document 07: unsupported hardware is skipped and reported as
        // unmeasured -- never silently counted as a pass.
        if !moxie_kernels::compiled_sm().contains(&cap.sm()) {
            let why = format!(
                "{} is not among the compiled architectures ({}); UNMEASURED",
                cap.sm(),
                moxie_kernels::KERNEL_ARCHS
            );
            for name in CASES {
                results.push(CaseResult {
                    device: ordinal,
                    sm: cap.sm(),
                    case: name,
                    outcome: Outcome::Skipped(why.clone()),
                });
            }
            continue;
        }

        results.push(case(&cap, "axpy_f32", axpy(&cap)));
        results.push(case(&cap, "bf16_round_trip", bf16(&cap)));
        results.push(case(&cap, "arch_mismatch_is_typed", arch_mismatch(&cap)));
        results.push(case(&cap, "stream_event_completion", stream_event(&cap)));
        results.push(case(&cap, "event_backed_lease", backed_lease(&cap)));
        results.push(case(
            &cap,
            "admitted_device_arena",
            admitted_device_arena(&cap),
        ));
        results.push(case(
            &cap,
            "selected_bf16_device_chain",
            selected_bf16_device_chain(&cap),
        ));
        results.push(case(
            &cap,
            "bf16_semantic_numerics",
            bf16_semantic_numerics(&cap),
        ));
        results.push(case(
            &cap,
            "lease_rejects_foreign_completion",
            lease_quarantine(&cap),
        ));
        results.push(case(&cap, "non_ptx_text_rejected", ptx_rejection(&cap)));
        results.push(case(
            &cap,
            "rank_context_is_exclusive",
            rank_exclusivity(&cap),
        ));
        results.push(case(
            &cap,
            "concurrent_handoff_is_exclusive",
            concurrent_handoff(&cap),
        ));
        results.push(case(&cap, "measurement_is_live", measurement_is_live(&cap)));
    }

    println!("\n--- results ---");
    let (mut failed, mut skipped, mut passed) = (0usize, 0usize, 0usize);
    for r in &results {
        let detail = match &r.outcome {
            Outcome::Passed => String::new(),
            Outcome::Failed(m) | Outcome::Skipped(m) => format!("  {m}"),
        };
        println!(
            "{:<5} device {} {:<7} {}{}",
            r.outcome.label(),
            r.device,
            r.sm,
            r.case,
            detail
        );
        match r.outcome {
            Outcome::Passed => passed += 1,
            Outcome::Failed(_) => failed += 1,
            Outcome::Skipped(_) => skipped += 1,
        }
    }
    println!("\n{passed} passed, {failed} failed, {skipped} skipped/unmeasured");

    // Acceptance gate: every required architecture must have run *every* case to
    // a pass on at least one real device of that architecture. Absent hardware,
    // an all-skipped device and a partially-run device are all failures of the
    // gate, not silent successes.
    println!("\narchitectures exercised: {}", seen_sm.join(", "));
    let mut unqualified = Vec::new();
    for arch in &required {
        let qualified = CASES.iter().all(|case_name| {
            results.iter().any(|r| {
                r.sm == *arch && r.case == *case_name && matches!(r.outcome, Outcome::Passed)
            })
        });
        if qualified {
            println!("QUALIFIED   {arch}: every case passed on a real device");
        } else {
            let why = if seen_sm.contains(arch) {
                "present, but not every case passed"
            } else {
                "no installed device has this architecture"
            };
            println!("UNQUALIFIED {arch}: {why}");
            unqualified.push(arch.clone());
        }
    }

    if failed > 0 || !unqualified.is_empty() {
        println!(
            "\ntest-gpu FAILED: {failed} case(s) failed; {} required architecture(s) unqualified",
            unqualified.len()
        );
        println!(
            "A skipped or absent architecture is unmeasured, never passing (document 07). \
             Run the missing profile on matching hardware, or narrow the claim."
        );
        return 1;
    }
    println!(
        "\ntest-gpu passed: {} required architecture(s) qualified",
        required.len()
    );
    0
}

/// Reduced real-chain entry point for Compute Sanitizer. It deliberately runs
/// no smoke/fault cases, so memcheck observes exactly the H8/H17 plan path.
pub fn run_chain() -> i32 {
    let count = match moxie_cuda::device_count() {
        Ok(count) if count > 0 => count,
        Ok(_) => {
            eprintln!("no CUDA devices visible: chain sanitizer measured nothing");
            return 2;
        }
        Err(error) => {
            eprintln!("cannot enumerate devices: {error}");
            return 2;
        }
    };
    let mut failed = 0;
    for ordinal in 0..count {
        let cap = match query_device(ordinal) {
            Ok(cap) => cap,
            Err(error) => {
                eprintln!("FAIL device {ordinal}: {error}");
                failed += 1;
                continue;
            }
        };
        match selected_bf16_device_chain(&cap) {
            Ok(Outcome::Passed) => println!("PASS {} {} reduced BF16 chain", cap.uuid, cap.sm()),
            Ok(other) => {
                eprintln!("FAIL {} {}: {other:?}", cap.uuid, cap.sm());
                failed += 1;
            }
            Err(error) => {
                eprintln!("FAIL {} {}: {error}", cap.uuid, cap.sm());
                failed += 1;
            }
        }
    }
    if failed == 0 { 0 } else { 1 }
}

/// Accept `sm86`, `86` and `sm_86` for the same profile.
fn normalise_profile(p: &str) -> String {
    let digits: String = p.chars().filter(|c| c.is_ascii_digit()).collect();
    format!("sm_{digits}")
}

fn case(cap: &DeviceCapability, name: &'static str, r: Result<Outcome, Error>) -> CaseResult {
    CaseResult {
        device: cap.ordinal,
        sm: cap.sm(),
        case: name,
        outcome: r.unwrap_or_else(|e| Outcome::Failed(e.to_string())),
    }
}

/// Allocate, copy, launch, copy back, verify against a host oracle.
fn axpy(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const N: usize = 4096;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN)?),
    )?;
    let func = module.function(moxie_kernels::AXPY_F32)?;

    let x: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
    let y_in: Vec<f32> = (0..N).map(|i| (i as f32) * -0.25).collect();
    let a = 3.0f32;

    let mut dx = DeviceBuffer::alloc(&ctx, N * 4)?;
    let mut dy = DeviceBuffer::alloc(&ctx, N * 4)?;
    dx.copy_from_host(bytemuck_f32(&x))?;
    dy.copy_from_host(bytemuck_f32(&y_in))?;

    let mut px = dx.device_ptr();
    let mut py = dy.device_ptr();
    let mut pa = a;
    let mut pn = N as u32;
    let mut params: [*mut c_void; 4] = [
        (&raw mut px).cast(),
        (&raw mut py).cast(),
        (&raw mut pa).cast(),
        (&raw mut pn).cast(),
    ];
    // SAFETY: the parameter list matches `moxie_smoke_axpy_f32(const float*,
    // float*, float, unsigned)` in count, order and type. Both device pointers
    // address N*4 bytes, which is exactly what the kernel indexes for i < N.
    unsafe {
        func.launch_blocking((N.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &mut params)?;
    }

    let mut out = vec![0f32; N];
    dy.copy_to_host(bytemuck_f32_mut(&mut out))?;

    for i in 0..N {
        let want = a * x[i] + y_in[i];
        if (out[i] - want).abs() > 1e-4 {
            return Ok(Outcome::Failed(format!(
                "index {i}: got {}, want {want}",
                out[i]
            )));
        }
    }
    Ok(Outcome::Passed)
}

/// Device f32 -> bf16 rounding must agree with the host oracle bit for bit.
fn bf16(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let inputs: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        // Exact tie: mantissa halfway between two bf16 values. Round-to-nearest-
        // even must pick the even one. This is the case a naive truncation gets
        // wrong, which is why document 03 pins the rounding rule.
        f32::from_bits(0x3F80_8000),
        f32::from_bits(0x3F81_8000),
        f32::from_bits(0x3F80_7FFF),
        1e-38,
        3.4e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let n = inputs.len();

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN)?),
    )?;
    let func = module.function(moxie_kernels::F32_TO_BF16_BITS)?;

    let mut dsrc = DeviceBuffer::alloc(&ctx, n * 4)?;
    let ddst = DeviceBuffer::alloc(&ctx, n * 2)?;
    dsrc.copy_from_host(bytemuck_f32(&inputs))?;

    let mut ps = dsrc.device_ptr();
    let mut pd = ddst.device_ptr();
    let mut pn = n as u32;
    let mut params: [*mut c_void; 3] = [
        (&raw mut ps).cast(),
        (&raw mut pd).cast(),
        (&raw mut pn).cast(),
    ];
    // SAFETY: matches `moxie_smoke_f32_to_bf16_bits(const float*, unsigned short*,
    // unsigned)`. Source holds n*4 bytes, destination n*2, and the kernel writes
    // one u16 per i < n.
    unsafe {
        func.launch_blocking((1, 1, 1), (n as u32, 1, 1), 0, &mut params)?;
    }

    let mut got = vec![0u16; n];
    ddst.copy_to_host(bytemuck_u16_mut(&mut got))?;

    for (i, v) in inputs.iter().enumerate() {
        let want = host_f32_to_bf16_bits(*v);
        if got[i] != want {
            return Ok(Outcome::Failed(format!(
                "input {v:e} (0x{:08x}): device 0x{:04x}, host oracle 0x{want:04x}",
                v.to_bits(),
                got[i]
            )));
        }
    }
    Ok(Outcome::Passed)
}

/// Round-to-nearest-even f32 -> bf16, computed independently of CUDA.
///
/// Document 03 pins this conversion. Duplicated in `moxie-format` as the shared
/// oracle; kept here too so this lane does not depend on that crate agreeing.
fn host_f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Quiet NaN, preserving the sign, as the hardware does.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits + 0x7FFF + lsb;
    (rounded >> 16) as u16
}

/// Loading an image with no binary for this device must be a typed error.
fn arch_mismatch(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let image = ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN_SM86_ONLY)?);
    let r = Module::load(&ctx, image);
    let is_sm86 = cap.sm() == "sm_86";
    match (is_sm86, r) {
        (true, Ok(_)) => Ok(Outcome::Passed),
        (true, Err(e)) => Ok(Outcome::Failed(format!(
            "sm_86 device rejected the sm_86 image: {e}"
        ))),
        (false, Ok(_)) => Ok(Outcome::Failed(
            "an sm_86-only image loaded on a non-sm_86 device; \
             an architecture mismatch went undetected"
                .into(),
        )),
        (false, Err(e)) => {
            // The point of the case: the failure must be typed, and typed as a
            // kernel/module problem rather than a generic numerical one.
            if e.kind() == "unsupported_kernel" {
                Ok(Outcome::Passed)
            } else {
                Ok(Outcome::Failed(format!(
                    "mismatch surfaced as {} rather than unsupported_kernel: {e}",
                    e.kind()
                )))
            }
        }
    }
}

/// Bounded stream/event completion smoke (document 06 M0.4).
///
/// ADR 0001 recorded streams and events as *declarations* only. This exercises
/// them: enqueue an asynchronous copy and a launch on a non-default stream,
/// record an event after them, observe that the event reports completion after
/// synchronising, and verify the device result. It proves the mechanism exists
/// and is typed. It is **not** the event-retained lease from R07 and does not
/// measure overlap.
fn stream_event(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const N: usize = 1024;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN)?),
    )?;
    let func = module.function(moxie_kernels::AXPY_F32)?;

    let stream = Stream::new(&ctx)?;
    let start = Event::new(&ctx)?;
    let done = Event::new(&ctx)?;

    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y_in = vec![1.0f32; N];
    let a = 2.0f32;

    let dx = DeviceBuffer::alloc(&ctx, N * 4)?;
    let dy = DeviceBuffer::alloc(&ctx, N * 4)?;

    start.record(&stream)?;
    // SAFETY: for both async copies, `x`, `y_in` and both device buffers are
    // live until after `done.synchronize()` below, which is the completion the
    // contract requires. Nothing reads or moves them in between.
    unsafe {
        dx.copy_from_host_async(bytemuck_f32(&x), &stream)?;
        dy.copy_from_host_async(bytemuck_f32(&y_in), &stream)?;
    }

    let mut px = dx.device_ptr();
    let mut py = dy.device_ptr();
    let mut pa = a;
    let mut pn = N as u32;
    let mut params: [*mut c_void; 4] = [
        (&raw mut px).cast(),
        (&raw mut py).cast(),
        (&raw mut pa).cast(),
        (&raw mut pn).cast(),
    ];
    // This launch is on the default stream and blocks, which orders it after the
    // stream work only because the synchronise below runs first.
    stream.synchronize()?;
    // SAFETY: same signature match as `axpy`, and the stream work above has
    // completed, so nothing this launch touches is still in flight.
    unsafe {
        func.launch_blocking((N.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &mut params)?;
    }
    done.record(&stream)?;
    done.synchronize()?;

    if !done.is_complete()? {
        return Ok(Outcome::Failed(
            "event reported incomplete after cuEventSynchronize returned".into(),
        ));
    }
    let ms = Event::elapsed_ms(&start, &done)?;
    if !ms.is_finite() || ms < 0.0 {
        return Ok(Outcome::Failed(format!(
            "elapsed time between two completed events is {ms}"
        )));
    }

    let mut out = vec![0f32; N];
    dy.copy_to_host(bytemuck_f32_mut(&mut out))?;
    for i in 0..N {
        let want = a * x[i] + y_in[i];
        if (out[i] - want).abs() > 1e-4 {
            return Ok(Outcome::Failed(format!(
                "index {i}: got {}, want {want} after stream/event path",
                out[i]
            )));
        }
    }
    Ok(Outcome::Passed)
}

/// Admit one upload envelope against the live device reading. A free function
/// rather than a closure so each call borrows the ledger briefly instead of
/// holding it across the leases below.
fn admit_upload(
    ledger: &mut Ledger,
    scope: Scope,
    label: &str,
    bytes: u64,
) -> Result<moxie_memory::Reservation, Error> {
    let mut req = PlanRequest::new(label, ["run"])?;
    req.buffer(BufferRequest::new(
        format!("{label}-buf"),
        scope,
        Tier::Device(DeviceTier::TransferStaging),
        bytes,
        StageSpan { first: 0, last: 0 },
    ))?;
    req.buffer(BufferRequest::new(
        format!("{label}-source"),
        Scope::Host,
        Tier::Host(HostTier::Pageable),
        bytes,
        StageSpan { first: 0, last: 0 },
    ))?;
    Ok(ledger.admit(&req)?)
}

/// The event-retained lease, against a real driver (task 0009, R07/R08).
///
/// Two async uploads on one stream, each bound to a lease by the event
/// recorded after it. The source buffers are reused only after their lease
/// retires — one directly, one through a turn sweep — and the device bytes
/// are read back against the host values. Refusal-before-completion is
/// timing-dependent on hardware and is proven by the host retirement tests
/// instead; what this proves is that the mechanism binds real bytes to real
/// completion and releases both sides.
fn backed_lease(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const N: usize = 1024;
    const BYTES: usize = N * 4;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let measurement = ctx.measure()?;
    let scope = Scope::Device(measurement.uuid);
    let snapshot = CapacitySnapshot::measured(&measurement, 1 << 20)?;
    let host = moxie_host::read()?;
    let host_snapshot = CapacitySnapshot::measured_host(&host, 1 << 20)?;
    let mut ledger = Ledger::new([snapshot, host_snapshot])?;
    let stream = Stream::new(&ctx)?;

    // First upload: retired directly after its event is observed. Readback
    // happens before retirement through the retained upload; retirement then
    // settles the buffer and returns the source for legal reuse.
    let want_a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mut lease_a = stage_upload(&mut ledger, &ctx, &stream, scope, "upload-a", &want_a)?;
    lease_a.synchronize()?;
    let mut out_a = vec![0f32; N];
    lease_a.readback(bytemuck_f32_mut(&mut out_a))?;
    for (i, (&got, &want)) in out_a.iter().zip(want_a.iter()).enumerate() {
        if got != want {
            return Ok(Outcome::Failed(format!(
                "index {i}: device bytes differ after leased upload"
            )));
        }
    }
    let (_, source_a) = lease_a.retire(&mut ledger).map_err(|r| r.error)?;
    if source_a.len() != BYTES {
        return Ok(Outcome::Failed(
            "retirement did not return the source".into(),
        ));
    }

    // Second upload: retired through a turn sweep, the R08 shape.
    let want_b: Vec<f32> = (0..N).map(|i| 1.0 + i as f32).collect();
    let mut lease_b = stage_upload(&mut ledger, &ctx, &stream, scope, "upload-b", &want_b)?;
    let mut out_b = vec![0f32; N];
    lease_b.readback(bytemuck_f32_mut(&mut out_b))?;
    let mut turn = Turn::new("upload-turn")?;
    turn.hold(lease_b);
    for (i, (&got, &want)) in out_b.iter().zip(want_b.iter()).enumerate() {
        if got != want {
            return Ok(Outcome::Failed(format!(
                "index {i}: device bytes differ after swept upload"
            )));
        }
    }
    let report = turn.release_turn(&mut ledger);
    if !report.is_clean() {
        return Ok(Outcome::Failed(format!(
            "turn sweep held {} lease(s): {:?}",
            report.held.len(),
            report.held.iter().map(|h| &h.label).collect::<Vec<_>>()
        )));
    }
    if report.retired.len() != 1 || !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "leases retired but bytes still charged".into(),
        ));
    }
    // Settlement returned the sources; the buffers are gone, so the empty
    // ledger tells the truth.
    if report.retired[0].resource.len() != BYTES {
        return Ok(Outcome::Failed("sweep did not return the source".into()));
    }
    Ok(Outcome::Passed)
}

/// One admitted physical allocation, three bounded ranges, and no per-range
/// allocation fallback (task 0010). Distinct patterns prove checked offsets;
/// transfer and generation checks prove persistent identity and safe reuse.
fn admitted_device_arena(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const ARENA_BYTES: u64 = 4096;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let measurement = ctx.measure()?;
    let scope = Scope::Device(measurement.uuid);
    let snapshot = CapacitySnapshot::measured(&measurement, 1 << 20)?;
    let host = moxie_host::read()?;
    let host_snapshot = CapacitySnapshot::measured_host(&host, 1 << 20)?;
    let mut ledger = Ledger::new([snapshot, host_snapshot])?;
    let mut request = PlanRequest::new("device arena", ["resident"])?;
    request.buffer(BufferRequest::new(
        "physical arena",
        scope,
        Tier::Device(DeviceTier::PackedResidentWeights),
        ARENA_BYTES,
        StageSpan { first: 0, last: 0 },
    ))?;
    request.buffer(BufferRequest::new(
        "one retained source",
        Scope::Host,
        Tier::Host(HostTier::Pageable),
        1792,
        StageSpan { first: 0, last: 0 },
    ))?;
    let reservation = ledger.admit(&request)?;
    let mut arena = DeviceArena::create(
        &ledger,
        reservation,
        &ctx,
        DeviceTier::PackedResidentWeights,
        ARENA_BYTES,
        "test-gpu arena",
    )
    .map_err(|r| r.error)?;
    let stream = Stream::new(&ctx)?;

    let first = arena.allocate(1024, 256, "importer").map_err(|r| r.error)?;
    let generation = first.key().generation;
    let second = arena
        .allocate(1280, 128, "workspace")
        .map_err(|r| r.error)?;
    let third = arena
        .allocate(1792, 64, "persistent")
        .map_err(|r| r.error)?;
    let refused = arena.allocate(1, 1, "must refuse").unwrap_err();
    if refused.occupancy.free_bytes != 0 {
        return Ok(Outcome::Failed(
            "arena exhaustion did not report zero free bytes".into(),
        ));
    }

    let first = upload_arena_range(first, vec![0x11; 1024], &stream, &ctx, false)?;
    let first_key = first.key();
    let first = arena.transfer(first, "executor").map_err(|r| r.error)?;
    if first.key() != first_key || first.owner() != "executor" {
        return Ok(Outcome::Failed(
            "persistent transfer changed identity or missed its owner".into(),
        ));
    }
    let second = upload_arena_range(second, vec![0x22; 1280], &stream, &ctx, true)?;
    let third = upload_arena_range(third, vec![0x33; 1792], &stream, &ctx, false)?;

    arena.release(second).map_err(|r| r.error)?;
    arena.release(first).map_err(|r| r.error)?;
    arena.release(third).map_err(|r| r.error)?;
    let whole = arena
        .allocate(ARENA_BYTES, 256, "coalesced")
        .map_err(|r| r.error)?;
    if whole.offset() != 0 || whole.key().generation <= generation {
        return Ok(Outcome::Failed(
            "coalesced full-range reuse did not advance its generation".into(),
        ));
    }
    arena.release(whole).map_err(|r| r.error)?;
    arena.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "closed arena left its parent reservation charged".into(),
        ));
    }
    Ok(Outcome::Passed)
}

const CHAIN_ROWS: SymbolId = SymbolId(1_212);
const CHAIN_ORACLE: OracleId = OracleId("task-0012-device-chain");

struct ChainFixture {
    graph: Graph,
    x: ValueId,
    weight: ValueId,
    gain: ValueId,
}

fn chain_graph(hidden: u64, eps: f32) -> Result<ChainFixture, Error> {
    let mut registry = OracleRegistry::new();
    for op in [Op::Linear, Op::RmsNorm, Op::Residual] {
        registry.register(
            op,
            CHAIN_ORACLE,
            OracleEvidence {
                implementation: "moxie-oracles",
                test_module: "task-0012-device-chain",
            },
        )?;
    }
    let activation = |shape| {
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            shape,
        )
    };
    let weight_spec = |shape| {
        TensorSpec::new(
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            shape,
        )
    };
    let mut builder = GraphBuilder::new(CHAIN_ORACLE, CHAIN_ROWS);
    let x = builder.input(
        "x",
        activation(vec![Dim::symbol(CHAIN_ROWS), Dim::constant(hidden)]),
    );
    let weight = builder.weight(
        "weight",
        weight_spec(vec![Dim::constant(hidden), Dim::constant(hidden)]),
    )?;
    let linear = builder.node(
        OpParams::Linear {
            in_features: hidden,
            out_features: hidden,
            bias: false,
        },
        &[x, weight],
    )?;
    let gain = builder.weight("gain", weight_spec(vec![Dim::constant(hidden)]))?;
    let norm = builder.node(OpParams::RmsNorm { hidden, eps }, &[linear, gain])?;
    let output = builder.node(OpParams::Residual, &[x, norm])?;
    Ok(ChainFixture {
        graph: builder.finish(output, &registry)?,
        x,
        weight,
        gain,
    })
}

/// The first selected semantic chain (task 0012), on every visible UUID.
///
/// It covers the exact decode and odd-tail prefill shapes, retained uploads,
/// one event, final-only readback, immutable weight reuse and explicit close.
fn selected_bf16_device_chain(cap: &DeviceCapability) -> Result<Outcome, Error> {
    for (rows, hidden, eps, exact) in [(1usize, 8usize, 3.5f32, true), (5, 17, 1e-5, false)] {
        let fixture = chain_graph(hidden as u64, eps)?;
        let workload = ResourceWorkload {
            phase: if rows == 1 {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            rows: rows as u64,
            visible_tokens: if rows == 5 { 32_768 } else { 1 },
            branch_rows: rows as u64,
            output: fixture.graph.output(),
            device: cap.uuid,
        };
        let catalogue = moxie_kernels::bf16_chain_catalogue();
        let candidate = lower_selected(&fixture.graph, workload, cap, &catalogue)?;
        let expected_total = if hidden == 8 { 1536 } else { 2048 };
        if candidate.combined_arena_bytes() != expected_total
            || candidate.workspace().logical_bytes != (rows * 4) as u64
            || candidate.workspace().physical_bytes != 256
        {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: wrong selected bytes: total={}, workspace={}/{}",
                candidate.combined_arena_bytes(),
                candidate.workspace().logical_bytes,
                candidate.workspace().physical_bytes
            )));
        }

        let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
        let stream = Stream::new(&ctx)?;
        let measurement = ctx.measure()?;
        let device_snapshot = CapacitySnapshot::measured(&measurement, 1 << 20)?;
        let host_snapshot = CapacitySnapshot::measured_host(&moxie_host::read()?, 1 << 20)?;
        let mut ledger = Ledger::new([device_snapshot, host_snapshot])?;
        let before = ctx.memory_info()?.0;
        let mut plan = match SelectedReservedPlan::admit(
            candidate,
            &fixture.graph,
            cap,
            &catalogue,
            &mut ledger,
            &ctx,
        ) {
            Ok(plan) => plan,
            Err(SelectedAdmitRefused::Invalid { error, .. })
            | Err(SelectedAdmitRefused::Held { error, .. }) => return Err(error),
            Err(SelectedAdmitRefused::Rejected { rejection, .. }) => {
                return Err((*rejection).into());
            }
        };
        let during = ctx.memory_info()?.0;
        if during >= before {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: the admitted physical arena spent no visible device memory"
            )));
        }

        let (x, weight, gain) = chain_values(rows, hidden, exact);
        let want = interpreter_chain(&fixture, rows, hidden, &x, &weight, &gain)?;
        let bindings = vec![
            owned_binding(
                fixture.x,
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                vec![rows as u64, hidden as u64],
                cap,
                &x,
            ),
            owned_binding(
                fixture.weight,
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                vec![hidden as u64, hidden as u64],
                cap,
                &weight,
            ),
            owned_binding(
                fixture.gain,
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                vec![hidden as u64],
                cap,
                &gain,
            ),
        ];
        let lease = plan
            .launch(&fixture.graph, cap, &catalogue, &ctx, &stream, bindings)
            .map_err(|refused| refused.error)?;
        let first = lease.finish().map_err(|refused| refused.error)?;
        if first.launch_order != ["linear", "rms-reduce", "rms-apply", "residual"] {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: launch order was {:?}",
                first.launch_order
            )));
        }
        let first_bits = decode_u16(&first.output);
        let distances: Vec<u16> = first_bits
            .iter()
            .zip(&want.residual)
            .map(|(got, want)| bf16_ulp_distance(*got, *want))
            .collect();
        let (max_ulp, rms_ulp, p99_ulp) = ulp_summary(&distances);
        if (exact && first_bits != want.residual) || (!exact && max_ulp > 1) {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: final output max ULP {max_ulp}, exact={exact}"
            )));
        }
        plan = first.plan;
        if plan.bound_weight_count() != 2 || first.returned_inputs.len() != 1 {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: completion retained {} weights and returned {} inputs",
                plan.bound_weight_count(),
                first.returned_inputs.len()
            )));
        }

        // The second execution supplies x only. The immutable device weights
        // remain plan-owned and cannot be uploaded or rebound a second time.
        let second_bindings = vec![owned_binding(
            fixture.x,
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![rows as u64, hidden as u64],
            cap,
            &x,
        )];
        let second = plan
            .launch(
                &fixture.graph,
                cap,
                &catalogue,
                &ctx,
                &stream,
                second_bindings,
            )
            .map_err(|refused| refused.error)?
            .finish()
            .map_err(|refused| refused.error)?;
        if decode_u16(&second.output) != first_bits || second.plan.bound_weight_count() != 2 {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: immutable-weight reuse changed the result"
            )));
        }
        second
            .plan
            .close(&mut ledger)
            .map_err(|refused| refused.error)?;
        if !ledger.outstanding().is_empty() {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: close left {} reservations",
                ledger.outstanding().len()
            )));
        }
        let after = ctx.memory_info()?.0;
        if after < during.saturating_add(expected_total) {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden}: close did not reconcile the {expected_total}-byte arena"
            )));
        }
        println!(
            "  selected chain {} rows={rows} H={hidden} arena={expected_total} B workspace={}/256 B ULP[max={max_ulp} rms={rms_ulp:.6} p99={p99_ulp}] second_execution=reused_weights",
            cap.uuid,
            rows * 4
        );
    }
    let overflow = selected_bf16_overflow_is_numerical(cap)?;
    if overflow != Outcome::Passed {
        return Ok(overflow);
    }
    selected_bf16_underflows_are_numerical(cap)
}

/// Finite BF16 operands can overflow the FP32 RMS reduction. The selected
/// chain must carry that invalid intermediate to the final bounded readback,
/// return a typed numerical refusal, and remain explicitly recoverable only
/// after the completion event has been observed.
fn selected_bf16_overflow_is_numerical(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let x = vec![1.0; 8];
    let weight = vec![2.0f32.powi(60); 64];
    let gain = vec![1.0; 8];
    let fixture = chain_graph(8, 1e-5)?;
    let oracle_error = interpreter_chain(&fixture, 1, 8, &x, &weight, &gain)
        .expect_err("the interpreter must reject an infinite RMS denominator");
    if oracle_error.kind() != "numerical" {
        return Ok(Outcome::Failed(format!(
            "RMS overflow oracle returned {} instead of numerical",
            oracle_error.kind()
        )));
    }
    selected_bf16_invalid_rms_is_numerical(cap, "RMS-overflow", 1e-5, &x, &weight, &gain)
}

/// Underflow is outside the fixed relative-error proof even when every
/// external BF16 value and the eventual device result are finite. Refuse both
/// owner-reproduced cases instead of accepting an output past the fixed bound.
fn selected_bf16_underflows_are_numerical(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let mut weight = vec![0.0; 64];
    for diagonal in 0..8 {
        weight[diagonal * 8 + diagonal] = 1.0;
    }
    for (label, input, gain, epsilon) in [
        (
            "RMS-reduction-underflow",
            2.0f32.powi(-75),
            1.0,
            f32::from_bits(1),
        ),
        (
            "RMS-scaling-underflow",
            2.0f32.powi(-80),
            2.0f32.powi(-70),
            2.0f32.powi(-126),
        ),
    ] {
        let outcome = selected_bf16_invalid_rms_is_numerical(
            cap,
            label,
            epsilon,
            &[input; 8],
            &weight,
            &[gain; 8],
        )?;
        if outcome != Outcome::Passed {
            return Ok(outcome);
        }
    }
    Ok(Outcome::Passed)
}

fn selected_bf16_invalid_rms_is_numerical(
    cap: &DeviceCapability,
    label: &'static str,
    epsilon: f32,
    x: &[f32],
    weight: &[f32],
    gain: &[f32],
) -> Result<Outcome, Error> {
    let rows = 1usize;
    let hidden = 8usize;
    let fixture = chain_graph(hidden as u64, epsilon)?;
    let workload = ResourceWorkload {
        phase: Phase::Decode,
        rows: rows as u64,
        visible_tokens: 1,
        branch_rows: rows as u64,
        output: fixture.graph.output(),
        device: cap.uuid,
    };
    let catalogue = moxie_kernels::bf16_chain_catalogue();
    let candidate = lower_selected(&fixture.graph, workload, cap, &catalogue)?;

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let device_snapshot = CapacitySnapshot::measured(&ctx.measure()?, 1 << 20)?;
    let host_snapshot = CapacitySnapshot::measured_host(&moxie_host::read()?, 1 << 20)?;
    let mut ledger = Ledger::new([device_snapshot, host_snapshot])?;
    let plan = match SelectedReservedPlan::admit(
        candidate,
        &fixture.graph,
        cap,
        &catalogue,
        &mut ledger,
        &ctx,
    ) {
        Ok(plan) => plan,
        Err(SelectedAdmitRefused::Invalid { error, .. })
        | Err(SelectedAdmitRefused::Held { error, .. }) => return Err(error),
        Err(SelectedAdmitRefused::Rejected { rejection, .. }) => return Err((*rejection).into()),
    };
    let bindings = vec![
        owned_binding(
            fixture.x,
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![rows as u64, hidden as u64],
            cap,
            x,
        ),
        owned_binding(
            fixture.weight,
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            vec![hidden as u64, hidden as u64],
            cap,
            weight,
        ),
        owned_binding(
            fixture.gain,
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            vec![hidden as u64],
            cap,
            gain,
        ),
    ];
    let refusal = plan
        .launch(&fixture.graph, cap, &catalogue, &ctx, &stream, bindings)
        .map_err(|refused| refused.error)?
        .finish()
        .expect_err("an invalid RMS intermediate must not become a successful finite result");
    if refusal.error.kind() != "numerical" {
        return Ok(Outcome::Failed(format!(
            "{label} returned {} instead of numerical",
            refusal.error.kind()
        )));
    }
    let (_, operation) = refusal.lease.retire().map_err(|refused| refused.error)?;
    let (plan, returned_inputs) = operation.into_parts();
    if plan.bound_weight_count() != 2 || returned_inputs.len() != 1 {
        return Ok(Outcome::Failed(format!(
            "{label} recovery lost its completed binding ownership"
        )));
    }
    plan.close(&mut ledger).map_err(|refused| refused.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(format!(
            "{label} recovery left its reservation charged"
        )));
    }
    println!(
        "  selected chain {} {label}=typed-numerical recovered_weights=2",
        cap.uuid,
    );
    Ok(Outcome::Passed)
}

fn chain_values(rows: usize, hidden: usize, exact: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if exact {
        let x = (0..rows * hidden)
            .map(|i| if i % 2 == 0 { 3.0 } else { 4.0 })
            .collect();
        let weight = (0..hidden * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        return (x, weight, vec![1.0; hidden]);
    }
    let x = (0..rows * hidden)
        .map(|i| {
            let raw = ((i * 29 + 7) % 31) as f32 - 15.0;
            bf16_value(host_f32_to_bf16_bits(raw / 8.0))
        })
        .collect();
    let weight = (0..hidden * hidden)
        .map(|i| {
            let raw = ((i * 17 + i / hidden * 3) % 13) as f32 - 6.0;
            bf16_value(host_f32_to_bf16_bits(raw / 8.0))
        })
        .collect();
    let gain = (0..hidden)
        .map(|i| [0.5, 0.75, 1.0, 1.25, -0.5][i % 5])
        .collect();
    (x, weight, gain)
}

#[derive(Debug)]
struct InterpreterChain {
    linear: Vec<u16>,
    norm: Vec<u16>,
    residual: Vec<u16>,
}

fn interpreter_chain(
    fixture: &ChainFixture,
    rows: usize,
    hidden: usize,
    x: &[f32],
    weight: &[f32],
    gain: &[f32],
) -> Result<InterpreterChain, Error> {
    let mut bindings = Bindings::new();
    bindings.set(
        fixture.x,
        Value::Float(HostTensor::bf16(x.to_vec(), vec![rows, hidden])?),
    );
    bindings.set(
        fixture.weight,
        Value::Float(HostTensor::bf16(weight.to_vec(), vec![hidden, hidden])?),
    );
    bindings.set(
        fixture.gain,
        Value::Float(HostTensor::bf16(gain.to_vec(), vec![hidden])?),
    );
    let trace = Interpreter::new().run_stateless(&fixture.graph, &bindings)?;
    let bits = |value: ValueId| -> Result<Vec<u16>, Error> {
        Ok(trace
            .node_output(value)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("interpreter trace omitted value {}", value.0),
            })?
            .as_float()?
            .data()
            .iter()
            .map(|value| host_f32_to_bf16_bits(*value))
            .collect())
    };
    Ok(InterpreterChain {
        linear: bits(fixture.graph.nodes()[0].output)?,
        norm: bits(fixture.graph.nodes()[1].output)?,
        residual: bits(fixture.graph.nodes()[2].output)?,
    })
}

fn owned_binding(
    value: ValueId,
    role: ValueRole,
    shape: Vec<u64>,
    cap: &DeviceCapability,
    values: &[f32],
) -> OwnedBinding {
    OwnedBinding {
        value,
        role,
        shape,
        layout: TensorLayout::ContiguousRowMajorV1,
        device: cap.uuid,
        bytes: encode_bf16(values),
    }
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&host_f32_to_bf16_bits(*value).to_le_bytes());
    }
    bytes
}

fn decode_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes([word[0], word[1]]))
        .collect()
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn bf16_ulp_distance(left: u16, right: u16) -> u16 {
    fn ordered(value: u16) -> i32 {
        if value & 0x8000 == 0 {
            0x8000 + value as i32
        } else {
            0x8000 - (value & 0x7fff) as i32
        }
    }
    (ordered(left) - ordered(right)).unsigned_abs() as u16
}

fn ulp_summary(distances: &[u16]) -> (u16, f64, u16) {
    let max = distances.iter().copied().max().unwrap_or(u16::MAX);
    let rms = (distances
        .iter()
        .map(|distance| f64::from(*distance).powi(2))
        .sum::<f64>()
        / distances.len() as f64)
        .sqrt();
    let mut sorted = distances.to_vec();
    sorted.sort_unstable();
    let p99 = sorted[((sorted.len() - 1) * 99).div_ceil(100)];
    (max, rms, p99)
}

#[derive(Clone, Copy)]
struct BoundSummary {
    count: usize,
    max_abs: f64,
    rms_abs: f64,
    p99_abs: f64,
    max_normalized: f64,
}

impl core::fmt::Display for BoundSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "count={} max={:.3e} rms={:.3e} p99={:.3e} normalized_max={:.6}",
            self.count, self.max_abs, self.rms_abs, self.p99_abs, self.max_normalized
        )
    }
}

/// Per-primitive FP64 equation gates. These intentionally use separate
/// readbacks from the integrated chain case, whose one-D2H census stays exact.
fn bf16_semantic_numerics(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::BF16_CHAIN_FATBIN)?),
    )?;
    for (rows, hidden, eps, exact) in [
        (1usize, 8usize, 3.5f32, true),
        (5, 17, 1e-5, false),
        (64, 1024, 1e-5, false),
    ] {
        let (x, weight, gain) = chain_values(rows, hidden, exact);
        let fixture = chain_graph(hidden as u64, eps)?;
        let interpreter = interpreter_chain(&fixture, rows, hidden, &x, &weight, &gain)?;
        let linear_bits = launch_linear(&ctx, &module, rows, hidden, &x, &weight)?;
        let linear = decode_bf16(&linear_bits);
        let (linear_want, linear_bounds) = linear_equation(rows, hidden, &x, &weight);
        let linear_summary = bound_summary(&linear, &linear_want, &linear_bounds);
        if !primitive_accepted(
            &linear_bits,
            &linear_want,
            &linear_bounds,
            exact.then_some(interpreter.linear.as_slice()),
        ) {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden} Linear exceeded its fixed bound: {linear_summary}"
            )));
        }

        let norm_bits = launch_rms(&ctx, &module, rows, hidden, eps, &linear, &gain)?;
        let norm = decode_bf16(&norm_bits);
        let (norm_want, norm_bounds) = rms_equation(rows, hidden, eps, &linear, &gain);
        let norm_summary = bound_summary(&norm, &norm_want, &norm_bounds);
        if !primitive_accepted(
            &norm_bits,
            &norm_want,
            &norm_bounds,
            exact.then_some(interpreter.norm.as_slice()),
        ) {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden} RMSNorm exceeded its fixed bound: {norm_summary}"
            )));
        }

        let residual_bits = launch_residual(&ctx, &module, &x, &norm)?;
        let residual = decode_bf16(&residual_bits);
        let (residual_want, residual_bounds) = residual_equation(&x, &norm);
        let residual_summary = bound_summary(&residual, &residual_want, &residual_bounds);
        if !primitive_accepted(
            &residual_bits,
            &residual_want,
            &residual_bounds,
            exact.then_some(interpreter.residual.as_slice()),
        ) {
            return Ok(Outcome::Failed(format!(
                "rows={rows} H={hidden} Residual exceeded its fixed bound: {residual_summary}"
            )));
        }
        println!(
            "  numerical {} rows={rows} H={hidden}: Linear [{linear_summary}]; RMSNorm [{norm_summary}]; Residual [{residual_summary}]",
            cap.uuid
        );
    }

    let overflow_x = vec![1.0; 8];
    let overflow_weight = vec![2.0f32.powi(60); 64];
    let overflow_gain = vec![1.0; 8];
    let overflow_linear = decode_bf16(&launch_linear(
        &ctx,
        &module,
        1,
        8,
        &overflow_x,
        &overflow_weight,
    )?);
    match launch_rms(&ctx, &module, 1, 8, 1e-5, &overflow_linear, &overflow_gain) {
        Err(error) if error.kind() == "numerical" => {}
        Err(error) => {
            return Ok(Outcome::Failed(format!(
                "primitive RMS overflow returned {} instead of numerical",
                error.kind()
            )));
        }
        Ok(_) => {
            return Ok(Outcome::Failed(
                "primitive RMS overflow became a successful output".into(),
            ));
        }
    }
    println!("  numerical {} RMS-overflow=typed-numerical", cap.uuid);

    // The first two fixtures are the independently reproduced failures. The
    // third keeps the square reduction normal so the scaling-underflow guard
    // is exercised independently rather than being masked by the row marker.
    for (label, input, gain, epsilon) in [
        (
            "RMS-reduction-underflow",
            2.0f32.powi(-75),
            1.0,
            f32::from_bits(1),
        ),
        (
            "RMS-scaling-underflow",
            2.0f32.powi(-80),
            2.0f32.powi(-70),
            2.0f32.powi(-126),
        ),
        (
            "RMS-scaling-underflow-isolated",
            2.0f32.powi(-60),
            2.0f32.powi(-90),
            2.0f32.powi(-126),
        ),
    ] {
        let input = vec![input; 8];
        let gain = vec![gain; 8];
        let (want, bounds) = rms_equation(1, 8, epsilon, &input, &gain);
        if want.iter().any(|value| !value.is_finite())
            || bounds.iter().any(|value| !value.is_finite())
        {
            return Ok(Outcome::Failed(format!(
                "{label} fixture has a nonfinite independent reference"
            )));
        }
        match launch_rms(&ctx, &module, 1, 8, epsilon, &input, &gain) {
            Err(error) if error.kind() == "numerical" => {}
            Err(error) => {
                return Ok(Outcome::Failed(format!(
                    "primitive {label} returned {} instead of numerical",
                    error.kind()
                )));
            }
            Ok(_) => {
                return Ok(Outcome::Failed(format!(
                    "primitive {label} became a successful output"
                )));
            }
        }
        println!("  numerical {} {label}=typed-numerical", cap.uuid);
    }
    Ok(Outcome::Passed)
}

fn launch_linear(
    ctx: &RankContext,
    module: &Module<'_>,
    rows: usize,
    hidden: usize,
    x: &[f32],
    weight: &[f32],
) -> Result<Vec<u16>, Error> {
    let x_bytes = encode_bf16(x);
    let weight_bytes = encode_bf16(weight);
    let mut dx = DeviceBuffer::alloc(ctx, x_bytes.len())?;
    let mut dw = DeviceBuffer::alloc(ctx, weight_bytes.len())?;
    let out_bytes = rows * hidden * 2;
    let output = DeviceBuffer::alloc(ctx, out_bytes)?;
    dx.copy_from_host(&x_bytes)?;
    dw.copy_from_host(&weight_bytes)?;
    let mut px = dx.device_ptr();
    let mut pw = dw.device_ptr();
    let mut po = output.device_ptr();
    let mut prows = rows as u64;
    let mut phidden = hidden as u64;
    let mut poutput = hidden as u64;
    let mut params: [*mut c_void; 6] = [
        (&raw mut px).cast(),
        (&raw mut pw).cast(),
        (&raw mut po).cast(),
        (&raw mut prows).cast(),
        (&raw mut phidden).cast(),
        (&raw mut poutput).cast(),
    ];
    let function = module.function(moxie_kernels::BF16_LINEAR)?;
    // SAFETY: the buffers and dimensions match the closed v1 symbol ABI.
    unsafe {
        function.launch_blocking(
            ((rows * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
    }
    let mut bits = vec![0u16; rows * hidden];
    output.copy_to_host(bytemuck_u16_mut(&mut bits))?;
    finite_primitive("Linear", bits)
}

fn launch_rms(
    ctx: &RankContext,
    module: &Module<'_>,
    rows: usize,
    hidden: usize,
    eps: f32,
    input: &[f32],
    gain: &[f32],
) -> Result<Vec<u16>, Error> {
    let input_bytes = encode_bf16(input);
    let gain_bytes = encode_bf16(gain);
    let mut di = DeviceBuffer::alloc(ctx, input_bytes.len())?;
    let mut dg = DeviceBuffer::alloc(ctx, gain_bytes.len())?;
    let sums = DeviceBuffer::alloc(ctx, rows * 4)?;
    let output = DeviceBuffer::alloc(ctx, rows * hidden * 2)?;
    di.copy_from_host(&input_bytes)?;
    dg.copy_from_host(&gain_bytes)?;
    let mut pi = di.device_ptr();
    let mut ps = sums.device_ptr();
    let mut prows = rows as u64;
    let mut phidden = hidden as u64;
    let mut reduce: [*mut c_void; 4] = [
        (&raw mut pi).cast(),
        (&raw mut ps).cast(),
        (&raw mut prows).cast(),
        (&raw mut phidden).cast(),
    ];
    // SAFETY: the buffers and dimensions match the closed RMS-sum v1 ABI.
    unsafe {
        module
            .function(moxie_kernels::BF16_RMS_SUM)?
            .launch_blocking((rows.div_ceil(64) as u32, 1, 1), (64, 1, 1), 0, &mut reduce)?;
    }
    let mut pg = dg.device_ptr();
    let mut po = output.device_ptr();
    let mut peps = eps;
    let mut apply: [*mut c_void; 7] = [
        (&raw mut pi).cast(),
        (&raw mut pg).cast(),
        (&raw mut ps).cast(),
        (&raw mut po).cast(),
        (&raw mut prows).cast(),
        (&raw mut phidden).cast(),
        (&raw mut peps).cast(),
    ];
    // SAFETY: the buffers and dimensions match the closed RMS-apply v1 ABI.
    unsafe {
        module
            .function(moxie_kernels::BF16_RMS_APPLY)?
            .launch_blocking(
                ((rows * hidden).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut apply,
            )?;
    }
    let mut bits = vec![0u16; rows * hidden];
    output.copy_to_host(bytemuck_u16_mut(&mut bits))?;
    finite_primitive("RMSNorm", bits)
}

fn launch_residual(
    ctx: &RankContext,
    module: &Module<'_>,
    left: &[f32],
    right: &[f32],
) -> Result<Vec<u16>, Error> {
    let left_bytes = encode_bf16(left);
    let right_bytes = encode_bf16(right);
    let mut dl = DeviceBuffer::alloc(ctx, left_bytes.len())?;
    let mut dr = DeviceBuffer::alloc(ctx, right_bytes.len())?;
    let output = DeviceBuffer::alloc(ctx, left_bytes.len())?;
    dl.copy_from_host(&left_bytes)?;
    dr.copy_from_host(&right_bytes)?;
    let mut pl = dl.device_ptr();
    let mut pr = dr.device_ptr();
    let mut po = output.device_ptr();
    let mut elements = left.len() as u64;
    let mut params: [*mut c_void; 4] = [
        (&raw mut pl).cast(),
        (&raw mut pr).cast(),
        (&raw mut po).cast(),
        (&raw mut elements).cast(),
    ];
    // SAFETY: the buffers and element count match the closed residual v1 ABI.
    unsafe {
        module
            .function(moxie_kernels::BF16_RESIDUAL)?
            .launch_blocking(
                (left.len().div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut params,
            )?;
    }
    let mut bits = vec![0u16; left.len()];
    output.copy_to_host(bytemuck_u16_mut(&mut bits))?;
    finite_primitive("Residual", bits)
}

fn finite_primitive(operation: &'static str, bits: Vec<u16>) -> Result<Vec<u16>, Error> {
    if bits.iter().any(|value| !bf16_value(*value).is_finite()) {
        return Err(Error::Numerical {
            detail: format!("{operation} produced a nonfinite BF16 primitive output"),
        });
    }
    Ok(bits)
}

fn linear_equation(rows: usize, hidden: usize, x: &[f32], weight: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut want = Vec::with_capacity(rows * hidden);
    let mut bounds = Vec::with_capacity(rows * hidden);
    for row in 0..rows {
        for output in 0..hidden {
            let mut sum = 0.0f64;
            let mut scale = 0.0f64;
            for k in 0..hidden {
                let term = x[row * hidden + k] as f64 * weight[output * hidden + k] as f64;
                sum += term;
                scale += term.abs();
            }
            want.push(sum);
            bounds.push(gamma(hidden as u64 + 1) * scale + BF16_U * sum.abs() + BF16_ETA);
        }
    }
    (want, bounds)
}

fn rms_equation(
    rows: usize,
    hidden: usize,
    eps: f32,
    input: &[f32],
    gain: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let mut want = Vec::with_capacity(rows * hidden);
    let mut bounds = Vec::with_capacity(rows * hidden);
    for row in 0..rows {
        let mut sum = 0.0f64;
        for k in 0..hidden {
            let value = input[row * hidden + k] as f64;
            sum += value * value;
        }
        let denom = (sum / hidden as f64 + eps as f64).sqrt();
        for k in 0..hidden {
            let value = input[row * hidden + k] as f64 * gain[k] as f64 / denom;
            want.push(value);
            bounds.push((gamma(hidden as u64 + 4) + BF16_U) * value.abs() + BF16_ETA);
        }
    }
    (want, bounds)
}

fn residual_equation(left: &[f32], right: &[f32]) -> (Vec<f64>, Vec<f64>) {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let want = *left as f64 + *right as f64;
            let bound = gamma(1) * ((*left as f64).abs() + (*right as f64).abs())
                + BF16_U * want.abs()
                + BF16_ETA;
            (want, bound)
        })
        .unzip()
}

const BF16_U: f64 = 1.0 / 256.0;
const BF16_ETA: f64 = 4.591_774_807_899_561e-41;

fn gamma(steps: u64) -> f64 {
    let product = steps as f64 * 2.0f64.powi(-24);
    product / (1.0 - product)
}

fn decode_bf16(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|value| bf16_value(*value)).collect()
}

fn bound_summary(got: &[f32], want: &[f64], bounds: &[f64]) -> BoundSummary {
    if got.iter().any(|value| !value.is_finite())
        || want.iter().any(|value| !value.is_finite())
        || bounds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return BoundSummary {
            count: got.len(),
            max_abs: f64::INFINITY,
            rms_abs: f64::INFINITY,
            p99_abs: f64::INFINITY,
            max_normalized: f64::INFINITY,
        };
    }
    let mut absolute: Vec<f64> = got
        .iter()
        .zip(want)
        .map(|(got, want)| (*got as f64 - *want).abs())
        .collect();
    let max_abs = absolute.iter().copied().fold(0.0, f64::max);
    let rms_abs =
        (absolute.iter().map(|value| value * value).sum::<f64>() / absolute.len() as f64).sqrt();
    absolute.sort_by(f64::total_cmp);
    let p99_index = ((absolute.len() - 1) * 99).div_ceil(100);
    let max_normalized = got
        .iter()
        .zip(want)
        .zip(bounds)
        .map(|((got, want), bound)| (*got as f64 - *want).abs() / *bound)
        .fold(0.0, f64::max);
    BoundSummary {
        count: got.len(),
        max_abs,
        rms_abs,
        p99_abs: absolute[p99_index],
        max_normalized,
    }
}

fn primitive_accepted(
    got_bits: &[u16],
    want: &[f64],
    bounds: &[f64],
    exact_bits: Option<&[u16]>,
) -> bool {
    if got_bits.len() != want.len() || want.len() != bounds.len() {
        return false;
    }
    let got = decode_bf16(got_bits);
    let summary = bound_summary(&got, want, bounds);
    summary.max_normalized <= 1.0 && exact_bits.is_none_or(|expected| got_bits == expected)
}

fn upload_arena_range<'ctx>(
    range: moxie_executor::DeviceRange<'ctx>,
    source: Vec<u8>,
    stream: &Stream<'ctx>,
    ctx: &'ctx RankContext,
    cancel: bool,
) -> Result<moxie_executor::DeviceRange<'ctx>, Error> {
    let expected = source.clone();
    let mut lease = range
        .prepare_upload(source, "test-gpu range upload")
        .map_err(|r| r.error)?;
    lease.submit(stream, Event::new(ctx)?)?;
    if cancel {
        lease.cancel();
    }
    let mut readback = vec![0; expected.len()];
    lease.readback(&mut readback)?;
    if readback != expected {
        return Err(Error::InvalidRequest {
            field: "arena",
            detail: "range readback differs at its checked offset".into(),
        });
    }
    let (_, upload) = lease.retire().map_err(|r| r.error)?;
    let (range, returned) = upload.finish();
    if returned != expected {
        return Err(Error::InvalidRequest {
            field: "arena",
            detail: "retirement did not return the retained source".into(),
        });
    }
    Ok(range)
}

/// Admit one upload envelope, prepare its staging, and submit both under a
/// lease, in that order: admission first, then allocation, then the single
/// submit that enqueues, records, tracks and retains together. A free
/// function rather than a closure so each step borrows the ledger briefly
/// instead of holding it across the leases below.
fn stage_upload<'ctx>(
    ledger: &mut Ledger,
    ctx: &'ctx RankContext,
    stream: &Stream<'ctx>,
    scope: Scope,
    label: &str,
    values: &[f32],
) -> Result<Lease<Event<'ctx>, Upload<'ctx>>, Error> {
    let reservation = admit_upload(ledger, scope, label, (values.len() * 4) as u64)?;
    let lease = Lease::acquire(ledger, reservation, label)?;
    let mut lease = lease
        .prepare_upload(ctx, bytemuck_f32(values).to_vec())
        .map_err(|r| r.error)?;
    lease.submit(stream, Event::new(ctx)?)?;
    Ok(lease)
}

/// Foreign completion is refused before copying, and the unsubmitted lease
/// remains recoverable. Post-copy record failure is covered by driver_faults.
fn lease_quarantine(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let count = moxie_cuda::device_count()?;
    if count < 2 {
        return Ok(Outcome::Skipped(
            "quarantine probe needs a second device".into(),
        ));
    }
    let other = (cap.ordinal + 1) % count;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let measurement = ctx.measure()?;
    let scope = Scope::Device(measurement.uuid);
    let snapshot = CapacitySnapshot::measured(&measurement, 1 << 20)?;
    let host = moxie_host::read()?;
    let host_snapshot = CapacitySnapshot::measured_host(&host, 1 << 20)?;
    let mut ledger = Ledger::new([snapshot, host_snapshot])?;
    let stream = Stream::new(&ctx)?;
    // A second live context on another device. Rank and device both differ,
    // so neither exclusivity rule fires.
    let ctx2 = RankContext::acquire(RankId(500 + other), other)?;
    let foreign = Event::new(&ctx2)?;

    let reservation = admit_upload(&mut ledger, scope, "quarantine", 64)?;
    let lease = Lease::acquire(&ledger, reservation, "foreign-event")?;
    let mut lease = lease
        .prepare_upload(&ctx, vec![3u8; 64])
        .map_err(|r| r.error)?;
    let error = lease
        .submit(&stream, foreign)
        .expect_err("foreign event must refuse");
    if error.kind() != "invalid_request" || lease.state() != moxie_executor::LeaseState::Live {
        return Ok(Outcome::Failed(format!(
            "wrong pre-submission refusal: {error}"
        )));
    }
    lease.retire(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "unsubmitted lease failed to release".into(),
        ));
    }
    Ok(Outcome::Passed)
}
/// The PTX boundary, in both halves, against a real driver.
///
/// 1. Text that is not PTX, and text that begins with a binary image magic,
///    are refused by `PtxSource` before the driver is called at all. The second
///    half matters most: `cuModuleLoadData` sniffs the leading bytes and decides
///    for itself which parser to use, so without that check a `&CStr` holding
///    ELF magic reached the binary-image parser through the text path.
/// 2. Text that *is* shaped like a PTX module but does not compile reaches the
///    driver and comes back as a typed `UnsupportedKernel`, not as a crash or a
///    generic numerical error.
///
/// It needs a device, so it lives in this lane rather than in a unit test.
fn ptx_rejection(cap: &DeviceCapability) -> Result<Outcome, Error> {
    // Half one: refused before any driver call.
    let not_ptx = CString::new("this is not ptx").expect("no interior NUL");
    if PtxSource::new(&not_ptx).is_ok() {
        return Ok(Outcome::Failed(
            "text with no .version directive was accepted as PTX".into(),
        ));
    }
    let elf_magic =
        CString::new([0x7Fu8, b'E', b'L', b'F', b'\n', b'.', b'v'].as_slice()).expect("no NUL");
    if PtxSource::new(&elf_magic).is_ok() {
        return Ok(Outcome::Failed(
            "a buffer beginning with ELF magic was accepted as PTX; the driver would \
             parse it as a binary image"
                .into(),
        ));
    }

    // Half two: shaped like PTX, does not compile, must be typed.
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let broken =
        CString::new(".version 8.0\n.target sm_86\n.address_size 64\nnot_an_instruction\n")
            .expect("no interior NUL");
    let src = PtxSource::new(&broken)?;
    match Module::load(&ctx, ModuleImage::Ptx(src)) {
        Ok(_) => Ok(Outcome::Failed(
            "the driver compiled deliberately invalid PTX".into(),
        )),
        Err(e) if e.kind() == "unsupported_kernel" => Ok(Outcome::Passed),
        Err(e) => Ok(Outcome::Failed(format!(
            "invalid PTX surfaced as {}: {e}",
            e.kind()
        ))),
    }
}

// Small local reinterpretation helpers. Deliberately not a dependency: these are
// the only three shapes this lane needs.
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and no invalid bit patterns; the resulting
    // slice covers exactly the same allocation, with alignment 1.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
}

fn bytemuck_f32_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern written back is a valid f32.
    unsafe {
        core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), core::mem::size_of_val(v))
    }
}

fn bytemuck_u16_mut(v: &mut [u16]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern is a valid u16.
    unsafe {
        core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), core::mem::size_of_val(v))
    }
}

/// One rank, one GPU, and the claim released when the context drops.
///
/// Document 01 asks for unsafe CUDA state to be isolated behind rank-owned
/// contexts. A second context on one card is exactly the shared state that
/// isolation removes, and the driver will happily hand one out, so this proves
/// the engine refuses instead.
fn rank_exclusivity(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let held = RankContext::acquire(RankId(0), cap.ordinal)?;
    if held.uuid() != cap.uuid {
        return Err(Error::Numerical {
            detail: format!("acquired {} while querying {}", held.uuid(), cap.uuid),
        });
    }

    // A second rank cannot take a device that rank 0 holds.
    match RankContext::acquire(RankId(1), cap.ordinal) {
        Ok(_) => {
            return Err(Error::Numerical {
                detail: "a second rank acquired a device rank 0 already holds".into(),
            });
        }
        Err(e) => {
            let text = e.to_string();
            if !text.contains(&cap.uuid.to_string()) || !text.contains("rank 0") {
                return Err(Error::Numerical {
                    detail: format!("refusal names neither the device nor the holder: {text}"),
                });
            }
        }
    }

    // The refusal left the first context usable.
    let (free, total) = held.memory_info()?;
    if total == 0 || free > total {
        return Err(Error::Numerical {
            detail: format!("holder unusable after the refusal: {free} free of {total}"),
        });
    }

    // Rank 0 already holds this device, so it may not take another one.
    if moxie_cuda::device_count()? > 1 {
        let other = (cap.ordinal + 1) % moxie_cuda::device_count()?;
        if RankContext::acquire(RankId(0), other).is_ok() {
            return Err(Error::Numerical {
                detail: "one rank acquired two devices".into(),
            });
        }
    }

    // Dropping releases the claim, and a later rank takes the card.
    drop(held);
    let next = RankContext::acquire(RankId(7), cap.ordinal)?;
    if next.rank() != RankId(7) || next.uuid() != cap.uuid {
        return Err(Error::Numerical {
            detail: "the released device came back as a different rank or device".into(),
        });
    }
    Ok(Outcome::Passed)
}

/// A measurement is a reading, not a constant.
///
/// Allocating on the device must move `free_bytes`. If it does not, the number
/// reaching the ledger is decoration, and every admission decision made from it
/// would be fiction.
fn measurement_is_live(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const BYTES: usize = 64 * 1024 * 1024;
    let ctx = RankContext::acquire(RankId(0), cap.ordinal)?;
    let before = ctx.measure()?;
    if before.uuid != cap.uuid || before.ordinal_label != cap.ordinal {
        return Err(Error::Numerical {
            detail: "the measurement does not identify the device it came from".into(),
        });
    }
    if before.total_bytes == 0 || before.free_bytes > before.total_bytes {
        return Err(Error::Numerical {
            detail: format!(
                "{} B free of {} B total",
                before.free_bytes, before.total_bytes
            ),
        });
    }

    let buffer = DeviceBuffer::alloc(&ctx, BYTES)?;
    let after = ctx.measure()?;
    if after.total_bytes != before.total_bytes {
        return Err(Error::Numerical {
            detail: "total memory changed under an allocation".into(),
        });
    }
    if after.free_bytes >= before.free_bytes {
        return Err(Error::Numerical {
            detail: format!(
                "allocating {BYTES} B did not reduce free memory: {} then {}",
                before.free_bytes, after.free_bytes
            ),
        });
    }
    drop(buffer);
    Ok(Outcome::Passed)
}

/// A device held on one thread is refused on another, and handed over cleanly
/// once the holder is gone.
///
/// The exclusivity *window* -- that the claim outlives the driver teardown -- is
/// proved deterministically by `moxie_cuda::claims`' host tests, which can pause
/// inside the teardown. This case proves the same property across real threads
/// on real hardware, where the timing is not ours to choose.
fn concurrent_handoff(cap: &DeviceCapability) -> Result<Outcome, Error> {
    use std::sync::mpsc;

    let ordinal = cap.ordinal;
    let (ready_tx, ready_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel();

    // The context is `!Send`, so it is acquired, held and dropped entirely
    // inside the thread that owns it.
    let holder = std::thread::spawn(move || -> Result<(), Error> {
        let ctx = RankContext::acquire(RankId(20), ordinal)?;
        let uuid = ctx.uuid();
        ready_tx.send(uuid).expect("the main thread is waiting");
        go_rx
            .recv()
            .expect("the main thread signals before joining");
        drop(ctx);
        Ok(())
    });

    let uuid = ready_rx.recv().map_err(|_| Error::Numerical {
        detail: "the holding thread failed before it acquired the device".into(),
    })?;

    // Held by another thread: refused, and the refusal names the holder.
    match RankContext::acquire(RankId(21), ordinal) {
        Ok(_) => {
            let _ = go_tx.send(());
            let _ = holder.join();
            return Err(Error::Numerical {
                detail: "a second thread acquired a device another rank holds".into(),
            });
        }
        Err(e) => {
            let text = e.to_string();
            if !text.contains("rank 20") {
                let _ = go_tx.send(());
                let _ = holder.join();
                return Err(Error::Numerical {
                    detail: format!("the refusal does not name the holding rank: {text}"),
                });
            }
        }
    }

    go_tx.send(()).expect("the holder is waiting");
    holder.join().map_err(|_| Error::Numerical {
        detail: "the holding thread panicked".into(),
    })??;

    // The handover completed: the card is available, and usable.
    let taken = RankContext::acquire(RankId(21), ordinal)?;
    if taken.uuid() != uuid {
        return Err(Error::Numerical {
            detail: "the handed-over device is not the one that was released".into(),
        });
    }
    taken.measure()?;
    Ok(Outcome::Passed)
}

#[cfg(test)]
mod task_0012_negative_fixtures {
    use super::*;

    #[test]
    fn nonfinite_metrics_fail_closed() {
        let summary = bound_summary(&[f32::NAN], &[0.0], &[1.0]);
        assert!(summary.max_normalized.is_infinite());
        let summary = bound_summary(&[0.0], &[f64::INFINITY], &[1.0]);
        assert!(summary.max_normalized.is_infinite());
    }

    #[test]
    fn forbidden_semantic_substitutions_fail_the_real_acceptance_gate() {
        // This BF16 vector lands exactly on a BF16 tie in ascending FP32
        // accumulation. Reversing the adds nudges it above the tie, changing
        // the stored BF16 bit pattern even though both results satisfy the
        // ordinary analytical rounding bound.
        let terms = [
            6.1875f32,
            f32::from_bits(0x3580_0000), // 2^-20
            7.375,
            4.875,
            f32::from_bits(0x3600_0000), // 2^-19
            7.0625,
            1.375,
            7.75,
        ];
        let mut ascending = 0.0f32;
        for value in &terms {
            ascending += *value;
        }
        let mut reversed = 0.0f32;
        for value in terms.iter().rev() {
            reversed += *value;
        }
        let mut ascending_bits = vec![0; 8];
        ascending_bits[0] = host_f32_to_bf16_bits(ascending);
        let mut reversed_bits = vec![0; 8];
        reversed_bits[0] = host_f32_to_bf16_bits(reversed);
        let mut identity_row = vec![0.0; 64];
        identity_row[..8].fill(1.0);
        let (linear_want, linear_bounds) = linear_equation(1, 8, &terms, &identity_row);
        assert!(primitive_accepted(
            &ascending_bits,
            &linear_want,
            &linear_bounds,
            Some(&ascending_bits),
        ));
        assert!(!primitive_accepted(
            &reversed_bits,
            &linear_want,
            &linear_bounds,
            Some(&ascending_bits),
        ));

        // Feed a deliberately unrounded Linear result into RMSNorm. The same
        // per-node fixed-bound gate used above rejects the substituted output.
        let unrounded = [-2.312_744_1f32, 1.878_906_2];
        let rounded: Vec<_> = unrounded
            .iter()
            .map(|value| bf16_value(host_f32_to_bf16_bits(*value)))
            .collect();
        let mut sum = 0.0f32;
        for value in unrounded {
            sum += value * value;
        }
        let denom = (sum / 2.0 + 1e-5).sqrt();
        let omitted_boundary_bits: Vec<_> = unrounded
            .iter()
            .map(|value| host_f32_to_bf16_bits(*value / denom))
            .collect();
        let (rms_want, rms_bounds) = rms_equation(1, 2, 1e-5, &rounded, &[1.0, 1.0]);
        assert!(!primitive_accepted(
            &omitted_boundary_bits,
            &rms_want,
            &rms_bounds,
            None,
        ));

        let norm_input = [10.0f32, 11.0, 12.0, 13.0];
        let (rms_want, rms_bounds) = rms_equation(1, 4, 1e-5, &norm_input, &[1.0; 4]);
        let mean = norm_input.iter().sum::<f32>() / 4.0;
        let variance = norm_input
            .iter()
            .map(|value| (*value - mean) * (*value - mean))
            .sum::<f32>()
            / 4.0;
        let layer_denom = (variance + 1e-5).sqrt();
        let layer_bits: Vec<_> = norm_input
            .iter()
            .map(|value| host_f32_to_bf16_bits((*value - mean) / layer_denom))
            .collect();
        assert!(!primitive_accepted(
            &layer_bits,
            &rms_want,
            &rms_bounds,
            None,
        ));

        let input = [1.0f32, -2.0];
        let residual = [10.0f32, 20.0];
        let once: Vec<_> = input.iter().zip(residual).map(|(a, b)| *a + b).collect();
        let twice: Vec<_> = once.iter().zip(residual).map(|(a, b)| *a + b).collect();
        let twice_bits: Vec<_> = twice
            .iter()
            .map(|value| host_f32_to_bf16_bits(*value))
            .collect();
        let (residual_want, residual_bounds) = residual_equation(&input, &residual);
        assert!(!primitive_accepted(
            &twice_bits,
            &residual_want,
            &residual_bounds,
            None,
        ));
    }
}
