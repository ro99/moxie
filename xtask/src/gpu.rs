//! The real-hardware lane: `cargo xtask-cuda test-gpu [--profile sm_NN]`.
//!
//! Document 07: "unsupported hardware is skipped/unmeasured, never passed" and
//! "a test that catches a device error then exits successfully is not a passing
//! production test". So this lane reports pass/fail/skip per device explicitly,
//! returns non-zero when any *attempted* case failed, **and** returns non-zero
//! when a required architecture produced no passing case at all.
//!
//! That last rule matters because an all-skipped run must not look like a
//! qualification: a CI wrapper reading exit 0 cannot otherwise distinguish
//! "every required architecture passed" from "every required architecture was
//! missing."

use core::ffi::c_void;
use std::ffi::CString;

use moxie_cuda::{
    DeviceBuffer, Event, Module, ModuleImage, PtxSource, RankContext, Stream, TrustedImage,
    query_device,
};
use moxie_executor::{
    AttentionLayer, DeviceArena, Lease, OwnedBinding, PageGeometry, PagedAttentionInputs,
    PagedAttentionLaunch, PagedAttentionRun, PagedAttentionStep, SelectedAdmitRefused,
    SelectedReservedPlan, Staging, Turn, TwoBlockAttention, Upload, select_paged_attention_kernel,
};
use moxie_graph::{
    Bindings, Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry,
    TensorSpec, ValueId, ValueRole,
};
use moxie_interp::{HostTensor, Interpreter, Value};
use moxie_memory::{
    BufferRequest, CapacitySnapshot, HostBackedPlan, Ledger, PlanRequest, StageSpan,
};
use moxie_plan::{
    PagedStateCapacity, Phase, ResourceWorkload, StateRequirementKind, Visibility, lower_selected,
};
use moxie_types::{
    ActivationPrecision, DeviceCapability, DeviceTier, Dim, Error, HostTier, PagePlacement,
    Precision, RankId, Scope, SymbolId, TensorLayout, Tier, WeightPrecision,
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
    "graph_capture_replay",
    "graph_memory_within_bound",
    "event_backed_lease",
    "admitted_device_arena",
    "selected_bf16_device_chain",
    "bf16_semantic_numerics",
    "lease_rejects_foreign_completion",
    "non_ptx_text_rejected",
    "rank_context_is_exclusive",
    "concurrent_handoff_is_exclusive",
    "measurement_is_live",
    "grouped_expert_mlp",
    "affine_linear_w4a16_w8a16",
    "paged_attention",
    "paged_attention_indirect",
    "paged_attention_host_streaming",
    "paged_attention_host_streaming_n3",
    "paged_attention_32k",
    "paged_attention_state_lifecycle",
    "paged_attention_device_cow",
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
        results.push(case(
            &cap,
            "graph_capture_replay",
            graph_capture_replay(&cap),
        ));
        results.push(case(
            &cap,
            "graph_memory_within_bound",
            graph_memory_within_bound(&cap),
        ));
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
        results.push(case(&cap, "grouped_expert_mlp", grouped_expert_mlp(&cap)));
        results.push(case(&cap, "affine_linear_w4a16_w8a16", affine_linear(&cap)));
        results.push(case(&cap, "paged_attention", paged_attention(&cap)));
        results.push(case(
            &cap,
            "paged_attention_indirect",
            paged_attention_indirect(&cap),
        ));
        results.push(case(
            &cap,
            "paged_attention_host_streaming",
            paged_attention_host_streaming(&cap),
        ));
        results.push(case(
            &cap,
            "paged_attention_host_streaming_n3",
            paged_attention_host_streaming_n3(&cap),
        ));
        results.push(case(&cap, "paged_attention_32k", paged_attention_32k(&cap)));
        results.push(case(
            &cap,
            "paged_attention_state_lifecycle",
            paged_attention_state_lifecycle(&cap),
        ));
        results.push(case(
            &cap,
            "paged_attention_device_cow",
            paged_attention_device_cow(&cap),
        ));
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

/// Captured launches replay in order and an invalidated capture releases its stream.
fn graph_capture_replay(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const N: usize = 1024;
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let package = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN)?),
    )?
    .resolve_all(&[moxie_kernels::AXPY_F32.to_string()])?;
    let stream = Stream::new(&ctx)?;

    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y_initial = vec![1.0f32; N];
    let a = 2.0f32;
    let mut dx = DeviceBuffer::alloc(&ctx, N * 4)?;
    let mut dy = DeviceBuffer::alloc(&ctx, N * 4)?;
    dx.copy_from_host(bytemuck_f32(&x))?;
    dy.copy_from_host(bytemuck_f32(&y_initial))?;

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
    stream.begin_capture()?;
    // SAFETY: the parameters match the AXPY symbol; both device buffers and
    // the module stay live through the graph's launches and synchronization.
    unsafe {
        package.launch_async(
            0,
            &stream,
            (N.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
        package.launch_async(
            0,
            &stream,
            (N.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
    }
    let graph = stream.end_capture()?;
    // SAFETY: this graph names only the live buffers above, neither of which
    // the host accesses until the following stream synchronization completes.
    unsafe {
        graph.launch(&stream)?;
        graph.launch(&stream)?;
    }
    stream.synchronize()?;

    let mut expected = y_initial;
    for _ in 0..4 {
        for (y, &x) in expected.iter_mut().zip(&x) {
            *y += a * x;
        }
    }
    let mut output = vec![0.0f32; N];
    dy.copy_to_host(bytemuck_f32_mut(&mut output))?;
    for (index, (&got, &want)) in output.iter().zip(&expected).enumerate() {
        if got != want {
            return Ok(Outcome::Failed(format!(
                "index {index}: graph replay got {got}, want {want} after four AXPY launches"
            )));
        }
    }

    stream.begin_capture()?;
    // SAFETY: the same live buffers and package are used; this launch is
    // intentionally enqueued during capture so synchronize invalidates a
    // non-empty graph.
    unsafe {
        package.launch_async(
            0,
            &stream,
            (N.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
    }
    if stream.synchronize().is_ok() {
        return Ok(Outcome::Failed(
            "synchronize succeeded during stream capture".into(),
        ));
    }
    if let Ok(unexpected_graph) = stream.end_capture() {
        drop(unexpected_graph);
        return Ok(Outcome::Failed(
            "end_capture succeeded after capture was invalidated by synchronize".into(),
        ));
    }
    // SAFETY: the same live buffers and package are used; the failed capture
    // has ended, and the work is observed before the function returns.
    unsafe {
        package.launch_async(
            0,
            &stream,
            (N.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
    }
    stream.synchronize()?;

    let mut expected_after_recovery = expected;
    for (y, &x) in expected_after_recovery.iter_mut().zip(&x) {
        *y += a * x;
    }
    dy.copy_to_host(bytemuck_f32_mut(&mut output))?;
    for (index, (&got, &want)) in output.iter().zip(&expected_after_recovery).enumerate() {
        if got != want {
            return Ok(Outcome::Failed(format!(
                "index {index}: ordinary launch after invalidated capture got {got}, want {want}"
            )));
        }
    }
    Ok(Outcome::Passed)
}

/// Measure CUDA graph-pool use for kernel nodes and executable graphs.
fn graph_memory_within_bound(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const ELEMENTS: usize = 256;
    const KERNEL_NODES: u64 = 16_384;
    const GRAPH_COUNT: usize = 256;

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let package = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::SMOKE_FATBIN)?),
    )?
    .resolve_all(&[moxie_kernels::AXPY_F32.to_string()])?;
    let stream = Stream::new(&ctx)?;
    let x = vec![1.0f32; ELEMENTS];
    let y = vec![1.0f32; ELEMENTS];
    let a = 2.0f32;
    let mut dx = DeviceBuffer::alloc(&ctx, ELEMENTS * size_of::<f32>())?;
    let mut dy = DeviceBuffer::alloc(&ctx, ELEMENTS * size_of::<f32>())?;
    dx.copy_from_host(bytemuck_f32(&x))?;
    dy.copy_from_host(bytemuck_f32(&y))?;

    let mut px = dx.device_ptr();
    let mut py = dy.device_ptr();
    let mut pa = a;
    let mut pn = u32::try_from(ELEMENTS).expect("vector length fits kernel ABI");
    let mut params: [*mut c_void; 4] = [
        (&raw mut px).cast(),
        (&raw mut py).cast(),
        (&raw mut pa).cast(),
        (&raw mut pn).cast(),
    ];
    let block = (pn, 1, 1);
    let mut graphs = Vec::with_capacity(GRAPH_COUNT + 1);

    stream.synchronize()?;
    let node_free_before = ctx.memory_info()?.0;
    stream.begin_capture()?;
    // SAFETY: the arguments match AXPY; the module and both buffers stay live
    // until every captured graph has been measured and dropped.
    unsafe {
        for _ in 0..KERNEL_NODES {
            package.launch_async(0, &stream, (1, 1, 1), block, 0, &mut params)?;
        }
    }
    graphs.push(stream.end_capture()?);
    stream.synchronize()?;
    let node_free_after = ctx.memory_info()?.0;
    let node_delta = i128::from(node_free_before) - i128::from(node_free_after);

    let graph_free_before = ctx.memory_info()?.0;
    for _ in 0..GRAPH_COUNT {
        stream.begin_capture()?;
        // SAFETY: this uses the same live module, parameters, and buffers.
        unsafe {
            package.launch_async(0, &stream, (1, 1, 1), block, 0, &mut params)?;
        }
        graphs.push(stream.end_capture()?);
    }
    stream.synchronize()?;
    let graph_free_after = ctx.memory_info()?.0;
    let graph_delta = i128::from(graph_free_before) - i128::from(graph_free_after);

    println!(
        "graph-memory gpu={} node_free_before_bytes={} node_free_after_bytes={} node_delta_bytes={} graph_free_before_bytes={} graph_free_after_bytes={} graph_delta_bytes={}",
        cap.uuid,
        node_free_before,
        node_free_after,
        node_delta,
        graph_free_before,
        graph_free_after,
        graph_delta,
    );
    if node_delta < 0 || graph_delta < 0 {
        return Ok(Outcome::Failed(
            "free memory increased during graph memory measurement".into(),
        ));
    }
    let graph_count = u64::try_from(GRAPH_COUNT).map_err(|_| Error::InvalidRequest {
        field: "graph_memory",
        detail: "graph count exceeds u64".into(),
    })?;
    let node_delta = u64::try_from(node_delta).map_err(|_| Error::InvalidRequest {
        field: "graph_memory",
        detail: "node memory delta exceeds u64".into(),
    })?;
    let graph_delta = u64::try_from(graph_delta).map_err(|_| Error::InvalidRequest {
        field: "graph_memory",
        detail: "graph memory delta exceeds u64".into(),
    })?;
    let kernel_bound_bytes = SelectedReservedPlan::CAPTURED_KERNEL_BOUND_BYTES;
    let graph_bound_bytes = SelectedReservedPlan::CAPTURED_GRAPH_BOUND_BYTES;
    let node_bound = KERNEL_NODES
        .checked_mul(kernel_bound_bytes)
        .and_then(|bytes| bytes.checked_add(graph_bound_bytes))
        .ok_or_else(|| Error::InvalidRequest {
            field: "graph_memory",
            detail: "kernel graph memory bound overflowed".into(),
        })?;
    let graph_bound = graph_count
        .checked_mul(
            kernel_bound_bytes
                .checked_add(graph_bound_bytes)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "graph_memory",
                    detail: "per-graph memory bound overflowed".into(),
                })?,
        )
        .ok_or_else(|| Error::InvalidRequest {
            field: "graph_memory",
            detail: "graph memory bound overflowed".into(),
        })?;
    if node_delta > node_bound {
        return Ok(Outcome::Failed(format!(
            "node delta {node_delta} exceeds bound {node_bound}"
        )));
    }
    if graph_delta > graph_bound {
        return Ok(Outcome::Failed(format!(
            "graph delta {graph_delta} exceeds bound {graph_bound}"
        )));
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
const CHAIN_ORACLE: OracleId = OracleId("device-chain");
const ATTENTION_ROWS: SymbolId = SymbolId(3_038);
const ATTENTION_ORACLE: OracleId = OracleId("selected-attention");

fn selected_attention_graph(
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    scale: f32,
    visibility: Visibility,
) -> Result<Graph, Error> {
    let mut registry = OracleRegistry::new();
    registry.register(
        Op::Attention,
        ATTENTION_ORACLE,
        OracleEvidence {
            implementation: "moxie-oracles",
            test_module: "selected-attention",
        },
    )?;
    let activation = |width| {
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(ATTENTION_ROWS), Dim::constant(width)],
        )
    };
    let mut builder = GraphBuilder::new(ATTENTION_ORACLE, ATTENTION_ROWS);
    let query = builder.input("query", activation(heads * head_dim));
    let keys = builder.input("keys", activation(kv_heads * head_dim));
    let values = builder.input("values", activation(kv_heads * head_dim));
    let positions = builder.input(
        "positions",
        TensorSpec::new(
            ValueRole::Index(moxie_graph::IndexEncoding::U64),
            vec![Dim::symbol(ATTENTION_ROWS)],
        ),
    );
    let output = builder.node(
        OpParams::Attention {
            heads,
            kv_heads,
            head_dim,
            scale,
            visibility,
            layer: 0,
        },
        &[query, keys, values, positions],
    )?;
    builder.finish(output, &registry)
}

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
                test_module: "device-chain",
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
    let norm = builder.node(
        OpParams::RmsNorm {
            hidden,
            eps,
            group: 1,
        },
        &[linear, gain],
    )?;
    let output = builder.node(OpParams::Residual { scale: 1.0 }, &[x, norm])?;
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
            paged_state_capacity: None,
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
                return Err(rejection.into());
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
        paged_state_capacity: None,
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
        Err(SelectedAdmitRefused::Rejected { rejection, .. }) => return Err(rejection.into()),
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
                detail: format!("interpreter trace omitted value {}", value.0).into(),
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

/// Task 0021's grouped expert kernel against task 0019's oracle, on real
/// hardware, for both gate transforms.
///
/// The declared gate is **bitwise** equality of the BF16 slot outputs. Nothing
/// here is a checkpoint and nothing here is a model: the weights are bytes this
/// function invents, and a synthetic routed block is not MoE support.
fn grouped_expert_mlp(cap: &DeviceCapability) -> Result<Outcome, Error> {
    use moxie_graph::ExpertActivation;
    use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
    use moxie_oracles::route;

    const HIDDEN: u64 = 96;
    const INTERMEDIATE: u64 = 24;
    const EXPERTS: u64 = 3;
    const ASSIGNMENTS: u64 = 5;

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::EXPERT_MLP_FATBIN)?),
    )?;

    // Deterministic BF16-representable values, so "the fixture" is a fact.
    let mut state = 0x9e37_79b9_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        bf16_round(((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5)
    };
    let x: Vec<f32> = (0..(ASSIGNMENTS * HIDDEN) as usize)
        .map(|_| next())
        .collect();
    let gate_up: Vec<f32> = (0..(EXPERTS * 2 * INTERMEDIATE * HIDDEN) as usize)
        .map(|_| next())
        .collect();
    let down: Vec<f32> = (0..(EXPERTS * HIDDEN * INTERMEDIATE) as usize)
        .map(|_| next())
        .collect();

    let encode = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
            .collect()
    };

    let mut compared = 0usize;
    for (activation, symbol) in [
        (
            ExpertActivation::GeGlu,
            moxie_kernels::BF16_EXPERT_PROJECT_GELU,
        ),
        (
            ExpertActivation::SwiGlu,
            moxie_kernels::BF16_EXPERT_PROJECT_SILU,
        ),
    ] {
        // One expert per assignment, cycling, so the row index and the expert
        // slice are exercised independently of each other.
        let expert = 1u32;
        let rows: Vec<u32> = (0..ASSIGNMENTS as u32).collect();
        let slots: Vec<u32> = (0..ASSIGNMENTS as u32).rev().collect();

        let mut d_x = DeviceBuffer::alloc(&ctx, (ASSIGNMENTS * HIDDEN * 2) as usize)?;
        d_x.copy_from_host(&encode(&x))?;
        let gu_stride = (2 * INTERMEDIATE * HIDDEN) as usize;
        let d_stride = (HIDDEN * INTERMEDIATE) as usize;
        let mut d_gate_up = DeviceBuffer::alloc(&ctx, gu_stride * 2)?;
        d_gate_up.copy_from_host(&encode(
            &gate_up[expert as usize * gu_stride..(expert as usize + 1) * gu_stride],
        ))?;
        let mut d_down = DeviceBuffer::alloc(&ctx, d_stride * 2)?;
        d_down.copy_from_host(&encode(
            &down[expert as usize * d_stride..(expert as usize + 1) * d_stride],
        ))?;
        let mut d_rows = DeviceBuffer::alloc(&ctx, rows.len() * 4)?;
        d_rows.copy_from_host(
            &rows
                .iter()
                .flat_map(|r| r.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut d_slots = DeviceBuffer::alloc(&ctx, slots.len() * 4)?;
        d_slots.copy_from_host(
            &slots
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let d_workspace = DeviceBuffer::alloc(&ctx, (ASSIGNMENTS * INTERMEDIATE * 4) as usize)?;
        let d_out = DeviceBuffer::alloc(&ctx, (ASSIGNMENTS * HIDDEN * 2) as usize)?;

        let project = module.function(symbol)?;
        let reduce = module.function(moxie_kernels::BF16_EXPERT_DOWN)?;
        let (mut x_ptr, mut row_ptr, mut gu_ptr, mut ws_ptr) = (
            d_x.device_ptr(),
            d_rows.device_ptr(),
            d_gate_up.device_ptr(),
            d_workspace.device_ptr(),
        );
        let (mut down_ptr, mut slot_ptr, mut out_ptr) = (
            d_down.device_ptr(),
            d_slots.device_ptr(),
            d_out.device_ptr(),
        );
        let (mut count, mut hidden, mut intermediate) = (ASSIGNMENTS, HIDDEN, INTERMEDIATE);
        let block = 64u32;
        let lanes = ASSIGNMENTS * INTERMEDIATE;
        let mut params: [*mut c_void; 7] = [
            (&raw mut x_ptr).cast(),
            (&raw mut row_ptr).cast(),
            (&raw mut gu_ptr).cast(),
            (&raw mut ws_ptr).cast(),
            (&raw mut count).cast(),
            (&raw mut hidden).cast(),
            (&raw mut intermediate).cast(),
        ];
        // SAFETY: the symbol's ABI is the one declared in `expert_mlp.cu`; every
        // pointer is a live buffer of the size the kernel indexes, and the grid
        // covers exactly the element count.
        unsafe {
            project.launch_blocking(
                (lanes.div_ceil(u64::from(block)) as u32, 1, 1),
                (block, 1, 1),
                0,
                &mut params,
            )?;
        }
        let components = ASSIGNMENTS * HIDDEN;
        let mut down_params: [*mut c_void; 7] = [
            (&raw mut ws_ptr).cast(),
            (&raw mut down_ptr).cast(),
            (&raw mut slot_ptr).cast(),
            (&raw mut out_ptr).cast(),
            (&raw mut count).cast(),
            (&raw mut hidden).cast(),
            (&raw mut intermediate).cast(),
        ];
        // SAFETY: as above, for the second symbol.
        unsafe {
            reduce.launch_blocking(
                (components.div_ceil(u64::from(block)) as u32, 1, 1),
                (block, 1, 1),
                0,
                &mut down_params,
            )?;
        }
        stream.synchronize()?;

        let mut got = vec![0u8; (ASSIGNMENTS * HIDDEN * 2) as usize];
        d_out.copy_to_host(&mut got)?;
        let got = decode_u16(&got);

        let spec = route::ExpertSpec {
            experts: EXPERTS as usize,
            hidden: HIDDEN as usize,
            intermediate: INTERMEDIATE as usize,
            activation,
        };
        for (assignment, (row, slot)) in rows.iter().zip(&slots).enumerate() {
            let want = route::expert_row(
                &x[*row as usize * HIDDEN as usize..(*row as usize + 1) * HIDDEN as usize],
                &gate_up,
                &down,
                expert,
                spec,
            )
            .map_err(|e| Error::Numerical {
                detail: format!("oracle: {e}"),
            })?;
            let want: Vec<u16> = want.iter().map(|v| to_bf16_bits(*v)).collect();
            let start = *slot as usize * HIDDEN as usize;
            if got[start..start + HIDDEN as usize] != want[..] {
                return Ok(Outcome::Failed(format!(
                    "{activation:?} assignment {assignment} (row {row} -> slot {slot}) differs \
                     from the oracle"
                )));
            }
            compared += HIDDEN as usize;
        }
    }

    // The count is part of the claim: "matches the oracle" over five components
    // would be a different statement.
    println!("      grouped expert: {compared} BF16 components bit-identical");
    Ok(Outcome::Passed)
}

/// Task 0028: the shared quantized linear, qualified per architecture.
///
/// One symbol, two widths, two group rules and both zero-point modes, against
/// task 0024's decoder on the host. The weights and activations are written
/// here: this qualifies a **kernel**, and is not model support or a quality
/// claim of any kind.
///
/// The executor's own device test covers admission, residency and the memory
/// bound. What this case adds is the thing document 07 asks this lane for --
/// a per-architecture pass or a recorded absence, with no way for an SM86 run
/// to stand in for SM120.
fn affine_linear(cap: &DeviceCapability) -> Result<Outcome, Error> {
    use moxie_format::affine::{
        AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints, pack_row,
    };
    use moxie_format::scale::{ScaleDtype, ScaleValues};
    use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::AFFINE_LINEAR_FATBIN)?),
    )?;
    let kernel = module.function(moxie_kernels::AFFINE_LINEAR)?;

    let mut state = 0x0028_0028u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u32 << 24) as f32)
    };

    let mut compared = 0usize;
    for (width, group, asymmetric, rows, out_features, in_features) in [
        (IntWidth::Int4, 32u32, true, 5usize, 48usize, 100usize),
        (IntWidth::Int8, 128, false, 3, 32, 300),
    ] {
        let descriptor = AffineDescriptor {
            width,
            out_features,
            in_features,
            grouping: Grouping::Contiguous { size: group },
            group_index: None,
            scale_dtype: ScaleDtype::Bf16,
        };
        let (lo, hi) = width.code_range();
        let span = (hi - lo + 1) as usize;
        let mut codes = Vec::new();
        for o in 0..out_features {
            let row: Vec<i32> = (0..in_features)
                .map(|k| lo + ((o * in_features + k) % span) as i32)
                .collect();
            codes.extend_from_slice(&pack_row(width, &row).map_err(numerical)?);
        }
        let entries = descriptor.group_entries().map_err(numerical)?;
        let scales: Vec<u16> = (0..entries)
            .map(|_| to_bf16_bits(0.01 + next() * 0.05))
            .collect();
        let zero_points = if asymmetric {
            ZeroPoints::PerGroup(
                (0..entries)
                    .map(|i| (lo + ((i * 5) % span) as i32) as i16)
                    .collect(),
            )
        } else {
            ZeroPoints::Symmetric
        };
        let tensor = AffineTensor::new(
            descriptor,
            codes,
            ScaleValues::Bf16(scales.clone()),
            zero_points.clone(),
        )
        .map_err(numerical)?;

        let x: Vec<f32> = (0..rows * in_features)
            .map(|_| bf16_round(next() - 0.5))
            .collect();
        let mut d_x = DeviceBuffer::alloc(&ctx, rows * in_features * 2)?;
        d_x.copy_from_host(
            &x.iter()
                .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut d_codes = DeviceBuffer::alloc(&ctx, tensor.codes().len())?;
        d_codes.copy_from_host(tensor.codes())?;
        let mut d_scales = DeviceBuffer::alloc(&ctx, scales.len() * 2)?;
        d_scales.copy_from_host(
            &scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let zero_bytes: Vec<u8> = match &zero_points {
            ZeroPoints::Symmetric => Vec::new(),
            ZeroPoints::PerGroup(z) => z.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        // A symmetric tensor has **no** zero-point component. The kernel gets a
        // null pointer, not a buffer of zeros, so the absent section is absent
        // on the device too.
        let d_zero = if zero_bytes.is_empty() {
            None
        } else {
            let mut buffer = DeviceBuffer::alloc(&ctx, zero_bytes.len())?;
            buffer.copy_from_host(&zero_bytes)?;
            Some(buffer)
        };
        let d_out = DeviceBuffer::alloc(&ctx, rows * out_features * 2)?;

        let (mut x_ptr, mut code_ptr, mut scale_ptr) = (
            d_x.device_ptr(),
            d_codes.device_ptr(),
            d_scales.device_ptr(),
        );
        let mut zero_ptr = d_zero.as_ref().map_or(0u64, DeviceBuffer::device_ptr);
        // The activation-order map, absent here: these fixtures are contiguous
        // tensors. A **null pointer is the operand**, not a missing argument —
        // this case lost the parameter when task 0035 versioned the ABI to v2,
        // which bound the output buffer to `group_index` and made the kernel
        // read group identities out of its own output. The result was an
        // illegal access that poisoned the process context, so every later case
        // on every later device failed with it. `moxie-executor`'s own device
        // tests passed throughout, because the executor's binding was updated
        // and this hand-written one was not.
        let mut map_ptr = 0u64;
        let mut out_ptr = d_out.device_ptr();
        let mut rows_u = rows as u64;
        let mut in_u = in_features as u64;
        let mut out_u = out_features as u64;
        let mut stride = width.row_stride(in_features) as u64;
        let mut groups = (entries / out_features) as u64;
        let mut bits = width.bits();
        let mut group_size = group;
        let mut scale_kind = 1u32;
        let mut params: [*mut c_void; 14] = [
            (&raw mut x_ptr).cast(),
            (&raw mut code_ptr).cast(),
            (&raw mut scale_ptr).cast(),
            (&raw mut zero_ptr).cast(),
            (&raw mut map_ptr).cast(),
            (&raw mut out_ptr).cast(),
            (&raw mut rows_u).cast(),
            (&raw mut in_u).cast(),
            (&raw mut out_u).cast(),
            (&raw mut stride).cast(),
            (&raw mut groups).cast(),
            (&raw mut bits).cast(),
            (&raw mut group_size).cast(),
            (&raw mut scale_kind).cast(),
        ];
        let tile = moxie_kernels::AFFINE_LINEAR_TILE;
        // SAFETY: the symbol's ABI is the one declared in `affine_linear.cu`;
        // every pointer is a live buffer of the size the kernel indexes, and
        // the grid covers exactly the output tiles.
        unsafe {
            kernel.launch_blocking(
                (out_u.div_ceil(tile) as u32, rows_u.div_ceil(tile) as u32, 1),
                (32, 1, 1),
                0,
                &mut params,
            )?;
        }
        stream.synchronize()?;
        let mut got = vec![0u8; rows * out_features * 2];
        d_out.copy_to_host(&mut got)?;
        let got = decode_u16(&got);

        // The oracle: the whole weight reconstructed on the host through task
        // 0024's decoder, rounded to BF16, then multiplied in ascending order
        // with FP32 accumulation. It shares no code with the kernel.
        let weight = tensor.reconstruct().map_err(numerical)?;
        for m in 0..rows {
            for n in 0..out_features {
                let mut acc = 0f32;
                let mut abs = 0f32;
                for k in 0..in_features {
                    let term = x[m * in_features + k] * bf16_round(weight[n * in_features + k]);
                    acc += term;
                    abs += term.abs();
                }
                let reference = bf16_round(acc);
                let device = f32::from_bits(u32::from(got[m * out_features + n]) << 16);
                let difference = (device - reference).abs();
                // The same spacing function the attention gate uses. It was
                // duplicated here and wrong below `2^-126` in both copies; see
                // `bf16_ulp`. The correction only ever *relaxes* the bound, and
                // only for subnormal expected values, so no verdict this lane
                // has ever produced changes.
                let ulp = bf16_ulp(reference);
                // The owner's two-clause gate of 2026-09-14: 2 ULP at the
                // oracle's magnitude, or within the reduction's own resolution
                // where the result has cancelled below what BF16 can express.
                if difference > 2.0 * ulp && difference > abs / 256.0 {
                    return Err(Error::Numerical {
                        detail: format!(
                            "{} ({m},{n}): device {device} against oracle {reference}, \
                             {difference} apart; 2 ULP is {} and the reduction's \
                             resolution is {}",
                            width.profile(),
                            2.0 * ulp,
                            abs / 256.0
                        ),
                    });
                }
                compared += 1;
            }
        }
    }
    if compared != 5 * 48 + 3 * 32 {
        return Err(Error::Numerical {
            detail: format!("{compared} element(s) compared; the case checks 336"),
        });
    }
    Ok(Outcome::Passed)
}

/// One attention fixture: BF16 keys, values and queries for a paged layer.
///
/// Synthetic, and that is stated rather than implied: no checkpoint is read
/// here and no output-quality claim follows from any of it. What these cases
/// prove is that the kernel computes the attention equation the host oracle
/// defines, over paged state whose physical pages are deliberately not in
/// logical order.
struct AttentionFixture {
    geometry: PageGeometry,
    heads: u64,
    keys: Vec<u16>,
    values: Vec<u16>,
}

impl AttentionFixture {
    fn build(geometry: PageGeometry, heads: u64, rows: u64, seed: u64) -> Self {
        use moxie_kernels::cpu_expert::to_bf16_bits;
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 41) as f32) / ((1u32 << 22) as f32) - 1.0
        };
        let width = (rows * geometry.kv_heads * geometry.head_dim) as usize;
        let mut keys = Vec::with_capacity(width);
        let mut values = Vec::with_capacity(width);
        for _ in 0..width {
            keys.push(to_bf16_bits(next()));
            values.push(to_bf16_bits(next()));
        }
        Self {
            geometry,
            heads,
            keys,
            values,
        }
    }

    /// The BF16 bytes for `rows` rows starting at absolute position `first`.
    fn payload(&self, first: u64, rows: u64) -> (Vec<u8>, Vec<u8>) {
        let per_row = (self.geometry.kv_heads * self.geometry.head_dim) as usize;
        let start = first as usize * per_row;
        let end = start + rows as usize * per_row;
        let to_bytes = |v: &[u16]| v.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>();
        (
            to_bytes(&self.keys[start..end]),
            to_bytes(&self.values[start..end]),
        )
    }

    /// A query block, a fixed function of the **absolute** position, so a
    /// chunked launch and a whole one are given exactly the same numbers.
    fn query(&self, first: u64, rows: u64) -> Vec<u16> {
        use moxie_kernels::cpu_expert::to_bf16_bits;
        let mut out = Vec::with_capacity((rows * self.heads * self.geometry.head_dim) as usize);
        for row in 0..rows {
            let position = first + row;
            for head in 0..self.heads {
                for d in 0..self.geometry.head_dim {
                    let mix = (position.wrapping_mul(31) ^ head.wrapping_mul(7) ^ (d * 13)) % 61;
                    out.push(to_bf16_bits((mix as f32 - 30.0) / 24.0));
                }
            }
        }
        out
    }

    fn query_bytes(&self, first: u64, rows: u64) -> Vec<u8> {
        self.query(first, rows)
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect()
    }

    fn head_slice(&self, source: &[u16], row: u64, kv_head: u64) -> Vec<f32> {
        let per_row = (self.geometry.kv_heads * self.geometry.head_dim) as usize;
        let start = row as usize * per_row + (kv_head * self.geometry.head_dim) as usize;
        source[start..start + self.geometry.head_dim as usize]
            .iter()
            .map(|b| bf16_value(*b))
            .collect()
    }
}

/// A page table whose physical pages are deliberately not in logical order.
///
/// An identity mapping proves nothing: every offset would be right even if the
/// table were ignored. This one reverses the order and is validated by
/// `publish_page_table`, so a kernel that quietly assumed contiguity fails.
fn shuffled_pages(pages: u64) -> Vec<u32> {
    (0..pages).rev().map(|p| p as u32).collect()
}

/// The placements a mapping implies for `rows` rows from `first`.
///
/// These cases hand the run a **reversed** table and then place rows through
/// it, which is the property they exist to check: the run performs whatever
/// mapping it is given and assumes nothing about it. The authority-driven path
/// — where `moxie_state::DeviceKvSequence` produces both the table and the
/// placements — is what `paged_attention_32k` and the executor's own device
/// tests cover.
fn placements_through(
    table: &[u32],
    page_tokens: u64,
    first: u64,
    rows: u64,
) -> Result<Vec<PagePlacement>, Error> {
    let mut out = Vec::new();
    let mut done = 0;
    while done < rows {
        let position = first + done;
        let slot = position % page_tokens;
        let run = (page_tokens - slot).min(rows - done);
        let logical = (position / page_tokens) as usize;
        let physical = *table.get(logical).ok_or(Error::InvalidRequest {
            field: "page_table",
            detail: "the mapping does not cover this row".into(),
        })?;
        out.push(PagePlacement {
            position,
            physical_page: u64::from(physical),
            slot,
            rows: run,
        });
        done += run;
    }
    Ok(out)
}

/// Compare one launch's device output against the independent FP64 equation.
///
/// The expected value comes from `moxie_oracles::online_softmax`, cut into
/// blocks of a width the kernel does not use, so the two agree on the *answer*
/// rather than on a schedule. The tolerance is the predeclared
/// `attention_error_bound` at the layer's declared scale, plus half a BF16 ulp
/// for the one output narrowing the descriptor already declares. That addition
/// is the rounding boundary, not a widening of the bound: ADR 0028's clauses
/// are untouched.
fn check_attention(
    fixture: &AttentionFixture,
    launch: &PagedAttentionLaunch,
    output: &[u8],
    label: &str,
) -> Result<moxie_oracles::metric::ErrorSummary, Error> {
    use moxie_oracles::attention::attention_error_bounds_at;
    use moxie_oracles::online_softmax::attend_row_blocked;

    let head_dim = fixture.geometry.head_dim as usize;
    let group = fixture.heads / fixture.geometry.kv_heads;
    let got = decode_u16(output);
    let block = fixture.query(launch.first_position(), launch.rows());
    let mut device_values: Vec<f32> = Vec::new();
    let mut oracle_values: Vec<f64> = Vec::new();
    for row in 0..launch.rows() {
        let visible: Vec<u64> = (launch.history_base()
            ..launch.history_base() + launch.history_rows())
            .filter(|key| launch.allows(row, *key))
            .collect();
        if visible.is_empty() {
            return Err(Error::Numerical {
                detail: format!("{label}: row {row} sees nothing, which this gate cannot check"),
            });
        }
        for head in 0..fixture.heads {
            let kv_head = head / group;
            let start = ((row * fixture.heads + head) * fixture.geometry.head_dim) as usize;
            let query_row: Vec<f32> = block[start..start + head_dim]
                .iter()
                .map(|b| bf16_value(*b))
                .collect();
            let keys: Vec<Vec<f32>> = visible
                .iter()
                .map(|k| fixture.head_slice(&fixture.keys, *k, kv_head))
                .collect();
            let values: Vec<Vec<f32>> = visible
                .iter()
                .map(|k| fixture.head_slice(&fixture.values, *k, kv_head))
                .collect();
            let key_views: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
            let value_views: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
            let allowed = vec![true; visible.len()];
            let want = attend_row_blocked(
                &query_row,
                &key_views,
                &value_views,
                &allowed,
                launch.scale(),
                // Neither the kernel's tile nor the page width.
                37,
            )
            .map_err(|e| Error::Numerical {
                detail: format!("{label}: the oracle refused: {e}"),
            })?;
            // Every component's bound in one pass. The per-component entry
            // point recomputes the score-error term for each lane, which at
            // the 32,768-row gate is four billion operations per head; the two
            // are asserted bitwise equal in the oracle's own fixtures.
            let bounds =
                attention_error_bounds_at(&query_row, &key_views, &value_views, launch.scale())
                    .map_err(|e| Error::Numerical {
                        detail: format!("{label}: the bound refused: {e}"),
                    })?;
            for d in 0..head_dim {
                let device = bf16_value(got[start + d]);
                if !device.is_finite() {
                    return Err(Error::Numerical {
                        detail: format!("{label}: row {row} head {head} lane {d} is {device}"),
                    });
                }
                let bound = bounds[d];
                let difference = (f64::from(device) - want[d]).abs();
                let half_ulp = 0.5 * f64::from(bf16_ulp(want[d] as f32));
                if difference > bound + half_ulp {
                    return Err(Error::Numerical {
                        detail: format!(
                            "{label}: row {row} head {head} lane {d}: device {device} against \
                             oracle {}, {difference:.3e} apart; the bound is {bound:.3e} and \
                             the BF16 boundary is {half_ulp:.3e}",
                            want[d]
                        ),
                    });
                }
                device_values.push(device);
                oracle_values.push(want[d]);
            }
        }
    }
    Ok(moxie_oracles::metric::ErrorSummary::absolute(
        &device_values,
        &oracle_values,
    ))
}

/// One BF16 ulp at this magnitude.
///
/// BF16 carries eight significand bits, so at a value whose F32 exponent field
/// is `e` the spacing is `2^(e-134)` — `f32::from_bits((e - 7) << 23)` while
/// that is a normal F32, and an F32 **subnormal** below it. The floor is BF16's
/// own smallest subnormal, `2^-133`, which is `f32::from_bits(1 << 16)`.
///
/// Returning the F32 subnormal floor (`f32::from_bits(1)`, `2^-149`) for every
/// exponent field at or below seven would be sixteen binades too small — wrong
/// in the strict direction. That cannot make a gate accept a wrong answer, but
/// it can make one reject a correct kernel whose expected value is subnormal —
/// exactly the region document 07 asks to be stressed rather than avoided.
fn bf16_ulp(value: f32) -> f32 {
    let exponent = (value.abs().to_bits() >> 23) & 0xFF;
    match exponent {
        // The expected value is itself an F32 subnormal: no BF16 spacing is
        // finer than BF16's own smallest subnormal.
        0 => f32::from_bits(1 << 16),
        // `2^(e-134)`, written as the F32 subnormal it is.
        1..=7 => f32::from_bits(1 << (15 + exponent)),
        _ => f32::from_bits((exponent - 7) << 23),
    }
}

/// Admit a ledger with this device's measured capacity and the host's.
fn measured_ledger(ctx: &RankContext) -> Result<Ledger, Error> {
    let measurement = ctx.measure()?;
    let snapshot = CapacitySnapshot::measured(&measurement, 1 << 20)?;
    let host = moxie_host::read()?;
    let host_snapshot = CapacitySnapshot::measured_host(&host, 1 << 20)?;
    Ledger::new([snapshot, host_snapshot])
}

/// Task 0037: paged attention over persistent admitted device state.
///
/// Head dimensions 64 and 128, multi-head and a 4:1 grouped ratio, full causal
/// and sliding visibility, whole and chunked prefill, single-row decode, page
/// tails and a page table that is not the identity. Every component is checked
/// against the FP64 oracle under the predeclared bound.
fn paged_attention(cap: &DeviceCapability) -> Result<Outcome, Error> {
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();

    struct Case {
        label: &'static str,
        geometry: PageGeometry,
        heads: u64,
        scale: f32,
        visibility: Visibility,
        rows: u64,
        history: u64,
        first_position: u64,
        /// Appends to build the history with, in order. They must sum to it.
        appends: &'static [u64],
    }

    impl Case {
        fn layer(&self) -> AttentionLayer {
            AttentionLayer {
                geometry: self.geometry,
                heads: self.heads,
                scale: self.scale,
                visibility: self.visibility,
            }
        }
    }

    let cases = [
        // Multi-head, head dimension 64, a history that ends mid-page, and a
        // whole-prefill launch from position zero.
        Case {
            label: "mha-64-whole-prefill",
            geometry: PageGeometry {
                kv_heads: 4,
                head_dim: 64,
                page_tokens: 16,
                pages: 4,
            },
            heads: 4,
            scale: moxie_plan::reciprocal_sqrt_scale(64),
            visibility: Visibility::Causal,
            rows: 40,
            history: 40,
            first_position: 0,
            appends: &[16, 9, 15],
        },
        // Grouped 4:1, head dimension 128, a single decode row at the end of a
        // history whose last page holds four rows.
        Case {
            label: "gqa-128-decode",
            geometry: PageGeometry {
                kv_heads: 2,
                head_dim: 128,
                page_tokens: 32,
                pages: 4,
            },
            heads: 8,
            scale: moxie_plan::reciprocal_sqrt_scale(128),
            visibility: Visibility::Causal,
            rows: 1,
            history: 100,
            first_position: 99,
            appends: &[1, 63, 36],
        },
        // The widest head dimension the catalogue advertises. Declaring a
        // shape domain and qualifying a subset of it is the gap document 07
        // calls out by name, so the boundary of the claim is measured rather
        // than assumed.
        Case {
            label: "mha-256-widest-declared",
            geometry: PageGeometry {
                kv_heads: 1,
                head_dim: moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM,
                page_tokens: 8,
                pages: 4,
            },
            heads: 2,
            scale: moxie_plan::reciprocal_sqrt_scale(moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM),
            visibility: Visibility::Causal,
            rows: 3,
            history: 29,
            first_position: 26,
            appends: &[8, 12, 9],
        },
        // A head dimension the 32-lane loop genuinely cannot divide: 100 is
        // three components for most lanes and four for the first four, so the
        // remainder path is exercised rather than described. 96 is 3x32 and
        // would not exercise it, which is why 100 is the number here.
        Case {
            label: "gqa-100-remainder",
            geometry: PageGeometry {
                kv_heads: 2,
                head_dim: 100,
                page_tokens: 16,
                pages: 3,
            },
            heads: 6,
            scale: 1.0,
            visibility: Visibility::Causal,
            rows: 2,
            history: 33,
            first_position: 31,
            appends: &[16, 1, 16],
        },
        // A remainder **and** a partially used second accumulator slot: 200 is
        // 6x32+8 across the lane loop, and 72 of the block's 128 threads own a
        // second output component while the rest own one. Neither the 100-wide
        // case (one slot) nor the 256-wide one (two full slots) reaches that.
        Case {
            label: "mha-200-partial-slot",
            geometry: PageGeometry {
                kv_heads: 1,
                head_dim: 200,
                page_tokens: 8,
                pages: 3,
            },
            heads: 2,
            scale: moxie_plan::reciprocal_sqrt_scale(200),
            visibility: Visibility::SlidingWindow { window: 12 },
            rows: 2,
            history: 21,
            first_position: 19,
            appends: &[8, 13],
        },
        // A sliding window with a declared scale of exactly 1.0 -- the
        // Gemma-shaped layer the bound repair was about -- over a history whose
        // early pages are entirely outside the window.
        Case {
            label: "sliding-20-scale-one",
            geometry: PageGeometry {
                kv_heads: 1,
                head_dim: 64,
                page_tokens: 8,
                pages: 9,
            },
            heads: 4,
            scale: 1.0,
            visibility: Visibility::SlidingWindow { window: 20 },
            rows: 5,
            history: 71,
            first_position: 66,
            appends: &[8, 40, 23],
        },
    ];

    let mut checked = 0usize;
    for case in &cases {
        let fixture = AttentionFixture::build(case.geometry, case.heads, case.history, 0x0037_0001);
        let launch = PagedAttentionLaunch::new(
            case.layer(),
            case.rows,
            case.first_position,
            0,
            case.history,
        )?;
        let descriptor = select_paged_attention_kernel(&catalogue, cap, &launch)?;
        let mut ledger = measured_ledger(&ctx)?;
        let run = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor,
            case.geometry,
            case.heads,
            case.rows,
            Staging::Host,
        )
        .map_err(|r| r.error)?;
        // No state authority anywhere in this gate: the shuffled table below
        // is deliberately not one any retention policy would ever produce --
        // stressing the kernel against the FP64 oracle is not a state
        // decision -- so this drives the run directly through
        // `RawPagedFixture` rather than the authority's writer.
        let mut run = moxie_executor::paged_attention::device::RawPagedFixture::new(run);
        let table = shuffled_pages(case.geometry.pages);
        run.publish_page_table(&stream, 0, table.clone())
            .map_err(|r| r.error)?;

        let mut written = 0u64;
        for rows in case.appends {
            let (keys, values) = fixture.payload(written, *rows);
            let placements = placements_through(&table, case.geometry.page_tokens, written, *rows)?;
            run.write_rows(&stream, &placements, keys, values)
                .map_err(|r| r.error)?;
            written += rows;
        }
        if run.run().written_rows() != case.history {
            return Ok(Outcome::Failed(format!(
                "{}: wrote {} row(s) of {}",
                case.label,
                run.run().written_rows(),
                case.history
            )));
        }

        // An append that cannot fit is refused *before* the device is touched:
        // the frontier does not move and every prior byte is unchanged.
        let whole_history = placements_through(&table, case.geometry.page_tokens, 0, case.history)?;
        let before = run.run().read_rows(&whole_history)?;
        // A placement naming a page this run was never admitted for. Refused
        // before the device is touched, with the rows handed back.
        let outside = vec![PagePlacement {
            position: case.history,
            physical_page: case.geometry.pages,
            slot: 0,
            rows: 1,
        }];
        let (keys, values) = fixture.payload(0, 1);
        let refused = run
            .write_rows(&stream, &outside, keys, values)
            .expect_err("a placement outside the admitted pages must be refused");
        if refused.retained_source() {
            return Ok(Outcome::Failed(format!(
                "{}: a refusal before enqueue kept the caller's rows",
                case.label
            )));
        }
        if run.run().written_rows() != case.history
            || run.run().read_rows(&whole_history)? != before
        {
            return Ok(Outcome::Failed(format!(
                "{}: a refused write moved the high-water mark or changed written bytes",
                case.label
            )));
        }

        // Done writing directly: hand the run back to attend and close
        // through its own public API, which never needed narrowing.
        let mut run = run.into_inner();
        let whole = run
            .attend(
                &stream,
                &launch,
                fixture.query_bytes(case.first_position, case.rows),
            )
            .map_err(|r| r.error)?;
        let summary = check_attention(&fixture, &launch, &whole, case.label)?;
        println!(
            "    {} {} heads={} kv={} d={} pages={}x{} state={} B {summary}",
            cap.sm(),
            case.label,
            case.heads,
            case.geometry.kv_heads,
            case.geometry.head_dim,
            case.geometry.pages,
            case.geometry.page_tokens,
            run.arena_bytes()
        );
        checked += summary.count;

        // The same rows, cut into uneven chunks. Every query row is
        // independent of how the launch was cut, so the bytes must be
        // identical -- not merely close.
        if case.rows > 1 {
            let mut chunked: Vec<u8> = Vec::new();
            let mut done = 0u64;
            for width in [1u64, 2, 3] {
                if done >= case.rows {
                    break;
                }
                let rows = width.min(case.rows - done);
                let chunk_launch = launch.at(rows, case.first_position + done)?;
                let out = run
                    .attend(
                        &stream,
                        &chunk_launch,
                        fixture.query_bytes(chunk_launch.first_position(), rows),
                    )
                    .map_err(|r| r.error)?;
                check_attention(&fixture, &chunk_launch, &out, case.label)?;
                chunked.extend_from_slice(&out);
                done += rows;
            }
            let remaining = case.rows - done;
            if remaining > 0 {
                let chunk_launch = launch.at(remaining, case.first_position + done)?;
                let out = run
                    .attend(
                        &stream,
                        &chunk_launch,
                        fixture.query_bytes(chunk_launch.first_position(), remaining),
                    )
                    .map_err(|r| r.error)?;
                chunked.extend_from_slice(&out);
            }
            if chunked != whole {
                return Ok(Outcome::Failed(format!(
                    "{}: chunked prefill differs from whole prefill",
                    case.label
                )));
            }
        }

        // A launch declaring more history than was committed is refused rather
        // than attending over pages nothing wrote.
        // One row past the frontier. The launch itself is legal -- the
        // geometry admits it -- and the run must refuse it because those rows
        // were never committed.
        let beyond = launch.over(0, case.history + 1)?;
        if run
            .attend(
                &stream,
                &beyond,
                fixture.query_bytes(case.first_position, case.rows),
            )
            .is_ok()
        {
            return Ok(Outcome::Failed(format!(
                "{}: a launch past the frontier was accepted",
                case.label
            )));
        }
        run.close(&mut ledger).map_err(|r| r.error)?;
        if !ledger.outstanding().is_empty() {
            return Ok(Outcome::Failed(format!(
                "{}: a closed run left bytes charged",
                case.label
            )));
        }
    }
    if checked == 0 {
        return Ok(Outcome::Failed("no component was compared".into()));
    }
    Ok(Outcome::Passed)
}

#[derive(Clone, Copy)]
struct IndirectAttentionCase {
    label: &'static str,
    geometry: PageGeometry,
    heads: u64,
    scale: f32,
    window: u32,
    rows: u64,
    history: u64,
    first_position: u64,
}

fn attention_page_images(
    fixture: &AttentionFixture,
    table: &[u32],
    history: u64,
) -> (Vec<u8>, Vec<u8>) {
    let row_bytes = (fixture.geometry.kv_heads * fixture.geometry.head_dim * 2) as usize;
    let page_bytes = fixture.geometry.page_tokens as usize * row_bytes;
    let image_bytes = fixture.geometry.pages as usize * page_bytes;
    let mut key_pages = vec![0; image_bytes];
    let mut value_pages = vec![0; image_bytes];
    let (keys, values) = fixture.payload(0, history);
    for row in 0..history as usize {
        let physical = table[row / fixture.geometry.page_tokens as usize] as usize;
        let slot = row % fixture.geometry.page_tokens as usize;
        let source = row * row_bytes;
        let destination = physical * page_bytes + slot * row_bytes;
        key_pages[destination..destination + row_bytes]
            .copy_from_slice(&keys[source..source + row_bytes]);
        value_pages[destination..destination + row_bytes]
            .copy_from_slice(&values[source..source + row_bytes]);
    }
    (key_pages, value_pages)
}

fn words_u64_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn words_u32_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn paged_attention_indirect(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const BASELINE_SHA256: [(&str, &str); 3] = [
        (
            "gqa-128-decode",
            "1469fabdc480f04b00e5edb73aff3efebe102d55b298e4bfc860d12df1e7d8bd",
        ),
        (
            "gqa-128-prefill-chunk",
            "01022cb71c03d274ab38d44e79a5b4ae642c123843b540d5dbd75ecbe803707c",
        ),
        (
            "sliding-20-scale-one",
            "60c641071b498e5771677b69be0f3b5159b6c66750a2c99e24b938b028d0d716",
        ),
    ];
    const APPEND_MAX_ROWS: u32 = 16;

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let module = Module::load(
        &ctx,
        ModuleImage::Binary(smoke_image(moxie_kernels::PAGED_ATTENTION_FATBIN)?),
    )?;
    let v1 = module.function(moxie_kernels::PAGED_ATTENTION)?;
    let indirect = module.function(moxie_kernels::PAGED_ATTENTION_INDIRECT)?;
    let append = module.function(moxie_kernels::KV_APPEND_INDIRECT)?;

    let cases = [
        IndirectAttentionCase {
            label: "gqa-128-decode",
            geometry: PageGeometry {
                kv_heads: 2,
                head_dim: 128,
                page_tokens: 32,
                pages: 4,
            },
            heads: 8,
            scale: moxie_plan::reciprocal_sqrt_scale(128),
            window: 0,
            rows: 1,
            history: 100,
            first_position: 99,
        },
        IndirectAttentionCase {
            label: "gqa-128-prefill-chunk",
            geometry: PageGeometry {
                kv_heads: 2,
                head_dim: 128,
                page_tokens: 32,
                pages: 4,
            },
            heads: 8,
            scale: moxie_plan::reciprocal_sqrt_scale(128),
            window: 0,
            rows: 8,
            history: 100,
            first_position: 92,
        },
        IndirectAttentionCase {
            label: "sliding-20-scale-one",
            geometry: PageGeometry {
                kv_heads: 1,
                head_dim: 64,
                page_tokens: 8,
                pages: 9,
            },
            heads: 4,
            scale: 1.0,
            window: 20,
            rows: 5,
            history: 71,
            first_position: 66,
        },
    ];

    for case in &cases {
        let fixture = AttentionFixture::build(case.geometry, case.heads, case.history, 0x0037_0001);
        let table = shuffled_pages(case.geometry.pages);
        let (key_bytes, value_bytes) = attention_page_images(&fixture, &table, case.history);
        let table_bytes = words_u32_bytes(&table);
        let query_bytes = fixture.query_bytes(case.first_position, case.rows);
        let output_bytes = (case.rows * case.heads * case.geometry.head_dim * 2) as usize;
        let padded_bytes =
            (u64::from(APPEND_MAX_ROWS) * case.heads * case.geometry.head_dim * 2) as usize;

        let mut query = DeviceBuffer::alloc(&ctx, query_bytes.len())?;
        query.copy_from_host(&query_bytes)?;
        let mut key_pages = DeviceBuffer::alloc(&ctx, key_bytes.len())?;
        key_pages.copy_from_host(&key_bytes)?;
        let mut value_pages = DeviceBuffer::alloc(&ctx, value_bytes.len())?;
        value_pages.copy_from_host(&value_bytes)?;
        let mut page_table = DeviceBuffer::alloc(&ctx, table_bytes.len())?;
        page_table.copy_from_host(&table_bytes)?;
        let reference_output = DeviceBuffer::alloc(&ctx, output_bytes)?;
        let mut reference_ptr = reference_output.device_ptr();
        let mut query_ptr = query.device_ptr();
        let mut key_ptr = key_pages.device_ptr();
        let mut value_ptr = value_pages.device_ptr();
        let mut table_ptr = page_table.device_ptr();
        let mut rows = case.rows;
        let mut first_position = case.first_position;
        let mut history_base = 0u64;
        let mut history_rows = case.history;
        let mut heads = case.heads as u32;
        let mut kv_heads = case.geometry.kv_heads as u32;
        let mut head_dim = case.geometry.head_dim as u32;
        let mut page_tokens = case.geometry.page_tokens as u32;
        let mut window = case.window;
        let mut scale = case.scale;
        let mut reference_params: [*mut c_void; 15] = [
            (&raw mut query_ptr).cast(),
            (&raw mut key_ptr).cast(),
            (&raw mut value_ptr).cast(),
            (&raw mut table_ptr).cast(),
            (&raw mut reference_ptr).cast(),
            (&raw mut rows).cast(),
            (&raw mut first_position).cast(),
            (&raw mut history_base).cast(),
            (&raw mut history_rows).cast(),
            (&raw mut heads).cast(),
            (&raw mut kv_heads).cast(),
            (&raw mut head_dim).cast(),
            (&raw mut page_tokens).cast(),
            (&raw mut window).cast(),
            (&raw mut scale).cast(),
        ];
        // SAFETY: the parameter order and types match the unchanged v1 ABI;
        // every buffer covers the declared geometry and the blocking launch
        // observes completion before any buffer can drop.
        unsafe {
            v1.launch_blocking(
                (case.rows as u32, case.heads as u32, 1),
                (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                0,
                &mut reference_params,
            )?;
        }
        let mut reference = vec![0; output_bytes];
        reference_output.copy_to_host(&mut reference)?;
        let hash = moxie_format::sha256_hex(&reference);
        let expected = BASELINE_SHA256
            .iter()
            .find(|(label, _)| *label == case.label)
            .map(|(_, hash)| *hash)
            .ok_or_else(|| Error::InvalidRequest {
                field: "baseline",
                detail: format!("no pre-refactor bytes recorded for {}", case.label),
            })?;
        if hash != expected {
            return Ok(Outcome::Failed(format!(
                "{}: v1 output SHA-256 {hash} differs from captured {expected}",
                case.label
            )));
        }

        let step_values = [case.rows, case.first_position, 0, case.history];
        let step_bytes = words_u64_bytes(&step_values);
        let mut step = DeviceBuffer::alloc(&ctx, step_bytes.len())?;
        step.copy_from_host(&step_bytes)?;
        let indirect_output = DeviceBuffer::alloc(&ctx, output_bytes)?;
        let mut output_ptr = indirect_output.device_ptr();
        let mut step_ptr = step.device_ptr();
        let mut indirect_params: [*mut c_void; 12] = [
            (&raw mut query_ptr).cast(),
            (&raw mut key_ptr).cast(),
            (&raw mut value_ptr).cast(),
            (&raw mut table_ptr).cast(),
            (&raw mut output_ptr).cast(),
            (&raw mut step_ptr).cast(),
            (&raw mut heads).cast(),
            (&raw mut kv_heads).cast(),
            (&raw mut head_dim).cast(),
            (&raw mut page_tokens).cast(),
            (&raw mut window).cast(),
            (&raw mut scale).cast(),
        ];
        // SAFETY: the parameter order matches the indirect ABI, step contains
        // four u64 values, and the exact-row grid covers every query block.
        unsafe {
            indirect.launch_blocking(
                (case.rows as u32, case.heads as u32, 1),
                (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                0,
                &mut indirect_params,
            )?;
        }
        let mut got = vec![0; output_bytes];
        indirect_output.copy_to_host(&mut got)?;
        if got != reference {
            return Ok(Outcome::Failed(format!(
                "{}: indirect output differs from v1",
                case.label
            )));
        }

        let sentinel = vec![0xa5; padded_bytes];
        let mut padded_output = DeviceBuffer::alloc(&ctx, padded_bytes)?;
        padded_output.copy_from_host(&sentinel)?;
        output_ptr = padded_output.device_ptr();
        // SAFETY: as above, with a max_rows grid; the kernel returns before
        // reading query/output rows at or beyond step[0].
        unsafe {
            indirect.launch_blocking(
                (APPEND_MAX_ROWS, case.heads as u32, 1),
                (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                0,
                &mut indirect_params,
            )?;
        }
        let mut padded = vec![0; padded_bytes];
        padded_output.copy_to_host(&mut padded)?;
        if padded[..output_bytes] != reference || padded[output_bytes..] != sentinel[output_bytes..]
        {
            return Ok(Outcome::Failed(format!(
                "{}: max_rows launch changed output bytes or rows beyond step[0]",
                case.label
            )));
        }
        println!(
            "    {} {} v1_sha256={} indirect=byte-identical max_rows={APPEND_MAX_ROWS}",
            cap.uuid, case.label, hash
        );
    }

    struct AppendCase {
        label: &'static str,
        geometry: PageGeometry,
        first: u64,
        rows: u64,
        aligned: bool,
    }
    let append_cases = [
        AppendCase {
            label: "aligned-row",
            geometry: PageGeometry {
                kv_heads: 2,
                head_dim: 128,
                page_tokens: 32,
                pages: 4,
            },
            first: 0,
            rows: 3,
            aligned: true,
        },
        AppendCase {
            label: "misaligned-row",
            geometry: PageGeometry {
                kv_heads: 1,
                head_dim: 70,
                page_tokens: 4,
                pages: 3,
            },
            first: 2,
            rows: 3,
            aligned: false,
        },
    ];
    for case in &append_cases {
        let fixture = AttentionFixture::build(
            case.geometry,
            case.geometry.kv_heads,
            case.first + case.rows,
            0x0037_0001,
        );
        let table = shuffled_pages(case.geometry.pages);
        let placements =
            placements_through(&table, case.geometry.page_tokens, case.first, case.rows)?;
        let (keys, values) = fixture.payload(case.first, case.rows);
        let row_bytes = case.geometry.kv_heads * case.geometry.head_dim * 2;
        let page_bytes = case.geometry.page_tokens * row_bytes;
        let image_bytes = (case.geometry.pages * page_bytes) as usize;
        if (row_bytes % 16 == 0) != case.aligned {
            return Ok(Outcome::Failed(format!(
                "{}: row_bytes={row_bytes} does not match the alignment fixture",
                case.label
            )));
        }

        let mut source_keys = DeviceBuffer::alloc(&ctx, keys.len())?;
        source_keys.copy_from_host(&keys)?;
        let mut source_values = DeviceBuffer::alloc(&ctx, values.len())?;
        source_values.copy_from_host(&values)?;
        let zeros = vec![0; image_bytes];
        let mut reference_keys = DeviceBuffer::alloc(&ctx, image_bytes)?;
        reference_keys.copy_from_host(&zeros)?;
        let mut reference_values = DeviceBuffer::alloc(&ctx, image_bytes)?;
        reference_values.copy_from_host(&zeros)?;
        let mut indirect_keys = DeviceBuffer::alloc(&ctx, image_bytes)?;
        indirect_keys.copy_from_host(&zeros)?;
        let mut indirect_values = DeviceBuffer::alloc(&ctx, image_bytes)?;
        indirect_values.copy_from_host(&zeros)?;

        let mut done = 0u64;
        let mut offsets = Vec::with_capacity((case.rows * 2) as usize);
        for placement in &placements {
            let destination =
                (placement.physical_page * case.geometry.page_tokens + placement.slot) * row_bytes;
            let source = done * row_bytes;
            let bytes = placement.rows * row_bytes;
            // SAFETY: all source/destination ranges are within their allocations;
            // both allocations and the stream stay live through synchronization.
            let key_copy = unsafe {
                reference_keys.copy_from_device_async_at(
                    destination as usize,
                    &source_keys,
                    source as usize,
                    bytes as usize,
                    &stream,
                )
            };
            // SAFETY: the value source and destination are within their
            // allocations and remain live with the stream through sync.
            let value_copy = unsafe {
                reference_values.copy_from_device_async_at(
                    destination as usize,
                    &source_values,
                    source as usize,
                    bytes as usize,
                    &stream,
                )
            };
            if let Err(error) = key_copy.and(value_copy) {
                let _ = stream.synchronize();
                return Err(error);
            }
            for offset in 0..placement.rows {
                let row_offset = destination + offset * row_bytes;
                offsets.push(row_offset);
                offsets.push(row_offset);
            }
            done += placement.rows;
        }
        stream.synchronize()?;

        let step_values = [case.rows, 0, 0, 0];
        let step_bytes = words_u64_bytes(&step_values);
        let offset_bytes = words_u64_bytes(&offsets);
        let mut step = DeviceBuffer::alloc(&ctx, step_bytes.len())?;
        step.copy_from_host(&step_bytes)?;
        let mut device_offsets = DeviceBuffer::alloc(&ctx, offset_bytes.len())?;
        device_offsets.copy_from_host(&offset_bytes)?;
        if case.aligned {
            if row_bytes % 16 != 0
                || source_keys.device_ptr() % 16 != 0
                || source_values.device_ptr() % 16 != 0
                || offsets.iter().any(|offset| offset % 16 != 0)
            {
                return Ok(Outcome::Failed(format!(
                    "{}: the aligned fixture did not align every row address",
                    case.label
                )));
            }
        } else if row_bytes % 16 == 0
            || (source_keys.device_ptr() + row_bytes) % 16 == 0
            || (source_values.device_ptr() + row_bytes) % 16 == 0
            || (indirect_keys.device_ptr() + offsets[0]) % 16 == 0
        {
            return Ok(Outcome::Failed(format!(
                "{}: the fixture failed to exercise an unaligned row address",
                case.label
            )));
        }

        let mut key_source_ptr = source_keys.device_ptr();
        let mut value_source_ptr = source_values.device_ptr();
        let mut key_pages_ptr = indirect_keys.device_ptr();
        let mut value_pages_ptr = indirect_values.device_ptr();
        let mut step_ptr = step.device_ptr();
        let mut offsets_ptr = device_offsets.device_ptr();
        let mut row_bytes_arg = row_bytes;
        let mut params: [*mut c_void; 7] = [
            (&raw mut key_source_ptr).cast(),
            (&raw mut value_source_ptr).cast(),
            (&raw mut key_pages_ptr).cast(),
            (&raw mut value_pages_ptr).cast(),
            (&raw mut step_ptr).cast(),
            (&raw mut offsets_ptr).cast(),
            (&raw mut row_bytes_arg).cast(),
        ];
        // SAFETY: the launch follows the completed D2D reference copies; source,
        // offset and page buffers cover every row selected by step[0].
        unsafe {
            append.launch_blocking(
                (APPEND_MAX_ROWS, 1, 1),
                (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                0,
                &mut params,
            )?;
        }
        let mut expected_keys = vec![0; image_bytes];
        let mut expected_values = vec![0; image_bytes];
        let mut got_keys = vec![0; image_bytes];
        let mut got_values = vec![0; image_bytes];
        reference_keys.copy_to_host(&mut expected_keys)?;
        reference_values.copy_to_host(&mut expected_values)?;
        indirect_keys.copy_to_host(&mut got_keys)?;
        indirect_values.copy_to_host(&mut got_values)?;
        if expected_keys != got_keys || expected_values != got_values {
            return Ok(Outcome::Failed(format!(
                "{}: indirect append changed page bytes",
                case.label
            )));
        }
        println!(
            "    {} {} row_bytes={row_bytes} pages=byte-identical",
            cap.uuid, case.label
        );
    }
    Ok(Outcome::Passed)
}

/// Task 0041: one resident page plus one host page, with the device partial
/// producer kept separate from the accepted single-shot kernel.
///
/// The host page is read from `moxie_state::PagedSequence`, the resident page
/// is committed through `DeviceKvSequence`. The executor returns the two raw
/// device partials; this composition root widens them and calls the shared
/// `Partial::merge`. The raw merge is checked against
/// `attention_error_bounds_at` without a BF16 allowance. The cross-path check
/// first applies the declared BF16 output boundary to that raw result, then
/// compares like representations with the same per-lane bound.
fn paged_attention_host_streaming(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 64;
    const PAGE_TOKENS: u64 = 8;
    const RESIDENT_ROWS: u64 = PAGE_TOKENS;
    const STAGED_ROWS: u64 = PAGE_TOKENS;
    const TOTAL_ROWS: u64 = RESIDENT_ROWS + STAGED_ROWS;
    let geometry = PageGeometry {
        kv_heads: 2,
        head_dim: HEAD_DIM,
        page_tokens: PAGE_TOKENS,
        pages: 2,
    };
    let scale = moxie_plan::reciprocal_sqrt_scale(HEAD_DIM);
    let fixture = AttentionFixture::build(geometry, HEADS, TOTAL_ROWS, 0x0041_0001);
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();

    let stream_geometry = PageGeometry {
        pages: 1,
        ..geometry
    };
    let stream_layer = AttentionLayer {
        geometry: stream_geometry,
        heads: HEADS,
        scale,
        visibility: Visibility::Causal,
    };
    let stream_probe = PagedAttentionLaunch::new(stream_layer, 1, 0, 0, 1)?;
    let stream_descriptor = select_paged_attention_kernel(&catalogue, cap, &stream_probe)?;

    let mut ledger = measured_ledger(&ctx)?;
    let host_geometry = moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: TOTAL_ROWS as usize,
        tentative_rows: TOTAL_ROWS as usize,
    };
    let mut host = moxie_state::PagedSequence::new(&mut ledger, host_geometry)?;
    let host_txn = host.begin()?;
    host.append_prompt(TOTAL_ROWS)?;
    for position in 0..TOTAL_ROWS {
        let (keys, values) = fixture.payload(position, 1);
        let rows = [moxie_state::KvRow {
            key: &keys,
            value: &values,
        }];
        host.append(
            host_txn,
            position,
            &rows,
            &std::sync::atomic::AtomicBool::new(false),
        )?;
    }
    host.commit_prefix(host_txn, 0)?;

    let mut streamed_run = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        stream_descriptor,
        stream_geometry,
        HEADS,
        RESIDENT_ROWS,
        RESIDENT_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::TwoBlock,
    )
    .map_err(|refused| refused.error)?;
    let mut resident_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: RESIDENT_ROWS as usize,
        tentative_rows: RESIDENT_ROWS as usize,
    })?;
    append_authority_rows(
        &mut resident_state,
        &mut streamed_run,
        &stream,
        &fixture,
        RESIDENT_ROWS,
    )?;
    let (host_keys, host_values) = host.read_block(0, RESIDENT_ROWS, STAGED_ROWS)?;
    let expected_transfer = host_keys
        .len()
        .checked_add(host_values.len())
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(moxie_types::DimError::Overflow)? as u64;
    let stream_launch =
        PagedAttentionLaunch::two_block_stream(stream_layer, 1, TOTAL_ROWS - 1, 0, STAGED_ROWS)?;
    let streamed = streamed_run
        .attend_two_block(
            &stream,
            &stream_launch,
            fixture.query_bytes(TOTAL_ROWS - 1, 1),
            host_keys,
            host_values,
        )
        .map_err(|refused| refused.error)?;
    if streamed.host_to_device_bytes != expected_transfer {
        return Ok(Outcome::Failed(format!(
            "{} streamed {} B but the staged block transferred {} B",
            cap.uuid, streamed.host_to_device_bytes, expected_transfer
        )));
    }

    // The comparison uses the same independent FP64 equation and unchanged
    // attention_error_bound gate as the accepted single-shot case.
    let streamed_output = merge_device_partials(&streamed, HEADS, HEAD_DIM)?;
    let (stream_summary, pairwise_bounds) =
        check_streamed_attention(&fixture, &stream_launch, &streamed_output, "host-streamed")?;
    // The comparison run is an alternative execution plan, not a concurrent
    // consumer. Return the bounded streaming arena before admitting the
    // two-page single-shot arena so the ledger measures the actual peak.
    streamed_run
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;

    // A second run with both pages resident exercises the already-qualified
    // single-shot symbol over the identical logical history.
    let full_layer = AttentionLayer {
        geometry,
        heads: HEADS,
        scale,
        visibility: Visibility::Causal,
    };
    let full_launch = PagedAttentionLaunch::new(full_layer, 1, TOTAL_ROWS - 1, 0, TOTAL_ROWS)?;
    let full_descriptor = select_paged_attention_kernel(&catalogue, cap, &full_launch)?;
    let mut full_run = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        full_descriptor,
        geometry,
        HEADS,
        TOTAL_ROWS,
        TOTAL_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|refused| refused.error)?;
    let mut full_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: TOTAL_ROWS as usize,
        tentative_rows: TOTAL_ROWS as usize,
    })?;
    append_authority_rows(
        &mut full_state,
        &mut full_run,
        &stream,
        &fixture,
        TOTAL_ROWS,
    )?;
    let one_shot = full_run
        .attend(
            &stream,
            &full_launch,
            fixture.query_bytes(TOTAL_ROWS - 1, 1),
        )
        .map_err(|refused| refused.error)?;
    let one_shot_summary = check_attention(&fixture, &full_launch, &one_shot, "single-shot")?;
    let one_shot_values = decode_u16(&one_shot);
    let one_shot_values: Vec<f64> = one_shot_values
        .into_iter()
        .map(|bits| f64::from(bf16_value(bits)))
        .collect();
    if streamed_output.len() != one_shot_values.len()
        || streamed_output.len() != pairwise_bounds.len()
    {
        return Ok(Outcome::Failed(
            "streamed and single-shot output widths differ".into(),
        ));
    }
    let narrowed_streamed: Vec<f64> = streamed_output
        .iter()
        .map(|value| f64::from(bf16_value(host_f32_to_bf16_bits(*value as f32))))
        .collect();
    let mut max_pairwise: f64 = 0.0;
    for (index, ((streamed, one_shot), bound)) in narrowed_streamed
        .iter()
        .zip(one_shot_values.iter())
        .zip(pairwise_bounds.iter())
        .enumerate()
    {
        if !streamed.is_finite() {
            return Ok(Outcome::Failed(format!(
                "streamed BF16 output {index} is not finite"
            )));
        }
        let difference = (streamed - one_shot).abs();
        max_pairwise = max_pairwise.max(difference);
        if difference > *bound {
            return Ok(Outcome::Failed(format!(
                "streamed and single-shot output {index} differ by {difference:.3e}, \
                 beyond attention bound {bound:.3e}"
            )));
        }
    }
    println!(
        "    {} host-streamed two-block transfer={} B stream={} single-shot={} max_pairwise={:.3e}",
        cap.sm(),
        streamed.host_to_device_bytes,
        stream_summary.max,
        one_shot_summary.max,
        max_pairwise
    );

    full_run
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;
    // Device state is the task-0038 authority and releases on drop; the host
    // paged store owns the ledger-backed close explicitly.
    host.close(&mut ledger).map_err(|refused| refused.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "host-backed proof left an admission charge outstanding".into(),
        ));
    }

    // Inject a failure after the key copy but before the value/table copies.
    // The run is deliberately quarantined: submitted work may still read the
    // retained host vectors, so it cannot be reused or closed as successful.
    let mut fault_ledger = measured_ledger(&ctx)?;
    let mut fault_run = PagedAttentionRun::admit_for_sequence(
        &mut fault_ledger,
        &ctx,
        select_paged_attention_kernel(&catalogue, cap, &stream_probe)?,
        stream_geometry,
        HEADS,
        RESIDENT_ROWS,
        RESIDENT_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::TwoBlock,
    )
    .map_err(|refused| refused.error)?;
    let mut fault_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: RESIDENT_ROWS as usize,
        tentative_rows: RESIDENT_ROWS as usize,
    })?;
    append_authority_rows(
        &mut fault_state,
        &mut fault_run,
        &stream,
        &fixture,
        RESIDENT_ROWS,
    )?;
    let (fault_keys, fault_values) = fixture.payload(RESIDENT_ROWS, STAGED_ROWS);
    fault_run.inject_staging_failure();
    let refused = fault_run
        .attend_two_block(
            &stream,
            &stream_launch,
            fixture.query_bytes(TOTAL_ROWS - 1, 1),
            fault_keys,
            fault_values,
        )
        .expect_err("an injected partial staging failure was accepted");
    if !refused.retained_source() {
        return Ok(Outcome::Failed(
            "a partial staging failure returned host bytes that may still be in flight".into(),
        ));
    }
    // `fault_run` remains quarantined by design and is dropped without a
    // release path: the unknown completion must keep its arena charged.
    drop(fault_run);
    Ok(Outcome::Passed)
}

/// Task 0042: one resident page plus three host pages, using the same
/// separately-qualified partial ABI once per block and one reused staging
/// allocation. This is deliberately a second case so task 0041's exact
/// two-block path remains an unchanged regression.
fn paged_attention_host_streaming_n3(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 64;
    const PAGE_TOKENS: u64 = 8;
    const RESIDENT_ROWS: u64 = PAGE_TOKENS;
    const STAGED_BLOCKS: u64 = 3;
    const TOTAL_ROWS: u64 = RESIDENT_ROWS + STAGED_BLOCKS * PAGE_TOKENS;
    let geometry = PageGeometry {
        kv_heads: 2,
        head_dim: HEAD_DIM,
        page_tokens: PAGE_TOKENS,
        pages: TOTAL_ROWS / PAGE_TOKENS,
    };
    let scale = moxie_plan::reciprocal_sqrt_scale(HEAD_DIM);
    let fixture = AttentionFixture::build(geometry, HEADS, TOTAL_ROWS, 0x0042_0001);
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();

    let stream_geometry = PageGeometry {
        pages: 1,
        ..geometry
    };
    let stream_layer = AttentionLayer {
        geometry: stream_geometry,
        heads: HEADS,
        scale,
        visibility: Visibility::Causal,
    };
    let stream_probe = PagedAttentionLaunch::new(stream_layer, 1, 0, 0, 1)?;
    let stream_descriptor = select_paged_attention_kernel(&catalogue, cap, &stream_probe)?;

    let mut ledger = measured_ledger(&ctx)?;
    let host_geometry = moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: TOTAL_ROWS as usize,
        tentative_rows: TOTAL_ROWS as usize,
    };
    let mut host = moxie_state::PagedSequence::new(&mut ledger, host_geometry)?;
    let host_txn = host.begin()?;
    host.append_prompt(TOTAL_ROWS)?;
    for position in 0..TOTAL_ROWS {
        let (keys, values) = fixture.payload(position, 1);
        let rows = [moxie_state::KvRow {
            key: &keys,
            value: &values,
        }];
        host.append(
            host_txn,
            position,
            &rows,
            &std::sync::atomic::AtomicBool::new(false),
        )?;
    }
    host.commit_prefix(host_txn, 0)?;

    let mut streamed_run = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        stream_descriptor,
        stream_geometry,
        HEADS,
        RESIDENT_ROWS,
        RESIDENT_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::HostBacked {
            max_staged_blocks: STAGED_BLOCKS,
        },
    )
    .map_err(|refused| refused.error)?;
    let admitted_stream_bytes = streamed_run.arena_bytes();
    let mut resident_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: RESIDENT_ROWS as usize,
        tentative_rows: RESIDENT_ROWS as usize,
    })?;
    append_authority_rows(
        &mut resident_state,
        &mut streamed_run,
        &stream,
        &fixture,
        RESIDENT_ROWS,
    )?;

    let graph = selected_attention_graph(
        HEADS,
        geometry.kv_heads,
        geometry.head_dim,
        scale,
        Visibility::Causal,
    )?;
    let state_layout = resident_state.layout(0)?;
    let state_page_tokens = resident_state.geometry()?.page_tokens as u64;
    let workload = ResourceWorkload {
        phase: Phase::Decode,
        rows: 1,
        visible_tokens: TOTAL_ROWS,
        branch_rows: 1,
        output: graph.output(),
        device: cap.uuid,
        paged_state_capacity: Some(PagedStateCapacity {
            resident_rows: state_layout.capacity_rows,
            page_tokens: state_page_tokens,
            max_staged_blocks: HostBackedPlan::MAX_STAGED_BLOCKS,
        }),
    };
    let candidate = lower_selected(&graph, workload, cap, &catalogue)?;
    let [requirement] = candidate.base().state() else {
        return Ok(Outcome::Failed(
            "streaming graph plan did not report its state requirement".into(),
        ));
    };
    let planned_staged_blocks = match requirement.kind {
        StateRequirementKind::KvPagesHostBacked { staged_blocks } => staged_blocks,
        kind => {
            return Ok(Outcome::Failed(format!(
                "streaming graph plan reported {kind:?}, expected {STAGED_BLOCKS} staged blocks"
            )));
        }
    };
    if planned_staged_blocks != STAGED_BLOCKS {
        return Ok(Outcome::Failed(format!(
            "streaming graph plan reported {planned_staged_blocks} staged blocks, expected {STAGED_BLOCKS}"
        )));
    }
    let selected_plan =
        match SelectedReservedPlan::admit(candidate, &graph, cap, &catalogue, &mut ledger, &ctx) {
            Ok(plan) => plan,
            Err(
                SelectedAdmitRefused::Invalid { error, .. }
                | SelectedAdmitRefused::Held { error, .. },
            ) => return Err(error),
            Err(SelectedAdmitRefused::Rejected { rejection, .. }) => {
                return Err(Error::CapacityExceeded {
                    tier: None,
                    requested_bytes: rejection.shortfall_bytes,
                    available_bytes: 0,
                });
            }
        };

    let stream_launch =
        PagedAttentionLaunch::n_block_stream(stream_layer, 1, TOTAL_ROWS - 1, 0, TOTAL_ROWS)?;
    if stream_launch.staged_blocks() != planned_staged_blocks {
        return Ok(Outcome::Failed(format!(
            "N-block launch derived {} staged blocks, plan admitted {planned_staged_blocks}",
            stream_launch.staged_blocks(),
        )));
    }
    let (streamed_output, actual_transfer, expected_per_block) = {
        let (mut n_stream, resident_partials) = streamed_run
            .start_n_block(
                &stream,
                &stream_launch,
                fixture.query_bytes(TOTAL_ROWS - 1, 1),
            )
            .map_err(|refused| refused.error)?;
        let mut merged = device_partials_to_oracle(&resident_partials, HEADS, HEAD_DIM)?;
        let mut expected_per_block = Vec::new();
        for block in 0..STAGED_BLOCKS {
            // This read is intentionally inside the step loop. The prior
            // stage/launch/readback and oracle fold have settled before the next
            // host page is obtained, so this source cannot read ahead.
            let first = RESIDENT_ROWS + block * PAGE_TOKENS;
            let (keys, values) = host.read_block(0, first, PAGE_TOKENS)?;
            let transfer = keys
                .len()
                .checked_add(values.len())
                .and_then(|bytes| bytes.checked_add(4))
                .ok_or(moxie_types::DimError::Overflow)? as u64;
            expected_per_block.push(transfer);
            let partials = n_stream
                .stage_next(keys, values)
                .map_err(|refused| refused.error)?;
            merge_device_partials_into(&mut merged, &partials, HEADS, HEAD_DIM)?;
            drop(partials);
        }
        if n_stream.remaining_blocks() != 0 {
            return Ok(Outcome::Failed(format!(
                "N-block stream stopped with {} staged block(s) remaining",
                n_stream.remaining_blocks()
            )));
        }
        let actual_transfer = n_stream.host_to_device_bytes();
        let streamed_output = finish_device_partials(merged)?;
        (streamed_output, actual_transfer, expected_per_block)
    };
    let expected_transfer = expected_per_block.iter().sum::<u64>();
    if actual_transfer != expected_transfer {
        return Ok(Outcome::Failed(format!(
            "N-block transfer accounting was {} B; expected {} B across {:?}",
            actual_transfer, expected_transfer, expected_per_block
        )));
    }
    if streamed_run.arena_bytes() != admitted_stream_bytes {
        return Ok(Outcome::Failed(
            "repeated staging changed the admitted device arena size".into(),
        ));
    }
    let (stream_summary, pairwise_bounds) = check_streamed_attention(
        &fixture,
        &stream_launch,
        &streamed_output,
        "host-streamed-n3",
    )?;

    streamed_run
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;
    selected_plan
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;
    // Compare the N=3 run's physical arena with the exact one-page staging
    // shape. Equal sizes are the measured one-buffer property, not a comment.
    let one_block_bound = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        select_paged_attention_kernel(&catalogue, cap, &stream_probe)?,
        stream_geometry,
        HEADS,
        RESIDENT_ROWS,
        Staging::TwoBlock,
    )
    .map_err(|refused| refused.error)?;
    let one_block_bytes = one_block_bound.arena_bytes();
    one_block_bound
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;
    if admitted_stream_bytes != one_block_bytes {
        return Ok(Outcome::Failed(format!(
            "N=3 staging arena is {admitted_stream_bytes} B, not the one-buffer {one_block_bytes} B"
        )));
    }

    // The same complete history through the accepted single-shot symbol.
    let full_layer = AttentionLayer {
        geometry,
        heads: HEADS,
        scale,
        visibility: Visibility::Causal,
    };
    let full_launch = PagedAttentionLaunch::new(full_layer, 1, TOTAL_ROWS - 1, 0, TOTAL_ROWS)?;
    let full_descriptor = select_paged_attention_kernel(&catalogue, cap, &full_launch)?;
    let mut full_run = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        full_descriptor,
        geometry,
        HEADS,
        TOTAL_ROWS,
        TOTAL_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|refused| refused.error)?;
    let mut full_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: TOTAL_ROWS as usize,
        tentative_rows: TOTAL_ROWS as usize,
    })?;
    append_authority_rows(
        &mut full_state,
        &mut full_run,
        &stream,
        &fixture,
        TOTAL_ROWS,
    )?;
    let one_shot = full_run
        .attend(
            &stream,
            &full_launch,
            fixture.query_bytes(TOTAL_ROWS - 1, 1),
        )
        .map_err(|refused| refused.error)?;
    let one_shot_summary = check_attention(&fixture, &full_launch, &one_shot, "single-shot-n3")?;
    let one_shot_values: Vec<f64> = decode_u16(&one_shot)
        .into_iter()
        .map(|bits| f64::from(bf16_value(bits)))
        .collect();
    if streamed_output.len() != one_shot_values.len()
        || streamed_output.len() != pairwise_bounds.len()
    {
        return Ok(Outcome::Failed(
            "N-block streamed and single-shot output widths differ".into(),
        ));
    }
    let narrowed_streamed: Vec<f64> = streamed_output
        .iter()
        .map(|value| f64::from(bf16_value(host_f32_to_bf16_bits(*value as f32))))
        .collect();
    let mut max_pairwise: f64 = 0.0;
    for (index, ((streamed, one_shot), bound)) in narrowed_streamed
        .iter()
        .zip(one_shot_values.iter())
        .zip(pairwise_bounds.iter())
        .enumerate()
    {
        if !streamed.is_finite() {
            return Ok(Outcome::Failed(format!(
                "N-block streamed BF16 output {index} is not finite"
            )));
        }
        let difference = (streamed - one_shot).abs();
        max_pairwise = max_pairwise.max(difference);
        if difference > *bound {
            return Ok(Outcome::Failed(format!(
                "N-block streamed and single-shot output {index} differ by {difference:.3e}, \
                 beyond attention bound {bound:.3e}"
            )));
        }
    }
    println!(
        "    {} host-streamed N=3 transfer={} B per_block={:?} stream={} single-shot={} \
         staging_arena={} B max_pairwise={:.3e}",
        cap.sm(),
        actual_transfer,
        expected_per_block,
        stream_summary.max,
        one_shot_summary.max,
        admitted_stream_bytes,
        max_pairwise
    );
    full_run
        .close(&mut ledger)
        .map_err(|refused| refused.error)?;

    // Fail after the first staged block, so a middle iteration is the one that
    // refuses. The partial from the first block never escapes this operation,
    // and the currently copied block remains owned by the quarantined run.
    let mut fault_ledger = measured_ledger(&ctx)?;
    let mut fault_run = PagedAttentionRun::admit_for_sequence(
        &mut fault_ledger,
        &ctx,
        select_paged_attention_kernel(&catalogue, cap, &stream_probe)?,
        stream_geometry,
        HEADS,
        RESIDENT_ROWS,
        RESIDENT_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::HostBacked {
            max_staged_blocks: STAGED_BLOCKS,
        },
    )
    .map_err(|refused| refused.error)?;
    let mut fault_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: RESIDENT_ROWS as usize,
        tentative_rows: RESIDENT_ROWS as usize,
    })?;
    append_authority_rows(
        &mut fault_state,
        &mut fault_run,
        &stream,
        &fixture,
        RESIDENT_ROWS,
    )?;
    fault_run.inject_staging_failure_after(1);
    let (failed_at, refused) = {
        let (mut fault_stream, _resident_partials) = fault_run
            .start_n_block(
                &stream,
                &stream_launch,
                fixture.query_bytes(TOTAL_ROWS - 1, 1),
            )
            .map_err(|refused| refused.error)?;
        let mut failed_at = None;
        for block in 0..STAGED_BLOCKS {
            let first = RESIDENT_ROWS + block * PAGE_TOKENS;
            let (keys, values) = host.read_block(0, first, PAGE_TOKENS)?;
            match fault_stream.stage_next(keys, values) {
                Ok(partials) => drop(partials),
                Err(refused) => {
                    failed_at = Some((block, refused));
                    break;
                }
            }
        }
        failed_at.ok_or_else(|| Error::InvalidRequest {
            field: "fault",
            detail: "a mid-sequence N-block staging failure was accepted".into(),
        })?
    };
    if failed_at != 1 {
        return Ok(Outcome::Failed(format!(
            "the injected N-block staging failure happened at block {failed_at}, expected 1"
        )));
    }
    if !refused.retained_source() {
        return Ok(Outcome::Failed(
            "a mid-sequence staging failure returned bytes that may still be in flight".into(),
        ));
    }
    drop(fault_run);

    host.close(&mut ledger).map_err(|refused| refused.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "N-block host-backed proof left an admission charge outstanding".into(),
        ));
    }
    Ok(Outcome::Passed)
}

fn check_streamed_attention(
    fixture: &AttentionFixture,
    launch: &PagedAttentionLaunch,
    output: &[f64],
    label: &str,
) -> Result<(moxie_oracles::metric::ErrorSummary, Vec<f64>), Error> {
    use moxie_oracles::attention::attention_error_bounds_at;
    use moxie_oracles::online_softmax::attend_row_blocked;

    let head_dim = fixture.geometry.head_dim as usize;
    let group = fixture.heads / fixture.geometry.kv_heads;
    let block = fixture.query(launch.first_position(), launch.rows());
    let mut got = Vec::new();
    let mut want_all = Vec::new();
    let mut bounds_all = Vec::new();
    for row in 0..launch.rows() {
        let visible: Vec<u64> = (launch.history_base()
            ..launch.history_base() + launch.history_rows())
            .filter(|key| launch.allows(row, *key))
            .collect();
        if visible.is_empty() {
            return Err(Error::Numerical {
                detail: format!("{label}: row {row} sees nothing"),
            });
        }
        for head in 0..fixture.heads {
            let kv_head = head / group;
            let start = ((row * fixture.heads + head) * fixture.geometry.head_dim) as usize;
            let query_row: Vec<f32> = block[start..start + head_dim]
                .iter()
                .map(|bits| bf16_value(*bits))
                .collect();
            let keys: Vec<Vec<f32>> = visible
                .iter()
                .map(|position| fixture.head_slice(&fixture.keys, *position, kv_head))
                .collect();
            let values: Vec<Vec<f32>> = visible
                .iter()
                .map(|position| fixture.head_slice(&fixture.values, *position, kv_head))
                .collect();
            let key_views: Vec<&[f32]> = keys.iter().map(Vec::as_slice).collect();
            let value_views: Vec<&[f32]> = values.iter().map(Vec::as_slice).collect();
            let allowed = vec![true; visible.len()];
            let want = attend_row_blocked(
                &query_row,
                &key_views,
                &value_views,
                &allowed,
                launch.scale(),
                37,
            )
            .map_err(numerical)?;
            let bounds =
                attention_error_bounds_at(&query_row, &key_views, &value_views, launch.scale())
                    .map_err(numerical)?;
            for d in 0..head_dim {
                let index = start + d;
                let device = *output.get(index).ok_or_else(|| Error::InvalidRequest {
                    field: "output",
                    detail: "streamed output is shorter than its launch".into(),
                })?;
                if !device.is_finite() {
                    return Err(Error::Numerical {
                        detail: format!("{label}: output {index} is {device}"),
                    });
                }
                let difference = (device - want[d]).abs();
                if difference > bounds[d] {
                    return Err(Error::Numerical {
                        detail: format!(
                            "{label}: row {row} head {head} lane {d}: {device} against \
                             oracle {}, {difference:.3e} apart; bound {:.3e}",
                            want[d], bounds[d]
                        ),
                    });
                }
                got.push(device as f32);
                want_all.push(want[d]);
                bounds_all.push(bounds[d]);
            }
        }
    }
    if output.len() != got.len() {
        return Err(Error::InvalidRequest {
            field: "output",
            detail: format!(
                "{} value(s) where the launch produces {}",
                output.len(),
                got.len()
            ),
        });
    }
    Ok((
        moxie_oracles::metric::ErrorSummary::absolute(&got, &want_all),
        bounds_all,
    ))
}

fn merge_device_partials(
    streamed: &TwoBlockAttention,
    heads: u64,
    head_dim: u64,
) -> Result<Vec<f64>, Error> {
    merge_device_partial_chain(
        &[streamed.resident.as_slice(), streamed.staged.as_slice()],
        heads,
        head_dim,
    )
}

fn device_partials_to_oracle(
    partials: &[moxie_executor::DevicePartial],
    heads: u64,
    head_dim: u64,
) -> Result<Vec<moxie_oracles::online_softmax::Partial>, Error> {
    let heads = usize::try_from(heads).map_err(|_| Error::Dim(moxie_types::DimError::Overflow))?;
    let head_dim =
        usize::try_from(head_dim).map_err(|_| Error::Dim(moxie_types::DimError::Overflow))?;
    if heads == 0 || head_dim == 0 || !partials.len().is_multiple_of(heads) {
        return Err(Error::InvalidRequest {
            field: "partial",
            detail: "partial count is not a whole nonempty head group".into(),
        });
    }
    partials
        .iter()
        .map(|partial| {
            if partial.weighted.len() != head_dim {
                return Err(Error::InvalidRequest {
                    field: "partial",
                    detail: "device partial value widths differ from the launch".into(),
                });
            }
            Ok(moxie_oracles::online_softmax::Partial {
                max: f64::from(partial.max),
                sum: f64::from(partial.sum),
                weighted: partial
                    .weighted
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
            })
        })
        .collect()
}

fn merge_device_partials_into(
    running: &mut [moxie_oracles::online_softmax::Partial],
    incoming: &[moxie_executor::DevicePartial],
    heads: u64,
    head_dim: u64,
) -> Result<(), Error> {
    let incoming = device_partials_to_oracle(incoming, heads, head_dim)?;
    if running.len() != incoming.len() {
        return Err(Error::InvalidRequest {
            field: "partial",
            detail: "the partial chain has mismatched head counts".into(),
        });
    }
    for (left, right) in running.iter_mut().zip(incoming.iter()) {
        *left = left.merge(right)?;
    }
    Ok(())
}

fn finish_device_partials(
    partials: Vec<moxie_oracles::online_softmax::Partial>,
) -> Result<Vec<f64>, Error> {
    let mut output = Vec::new();
    for partial in partials {
        output.extend(partial.finish()?);
    }
    Ok(output)
}

fn merge_device_partial_chain(
    blocks: &[&[moxie_executor::DevicePartial]],
    heads: u64,
    head_dim: u64,
) -> Result<Vec<f64>, Error> {
    use moxie_oracles::online_softmax::Partial;

    let Some(first) = blocks.first() else {
        return Err(Error::InvalidRequest {
            field: "partial",
            detail: "the partial chain is empty".into(),
        });
    };
    if blocks.len() < 2 || blocks.iter().any(|block| block.len() != first.len()) {
        return Err(Error::InvalidRequest {
            field: "partial",
            detail: "the partial chain has mismatched block counts".into(),
        });
    }
    let heads = usize::try_from(heads).map_err(|_| Error::Dim(moxie_types::DimError::Overflow))?;
    let head_dim =
        usize::try_from(head_dim).map_err(|_| Error::Dim(moxie_types::DimError::Overflow))?;
    if heads == 0 || head_dim == 0 || !first.len().is_multiple_of(heads) {
        return Err(Error::InvalidRequest {
            field: "partial",
            detail: "partial count is not a whole nonempty head group".into(),
        });
    }
    let mut output = Vec::with_capacity(first.len() * head_dim);
    for index in 0..first.len() {
        let to_partial = |partial: &moxie_executor::DevicePartial| {
            if partial.weighted.len() != head_dim {
                return Err(Error::InvalidRequest {
                    field: "partial",
                    detail: "device partial value widths differ from the launch".into(),
                });
            }
            Ok(Partial {
                max: f64::from(partial.max),
                sum: f64::from(partial.sum),
                weighted: partial
                    .weighted
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
            })
        };
        let mut merged = to_partial(&blocks[0][index])?;
        for block in blocks.iter().skip(1) {
            merged = merged.merge(&to_partial(&block[index])?)?;
        }
        output.extend(merged.finish()?);
    }
    Ok(output)
}

fn append_authority_rows<'ctx>(
    sequence: &mut moxie_state::DeviceKvSequence,
    run: &mut PagedAttentionRun<'ctx>,
    stream: &Stream<'ctx>,
    fixture: &AttentionFixture,
    rows: u64,
) -> Result<(), Error> {
    let txn = sequence.begin()?;
    let first = sequence.published_rows()?;
    let (keys, values) = fixture.payload(first, rows);
    moxie_executor::paged_attention::device::append_paged_layer(
        sequence,
        txn,
        0,
        rows,
        run,
        stream,
        moxie_executor::paged_attention::device::PagedKvRows { keys, values },
    )
    .map_err(|refused| refused.error)?;
    moxie_executor::paged_attention::device::commit_paged_layer(sequence, txn, rows, run, stream)
}

fn attend_authority<'ctx>(
    run: &mut PagedAttentionRun<'ctx>,
    stream: &Stream<'ctx>,
    state: &moxie_state::DeviceKvSequence,
    fixture: &AttentionFixture,
    layer: AttentionLayer,
    position: u64,
    label: &str,
) -> Result<Vec<u8>, Error> {
    let retained = state.retained(0)?;
    let committed = state.committed_rows()?;
    if retained.end != committed {
        return Err(Error::InvalidRequest {
            field: "state",
            detail: "a closed transaction retained rows past its committed frontier".into(),
        });
    }
    let launch = PagedAttentionLaunch::new(
        layer,
        1,
        position,
        retained.start,
        committed - retained.start,
    )?;
    let output = run
        .attend(stream, &launch, fixture.query_bytes(position, 1))
        .map_err(|r| r.error)?;
    check_attention(fixture, &launch, &output, label)?;
    Ok(output)
}

fn device_state_row_placements(
    state: &moxie_state::DeviceKvSequence,
    first: u64,
    rows: u64,
) -> Result<Vec<PagePlacement>, Error> {
    (first
        ..first
            .checked_add(rows)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?)
        .map(|position| state.placement_of(0, position))
        .collect()
}

fn device_branch_row_placements(
    branch: &moxie_state::DeviceBranch<'_>,
    first: u64,
    rows: u64,
) -> Result<Vec<PagePlacement>, Error> {
    (first
        ..first
            .checked_add(rows)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?)
        .map(|position| branch.placement_of(0, position))
        .collect()
}

/// Task 0037 acceptance 3: **32,768 actual BF16 key/value rows**.
///
/// Not an admitted capacity, not a declared maximum and not a short history
/// with a long label: 32,768 rows are materialized, appended and attended over,
/// and then row 32,768 is appended and the next decode produced. Whole and
/// chunked construction are compared, and the three numbers a cache can confuse
/// -- admitted capacity, committed rows and visible rows -- are reported
/// separately because they are different facts.
///
/// No timing appears here. O6 and O7 are open; this gate is about correctness
/// and bounded state.
fn paged_attention_32k(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const CONTEXT: u64 = 32_768;
    const SECOND_TURN_PREFILL_ROWS: u64 = 255;
    const SECOND_TURN_ROWS: u64 = SECOND_TURN_PREFILL_ROWS + 1;
    let geometry = PageGeometry {
        kv_heads: 2,
        head_dim: 128,
        page_tokens: 256,
        // One page beyond the context, for the row that is appended after it.
        pages: CONTEXT / 256 + 1,
    };
    let heads = 8;
    let scale = moxie_plan::reciprocal_sqrt_scale(128);
    // The first-turn multi-row prefill chunk at the far end of the history,
    // which is also what proves a multi-row launch and a one-row decode agree
    // at 32K. The second turn below prefills 255 rows, then appends and
    // decodes the page-tail row separately.
    let chunk_rows = 24u64;

    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();
    let fixture = AttentionFixture::build(geometry, heads, CONTEXT + SECOND_TURN_ROWS, 0x0037_8000);

    // Task 0038: the state authority owns this history. It places every row,
    // publishes the mapping for its own retained range, and is the only thing
    // that says what is committed; the run performs and launches. A gate that
    // drove the run directly would be testing a path nothing uses.
    let authority = || -> Result<moxie_state::DeviceKvSequence, Error> {
        let sequence = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
            layers: vec![moxie_state::LayerKv {
                kv_heads: geometry.kv_heads as usize,
                key_dim: geometry.head_dim as usize,
                value_dim: geometry.head_dim as usize,
                retention: moxie_state::Retention::All,
            }],
            precision: Precision::Bf16,
            page_tokens: geometry.page_tokens as usize,
            max_tokens: (geometry.pages * geometry.page_tokens) as usize,
            tentative_rows: (geometry.pages * geometry.page_tokens) as usize,
        })?;
        if sequence.layout(0)?.pages != geometry.pages {
            return Err(Error::InvalidRequest {
                field: "geometry",
                detail: "the authority and the run describe different page counts".into(),
            });
        }
        Ok(sequence)
    };

    let decode = |first_position: u64, history_rows: u64, visibility: Visibility| {
        PagedAttentionLaunch::new(
            AttentionLayer {
                geometry,
                heads,
                scale,
                visibility,
            },
            1,
            first_position,
            0,
            history_rows,
        )
    };
    let descriptor =
        select_paged_attention_kernel(&catalogue, cap, &decode(0, 1, Visibility::Causal)?)?;

    let root_lineage_capacity = geometry
        .pages
        .checked_mul(geometry.page_tokens)
        .and_then(|positions| positions.checked_add(1))
        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;

    // Build A: the whole history in one append.
    let mut ledger = measured_ledger(&ctx)?;
    let host_before_root_admission = ledger.scope_committed(Scope::Host);
    let mut whole = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        CONTEXT,
        root_lineage_capacity,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let root_admitted_host_bytes = ledger
        .scope_committed(Scope::Host)
        .checked_sub(host_before_root_admission)
        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
    let expected_root_host_bytes =
        moxie_state::DeviceKvSequence::root_host_metadata_bytes(root_lineage_capacity)?;
    let mut whole_state = authority()?;
    let root_capacity = usize::try_from(root_lineage_capacity)
        .map_err(|_| Error::Dim(moxie_types::DimError::Overflow))?;
    if whole_state.state()?.lineage_capacity(moxie_state::ROOT)? != root_capacity {
        return Ok(Outcome::Failed(
            "the 32K root lineage was not reserved to the admitted maximum".into(),
        ));
    }
    append_authority_rows(&mut whole_state, &mut whole, &stream, &fixture, CONTEXT)?;
    if whole_state.committed_rows()? != CONTEXT
        || whole.written_rows() != CONTEXT
        || whole_state.state()?.lineage_capacity(moxie_state::ROOT)? != root_capacity
    {
        return Ok(Outcome::Failed(format!(
            "the authority committed {} row(s) and the run wrote {}, not {CONTEXT}",
            whole_state.committed_rows()?,
            whole.written_rows()
        )));
    }

    // Build B: the same rows, appended in uneven chunks that cross pages and
    // leave a partial page open in the middle.
    let mut chunked = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        CONTEXT,
        root_lineage_capacity,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let mut chunked_state = authority()?;
    let mut written = 0u64;
    for rows in [1u64, 255, 256, 7_000, 25_256] {
        append_authority_rows(&mut chunked_state, &mut chunked, &stream, &fixture, rows)?;
        written += rows;
    }
    if written != CONTEXT || chunked_state.committed_rows()? != CONTEXT {
        return Ok(Outcome::Failed(format!(
            "chunked construction committed {} row(s) after {written}",
            chunked_state.committed_rows()?
        )));
    }

    // The last row of the 32,768 attends over the whole history. Whole and
    // chunked construction must produce the same bytes, not merely close ones.
    let last = decode(CONTEXT - 1, CONTEXT, Visibility::Causal)?;
    let query = fixture.query_bytes(CONTEXT - 1, 1);
    let from_whole = whole
        .attend(&stream, &last, query.clone())
        .map_err(|r| r.error)?;
    let from_chunked = chunked.attend(&stream, &last, query).map_err(|r| r.error)?;
    if from_whole != from_chunked {
        return Ok(Outcome::Failed(
            "whole and chunked construction disagree at 32,768 rows".into(),
        ));
    }
    let summary = check_attention(&fixture, &last, &from_whole, "32k-decode")?;
    println!(
        "    {} 32k-decode visible={} committed={} capacity={} state={} B {summary}",
        cap.sm(),
        last.history_rows(),
        whole_state.committed_rows()?,
        whole.capacity_rows()?,
        whole.arena_bytes()
    );

    // A multi-row prefill chunk at the far end, against the same rows decoded
    // one at a time. Same operation, different row counts, identical bytes.
    let chunk = last.at(chunk_rows, CONTEXT - chunk_rows)?;
    let block = whole
        .attend(
            &stream,
            &chunk,
            fixture.query_bytes(chunk.first_position(), chunk_rows),
        )
        .map_err(|r| r.error)?;
    let lane_bytes = (heads * geometry.head_dim * 2) as usize;
    for row in 0..chunk_rows {
        let position = chunk.first_position() + row;
        let one = whole
            .attend(
                &stream,
                &decode(position, CONTEXT, Visibility::Causal)?,
                fixture.query_bytes(position, 1),
            )
            .map_err(|r| r.error)?;
        let start = row as usize * lane_bytes;
        if one != block[start..start + lane_bytes] {
            return Ok(Outcome::Failed(format!(
                "row {position} differs between a {chunk_rows}-row chunk and a single decode"
            )));
        }
    }

    // Task 0050: continue from the exact committed 32K parent through the
    // accepted eager device-COW path. The later turn prefills 255 rows across
    // the new page, then appends and decodes its page-tail row separately.
    // That keeps the decode-after operation distinct from the prefill while
    // retaining the full 256-row second-turn envelope.
    let parent_frontier_before_turn = whole_state.committed_rows()?;
    let parent_retained_before_turn = whole_state.retained(0)?;
    if parent_frontier_before_turn != CONTEXT || parent_retained_before_turn != (0..CONTEXT) {
        return Ok(Outcome::Failed(format!(
            "the 32K parent is {:?} with retained {:?} before the second turn",
            parent_frontier_before_turn, parent_retained_before_turn
        )));
    }
    let parent_placements = device_state_row_placements(&whole_state, 0, CONTEXT)?;
    let parent_before = whole.read_rows(&parent_placements)?;
    let parent_arena_bytes = whole.arena_bytes();
    let second_prefill = PagedAttentionLaunch::new(
        AttentionLayer {
            geometry,
            heads,
            scale,
            visibility: Visibility::Causal,
        },
        SECOND_TURN_PREFILL_ROWS,
        CONTEXT,
        0,
        CONTEXT + SECOND_TURN_PREFILL_ROWS,
    )?;
    let decode_position = CONTEXT + SECOND_TURN_PREFILL_ROWS;
    let second_decode = PagedAttentionLaunch::new(
        AttentionLayer {
            geometry,
            heads,
            scale,
            visibility: Visibility::Causal,
        },
        1,
        decode_position,
        0,
        CONTEXT + SECOND_TURN_ROWS,
    )?;
    let prefill_tail_position = CONTEXT + SECOND_TURN_PREFILL_ROWS - 1;

    // First second-turn construction: one whole append in the forked child.
    let fork_lineage_capacity = root_lineage_capacity;
    let without_fork = moxie_executor::paged_attention::device::resource_request(
        &geometry,
        heads,
        CONTEXT,
        Staging::Host,
        &ctx,
    )?;
    let with_root = moxie_executor::paged_attention::device::resource_request_for_sequence(
        &geometry,
        heads,
        CONTEXT,
        root_lineage_capacity,
        Staging::Host,
        &ctx,
    )?;
    let with_fork = moxie_executor::paged_attention::device::resource_request_for_fork(
        &geometry,
        heads,
        CONTEXT,
        fork_lineage_capacity,
        Staging::Host,
        &ctx,
    )?;
    let base_host_peak = ledger
        .preview(&without_fork)?
        .scope(Scope::Host)
        .map(|report| report.request_peak_bytes);
    let root_host_peak = ledger
        .preview(&with_root)?
        .scope(Scope::Host)
        .map(|report| report.request_peak_bytes);
    let fork_host_peak = ledger
        .preview(&with_fork)?
        .scope(Scope::Host)
        .map(|report| report.request_peak_bytes);
    let (Some(base_host_peak), Some(root_host_peak), Some(fork_host_peak)) =
        (base_host_peak, root_host_peak, fork_host_peak)
    else {
        return Ok(Outcome::Failed(
            "the 32K root or fork request omitted its pageable host scope".into(),
        ));
    };
    if root_host_peak.checked_sub(base_host_peak) != Some(expected_root_host_bytes)
        || root_admitted_host_bytes != root_host_peak
    {
        return Ok(Outcome::Failed(format!(
            "the 32K root request charged {:?} B beyond base metadata and reserved {root_admitted_host_bytes} B, expected {expected_root_host_bytes} B and {root_host_peak} B",
            root_host_peak.checked_sub(base_host_peak),
        )));
    }
    let expected_fork_bytes =
        moxie_state::DeviceKvSequence::fork_host_metadata_bytes(fork_lineage_capacity, 1)?;
    if fork_host_peak.checked_sub(base_host_peak) != Some(expected_fork_bytes) {
        return Ok(Outcome::Failed(format!(
            "the 32K fork request charged {:?} B beyond base metadata, expected {expected_fork_bytes} B",
            fork_host_peak.checked_sub(base_host_peak)
        )));
    }
    let host_before_fork_admission = ledger.scope_committed(Scope::Host);
    let mut whole_turn = PagedAttentionRun::admit_for_fork(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        CONTEXT,
        fork_lineage_capacity,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    if ledger
        .scope_committed(Scope::Host)
        .checked_sub(host_before_fork_admission)
        != Some(fork_host_peak)
    {
        return Ok(Outcome::Failed(
            "the admitted 32K child reservation disagreed with its fork request".into(),
        ));
    }
    let whole_child_id = moxie_executor::paged_attention::device::fork_paged_layer(
        &mut whole_state,
        CONTEXT,
        &whole,
        &mut whole_turn,
        &stream,
    )?;
    if whole_state.state()?.lineage_capacity(whole_child_id)? != root_capacity {
        return Ok(Outcome::Failed(
            "the 32K child lineage was not reserved to the admitted maximum".into(),
        ));
    }
    {
        let branch = whole_state.branch(whole_child_id)?;
        let inherited =
            whole_turn.read_rows(&device_branch_row_placements(&branch, 0, CONTEXT)?)?;
        if inherited != parent_before {
            return Ok(Outcome::Failed(
                "the 32K child did not inherit the parent's exact device bytes".into(),
            ));
        }
        let parent_last = branch.placement_of(0, CONTEXT - 1)?;
        let expected_new_page = CONTEXT / geometry.page_tokens;
        if parent_last.physical_page != expected_new_page - 1
            || parent_last.slot != geometry.page_tokens - 1
        {
            return Ok(Outcome::Failed(format!(
                "the 32K parent page boundary was wrong: parent={parent_last:?}"
            )));
        }
    }
    if whole.read_rows(&parent_placements)? != parent_before
        || whole_state.committed_rows()? != parent_frontier_before_turn
        || whole_state.retained(0)? != parent_retained_before_turn
    {
        return Ok(Outcome::Failed(
            "creating the 32K COW child changed the parent's bytes or frontier".into(),
        ));
    }
    {
        let mut branch = whole_state.branch(whole_child_id)?;
        let txn = branch.begin()?;
        let first = branch.published_rows()?;
        if first != CONTEXT {
            return Ok(Outcome::Failed(format!(
                "the whole second-turn child began at row {first}, not {CONTEXT}"
            )));
        }
        let (keys, values) = fixture.payload(first, SECOND_TURN_PREFILL_ROWS);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            SECOND_TURN_PREFILL_ROWS,
            &mut whole_turn,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            SECOND_TURN_PREFILL_ROWS,
            &mut whole_turn,
            &stream,
        )?;
        let child_first = branch.placement_of(0, CONTEXT)?;
        let child_prefill_tail = branch.placement_of(0, prefill_tail_position)?;
        let expected_new_page = CONTEXT / geometry.page_tokens;
        if child_first.physical_page != expected_new_page
            || child_first.slot != 0
            || child_prefill_tail.physical_page != expected_new_page
            || child_prefill_tail.slot != geometry.page_tokens - 2
        {
            return Ok(Outcome::Failed(format!(
                "the second-turn page boundary was wrong: child_first={child_first:?}, \
                 child_prefill_tail={child_prefill_tail:?}"
            )));
        }
        if branch.committed_rows()? != CONTEXT + SECOND_TURN_PREFILL_ROWS
            || branch.retained(0)? != (0..CONTEXT + SECOND_TURN_PREFILL_ROWS)
        {
            return Ok(Outcome::Failed(
                "the whole second-turn child did not commit its prefill".into(),
            ));
        }
    }
    if whole_state.state()?.lineage_capacity(whole_child_id)? != root_capacity {
        return Ok(Outcome::Failed(
            "the 32K child lineage reallocated during its second-turn append".into(),
        ));
    }
    let mut whole_prefill_output =
        Vec::with_capacity(SECOND_TURN_PREFILL_ROWS as usize * lane_bytes);
    let mut first = CONTEXT;
    while first < CONTEXT + SECOND_TURN_PREFILL_ROWS {
        let rows = chunk_rows.min(CONTEXT + SECOND_TURN_PREFILL_ROWS - first);
        let launch = second_prefill.at(rows, first)?;
        let output = whole_turn
            .attend(&stream, &launch, fixture.query_bytes(first, rows))
            .map_err(|r| r.error)?;
        whole_prefill_output.extend_from_slice(&output);
        first += rows;
    }
    let first_second_summary = check_attention(
        &fixture,
        &second_prefill.at(1, CONTEXT)?,
        &whole_prefill_output[..lane_bytes],
        "32k-second-turn-first-page-row",
    )?;
    let prefill_tail_start = (SECOND_TURN_PREFILL_ROWS as usize - 1) * lane_bytes;
    let last_second_summary = check_attention(
        &fixture,
        &second_prefill.at(1, prefill_tail_position)?,
        &whole_prefill_output[prefill_tail_start..prefill_tail_start + lane_bytes],
        "32k-second-turn-prefill-tail-row",
    )?;
    {
        let mut branch = whole_state.branch(whole_child_id)?;
        let txn = branch.begin()?;
        let first = branch.published_rows()?;
        if first != decode_position {
            return Ok(Outcome::Failed(format!(
                "the whole second-turn decode row began at {first}, not {decode_position}"
            )));
        }
        let (keys, values) = fixture.payload(first, 1);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            1,
            &mut whole_turn,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            1,
            &mut whole_turn,
            &stream,
        )?;
        let child_decode_tail = branch.placement_of(0, decode_position)?;
        let expected_new_page = CONTEXT / geometry.page_tokens;
        if child_decode_tail.physical_page != expected_new_page
            || child_decode_tail.slot != geometry.page_tokens - 1
            || branch.committed_rows()? != CONTEXT + SECOND_TURN_ROWS
            || branch.retained(0)? != (0..CONTEXT + SECOND_TURN_ROWS)
        {
            return Ok(Outcome::Failed(format!(
                "the whole second-turn decode row was not the page tail: \
                 placement={child_decode_tail:?}, committed={} retained={:?}",
                branch.committed_rows()?,
                branch.retained(0)?
            )));
        }
    }
    let whole_decode_output = whole_turn
        .attend(
            &stream,
            &second_decode,
            fixture.query_bytes(decode_position, 1),
        )
        .map_err(|r| r.error)?;
    let second_decode_summary = check_attention(
        &fixture,
        &second_decode,
        &whole_decode_output,
        "32k-second-turn-decode",
    )?;
    // Adversarial isolation check: an implementation that aliases the parent
    // pages would survive only-new-page appends. Rewrite the child's final
    // inherited row after the second-turn result has been checked, and prove
    // that the child changes to the alternate bytes while the parent prefix
    // remains byte-identical.
    let divergence_fixture =
        AttentionFixture::build(geometry, heads, CONTEXT + SECOND_TURN_ROWS, 0x0050_0001);
    let parent_prefix_before_divergence =
        whole.read_rows(&device_state_row_placements(&whole_state, 0, CONTEXT - 1)?)?;
    let parent_last_row_before_divergence =
        whole.read_rows(&device_state_row_placements(&whole_state, CONTEXT - 1, 1)?)?;
    let (divergent_keys, divergent_values) = divergence_fixture.payload(CONTEXT - 1, 1);
    let mut expected_divergent_row = divergent_keys.clone();
    expected_divergent_row.extend_from_slice(&divergent_values);
    {
        let mut branch = whole_state.branch(whole_child_id)?;
        branch.truncate(CONTEXT - 1)?;
        let txn = branch.begin()?;
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            1,
            &mut whole_turn,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows {
                keys: divergent_keys,
                values: divergent_values,
            },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            1,
            &mut whole_turn,
            &stream,
        )?;
        let child_prefix_after_divergence =
            whole_turn.read_rows(&device_branch_row_placements(&branch, 0, CONTEXT - 1)?)?;
        let child_last_row_after_divergence =
            whole_turn.read_rows(&device_branch_row_placements(&branch, CONTEXT - 1, 1)?)?;
        if child_prefix_after_divergence != parent_prefix_before_divergence
            || child_last_row_after_divergence != expected_divergent_row
            || child_last_row_after_divergence == parent_last_row_before_divergence
        {
            return Ok(Outcome::Failed(
                "32K child divergence did not isolate the inherited final row".into(),
            ));
        }
    }
    let whole_turn_arena_bytes = whole_turn.arena_bytes();
    if whole_state.state()?.lineage_capacity(whole_child_id)? != root_capacity {
        return Ok(Outcome::Failed(
            "the 32K child lineage reallocated during the later-turn steps".into(),
        ));
    }
    if whole.read_rows(&parent_placements)? != parent_before
        || whole_state.committed_rows()? != parent_frontier_before_turn
        || whole_state.retained(0)? != parent_retained_before_turn
    {
        return Ok(Outcome::Failed(
            "whole second-turn execution changed the parent's bytes or frontier".into(),
        ));
    }
    let parent_frontier_before_turn_discard = whole_state.committed_rows()?;
    let parent_retained_before_turn_discard = whole_state.retained(0)?;
    whole_state.discard_branch(whole_child_id)?;
    if whole_state.committed_rows()? != parent_frontier_before_turn_discard
        || whole_state.retained(0)? != parent_retained_before_turn_discard
        || whole.read_rows(&parent_placements)? != parent_before
    {
        return Ok(Outcome::Failed(
            "discarding the whole second-turn child changed the parent".into(),
        ));
    }
    whole_turn.close(&mut ledger).map_err(|r| r.error)?;

    // Second second-turn construction: fork the same parent again and append
    // the identical page in uneven chunks. Exact output parity with the whole
    // append catches a chunk-boundary mistake; the parent readback below
    // keeps this branch's existence and cleanup in the same isolation proof.
    let mut chunked_turn = PagedAttentionRun::admit_for_fork(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        CONTEXT,
        fork_lineage_capacity,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let chunked_child_id = moxie_executor::paged_attention::device::fork_paged_layer(
        &mut whole_state,
        CONTEXT,
        &whole,
        &mut chunked_turn,
        &stream,
    )?;
    if whole_state.state()?.lineage_capacity(chunked_child_id)? != root_capacity {
        return Ok(Outcome::Failed(
            "the chunked 32K child lineage was not reserved to the admitted maximum".into(),
        ));
    }
    {
        let branch = whole_state.branch(chunked_child_id)?;
        if chunked_turn.read_rows(&device_branch_row_placements(&branch, 0, CONTEXT)?)?
            != parent_before
        {
            return Ok(Outcome::Failed(
                "the chunked second-turn child did not inherit the parent bytes".into(),
            ));
        }
    }
    let second_turn_chunks = [1u64, 63, 64, 127];
    for rows in second_turn_chunks {
        let mut branch = whole_state.branch(chunked_child_id)?;
        let txn = branch.begin()?;
        let first = branch.published_rows()?;
        let (keys, values) = fixture.payload(first, rows);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            rows,
            &mut chunked_turn,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            rows,
            &mut chunked_turn,
            &stream,
        )?;
    }
    if whole_state.state()?.lineage_capacity(chunked_child_id)? != root_capacity {
        return Ok(Outcome::Failed(
            "the chunked 32K child lineage reallocated during append".into(),
        ));
    }
    {
        let branch = whole_state.branch(chunked_child_id)?;
        if branch.committed_rows()? != CONTEXT + SECOND_TURN_PREFILL_ROWS
            || branch.retained(0)? != (0..CONTEXT + SECOND_TURN_PREFILL_ROWS)
        {
            return Ok(Outcome::Failed(
                "the chunked second-turn child did not commit its prefill".into(),
            ));
        }
    }
    let mut chunked_prefill_output =
        Vec::with_capacity(SECOND_TURN_PREFILL_ROWS as usize * lane_bytes);
    let mut first = CONTEXT;
    while first < CONTEXT + SECOND_TURN_PREFILL_ROWS {
        let rows = chunk_rows.min(CONTEXT + SECOND_TURN_PREFILL_ROWS - first);
        let launch = second_prefill.at(rows, first)?;
        let output = chunked_turn
            .attend(&stream, &launch, fixture.query_bytes(first, rows))
            .map_err(|r| r.error)?;
        chunked_prefill_output.extend_from_slice(&output);
        first += rows;
    }
    if chunked_prefill_output != whole_prefill_output {
        return Ok(Outcome::Failed(
            "whole and chunked second-turn prefill outputs differ".into(),
        ));
    }
    {
        let mut branch = whole_state.branch(chunked_child_id)?;
        let txn = branch.begin()?;
        let first = branch.published_rows()?;
        if first != decode_position {
            return Ok(Outcome::Failed(format!(
                "the chunked second-turn decode row began at {first}, not {decode_position}"
            )));
        }
        let (keys, values) = fixture.payload(first, 1);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            1,
            &mut chunked_turn,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            1,
            &mut chunked_turn,
            &stream,
        )?;
        let child_decode_tail = branch.placement_of(0, decode_position)?;
        let expected_new_page = CONTEXT / geometry.page_tokens;
        if child_decode_tail.physical_page != expected_new_page
            || child_decode_tail.slot != geometry.page_tokens - 1
            || branch.committed_rows()? != CONTEXT + SECOND_TURN_ROWS
            || branch.retained(0)? != (0..CONTEXT + SECOND_TURN_ROWS)
        {
            return Ok(Outcome::Failed(format!(
                "the chunked second-turn decode row was not the page tail: \
                 placement={child_decode_tail:?}, committed={} retained={:?}",
                branch.committed_rows()?,
                branch.retained(0)?
            )));
        }
    }
    let chunked_decode_output = chunked_turn
        .attend(
            &stream,
            &second_decode,
            fixture.query_bytes(decode_position, 1),
        )
        .map_err(|r| r.error)?;
    if chunked_decode_output != whole_decode_output {
        return Ok(Outcome::Failed(
            "whole and chunked second-turn decode outputs differ".into(),
        ));
    }
    check_attention(
        &fixture,
        &second_decode,
        &chunked_decode_output,
        "32k-second-turn-chunked-decode",
    )?;
    let chunked_turn_arena_bytes = chunked_turn.arena_bytes();
    if whole_turn_arena_bytes != chunked_turn_arena_bytes
        || parent_arena_bytes != whole_turn_arena_bytes
    {
        return Ok(Outcome::Failed(format!(
            "second-turn arena sizes differ: parent={parent_arena_bytes} B, \
             whole_child={whole_turn_arena_bytes} B, chunked_child={chunked_turn_arena_bytes} B"
        )));
    }
    if whole.read_rows(&parent_placements)? != parent_before
        || whole_state.committed_rows()? != parent_frontier_before_turn
        || whole_state.retained(0)? != parent_retained_before_turn
    {
        return Ok(Outcome::Failed(
            "chunked second-turn execution changed the parent's bytes or frontier".into(),
        ));
    }
    let parent_frontier_before_chunked_discard = whole_state.committed_rows()?;
    let parent_retained_before_chunked_discard = whole_state.retained(0)?;
    whole_state.discard_branch(chunked_child_id)?;
    if whole_state.committed_rows()? != parent_frontier_before_chunked_discard
        || whole_state.retained(0)? != parent_retained_before_chunked_discard
        || whole.read_rows(&parent_placements)? != parent_before
    {
        return Ok(Outcome::Failed(
            "discarding the chunked second-turn child changed the parent".into(),
        ));
    }
    chunked_turn.close(&mut ledger).map_err(|r| r.error)?;
    println!(
        "    {} 32k-second-turn prefill_rows={} decode_rows=1 total={} first={first_second_summary} tail={last_second_summary} decode={second_decode_summary} parent_bytes={} B child_arena_whole={} B child_arena_chunked={} B",
        cap.sm(),
        SECOND_TURN_PREFILL_ROWS,
        CONTEXT + SECOND_TURN_ROWS,
        parent_before.len(),
        whole_turn_arena_bytes,
        chunked_turn_arena_bytes,
    );

    // Append row 32,768 -- the row after the context -- and decode it.
    append_authority_rows(&mut whole_state, &mut whole, &stream, &fixture, 1)?;
    if whole_state.committed_rows()? != CONTEXT + 1 {
        return Ok(Outcome::Failed(format!(
            "the frontier is {} after appending row {CONTEXT}",
            whole_state.committed_rows()?
        )));
    }
    let next = decode(CONTEXT, CONTEXT + 1, Visibility::Causal)?;
    let after = whole
        .attend(&stream, &next, fixture.query_bytes(CONTEXT, 1))
        .map_err(|r| r.error)?;
    let summary = check_attention(&fixture, &next, &after, "32k-append-decode")?;
    println!(
        "    {} 32k-append-decode visible={} committed={} capacity={} {summary}",
        cap.sm(),
        next.history_rows(),
        whole_state.committed_rows()?,
        whole.capacity_rows()?
    );

    // The three numbers are different, and a sliding launch is where that
    // becomes visible: the same committed history, a fraction of it visible.
    let windowed = decode(
        CONTEXT,
        CONTEXT + 1,
        Visibility::SlidingWindow { window: 4_096 },
    )?;
    let slid = whole
        .attend(&stream, &windowed, fixture.query_bytes(CONTEXT, 1))
        .map_err(|r| r.error)?;
    let summary = check_attention(&fixture, &windowed, &slid, "32k-sliding")?;
    let visible = (0..=CONTEXT).filter(|k| windowed.allows(0, *k)).count();
    if visible != 4_096 || whole_state.committed_rows()? != CONTEXT + 1 {
        return Ok(Outcome::Failed(format!(
            "{visible} visible row(s) against {} committed",
            whole_state.committed_rows()?
        )));
    }
    if slid == after {
        return Ok(Outcome::Failed(
            "a 4,096-row window produced the same answer as the whole history".into(),
        ));
    }
    println!(
        "    {} 32k-sliding visible={visible} committed={} capacity={} {summary}",
        cap.sm(),
        whole_state.committed_rows()?,
        whole.capacity_rows()?
    );

    whole.close(&mut ledger).map_err(|r| r.error)?;
    chunked.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed("a closed run left bytes charged".into()));
    }
    Ok(Outcome::Passed)
}

/// Task 0038 acceptance 2: one state lifecycle after the page ring wraps.
fn paged_attention_state_lifecycle(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const INITIAL_ROWS: u64 = 100;
    const MAX_ROWS: u64 = 128;
    const WINDOW: usize = 16;
    const TENTATIVE: usize = 8;

    let geometry = PageGeometry {
        kv_heads: 2,
        head_dim: 64,
        page_tokens: 8,
        pages: 4,
    };
    let heads = 4;
    let layer = AttentionLayer {
        geometry,
        heads,
        scale: moxie_plan::reciprocal_sqrt_scale(geometry.head_dim),
        visibility: Visibility::SlidingWindow {
            window: WINDOW as u64,
        },
    };
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();
    let probe = PagedAttentionLaunch::new(layer, 1, 0, 0, 1)?;
    let descriptor = select_paged_attention_kernel(&catalogue, cap, &probe)?;
    let mut ledger = measured_ledger(&ctx)?;
    let mut run = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        descriptor,
        geometry,
        heads,
        TENTATIVE as u64,
        MAX_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let mut state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::Window { window: WINDOW },
        }],
        precision: Precision::Bf16,
        page_tokens: geometry.page_tokens as usize,
        max_tokens: MAX_ROWS as usize,
        tentative_rows: TENTATIVE,
    })?;
    if state.layout(0)?.pages != geometry.pages {
        return Ok(Outcome::Failed(format!(
            "state admitted {} pages but the run admitted {}",
            state.layout(0)?.pages,
            geometry.pages
        )));
    }

    let mut fixture = AttentionFixture::build(geometry, heads, MAX_ROWS, 0x0038_0001);
    let mut written = 0;
    while written < INITIAL_ROWS {
        let rows = (INITIAL_ROWS - written).min(TENTATIVE as u64);
        append_authority_rows(&mut state, &mut run, &stream, &fixture, rows)?;
        written += rows;
    }

    let retained = state.retained(0)?;
    if retained.start == 0 || retained != (80..INITIAL_ROWS) {
        return Ok(Outcome::Failed(format!(
            "the wrapped sequence retained {retained:?}, expected 80..{INITIAL_ROWS}"
        )));
    }
    let before_abort = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        99,
        "state-before-abort",
    )?;

    // Write a real tentative append, then abort it. The physical run may keep
    // the observed high-water mark, but the authority must restore its own
    // frontier and the previously checked answer bit for bit.
    let txn = state.begin()?;
    let first = state.published_rows()?;
    let (keys, values) = fixture.payload(first, 4);
    moxie_executor::paged_attention::device::append_paged_layer(
        &mut state,
        txn,
        0,
        4,
        &mut run,
        &stream,
        moxie_executor::paged_attention::device::PagedKvRows { keys, values },
    )
    .map_err(|refused| refused.error)?;
    if state.committed_rows()? != INITIAL_ROWS || state.published_rows()? != INITIAL_ROWS + 4 {
        return Ok(Outcome::Failed(
            "a tentative append was reported as committed or not published".into(),
        ));
    }
    state.abort(txn)?;
    if state.committed_rows()? != INITIAL_ROWS
        || state.published_rows()? != INITIAL_ROWS
        || state.retained(0)? != retained
    {
        return Ok(Outcome::Failed(
            "abort did not restore the frontier and retained range".into(),
        ));
    }
    let after_abort = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        99,
        "state-after-abort",
    )?;
    if after_abort != before_abort {
        return Ok(Outcome::Failed(
            "abort changed the answer over committed history".into(),
        ));
    }

    // A causal decode at 87 is independent of the suffix above it. Truncating
    // to 88 must therefore leave this already-oracle-checked answer exact.
    let before_truncate = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        87,
        "state-before-truncate",
    )?;
    let old_suffix = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        95,
        "state-old-suffix",
    )?;
    state.truncate(88)?;
    let after_truncate = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        87,
        "state-after-truncate",
    )?;
    if after_truncate != before_truncate {
        return Ok(Outcome::Failed(
            "truncate changed the retained prefix's answer".into(),
        ));
    }

    // Replace positions 88..96 with a different deterministic suffix, append
    // it through the authority, and check every resulting component against
    // the FP64 oracle built from that mixed prefix/suffix history.
    let replacement = AttentionFixture::build(geometry, heads, MAX_ROWS, 0x0038_9001);
    let row_width = (geometry.kv_heads * geometry.head_dim) as usize;
    let start = 88 * row_width;
    let end = 96 * row_width;
    fixture.keys[start..end].copy_from_slice(&replacement.keys[start..end]);
    fixture.values[start..end].copy_from_slice(&replacement.values[start..end]);
    append_authority_rows(&mut state, &mut run, &stream, &fixture, 8)?;
    let after_reappend = attend_authority(
        &mut run,
        &stream,
        &state,
        &fixture,
        layer,
        95,
        "state-reappend",
    )?;
    if after_reappend == old_suffix {
        return Ok(Outcome::Failed(
            "reappend answered with the suffix that truncation discarded".into(),
        ));
    }
    if state.retained(0)?.start == 0 {
        return Ok(Outcome::Failed(
            "the lifecycle finished without a reclaimed history base".into(),
        ));
    }

    let graph = selected_attention_graph(
        heads,
        geometry.kv_heads,
        geometry.head_dim,
        layer.scale,
        layer.visibility,
    )?;
    let workload = ResourceWorkload {
        phase: Phase::Decode,
        rows: 1,
        visible_tokens: WINDOW as u64,
        branch_rows: 1,
        output: graph.output(),
        device: ctx.uuid(),
        paged_state_capacity: None,
    };
    let candidate = lower_selected(&graph, workload, cap, &catalogue)?;
    let mut plan =
        match SelectedReservedPlan::admit(candidate, &graph, cap, &catalogue, &mut ledger, &ctx) {
            Ok(plan) => plan,
            Err(
                SelectedAdmitRefused::Invalid { error, .. }
                | SelectedAdmitRefused::Held { error, .. },
            ) => return Err(error),
            Err(SelectedAdmitRefused::Rejected { rejection, .. }) => {
                return Err(Error::CapacityExceeded {
                    tier: None,
                    requested_bytes: rejection.shortfall_bytes,
                    available_bytes: 0,
                });
            }
        };
    let position = state.published_rows()?;
    let transaction = state.begin()?;
    let (keys, values) = fixture.payload(position, 1);
    plan.execute_paged_attention(PagedAttentionStep {
        graph: &graph,
        capability: cap,
        catalogue: &catalogue,
        ctx: &ctx,
        stream: &stream,
        state: &mut state,
        transaction,
        run: &mut run,
        inputs: PagedAttentionInputs {
            query: fixture.query_bytes(position, 1),
            keys,
            values,
            positions: vec![position],
        },
    })
    .map_err(|refused| refused.error)?;
    let retained = state.layer_retained(0)?;
    let launch = PagedAttentionLaunch::new(
        layer,
        1,
        position,
        retained.start,
        retained.end - retained.start,
    )?;
    let mut selected_output = vec![0; launch.output_bytes()? as usize];
    plan.read_paged_attention_output(&mut selected_output)?;
    check_attention(
        &fixture,
        &launch,
        &selected_output,
        "selected-state-lifecycle",
    )?;
    moxie_executor::paged_attention::device::commit_paged_layer(
        &mut state,
        transaction,
        1,
        &mut run,
        &stream,
    )?;
    plan.close(&mut ledger).map_err(|refused| refused.error)?;

    println!(
        "    {} state-lifecycle base={} committed={} written={} capacity={}",
        cap.sm(),
        state.retained(0)?.start,
        state.committed_rows()?,
        run.written_rows(),
        run.capacity_rows()?
    );
    run.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "a closed lifecycle run left bytes charged".into(),
        ));
    }
    Ok(Outcome::Passed)
}

/// Task 0045: eager device COW fork.
///
/// `DeviceKvSequence` creates the child decisions first, then the executor
/// copies the parent's complete admitted page storage through the existing
/// writer callback. The proof reads back device bytes before and after child
/// append/truncate/reappend operations; counters alone are deliberately not
/// accepted as evidence.
fn paged_attention_device_cow(cap: &DeviceCapability) -> Result<Outcome, Error> {
    const FORK_AT: u64 = 8;
    const PAGE_TOKENS: u64 = 4;
    const PAGES: u64 = 4;
    const MAX_ROWS: u64 = PAGE_TOKENS * PAGES;
    let geometry = PageGeometry {
        kv_heads: 2,
        head_dim: 64,
        page_tokens: PAGE_TOKENS,
        pages: PAGES,
    };
    let heads = 4;
    let layer = AttentionLayer {
        geometry,
        heads,
        scale: moxie_plan::reciprocal_sqrt_scale(geometry.head_dim),
        visibility: Visibility::Causal,
    };
    let ctx = RankContext::acquire(RankId(cap.ordinal), cap.ordinal)?;
    let stream = Stream::new(&ctx)?;
    let catalogue = moxie_kernels::paged_attention_catalogue();
    let probe = PagedAttentionLaunch::new(layer, 1, 0, 0, 1)?;
    let descriptor = select_paged_attention_kernel(&catalogue, cap, &probe)?;
    let mut ledger = measured_ledger(&ctx)?;
    let mut parent = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        FORK_AT,
        MAX_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let mut state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: MAX_ROWS as usize,
        tentative_rows: MAX_ROWS as usize,
    })?;
    let parent_fixture = AttentionFixture::build(geometry, heads, MAX_ROWS, 0x0045_0001);
    append_authority_rows(&mut state, &mut parent, &stream, &parent_fixture, FORK_AT)?;
    let parent_frontier_before_child = state.committed_rows()?;
    let parent_retained_before_child = state.retained(0)?;
    let parent_placements = device_state_row_placements(&state, 0, FORK_AT)?;
    let parent_before = parent.read_rows(&parent_placements)?;

    let mut child = PagedAttentionRun::admit_for_fork(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        FORK_AT,
        MAX_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let child_id = moxie_executor::paged_attention::device::fork_paged_layer(
        &mut state, FORK_AT, &parent, &mut child, &stream,
    )?;
    let child_before = {
        let branch = state.branch(child_id)?;
        let placements = device_branch_row_placements(&branch, 0, FORK_AT)?;
        child.read_rows(&placements)?
    };
    if child_before != parent_before {
        return Ok(Outcome::Failed(
            "device fork did not reproduce the parent's inherited bytes".into(),
        ));
    }

    // The child gets a different suffix, then truncates and diverges again.
    // Neither operation may alter the inherited prefix in its own copy.
    let child_fixture = AttentionFixture::build(geometry, heads, MAX_ROWS, 0x0045_1001);
    let child_fixture_after_truncate =
        AttentionFixture::build(geometry, heads, MAX_ROWS, 0x0045_2001);
    {
        let mut branch = state.branch(child_id)?;
        let txn = branch.begin()?;
        let (keys, values) = child_fixture.payload(FORK_AT, 1);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            1,
            &mut child,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            1,
            &mut child,
            &stream,
        )?;
        branch.truncate(FORK_AT)?;
        let txn = branch.begin()?;
        let (keys, values) = child_fixture_after_truncate.payload(FORK_AT, 1);
        moxie_executor::paged_attention::device::append_paged_branch(
            &mut branch,
            txn,
            0,
            1,
            &mut child,
            &stream,
            moxie_executor::paged_attention::device::PagedKvRows { keys, values },
        )
        .map_err(|refused| refused.error)?;
        moxie_executor::paged_attention::device::commit_paged_branch(
            &mut branch,
            txn,
            1,
            &mut child,
            &stream,
        )?;
        let placements = device_branch_row_placements(&branch, 0, FORK_AT)?;
        if child.read_rows(&placements)? != parent_before {
            return Ok(Outcome::Failed(
                "child divergence changed inherited device bytes".into(),
            ));
        }
    }
    if state.committed_rows()? != parent_frontier_before_child
        || state.retained(0)? != parent_retained_before_child
    {
        return Ok(Outcome::Failed(
            "child operations changed the parent's frontier or retained range".into(),
        ));
    }

    // The parent now writes its own suffix. Read its original prefix again
    // after the child has diverged, which is the other half of isolation.
    append_authority_rows(&mut state, &mut parent, &stream, &parent_fixture, 1)?;
    let parent_after = parent.read_rows(&device_state_row_placements(&state, 0, FORK_AT)?)?;
    if parent_after != parent_before {
        return Ok(Outcome::Failed(
            "child divergence changed the parent's device bytes".into(),
        ));
    }
    let parent_frontier_before_discard = state.committed_rows()?;
    let parent_retained_before_discard = state.retained(0)?;
    state.discard_branch(child_id)?;
    if state.committed_rows()? != parent_frontier_before_discard
        || state.retained(0)? != parent_retained_before_discard
    {
        return Ok(Outcome::Failed(
            "discarding the child changed the parent's frontier or retained range".into(),
        ));
    }
    child.close(&mut ledger).map_err(|r| r.error)?;
    let parent_after_discard = parent.read_rows(&parent_placements)?;
    parent.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "normal device fork cleanup left bytes charged".into(),
        ));
    }
    if parent_after_discard != parent_before {
        return Ok(Outcome::Failed(
            "discarding the child changed the parent's device bytes".into(),
        ));
    }

    // A wrapped, windowed parent carries a retained floor into the child. The
    // child must copy the visible device pages, while a placement below that
    // floor remains a typed reclaimed-row refusal rather than reading stale
    // bytes from the copied ring.
    let window_geometry = PageGeometry {
        kv_heads: 2,
        head_dim: 64,
        page_tokens: PAGE_TOKENS,
        pages: 3,
    };
    let window_layer = AttentionLayer {
        geometry: window_geometry,
        heads,
        scale: moxie_plan::reciprocal_sqrt_scale(window_geometry.head_dim),
        visibility: Visibility::SlidingWindow { window: 4 },
    };
    let window_descriptor = select_paged_attention_kernel(
        &catalogue,
        cap,
        &PagedAttentionLaunch::new(window_layer, 1, 0, 0, 1)?,
    )?;
    let mut window_parent = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        window_descriptor.try_clone()?,
        window_geometry,
        heads,
        4,
        16u64
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let mut window_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: window_geometry.kv_heads as usize,
            key_dim: window_geometry.head_dim as usize,
            value_dim: window_geometry.head_dim as usize,
            retention: moxie_state::Retention::Window { window: 4 },
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: 16,
        tentative_rows: 4,
    })?;
    let window_fixture = AttentionFixture::build(window_geometry, heads, 16, 0x0045_3001);
    for _ in 0..3 {
        append_authority_rows(
            &mut window_state,
            &mut window_parent,
            &stream,
            &window_fixture,
            4,
        )?;
    }
    let window_retained = window_state.retained(0)?;
    if window_retained != (4..12) {
        return Ok(Outcome::Failed(format!(
            "windowed device parent retained {window_retained:?}, expected 4..12"
        )));
    }
    let window_before = window_parent.read_rows(&device_state_row_placements(
        &window_state,
        window_retained.start,
        window_retained.end - window_retained.start,
    )?)?;
    let mut window_child = PagedAttentionRun::admit_for_fork(
        &mut ledger,
        &ctx,
        window_descriptor,
        window_geometry,
        heads,
        4,
        16u64
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let window_child_id = moxie_executor::paged_attention::device::fork_paged_layer(
        &mut window_state,
        12,
        &window_parent,
        &mut window_child,
        &stream,
    )?;
    {
        let window_branch = window_state.branch(window_child_id)?;
        if window_branch.placement_of(0, 0).is_ok() {
            return Ok(Outcome::Failed(
                "windowed child exposed a row below the parent's retained floor".into(),
            ));
        }
        let placements = device_branch_row_placements(&window_branch, 4, 8)?;
        if window_child.read_rows(&placements)? != window_before {
            return Ok(Outcome::Failed(
                "windowed child did not reproduce the retained device bytes".into(),
            ));
        }
    }
    window_state.discard_branch(window_child_id)?;
    window_child.close(&mut ledger).map_err(|r| r.error)?;
    window_parent.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "windowed device fork cleanup left bytes charged".into(),
        ));
    }

    // Fault after the first physical page copy. The logical branch has
    // already been created at this point; a passing cleanup assertion must
    // therefore cover both SequenceState and the child run's allocation.
    let mut fault_parent = PagedAttentionRun::admit_for_sequence(
        &mut ledger,
        &ctx,
        descriptor.try_clone()?,
        geometry,
        heads,
        PAGE_TOKENS,
        MAX_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    let mut fault_state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
        layers: vec![moxie_state::LayerKv {
            kv_heads: geometry.kv_heads as usize,
            key_dim: geometry.head_dim as usize,
            value_dim: geometry.head_dim as usize,
            retention: moxie_state::Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: PAGE_TOKENS as usize,
        max_tokens: MAX_ROWS as usize,
        tentative_rows: MAX_ROWS as usize,
    })?;
    append_authority_rows(
        &mut fault_state,
        &mut fault_parent,
        &stream,
        &parent_fixture,
        PAGE_TOKENS,
    )?;
    let fault_placements = device_state_row_placements(&fault_state, 0, PAGE_TOKENS)?;
    let fault_before = fault_parent.read_rows(&fault_placements)?;
    let fault_root_frontier = fault_state.committed_rows()?;
    let fault_root_retained = fault_state.retained(0)?;
    let parent_charge = ledger.outstanding_count();
    let mut fault_child = PagedAttentionRun::admit_for_fork(
        &mut ledger,
        &ctx,
        descriptor,
        geometry,
        heads,
        PAGE_TOKENS,
        MAX_ROWS
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
        Staging::Host,
    )
    .map_err(|r| r.error)?;
    fault_child.inject_branch_copy_failure_after(1);
    let fork_error = moxie_executor::paged_attention::device::fork_paged_layer(
        &mut fault_state,
        PAGE_TOKENS,
        &fault_parent,
        &mut fault_child,
        &stream,
    );
    if fork_error.is_ok() {
        return Ok(Outcome::Failed(
            "the injected post-fork device copy did not refuse".into(),
        ));
    }
    if fault_state.state()?.branch_ids() != vec![moxie_state::ROOT]
        || fault_state.committed_rows()? != fault_root_frontier
        || fault_state.retained(0)? != fault_root_retained
        || fault_parent.read_rows(&fault_placements)? != fault_before
    {
        return Ok(Outcome::Failed(
            "post-fork copy failure changed logical or parent state".into(),
        ));
    }
    fault_child.close(&mut ledger).map_err(|r| r.error)?;
    if ledger.outstanding_count() != parent_charge {
        return Ok(Outcome::Failed(
            "post-fork copy failure stranded the child charge".into(),
        ));
    }
    fault_parent.close(&mut ledger).map_err(|r| r.error)?;
    if !ledger.outstanding().is_empty() {
        return Ok(Outcome::Failed(
            "faulted device fork cleanup left bytes charged".into(),
        ));
    }
    let normal_copy_bytes = geometry
        .page_bytes()?
        .checked_mul(PAGES * 2)
        .and_then(|bytes| bytes.checked_add((FORK_AT / PAGE_TOKENS) * 4))
        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
    let window_copy_bytes = window_geometry
        .page_bytes()?
        .checked_mul(3 * 2)
        .and_then(|bytes| bytes.checked_add((8 / PAGE_TOKENS) * 4))
        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
    let fault_copy_bytes = geometry
        .page_bytes()?
        .checked_mul(2)
        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
    println!(
        "    {} device-cow inherited={} rows, copy_bytes={} window_copy_bytes={} fault_bytes_before_refusal={}, post-fork fault cleaned",
        cap.sm(),
        FORK_AT,
        normal_copy_bytes,
        window_copy_bytes,
        fault_copy_bytes
    );
    Ok(Outcome::Passed)
}

fn numerical(error: Error) -> Error {
    Error::Numerical {
        detail: format!("the canonical fixture is malformed: {error}"),
    }
}

#[cfg(test)]
mod nonfinite_metric_fixtures {
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
