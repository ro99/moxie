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
use moxie_types::{DeviceCapability, Error, RankId};

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
    "non_ptx_text_rejected",
    "rank_context_is_exclusive",
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
        results.push(case(&cap, "non_ptx_text_rejected", ptx_rejection(&cap)));
        results.push(case(
            &cap,
            "rank_context_is_exclusive",
            rank_exclusivity(&cap),
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

    let mut dx = DeviceBuffer::alloc(&ctx, N * 4)?;
    let mut dy = DeviceBuffer::alloc(&ctx, N * 4)?;

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
