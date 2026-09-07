//! The real-hardware lane: `cargo xtask test-gpu`.
//!
//! Document 07: "unsupported hardware is skipped/unmeasured, never passed" and
//! "a test that catches a device error then exits successfully is not a passing
//! production test". So this lane reports pass/fail/skip per device explicitly
//! and returns a non-zero exit code when any *attempted* case failed.

use core::ffi::c_void;

use moxie_cuda::{DeviceBuffer, DeviceContext, Module, query_device};
use moxie_types::{DeviceCapability, Error};

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

/// Run every M0 GPU case on every visible device.
pub fn run() -> i32 {
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
        if !moxie_kernels::qualified_sm().contains(&cap.sm()) {
            let why = format!(
                "{} is not among the compiled architectures ({}); UNMEASURED",
                cap.sm(),
                moxie_kernels::KERNEL_ARCHS
            );
            for name in ["axpy_f32", "bf16_round_trip", "arch_mismatch_is_typed"] {
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
    }

    println!("\n--- results ---");
    let mut failed = 0;
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
        if matches!(r.outcome, Outcome::Failed(_)) {
            failed += 1;
        }
    }

    // Exit gate: "one real CUDA launch on each installed architecture".
    println!("\narchitectures exercised: {}", seen_sm.join(", "));
    for arch in moxie_kernels::qualified_sm() {
        if !seen_sm.contains(&arch) {
            println!(
                "NOTE  {arch} is compiled but no installed device has it; \
                 that architecture is UNMEASURED, not passing"
            );
        }
    }

    if failed > 0 {
        println!("\n{failed} case(s) failed");
        1
    } else {
        println!("\nall attempted cases passed");
        0
    }
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
    let ctx = DeviceContext::new(cap.ordinal)?;
    let module = Module::load(&ctx, moxie_kernels::M0_SMOKE_FATBIN)?;
    let func = module.function(moxie_kernels::AXPY_F32)?;

    let x: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
    let y_in: Vec<f32> = (0..N).map(|i| (i as f32) * -0.25).collect();
    let a = 3.0f32;

    let mut dx = DeviceBuffer::alloc(&ctx, N * 4)?;
    let mut dy = DeviceBuffer::alloc(&ctx, N * 4)?;
    dx.copy_from_host(&ctx, bytemuck_f32(&x))?;
    dy.copy_from_host(&ctx, bytemuck_f32(&y_in))?;

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
    // SAFETY: the parameter list matches `moxie_m0_axpy_f32(const float*,
    // float*, float, unsigned)` in count, order and type. Both device pointers
    // address N*4 bytes, which is exactly what the kernel indexes for i < N.
    unsafe {
        func.launch_blocking(
            &ctx,
            (N.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )?;
    }

    let mut out = vec![0f32; N];
    dy.copy_to_host(&ctx, bytemuck_f32_mut(&mut out))?;

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

    let ctx = DeviceContext::new(cap.ordinal)?;
    let module = Module::load(&ctx, moxie_kernels::M0_SMOKE_FATBIN)?;
    let func = module.function(moxie_kernels::F32_TO_BF16_BITS)?;

    let mut dsrc = DeviceBuffer::alloc(&ctx, n * 4)?;
    let ddst = DeviceBuffer::alloc(&ctx, n * 2)?;
    dsrc.copy_from_host(&ctx, bytemuck_f32(&inputs))?;

    let mut ps = dsrc.device_ptr();
    let mut pd = ddst.device_ptr();
    let mut pn = n as u32;
    let mut params: [*mut c_void; 3] = [
        (&raw mut ps).cast(),
        (&raw mut pd).cast(),
        (&raw mut pn).cast(),
    ];
    // SAFETY: matches `moxie_m0_f32_to_bf16_bits(const float*, unsigned short*,
    // unsigned)`. Source holds n*4 bytes, destination n*2, and the kernel writes
    // one u16 per i < n.
    unsafe {
        func.launch_blocking(&ctx, (1, 1, 1), (n as u32, 1, 1), 0, &mut params)?;
    }

    let mut got = vec![0u16; n];
    ddst.copy_to_host(&ctx, bytemuck_u16_mut(&mut got))?;

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
    let ctx = DeviceContext::new(cap.ordinal)?;
    let r = Module::load(&ctx, moxie_kernels::M0_SMOKE_FATBIN_SM86_ONLY);
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

// Small local reinterpretation helpers. Deliberately not a dependency: these are
// the only three shapes M0 needs.
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
