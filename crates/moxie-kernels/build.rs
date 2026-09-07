//! Compile the smoke kernels to fatbins, one per target set.
//!
//! Two images are produced deliberately:
//!
//! * `smoke.fatbin`      -- every architecture the product targets.
//! * `smoke_sm86.fatbin` -- SM86 only.
//!
//! The second is not redundant. Document 03 warns that "SM100/Hopper recipes are
//! not automatically compatible" with SM120 and that the 5060 Ti must be
//! qualified separately. The SM86-only image gives the GPU lane a *real*
//! architecture mismatch to load, proving that such a mismatch surfaces as a
//! typed `UnsupportedKernel` rather than silently doing nothing.
//!
//! Architectures are listed here explicitly rather than discovered, so that the
//! build does not quietly change when run on a different machine.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Compute capabilities this build targets. Changing this list is a support
/// matrix change (document 07), not a build tweak.
const ARCHS: &[&str] = &["86", "120"];

fn main() {
    println!("cargo:rerun-if-changed=cuda/smoke.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| {
        let home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        format!("{home}/bin/nvcc")
    });

    if !PathBuf::from(&nvcc).exists() {
        // A missing toolkit must be a loud, specific failure. Document 07: a
        // skipped GPU lane is never a passing result, so we do not emit an empty
        // fatbin and carry on.
        panic!(
            "nvcc not found at {nvcc}. Set NVCC or CUDA_HOME. \
             moxie-kernels cannot be built without the pinned CUDA toolkit."
        );
    }

    build_fatbin(&nvcc, &out, "smoke.fatbin", ARCHS);
    build_fatbin(&nvcc, &out, "smoke_sm86.fatbin", &["86"]);

    println!(
        "cargo:rustc-env=MOXIE_SMOKE_FATBIN={}",
        out.join("smoke.fatbin").display()
    );
    println!(
        "cargo:rustc-env=MOXIE_SMOKE_FATBIN_SM86={}",
        out.join("smoke_sm86.fatbin").display()
    );
    println!("cargo:rustc-env=MOXIE_KERNEL_ARCHS={}", ARCHS.join(","));
}

fn build_fatbin(nvcc: &str, out: &Path, name: &str, archs: &[&str]) {
    let dst = out.join(name);
    let mut cmd = Command::new(nvcc);
    cmd.arg("-fatbin").arg("-O3").arg("--std=c++17");
    for a in archs {
        // Embed both SASS (`sm_NN`) and PTX (`compute_NN`). The PTX lets a
        // future architecture JIT rather than fail outright; the SASS is what
        // actually runs on the architectures we qualified.
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{a},code=sm_{a}"));
    }
    cmd.arg("cuda/smoke.cu").arg("-o").arg(&dst);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed building {name}: {cmd:?}");
}
