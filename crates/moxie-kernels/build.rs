//! Compile the smoke kernels to fatbins, one per target set.
//!
//! Runs only with the `fatbin` feature. Without it this script emits nothing, so
//! the host lane builds and tests this crate with no CUDA toolkit present
//! (document 07). The GPU lane turns the feature on and gets a hard failure if
//! the toolkit is missing or is not the pinned version.
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
//! Both images are **SASS only**: `-gencode arch=compute_NN,code=sm_NN` embeds
//! cubin for `sm_NN` and no PTX. That is deliberate and load-bearing for the
//! mismatch case -- embedding `code=compute_NN` as well would let the driver JIT
//! the SM86-only image onto an SM120 device, and the architecture-mismatch
//! assertion would silently stop asserting anything. An earlier comment here
//! claimed PTX was embedded; it never was.

use std::path::{Path, PathBuf};
use std::process::Command;

include!("archs.rs");

/// The pinned toolkit, from docs/evidence/toolchain.md. A freely overridable
/// `NVCC` path does not pin a toolchain; checking what the binary actually
/// reports does. Changing this is an ADR, not a build tweak.
const PINNED_NVCC_RELEASE: &str = "release 13.0, V13.0.88";

fn main() {
    println!("cargo:rerun-if-changed=cuda/smoke.cu");
    println!("cargo:rerun-if-changed=cuda/bf16_chain.cu");
    println!("cargo:rerun-if-changed=cuda/expert_mlp.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=archs.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    println!(
        "cargo:rustc-env=MOXIE_KERNEL_TARGET_ARCHS={}",
        ARCHS.join(",")
    );

    if std::env::var_os("CARGO_FEATURE_FATBIN").is_none() {
        return;
    }

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

    let nvcc_version = tool_version(&nvcc, &["--version"]);
    assert!(
        nvcc_version.contains(PINNED_NVCC_RELEASE),
        "nvcc at {nvcc} reports {nvcc_version:?}, which does not contain the pinned \
         {PINNED_NVCC_RELEASE:?}. docs/evidence/toolchain.md pins the toolkit; \
         changing it is an ADR."
    );

    // nvcc drives a host compiler for the C++ it emits. Recording which one is
    // part of the build identity document 07 requires -- but only if the
    // recorded compiler is the one actually used. The first version read
    // `CUDAHOSTCXX` to *describe* the compiler and then never passed that
    // selection to nvcc, so the record was a guess about nvcc's default. It is
    // now passed explicitly with `-ccbin`, which makes the recorded version a
    // fact about this build rather than an assumption about the environment.
    let host_cc = std::env::var("CUDAHOSTCXX")
        .or_else(|_| std::env::var("NVCC_CCBIN"))
        .unwrap_or_else(|_| "c++".to_string());
    let host_cc_version = tool_version(&host_cc, &["--version"]);

    // The known-answer vectors live as unit tests beside the canonical
    // implementation in `moxie-format::sha256`; they run in the host lane.
    // Re-checking them here as well keeps a corrupt include loud at image
    // build time rather than silent.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let full = build_fatbin(&nvcc, &host_cc, &out, "smoke.fatbin", ARCHS);
    let sm86 = build_fatbin(&nvcc, &host_cc, &out, "smoke_sm86.fatbin", &["86"]);
    let semantic = build_source_fatbin(
        &nvcc,
        &host_cc,
        &out,
        "bf16_chain.fatbin",
        "cuda/bf16_chain.cu",
        ARCHS,
    );

    let experts = build_source_fatbin(
        &nvcc,
        &host_cc,
        &out,
        "expert_mlp.fatbin",
        "cuda/expert_mlp.cu",
        ARCHS,
    );

    println!(
        "cargo:rustc-env=MOXIE_SMOKE_FATBIN={}",
        out.join("smoke.fatbin").display()
    );
    println!(
        "cargo:rustc-env=MOXIE_SMOKE_FATBIN_SM86={}",
        out.join("smoke_sm86.fatbin").display()
    );
    println!("cargo:rustc-env=MOXIE_KERNEL_ARCHS={}", ARCHS.join(","));
    println!("cargo:rustc-env=MOXIE_SMOKE_FATBIN_SHA256={full}");
    println!("cargo:rustc-env=MOXIE_SMOKE_FATBIN_SM86_SHA256={sm86}");
    println!(
        "cargo:rustc-env=MOXIE_BF16_CHAIN_FATBIN={}",
        out.join("bf16_chain.fatbin").display()
    );
    println!("cargo:rustc-env=MOXIE_BF16_CHAIN_FATBIN_SHA256={semantic}");
    println!(
        "cargo:rustc-env=MOXIE_EXPERT_MLP_FATBIN={}",
        out.join("expert_mlp.fatbin").display()
    );
    println!("cargo:rustc-env=MOXIE_EXPERT_MLP_FATBIN_SHA256={experts}");
    println!(
        "cargo:rustc-env=MOXIE_NVCC_VERSION={}",
        one_line(&nvcc_version)
    );
    println!(
        "cargo:rustc-env=MOXIE_HOST_COMPILER_VERSION={}",
        one_line(&host_cc_version)
    );
}

/// Run `tool args...` and return its combined version output.
fn tool_version(tool: &str, args: &[&str]) -> String {
    let out = Command::new(tool)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("cannot run {tool} {args:?}: {e}"));
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

/// The most informative single line of a version banner, for an env var.
fn one_line(version: &str) -> String {
    version
        .lines()
        .find(|l| l.contains("release") || l.contains("version"))
        .unwrap_or_else(|| version.lines().next().unwrap_or(""))
        .trim()
        .to_string()
}

/// Compile one image and return its SHA-256, lowercase hex.
fn build_fatbin(nvcc: &str, host_cc: &str, out: &Path, name: &str, archs: &[&str]) -> String {
    build_source_fatbin(nvcc, host_cc, out, name, "cuda/smoke.cu", archs)
}

fn build_source_fatbin(
    nvcc: &str,
    host_cc: &str,
    out: &Path,
    name: &str,
    source: &str,
    archs: &[&str],
) -> String {
    let dst = out.join(name);
    let mut cmd = Command::new(nvcc);
    // `-ccbin` names the host compiler explicitly, so `HOST_COMPILER_VERSION`
    // records the one that actually ran.
    cmd.arg("-ccbin").arg(host_cc);
    cmd.arg("-fatbin").arg("-O3").arg("--std=c++17");
    for a in archs {
        // SASS only, on purpose. See the module comment: adding `code=compute_NN`
        // would embed PTX and let the driver JIT the SM86-only image onto SM120,
        // destroying the architecture-mismatch assertion.
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{a},code=sm_{a}"));
    }
    cmd.arg(source).arg("-o").arg(&dst);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed building {name}: {cmd:?}");

    let bytes =
        std::fs::read(&dst).unwrap_or_else(|e| panic!("cannot read {}: {e}", dst.display()));
    sha256_hex(&bytes)
}

// --- SHA-256: the single canonical implementation lives in
// `crates/moxie-format/src/sha256_raw.rs` (task 0005's deletion plan). This build
// script includes that file rather than duplicating it, so the digest that
// identifies a kernel image is textually the same function that checks a
// manifest checksum. `grep "fn sha256_hex"` finds one definition.
include!("../moxie-format/src/sha256_raw.rs");
