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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=archs.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

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
    // part of the build identity document 07 requires; it is reported, not
    // pinned, because nvcc chooses it from the environment.
    let host_cc = std::env::var("CUDAHOSTCXX")
        .or_else(|_| std::env::var("NVCC_CCBIN"))
        .unwrap_or_else(|_| "c++".to_string());
    let host_cc_version = tool_version(&host_cc, &["--version"]);

    check_known_answers();
    let full = build_fatbin(&nvcc, &out, "smoke.fatbin", ARCHS);
    let sm86 = build_fatbin(&nvcc, &out, "smoke_sm86.fatbin", &["86"]);

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
fn build_fatbin(nvcc: &str, out: &Path, name: &str, archs: &[&str]) -> String {
    let dst = out.join(name);
    let mut cmd = Command::new(nvcc);
    cmd.arg("-fatbin").arg("-O3").arg("--std=c++17");
    for a in archs {
        // SASS only, on purpose. See the module comment: adding `code=compute_NN`
        // would embed PTX and let the driver JIT the SM86-only image onto SM120,
        // destroying the architecture-mismatch assertion.
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{a},code=sm_{a}"));
    }
    cmd.arg("cuda/smoke.cu").arg("-o").arg(&dst);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed building {name}: {cmd:?}");

    let bytes =
        std::fs::read(&dst).unwrap_or_else(|e| panic!("cannot read {}: {e}", dst.display()));
    sha256_hex(&bytes)
}

// --- SHA-256, so the build has no third-party dependency and no shell-out ---
//
// Document 07 requires a recorded image identity. FIPS 180-4; the constants are
// the standard ones, and `check_known_answers` runs the two standard vectors
// before any image is hashed -- a home-grown digest that has never been checked
// against a published vector is not an identity.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// FIPS 180-4 published vectors, checked at build time.
fn check_known_answers() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Multi-block, to exercise the padding path an image will take.
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (dst, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *dst = dst.wrapping_add(v);
        }
    }

    h.iter().map(|w| format!("{w:08x}")).collect()
}
