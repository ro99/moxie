//! `cargo xtask spec-check` -- is the normative specification actually here?
//!
//! `docs/spec/01-09` and the architectural diagnosis are kept local and
//! untracked at the owner's direction (`docs/README.md`), so a fresh clone has
//! the living records and **not** the specification. Nothing previously noticed
//! that. An agent in such a clone would read AGENTS.md, fail to open the
//! documents it points at, and carry on inferring the contracts -- which is R25
//! happening again.
//!
//! This command does not change the publication policy, and deliberately does
//! not copy, fetch or generate anything. It reports presence and content
//! digests against a tracked manifest, so that a missing or altered normative
//! document is a loud failure at setup rather than a silent divergence later.
//!
//! Updating the manifest after an ADR amends a document is the intended
//! workflow: `cargo xtask spec-check --update` rewrites it, and the diff is what
//! review looks at.

/// The tracked digest manifest, relative to the workspace root.
const MANIFEST: &str = "docs/evidence/specification-version.md";

/// Documents that must exist. Absence of any one of them fails the check.
const REQUIRED: &[&str] = &[
    "docs/spec/01-product-and-decisions.md",
    "docs/spec/02-architecture-and-common-api.md",
    "docs/spec/03-memory-formats-and-cuda.md",
    "docs/spec/04-attention-parallelism-and-speculation.md",
    "docs/spec/05-sampling-api-and-cli.md",
    "docs/spec/06-implementation-roadmap.md",
    "docs/spec/07-validation-and-performance.md",
    "docs/spec/08-strata-reference-map.md",
    "docs/spec/09-agent-playbooks.md",
    "docs/spec/strata-arch-diagnosis.md",
];

pub fn run(update: bool) -> i32 {
    let root = super::workspace_root();
    let manifest_path = root.join(MANIFEST);

    let mut present: Vec<(String, String)> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    for rel in REQUIRED {
        match std::fs::read(root.join(rel)) {
            Ok(bytes) => present.push((rel.to_string(), sha256_hex(&bytes))),
            Err(_) => missing.push(rel),
        }
    }

    if update {
        if !missing.is_empty() {
            eprintln!(
                "refusing to update the manifest: {} document(s) missing",
                missing.len()
            );
            for m in &missing {
                eprintln!("  {m}");
            }
            return 1;
        }
        let body = render_manifest(&present);
        if let Err(e) = std::fs::write(&manifest_path, body) {
            eprintln!("cannot write {}: {e}", manifest_path.display());
            return 1;
        }
        println!("wrote {} ({} document(s))", MANIFEST, present.len());
        return 0;
    }

    let recorded = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => parse_manifest(&t),
        Err(e) => {
            println!("FAIL  cannot read {MANIFEST}: {e}");
            println!("      Run `cargo xtask spec-check --update` with the pack in place.");
            return 1;
        }
    };

    let mut failed = 0;
    for m in &missing {
        println!("FAIL  {m} is missing");
        failed += 1;
    }
    if !missing.is_empty() {
        println!(
            "\nThe specification is kept local and untracked (docs/README.md), so a fresh\n\
             clone does not have it. Copy the pack in before implementing anything: these\n\
             documents are normative, and inferring their contracts from the code is what\n\
             R25 records as a cause of the legacy project's drift."
        );
    }

    for (rel, digest) in &present {
        match recorded.iter().find(|(r, _)| r == rel) {
            None => {
                println!("FAIL  {rel} is present but not in {MANIFEST}");
                failed += 1;
            }
            Some((_, want)) if want != digest => {
                println!("FAIL  {rel} differs from the recorded digest");
                println!("      recorded {want}");
                println!("      actual   {digest}");
                failed += 1;
            }
            Some(_) => println!("PASS  {rel}"),
        }
    }
    for (rel, _) in &recorded {
        if !REQUIRED.contains(&rel.as_str()) {
            println!("FAIL  {MANIFEST} records {rel}, which is not a required document");
            failed += 1;
        }
    }

    if failed > 0 {
        println!(
            "\n{failed} failure(s). A document that changed without an ADR is an amendment\n\
             by edit, which docs/README.md forbids: \"Do not edit spec/01-09 to match what\n\
             was built.\" If an ADR did amend one, re-run with --update and commit the diff."
        );
        1
    } else {
        println!(
            "\nspec-check passed: {} document(s) present and unchanged",
            present.len()
        );
        0
    }
}

fn render_manifest(entries: &[(String, String)]) -> String {
    let mut s = String::new();
    s.push_str(
        "# Specification version and digests\n\n\
         The normative reference documents are kept local and untracked at the owner's\n\
         direction; see [docs/README.md](../README.md). This file is tracked, so a fresh\n\
         clone can tell whether it has the specification and whether the copy it has is the\n\
         one the living records were written against.\n\n\
         `cargo xtask spec-check` verifies it. It reads and hashes; it never fetches,\n\
         copies or generates a document, and it does not change the publication policy.\n\n\
         Regenerate with `cargo xtask spec-check --update` **after** an ADR amends a\n\
         document, and commit the diff alongside that ADR. A digest that changes with no\n\
         ADR is an amendment by edit, which the placement contract forbids.\n\n\
         SHA-256 of the raw file bytes:\n\n\
         | Document | SHA-256 |\n|---|---|\n",
    );
    for (rel, digest) in entries {
        s.push_str(&format!("| `{rel}` | `{digest}` |\n"));
    }
    s
}

/// Read `| `path` | `digest` |` rows back out.
fn parse_manifest(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let path = cells[1].trim_matches('`');
        let digest = cells[2].trim_matches('`');
        if path.starts_with("docs/spec/") && digest.len() == 64 {
            out.push((path.to_string(), digest.to_string()));
        }
    }
    out
}

// --- SHA-256, FIPS 180-4 ---------------------------------------------------
//
// Same reasoning as `moxie-kernels/build.rs`: no third-party dependency and no
// shell-out. Checked against the published vectors by a unit test below.

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

pub fn sha256_hex(data: &[u8]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_published_vectors() {
        // A home-grown digest that has never been checked against a published
        // vector is not an identity.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Multi-block, crossing the length-padding boundary.
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn the_manifest_round_trips() {
        let entries = vec![
            (
                "docs/spec/01-product-and-decisions.md".to_string(),
                sha256_hex(b"one"),
            ),
            (
                "docs/spec/09-agent-playbooks.md".to_string(),
                sha256_hex(b"nine"),
            ),
        ];
        let parsed = parse_manifest(&render_manifest(&entries));
        assert_eq!(parsed, entries);
    }

    #[test]
    fn the_required_list_is_the_whole_pack() {
        // Nine numbered documents plus the diagnosis. A document dropped from
        // this list stops being checked, so the count is asserted.
        assert_eq!(REQUIRED.len(), 10);
        let mut seen = REQUIRED.to_vec();
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(n, seen.len());
    }
}
