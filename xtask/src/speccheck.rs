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

// --- SHA-256 -------------------------------------------------------------
//
// Task 0005's deletion plan: the canonical implementation lives in
// `moxie-format::sha256` and this command consumes it. The published-vector
// test moved with it.
use moxie_format::sha256_hex;

#[cfg(test)]
mod tests {
    use super::*;

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
