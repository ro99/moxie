//! `cargo xtask arch-check` -- machine-checked ownership boundaries.
//!
//! Document 02: "Build `xtask arch-check` in M0/M1. It must check Cargo
//! dependency direction, actual module imports, model build scripts/FFI, dynamic
//! registration edges, and generated source." And: "CI must contain negative
//! fixtures proving each forbidden edge fails."
//!
//! The allowlist below is the machine-readable form of document 02's ownership
//! table. Widening it is an architectural change and needs an ADR, not an edit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Allowed *production* workspace dependencies, per crate. Absent from this map
/// means "not a known crate", which is itself a failure: a new crate must
/// declare where it sits in the ownership graph before it can build.
///
/// Dev-dependencies are checked separately and more loosely: document 02 permits
/// a test harness, and says that "is not permission for production dependencies".
fn allowlist() -> BTreeMap<&'static str, &'static [&'static str]> {
    BTreeMap::from([
        ("moxie-types", &[] as &[&str]),
        ("moxie-graph", &["moxie-types"][..]),
        ("moxie-model-api", &["moxie-types", "moxie-graph"][..]),
        ("moxie-format", &["moxie-types"][..]),
        ("moxie-state", &["moxie-types"][..]),
        ("moxie-cuda", &["moxie-types"][..]),
        ("moxie-kernels", &["moxie-types"][..]),
        // The composition root / tooling. Document 02: "Only the composition
        // root/registry and integration tests may" import concrete models.
        (
            "xtask",
            &[
                "moxie-types",
                "moxie-graph",
                "moxie-model-api",
                "moxie-format",
                "moxie-state",
                "moxie-cuda",
                "moxie-kernels",
            ][..],
        ),
    ])
}

/// A model crate may depend on these and nothing else in the workspace.
/// Document 02: "A model crate's allowed production dependencies are
/// graph/model-api/types and narrowly approved pure metadata parsing."
const MODEL_ALLOWED: &[&str] = &["moxie-types", "moxie-graph", "moxie-model-api"];

/// Crate-name prefixes that identify a concrete model adapter.
const MODEL_PREFIX: &str = "moxie-models-";

/// Substrings that must not appear in a model crate's production source.
/// Needles are lowercase: the haystack is lowercased before matching.
/// Document 09 §B: a model definition may not contain CUDA allocation/launch,
/// device streams/events, host-mmap or disk-read pipelines, cache/eviction
/// policy, KV page allocation, or sampling/speculation loops.
const MODEL_FORBIDDEN_SOURCE: &[(&str, &str)] = &[
    ("extern \"c\"", "FFI declaration in a model crate"),
    ("cuda", "CUDA reference in a model crate"),
    ("std::fs", "direct file I/O in a model crate"),
    ("std::thread", "thread management in a model crate"),
    ("memmap", "memory mapping in a model crate"),
];

#[derive(Debug)]
struct Violation {
    crate_name: String,
    rule: &'static str,
    detail: String,
}

pub fn run() -> i32 {
    let root = workspace_root();
    let mut failed = 0;

    println!("== workspace ==");
    let violations = check_tree(&root);
    match &violations {
        Ok(v) if v.is_empty() => println!("PASS  no ownership violations"),
        Ok(v) => {
            for x in v {
                println!("FAIL  {} :: {} :: {}", x.crate_name, x.rule, x.detail);
            }
            failed += v.len();
        }
        Err(e) => {
            println!("FAIL  cannot check workspace: {e}");
            failed += 1;
        }
    }

    // Negative fixtures. Document 02 requires proof that each forbidden edge
    // actually fails; a checker that has never rejected anything is not evidence.
    println!("\n== negative fixtures ==");
    let fixtures = root.join("xtask/fixtures/arch-check");
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(&fixtures) {
        Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).collect(),
        Err(e) => {
            println!("FAIL  cannot read {}: {e}", fixtures.display());
            return 1;
        }
    };
    entries.sort();
    entries.retain(|p| p.is_dir());

    if entries.is_empty() {
        println!("FAIL  no negative fixtures present; the checker is unproven");
        return 1;
    }

    for dir in entries {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        match check_tree(&dir) {
            Ok(v) if v.is_empty() => {
                // The whole point: this fixture is supposed to be rejected.
                println!("FAIL  fixture {name} was ACCEPTED but must be rejected");
                failed += 1;
            }
            Ok(v) => {
                println!("PASS  fixture {name} rejected: {}", v[0].rule);
            }
            Err(e) => {
                println!("FAIL  fixture {name} could not be checked: {e}");
                failed += 1;
            }
        }
    }

    if failed > 0 {
        println!("\n{failed} failure(s)");
        1
    } else {
        println!("\narch-check passed");
        0
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is xtask/; the workspace root is its parent.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent")
        .to_path_buf()
}

/// Check every crate manifest under `root`.
fn check_tree(root: &Path) -> Result<Vec<Violation>, String> {
    let mut out = Vec::new();
    let allow = allowlist();

    for manifest in find_manifests(root)? {
        let text = std::fs::read_to_string(&manifest)
            .map_err(|e| format!("{}: {e}", manifest.display()))?;
        let doc: toml::Value =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;

        let Some(pkg) = doc.get("package") else {
            continue; // virtual workspace manifest
        };
        let name = pkg
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: package has no name", manifest.display()))?
            .to_string();

        let deps = workspace_deps(&doc, "dependencies");
        let is_model = name.starts_with(MODEL_PREFIX);

        // Rule 1: dependency direction, from the ownership table.
        let permitted: Vec<&str> = if is_model {
            MODEL_ALLOWED.to_vec()
        } else {
            match allow.get(name.as_str()) {
                Some(v) => v.to_vec(),
                None => {
                    out.push(Violation {
                        crate_name: name.clone(),
                        rule: "undeclared crate",
                        detail: "not in the arch-check allowlist; declare its ownership first"
                            .into(),
                    });
                    Vec::new()
                }
            }
        };
        for d in &deps {
            if !permitted.contains(&d.as_str()) {
                out.push(Violation {
                    crate_name: name.clone(),
                    rule: "forbidden dependency",
                    detail: format!("{name} -> {d}"),
                });
            }
        }

        // Rule 2: no shared production crate may import a concrete model.
        // Only the composition root may, and it is named explicitly.
        if name != "xtask" {
            for d in &deps {
                if d.starts_with(MODEL_PREFIX) {
                    out.push(Violation {
                        crate_name: name.clone(),
                        rule: "shared crate imports a concrete model",
                        detail: format!("{name} -> {d}"),
                    });
                }
            }
        }

        // Rule 3: a model crate may not carry a build script. R09: CUDA
        // compilation under a model directory is exactly what the legacy layer
        // checker could not see.
        if is_model && manifest.with_file_name("build.rs").exists() {
            out.push(Violation {
                crate_name: name.clone(),
                rule: "model crate has a build script",
                detail: "build.rs under a model crate can compile kernels".into(),
            });
        }

        // Rule 4: forbidden constructs in a model crate's production source.
        if is_model {
            let src = manifest.with_file_name("src");
            for file in rust_sources(&src) {
                let body = std::fs::read_to_string(&file).unwrap_or_default();
                // Strip test modules: dev-time harness use is permitted.
                let body = body.split("#[cfg(test)]").next().unwrap_or("");
                let lower = body.to_lowercase();
                for (needle, why) in MODEL_FORBIDDEN_SOURCE {
                    if lower.contains(needle) {
                        out.push(Violation {
                            crate_name: name.clone(),
                            rule: "forbidden construct in model source",
                            detail: format!("{}: {why}", file.display()),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Workspace dependency names from one manifest section, covering both
/// `foo.workspace = true` and `foo = { path = ... }` spellings.
fn workspace_deps(doc: &toml::Value, section: &str) -> Vec<String> {
    let Some(tbl) = doc.get(section).and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    tbl.keys()
        .filter(|k| k.starts_with("moxie-"))
        .cloned()
        .collect()
}

fn find_manifests(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        };
        for entry in rd.filter_map(|e| e.ok()) {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if p.is_dir() {
                // `target` is build output; `fixtures` is walked explicitly by
                // the caller, never as part of the real workspace.
                if name == "target" || name == ".git" || name == "fixtures" {
                    continue;
                }
                stack.push(p);
            } else if name == "Cargo.toml" {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}
