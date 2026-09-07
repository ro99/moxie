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

/// Everything a crate may depend on in production, workspace and third-party
/// alike. Absent from this map means "not a known crate", which is itself a
/// failure: a new crate must declare where it sits in the ownership graph
/// before it can build.
///
/// Third-party crates are listed for the same reason workspace crates are.
/// Document 02 denies a model adapter "CUDA, storage, engine, sampling,
/// allocator, thread/task runtimes" -- and almost every one of those is
/// reachable from crates.io, so an allowlist that only inspects `moxie-*`
/// dependencies enforces nothing. An empty `third_party` is the default and
/// widening one is an ADR.
struct Allowed {
    workspace: &'static [&'static str],
    third_party: &'static [&'static str],
}

fn allowlist() -> BTreeMap<&'static str, Allowed> {
    const NONE: &[&str] = &[];
    BTreeMap::from([
        (
            "moxie-types",
            Allowed {
                workspace: NONE,
                third_party: NONE,
            },
        ),
        (
            "moxie-graph",
            Allowed {
                workspace: &["moxie-types"],
                third_party: NONE,
            },
        ),
        (
            "moxie-model-api",
            Allowed {
                workspace: &["moxie-types", "moxie-graph"],
                third_party: NONE,
            },
        ),
        (
            "moxie-format",
            Allowed {
                workspace: &["moxie-types"],
                third_party: NONE,
            },
        ),
        (
            "moxie-state",
            Allowed {
                workspace: &["moxie-types"],
                third_party: NONE,
            },
        ),
        (
            "moxie-cuda",
            Allowed {
                workspace: &["moxie-types"],
                third_party: NONE,
            },
        ),
        (
            "moxie-kernels",
            Allowed {
                workspace: &["moxie-types"],
                third_party: NONE,
            },
        ),
        // The composition root / tooling. Document 02: "Only the composition
        // root/registry and integration tests may" import concrete models.
        (
            "xtask",
            Allowed {
                workspace: &[
                    "moxie-types",
                    "moxie-graph",
                    "moxie-model-api",
                    "moxie-format",
                    "moxie-state",
                    "moxie-cuda",
                    "moxie-kernels",
                ],
                // Manifest parsing for this checker. Nothing else.
                third_party: &["toml"],
            },
        ),
    ])
}

/// A model crate may depend on these and nothing else in the workspace.
/// Document 02: "A model crate's allowed production dependencies are
/// graph/model-api/types and narrowly approved pure metadata parsing."
const MODEL_ALLOWED: &[&str] = &["moxie-types", "moxie-graph", "moxie-model-api"];

/// Third-party crates a model adapter may depend on in production.
///
/// Document 02 allows "narrowly approved pure metadata parsing" and nothing
/// else. Nothing is approved yet: approving one is an ADR naming the crate and
/// why the parsing it does is pure. Empty is the correct default, not an
/// oversight.
const MODEL_ALLOWED_THIRD_PARTY: &[&str] = &[];

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

        // Both production sections. Document 02 requires enforcement of "actual
        // module imports, model build scripts/FFI, dynamic registration edges,
        // and generated code" -- a build dependency is how generated code and
        // kernel compilation get in, so it is production, not tooling.
        let mut deps = all_deps(&doc, "dependencies");
        deps.extend(all_deps(&doc, "build-dependencies"));
        // `dev-dependencies` are deliberately not checked: document 02 permits a
        // test harness there, and integration tests are one of the two places
        // allowed to import a concrete model.
        let is_model = name.starts_with(MODEL_PREFIX);

        // Rule 1: dependency direction, from the ownership table.
        let (allow_ws, allow_tp): (Vec<&str>, Vec<&str>) = if is_model {
            (MODEL_ALLOWED.to_vec(), MODEL_ALLOWED_THIRD_PARTY.to_vec())
        } else {
            match allow.get(name.as_str()) {
                Some(a) => (a.workspace.to_vec(), a.third_party.to_vec()),
                None => {
                    out.push(Violation {
                        crate_name: name.clone(),
                        rule: "undeclared crate",
                        detail: "not in the arch-check allowlist; declare its ownership first"
                            .into(),
                    });
                    (Vec::new(), Vec::new())
                }
            }
        };
        for d in &deps {
            let permitted = if d.starts_with("moxie-") {
                allow_ws.contains(&d.as_str())
            } else {
                allow_tp.contains(&d.as_str())
            };
            if !permitted {
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
                let body = strip_cfg_test(&body);
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

/// Every dependency name in one manifest section, workspace and third-party
/// alike, covering the `foo.workspace = true`, `foo = "1.0"` and
/// `foo = { path = ... }` spellings.
///
/// Deliberately unfiltered. An earlier version kept only `moxie-*` names, which
/// meant a model crate could depend on a CUDA wrapper or an async runtime from
/// crates.io and pass -- the exact dependencies document 02 denies it.
fn all_deps(doc: &toml::Value, section: &str) -> Vec<String> {
    let Some(tbl) = doc.get(section).and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    tbl.keys().cloned().collect()
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

/// Remove `#[cfg(test)]` items, keeping everything else.
///
/// An earlier version used `split("#[cfg(test)]").next()`, which kept only the
/// text *before the first* test module and silently discarded every line after
/// it. A model crate could put FFI declarations, file I/O and a cache below a
/// test module and pass the check. This brace-matches instead, so only the
/// attributed item is removed.
///
/// The matcher skips line comments, block comments and string/char literals so
/// that a brace inside one does not throw off the count. A raw string with an
/// unbalanced brace and a `#` delimiter could still fool it; this scan is a
/// secondary net behind the dependency rules, not the only barrier.
fn strip_cfg_test(src: &str) -> String {
    const ATTR: &str = "#[cfg(test)]";
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        if src[i..].starts_with(ATTR) {
            // Find the item's opening brace, then its match.
            match find_item_end(bytes, i + ATTR.len()) {
                Some(end) => {
                    i = end;
                    continue;
                }
                // No brace found (e.g. `#[cfg(test)] use ...;`): drop just the
                // attribute and carry on.
                None => {
                    i += ATTR.len();
                    continue;
                }
            }
        }
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// From `start`, find the first `{` and return the index just past its match.
fn find_item_end(b: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    // Locate the opening brace, bailing out if a `;` ends the item first.
    while i < b.len() && b[i] != b'{' {
        if b[i] == b';' {
            return None;
        }
        i += 1;
    }
    if i >= b.len() {
        return None;
    }
    let mut depth = 0usize;
    while i < b.len() {
        match b[i] {
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            b'\'' => {
                // A char literal, or a lifetime like `'a`. Only skip when it
                // closes within a few bytes, which a lifetime never does.
                let mut j = i + 1;
                let mut closed = false;
                while j < b.len() && j <= i + 4 {
                    if b[j] == b'\\' {
                        j += 2;
                        continue;
                    }
                    if b[j] == b'\'' {
                        closed = true;
                        break;
                    }
                    j += 1;
                }
                i = if closed { j + 1 } else { i + 1 };
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => i += 1,
        }
    }
    None
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
