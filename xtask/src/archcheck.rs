//! `cargo xtask arch-check` -- machine-checked ownership boundaries.
//!
//! Document 02: "Build `xtask arch-check` in M0/M1. It must check Cargo
//! dependency direction, actual module imports, model build scripts/FFI, dynamic
//! registration edges, and generated source." And: "CI must contain negative
//! fixtures proving each forbidden edge fails."
//!
//! The allowlist below is the machine-readable form of document 02's ownership
//! table. Widening it is an architectural change and needs an ADR, not an edit.
//!
//! ## What the M0 review found, and what changed
//!
//! The first version read manifest *keys* in the two top-level dependency
//! tables. Three complete model manifests passed it with zero violations:
//!
//! * a real dependency hidden in `[target.'cfg(unix)'.dependencies]`;
//! * a forbidden package behind an allowed alias, via `package = "..."`;
//! * a build script at a custom path, declared as `build = "codegen.rs"`.
//!
//! So this module now resolves a dependency's **identity** rather than trusting
//! the key it is written under, reads **every** production dependency table
//! including target-specific ones, follows workspace inheritance, and takes the
//! build script and source layout from the manifest's own declarations. Each of
//! the three cases is a negative fixture below, and each fixture states the rule
//! it must trigger, so a fixture cannot pass by being rejected for the wrong
//! reason.
//!
//! Deliberately still absent: running `cargo metadata` on a fixture. It would
//! resolve identities for us, but it also wants a lock file, a registry and a
//! writable directory, and none of that belongs in a check that has to run
//! offline on an untrusted tree. Nothing here executes a build script.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Stable rule identifiers. A fixture names the rule it must trigger, and the
/// runner compares against these, so renaming one is a visible change to every
/// fixture rather than a silent weakening of the check.
pub mod rule {
    pub const FORBIDDEN_DEPENDENCY: &str = "forbidden dependency";
    pub const SHARED_IMPORTS_MODEL: &str = "shared crate imports a concrete model";
    pub const MODEL_BUILD_SCRIPT: &str = "model crate has a build script";
    pub const MODEL_FORBIDDEN_SOURCE: &str = "forbidden construct in model source";
    pub const UNDECLARED_CRATE: &str = "undeclared crate";
    pub const UNRESOLVABLE_DEPENDENCY: &str = "unresolvable dependency";

    pub const ALL: &[&str] = &[
        FORBIDDEN_DEPENDENCY,
        SHARED_IMPORTS_MODEL,
        MODEL_BUILD_SCRIPT,
        MODEL_FORBIDDEN_SOURCE,
        UNDECLARED_CRATE,
        UNRESOLVABLE_DEPENDENCY,
    ];
}

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
            "moxie-oracles",
            Allowed {
                workspace: &["moxie-types", "moxie-graph"],
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
                    "moxie-oracles",
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

/// Directory names that hold dev-only code, which document 02 exempts: "Test
/// fixtures may use a harness through dev dependencies". Kept narrow on purpose
/// -- everything else under a crate is production and is scanned.
const DEV_ONLY_DIRS: &[&str] = &["tests", "benches"];

#[derive(Debug)]
struct Violation {
    crate_name: String,
    rule: &'static str,
    detail: String,
}

pub fn run() -> i32 {
    let root = crate::workspace_root();
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
    //
    // Each fixture declares the rule it must trigger, under
    // `[package.metadata.moxie-arch-check] expect-rule = "..."`. Accepting "any
    // violation at all" is how a fixture keeps passing after the rule it was
    // written for stops firing -- the M0 review's F1 in miniature.
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

    let mut covered: Vec<&str> = Vec::new();
    for dir in &entries {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let expected = match expected_rule(dir) {
            Ok(r) => r,
            Err(e) => {
                println!("FAIL  fixture {name}: {e}");
                failed += 1;
                continue;
            }
        };
        match check_tree(dir) {
            Ok(v) if v.is_empty() => {
                // The whole point: this fixture is supposed to be rejected.
                println!("FAIL  fixture {name} was ACCEPTED but must be rejected ({expected})");
                failed += 1;
            }
            Ok(v) => {
                if let Some(hit) = v.iter().find(|x| x.rule == expected) {
                    println!(
                        "PASS  fixture {name} rejected: {} :: {}",
                        hit.rule, hit.detail
                    );
                    if !covered.contains(&hit.rule) {
                        covered.push(hit.rule);
                    }
                } else {
                    let got: Vec<&str> = v.iter().map(|x| x.rule).collect();
                    println!(
                        "FAIL  fixture {name} was rejected for the wrong reason: \
                         expected {expected:?}, got {got:?}"
                    );
                    failed += 1;
                }
            }
            Err(e) => {
                println!("FAIL  fixture {name} could not be checked: {e}");
                failed += 1;
            }
        }
    }

    // A rule with no fixture has never been observed to fire.
    println!("\n== rule coverage ==");
    for r in rule::ALL {
        if covered.contains(r) {
            println!("PASS  {r}");
        } else {
            println!("NOTE  {r}: no negative fixture exercises this rule");
        }
    }

    if failed > 0 {
        println!("\n{failed} failure(s)");
        1
    } else {
        println!(
            "\narch-check passed: {} fixture(s), {} rule(s) exercised",
            entries.len(),
            covered.len()
        );
        0
    }
}

/// The rule a fixture declares it must trigger.
fn expected_rule(dir: &Path) -> Result<&'static str, String> {
    let manifest = dir.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("cannot read {}: {e}", manifest.display()))?;
    let doc: toml::Value = toml::from_str(&text).map_err(|e| e.to_string())?;
    let declared = doc
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("moxie-arch-check"))
        .and_then(|m| m.get("expect-rule"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            "no [package.metadata.moxie-arch-check] expect-rule; a fixture must say \
             which rule it proves"
                .to_string()
        })?;
    rule::ALL
        .iter()
        .find(|r| **r == declared)
        .copied()
        .ok_or_else(|| format!("expect-rule {declared:?} is not a known rule; see rule::ALL"))
}

// --- manifest analysis -------------------------------------------------------

/// One production dependency edge, with every name it could actually resolve to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DepEdge {
    /// The manifest table this came from, for the message.
    section: String,
    /// The key it is written under. This is the name the crate says `use` on.
    alias: String,
    /// Every package name this edge can resolve to: the alias, an explicit
    /// `package = "..."` rename, and the `[package] name` of a `path` target's
    /// own manifest. All of them must be permitted -- checking only the alias is
    /// what let `moxie-types = { package = "moxie-cuda", ... }` through.
    identities: Vec<String>,
    /// Set when a `path` could not be resolved to a manifest. Fails closed.
    unresolved: Option<String>,
}

/// Collect every production dependency edge in one manifest.
///
/// Covers `[dependencies]`, `[build-dependencies]` and every
/// `[target.<cfg>.dependencies]` / `[target.<cfg>.build-dependencies]`.
///
/// `dev-dependencies` are deliberately excluded, in every table: document 02
/// permits a test harness there, and integration tests are one of the two places
/// allowed to import a concrete model.
///
/// A build dependency is production, not tooling: it is how generated code and
/// kernel compilation get in.
fn production_deps(
    doc: &toml::Value,
    manifest_dir: &Path,
    workspace: Option<&Workspace<'_>>,
) -> Vec<DepEdge> {
    let mut out = Vec::new();
    for section in ["dependencies", "build-dependencies"] {
        collect_section(doc, section, section, manifest_dir, workspace, &mut out);
    }
    if let Some(targets) = doc.get("target").and_then(|v| v.as_table()) {
        for (cfg, table) in targets {
            for section in ["dependencies", "build-dependencies"] {
                collect_section(
                    table,
                    section,
                    &format!("target.{cfg}.{section}"),
                    manifest_dir,
                    workspace,
                    &mut out,
                );
            }
        }
    }
    out
}

fn collect_section(
    parent: &toml::Value,
    key: &str,
    label: &str,
    manifest_dir: &Path,
    workspace: Option<&Workspace<'_>>,
    out: &mut Vec<DepEdge>,
) {
    let Some(tbl) = parent.get(key).and_then(|v| v.as_table()) else {
        return;
    };
    for (alias, spec) in tbl {
        out.push(resolve_edge(alias, spec, label, manifest_dir, workspace));
    }
}

/// The workspace root manifest and the directory it sits in.
///
/// Both halves matter: a `path` written in `[workspace.dependencies]` is
/// relative to the **workspace root**, not to the member that inherits it.
/// Resolving it against the member's directory produces a path that does not
/// exist, which under this checker's fail-closed rule is a violation -- so
/// getting this wrong turns every inherited dependency in the workspace into a
/// false positive.
#[derive(Debug, Clone, Copy)]
struct Workspace<'a> {
    root: &'a Path,
    doc: &'a toml::Value,
}

fn resolve_edge(
    alias: &str,
    spec: &toml::Value,
    section: &str,
    manifest_dir: &Path,
    workspace: Option<&Workspace<'_>>,
) -> DepEdge {
    let mut identities = vec![alias.to_string()];
    let mut unresolved = None;

    // `foo.workspace = true` inherits the real definition from the workspace
    // root, which is where a rename or a path actually lives.
    let inherits = spec
        .get("workspace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (effective, base): (Option<&toml::Value>, &Path) = if inherits {
        match workspace
            .and_then(|w| w.doc.get("workspace"))
            .and_then(|w| w.get("dependencies"))
            .and_then(|d| d.get(alias))
        {
            Some(v) => (Some(v), workspace.map(|w| w.root).unwrap_or(manifest_dir)),
            None => {
                unresolved = Some(format!(
                    "{alias} inherits from [workspace.dependencies], which was not found"
                ));
                (None, manifest_dir)
            }
        }
    } else {
        (Some(spec), manifest_dir)
    };

    if let Some(v) = effective {
        if let Some(renamed) = v.get("package").and_then(|p| p.as_str()) {
            identities.push(renamed.to_string());
        }
        if let Some(rel) = v.get("path").and_then(|p| p.as_str()) {
            // The strongest identity available offline: whatever that manifest
            // calls itself. An alias and a `package` key are both writable by
            // the crate under inspection; the target's own name is not.
            let target = base.join(rel).join("Cargo.toml");
            match std::fs::read_to_string(&target)
                .ok()
                .and_then(|t| toml::from_str::<toml::Value>(&t).ok())
                .and_then(|d| {
                    d.get("package")
                        .and_then(|p| p.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                }) {
                Some(name) => identities.push(name),
                None => {
                    unresolved = Some(format!(
                        "path dependency {alias} -> {rel} has no readable manifest at {}",
                        target.display()
                    ));
                }
            }
        }
    }

    identities.sort();
    identities.dedup();
    DepEdge {
        section: section.to_string(),
        alias: alias.to_string(),
        identities,
        unresolved,
    }
}

/// Every build script this manifest declares or implies.
///
/// `[package] build` may be absent (autodetect `build.rs`), `false` (none), a
/// path, or a list of paths. Reading only for a file literally named `build.rs`
/// is what let `build = "codegen.rs"` through.
fn build_scripts(doc: &toml::Value, manifest_dir: &Path) -> Vec<String> {
    let Some(pkg) = doc.get("package") else {
        return Vec::new();
    };
    match pkg.get("build") {
        None => {
            if manifest_dir.join("build.rs").exists() {
                vec!["build.rs (autodetected)".to_string()]
            } else {
                Vec::new()
            }
        }
        Some(toml::Value::Boolean(false)) => Vec::new(),
        Some(toml::Value::Boolean(true)) => vec!["build.rs (build = true)".to_string()],
        Some(toml::Value::String(p)) => vec![p.clone()],
        Some(toml::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(other) => vec![format!("unrecognised build key: {other}")],
    }
}

/// Production Rust sources of one crate.
///
/// Everything under the crate directory except build output and the narrow
/// dev-only trees document 02 exempts. Taking `src/` alone missed a `[lib] path`
/// or a `[[bin]] path` pointing somewhere else entirely.
fn production_sources(manifest_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![manifest_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if p.is_dir() {
                if name == "target" || name == ".git" || DEV_ONLY_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Check every crate manifest under `root`.
fn check_tree(root: &Path) -> Result<Vec<Violation>, String> {
    let mut out = Vec::new();
    let allow = allowlist();

    // The workspace manifest, if this tree has one. Needed to resolve
    // `foo.workspace = true` to a real package.
    let workspace_doc: Option<toml::Value> = std::fs::read_to_string(root.join("Cargo.toml"))
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .filter(|d: &toml::Value| d.get("workspace").is_some());
    let workspace = workspace_doc.as_ref().map(|doc| Workspace { root, doc });

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
        let dir = manifest.parent().expect("manifest has a directory");

        let deps = production_deps(&doc, dir, workspace.as_ref());
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
                        rule: rule::UNDECLARED_CRATE,
                        detail: "not in the arch-check allowlist; declare its ownership first"
                            .into(),
                    });
                    (Vec::new(), Vec::new())
                }
            }
        };

        for d in &deps {
            if let Some(why) = &d.unresolved {
                // Fail closed: an edge whose identity cannot be established is
                // not an edge that can be permitted.
                out.push(Violation {
                    crate_name: name.clone(),
                    rule: rule::UNRESOLVABLE_DEPENDENCY,
                    detail: format!("[{}] {why}", d.section),
                });
                continue;
            }
            // Every identity must be permitted. An alias is not a permission.
            for id in &d.identities {
                let permitted = if id.starts_with("moxie-") {
                    allow_ws.contains(&id.as_str())
                } else {
                    allow_tp.contains(&id.as_str())
                };
                if !permitted {
                    let via = if *id == d.alias {
                        String::new()
                    } else {
                        format!(" (as `{}`)", d.alias)
                    };
                    out.push(Violation {
                        crate_name: name.clone(),
                        rule: rule::FORBIDDEN_DEPENDENCY,
                        detail: format!("[{}] {name} -> {id}{via}", d.section),
                    });
                }
            }
        }

        // Rule 2: no shared production crate may import a concrete model.
        // Only the composition root may, and it is named explicitly.
        if name != "xtask" {
            for d in &deps {
                for id in &d.identities {
                    if id.starts_with(MODEL_PREFIX) {
                        out.push(Violation {
                            crate_name: name.clone(),
                            rule: rule::SHARED_IMPORTS_MODEL,
                            detail: format!("[{}] {name} -> {id}", d.section),
                        });
                    }
                }
            }
        }

        // Rule 3: a model crate may not carry a build script, at any path. R09:
        // CUDA compilation under a model directory is exactly what the legacy
        // layer checker could not see.
        if is_model {
            for script in build_scripts(&doc, dir) {
                out.push(Violation {
                    crate_name: name.clone(),
                    rule: rule::MODEL_BUILD_SCRIPT,
                    detail: format!("{script} under a model crate can compile kernels"),
                });
            }
        }

        // Rule 4: forbidden constructs in a model crate's production source.
        if is_model {
            for file in production_sources(dir) {
                let body = std::fs::read_to_string(&file).unwrap_or_default();
                // Strip test modules: dev-time harness use is permitted.
                let body = strip_cfg_test(&body);
                let lower = body.to_lowercase();
                for (needle, why) in MODEL_FORBIDDEN_SOURCE {
                    if lower.contains(needle) {
                        out.push(Violation {
                            crate_name: name.clone(),
                            rule: rule::MODEL_FORBIDDEN_SOURCE,
                            detail: format!("{}: {why}", file.display()),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
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
/// secondary net behind the dependency rules, not the only barrier, and it is
/// deliberately not being grown into a Rust parser.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> toml::Value {
        toml::from_str(s).expect("fixture manifest parses")
    }

    fn identities(edges: &[DepEdge]) -> Vec<String> {
        let mut all: Vec<String> = edges
            .iter()
            .flat_map(|e| e.identities.iter().cloned())
            .collect();
        all.sort();
        all.dedup();
        all
    }

    #[test]
    fn a_target_specific_table_is_a_production_dependency() {
        // Review case A. This manifest returned zero violations before.
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [target.'cfg(unix)'.dependencies]
            some-cuda-crate = "1"
            [target.'cfg(windows)'.build-dependencies]
            other = "1"
            "#,
        );
        let deps = production_deps(&doc, Path::new("/nonexistent"), None);
        let ids = identities(&deps);
        assert!(ids.contains(&"some-cuda-crate".to_string()), "{ids:?}");
        assert!(ids.contains(&"other".to_string()), "{ids:?}");
        assert!(deps.iter().any(|d| d.section.starts_with("target.")));
    }

    #[test]
    fn dev_dependencies_stay_exempt_in_every_table() {
        // Document 02 permits a test harness. That exemption must not widen when
        // target tables started being read.
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dev-dependencies]
            harness = "1"
            [target.'cfg(unix)'.dev-dependencies]
            other-harness = "1"
            "#,
        );
        assert!(production_deps(&doc, Path::new("/nonexistent"), None).is_empty());
    }

    #[test]
    fn a_renamed_package_is_identified_by_its_real_name() {
        // Review case B: a forbidden package behind an allowed alias.
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dependencies]
            moxie-types = { package = "moxie-cuda", version = "0.0.0" }
            "#,
        );
        let deps = production_deps(&doc, Path::new("/nonexistent"), None);
        let ids = identities(&deps);
        assert!(ids.contains(&"moxie-cuda".to_string()), "{ids:?}");
        // The alias is still reported, so a `use moxie_types::...` edge is not
        // lost either.
        assert!(ids.contains(&"moxie-types".to_string()), "{ids:?}");
    }

    #[test]
    fn a_path_dependency_is_identified_by_the_target_manifests_own_name() {
        let dir = std::env::temp_dir().join(format!("moxie-archcheck-{}", std::process::id()));
        let target = dir.join("real");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join("Cargo.toml"),
            "[package]\nname = \"moxie-cuda\"\nversion = \"0.0.0\"\n",
        )
        .unwrap();

        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dependencies]
            innocent = { path = "real" }
            "#,
        );
        let ids = identities(&production_deps(&doc, &dir, None));
        assert!(ids.contains(&"moxie-cuda".to_string()), "{ids:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unresolvable_path_fails_closed() {
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dependencies]
            mystery = { path = "nowhere" }
            "#,
        );
        let deps = production_deps(&doc, Path::new("/nonexistent"), None);
        assert!(deps[0].unresolved.is_some());
    }

    #[test]
    fn workspace_inheritance_resolves_to_the_real_definition() {
        let ws = parse(
            r#"
            [workspace]
            members = []
            [workspace.dependencies]
            innocent = { package = "moxie-cuda", version = "0.0.0" }
            "#,
        );
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dependencies]
            innocent = { workspace = true }
            "#,
        );
        let ws = Workspace {
            root: Path::new("/nonexistent"),
            doc: &ws,
        };
        let ids = identities(&production_deps(&doc, Path::new("/nonexistent"), Some(&ws)));
        assert!(ids.contains(&"moxie-cuda".to_string()), "{ids:?}");
    }

    #[test]
    fn an_inherited_path_resolves_against_the_workspace_root() {
        // Not the member's directory. Getting this wrong makes every
        // `foo.workspace = true` in the real workspace unresolvable, and this
        // checker's fail-closed rule then reports each one as a violation.
        let dir = std::env::temp_dir().join(format!("moxie-archcheck-ws-{}", std::process::id()));
        let member = dir.join("crates/member");
        let target = dir.join("crates/moxie-types");
        std::fs::create_dir_all(&member).unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join("Cargo.toml"),
            "[package]\nname = \"moxie-types\"\nversion = \"0.0.0\"\n",
        )
        .unwrap();

        let ws_doc = parse(
            r#"
            [workspace]
            members = ["crates/member"]
            [workspace.dependencies]
            moxie-types = { path = "crates/moxie-types" }
            "#,
        );
        let ws = Workspace {
            root: &dir,
            doc: &ws_doc,
        };
        let doc = parse(
            r#"
            [package]
            name = "member"
            version = "0.0.0"
            [dependencies]
            moxie-types = { workspace = true }
            "#,
        );
        let deps = production_deps(&doc, &member, Some(&ws));
        assert_eq!(deps[0].unresolved, None, "{deps:?}");
        assert!(deps[0].identities.contains(&"moxie-types".to_string()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unresolvable_inherited_dependency_fails_closed() {
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [dependencies]
            innocent = { workspace = true }
            "#,
        );
        let deps = production_deps(&doc, Path::new("/nonexistent"), None);
        assert!(deps[0].unresolved.is_some());
    }

    #[test]
    fn a_custom_build_path_is_still_a_build_script() {
        // Review case C: `build = "codegen.rs"`, with no file named build.rs.
        let doc = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            build = "codegen.rs"
            "#,
        );
        assert_eq!(
            build_scripts(&doc, Path::new("/nonexistent")),
            vec!["codegen.rs".to_string()]
        );
    }

    #[test]
    fn build_false_declares_no_script_and_a_list_declares_several() {
        let none = parse("[package]\nname = \"x\"\nversion = \"0\"\nbuild = false\n");
        assert!(build_scripts(&none, Path::new("/nonexistent")).is_empty());

        let many =
            parse("[package]\nname = \"x\"\nversion = \"0\"\nbuild = [\"a.rs\", \"b.rs\"]\n");
        assert_eq!(build_scripts(&many, Path::new("/nonexistent")).len(), 2);
    }

    #[test]
    fn every_rule_name_is_distinct() {
        let mut names = rule::ALL.to_vec();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate rule identifier");
    }

    #[test]
    fn cfg_test_stripping_keeps_code_after_the_test_module() {
        let src = "fn a() {}\n#[cfg(test)]\nmod t { fn hidden() { let _ = \"cuda\"; } }\nfn b() { let _ = \"cuda\"; }\n";
        let stripped = strip_cfg_test(src);
        assert!(!stripped.contains("hidden"));
        assert!(stripped.contains("fn b()"));
        assert_eq!(stripped.matches("cuda").count(), 1);
    }
}
