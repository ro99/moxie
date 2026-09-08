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

use std::collections::{BTreeMap, BTreeSet};
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
                // What this checker needs to read manifests and to parse model
                // source structurally: `toml`, and `syn` with the two crates it
                // is built on. Nothing else, and none of them is reachable from
                // a production crate. See ADR 0004.
                third_party: &["toml", "syn", "quote", "proc-macro2"],
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

/// Module paths a model crate may not reach.
///
/// Matched **structurally**, against paths recovered from the parsed source:
/// the fully-qualified names a `use` declaration introduces, and every
/// `a::b`-shaped path in the token stream. Never against text. Two M0 reviews
/// showed why -- `use std::{fs};` followed by `fs::read(p)` contains the string
/// `std::fs` nowhere, and `use std::{ /* comment */ fs };` defeats a hand-
/// written parser that works on characters. Groups, renames, globs, comments and
/// inline calls all reduce to the same path here.
///
/// A prefix matches itself and everything under it: `std::fs` catches
/// `std::fs::read`, and a glob at or above it (`use std::*`) catches it too.
///
/// Document 09 §B: a model definition may not contain CUDA allocation/launch,
/// device streams/events, host-mmap or disk-read pipelines, cache/eviction
/// policy, KV page allocation, or sampling/speculation loops.
const MODEL_FORBIDDEN_PATHS: &[(&str, &str)] = &[
    ("std::fs", "direct file I/O in a model crate"),
    ("std::thread", "thread management in a model crate"),
    ("std::process", "process control in a model crate"),
    ("std::net", "network access in a model crate"),
    ("memmap", "memory mapping in a model crate"),
    ("memmap2", "memory mapping in a model crate"),
    ("libc", "raw platform bindings in a model crate"),
];

/// Directory names that hold dev-only code, which document 02 exempts: "Test
/// fixtures may use a harness through dev dependencies".
///
/// The exemption is by *role*, not by name. A directory called `tests` that a
/// manifest declares as a production target -- `[lib] path = "tests/production.rs"`
/// -- is production, and [`production_sources`] withdraws the exemption for it.
/// The second review found a model crate hiding its library there.
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

// --- production source discovery and structural analysis ---------------------
//
// The third M0 review compiled two model crates that this checker accepted:
//
//     use std::{ /* checkpoint files */ fs };   // comment retained in the path
//     #[path = "../tests/reader.rs"] pub mod reader;   // module outside any target dir
//
// Both are ordinary Rust. The first defeated a hand-written import parser that
// scanned text; the second defeated a directory walk that decided what was
// production by directory *name*. Neither is the documented re-export
// limitation, and neither is fixable by another special case.
//
// So imports and module declarations are now parsed with `syn`, and production
// sources are found by following module declarations from the crate's Cargo
// targets. Comments and string literals cannot survive tokenisation, so that
// whole class of evasion is gone rather than patched. See ADR 0004.

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;

/// Paths the manifest declares as **production** targets.
///
/// `[lib]`, `[[bin]]` and `[[example]]` are production; `[[test]]` and
/// `[[bench]]` are the dev harness document 02 exempts. A declared path may
/// point anywhere, including into a directory whose name suggests it is a test
/// tree, which is what makes this function load-bearing rather than cosmetic.
fn declared_target_paths(doc: &toml::Value, manifest_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut push = |v: &toml::Value| {
        if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
            out.push(manifest_dir.join(p));
        }
    };
    if let Some(lib) = doc.get("lib") {
        push(lib);
    }
    for section in ["bin", "example"] {
        if let Some(arr) = doc.get(section).and_then(|v| v.as_array()) {
            for t in arr {
                push(t);
            }
        }
    }
    out
}

/// The crate roots to start module traversal from.
///
/// Declared targets when the manifest names any, plus Cargo's default layout.
/// Both, not either: a manifest may declare `[[bin]]` explicitly and still have
/// an implicit `src/lib.rs`.
fn crate_roots(doc: &toml::Value, manifest_dir: &Path) -> Vec<PathBuf> {
    let mut roots = declared_target_paths(doc, manifest_dir);
    for default in ["src/lib.rs", "src/main.rs"] {
        let p = manifest_dir.join(default);
        if p.is_file() {
            roots.push(p);
        }
    }
    for dir in ["src/bin", "examples"] {
        if let Ok(rd) = std::fs::read_dir(manifest_dir.join(dir)) {
            for e in rd.filter_map(|e| e.ok()) {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "rs") {
                    roots.push(p);
                }
            }
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

/// Where the child modules of `file` live.
///
/// A crate root and a `mod.rs` put their children beside them; any other module
/// file puts them in a directory named after it. Getting this wrong is how a
/// traversal silently stops one level down.
fn children_dir(file: &Path, is_crate_root: bool) -> PathBuf {
    let parent = file.parent().unwrap_or(Path::new(".")).to_path_buf();
    if is_crate_root {
        return parent;
    }
    match file.file_stem().and_then(|s| s.to_str()) {
        Some("mod") => parent,
        Some(stem) => parent.join(stem),
        None => parent,
    }
}

/// Whether an item is gated to test builds.
///
/// Document 02 permits a test harness. Reading it from the parsed attribute
/// replaces a hand-written brace matcher that had to skip comments, strings and
/// char literals to find where the attributed item ended.
fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && match &a.meta {
                syn::Meta::List(l) => l
                    .tokens
                    .clone()
                    .into_iter()
                    .any(|t| matches!(&t, TokenTree::Ident(i) if i == "test")),
                _ => false,
            }
    })
}

/// The `#[path = "..."]` override on a module declaration, if present.
fn path_attribute(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|a| {
        if !a.path().is_ident("path") {
            return None;
        }
        match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        }
    })
}

/// What one parsed source file contains that the ownership rules care about.
#[derive(Debug, Default)]
struct SourceFacts {
    /// Fully-qualified paths brought into scope by `use`, with a glob written
    /// as `prefix::*`.
    imports: Vec<String>,
    /// Every `a::b`-shaped path mentioned anywhere in non-test items, from the
    /// token stream. This is what catches `std::fs::read(p)` written inline with
    /// no import, and it cannot be fooled by a comment because comments are not
    /// tokens.
    mentions: Vec<String>,
    /// `extern "C"` blocks.
    foreign_blocks: usize,
    /// Source pulled in by `include!`, which is code outside the module tree.
    source_includes: Vec<String>,
    /// Child module files this file declares.
    children: Vec<PathBuf>,
    /// Module declarations that could not be resolved to a file.
    unresolved: Vec<String>,
}

/// Parse one file and collect its facts.
fn analyse_source(file: &Path, is_crate_root: bool) -> Result<SourceFacts, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let parsed = syn::parse_file(&text).map_err(|e| {
        format!(
            "{}: cannot parse as Rust ({e}); a model crate whose source does not parse \
             cannot be checked",
            file.display()
        )
    })?;
    let dir = children_dir(file, is_crate_root);
    let mut facts = SourceFacts::default();
    walk_items(&parsed.items, &dir, &mut facts);
    Ok(facts)
}

fn walk_items(items: &[syn::Item], dir: &Path, facts: &mut SourceFacts) {
    for item in items {
        match item {
            syn::Item::Use(u) => {
                if is_cfg_test(&u.attrs) {
                    continue;
                }
                flatten_use_tree("", &u.tree, &mut facts.imports);
                collect_mentions(u.to_token_stream(), &mut facts.mentions);
            }
            syn::Item::Mod(m) => {
                if is_cfg_test(&m.attrs) {
                    continue;
                }
                let over = path_attribute(&m.attrs);
                match &m.content {
                    // Inline module: its children live one level down, unless a
                    // `#[path]` names the directory for them.
                    Some((_, inner)) => {
                        let inner_dir = match &over {
                            Some(p) => dir.join(p),
                            None => dir.join(m.ident.to_string()),
                        };
                        walk_items(inner, &inner_dir, facts);
                    }
                    // Declaration: resolve it to a file.
                    None => match resolve_module(dir, &m.ident.to_string(), over.as_deref()) {
                        Some(p) => facts.children.push(p),
                        None => facts.unresolved.push(format!(
                            "`mod {};` in {} resolves to no file",
                            m.ident,
                            dir.display()
                        )),
                    },
                }
            }
            syn::Item::ForeignMod(f) => {
                if is_cfg_test(&f.attrs) {
                    continue;
                }
                facts.foreign_blocks += 1;
            }
            syn::Item::Macro(mac) => {
                if is_cfg_test(&mac.attrs) {
                    continue;
                }
                if mac.mac.path.is_ident("include") {
                    facts
                        .source_includes
                        .push(mac.mac.tokens.to_string().trim().to_string());
                }
                collect_mentions(mac.to_token_stream(), &mut facts.mentions);
            }
            other => collect_mentions(other.to_token_stream(), &mut facts.mentions),
        }
    }
}

/// Resolve `mod name;` to the file Cargo would compile.
fn resolve_module(dir: &Path, name: &str, path_attr: Option<&str>) -> Option<PathBuf> {
    let candidates = match path_attr {
        Some(p) => vec![dir.join(p)],
        None => vec![
            dir.join(format!("{name}.rs")),
            dir.join(name).join("mod.rs"),
        ],
    };
    candidates.into_iter().find(|c| c.is_file())
}

/// Flatten one `use` tree onto `prefix`.
///
/// Operating on the parsed tree rather than on text is the whole point: a
/// comment inside the braces, a rename, a nested group and a glob are all
/// distinct AST shapes here, and none of them can leak characters into a path.
fn flatten_use_tree(prefix: &str, tree: &syn::UseTree, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Path(p) => {
            flatten_use_tree(&join_path(prefix, &p.ident.to_string()), &p.tree, out)
        }
        syn::UseTree::Name(n) => out.push(join_path(prefix, &n.ident.to_string())),
        // A rename imports the original item; the local name is irrelevant here.
        syn::UseTree::Rename(r) => out.push(join_path(prefix, &r.ident.to_string())),
        syn::UseTree::Glob(_) => out.push(join_path(prefix, "*")),
        syn::UseTree::Group(g) => {
            for t in &g.items {
                flatten_use_tree(prefix, t, out);
            }
        }
    }
}

fn join_path(prefix: &str, tail: &str) -> String {
    if prefix.is_empty() {
        tail.to_string()
    } else {
        format!("{prefix}::{tail}")
    }
}

/// Every `a::b`-shaped path in a token stream.
///
/// Walks tokens, so a path inside a comment does not exist and a path inside a
/// string literal is a `Literal`, not a sequence of idents. Both were false
/// positives or false negatives for a text scan, depending on which way it erred.
fn collect_mentions(ts: TokenStream, out: &mut Vec<String>) {
    let tokens: Vec<TokenTree> = ts.into_iter().collect();
    let mut i = 0usize;
    while i < tokens.len() {
        if let TokenTree::Group(g) = &tokens[i] {
            collect_mentions(g.stream(), out);
            i += 1;
            continue;
        }
        let TokenTree::Ident(first) = &tokens[i] else {
            i += 1;
            continue;
        };
        // `ident (:: ident)+`
        let mut path = first.to_string();
        let mut j = i + 1;
        let mut segments = 1;
        while j + 2 < tokens.len() + 1 {
            let is_colon2 = matches!((tokens.get(j), tokens.get(j + 1)),
                (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)))
                    if a.as_char() == ':' && b.as_char() == ':');
            if !is_colon2 {
                break;
            }
            match tokens.get(j + 2) {
                Some(TokenTree::Ident(next)) => {
                    path.push_str("::");
                    path.push_str(&next.to_string());
                    segments += 1;
                    j += 3;
                }
                _ => break,
            }
        }
        if segments > 1 {
            out.push(path);
        }
        i = j.max(i + 1);
    }
}

/// Production Rust sources of one crate, and any problem found finding them.
///
/// Three sources, unioned:
///
/// * every module reachable from a Cargo production target, followed through
///   `mod` declarations including `#[path]` overrides -- this is what makes a
///   file production, regardless of which directory it sits in;
/// * the crate roots themselves;
/// * a walk of the crate directory excluding build output and the dev-only
///   trees, which keeps unreachable stray files in scope.
///
/// The dev exemption is by *role*: a `tests` directory holding a declared target
/// or a reachable module is production and is scanned.
fn production_sources(doc: &toml::Value, manifest_dir: &Path) -> (Vec<PathBuf>, Vec<String>) {
    let roots = crate_roots(doc, manifest_dir);
    let mut problems = Vec::new();
    let mut reachable: BTreeSet<PathBuf> = BTreeSet::new();

    let mut queue: Vec<(PathBuf, bool)> = roots.iter().map(|r| (r.clone(), true)).collect();
    while let Some((file, is_root)) = queue.pop() {
        if !file.is_file() || !reachable.insert(file.clone()) {
            continue;
        }
        match analyse_source(&file, is_root) {
            Ok(facts) => {
                problems.extend(facts.unresolved);
                for c in facts.children {
                    queue.push((c, false));
                }
            }
            Err(e) => problems.push(e),
        }
    }

    // The directory walk, with the exemption withdrawn from any dev-named
    // directory that actually holds production code.
    let exempt: Vec<PathBuf> = DEV_ONLY_DIRS
        .iter()
        .map(|d| manifest_dir.join(d))
        .filter(|dir| !reachable.iter().any(|p| p.starts_with(dir)))
        .collect();

    let mut out: BTreeSet<PathBuf> = reachable;
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
                if name == "target" || name == ".git" || exempt.contains(&p) {
                    continue;
                }
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.insert(p);
            }
        }
    }
    (out.into_iter().collect(), problems)
}

/// Whether an imported `path` reaches `needle`.
///
/// True when the path *is* the needle, sits under it, or is a glob at or above
/// it -- `use std::*;` brings `fs` into scope just as `use std::fs;` does.
fn path_reaches(path: &str, needle: &str) -> bool {
    if path == needle || path.starts_with(&format!("{needle}::")) {
        return true;
    }
    match path.strip_suffix("::*") {
        Some(base) => needle == base || needle.starts_with(&format!("{base}::")),
        None => false,
    }
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
        //
        // Structural throughout. Imports come from parsed `use` trees, inline
        // paths from the token stream, `extern "C"` from a foreign-module item,
        // and the files to look at from following `mod` declarations out of the
        // Cargo targets. A comment or a string cannot influence any of them.
        if is_model {
            let (files, problems) = production_sources(&doc, dir);
            for detail in problems {
                // Fail closed: source that cannot be found or parsed cannot be
                // cleared.
                out.push(Violation {
                    crate_name: name.clone(),
                    rule: rule::MODEL_FORBIDDEN_SOURCE,
                    detail,
                });
            }
            for file in files {
                let facts = match analyse_source(&file, true) {
                    Ok(f) => f,
                    Err(e) => {
                        out.push(Violation {
                            crate_name: name.clone(),
                            rule: rule::MODEL_FORBIDDEN_SOURCE,
                            detail: e,
                        });
                        continue;
                    }
                };
                let mut report = |detail: String| {
                    out.push(Violation {
                        crate_name: name.clone(),
                        rule: rule::MODEL_FORBIDDEN_SOURCE,
                        detail,
                    });
                };

                for (needle, why) in MODEL_FORBIDDEN_PATHS {
                    for path in facts.imports.iter().chain(facts.mentions.iter()) {
                        if path_reaches(path, needle) {
                            report(format!("{}: uses `{path}`: {why}", file.display()));
                            break;
                        }
                    }
                }
                if facts.foreign_blocks > 0 {
                    report(format!(
                        "{}: {} `extern` block(s): FFI declaration in a model crate",
                        file.display(),
                        facts.foreign_blocks
                    ));
                }
                for inc in &facts.source_includes {
                    // A model crate may not carry a build script, so there is no
                    // generated source for this to pull in; what it does do is
                    // introduce code a reader will not find by following `mod`.
                    report(format!(
                        "{}: `include!({inc})` brings in source outside the module tree",
                        file.display()
                    ));
                }

                // The one deliberately broad net that is not a path rule: a
                // model crate mentioning CUDA at all is a boundary breach, and
                // it is worth catching in a comment or a string too.
                let text = std::fs::read_to_string(&file).unwrap_or_default();
                if text.to_lowercase().contains("cuda") {
                    report(format!(
                        "{}: CUDA reference in a model crate",
                        file.display()
                    ));
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

    /// Parse a snippet as a crate root and return its facts.
    ///
    /// Each call gets its own directory: these tests run in parallel, and two
    /// sharing a path would delete each other's file.
    fn facts_of(src: &str) -> SourceFacts {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("moxie-facts-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("lib.rs");
        std::fs::write(&file, src).unwrap();
        let f = analyse_source(&file, true).expect("snippet parses");
        std::fs::remove_dir_all(&dir).ok();
        f
    }

    fn reaches_forbidden(f: &SourceFacts, needle: &str) -> bool {
        f.imports
            .iter()
            .chain(f.mentions.iter())
            .any(|p| path_reaches(p, needle))
    }

    #[test]
    fn every_spelling_of_a_forbidden_import_lands_on_one_rule() {
        // Two reviews' worth of evasions, plus the shapes around them. The last
        // two are the ones a character-level parser could not survive.
        let cases = [
            "use std::fs;",
            "use std::{fs};",
            "use std::{fs, thread};",
            "use std::fs as filesystem;",
            "use std::fs::read;",
            "use std::{io, fs::{read, write}};",
            "pub use std::fs;",
            "use ::std::fs;",
            "use std::*;",
            "use\n    std::{\n        fs,\n    };",
            "use std::{ /* checkpoint files */ fs };",
            "use std::{\n  // the weights live on disk\n  fs,\n};",
            "use std :: fs ;",
            "fn f(p: &str) { let _ = std::fs::read(p); }",
            "fn f(p: &str) { let _ = /* here */ std::fs::read(p); }",
        ];
        for src in cases {
            assert!(
                reaches_forbidden(&facts_of(src), "std::fs"),
                "{src:?} was not recognised"
            );
        }
    }

    #[test]
    fn an_innocent_import_is_not_flagged() {
        // A checker that flags everything is not enforcement either.
        let cases = [
            "use std::io::Result;",
            "use std::collections::BTreeMap;",
            "mod fs { pub fn read() {} }\nuse crate::fs;",
            "use moxie_graph::{Op, OracleRegistry};",
            "const DOC: &str = \"use std::fs; std::thread::spawn\";",
            "// use std::fs;\nfn f() {}",
            "/* use std::process; */\nfn f() {}",
            "const R: &str = r#\"std::fs::read\"#;",
        ];
        for src in cases {
            let f = facts_of(src);
            for (needle, _) in MODEL_FORBIDDEN_PATHS {
                assert!(
                    !reaches_forbidden(&f, needle),
                    "{src:?} was wrongly matched against {needle}: \
                     imports {:?} mentions {:?}",
                    f.imports,
                    f.mentions
                );
            }
        }
    }

    #[test]
    fn a_test_module_is_exempt_and_the_code_after_it_is_not() {
        // Document 02 permits a dev harness. The exemption must cover the test
        // module and stop there -- the failure the very first review found.
        let f = facts_of(
            "#[cfg(test)]\nmod tests { use std::fs; }\npub fn after(p: &str) { let _ = std::fs::read(p); }",
        );
        assert!(reaches_forbidden(&f, "std::fs"));

        let only_tests =
            facts_of("#[cfg(test)]\nmod tests { use std::fs; fn t() {} }\npub fn g() {}");
        assert!(!reaches_forbidden(&only_tests, "std::fs"));

        let gated_use = facts_of("#[cfg(test)]\nuse std::fs;\npub fn g() {}");
        assert!(!reaches_forbidden(&gated_use, "std::fs"));
    }

    #[test]
    fn nested_groups_flatten_to_full_paths() {
        let mut got = facts_of("use a::{b::{c, d as e}, f, g::*};").imports;
        got.sort();
        assert_eq!(
            got,
            vec![
                "a::b::c".to_string(),
                "a::b::d".to_string(),
                "a::f".to_string(),
                "a::g::*".to_string(),
            ]
        );
    }

    #[test]
    fn an_extern_block_is_found_as_an_item_not_as_text() {
        assert_eq!(
            facts_of("unsafe extern \"C\" { fn f(); }").foreign_blocks,
            1
        );
        // ... and a string that merely says so is not one.
        assert_eq!(
            facts_of("const S: &str = \"extern \\\"C\\\"\";").foreign_blocks,
            0
        );
        // A test-gated block is the dev harness.
        assert_eq!(
            facts_of("#[cfg(test)]\nunsafe extern \"C\" { fn f(); }").foreign_blocks,
            0
        );
    }

    #[test]
    fn included_source_is_reported_because_it_leaves_the_module_tree() {
        let f = facts_of("include!(\"generated.rs\");\npub fn g() {}");
        assert_eq!(f.source_includes.len(), 1);
        assert!(facts_of("pub fn g() {}").source_includes.is_empty());
    }

    #[test]
    fn a_glob_covers_what_it_brings_into_scope() {
        assert!(path_reaches("std::*", "std::fs"));
        assert!(path_reaches("std::fs::*", "std::fs"));
        assert!(path_reaches("std::fs", "std::fs"));
        assert!(path_reaches("std::fs::read", "std::fs"));
        assert!(!path_reaches("std::io::*", "std::fs"));
        assert!(!path_reaches("stdext::fs", "std::fs"));
        assert!(!path_reaches("crate::fs", "std::fs"));
    }

    #[test]
    fn module_declarations_resolve_to_the_files_cargo_compiles() {
        // Third review, case 2: a `#[path]` module pointing into a directory no
        // Cargo target names. Also the two ordinary layouts, so the traversal is
        // not only correct for the evasion.
        let dir = std::env::temp_dir().join(format!("moxie-mods-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("src/inner")).unwrap();
        std::fs::create_dir_all(dir.join("tests")).unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "#[path = \"../tests/reader.rs\"]\npub mod reader;\npub mod sibling;\npub mod inner;\n#[cfg(test)]\nmod tests;\n",
        )
        .unwrap();
        std::fs::write(dir.join("src/sibling.rs"), "pub fn s() {}").unwrap();
        std::fs::write(dir.join("src/inner/mod.rs"), "pub fn i() {}").unwrap();
        std::fs::write(dir.join("tests/reader.rs"), "pub fn r() {}").unwrap();
        std::fs::write(dir.join("tests/harness.rs"), "fn h() {}").unwrap();

        let manifest = parse("[package]\nname = \"moxie-models-test\"\nversion = \"0.0.0\"\n");
        let (files, problems) = production_sources(&manifest, &dir);
        assert!(problems.is_empty(), "{problems:?}");

        let has = |rel: &str| files.iter().any(|f| f.ends_with(rel));
        assert!(has("tests/reader.rs"), "the #[path] module: {files:?}");
        assert!(has("src/sibling.rs"), "foo.rs layout: {files:?}");
        assert!(has("inner/mod.rs"), "foo/mod.rs layout: {files:?}");
        assert!(
            !has("tests/harness.rs"),
            "an unreachable file in a dev directory stays exempt: {files:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unresolvable_module_declaration_fails_closed() {
        let dir = std::env::temp_dir().join(format!("moxie-badmod-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub mod missing;\n").unwrap();
        let manifest = parse("[package]\nname = \"moxie-models-test\"\nversion = \"0.0.0\"\n");
        let (_, problems) = production_sources(&manifest, &dir);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("missing"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn source_that_does_not_parse_is_refused_rather_than_skipped() {
        let dir = std::env::temp_dir().join(format!("moxie-badsrc-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("lib.rs");
        std::fs::write(&f, "fn broken( {").unwrap();
        assert!(analyse_source(&f, true).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_declared_library_outside_src_is_production() {
        // Second review, case 2: `[lib] path = "tests/production.rs"` put the
        // whole library in a directory the checker skipped by name.
        let dir = std::env::temp_dir().join(format!("moxie-archcheck-lib-{}", std::process::id()));
        let tests = dir.join("tests");
        let src = dir.join("src");
        std::fs::create_dir_all(&tests).unwrap();
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(tests.join("production.rs"), "pub fn f() {}").unwrap();
        std::fs::write(tests.join("harness.rs"), "fn t() {}").unwrap();
        std::fs::write(src.join("lib.rs"), "pub fn g() {}").unwrap();

        let declared = parse(
            r#"
            [package]
            name = "moxie-models-test"
            version = "0.0.0"
            [lib]
            path = "tests/production.rs"
            "#,
        );
        let (scanned, _) = production_sources(&declared, &dir);
        assert!(
            scanned.contains(&tests.join("production.rs")),
            "the declared library was skipped: {scanned:?}"
        );

        // With no production target in `tests`, the exemption stands.
        let ordinary = parse("[package]\nname = \"moxie-models-test\"\nversion = \"0.0.0\"\n");
        let (scanned, _) = production_sources(&ordinary, &dir);
        assert!(!scanned.iter().any(|p| p.starts_with(&tests)));
        assert!(scanned.contains(&src.join("lib.rs")));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_rule_name_is_distinct() {
        let mut names = rule::ALL.to_vec();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate rule identifier");
    }
}
