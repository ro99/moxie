//! The compact form of a selection: what `moxie-repack plan` writes.
//!
//! One stanza per tensor does not fit a real model. Measured on
//! `cyankiwi/Laguna-S-2.1-AWQ-INT4` -- 34,740 quantized modules and 2,029 BF16
//! tensors -- the version 1 document is **15,210,830 bytes**, and the bytes are
//! dominated by repeating what every module shares: 48.3% is the four
//! `weight_* = "shard"` lines, 25.4% is the module's own name written twice,
//! and 11.7% is `kind`/`width`/`group` repeated identically 34,740 times.
//!
//! So a plan states the common parts once:
//!
//! ```toml
//! version = 2
//! shards = ["model-00001-of-00015.safetensors", "..."]
//!
//! [weights]
//! width = "int4"
//! group = 32
//! zero_points = "packed-along-output"
//! # [module name, index into `shards`]
//! modules = [["model.layers.0.mlp.down_proj", 0], ["...", 1]]
//!
//! # only for a module whose tensors are not all in one shard
//! [[weights.split]]
//! module = "model.layers.3.mlp.up_proj"
//! weight_packed = 0
//! weight_scale = 1
//! weight_shape = 0
//! weight_zero_point = 0
//!
//! [bf16]
//! tensors = [["model.norm.weight", 14]]
//! ```
//!
//! Expansion happens **here** and produces exactly the `Selection` version 1
//! produces, so both spellings meet the same validation. Nothing downstream
//! knows which one it came from.
//!
//! [ADR 0026]: ../../../docs/decisions/adr/0026-generated-plans-and-automatic-budgets.md

use serde::Deserialize;

use moxie_types::{Error, Result};

use crate::compressed_tensors::{Granularity, PackQuantizedSpec, ZeroPointSource};
use crate::selection::{MAX_SELECTED_TENSORS, Selection};

fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed repack plan (detail unavailable: out of memory)",
        detail,
    )
}

/// The four names a `pack-quantized` module is made of, in the order a plan
/// spells them when it has to spell them.
pub const MODULE_SUFFIXES: [&str; 4] = [
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
];

#[derive(Deserialize)]
struct RawPlan {
    #[serde(default)]
    shards: Vec<String>,
    source: toml::Value,
    tokenizer: toml::Value,
    template: toml::Value,
    architecture: toml::Value,
    provenance: toml::Value,
    completeness: toml::Value,
    #[serde(default)]
    excluded: Vec<toml::Value>,
    weights: Option<RawWeights>,
    bf16: Option<RawBf16>,
    /// Present in a generated plan, absent in a hand-written selection. Read
    /// through `resolved_options`; accepted here so the document parses.
    #[serde(default)]
    #[allow(dead_code)]
    options: Option<toml::Value>,
}

#[derive(Deserialize)]
struct RawWeights {
    width: String,
    group: toml::Value,
    zero_points: String,
    #[serde(default)]
    modules: Vec<(String, usize)>,
    #[serde(default)]
    split: Vec<RawSplit>,
}

#[derive(Deserialize)]
struct RawSplit {
    module: String,
    weight_packed: usize,
    weight_scale: usize,
    weight_shape: usize,
    weight_zero_point: Option<usize>,
}

#[derive(Deserialize)]
struct RawBf16 {
    #[serde(default)]
    tensors: Vec<(String, usize)>,
}

/// How many tensors a plan declares, read from its own counts.
///
/// The number a caller admits memory against. Text length stopped being a
/// proxy for that when one line began describing a whole module.
pub fn declared_entries(text: &str) -> Result<usize> {
    let raw: RawPlan =
        toml::from_str(text).map_err(|e| invalid(format_args!("plan does not parse: {e}")))?;
    let modules = raw.weights.as_ref().map(|w| w.modules.len()).unwrap_or(0);
    let bf16 = raw.bf16.as_ref().map(|b| b.tensors.len()).unwrap_or(0);
    Ok(modules + bf16)
}

/// Expand a plan into the selection it describes.
pub fn expand(text: &str) -> Result<Selection> {
    let raw: RawPlan =
        toml::from_str(text).map_err(|e| invalid(format_args!("plan does not parse: {e}")))?;

    let modules = raw.weights.as_ref().map(|w| w.modules.len()).unwrap_or(0);
    let bf16 = raw.bf16.as_ref().map(|b| b.tensors.len()).unwrap_or(0);
    let entries = modules + bf16;
    if entries == 0 {
        return Err(invalid(format_args!(
            "this plan selects nothing: it names no modules and no bf16 tensors"
        )));
    }
    if entries > MAX_SELECTED_TENSORS {
        return Err(invalid(format_args!(
            "this plan names {entries} tensors, above the {MAX_SELECTED_TENSORS} cap checked \
             before expansion"
        )));
    }
    if raw.shards.is_empty() {
        return Err(invalid(format_args!(
            "this plan names no shards, so no entry in it can say where its bytes are"
        )));
    }
    let shard = |i: usize, what: &str| -> Result<String> {
        raw.shards.get(i).cloned().ok_or_else(|| {
            invalid(format_args!(
                "{what} points at shard {i}, and this plan names only {}",
                raw.shards.len()
            ))
        })
    };

    // Rebuild the version 1 document and hand it to the one validator. Slower
    // than constructing `Selection` directly and deliberately so: a second
    // construction path is a second set of rules to keep in agreement, and this
    // repository has paid for that twice.
    let mut out = String::new();
    out.push_str("version = 1\n");
    // `root` and `[options]` belong to the plan, not to the selection schema the
    // validator knows. They are what the second command reads instead of asking
    // again; strip them before handing the document to that validator.
    let mut source = raw.source.clone();
    if let toml::Value::Table(t) = &mut source {
        t.remove("root");
    }
    out.push_str(&section("source", &source)?);
    out.push_str(&section("tokenizer", &raw.tokenizer)?);
    out.push_str(&section("template", &raw.template)?);
    out.push_str(&section("architecture", &raw.architecture)?);
    out.push_str(&section("provenance", &raw.provenance)?);
    out.push_str(&section("completeness", &raw.completeness)?);
    for e in &raw.excluded {
        out.push_str(&array_entry("excluded", e)?);
    }

    if let Some(w) = &raw.weights {
        let group = match &w.group {
            toml::Value::Integer(n) => n.to_string(),
            toml::Value::String(s) if s == "per-channel" => "\"per-channel\"".to_string(),
            other => {
                return Err(invalid(format_args!(
                    "[weights].group is {other}; an integer group size or \"per-channel\" is what \
                     this reads"
                )));
            }
        };
        let split: std::collections::BTreeMap<&str, &RawSplit> =
            w.split.iter().map(|s| (s.module.as_str(), s)).collect();
        for (module, index) in &w.modules {
            out.push_str("\n[[tensor]]\n");
            out.push_str(&format!("role = {}\n", string(&format!("{module}.weight"))));
            out.push_str("kind = \"pack-quantized\"\n");
            out.push_str(&format!("module = {}\n", string(module)));
            out.push_str(&format!("width = \"{}\"\n", w.width));
            out.push_str(&format!("group = {group}\n"));
            out.push_str(&format!("zero_points = \"{}\"\n", w.zero_points));
            out.push_str("[tensor.files]\n");
            match split.get(module.as_str()) {
                Some(s) => {
                    out.push_str(&format!(
                        "weight_packed = {}\n",
                        string(&shard(s.weight_packed, module)?)
                    ));
                    out.push_str(&format!(
                        "weight_scale = {}\n",
                        string(&shard(s.weight_scale, module)?)
                    ));
                    out.push_str(&format!(
                        "weight_shape = {}\n",
                        string(&shard(s.weight_shape, module)?)
                    ));
                    if let Some(zp) = s.weight_zero_point {
                        out.push_str(&format!(
                            "weight_zero_point = {}\n",
                            string(&shard(zp, module)?)
                        ));
                    }
                }
                None => {
                    let file = string(&shard(*index, module)?);
                    for suffix in MODULE_SUFFIXES {
                        if suffix == "weight_zero_point" && w.zero_points == "symmetric" {
                            continue;
                        }
                        out.push_str(&format!("{suffix} = {file}\n"));
                    }
                }
            }
        }
    }
    if let Some(b) = &raw.bf16 {
        for (name, index) in &b.tensors {
            out.push_str("\n[[tensor]]\n");
            out.push_str(&format!("role = {}\n", string(name)));
            out.push_str("kind = \"bf16\"\n");
            out.push_str(&format!("name = {}\n", string(name)));
            out.push_str(&format!("file = {}\n", string(&shard(*index, name)?)));
        }
    }
    let mut selection = crate::selection::parse(&out)?;
    selection.source_bytes = text.len() as u64;
    Ok(selection)
}

/// The spec a plan's `[weights]` block states, for a caller that needs it
/// before expanding.
pub fn declared_spec(text: &str) -> Result<Option<PackQuantizedSpec>> {
    let raw: RawPlan =
        toml::from_str(text).map_err(|e| invalid(format_args!("plan does not parse: {e}")))?;
    let Some(w) = raw.weights else {
        return Ok(None);
    };
    let width = match w.width.as_str() {
        "int4" => crate::affine::IntWidth::Int4,
        "int8" => crate::affine::IntWidth::Int8,
        other => {
            return Err(invalid(format_args!(
                "[weights].width '{other}' is not read here"
            )));
        }
    };
    let granularity = match &w.group {
        toml::Value::Integer(32) => Granularity::Group { size: 32 },
        toml::Value::Integer(128) => Granularity::Group { size: 128 },
        toml::Value::String(s) if s == "per-channel" => Granularity::Channel,
        other => {
            return Err(invalid(format_args!(
                "[weights].group {other} is not read here"
            )));
        }
    };
    let zero_points = match w.zero_points.as_str() {
        "symmetric" => ZeroPointSource::Symmetric,
        "packed-along-output" => ZeroPointSource::PackedAlongOutput,
        other => {
            return Err(invalid(format_args!(
                "[weights].zero_points '{other}' is not read here"
            )));
        }
    };
    Ok(Some(PackQuantizedSpec {
        width,
        granularity,
        zero_points,
    }))
}

/// Re-emit one table as version 1 spells it.
fn section(name: &str, value: &toml::Value) -> Result<String> {
    let mut doc = toml::map::Map::new();
    doc.insert(name.to_string(), value.clone());
    toml::to_string(&toml::Value::Table(doc))
        .map(|s| format!("\n{s}"))
        .map_err(|e| invalid(format_args!("cannot re-emit [{name}]: {e}")))
}

fn array_entry(name: &str, value: &toml::Value) -> Result<String> {
    let mut doc = toml::map::Map::new();
    doc.insert(name.to_string(), toml::Value::Array(vec![value.clone()]));
    toml::to_string(&toml::Value::Table(doc))
        .map(|s| format!("\n{s}"))
        .map_err(|e| invalid(format_args!("cannot re-emit [[{name}]]: {e}")))
}

/// A TOML basic string, escaped.
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Where a plan says its checkpoint is.
pub fn source_root(text: &str) -> Result<Option<String>> {
    let raw: RawPlan =
        toml::from_str(text).map_err(|e| invalid(format_args!("plan does not parse: {e}")))?;
    Ok(raw
        .source
        .get("root")
        .and_then(|v| v.as_str())
        .map(str::to_string))
}

/// The budgets a plan resolved, in the order `Budgets` declares them.
///
/// `None` for a plan that carries no `[options]`: a hand-written document is
/// still a selection, and it says nothing about resources.
pub fn resolved_options(text: &str) -> Result<Option<[u64; 5]>> {
    #[derive(Deserialize)]
    struct WithOptions {
        options: Option<RawOptions>,
    }
    #[derive(Deserialize)]
    struct RawOptions {
        total_bytes: u64,
        header_bytes: u64,
        scratch_bytes: u64,
        chunk_file_bytes: u64,
        disk_bytes: u64,
    }
    let parsed: WithOptions =
        toml::from_str(text).map_err(|e| invalid(format_args!("plan does not parse: {e}")))?;
    Ok(parsed.options.map(|o| {
        [
            o.total_bytes,
            o.header_bytes,
            o.scratch_bytes,
            o.chunk_file_bytes,
            o.disk_bytes,
        ]
    }))
}
