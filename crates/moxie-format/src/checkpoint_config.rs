//! Reading a compressed-tensors checkpoint's `config.json`.
//!
//! A selection says which tensors to convert and how they are packed. Every one
//! of those facts is already written down in the checkpoint: the quantization
//! parameters live in `quantization_config`, and which modules are **not**
//! quantized lives in its `ignore` list. Making a user retype them was a
//! usability decision nobody made deliberately -- the selection exists so a
//! conversion is *explicit* ([ADR 0020]), which is a reason for the file to
//! exist, not a reason for a human to author it.
//!
//! This module reads only. It decides nothing about what to convert; it reports
//! what the checkpoint declares, and the caller composes a selection from it.
//!
//! [ADR 0020]: ../../../docs/decisions/adr/0020-user-managed-storage-and-canonical-materialization.md

use serde::Deserialize;

use crate::compressed_tensors::{Granularity, ZeroPointSource};
use moxie_types::{Error, Result};

use crate::invalid_static;

fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a checkpoint config this repacker will not read (detail unavailable: out of memory)",
        detail,
    )
}

/// The largest `config.json` this will read, before reading it.
pub const MAX_CONFIG_BYTES: usize = 8 * 1024 * 1024;

/// What a checkpoint says about how its weights are packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationDeclaration {
    /// `int4` or `int8`, from `num_bits`.
    pub bits: u32,
    /// How scales are shared, from `strategy` and `group_size`.
    pub granularity: Granularity,
    /// Whether zero points are stored.
    pub zero_points: ZeroPointSource,
    /// Modules the checkpoint says are **not** quantized.
    pub ignored: Vec<String>,
}

/// What a checkpoint says about itself, beyond its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointDeclaration {
    pub model_type: String,
    pub architecture: String,
    /// `None` when the checkpoint declares no quantization at all: a BF16
    /// checkpoint is a legitimate input and says so by omission.
    pub quantization: Option<QuantizationDeclaration>,
}

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    quantization_config: Option<RawQuantization>,
}

#[derive(Deserialize)]
struct RawQuantization {
    #[serde(default)]
    format: String,
    /// What the quantizer called itself. `compressed-tensors` checkpoints carry
    /// a `format`; auto-round and others identify themselves here instead, and
    /// a refusal that only quoted the empty `format` told the user nothing.
    #[serde(default)]
    quant_method: String,
    #[serde(default)]
    config_groups: std::collections::BTreeMap<String, RawGroup>,
    #[serde(default)]
    ignore: Vec<String>,
}

#[derive(Deserialize)]
struct RawGroup {
    weights: Option<RawWeights>,
}

#[derive(Deserialize)]
struct RawWeights {
    num_bits: Option<u32>,
    group_size: Option<i64>,
    symmetric: Option<bool>,
    strategy: Option<String>,
    #[serde(default)]
    actorder: Option<serde_json::Value>,
    #[serde(rename = "type")]
    #[serde(default)]
    kind: Option<String>,
}

/// Parse a checkpoint's `config.json`.
///
/// Refuses rather than guesses. Anything this repository has not measured --
/// an activation-order permutation, a packing format it does not read, more
/// than one quantization group -- is an error naming what it found, because a
/// generated selection that quietly dropped a permutation would produce an
/// artifact whose values are wrong in a way no checksum can see.
pub fn parse(text: &str) -> Result<CheckpointDeclaration> {
    if text.len() > MAX_CONFIG_BYTES {
        return Err(invalid(format_args!(
            "config.json is {} byte(s), above the {MAX_CONFIG_BYTES} cap checked before parsing",
            text.len()
        )));
    }
    let raw: RawConfig = serde_json::from_str(text)
        .map_err(|e| invalid(format_args!("config.json does not parse: {e}")))?;
    let architecture = raw
        .architectures
        .first()
        .cloned()
        .unwrap_or_else(|| raw.model_type.clone());

    let Some(q) = raw.quantization_config else {
        return Ok(CheckpointDeclaration {
            model_type: raw.model_type,
            architecture,
            quantization: None,
        });
    };
    if q.format != "pack-quantized" {
        let declared = if q.format.is_empty() {
            if q.quant_method.is_empty() {
                "no format and no quant_method".to_string()
            } else {
                format!("quant_method '{}'", q.quant_method)
            }
        } else {
            format!("format '{}'", q.format)
        };
        let known = match q.quant_method.as_str() {
            "auto-round" => {
                " auto-round stores GPTQ-style `qweight`, `qzeros` and `scales`, which is a                  different packing rather than a different spelling of this one: reading it needs                  a shared importer this repository does not have yet."
            }
            _ => "",
        };
        return Err(invalid(format_args!(
            "this checkpoint declares {declared}; only compressed-tensors 'pack-quantized' has \
             been measured here.{known} Guessing at a packing is how a conversion produces wrong \
             values that no checksum can see, so it is refused"
        )));
    }
    if q.config_groups.len() != 1 {
        return Err(invalid(format_args!(
            "this checkpoint declares {} quantization group(s); one has been measured, and a \
             selection covering several would have to say which tensors belong to which",
            q.config_groups.len()
        )));
    }
    let group = q.config_groups.values().next().expect("one group");
    let w = group.weights.as_ref().ok_or_else(|| {
        invalid_static("this checkpoint's quantization group declares no weights")
    })?;

    if let Some(order) = &w.actorder
        && !order.is_null()
    {
        return Err(invalid(format_args!(
            "this checkpoint declares actorder {order}: an activation-order permutation changes \
             which input column each code belongs to, and document 03 forbids ignoring one. It is \
             refused rather than dropped"
        )));
    }
    if let Some(kind) = &w.kind
        && kind != "int"
    {
        return Err(invalid(format_args!(
            "this checkpoint declares weight type '{kind}'; only 'int' has been measured here"
        )));
    }
    let bits = w
        .num_bits
        .ok_or_else(|| invalid_static("this checkpoint's weights declare no num_bits"))?;
    if bits != 4 && bits != 8 {
        return Err(invalid(format_args!(
            "this checkpoint declares {bits}-bit weights; 4 and 8 are what this reads"
        )));
    }
    let strategy = w.strategy.as_deref().unwrap_or("group");
    let granularity = match strategy {
        "channel" => Granularity::Channel,
        "group" => {
            let size = w.group_size.ok_or_else(|| {
                invalid_static("this checkpoint declares group strategy with no group_size")
            })?;
            match size {
                32 => Granularity::Group { size: 32 },
                128 => Granularity::Group { size: 128 },
                other => {
                    return Err(invalid(format_args!(
                        "this checkpoint declares group_size {other}; 32 and 128 are what this \
                         repository has measured"
                    )));
                }
            }
        }
        other => {
            return Err(invalid(format_args!(
                "this checkpoint declares strategy '{other}'; 'group' and 'channel' are what this \
                 reads"
            )));
        }
    };
    let zero_points = match w.symmetric {
        Some(true) => ZeroPointSource::Symmetric,
        _ => ZeroPointSource::PackedAlongOutput,
    };

    Ok(CheckpointDeclaration {
        model_type: raw.model_type,
        architecture,
        quantization: Some(QuantizationDeclaration {
            bits,
            granularity,
            zero_points,
            ignored: q.ignore,
        }),
    })
}

/// The largest `model.safetensors.index.json` this will read, before reading.
pub const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;

/// `weight_map` from a safetensors index: tensor name to shard file.
///
/// The index is the **authority** on what a checkpoint contains. Scanning a
/// directory and matching suffixes answers a different question, and the two
/// differ in ways that matter: on an auto-round checkpoint a suffix scan finds
/// no modules at all and would report a complete BF16-only plan.
pub fn parse_index(text: &str) -> Result<std::collections::BTreeMap<String, String>> {
    if text.len() > MAX_INDEX_BYTES {
        return Err(invalid(format_args!(
            "the safetensors index is {} byte(s), above the {MAX_INDEX_BYTES} cap checked before \
             parsing",
            text.len()
        )));
    }
    #[derive(Deserialize)]
    struct Index {
        weight_map: std::collections::BTreeMap<String, String>,
    }
    let parsed: Index = serde_json::from_str(text)
        .map_err(|e| invalid(format_args!("the safetensors index does not parse: {e}")))?;
    Ok(parsed.weight_map)
}
