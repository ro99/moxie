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
use moxie_types::{Error, Result, declared_value::DeclaredValue};

use crate::invalid_static;

fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a checkpoint config this repacker will not read (detail unavailable: out of memory)",
        detail,
    )
}

/// The largest `config.json` this will read, before reading it.
pub const MAX_CONFIG_BYTES: usize = 8 * 1024 * 1024;

/// Decode the checkpoint's text configuration into architecture-neutral
/// dotted fields. Model crates consume the plain values without depending on
/// JSON parsing.
pub fn declared_text_fields(
    text: &str,
) -> Result<std::collections::BTreeMap<String, DeclaredValue>> {
    use serde_json::Value;

    if text.len() > MAX_CONFIG_BYTES {
        return Err(invalid(format_args!(
            "config.json is {} byte(s), above the {MAX_CONFIG_BYTES} cap checked before parsing",
            text.len()
        )));
    }
    let root: Value = serde_json::from_str(text)
        .map_err(|e| invalid(format_args!("config.json does not parse: {e}")))?;
    let object = root.as_object().ok_or_else(|| {
        invalid_static("config.json root must be an object to read declared text fields")
    })?;
    let source = match object.get("text_config") {
        Some(Value::Object(text_config)) => text_config,
        Some(other) => {
            return Err(invalid(format_args!(
                "config.json text_config must be an object, found {other}"
            )));
        }
        None => object,
    };
    let mut fields = std::collections::BTreeMap::new();
    for (key, value) in source {
        flatten_declared(key, value, &mut fields)?;
    }
    Ok(fields)
}

fn flatten_declared(
    key: &str,
    value: &serde_json::Value,
    fields: &mut std::collections::BTreeMap<String, DeclaredValue>,
) -> Result<()> {
    match value {
        serde_json::Value::Object(object) => {
            for (child, value) in object {
                let dotted = format!("{key}.{child}");
                flatten_declared(&dotted, value, fields)?;
            }
        }
        // Everything else, arrays included: `declared_value` already
        // recurses into an array's own items, so restating that here would
        // be a second copy of the same match.
        value => insert_declared(key, declared_value(value)?, fields)?,
    }
    Ok(())
}

fn declared_value(value: &serde_json::Value) -> Result<DeclaredValue> {
    use serde_json::Value;

    Ok(match value {
        Value::Null => DeclaredValue::Null,
        Value::Bool(value) => DeclaredValue::Bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                DeclaredValue::Int(value)
            } else if value.is_u64() {
                return Err(invalid(format_args!(
                    "declared unsigned integer {value} does not fit Int(i64)"
                )));
            } else if let Some(value) = value.as_f64() {
                DeclaredValue::Float(value)
            } else {
                return Err(invalid(format_args!(
                    "declared number {value} cannot be represented"
                )));
            }
        }
        Value::String(value) => DeclaredValue::Str(value.clone()),
        Value::Array(items) => DeclaredValue::List(
            items
                .iter()
                .map(declared_value)
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::Object(_) => {
            return Err(invalid_static(
                "object values inside declared lists cannot be represented",
            ));
        }
    })
}

fn insert_declared(
    key: &str,
    value: DeclaredValue,
    fields: &mut std::collections::BTreeMap<String, DeclaredValue>,
) -> Result<()> {
    if fields.insert(key.to_string(), value).is_some() {
        return Err(invalid(format_args!(
            "nested config keys flatten to the duplicate field {key:?}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerSerialization {
    CompressedTensors,
    GptqV1 { requires_group_index: bool },
}

/// What a checkpoint says about how its weights are packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationDeclaration {
    pub serialization: IntegerSerialization,
    /// AutoRound regex overrides requiring unquantized source tensors.
    pub passthrough_patterns: Vec<String>,
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
    #[serde(default)]
    packing_format: String,
    #[serde(default)]
    autoround_version: String,
    bits: Option<u32>,
    group_size: Option<i64>,
    sym: Option<bool>,
    desc_act: Option<bool>,
    data_type: Option<String>,
    #[serde(default)]
    extra_config: std::collections::BTreeMap<String, RawOverride>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOverride {
    bits: u32,
    data_type: String,
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
    if q.quant_method == "auto-round" && q.packing_format == "auto_round:auto_gptq" {
        if !q.format.is_empty() || !q.config_groups.is_empty() {
            return Err(invalid_static(
                "AutoRound GPTQ declaration conflicts with compressed-tensors fields",
            ));
        }
        if q.autoround_version != "0.15.0"
            || q.data_type.as_deref() != Some("int")
            || q.sym.is_none()
        {
            return Err(invalid_static(
                "AutoRound GPTQ import requires the pinned 0.15.0 integer declaration and explicit symmetry",
            ));
        }
        let bits = q
            .bits
            .ok_or_else(|| invalid_static("AutoRound declares no bits"))?;
        if !matches!(bits, 4 | 8) {
            return Err(invalid_static("AutoRound importer supports INT4/INT8 only"));
        }
        let size = match q.group_size {
            Some(32) => 32,
            Some(128) => 128,
            _ => return Err(invalid_static("AutoRound importer requires group32/128")),
        };
        let mut passthrough_patterns = Vec::new();
        for (pattern, over) in q.extra_config {
            if over.bits != 16 || !matches!(over.data_type.as_str(), "fp" | "float") {
                return Err(invalid_static(
                    "AutoRound per-module quantized overrides need their own packing specification; only declared 16-bit passthrough overrides are supported",
                ));
            }
            passthrough_patterns.push(pattern);
        }
        return Ok(CheckpointDeclaration {
            model_type: raw.model_type,
            architecture,
            quantization: Some(QuantizationDeclaration {
                serialization: IntegerSerialization::GptqV1 {
                    requires_group_index: q.desc_act.unwrap_or(false),
                },
                passthrough_patterns,
                bits,
                granularity: Granularity::Group { size },
                // GPTQ carries explicit qzeros even for symmetric sources.
                zero_points: ZeroPointSource::PackedAlongOutput,
                ignored: q.ignore,
            }),
        });
    }
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
        return Err(invalid(format_args!(
            "this checkpoint declares {declared}; supported declarations are compressed-tensors \
             pack-quantized and pinned AutoRound 0.15.0 auto_round:auto_gptq. \
             Unsupported packing is refused rather than inferred"
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

    // compressed-tensors 0.17.0 ActivationOrdering: static aliases weight,
    // which changes calibration but preserves the saved column/group order.
    // group/dynamic instead needs weight_g_idx; it is not an alias of this lane.
    if let Some(order) = &w.actorder
        && !order.is_null()
        && !matches!(order.as_str(), Some("static" | "weight"))
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
            serialization: IntegerSerialization::CompressedTensors,
            passthrough_patterns: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(order: serde_json::Value) -> String {
        serde_json::json!({"model_type":"fixture","quantization_config":{
            "format":"pack-quantized", "config_groups":{"group_0":{"weights":{
                "num_bits":4,"group_size":128,"symmetric":true,"strategy":"group",
                "type":"int","actorder":order
            }}}
        }})
        .to_string()
    }

    #[test]
    fn static_and_weight_are_contiguous_serialization_aliases() {
        let baseline = parse(&config(serde_json::Value::Null)).unwrap();
        for order in ["static", "weight"] {
            assert_eq!(parse(&config(order.into())).unwrap(), baseline);
        }
    }

    #[test]
    fn mapped_unknown_and_malformed_ordering_is_not_discarded() {
        for order in [
            serde_json::json!("group"),
            serde_json::json!("dynamic"),
            serde_json::json!("unknown"),
            serde_json::json!(true),
            serde_json::json!({}),
        ] {
            assert!(
                parse(&config(order))
                    .unwrap_err()
                    .to_string()
                    .contains("actorder")
            );
        }
    }

    #[test]
    fn declared_text_fields_flatten_values_without_interpreting_them() {
        let fields = declared_text_fields(
            r#"{"outside":7,"text_config":{"hidden_size":12,"active":true,"name":"Gemma","unset":null,"layer_types":["sliding_attention","full_attention"],"rope_parameters":{"full_attention":{"rope_theta":1000000.0}}}}"#,
        )
        .unwrap();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields["hidden_size"], DeclaredValue::Int(12));
        assert_eq!(fields["active"], DeclaredValue::Bool(true));
        assert_eq!(fields["name"], DeclaredValue::Str("Gemma".into()));
        assert_eq!(fields["unset"], DeclaredValue::Null);
        assert_eq!(
            fields["layer_types"],
            DeclaredValue::List(vec![
                DeclaredValue::Str("sliding_attention".into()),
                DeclaredValue::Str("full_attention".into()),
            ])
        );
        assert_eq!(
            fields["rope_parameters.full_attention.rope_theta"],
            DeclaredValue::Float(1_000_000.0)
        );
        assert_eq!(
            declared_text_fields(r#"{"root":1,"nested":{"leaf":2}}"#).unwrap()["nested.leaf"],
            DeclaredValue::Int(2)
        );
        assert!(declared_text_fields("[]").is_err());
        assert!(declared_text_fields(r#"{"text_config":null}"#).is_err());
    }
}
