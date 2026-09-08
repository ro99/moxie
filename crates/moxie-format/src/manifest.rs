//! Canonical manifest v1: schema, parsing from `&str`, and every validation rule.
//!
//! Document 03 fixes *what* manifest v1 must contain; [ADR 0005] fixes the
//! encoding (TOML, one `manifest.toml` per artifact directory, payloads in
//! sibling chunk files). This module is the schema and its validation, pure and
//! I/O-free: it parses a `&str` and validates the value. It never opens a file,
//! never knows a path exists, and never names a model family. `moxie-storage`
//! owns the filesystem half.
//!
//! [ADR 0005]: ../../../docs/decisions/adr/0005-toml-manifest-with-separate-chunks.md
//!
//! ## TOML layout
//!
//! ```toml
//! schema_version = 1
//! required_features = []
//! endianness = "little"
//!
//! [source]
//! model = "example"
//! revision = "r1"
//! license = "apache-2.0"
//! [[source.files]]
//! path = "original.safetensors"
//! sha256 = "<64 hex>"
//!
//! [tokenizer]
//! name = "example-tok"
//! version = "1"
//! digest = "<64 hex>"
//!
//! [template]
//! name = "example-chat"
//! version = "1"
//! digest = "<64 hex>"
//!
//! [architecture]
//! name = "example"
//! version = "1"
//! [architecture.metadata]
//! hidden = 32
//!
//! [provenance]
//! scale_convention = "affine-v1"
//! quantizer = "none"
//! calibration = "none"
//!
//! [[tensors]]
//! role = "w"
//! shape = [4, 4]
//! precision = "bf16-v1"
//! chunk = "chunk0.bin"
//! offset = 0
//! length = 32
//! sha256 = "<64 hex>"
//! alignment = 16
//! logical_order = 0
//!
//! [[excluded]]
//! role = "old-head"
//! reason = " основана"
//!
//! [completeness]
//! status = "complete"
//! missing = []
//! ```
//!
//! Affine tensors additionally carry `group_rule`, `scale_dtype` and
//! `zero_point`, and optionally `group_index`. BF16 tensors must not carry any
//! of them. The reader validates affine descriptors and then refuses to *read*
//! the tensor with an error naming M3: describing is v1, decoding is M3.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use moxie_types::{Error, Result};
use serde::Deserialize;

use crate::affine::{AffineDescriptor, Grouping, IntWidth};
use crate::scale::ScaleDtype as PayloadScaleDtype;
use crate::sha256::sha256_hex;

/// Manifests this reader accepts.
pub const SCHEMA_VERSION: u32 = 1;
/// A manifest is kilobytes; four orders of magnitude of headroom constrains no
/// real artifact while bounding the parse itself.
pub const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
/// The explicit cap that makes the bound on later O(n)/O(n log n) validation
/// passes stated rather than incidental.
pub const MAX_TENSORS: usize = 1_048_576;
/// The opaque architecture tree is the one field with no schema, so it is the
/// one that can be adversarially shaped. Depth is checked as the tree is
/// walked, and the walk stops at the limit.
pub const MAX_ARCH_DEPTH: usize = 64;
pub const MAX_ARCH_NODES: usize = 65_536;
/// The reader's own closed set of understood required features. Empty today:
/// any entry is refused by name, which is what makes a future field safe to
/// add.
pub const KNOWN_REQUIRED_FEATURES: &[&str] = &[];
pub const PRECISION_BF16_V1: &str = "bf16-v1";
pub const PRECISION_AFFINE_INT4_V1: &str = "affine-int4-v1";
pub const PRECISION_AFFINE_INT8_V1: &str = "affine-int8-v1";
pub const KNOWN_PRECISIONS: &[&str] = &[
    PRECISION_BF16_V1,
    PRECISION_AFFINE_INT4_V1,
    PRECISION_AFFINE_INT8_V1,
];

/// Raw TOML shape. Every table denies unknown fields so an unrecognized key is
/// a rejection rather than a silent drop.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    // Pinned by `check_version` before this schema deserializes, and required
    // here so `deny_unknown_fields` keeps recognizing it. Never read after
    // the gate; that is the point.
    #[allow(dead_code)]
    schema_version: u32,
    required_features: Vec<String>,
    endianness: String,
    source: RawSource,
    tokenizer: RawIdentity,
    template: RawIdentity,
    architecture: RawArchitecture,
    provenance: RawProvenance,
    tensors: Vec<RawTensor>,
    excluded: Vec<RawExcluded>,
    completeness: RawCompleteness,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSource {
    model: String,
    revision: String,
    license: String,
    files: Vec<RawSourceFile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSourceFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIdentity {
    name: String,
    version: String,
    digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArchitecture {
    name: String,
    version: String,
    metadata: toml::Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProvenance {
    scale_convention: String,
    quantizer: String,
    calibration: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    role: String,
    shape: Vec<i64>,
    precision: String,
    chunk: String,
    offset: i64,
    length: i64,
    sha256: String,
    alignment: i64,
    logical_order: i64,
    group_rule: Option<String>,
    scale_dtype: Option<String>,
    zero_point: Option<String>,
    group_index: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExcluded {
    role: String,
    reason: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompleteness {
    status: String,
    missing: Vec<String>,
}

/// A validated manifest: the owned value every later stage consumes.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub required_features: Vec<String>,
    pub endianness: Endianness,
    pub source: Source,
    pub tokenizer: Identity,
    pub template: Identity,
    pub architecture: Architecture,
    pub provenance: Provenance,
    pub tensors: Vec<Tensor>,
    pub excluded: Vec<Excluded>,
    pub completeness: Completeness,
}

/// The deliberately opaque architecture-metadata tree.
///
/// There is no accessor that returns a typed field of it, only the whole tree:
/// interpreting it is a `moxie-models-*` job. The type is named so that a later
/// reader knows the omission is deliberate.
#[derive(Debug, Clone)]
pub struct OpaqueArchMetadata(pub toml::Value);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub model: String,
    pub revision: String,
    pub license: String,
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct Identity {
    pub name: String,
    pub version: String,
    pub digest: String,
}

#[derive(Debug, Clone)]
pub struct Architecture {
    pub name: String,
    pub version: String,
    pub metadata: OpaqueArchMetadata,
}

#[derive(Debug, Clone)]
pub struct Provenance {
    pub scale_convention: String,
    pub quantizer: String,
    pub calibration: String,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub role: String,
    pub shape: Vec<u64>,
    pub precision: TensorPrecision,
    pub chunk: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
    pub alignment: u64,
    pub logical_order: u64,
    pub affine: Option<AffineFields>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPrecision {
    Bf16V1,
    AffineInt4V1,
    AffineInt8V1,
}

impl TensorPrecision {
    pub fn name(self) -> &'static str {
        match self {
            TensorPrecision::Bf16V1 => PRECISION_BF16_V1,
            TensorPrecision::AffineInt4V1 => PRECISION_AFFINE_INT4_V1,
            TensorPrecision::AffineInt8V1 => PRECISION_AFFINE_INT8_V1,
        }
    }

    pub fn is_affine(self) -> bool {
        !matches!(self, TensorPrecision::Bf16V1)
    }
}

#[derive(Debug, Clone)]
pub struct AffineFields {
    pub group_rule: GroupRule,
    pub scale_dtype: ScaleDtype,
    pub zero_point: ZeroPointMode,
    pub group_index: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRule {
    Contiguous32,
    Contiguous128,
    PerChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDtype {
    F16,
    Bf16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPointMode {
    Symmetric,
    PerGroup,
}

#[derive(Debug, Clone)]
pub struct Excluded {
    pub role: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum Completeness {
    Complete,
    Partial { missing: Vec<String> },
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// Parse and fully validate a manifest from its TOML text.
///
/// This covers every rule that does not need chunk-file lengths: schema
/// version, required features, endianness, presence and shape of every section,
/// tensor-count and architecture-tree limits, role uniqueness, shape/length
/// agreement, alignment, overlap, descriptor rules, path-string confinement and
/// completeness shape. [`validate_chunks`] finishes the job once the chunk
/// files have been statted.
pub fn parse(text: &str) -> Result<Manifest> {
    // Version first: a future schema is refused by version before any v1
    // field is looked at, even when v1's fields are missing or changed.
    check_version(text)?;
    let raw: RawManifest = toml::from_str(text)
        .map_err(|e| invalid(format!("manifest does not parse as TOML v1: {e}")))?;
    validate_raw(raw)
}

/// Chunk-file lengths, by chunk name, from statting the artifact directory.
/// Passed in so this crate stays I/O-free: it validates numbers, never files.
pub fn validate_chunks(manifest: &Manifest, chunks: &BTreeMap<String, u64>) -> Result<()> {
    for t in &manifest.tensors {
        let len = chunks.get(&t.chunk).ok_or_else(|| {
            invalid(format!(
                "tensor '{}' names chunk '{}', which does not exist",
                t.role, t.chunk
            ))
        })?;
        let end = t.offset.checked_add(t.length).ok_or_else(|| {
            invalid(format!(
                "tensor '{}' offset {} + length {} overflows",
                t.role, t.offset, t.length
            ))
        })?;
        if end > *len {
            return Err(invalid(format!(
                "tensor '{}' describes bytes {}..{end} but chunk '{}' holds {len} bytes: truncation",
                t.role, t.offset, t.chunk
            )));
        }
    }
    Ok(())
}

/// The version gate, read before the version-specific schema: a future
/// writer's artifact must be refused by version even when it no longer
/// carries v1's fields, so this shapes only `schema_version` and tolerates
/// every other key.
#[derive(Debug, Deserialize)]
struct VersionOnly {
    schema_version: u32,
}

fn check_version(text: &str) -> Result<()> {
    let v: VersionOnly = toml::from_str(text).map_err(|e| {
        // Malformed TOML, or no version at all: still a rejection, and the
        // message says which.
        let msg = e.to_string();
        if msg.contains("schema_version") {
            invalid(format!("manifest carries no schema_version: {e}"))
        } else {
            invalid(format!("manifest does not parse as TOML v1: {e}"))
        }
    })?;
    if v.schema_version != SCHEMA_VERSION {
        return Err(invalid(format!(
            "schema_version is {}, this reader accepts only version {}: refused by version before any other field",
            v.schema_version, SCHEMA_VERSION
        )));
    }
    Ok(())
}

fn validate_raw(raw: RawManifest) -> Result<Manifest> {
    // The version was already gated by `check_version` before deserializing
    // this schema; re-checking here would only repeat it.
    for f in &raw.required_features {
        if !KNOWN_REQUIRED_FEATURES.contains(&f.as_str()) {
            return Err(invalid(format!(
                "unknown required feature '{f}': refused by name; a future writer's artifact needs a future reader"
            )));
        }
    }
    let endianness = match raw.endianness.as_str() {
        "little" => Endianness::Little,
        other => {
            return Err(invalid(format!(
                "endianness '{other}' is refused rather than byte-swapped: nothing here has ever been tested on one"
            )));
        }
    };
    if raw.tensors.len() > MAX_TENSORS {
        return Err(invalid(format!(
            "manifest lists {} tensors, above the {} cap checked before validation",
            raw.tensors.len(),
            MAX_TENSORS
        )));
    }
    check_arch_limits(&raw.architecture.metadata)?;

    let source = validate_source(raw.source)?;
    let tokenizer = validate_identity(raw.tokenizer, "tokenizer")?;
    let template = validate_identity(raw.template, "template")?;
    let architecture = Architecture {
        name: nonempty(raw.architecture.name, "architecture.name")?,
        version: nonempty(raw.architecture.version, "architecture.version")?,
        metadata: OpaqueArchMetadata(raw.architecture.metadata),
    };
    let provenance = Provenance {
        scale_convention: nonempty(
            raw.provenance.scale_convention,
            "provenance.scale_convention",
        )?,
        quantizer: nonempty(raw.provenance.quantizer, "provenance.quantizer")?,
        calibration: nonempty(raw.provenance.calibration, "provenance.calibration")?,
    };

    let mut tensors = Vec::with_capacity(raw.tensors.len());
    for t in raw.tensors {
        tensors.push(validate_tensor(t)?);
    }
    // Every role unique: a manifest whose second entry silently wins is refused.
    let mut roles = BTreeSet::new();
    for t in &tensors {
        if !roles.insert(t.role.clone()) {
            return Err(invalid(format!(
                "duplicate tensor role '{}': the second entry would silently win",
                t.role
            )));
        }
    }
    // Logical orders unique: they are an ordering, and duplicates are ambiguous.
    let mut orders = BTreeSet::new();
    for t in &tensors {
        if !orders.insert(t.logical_order) {
            return Err(invalid(format!(
                "duplicate logical_order {} (tensor '{}'): ordering must be unique",
                t.logical_order, t.role
            )));
        }
    }
    check_overlap(&tensors)?;

    let mut excluded = Vec::with_capacity(raw.excluded.len());
    for e in raw.excluded {
        excluded.push(Excluded {
            role: nonempty(e.role, "excluded.role")?,
            reason: nonempty(e.reason, "excluded.reason")?,
        });
    }

    let completeness = match raw.completeness.status.as_str() {
        "complete" => {
            if !raw.completeness.missing.is_empty() {
                return Err(invalid(
                    "completeness is 'complete' but lists missing tensors",
                ));
            }
            Completeness::Complete
        }
        "partial" => {
            if raw.completeness.missing.is_empty() {
                return Err(invalid(
                    "completeness is 'partial' but names no missing tensor",
                ));
            }
            for m in &raw.completeness.missing {
                if m.is_empty() {
                    return Err(invalid("completeness lists an empty missing role"));
                }
            }
            Completeness::Partial {
                missing: raw.completeness.missing,
            }
        }
        other => {
            return Err(invalid(format!(
                "completeness status '{other}': expected 'complete' or 'partial'"
            )));
        }
    };

    Ok(Manifest {
        required_features: raw.required_features,
        endianness,
        source,
        tokenizer,
        template,
        architecture,
        provenance,
        tensors,
        excluded,
        completeness,
    })
}

fn nonempty(s: String, field: &str) -> Result<String> {
    if s.is_empty() {
        return Err(invalid(format!(
            "{field} must be present and non-empty: absence is a rejection, not a default"
        )));
    }
    Ok(s)
}

fn validate_source(s: RawSource) -> Result<Source> {
    let mut files = Vec::with_capacity(s.files.len());
    if s.files.is_empty() {
        return Err(invalid(
            "source.files must list at least one source file checksum: recorded, never fetched",
        ));
    }
    for f in s.files {
        files.push(SourceFile {
            path: nonempty(f.path, "source.files.path")?,
            sha256: validate_sha256(f.sha256, "source.files.sha256")?,
        });
    }
    Ok(Source {
        model: nonempty(s.model, "source.model")?,
        revision: nonempty(s.revision, "source.revision")?,
        license: nonempty(s.license, "source.license")?,
        files,
    })
}

fn validate_identity(id: RawIdentity, what: &str) -> Result<Identity> {
    Ok(Identity {
        name: nonempty(id.name, &format!("{what}.name"))?,
        version: nonempty(id.version, &format!("{what}.version"))?,
        digest: validate_sha256(id.digest, &format!("{what}.digest"))?,
    })
}

fn validate_sha256(s: String, field: &str) -> Result<String> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{field} must be 64 lowercase hex digits, got {s:?}"
        )));
    }
    Ok(s.to_ascii_lowercase())
}

fn to_u64(v: i64, field: &str) -> Result<u64> {
    u64::try_from(v).map_err(|_| {
        invalid(format!(
            "{field} is {v}: negative values are never valid here"
        ))
    })
}

fn validate_tensor(t: RawTensor) -> Result<Tensor> {
    let role = nonempty(t.role, "tensors.role")?;
    validate_chunk_name(&t.chunk).map_err(|d| invalid(format!("tensor '{role}': {d}")))?;
    if t.shape.is_empty() {
        return Err(invalid(format!("tensor '{role}': shape must be non-empty")));
    }
    let mut shape = Vec::with_capacity(t.shape.len());
    for (i, d) in t.shape.iter().enumerate() {
        let d = to_u64(*d, &format!("tensor '{role}' shape[{i}]"))?;
        if d == 0 {
            return Err(invalid(format!(
                "tensor '{role}': shape[{i}] is 0; dimensions must be positive"
            )));
        }
        shape.push(d);
    }
    let precision = match t.precision.as_str() {
        PRECISION_BF16_V1 => TensorPrecision::Bf16V1,
        PRECISION_AFFINE_INT4_V1 => TensorPrecision::AffineInt4V1,
        PRECISION_AFFINE_INT8_V1 => TensorPrecision::AffineInt8V1,
        other => {
            return Err(invalid(format!(
                "tensor '{role}': precision '{other}' is not one of {KNOWN_PRECISIONS:?}"
            )));
        }
    };
    let offset = to_u64(t.offset, &format!("tensor '{role}' offset"))?;
    let length = to_u64(t.length, &format!("tensor '{role}' length"))?;
    if length == 0 {
        return Err(invalid(format!(
            "tensor '{role}': length 0 describes no bytes"
        )));
    }
    let sha256 = validate_sha256(t.sha256, &format!("tensor '{role}' sha256"))?;
    let alignment = to_u64(t.alignment, &format!("tensor '{role}' alignment"))?;
    if !alignment.is_power_of_two() {
        return Err(invalid(format!(
            "tensor '{role}': alignment {alignment} is not a power of two"
        )));
    }
    if offset % alignment != 0 {
        return Err(invalid(format!(
            "tensor '{role}': offset {offset} is not a multiple of alignment {alignment}"
        )));
    }
    let logical_order = to_u64(t.logical_order, &format!("tensor '{role}' logical_order"))?;

    // Checked shape product, before anything multiplies by it.
    let mut elements: u64 = 1;
    for d in &shape {
        elements = elements
            .checked_mul(*d)
            .ok_or_else(|| invalid(format!("tensor '{role}': shape product overflows u64")))?;
    }

    let affine = match precision {
        TensorPrecision::Bf16V1 => {
            if t.group_rule.is_some()
                || t.scale_dtype.is_some()
                || t.zero_point.is_some()
                || t.group_index.is_some()
            {
                return Err(invalid(format!(
                    "tensor '{role}': a bf16-v1 tensor must not carry affine fields (group_rule/scale_dtype/zero_point/group_index)"
                )));
            }
            let need = elements
                .checked_mul(2)
                .ok_or_else(|| invalid(format!("tensor '{role}': BF16 byte count overflows")))?;
            if need != length {
                return Err(invalid(format!(
                    "tensor '{role}': shape product {elements} * 2 BF16 bytes = {need}, but length is {length}: shape and byte count disagree"
                )));
            }
            None
        }
        TensorPrecision::AffineInt4V1 | TensorPrecision::AffineInt8V1 => {
            let group_rule = t.group_rule.ok_or_else(|| {
                invalid(format!(
                    "tensor '{role}': affine tensors must carry group_rule"
                ))
            })?;
            let group_rule = match group_rule.as_str() {
                "contiguous-32" => GroupRule::Contiguous32,
                "contiguous-128" => GroupRule::Contiguous128,
                "per-channel" => GroupRule::PerChannel,
                other => {
                    return Err(invalid(format!(
                        "tensor '{role}': group_rule '{other}' is outside the closed set {{contiguous-32, contiguous-128, per-channel}}; widening it is an ADR"
                    )));
                }
            };
            let scale_dtype = t.scale_dtype.ok_or_else(|| {
                invalid(format!(
                    "tensor '{role}': affine tensors must carry scale_dtype"
                ))
            })?;
            let scale_dtype = match scale_dtype.as_str() {
                "f16" => ScaleDtype::F16,
                "bf16" => ScaleDtype::Bf16,
                "f32" => ScaleDtype::F32,
                other => {
                    return Err(invalid(format!(
                        "tensor '{role}': scale_dtype '{other}' is outside {{f16, bf16, f32}}"
                    )));
                }
            };
            let zero_point = t.zero_point.ok_or_else(|| {
                invalid(format!(
                    "tensor '{role}': affine tensors must carry zero_point"
                ))
            })?;
            let zero_point = match zero_point.as_str() {
                "symmetric" => ZeroPointMode::Symmetric,
                "per-group" => ZeroPointMode::PerGroup,
                other => {
                    return Err(invalid(format!(
                        "tensor '{role}': zero_point '{other}' is outside {{symmetric, per-group}}"
                    )));
                }
            };
            let group_index = match t.group_index {
                None => None,
                Some(idx) => {
                    if shape.len() < 2 {
                        return Err(invalid(format!(
                            "tensor '{role}': group_index needs at least a 2-D shape to index input channels"
                        )));
                    }
                    let in_features = shape[shape.len() - 1];
                    if idx.len() as u64 != in_features {
                        return Err(invalid(format!(
                            "tensor '{role}': group_index has {} entries for {in_features} input channels",
                            idx.len()
                        )));
                    }
                    let groups = match group_rule {
                        GroupRule::PerChannel => 1u64,
                        GroupRule::Contiguous32 => in_features.div_ceil(32),
                        GroupRule::Contiguous128 => in_features.div_ceil(128),
                    };
                    let mut out = Vec::with_capacity(idx.len());
                    for (k, g) in idx.iter().enumerate() {
                        let g = to_u64(*g, &format!("tensor '{role}' group_index[{k}]"))?;
                        if g >= groups {
                            return Err(invalid(format!(
                                "tensor '{role}': group_index[{k}] is group {g}, but there are only {groups} groups"
                            )));
                        }
                        let g = u32::try_from(g).map_err(|_| {
                            invalid(format!(
                                "tensor '{role}': group_index[{k}] is group {g}, above u32"
                            ))
                        })?;
                        out.push(g);
                    }
                    Some(out)
                }
            };
            // NOTE: no `product(shape) * element_size == length` check for
            // affine tensors. Their chunk payload (codes, scales, zero points)
            // has no single element size, and its exact layout is M3's reader
            // contract. V1 reserves the byte range and validates the
            // descriptor; the reader refuses to read the tensor.
            //
            // The descriptor is then checked by the shared
            // `AffineDescriptor::validate`, not by a second copy of its
            // rules here: manifest-level checks above exist for message
            // quality (they name the tensor and field), but the shared
            // validator is the authority on descriptor coherence -- e.g. a
            // group with a scale that no input column maps to. Duplicated
            // validation already drifted once (an all-zero group-index map
            // passed here and failed there); the gate below is the fix.
            let width = match precision {
                TensorPrecision::AffineInt4V1 => IntWidth::Int4,
                TensorPrecision::AffineInt8V1 => IntWidth::Int8,
                TensorPrecision::Bf16V1 => unreachable!("affine branch"),
            };
            // Last dimension is input channels; the leading product is output
            // channels. Both come from the already-validated positive shape.
            // `try_from`, never `as`: truncating conversions have no place in
            // the validator.
            let in_features = usize::try_from(shape[shape.len() - 1])
                .map_err(|_| invalid(format!("tensor '{role}': input dimension does not fit")))?;
            let mut out_features: usize = 1;
            for d in &shape[..shape.len() - 1] {
                let d = usize::try_from(*d).map_err(|_| {
                    invalid(format!("tensor '{role}': a leading dimension does not fit"))
                })?;
                out_features = out_features.checked_mul(d).ok_or_else(|| {
                    invalid(format!(
                        "tensor '{role}': leading-dimension product overflows"
                    ))
                })?;
            }
            let desc = AffineDescriptor {
                width,
                out_features,
                in_features,
                grouping: match group_rule {
                    GroupRule::Contiguous32 => Grouping::Contiguous { size: 32 },
                    GroupRule::Contiguous128 => Grouping::Contiguous { size: 128 },
                    GroupRule::PerChannel => Grouping::PerOutputChannel,
                },
                group_index: group_index.clone(),
                scale_dtype: match scale_dtype {
                    ScaleDtype::F16 => PayloadScaleDtype::F16,
                    ScaleDtype::Bf16 => PayloadScaleDtype::Bf16,
                    ScaleDtype::F32 => PayloadScaleDtype::F32,
                },
            };
            desc.validate().map_err(|e| {
                invalid(format!(
                    "tensor '{role}': shared affine descriptor rejects it: {e}"
                ))
            })?;
            Some(AffineFields {
                group_rule,
                scale_dtype,
                zero_point,
                group_index,
            })
        }
    };

    Ok(Tensor {
        role,
        shape,
        precision,
        chunk: t.chunk,
        offset,
        length,
        sha256,
        alignment,
        logical_order,
        affine,
    })
}

/// No two tensors' byte ranges may intersect, per chunk -- including the
/// total-containment case a pairwise "starts inside" test misses.
fn check_overlap(tensors: &[Tensor]) -> Result<()> {
    let mut by_chunk: BTreeMap<&str, Vec<&Tensor>> = BTreeMap::new();
    for t in tensors {
        by_chunk.entry(t.chunk.as_str()).or_default().push(t);
    }
    for (chunk, mut list) in by_chunk {
        list.sort_by_key(|t| (t.offset, t.length));
        for w in list.windows(2) {
            let (a, b) = (w[0], w[1]);
            let a_end = a
                .offset
                .checked_add(a.length)
                .ok_or_else(|| invalid(format!("tensor '{}' offset + length overflows", a.role)))?;
            // Sorted by offset, so any intersection means b starts inside
            // a -- including b wholly contained in a.
            if b.offset < a_end {
                return Err(invalid(format!(
                    "tensors '{}' ({}..{a_end}) and '{}' ({}..) overlap in chunk '{chunk}', including containment",
                    a.role, a.offset, b.role, b.offset
                )));
            }
        }
    }
    Ok(())
}

/// Every chunk reference must be a single path component, checked on the
/// string before any path is joined: no separator, no `.` or `..`, not
/// absolute, no prefix or root component.
pub fn validate_chunk_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty() {
        return Err("chunk name is empty".into());
    }
    if name == "." || name == ".." {
        return Err(format!(
            "chunk name '{name}' is a directory traversal token"
        ));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(format!("chunk name '{name}' contains a separator"));
    }
    if name.starts_with('/') || name.starts_with('\\') {
        return Err(format!("chunk name '{name}' is absolute"));
    }
    if name.len() >= 2 && name.as_bytes()[1] == b':' {
        return Err(format!("chunk name '{name}' carries a drive prefix"));
    }
    if name.contains(':') {
        return Err(format!("chunk name '{name}' carries a prefix"));
    }
    if name.contains('\0') {
        return Err(format!("chunk name '{name}' contains a null byte"));
    }
    if name == "/" || name.starts_with("~/") {
        return Err(format!("chunk name '{name}' is absolute"));
    }
    Ok(())
}

/// Depth is checked as the tree is walked, and the walk stops at the limit.
/// `toml` itself is a non-recursive parser, so this is a bound on *our*
/// traversal and hashing of the value, not a stack-overflow guard for the
/// parser.
pub fn check_arch_limits(value: &toml::Value) -> Result<()> {
    let mut nodes = 0usize;
    check_arch_walk(value, 0, &mut nodes)
}

fn check_arch_walk(value: &toml::Value, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| invalid("architecture metadata node count overflows"))?;
    if *nodes > MAX_ARCH_NODES {
        return Err(invalid(format!(
            "architecture metadata has more than {MAX_ARCH_NODES} nodes: the walk stops at the limit"
        )));
    }
    if depth > MAX_ARCH_DEPTH {
        return Err(invalid(format!(
            "architecture metadata is deeper than {MAX_ARCH_DEPTH}: the walk stops at the limit"
        )));
    }
    match value {
        toml::Value::Array(items) => {
            for v in items {
                check_arch_walk(v, depth + 1, nodes)?;
            }
        }
        toml::Value::Table(map) => {
            for (_, v) in map {
                check_arch_walk(v, depth + 1, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Stable artifact identity: the descriptor fields document 03 lists, hashed
/// with SHA-256. Architecture metadata participates through a canonical
/// encoding with sorted keys, so key order in the TOML never changes identity.
///
/// Every field and collection is length-prefixed. Bare delimiters (NUL,
/// newline) are not enough: those bytes are legal inside parsed strings, so
/// `"x\\0y"+"z"` and `"x"+"y\\0z"` would hash identically. With each
/// segment carrying its own length, concatenation is injective and no two
/// distinct manifests share a digest.
pub fn artifact_identity(manifest: &Manifest) -> String {
    let mut w = IdentityWriter::new();
    w.tag("manifest-v1");
    w.field_u64("schema", SCHEMA_VERSION as u64);
    w.field_str("source-model", &manifest.source.model);
    w.field_str("source-revision", &manifest.source.revision);
    w.field_str("source-license", &manifest.source.license);
    for f in &manifest.source.files {
        w.tag("source-file");
        w.str(&f.path);
        w.str(&f.sha256);
    }
    w.field_str("tokenizer-name", &manifest.tokenizer.name);
    w.field_str("tokenizer-version", &manifest.tokenizer.version);
    w.field_str("tokenizer-digest", &manifest.tokenizer.digest);
    w.field_str("template-name", &manifest.template.name);
    w.field_str("template-version", &manifest.template.version);
    w.field_str("template-digest", &manifest.template.digest);
    w.field_str("arch-name", &manifest.architecture.name);
    w.field_str("arch-version", &manifest.architecture.version);
    w.tag("arch-metadata");
    w.value(&manifest.architecture.metadata.0);
    w.field_str("scale-convention", &manifest.provenance.scale_convention);
    w.field_str("quantizer", &manifest.provenance.quantizer);
    w.field_str("calibration", &manifest.provenance.calibration);
    let mut tensors: Vec<&Tensor> = manifest.tensors.iter().collect();
    tensors.sort_by(|a, b| {
        a.logical_order
            .cmp(&b.logical_order)
            .then(a.role.cmp(&b.role))
    });
    for t in tensors {
        w.tag("tensor");
        w.str(&t.role);
        w.u64(t.shape.len() as u64);
        for d in &t.shape {
            w.u64(*d);
        }
        w.str(t.precision.name());
        w.str(&t.chunk);
        w.u64(t.offset);
        w.u64(t.length);
        w.str(&t.sha256);
        w.u64(t.alignment);
        w.u64(t.logical_order);
        if let Some(a) = &t.affine {
            w.tag("affine");
            w.str(match a.group_rule {
                GroupRule::Contiguous32 => "contiguous-32",
                GroupRule::Contiguous128 => "contiguous-128",
                GroupRule::PerChannel => "per-channel",
            });
            w.str(match a.scale_dtype {
                ScaleDtype::F16 => "f16",
                ScaleDtype::Bf16 => "bf16",
                ScaleDtype::F32 => "f32",
            });
            w.str(match a.zero_point {
                ZeroPointMode::Symmetric => "symmetric",
                ZeroPointMode::PerGroup => "per-group",
            });
            match &a.group_index {
                None => w.tag("no-group-index"),
                Some(map) => {
                    w.tag("group-index");
                    w.u64(map.len() as u64);
                    for g in map {
                        w.u64(*g as u64);
                    }
                }
            }
        }
    }
    let mut excl: Vec<&Excluded> = manifest.excluded.iter().collect();
    excl.sort_by(|a, b| a.role.cmp(&b.role));
    for e in excl {
        w.tag("excluded");
        w.str(&e.role);
        w.str(&e.reason);
    }
    match &manifest.completeness {
        Completeness::Complete => w.tag("complete"),
        Completeness::Partial { missing } => {
            w.tag("partial");
            for m in missing {
                w.str(m);
            }
        }
    }
    sha256_hex(&w.finish())
}

/// Length-prefixed identity encoding: every segment carries its own byte
/// length, so concatenation of distinct manifests can never collide.
struct IdentityWriter {
    buf: Vec<u8>,
}

impl IdentityWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.buf.extend_from_slice(b);
    }

    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    fn tag(&mut self, t: &str) {
        self.str(t);
    }

    fn field_str(&mut self, name: &str, value: &str) {
        self.str(name);
        self.str(value);
    }

    fn field_u64(&mut self, name: &str, value: u64) {
        self.str(name);
        self.u64(value);
    }

    fn value(&mut self, v: &toml::Value) {
        match v {
            toml::Value::String(s) => {
                self.buf.push(0x73);
                self.str(s);
            }
            toml::Value::Integer(i) => {
                self.buf.push(0x69);
                self.buf.extend_from_slice(&i.to_le_bytes());
            }
            toml::Value::Float(f) => {
                self.buf.push(0x66);
                self.buf.extend_from_slice(&f.to_bits().to_le_bytes());
            }
            toml::Value::Boolean(b) => {
                self.buf.push(0x62);
                self.buf.push(u8::from(*b));
            }
            toml::Value::Datetime(d) => {
                self.buf.push(0x64);
                self.str(&d.to_string());
            }
            toml::Value::Array(items) => {
                self.buf.push(0x61);
                self.u64(items.len() as u64);
                for v in items {
                    self.value(v);
                }
            }
            toml::Value::Table(map) => {
                // Sorted keys: `toml::map::Map` preserves insertion order, so
                // sort here for stability.
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                self.buf.push(0x74);
                self.u64(keys.len() as u64);
                for k in keys {
                    self.str(k);
                    self.value(&map[k]);
                }
            }
        }
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}
