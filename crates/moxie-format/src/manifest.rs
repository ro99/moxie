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
use crate::payload::ZeroPointSection as PayloadZeroPoints;
use crate::scale::ScaleDtype as PayloadScaleDtype;
use crate::sha256::sha256_hex;

/// The schema a repack **writes**: canonical v2, safetensors shards plus this
/// manifest ([ADR 0025]).
///
/// [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md
pub const SCHEMA_VERSION: u32 = 2;

/// The schemas this reader **accepts**.
///
/// Version 1 -- raw chunk files -- keeps its meaning and its reader: ADR 0025
/// says existing v1 artifacts are neither reinterpreted nor converted. Nothing
/// writes one any more. The transitional read path expires at M11 item 4, or
/// earlier if a task establishes that no v1 artifact exists outside tests.
pub const SUPPORTED_SCHEMA_VERSIONS: &[u32] = &[1, 2];
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
    /// Version 1 placement: one byte range of one raw chunk file. Absent in
    /// version 2, where the components below say where the bytes are.
    #[serde(default)]
    chunk: Option<String>,
    #[serde(default)]
    offset: Option<i64>,
    #[serde(default)]
    length: Option<i64>,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    alignment: Option<i64>,
    /// Version 2 placement: one entry per physical safetensors tensor.
    #[serde(default)]
    components: Vec<RawComponent>,
    logical_order: i64,
    group_rule: Option<String>,
    scale_dtype: Option<String>,
    zero_point: Option<String>,
    group_index: Option<Vec<i64>>,
}

/// One physical tensor of a version-2 artifact.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawComponent {
    kind: String,
    file: String,
    name: String,
    sha256: String,
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
    pub logical_order: u64,
    pub affine: Option<AffineFields>,
    /// Where this tensor's bytes are, which is the one thing the two schema
    /// versions disagree about.
    pub placement: Placement,
}

/// Where a tensor's bytes live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Version 1: one byte range of one raw chunk file, one checksum.
    Chunk {
        chunk: String,
        offset: u64,
        length: u64,
        sha256: String,
        alignment: u64,
    },
    /// Version 2: one or more physical safetensors tensors, each in a named
    /// shard, each with its own checksum ([ADR 0025]).
    ///
    /// [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md
    Components(Vec<Component>),
}

/// One physical tensor of a version-2 artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub kind: crate::canonical::ComponentKind,
    /// The shard file, a single path component of the artifact directory.
    pub file: String,
    /// The tensor's name inside that shard.
    pub name: String,
    /// SHA-256 of this component's payload bytes -- its `data_offsets` range,
    /// and nothing else.
    pub sha256: String,
}

impl Tensor {
    /// The v1 byte range, when this tensor has one.
    pub fn chunk_range(&self) -> Option<(&str, u64, u64, &str)> {
        match &self.placement {
            Placement::Chunk {
                chunk,
                offset,
                length,
                sha256,
                ..
            } => Some((chunk.as_str(), *offset, *length, sha256.as_str())),
            Placement::Components(_) => None,
        }
    }

    /// The v2 components, when this tensor has them.
    pub fn components(&self) -> Option<&[Component]> {
        match &self.placement {
            Placement::Components(c) => Some(c),
            Placement::Chunk { .. } => None,
        }
    }
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

/// This module's refusals, composed **fallibly**.
///
/// Takes `fmt::Arguments` rather than a `String`, so the message is never
/// built by an allocation that can abort: [`crate::invalid_fmt`] grows its
/// buffer through `try_reserve` and falls back to a borrowed static detail.
/// Task 0024's review reached `SIGABRT` here by refusing one allocation while
/// this module refused a malformed artifact.
fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed canonical manifest (detail unavailable: out of memory)",
        detail,
    )
}

/// A refusal whose whole message is static: allocation-free, always available.
#[allow(dead_code)]
fn invalid_static(detail: &'static str) -> Error {
    crate::invalid_static(detail)
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
    let version = check_version(text)?;
    let raw: RawManifest = toml::from_str(text)
        .map_err(|e| invalid(format_args!("manifest does not parse as TOML v1: {e}")))?;
    validate_raw(raw, version)
}

/// Chunk-file lengths, by chunk name, from statting the artifact directory.
/// Passed in so this crate stays I/O-free: it validates numbers, never files.
pub fn validate_chunks(manifest: &Manifest, chunks: &BTreeMap<String, u64>) -> Result<()> {
    for t in &manifest.tensors {
        // Version 2 places its bytes in safetensors shards, whose own headers
        // carry the ranges; `moxie-storage` checks those against the shard it
        // opened. This rule is version 1's, and applies to version 1's rows.
        let Some((chunk, offset, length, _)) = t.chunk_range() else {
            continue;
        };
        let len = chunks.get(chunk).ok_or_else(|| {
            invalid(format_args!(
                "tensor '{}' names chunk '{chunk}', which does not exist",
                t.role
            ))
        })?;
        let end = offset.checked_add(length).ok_or_else(|| {
            invalid(format_args!(
                "tensor '{}' offset {offset} + length {length} overflows",
                t.role
            ))
        })?;
        if end > *len {
            return Err(invalid(format_args!(
                "tensor '{}' describes bytes {offset}..{end} but chunk '{chunk}' holds {len} bytes: truncation",
                t.role
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

fn check_version(text: &str) -> Result<u32> {
    let v: VersionOnly = toml::from_str(text).map_err(|e| {
        // Malformed TOML, or no version at all: still a rejection, and the
        // message says which.
        let msg = e.to_string();
        if msg.contains("schema_version") {
            invalid(format_args!("manifest carries no schema_version: {e}"))
        } else {
            invalid(format_args!("manifest does not parse as TOML v1: {e}"))
        }
    })?;
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&v.schema_version) {
        return Err(invalid(format_args!(
            "schema_version is {}, this reader accepts only {SUPPORTED_SCHEMA_VERSIONS:?}: refused by version before any other field",
            v.schema_version
        )));
    }
    Ok(v.schema_version)
}

fn validate_raw(raw: RawManifest, version: u32) -> Result<Manifest> {
    // The version was already gated by `check_version` before deserializing
    // this schema; re-checking here would only repeat it.
    for f in &raw.required_features {
        if !KNOWN_REQUIRED_FEATURES.contains(&f.as_str()) {
            return Err(invalid(format_args!(
                "unknown required feature '{f}': refused by name; a future writer's artifact needs a future reader"
            )));
        }
    }
    let endianness = match raw.endianness.as_str() {
        "little" => Endianness::Little,
        other => {
            return Err(invalid(format_args!(
                "endianness '{other}' is refused rather than byte-swapped: nothing here has ever been tested on one"
            )));
        }
    };
    if raw.tensors.len() > MAX_TENSORS {
        return Err(invalid(format_args!(
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
        tensors.push(validate_tensor(t, version)?);
    }
    // Every role unique: a manifest whose second entry silently wins is refused.
    let mut roles = BTreeSet::new();
    for t in &tensors {
        if !roles.insert(t.role.clone()) {
            return Err(invalid(format_args!(
                "duplicate tensor role '{}': the second entry would silently win",
                t.role
            )));
        }
    }
    // Logical orders unique: they are an ordering, and duplicates are ambiguous.
    let mut orders = BTreeSet::new();
    for t in &tensors {
        if !orders.insert(t.logical_order) {
            return Err(invalid(format_args!(
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
                return Err(invalid_static(
                    "completeness is 'complete' but lists missing tensors",
                ));
            }
            Completeness::Complete
        }
        "partial" => {
            if raw.completeness.missing.is_empty() {
                return Err(invalid_static(
                    "completeness is 'partial' but names no missing tensor",
                ));
            }
            for m in &raw.completeness.missing {
                if m.is_empty() {
                    return Err(invalid_static("completeness lists an empty missing role"));
                }
            }
            Completeness::Partial {
                missing: raw.completeness.missing,
            }
        }
        other => {
            return Err(invalid(format_args!(
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
        return Err(invalid(format_args!(
            "{field} must be present and non-empty: absence is a rejection, not a default"
        )));
    }
    Ok(s)
}

fn validate_source(s: RawSource) -> Result<Source> {
    let mut files = Vec::with_capacity(s.files.len());
    if s.files.is_empty() {
        return Err(invalid_static(
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
        return Err(invalid(format_args!(
            "{field} must be 64 lowercase hex digits, got {s:?}"
        )));
    }
    Ok(s.to_ascii_lowercase())
}

fn to_u64(v: i64, field: &str) -> Result<u64> {
    u64::try_from(v).map_err(|_| {
        invalid(format_args!(
            "{field} is {v}: negative values are never valid here"
        ))
    })
}

fn validate_tensor(t: RawTensor, version: u32) -> Result<Tensor> {
    let role = nonempty(t.role, "tensors.role")?;
    if t.shape.is_empty() {
        return Err(invalid(format_args!(
            "tensor '{role}': shape must be non-empty"
        )));
    }
    let mut shape = Vec::with_capacity(t.shape.len());
    for (i, d) in t.shape.iter().enumerate() {
        let d = to_u64(*d, &format!("tensor '{role}' shape[{i}]"))?;
        if d == 0 {
            return Err(invalid(format_args!(
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
            return Err(invalid(format_args!(
                "tensor '{role}': precision '{other}' is not one of {KNOWN_PRECISIONS:?}"
            )));
        }
    };
    // Exactly one placement, and it must be the one this schema version
    // describes. A row carrying both, or neither, is a manifest whose reader
    // would have to guess which half to believe.
    let v1_fields = t.chunk.is_some()
        || t.offset.is_some()
        || t.length.is_some()
        || t.sha256.is_some()
        || t.alignment.is_some();
    let placement = match version {
        1 => {
            if !t.components.is_empty() {
                return Err(invalid(format_args!(
                    "tensor '{role}': a version 1 manifest places bytes in a chunk, not in \
                     components"
                )));
            }
            let chunk = t.chunk.ok_or_else(|| {
                invalid(format_args!("tensor '{role}': version 1 requires a chunk"))
            })?;
            validate_chunk_name(&chunk)
                .map_err(|d| invalid(format_args!("tensor '{role}': {d}")))?;
            let offset = to_u64(
                t.offset
                    .ok_or_else(|| invalid(format_args!("tensor '{role}': no offset")))?,
                &format!("tensor '{role}' offset"),
            )?;
            let length = to_u64(
                t.length
                    .ok_or_else(|| invalid(format_args!("tensor '{role}': no length")))?,
                &format!("tensor '{role}' length"),
            )?;
            if length == 0 {
                return Err(invalid(format_args!(
                    "tensor '{role}': length 0 describes no bytes"
                )));
            }
            let sha256 = validate_sha256(
                t.sha256
                    .ok_or_else(|| invalid(format_args!("tensor '{role}': no sha256")))?,
                &format!("tensor '{role}' sha256"),
            )?;
            let alignment = to_u64(
                t.alignment
                    .ok_or_else(|| invalid(format_args!("tensor '{role}': no alignment")))?,
                &format!("tensor '{role}' alignment"),
            )?;
            if !alignment.is_power_of_two() {
                return Err(invalid(format_args!(
                    "tensor '{role}': alignment {alignment} is not a power of two"
                )));
            }
            if offset % alignment != 0 {
                return Err(invalid(format_args!(
                    "tensor '{role}': offset {offset} is not a multiple of alignment {alignment}"
                )));
            }
            Placement::Chunk {
                chunk,
                offset,
                length,
                sha256,
                alignment,
            }
        }
        _ => {
            if v1_fields {
                return Err(invalid(format_args!(
                    "tensor '{role}': a version 2 manifest places bytes in safetensors \
                     components, so chunk/offset/length/sha256/alignment have no meaning here"
                )));
            }
            if t.components.is_empty() {
                return Err(invalid(format_args!(
                    "tensor '{role}': a version 2 tensor needs at least one component"
                )));
            }
            let mut components = Vec::with_capacity(t.components.len());
            let mut kinds = BTreeSet::new();
            for c in t.components {
                let kind = crate::canonical::ComponentKind::parse(&c.kind).ok_or_else(|| {
                    invalid(format_args!(
                        "tensor '{role}': component kind '{}' is outside \
                         {{weights, codes, scales, zero_points}}",
                        c.kind
                    ))
                })?;
                if !kinds.insert(kind) {
                    return Err(invalid(format_args!(
                        "tensor '{role}': component '{}' appears twice",
                        c.kind
                    )));
                }
                validate_chunk_name(&c.file).map_err(|d| {
                    invalid(format_args!("tensor '{role}' component '{}': {d}", c.kind))
                })?;
                let name = nonempty(c.name, "component.name")?;
                let sha256 = validate_sha256(
                    c.sha256,
                    &format!("tensor '{role}' component '{}' sha256", c.kind),
                )?;
                components.push(Component {
                    kind,
                    file: c.file,
                    name,
                    sha256,
                });
            }
            Placement::Components(components)
        }
    };
    let logical_order = to_u64(t.logical_order, &format!("tensor '{role}' logical_order"))?;

    // Checked shape product, before anything multiplies by it.
    let mut elements: u64 = 1;
    for d in &shape {
        elements = elements
            .checked_mul(*d)
            .ok_or_else(|| invalid(format_args!("tensor '{role}': shape product overflows u64")))?;
    }

    let affine = match precision {
        TensorPrecision::Bf16V1 => {
            if t.group_rule.is_some()
                || t.scale_dtype.is_some()
                || t.zero_point.is_some()
                || t.group_index.is_some()
            {
                return Err(invalid(format_args!(
                    "tensor '{role}': a bf16-v1 tensor must not carry affine fields (group_rule/scale_dtype/zero_point/group_index)"
                )));
            }
            let need = elements.checked_mul(2).ok_or_else(|| {
                invalid(format_args!("tensor '{role}': BF16 byte count overflows"))
            })?;
            if let Placement::Chunk { length, .. } = &placement
                && need != *length
            {
                return Err(invalid(format_args!(
                    "tensor '{role}': shape product {elements} * 2 BF16 bytes = {need}, but length is {length}: shape and byte count disagree"
                )));
            }
            None
        }
        TensorPrecision::AffineInt4V1 | TensorPrecision::AffineInt8V1 => {
            let group_rule = t.group_rule.ok_or_else(|| {
                invalid(format_args!(
                    "tensor '{role}': affine tensors must carry group_rule"
                ))
            })?;
            let group_rule = match group_rule.as_str() {
                "contiguous-32" => GroupRule::Contiguous32,
                "contiguous-128" => GroupRule::Contiguous128,
                "per-channel" => GroupRule::PerChannel,
                other => {
                    return Err(invalid(format_args!(
                        "tensor '{role}': group_rule '{other}' is outside the closed set {{contiguous-32, contiguous-128, per-channel}}; widening it is an ADR"
                    )));
                }
            };
            let scale_dtype = t.scale_dtype.ok_or_else(|| {
                invalid(format_args!(
                    "tensor '{role}': affine tensors must carry scale_dtype"
                ))
            })?;
            let scale_dtype = match scale_dtype.as_str() {
                "f16" => ScaleDtype::F16,
                "bf16" => ScaleDtype::Bf16,
                "f32" => ScaleDtype::F32,
                other => {
                    return Err(invalid(format_args!(
                        "tensor '{role}': scale_dtype '{other}' is outside {{f16, bf16, f32}}"
                    )));
                }
            };
            let zero_point = t.zero_point.ok_or_else(|| {
                invalid(format_args!(
                    "tensor '{role}': affine tensors must carry zero_point"
                ))
            })?;
            let zero_point = match zero_point.as_str() {
                "symmetric" => ZeroPointMode::Symmetric,
                "per-group" => ZeroPointMode::PerGroup,
                other => {
                    return Err(invalid(format_args!(
                        "tensor '{role}': zero_point '{other}' is outside {{symmetric, per-group}}"
                    )));
                }
            };
            let group_index = match t.group_index {
                None => None,
                Some(idx) => {
                    if shape.len() < 2 {
                        return Err(invalid(format_args!(
                            "tensor '{role}': group_index needs at least a 2-D shape to index input channels"
                        )));
                    }
                    let in_features = shape[shape.len() - 1];
                    if idx.len() as u64 != in_features {
                        return Err(invalid(format_args!(
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
                            return Err(invalid(format_args!(
                                "tensor '{role}': group_index[{k}] is group {g}, but there are only {groups} groups"
                            )));
                        }
                        let g = u32::try_from(g).map_err(|_| {
                            invalid(format_args!(
                                "tensor '{role}': group_index[{k}] is group {g}, above u32"
                            ))
                        })?;
                        out.push(g);
                    }
                    Some(out)
                }
            };
            // The affine payload layout is no longer open. [ADR 0023] fixes
            // it -- codes, then scales, then zero points, contiguous -- so the
            // length check this validator could not make in M1 is made below,
            // after the descriptor is built: the three section sizes are a
            // function of the descriptor, and a `length` that disagrees with
            // them describes a range no writer could have produced.
            //
            // [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md
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
            let in_features = usize::try_from(shape[shape.len() - 1]).map_err(|_| {
                invalid(format_args!(
                    "tensor '{role}': input dimension does not fit"
                ))
            })?;
            let mut out_features: usize = 1;
            for d in &shape[..shape.len() - 1] {
                let d = usize::try_from(*d).map_err(|_| {
                    invalid(format_args!(
                        "tensor '{role}': a leading dimension does not fit"
                    ))
                })?;
                out_features = out_features.checked_mul(d).ok_or_else(|| {
                    invalid(format_args!(
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
                invalid(format_args!(
                    "tensor '{role}': shared affine descriptor rejects it: {e}"
                ))
            })?;
            // ADR 0023's arithmetic, from the shared codec rather than a
            // second copy of it here.
            let section = match zero_point {
                ZeroPointMode::Symmetric => PayloadZeroPoints::Absent,
                ZeroPointMode::PerGroup => PayloadZeroPoints::PerGroup,
            };
            let need = crate::payload::length_of(&desc, section).map_err(|e| {
                invalid(format_args!(
                    "tensor '{role}': its payload length cannot be computed: {e}"
                ))
            })?;
            if let Placement::Chunk { length, .. } = &placement
                && need != *length
            {
                return Err(invalid(format_args!(
                    "tensor '{role}': a {} {out_features}x{in_features} tensor with {} \
                     zero point(s) occupies {need} canonical byte(s), but length is {length}: \
                     codes, scales and zero points are contiguous sections of one range (ADR 0023)",
                    precision.name(),
                    match section {
                        PayloadZeroPoints::Absent => "no",
                        PayloadZeroPoints::PerGroup => "per-group",
                    }
                )));
            }
            Some(AffineFields {
                group_rule,
                scale_dtype,
                zero_point,
                group_index,
            })
        }
    };

    // Version 2: the component set must be exactly the one the descriptor
    // implies. The shard headers say how big each one is -- `moxie-storage`
    // checks that against what it opened -- but whether a tensor has zero
    // points at all is this manifest's own claim, and it must agree with
    // itself.
    if let Placement::Components(components) = &placement {
        let expected: Vec<crate::canonical::ComponentKind> = match (&affine, precision) {
            (None, _) => vec![crate::canonical::ComponentKind::Weights],
            (Some(a), _) => {
                let mut kinds = vec![
                    crate::canonical::ComponentKind::Codes,
                    crate::canonical::ComponentKind::Scales,
                ];
                if a.zero_point == ZeroPointMode::PerGroup {
                    kinds.push(crate::canonical::ComponentKind::ZeroPoints);
                }
                kinds
            }
        };
        let mut found: Vec<crate::canonical::ComponentKind> =
            components.iter().map(|c| c.kind).collect();
        found.sort();
        let mut want = expected.clone();
        want.sort();
        if found != want {
            return Err(invalid(format_args!(
                "tensor '{role}' is {} with {} zero point(s) and carries {found:?}; it must carry \
                 exactly {want:?}",
                precision.name(),
                match &affine {
                    Some(a) if a.zero_point == ZeroPointMode::PerGroup => "per-group",
                    _ => "no",
                }
            )));
        }
    }

    Ok(Tensor {
        role,
        shape,
        precision,
        logical_order,
        affine,
        placement,
    })
}

/// No two tensors' byte ranges may intersect, per chunk -- including the
/// total-containment case a pairwise "starts inside" test misses.
fn check_overlap(tensors: &[Tensor]) -> Result<()> {
    let mut by_chunk: BTreeMap<&str, Vec<(&Tensor, u64, u64)>> = BTreeMap::new();
    for t in tensors {
        // A version 2 tensor's ranges are the shard header's, and
        // `moxie-storage` checks them against the shard it opened. This rule
        // is version 1's, where the manifest is the only thing that knows.
        if let Some((chunk, offset, length, _)) = t.chunk_range() {
            by_chunk.entry(chunk).or_default().push((t, offset, length));
        }
    }
    for (chunk, mut list) in by_chunk {
        list.sort_by_key(|(_, offset, length)| (*offset, *length));
        for w in list.windows(2) {
            let (a, a_offset, a_length) = w[0];
            let (b, b_offset, _) = w[1];
            let a_end = a_offset.checked_add(a_length).ok_or_else(|| {
                invalid(format_args!(
                    "tensor '{}' offset + length overflows",
                    a.role
                ))
            })?;
            // Sorted by offset, so any intersection means b starts inside
            // a -- including b wholly contained in a.
            if b_offset < a_end {
                return Err(invalid(format_args!(
                    "tensors '{}' ({a_offset}..{a_end}) and '{}' ({b_offset}..) overlap in chunk '{chunk}', including containment",
                    a.role, b.role
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
        .ok_or_else(|| invalid_static("architecture metadata node count overflows"))?;
    if *nodes > MAX_ARCH_NODES {
        return Err(invalid(format_args!(
            "architecture metadata has more than {MAX_ARCH_NODES} nodes: the walk stops at the limit"
        )));
    }
    if depth > MAX_ARCH_DEPTH {
        return Err(invalid(format_args!(
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
        // Placement participates in identity, and the two shapes are tagged
        // differently so a v1 range and a v2 component set can never hash to
        // the same artifact.
        match &t.placement {
            Placement::Chunk {
                chunk,
                offset,
                length,
                sha256,
                alignment,
            } => {
                w.tag("chunk-placement");
                w.str(chunk);
                w.u64(*offset);
                w.u64(*length);
                w.str(sha256);
                w.u64(*alignment);
            }
            Placement::Components(components) => {
                w.tag("component-placement");
                w.u64(components.len() as u64);
                for c in components {
                    w.str(c.kind.name());
                    w.str(&c.file);
                    w.str(&c.name);
                    w.str(&c.sha256);
                }
            }
        }
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

// --- the writer half ---------------------------------------------------------

/// Encode a validated manifest as `manifest.toml`.
///
/// The inverse of [`parse`], and the only place canonical manifest text is
/// produced. Three properties make it safe to publish what this returns:
///
/// * **It goes through the same parser on the way out.** `encode` re-parses its
///   own output and compares the artifact identity before returning it, so a
///   manifest this function emits is one this reader accepts, by construction
///   rather than by test coverage. A writer that could emit text its own reader
///   rejects is a publication step that can fail after the bytes are durable.
/// * **It escapes nothing by hand.** The document is built as a `toml::Value`
///   and serialized by the `toml` crate, which ADR 0005 already admitted for
///   exactly this file. A role containing a quote, a newline or a non-ASCII
///   character is the serializer's problem, not a hand-written quoting rule's --
///   ADR 0004 is four reviews of what hand-written text handling costs here.
/// * **It is deterministic.** Table keys are serialized in sorted order and
///   tensors in the order the manifest holds them, so two encodes of one
///   manifest are byte-identical and a republish of unchanged content changes
///   no byte.
///
/// It does **not** decide content: completeness, exclusions, provenance and
/// identity fields are whatever the caller put in the `Manifest`. Deciding
/// those is the repacker's job and is where "a selected module is not a
/// complete model" is enforced.
pub fn encode(manifest: &Manifest) -> Result<String> {
    let mut doc = toml::map::Map::new();
    doc.insert(
        "schema_version".into(),
        toml::Value::Integer(i64::from(SCHEMA_VERSION)),
    );
    doc.insert(
        "required_features".into(),
        toml::Value::Array(
            manifest
                .required_features
                .iter()
                .map(|f| toml::Value::String(f.clone()))
                .collect(),
        ),
    );
    doc.insert(
        "endianness".into(),
        toml::Value::String(
            match manifest.endianness {
                Endianness::Little => "little",
            }
            .into(),
        ),
    );

    let mut source = toml::map::Map::new();
    source.insert(
        "model".into(),
        toml::Value::String(manifest.source.model.clone()),
    );
    source.insert(
        "revision".into(),
        toml::Value::String(manifest.source.revision.clone()),
    );
    source.insert(
        "license".into(),
        toml::Value::String(manifest.source.license.clone()),
    );
    source.insert(
        "files".into(),
        toml::Value::Array(
            manifest
                .source
                .files
                .iter()
                .map(|f| {
                    let mut t = toml::map::Map::new();
                    t.insert("path".into(), toml::Value::String(f.path.clone()));
                    t.insert("sha256".into(), toml::Value::String(f.sha256.clone()));
                    toml::Value::Table(t)
                })
                .collect(),
        ),
    );
    doc.insert("source".into(), toml::Value::Table(source));
    doc.insert("tokenizer".into(), identity_table(&manifest.tokenizer));
    doc.insert("template".into(), identity_table(&manifest.template));

    let mut arch = toml::map::Map::new();
    arch.insert(
        "name".into(),
        toml::Value::String(manifest.architecture.name.clone()),
    );
    arch.insert(
        "version".into(),
        toml::Value::String(manifest.architecture.version.clone()),
    );
    // The opaque tree, re-emitted exactly as it was parsed. This crate hashes
    // it into artifact identity and never reads a field of it, so it is copied
    // rather than interpreted here too.
    arch.insert("metadata".into(), manifest.architecture.metadata.0.clone());
    doc.insert("architecture".into(), toml::Value::Table(arch));

    let mut prov = toml::map::Map::new();
    prov.insert(
        "scale_convention".into(),
        toml::Value::String(manifest.provenance.scale_convention.clone()),
    );
    prov.insert(
        "quantizer".into(),
        toml::Value::String(manifest.provenance.quantizer.clone()),
    );
    prov.insert(
        "calibration".into(),
        toml::Value::String(manifest.provenance.calibration.clone()),
    );
    doc.insert("provenance".into(), toml::Value::Table(prov));

    let mut tensors = Vec::new();
    for t in &manifest.tensors {
        tensors.push(tensor_table(t)?);
    }
    doc.insert("tensors".into(), toml::Value::Array(tensors));

    doc.insert(
        "excluded".into(),
        toml::Value::Array(
            manifest
                .excluded
                .iter()
                .map(|e| {
                    let mut t = toml::map::Map::new();
                    t.insert("role".into(), toml::Value::String(e.role.clone()));
                    t.insert("reason".into(), toml::Value::String(e.reason.clone()));
                    toml::Value::Table(t)
                })
                .collect(),
        ),
    );

    let mut completeness = toml::map::Map::new();
    match &manifest.completeness {
        Completeness::Complete => {
            completeness.insert("status".into(), toml::Value::String("complete".into()));
            completeness.insert("missing".into(), toml::Value::Array(Vec::new()));
        }
        Completeness::Partial { missing } => {
            completeness.insert("status".into(), toml::Value::String("partial".into()));
            completeness.insert(
                "missing".into(),
                toml::Value::Array(
                    missing
                        .iter()
                        .map(|m| toml::Value::String(m.clone()))
                        .collect(),
                ),
            );
        }
    }
    doc.insert("completeness".into(), toml::Value::Table(completeness));

    let text = toml::to_string(&toml::Value::Table(doc))
        .map_err(|e| invalid(format_args!("manifest does not serialize: {e}")))?;

    // Out through the reader. A publication that emits text its own validator
    // refuses would fail at the worst possible moment -- after the payload is
    // durable -- and identity is compared as well as validity, so a field lost
    // in encoding is caught rather than published.
    let reparsed = parse(&text).map_err(|e| {
        invalid(format_args!(
            "the encoded manifest does not parse back: {e}"
        ))
    })?;
    if artifact_identity(&reparsed) != artifact_identity(manifest) {
        return Err(invalid_static(
            "the encoded manifest parses back to a different artifact identity",
        ));
    }
    Ok(text)
}

fn identity_table(id: &Identity) -> toml::Value {
    let mut t = toml::map::Map::new();
    t.insert("name".into(), toml::Value::String(id.name.clone()));
    t.insert("version".into(), toml::Value::String(id.version.clone()));
    t.insert("digest".into(), toml::Value::String(id.digest.clone()));
    toml::Value::Table(t)
}

fn tensor_table(t: &Tensor) -> Result<toml::Value> {
    let to_i64 = |v: u64, field: &str| -> Result<toml::Value> {
        i64::try_from(v)
            .map(toml::Value::Integer)
            .map_err(|_| invalid(format_args!("tensor '{}': {field} {v} exceeds i64", t.role)))
    };
    let mut e = toml::map::Map::new();
    e.insert("role".into(), toml::Value::String(t.role.clone()));
    let mut shape = Vec::with_capacity(t.shape.len());
    for d in &t.shape {
        shape.push(to_i64(*d, "shape dimension")?);
    }
    e.insert("shape".into(), toml::Value::Array(shape));
    e.insert(
        "precision".into(),
        toml::Value::String(t.precision.name().into()),
    );
    match &t.placement {
        Placement::Chunk {
            chunk,
            offset,
            length,
            sha256,
            alignment,
        } => {
            e.insert("chunk".into(), toml::Value::String(chunk.clone()));
            e.insert("offset".into(), to_i64(*offset, "offset")?);
            e.insert("length".into(), to_i64(*length, "length")?);
            e.insert("sha256".into(), toml::Value::String(sha256.clone()));
            e.insert("alignment".into(), to_i64(*alignment, "alignment")?);
        }
        Placement::Components(components) => {
            let mut rows = Vec::with_capacity(components.len());
            for c in components {
                let mut row = toml::map::Map::new();
                row.insert("kind".into(), toml::Value::String(c.kind.name().into()));
                row.insert("file".into(), toml::Value::String(c.file.clone()));
                row.insert("name".into(), toml::Value::String(c.name.clone()));
                row.insert("sha256".into(), toml::Value::String(c.sha256.clone()));
                rows.push(toml::Value::Table(row));
            }
            e.insert("components".into(), toml::Value::Array(rows));
        }
    }
    e.insert(
        "logical_order".into(),
        to_i64(t.logical_order, "logical_order")?,
    );
    if let Some(a) = &t.affine {
        e.insert(
            "group_rule".into(),
            toml::Value::String(
                match a.group_rule {
                    GroupRule::Contiguous32 => "contiguous-32",
                    GroupRule::Contiguous128 => "contiguous-128",
                    GroupRule::PerChannel => "per-channel",
                }
                .into(),
            ),
        );
        e.insert(
            "scale_dtype".into(),
            toml::Value::String(
                match a.scale_dtype {
                    ScaleDtype::F16 => "f16",
                    ScaleDtype::Bf16 => "bf16",
                    ScaleDtype::F32 => "f32",
                }
                .into(),
            ),
        );
        e.insert(
            "zero_point".into(),
            toml::Value::String(
                match a.zero_point {
                    ZeroPointMode::Symmetric => "symmetric",
                    ZeroPointMode::PerGroup => "per-group",
                }
                .into(),
            ),
        );
        if let Some(map) = &a.group_index {
            let mut idx = Vec::with_capacity(map.len());
            for g in map {
                idx.push(toml::Value::Integer(i64::from(*g)));
            }
            e.insert("group_index".into(), toml::Value::Array(idx));
        }
    }
    Ok(toml::Value::Table(e))
}
