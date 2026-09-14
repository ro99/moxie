//! The repack selection: what a user asked to be repacked, as text.
//!
//! A repack needs more than a command line can carry without becoming its own
//! language: per-tensor roles, source names, packing parameters and the
//! identity fields manifest v1 requires. So the selection is a small TOML
//! document the user writes and the repacker validates, and this module is its
//! schema -- I/O-free, like [`crate::manifest`] and [`crate::journal`]. The
//! program reads the file; this parses the text.
//!
//! It is a **selection**, not a catalog and not a discovery mechanism. Nothing
//! here scans a directory, expands a pattern or infers a role from a name:
//! every tensor published is one the user wrote down. That is what keeps
//! "no agent-initiated bulk conversion" (ADR 0020) a property of the tool
//! rather than a promise about how it is invoked.
//!
//! ```toml
//! version = 1
//!
//! [source]
//! model = "Laguna-S-2.1-AWQ-INT4"
//! revision = "bc59f497520b23759ce61cc5164ca28bcc4f53bc"
//! license = "apache-2.0"
//!
//! [tokenizer]
//! name = "none"
//! version = "not-selected"
//! digest = "<64 hex, or the absent-identity digest>"
//!
//! [template]
//! name = "none"
//! version = "not-selected"
//! digest = "<64 hex>"
//!
//! [architecture]
//! name = "laguna"
//! version = "1"
//! [architecture.metadata]
//! note = "opaque to every shared crate"
//!
//! [provenance]
//! scale_convention = "affine-v1"
//! quantizer = "none"
//! calibration = "none"
//!
//! [completeness]
//! status = "partial"
//! missing = ["every tensor of this revision except the selection below"]
//!
//! [[tensor]]
//! role = "model.layers.1.mlp.experts.0.down_proj.weight"
//! kind = "pack-quantized"
//! module = "model.layers.1.mlp.experts.0.down_proj"
//! width = "int4"
//! group = 32
//! zero_points = "packed-along-output"
//! [tensor.files]
//! weight_packed = "model-00001-of-00015.safetensors"
//! weight_scale = "model-00001-of-00015.safetensors"
//! weight_shape = "model-00001-of-00015.safetensors"
//! weight_zero_point = "model-00001-of-00015.safetensors"
//! ```

use std::collections::{BTreeMap, BTreeSet};

use moxie_types::{Error, Result};
use serde::Deserialize;

use crate::affine::IntWidth;
use crate::compressed_tensors::{Granularity, PackQuantizedSpec, ZeroPointSource};

/// Selections this parser accepts.
pub const SELECTION_VERSION: u32 = 1;

/// Text above this is refused before it is parsed.
pub const MAX_SELECTION_BYTES: usize = 4 * 1024 * 1024;

/// Selected tensors above this are refused before validation.
pub const MAX_SELECTED_TENSORS: usize = 65_536;

fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed repack selection (detail unavailable: out of memory)",
        detail,
    )
}

fn invalid_static(detail: &'static str) -> Error {
    crate::invalid_static(detail)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSelection {
    // Pinned by the version gate before this schema deserializes, and required
    // here so `deny_unknown_fields` keeps recognizing it.
    #[allow(dead_code)]
    version: u32,
    source: RawSource,
    tokenizer: RawIdentity,
    template: RawIdentity,
    architecture: RawArchitecture,
    provenance: RawProvenance,
    completeness: RawCompleteness,
    // Defaulted so that "no tensor selected" is this module's refusal, in its
    // own words, rather than a deserializer's missing-field message.
    #[serde(default)]
    tensor: Vec<RawTensor>,
    #[serde(default)]
    excluded: Vec<RawExcluded>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSource {
    model: String,
    revision: String,
    license: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIdentity {
    name: String,
    version: String,
    digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArchitecture {
    name: String,
    version: String,
    metadata: toml::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProvenance {
    scale_convention: String,
    quantizer: String,
    calibration: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompleteness {
    status: String,
    #[serde(default)]
    missing: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExcluded {
    role: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    role: String,
    kind: String,
    #[serde(default)]
    alignment: Option<i64>,
    // BF16 passthrough.
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    file: Option<String>,
    // pack-quantized.
    #[serde(default)]
    module: Option<String>,
    #[serde(default)]
    width: Option<String>,
    #[serde(default)]
    group: Option<toml::Value>,
    #[serde(default)]
    zero_points: Option<String>,
    #[serde(default)]
    files: Option<BTreeMap<String, String>>,
}

/// One selected tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedTensor {
    /// The canonical role this becomes.
    pub role: String,
    pub alignment: u64,
    pub kind: SelectionKind,
}

/// What a selected tensor is read from, and how it is interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionKind {
    /// A BF16 tensor copied through unchanged: ADR 0018's bit-identical repack
    /// in its simplest form.
    Bf16 {
        /// The source tensor's name in its shard.
        name: String,
        /// The shard file, relative to the source root.
        file: String,
    },
    /// A compressed-tensors `pack-quantized` module.
    PackQuantized {
        module: String,
        spec: PackQuantizedSpec,
        /// Each of the module's tensors and the shard it lives in. Task 0024
        /// measured that companions need not share a shard, so this is a map
        /// rather than one file name.
        files: BTreeMap<String, String>,
    },
}

/// What the whole selection says.
#[derive(Debug, Clone)]
pub struct Selection {
    pub model: String,
    pub revision: String,
    pub license: String,
    pub tokenizer: Identity,
    pub template: Identity,
    pub architecture_name: String,
    pub architecture_version: String,
    pub architecture_metadata: toml::Value,
    pub scale_convention: String,
    pub quantizer: String,
    pub calibration: String,
    pub completeness: Completeness,
    pub excluded: Vec<(String, String)>,
    pub tensors: Vec<SelectedTensor>,
    /// Bytes of the document this was parsed from.
    ///
    /// Kept because a caller admitting memory for the parsed form needs the
    /// size of what produced it: every name in those structures came from this
    /// text, and the cap on the text is a cap on all of them together.
    pub source_bytes: u64,
}

impl Selection {
    /// Bytes of the document this selection was parsed from.
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub name: String,
    pub version: String,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completeness {
    /// The selection is claimed to be the whole model.
    Complete,
    /// The usual case for a selection, with what it does not contain.
    Partial { missing: Vec<String> },
}

/// The four tensors a `pack-quantized` module is serialized as.
pub const PACK_QUANTIZED_SUFFIXES: [&str; 4] = [
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
];

impl Selection {
    /// Every source file this selection names, deduplicated and sorted.
    pub fn files(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        for t in &self.tensors {
            match &t.kind {
                SelectionKind::Bf16 { file, .. } => {
                    set.insert(file.clone());
                }
                SelectionKind::PackQuantized { files, .. } => {
                    for f in files.values() {
                        set.insert(f.clone());
                    }
                }
            }
        }
        set.into_iter().collect()
    }
}

/// Parse and validate a selection.
pub fn parse(text: &str) -> Result<Selection> {
    if text.len() > MAX_SELECTION_BYTES {
        return Err(invalid(format_args!(
            "selection is {} byte(s), above the {MAX_SELECTION_BYTES} cap checked before parsing",
            text.len()
        )));
    }
    // Version first, then the version's schema: a future selection is refused
    // by version even when its other fields have changed.
    #[derive(Deserialize)]
    struct VersionOnly {
        version: u32,
    }
    let v: VersionOnly = toml::from_str(text)
        .map_err(|e| invalid(format_args!("selection carries no version: {e}")))?;
    if v.version != SELECTION_VERSION {
        return Err(invalid(format_args!(
            "selection is version {}, this repacker accepts only version {SELECTION_VERSION}",
            v.version
        )));
    }
    let raw: RawSelection =
        toml::from_str(text).map_err(|e| invalid(format_args!("selection does not parse: {e}")))?;

    if raw.tensor.is_empty() {
        return Err(invalid_static(
            "a selection with no [[tensor]] selects nothing; publishing nothing is not a repack",
        ));
    }
    if raw.tensor.len() > MAX_SELECTED_TENSORS {
        return Err(invalid(format_args!(
            "selection names {} tensors, above the {MAX_SELECTED_TENSORS} cap checked before \
             validation",
            raw.tensor.len()
        )));
    }
    crate::manifest::check_arch_limits(&raw.architecture.metadata)?;

    let mut tensors = Vec::with_capacity(raw.tensor.len());
    let mut roles = BTreeSet::new();
    for t in raw.tensor {
        let role = nonempty(t.role, "tensor.role")?;
        if !roles.insert(role.clone()) {
            return Err(invalid(format_args!(
                "tensor role '{role}' is selected twice: the second would silently win"
            )));
        }
        let alignment = match t.alignment {
            None => 16,
            Some(a) => {
                let a = u64::try_from(a).map_err(|_| {
                    invalid(format_args!("tensor '{role}': alignment {a} is negative"))
                })?;
                if !a.is_power_of_two() {
                    return Err(invalid(format_args!(
                        "tensor '{role}': alignment {a} is not a power of two"
                    )));
                }
                a
            }
        };
        let kind = match t.kind.as_str() {
            "bf16" => {
                let name = nonempty(
                    t.name.ok_or_else(|| {
                        invalid(format_args!(
                            "tensor '{role}': a bf16 selection needs `name`"
                        ))
                    })?,
                    "tensor.name",
                )?;
                let file = source_file(
                    t.file.ok_or_else(|| {
                        invalid(format_args!(
                            "tensor '{role}': a bf16 selection needs `file`"
                        ))
                    })?,
                    &role,
                )?;
                for (field, present) in [
                    ("module", t.module.is_some()),
                    ("width", t.width.is_some()),
                    ("group", t.group.is_some()),
                    ("zero_points", t.zero_points.is_some()),
                    ("files", t.files.is_some()),
                ] {
                    if present {
                        return Err(invalid(format_args!(
                            "tensor '{role}': a bf16 selection must not carry `{field}`"
                        )));
                    }
                }
                SelectionKind::Bf16 { name, file }
            }
            "pack-quantized" => {
                let module = nonempty(
                    t.module.ok_or_else(|| {
                        invalid(format_args!(
                            "tensor '{role}': a pack-quantized selection needs `module`"
                        ))
                    })?,
                    "tensor.module",
                )?;
                for (field, present) in [("name", t.name.is_some()), ("file", t.file.is_some())] {
                    if present {
                        return Err(invalid(format_args!(
                            "tensor '{role}': a pack-quantized selection must not carry `{field}`; \
                             its four tensors are named by `files`"
                        )));
                    }
                }
                let width = match t.width.as_deref() {
                    Some("int4") => IntWidth::Int4,
                    Some("int8") => IntWidth::Int8,
                    other => {
                        return Err(invalid(format_args!(
                            "tensor '{role}': width {other:?} is outside {{int4, int8}}"
                        )));
                    }
                };
                let granularity = match &t.group {
                    Some(toml::Value::Integer(n)) => {
                        let size = u32::try_from(*n).map_err(|_| {
                            invalid(format_args!(
                                "tensor '{role}': group {n} is not a group size"
                            ))
                        })?;
                        if !crate::affine::ALLOWED_GROUP_SIZES.contains(&size) {
                            return Err(invalid(format_args!(
                                "tensor '{role}': group {size} is outside the closed set {:?}; \
                                 widening it is an ADR, not a selection",
                                crate::affine::ALLOWED_GROUP_SIZES
                            )));
                        }
                        Granularity::Group { size }
                    }
                    Some(toml::Value::String(s)) if s == "channel" => Granularity::Channel,
                    other => {
                        return Err(invalid(format_args!(
                            "tensor '{role}': group must be 32, 128 or \"channel\", got {other:?}"
                        )));
                    }
                };
                let zero_points = match t.zero_points.as_deref() {
                    Some("symmetric") => ZeroPointSource::Symmetric,
                    Some("packed-along-output") => ZeroPointSource::PackedAlongOutput,
                    other => {
                        return Err(invalid(format_args!(
                            "tensor '{role}': zero_points {other:?} is outside \
                             {{symmetric, packed-along-output}}"
                        )));
                    }
                };
                let declared = t.files.ok_or_else(|| {
                    invalid(format_args!(
                        "tensor '{role}': a pack-quantized selection needs [tensor.files]"
                    ))
                })?;
                let mut files = BTreeMap::new();
                let wanted: &[&str] = match zero_points {
                    ZeroPointSource::Symmetric => &PACK_QUANTIZED_SUFFIXES[..3],
                    ZeroPointSource::PackedAlongOutput => &PACK_QUANTIZED_SUFFIXES,
                };
                for suffix in wanted {
                    let file = declared.get(*suffix).cloned().ok_or_else(|| {
                        invalid(format_args!(
                            "tensor '{role}': [tensor.files] does not say which shard holds \
                             {suffix}"
                        ))
                    })?;
                    files.insert((*suffix).to_string(), source_file(file, &role)?);
                }
                for named in declared.keys() {
                    if !wanted.contains(&named.as_str()) {
                        return Err(invalid(format_args!(
                            "tensor '{role}': [tensor.files] names '{named}', which this \
                             serialization does not have -- a symmetric module has no \
                             weight_zero_point, and a name outside the four is not a tensor"
                        )));
                    }
                }
                SelectionKind::PackQuantized {
                    module,
                    spec: PackQuantizedSpec {
                        width,
                        granularity,
                        zero_points,
                    },
                    files,
                }
            }
            other => {
                return Err(invalid(format_args!(
                    "tensor '{role}': kind '{other}' is outside {{bf16, pack-quantized}}. \
                     AutoGPTQ/AutoRound packing and activation-order maps are M3 item 2's \
                     remainder and are refused by name rather than guessed at"
                )));
            }
        };
        tensors.push(SelectedTensor {
            role,
            alignment,
            kind,
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
                    "completeness is 'partial' but names nothing missing: a partial artifact has \
                     to say what it does not contain",
                ));
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

    let mut excluded = Vec::with_capacity(raw.excluded.len());
    for e in raw.excluded {
        excluded.push((
            nonempty(e.role, "excluded.role")?,
            nonempty(e.reason, "excluded.reason")?,
        ));
    }

    Ok(Selection {
        source_bytes: text.len() as u64,
        model: nonempty(raw.source.model, "source.model")?,
        revision: nonempty(raw.source.revision, "source.revision")?,
        license: nonempty(raw.source.license, "source.license")?,
        tokenizer: identity(raw.tokenizer, "tokenizer")?,
        template: identity(raw.template, "template")?,
        architecture_name: nonempty(raw.architecture.name, "architecture.name")?,
        architecture_version: nonempty(raw.architecture.version, "architecture.version")?,
        architecture_metadata: raw.architecture.metadata,
        scale_convention: nonempty(
            raw.provenance.scale_convention,
            "provenance.scale_convention",
        )?,
        quantizer: nonempty(raw.provenance.quantizer, "provenance.quantizer")?,
        calibration: nonempty(raw.provenance.calibration, "provenance.calibration")?,
        completeness,
        excluded,
        tensors,
    })
}

fn identity(raw: RawIdentity, what: &str) -> Result<Identity> {
    let digest = raw.digest.to_ascii_lowercase();
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format_args!(
            "{what}.digest must be 64 hex digits, got {:?}",
            raw.digest
        )));
    }
    Ok(Identity {
        name: nonempty(raw.name, what)?,
        version: nonempty(raw.version, what)?,
        digest,
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

/// A source file name, confined to the source root **on the string**, before
/// any path is joined.
///
/// The same rule [`crate::manifest::validate_chunk_name`] applies to chunk
/// references, for the same reason: a selection is user-supplied text, and a
/// traversal in it would read a file outside the root the user named. The
/// filesystem check that catches symlinks is the program's job; this one
/// catches the spelling without touching a disk.
fn source_file(name: String, role: &str) -> Result<String> {
    crate::manifest::validate_chunk_name(&name).map_err(|d| {
        invalid(format_args!(
            "tensor '{role}': source file reference rejected on the string, before joining: {d}"
        ))
    })?;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> String {
        r#"version = 1

[source]
model = "m"
revision = "r"
license = "l"

[tokenizer]
name = "none"
version = "not-selected"
digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[template]
name = "none"
version = "not-selected"
digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[architecture]
name = "a"
version = "1"
[architecture.metadata]
note = "opaque"

[provenance]
scale_convention = "affine-v1"
quantizer = "none"
calibration = "none"

[completeness]
status = "partial"
missing = ["everything else"]
"#
        .to_string()
    }

    fn with(tensors: &str) -> String {
        format!("{}{tensors}", header())
    }

    const PQ: &str = r#"
[[tensor]]
role = "w"
kind = "pack-quantized"
module = "m.0"
width = "int4"
group = 32
zero_points = "packed-along-output"
[tensor.files]
weight_packed = "s1.safetensors"
weight_scale = "s2.safetensors"
weight_shape = "s1.safetensors"
weight_zero_point = "s1.safetensors"
"#;

    #[test]
    fn a_pack_quantized_selection_parses_with_its_companions_in_different_shards() {
        let s = parse(&with(PQ)).expect("it parses");
        assert_eq!(s.tensors.len(), 1);
        assert_eq!(s.tensors[0].alignment, 16);
        match &s.tensors[0].kind {
            SelectionKind::PackQuantized {
                module,
                spec,
                files,
            } => {
                assert_eq!(module, "m.0");
                assert_eq!(spec.width, IntWidth::Int4);
                assert_eq!(spec.granularity, Granularity::Group { size: 32 });
                assert_eq!(spec.zero_points, ZeroPointSource::PackedAlongOutput);
                assert_eq!(files["weight_scale"], "s2.safetensors");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(s.files(), ["s1.safetensors", "s2.safetensors"]);
    }

    #[test]
    fn a_bf16_selection_parses_and_refuses_quantized_fields() {
        let ok = with(
            "\n[[tensor]]\nrole = \"n\"\nkind = \"bf16\"\nname = \"x.weight\"\nfile = \"s1.safetensors\"\n",
        );
        let s = parse(&ok).expect("it parses");
        assert_eq!(
            s.tensors[0].kind,
            SelectionKind::Bf16 {
                name: "x.weight".into(),
                file: "s1.safetensors".into()
            }
        );
        let bad = ok.replace(
            "file = \"s1.safetensors\"",
            "file = \"s1.safetensors\"\nwidth = \"int4\"",
        );
        let e = parse(&bad).unwrap_err();
        assert!(e.to_string().contains("must not carry `width`"), "{e}");
    }

    #[test]
    fn every_refusal_names_what_it_refused() {
        for (text, needle) in [
            (
                with(PQ).replace("version = 1", "version = 2"),
                "only version 1",
            ),
            (header(), "selects nothing"),
            (with(&PQ.replace("group = 32", "group = 64")), "closed set"),
            (
                with(&PQ.replace("group = 32", "group = \"per-tensor\"")),
                "must be 32, 128",
            ),
            (
                with(&PQ.replace("width = \"int4\"", "width = \"int2\"")),
                "outside {int4, int8}",
            ),
            (
                with(&PQ.replace(
                    "zero_points = \"packed-along-output\"",
                    "zero_points = \"guess\"",
                )),
                "outside {symmetric, packed-along-output}",
            ),
            (
                with(&PQ.replace("kind = \"pack-quantized\"", "kind = \"auto-gptq\"")),
                "refused by name rather than guessed at",
            ),
            (
                with(&PQ.replace("weight_scale = \"s2.safetensors\"\n", "")),
                "which shard holds weight_scale",
            ),
            (
                with(&PQ.replace(
                    "weight_shape = \"s1.safetensors\"",
                    "weight_shape = \"s1.safetensors\"\nweight_other = \"s1.safetensors\"",
                )),
                "which this serialization does not have",
            ),
            (
                with(&PQ.replace("\"s1.safetensors\"", "\"../elsewhere/s1.safetensors\"")),
                "rejected on the string",
            ),
            (
                with(&PQ.replace("\"s1.safetensors\"", "\"/etc/passwd\"")),
                "rejected on the string",
            ),
            (with(&format!("{PQ}{PQ}")), "selected twice"),
            (
                with(PQ).replace("missing = [\"everything else\"]", "missing = []"),
                "has to say what it does not contain",
            ),
            (
                with(PQ).replace("status = \"partial\"", "status = \"mostly\""),
                "expected 'complete' or 'partial'",
            ),
            (
                with(PQ).replace("digest = \"e3b0", "digest = \"zzzz"),
                "64 hex digits",
            ),
            (
                with(&PQ.replace("role = \"w\"", "role = \"\"")),
                "must be present and non-empty",
            ),
        ] {
            let e = parse(&text).unwrap_err();
            assert!(e.to_string().contains(needle), "{needle:?} not in {e}");
        }
    }

    /// A symmetric module has three tensors, not four, and naming a fourth is
    /// a disagreement rather than a spare file.
    #[test]
    fn a_symmetric_module_has_no_zero_point_file() {
        let sym = PQ
            .replace(
                "zero_points = \"packed-along-output\"",
                "zero_points = \"symmetric\"",
            )
            .replace("weight_zero_point = \"s1.safetensors\"\n", "");
        let s = parse(&with(&sym)).expect("it parses");
        match &s.tensors[0].kind {
            SelectionKind::PackQuantized { files, .. } => assert_eq!(files.len(), 3),
            other => panic!("{other:?}"),
        }
        let e = parse(&with(&sym.replace(
            "weight_shape = \"s1.safetensors\"",
            "weight_shape = \"s1.safetensors\"\nweight_zero_point = \"s1.safetensors\"",
        )))
        .unwrap_err();
        assert!(e.to_string().contains("does not have"), "{e}");
    }
}
