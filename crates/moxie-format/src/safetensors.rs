//! Safetensors container: a bounded, strict header reader.
//!
//! The layout is a `u64` little-endian header length, that many bytes of JSON,
//! then the tensor payload. Offsets inside the header are relative to the
//! payload start, not to the file.
//!
//! **This parses untrusted input.** The bytes come from a downloaded checkpoint,
//! so nothing is inferred from a tensor's name and every field is read and
//! checked: document 03 requires readers to "validate all integer arithmetic and
//! reject overlap, truncation, NaN scales, incompatible dimensions, and unknown
//! required features", with "no remote checkpoint code execution; parse
//! supported metadata safely".
//!
//! Only the JSON parse is borrowed ([ADR 0015]); every structural rule, bound
//! and arithmetic check below is this crate's.
//!
//! This module never opens a file. `moxie-format` is forbidden to touch the
//! filesystem -- `arch-check` enforces it -- so a caller supplies bytes and
//! `moxie-storage` is what reads them.
//!
//! [ADR 0015]: ../../../docs/decisions/adr/0015-serde-json-for-safetensors-headers.md

use std::collections::BTreeMap;

use moxie_types::{Error, HostTier, Result, Tier};

/// The largest header this reader will parse.
///
/// Read and checked **before** the parser sees anything, so a hostile length
/// cannot cause a large allocation ahead of validation. The Gemma 4 artifact's
/// shards, the largest inspected so far, have headers well under 1 MiB; 64 MiB
/// is far above any plausible real index and far below a denial of service.
pub const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Most dimensions a tensor may declare.
///
/// A structural limit, not a stylistic one. Each dimension costs two serialized
/// bytes (`"1,"`) and twenty-four bytes of peak heap -- a `u64` in a vector that
/// holds its old allocation alongside the new one while it grows -- so an
/// unbounded rank buys twelve bytes of memory per byte of header, which is more
/// than any other construct and more than a caller's budget can be derived
/// against. Independent review reached 1.7 MB of peak from a 131 KB header with
/// a single 65,537-dimension tensor.
///
/// Eight is well above any real artifact: the Gemma 4 shards' tensors are one-
/// and two-dimensional, and a rank above four is already unusual. Refusing the
/// rest is honest -- a 65,537-dimension tensor is not a tensor -- and it is
/// refused **while parsing**, before the dimensions are allocated.
pub const MAX_RANK: usize = 8;

/// Longest tensor name, in bytes. The artifact's longest is about seventy.
pub const MAX_NAME_BYTES: usize = 1024;

/// Most tensors one header may declare. A minimal entry is about 55 serialized
/// bytes, so [`MAX_HEADER_BYTES`] already implies roughly this; stating it
/// makes the bound a contract rather than an arithmetic accident.
pub const MAX_TENSORS: usize = 1 << 20;

/// Most `__metadata__` entries one header may declare.
pub const MAX_METADATA_ENTRIES: usize = 1 << 16;

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

fn capacity(requested: u64) -> Error {
    Error::CapacityExceeded {
        tier: Some(Tier::Host(HostTier::Pageable)),
        requested_bytes: requested,
        available_bytes: 0,
    }
}

/// The safetensors element types this reader accepts.
///
/// Unknown dtypes are a typed error rather than a skipped tensor: a reader that
/// ignored one would silently under-report an artifact's contents, and document
/// 03 requires that "unknown required semantics fail before execution".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    Bf16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl Dtype {
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::U16 => "U16",
            Dtype::I16 => "I16",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
            Dtype::U32 => "U32",
            Dtype::I32 => "I32",
            Dtype::F32 => "F32",
            Dtype::U64 => "U64",
            Dtype::I64 => "I64",
            Dtype::F64 => "F64",
        }
    }

    pub const fn bytes(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "U16" => Dtype::U16,
            "I16" => Dtype::I16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "U32" => Dtype::U32,
            "I32" => Dtype::I32,
            "F32" => Dtype::F32,
            "U64" => Dtype::U64,
            "I64" => Dtype::I64,
            "F64" => Dtype::F64,
            other => return Err(invalid(format!("unknown safetensors dtype {other:?}"))),
        })
    }
}

/// One tensor's declared placement. Offsets are relative to the payload start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range within the payload, half-open.
    pub begin: u64,
    pub end: u64,
}

impl TensorEntry {
    pub fn len(&self) -> u64 {
        self.end - self.begin
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of logical elements, checked.
    pub fn elements(&self) -> Result<u64> {
        let mut n: u64 = 1;
        for d in &self.shape {
            n = n
                .checked_mul(*d)
                .ok_or_else(|| invalid("tensor shape product overflows u64"))?;
        }
        Ok(n)
    }

    /// Absolute file offset of this tensor's first byte.
    pub fn file_offset(&self, header: &Header) -> u64 {
        header.payload_start + self.begin
    }
}

/// A parsed and fully validated header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    tensors: BTreeMap<String, TensorEntry>,
    metadata: BTreeMap<String, String>,
    /// File offset where the payload begins: `8 + header_len`.
    pub payload_start: u64,
    /// Declared payload length, from the file length the caller supplied.
    pub payload_len: u64,
}

/// The three fields a tensor entry must carry.
///
/// Deserialized **directly** from the header's map, never through an
/// intermediate `serde_json::Value`. That is load-bearing: building a `Value`
/// first collapses a duplicated `"dtype"` into whichever copy came last, so a
/// header declaring an unsupported dtype and then overwriting it with a
/// supported one would be accepted. Deserializing straight into this struct
/// makes serde's own duplicate-field rejection apply.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    dtype: String,
    shape: Shape,
    data_offsets: [u64; 2],
}

/// A shape, refused past [`MAX_RANK`] **as it is read**.
///
/// A plain `Vec<u64>` would be fully allocated before any check could run, so
/// the limit has to live in the deserializer: this is the difference between
/// refusing an absurd rank and paying for it first.
struct Shape(Vec<u64>);

impl<'de> serde::Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Shape;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "at most {MAX_RANK} dimensions")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Shape, A::Error> {
                let mut out = Vec::new();
                while let Some(d) = seq.next_element::<u64>()? {
                    if out.len() == MAX_RANK {
                        return Err(serde::de::Error::custom(format!(
                            "a tensor declares more than {MAX_RANK} dimensions"
                        )));
                    }
                    out.push(d);
                }
                Ok(Shape(out))
            }
        }
        d.deserialize_seq(V)
    }
}

/// What one key in the header's top-level map turned out to be.
enum RawItem {
    Tensor(RawEntry),
    Metadata(BTreeMap<String, String>),
}

/// The header's entries **in declaration order, duplicates preserved**.
///
/// A `BTreeMap` would silently keep one of two tensors sharing a name, and the
/// discarded one's bytes would vanish from every later check. Collecting into a
/// sequence first is what lets [`Header::parse`] refuse the duplicate instead.
struct RawHeader(Vec<(String, RawItem)>);

impl<'de> serde::Deserialize<'de> for RawHeader {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = RawHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<RawHeader, A::Error> {
                let mut out = Vec::new();
                let mut tensors = 0usize;
                while let Some(key) = map.next_key::<String>()? {
                    // Every limit is checked as the entry is read, so a header
                    // that exceeds one never costs what it asked for.
                    if key.len() > MAX_NAME_BYTES && key != "__metadata__" {
                        return Err(serde::de::Error::custom(format!(
                            "a tensor name is {} bytes, over the {MAX_NAME_BYTES}-byte limit",
                            key.len()
                        )));
                    }
                    let item = if key == "__metadata__" {
                        let m: BTreeMap<String, String> = map.next_value()?;
                        if m.len() > MAX_METADATA_ENTRIES {
                            return Err(serde::de::Error::custom(format!(
                                "__metadata__ has {} entries, over the \
                                 {MAX_METADATA_ENTRIES} limit",
                                m.len()
                            )));
                        }
                        RawItem::Metadata(m)
                    } else {
                        tensors += 1;
                        if tensors > MAX_TENSORS {
                            return Err(serde::de::Error::custom(format!(
                                "the header declares more than {MAX_TENSORS} tensors"
                            )));
                        }
                        RawItem::Tensor(map.next_value()?)
                    };
                    out.push((key, item));
                }
                Ok(RawHeader(out))
            }
        }
        d.deserialize_map(V)
    }
}

impl Header {
    /// Bytes a caller must read from the front of the file before `parse` can
    /// be called: the length prefix plus the declared header.
    ///
    /// Split out so a caller reads the prefix, learns the bound, and reads
    /// exactly that much -- rather than handing over a whole file.
    pub fn prefix_len(first_eight: &[u8]) -> Result<u64> {
        let bytes: [u8; 8] = first_eight
            .get(..8)
            .ok_or_else(|| invalid("a safetensors file needs at least an eight-byte length"))?
            .try_into()
            .expect("eight bytes");
        let header_len = u64::from_le_bytes(bytes);
        if header_len > MAX_HEADER_BYTES {
            return Err(capacity(header_len));
        }
        header_len
            .checked_add(8)
            .ok_or_else(|| invalid("header length overflows the file offset space"))
    }

    /// Parse and validate a header.
    ///
    /// `prefix` is the length prefix followed by exactly the declared header
    /// bytes; `file_len` is the whole file's length, which the payload bounds
    /// are checked against.
    pub fn parse(prefix: &[u8], file_len: u64) -> Result<Self> {
        let payload_start = Self::prefix_len(prefix)?;
        if prefix.len() as u64 != payload_start {
            return Err(invalid(format!(
                "header prefix is {} byte(s); the declared length needs {payload_start}",
                prefix.len()
            )));
        }
        if payload_start > file_len {
            return Err(invalid(format!(
                "header claims {payload_start} byte(s) of a {file_len}-byte file"
            )));
        }
        let payload_len = file_len - payload_start;

        let RawHeader(raw) = serde_json::from_slice(&prefix[8..]).map_err(|e| {
            invalid(format!(
                "safetensors header is not a valid tensor object: {e}"
            ))
        })?;

        let mut tensors: BTreeMap<String, TensorEntry> = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut seen_metadata = false;
        // Sorted by start, to check overlap in one pass rather than pairwise.
        let mut spans: Vec<(u64, u64, String)> = Vec::new();
        for (name, item) in raw {
            let entry = match item {
                RawItem::Metadata(map) => {
                    if seen_metadata {
                        return Err(invalid("the header declares __metadata__ twice"));
                    }
                    seen_metadata = true;
                    metadata = map;
                    continue;
                }
                RawItem::Tensor(entry) => entry,
            };
            // A duplicate name is ambiguous, not a last-one-wins choice: the
            // discarded entry's bytes would disappear from the overlap and
            // coverage checks below while the file still contains them.
            if tensors.contains_key(&name) {
                return Err(invalid(format!(
                    "the header declares tensor {name:?} twice"
                )));
            }
            let dtype = Dtype::parse(&entry.dtype)?;
            let [begin, end] = entry.data_offsets;
            if begin > end {
                return Err(invalid(format!(
                    "tensor {name:?} has a reversed range [{begin}, {end})"
                )));
            }
            if end > payload_len {
                return Err(invalid(format!(
                    "tensor {name:?} ends at {end}, past the {payload_len}-byte payload"
                )));
            }
            let tensor = TensorEntry {
                dtype,
                shape: entry.shape.0,
                begin,
                end,
            };
            // The declared range must be exactly what the shape and dtype need:
            // a tensor that merely *fits* would let a truncated or padded
            // artifact through, and every later read derives its length here.
            let need = tensor
                .elements()?
                .checked_mul(dtype.bytes() as u64)
                .ok_or_else(|| invalid(format!("tensor {name:?} byte length overflows u64")))?;
            if tensor.len() != need {
                return Err(invalid(format!(
                    "tensor {name:?} is {} shape-{:?} {} element(s) needing {need} byte(s), \
                     but its range spans {}",
                    dtype.name(),
                    tensor.shape,
                    tensor.elements()?,
                    tensor.len()
                )));
            }
            spans.push((begin, end, name.clone()));
            tensors.insert(name, tensor);
        }

        spans.sort_unstable();
        for pair in spans.windows(2) {
            let (_, a_end, a_name) = &pair[0];
            let (b_begin, _, b_name) = &pair[1];
            // Zero-length tensors can share a boundary; a nonempty overlap
            // means two tensors claim the same bytes.
            if b_begin < a_end {
                return Err(invalid(format!(
                    "tensors {a_name:?} and {b_name:?} overlap at byte {b_begin}"
                )));
            }
        }
        Ok(Self {
            tensors,
            metadata,
            payload_start,
            payload_len,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("this shard has no tensor {name:?}")))
    }

    /// Payload bytes every tensor accounts for, and whether they cover it.
    ///
    /// Reported rather than enforced: a shard whose payload has trailing slack
    /// is unusual and worth surfacing, but it is not itself corruption, and the
    /// inventory records the covered total as evidence.
    pub fn covered_bytes(&self) -> u64 {
        self.tensors.values().map(TensorEntry::len).sum()
    }

    pub fn covers_payload_exactly(&self) -> bool {
        self.covered_bytes() == self.payload_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a header the way a writer would, so the reader is tested against
    /// an independently constructed file rather than its own output.
    fn file(entries: &[(&str, &str, &[u64], u64, u64)], payload: usize) -> Vec<u8> {
        let mut json = String::from("{");
        for (i, (name, dtype, shape, begin, end)) in entries.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            json.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{begin},{end}]}}",
                shape.join(",")
            ));
        }
        json.push('}');
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json.as_bytes());
        out.resize(out.len() + payload, 0);
        out
    }

    #[test]
    fn a_well_formed_header_reports_every_tensor_and_its_absolute_offset() {
        let bytes = file(
            &[
                ("a", "I32", &[2, 3], 0, 24),
                ("b", "BF16", &[4], 24, 32),
                ("c", "I64", &[2], 32, 48),
            ],
            48,
        );
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        let h = Header::parse(&bytes[..prefix], bytes.len() as u64).unwrap();
        assert_eq!(h.tensors().len(), 3);
        assert_eq!(h.get("a").unwrap().dtype, Dtype::I32);
        assert_eq!(h.get("a").unwrap().shape, vec![2, 3]);
        assert_eq!(h.get("b").unwrap().file_offset(&h), prefix as u64 + 24);
        assert!(h.covers_payload_exactly());
        assert_eq!(h.covered_bytes(), 48);
        assert!(h.get("missing").is_err());
    }

    #[test]
    fn metadata_is_a_string_map_and_not_a_tensor() {
        let json = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(bytes.len() + 4, 0);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        let h = Header::parse(&bytes[..prefix], bytes.len() as u64).unwrap();
        assert_eq!(h.tensors().len(), 1);
        assert_eq!(h.metadata()["format"], "pt");
    }

    #[test]
    fn every_malformed_header_is_a_typed_error_and_not_a_panic() {
        // A range that does not match shape x dtype, in both directions.
        for (shape, begin, end) in [(&[2u64, 3u64][..], 0, 20), (&[2, 3][..], 0, 28)] {
            let bytes = file(&[("a", "I32", shape, begin, end)], 32);
            let prefix = Header::prefix_len(&bytes).unwrap() as usize;
            assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        }
        // Overlap.
        let bytes = file(&[("a", "U8", &[8], 0, 8), ("b", "U8", &[8], 4, 12)], 12);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        // Reversed range.
        let bytes = file(&[("a", "U8", &[0], 8, 4)], 12);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        // Past the payload.
        let bytes = file(&[("a", "U8", &[64], 0, 64)], 8);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        // Unknown dtype, refused rather than skipped.
        let bytes = file(&[("a", "FP8", &[4], 0, 4)], 4);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        // Missing field.
        let json = r#"{"a":{"dtype":"U8","shape":[4]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(bytes.len() + 4, 0);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix], bytes.len() as u64).is_err());
        // Not JSON at all.
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"[[[[");
        assert!(Header::parse(&bytes, bytes.len() as u64).is_err());
        // Truncated: fewer than eight bytes, and a prefix shorter than declared.
        assert!(Header::prefix_len(&[0, 1, 2]).is_err());
        let bytes = file(&[("a", "U8", &[4], 0, 4)], 4);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        assert!(Header::parse(&bytes[..prefix - 1], bytes.len() as u64).is_err());
        // A header longer than the file it claims to describe.
        let bytes = file(&[("a", "U8", &[4], 0, 4)], 4);
        assert!(Header::parse(&bytes[..prefix], 8).is_err());
    }

    /// The bound is applied to the declared length, before any allocation.
    #[test]
    fn an_implausible_header_length_is_capacity_exceeded_not_an_allocation() {
        let huge = (MAX_HEADER_BYTES + 1).to_le_bytes();
        assert!(matches!(
            Header::prefix_len(&huge),
            Err(Error::CapacityExceeded { requested_bytes, .. })
                if requested_bytes == MAX_HEADER_BYTES + 1
        ));
        // u64::MAX would overflow `+ 8` if it were not bounded first.
        assert!(matches!(
            Header::prefix_len(&u64::MAX.to_le_bytes()),
            Err(Error::CapacityExceeded { .. })
        ));
        // And the largest allowed length is still refused when the file is not
        // that big, by the ordinary bounds check rather than by the cap.
        let mut ok = MAX_HEADER_BYTES.to_le_bytes().to_vec();
        ok.resize(16, 0);
        assert!(Header::parse(&ok, 16).is_err());
    }

    /// Duplicates are ambiguous and are refused, not resolved last-one-wins.
    ///
    /// Independent review found both cases accepted: two tensors sharing a
    /// name collapsed into one, and a `"dtype":"FP8"` overwritten by
    /// `"dtype":"U8"` was parsed as `U8` -- so an unsupported dtype could be
    /// smuggled past the check that exists to refuse it.
    #[test]
    fn duplicate_names_and_duplicate_fields_are_both_refused() {
        let raw = |json: &str, payload: usize| {
            let mut b = (json.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(json.as_bytes());
            b.resize(b.len() + payload, 0);
            b
        };
        let parse = |b: &[u8]| {
            let p = Header::prefix_len(b).unwrap() as usize;
            Header::parse(&b[..p], b.len() as u64)
        };
        // Two tensors with one name. The second would have hidden the first.
        let b = raw(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            8,
        );
        assert!(parse(&b).is_err());
        // An unsupported dtype overwritten by a supported one.
        let b = raw(
            r#"{"a":{"dtype":"FP8","dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            4,
        );
        assert!(parse(&b).is_err());
        // The same for the other two structural fields, and for __metadata__.
        for json in [
            r#"{"a":{"dtype":"U8","shape":[4],"shape":[8],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,8],"data_offsets":[0,4]}}"#,
            r#"{"__metadata__":{"f":"a"},"__metadata__":{"f":"b"},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        ] {
            assert!(parse(&raw(json, 8)).is_err(), "{json}");
        }
        // A field the schema does not define is refused rather than ignored.
        let b = raw(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4],"extra":1}}"#,
            4,
        );
        assert!(parse(&b).is_err());
        // The negative control: the same header without a duplicate parses.
        let b = raw(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            4,
        );
        assert!(parse(&b).is_ok());
    }

    #[test]
    fn zero_length_tensors_may_share_a_boundary() {
        let bytes = file(&[("a", "U8", &[0], 4, 4), ("b", "U8", &[4], 0, 4)], 4);
        let prefix = Header::prefix_len(&bytes).unwrap() as usize;
        let h = Header::parse(&bytes[..prefix], bytes.len() as u64).unwrap();
        assert!(h.get("a").unwrap().is_empty());
        assert_eq!(h.covered_bytes(), 4);
    }
}
