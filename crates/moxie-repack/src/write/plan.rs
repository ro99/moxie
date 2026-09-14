//! The output plan: which tensor's payload goes where, decided before a byte
//! is written.
//!
//! Manifest v1 assigns a tensor to **one** chunk, so a tensor's payload is one
//! contiguous range of one chunk file and a tensor larger than the admitted
//! chunk-file limit is a refusal rather than a tensor split across rows. The
//! bounded work units that fill that range are the repacker's business; the
//! range is this module's.

use moxie_format::manifest::{AffineFields, TensorPrecision};
use moxie_format::sha256::StreamingSha256;
use moxie_types::{Error, Result};

use crate::write::WriteBudget;

fn invalid(detail: String) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// One tensor a caller wants published.
#[derive(Debug, Clone)]
pub struct TensorRequest {
    pub role: String,
    pub shape: Vec<u64>,
    pub precision: TensorPrecision,
    /// Present exactly for affine precisions; the manifest validator refuses a
    /// disagreement either way.
    pub affine: Option<AffineFields>,
    /// The canonical payload length, from the shared codec (ADR 0023).
    pub length: u64,
    /// Byte alignment for the payload's start inside its chunk.
    pub alignment: u64,
}

/// One tensor's place in the output.
#[derive(Debug, Clone)]
pub struct PlannedTensor {
    pub request: TensorRequest,
    pub chunk: String,
    pub offset: u64,
    pub logical_order: u64,
}

impl PlannedTensor {
    pub fn end(&self) -> u64 {
        self.offset + self.request.length
    }
}

/// Every tensor's place, plus the disk the run will need.
#[derive(Debug, Clone)]
pub struct OutputPlan {
    tensors: Vec<PlannedTensor>,
    chunks: Vec<(String, u64)>,
}

impl OutputPlan {
    /// Assign chunks and offsets.
    ///
    /// Tensors keep the caller's order, which becomes their `logical_order`:
    /// the selection's order is a decision the caller made and this must not
    /// silently reorder it.
    pub fn build(
        requests: Vec<TensorRequest>,
        budget: &WriteBudget,
        overhead_bytes: u64,
    ) -> Result<Self> {
        if requests.is_empty() {
            return Err(invalid(
                "an empty selection publishes nothing: a manifest describes at least one tensor"
                    .into(),
            ));
        }
        let mut tensors = Vec::with_capacity(requests.len());
        let mut chunks: Vec<(String, u64)> = Vec::new();
        let mut current: Option<(String, u64)> = None;
        for (order, request) in requests.into_iter().enumerate() {
            if request.length == 0 {
                return Err(invalid(format!(
                    "tensor '{}' has a zero-length payload: length 0 describes no bytes",
                    request.role
                )));
            }
            if !request.alignment.is_power_of_two() {
                return Err(invalid(format!(
                    "tensor '{}': alignment {} is not a power of two",
                    request.role, request.alignment
                )));
            }
            // A tensor that cannot fit one chunk file at all: refused here,
            // naming the limit, rather than split across manifest rows. The
            // chunk-file limit is a disk-plan parameter, so a larger tensor
            // needs a larger admitted limit -- not a different manifest shape.
            let alone = request
                .length
                .checked_add(request.alignment - 1)
                .ok_or_else(|| invalid(format!("tensor '{}': size overflows", request.role)))?;
            if alone > budget.chunk_file_bytes() {
                return Err(invalid(format!(
                    "tensor '{}' needs {} byte(s) but the admitted chunk-file limit is {}: a \
                     larger tensor requires a larger admitted file limit, never a tensor split \
                     across manifest rows",
                    request.role,
                    request.length,
                    budget.chunk_file_bytes()
                )));
            }
            let (name, used) = match current.take() {
                Some(open) => open,
                None => (format!("chunk{}.bin", chunks.len()), 0u64),
            };
            let offset = used.next_multiple_of(request.alignment);
            let end = offset
                .checked_add(request.length)
                .ok_or_else(|| invalid(format!("tensor '{}': range overflows", request.role)))?;
            if end > budget.chunk_file_bytes() {
                // Close this chunk and open the next one, then place it there.
                chunks.push((name, used));
                let name = format!("chunk{}.bin", chunks.len());
                tensors.push(PlannedTensor {
                    chunk: name.clone(),
                    offset: 0,
                    logical_order: order as u64,
                    request: request.clone(),
                });
                current = Some((name, request.length));
                continue;
            }
            tensors.push(PlannedTensor {
                chunk: name.clone(),
                offset,
                logical_order: order as u64,
                request,
            });
            current = Some((name, end));
        }
        if let Some(open) = current {
            chunks.push(open);
        }
        let payload: u64 = chunks.iter().map(|(_, len)| *len).sum();
        // The disk budget covers the **whole** plan, not the payload alone: an
        // independent review published 5,005 bytes against a 4,096-byte budget
        // because the journal and the staged manifest were reported in an
        // estimate and enforced nowhere. `overhead_bytes` is the caller's bound
        // on those two, and it is checked here rather than printed.
        let total = payload
            .checked_add(overhead_bytes)
            .ok_or_else(|| invalid("the disk plan overflows".into()))?;
        if total > budget.disk_bytes() {
            return Err(invalid(format!(
                "this selection needs {total} byte(s) of disk -- {payload} of payload plus at \
                 most {overhead_bytes} of journal and staged manifest -- above the admitted disk \
                 budget of {}",
                budget.disk_bytes()
            )));
        }
        Ok(Self { tensors, chunks })
    }

    pub fn tensors(&self) -> &[PlannedTensor] {
        &self.tensors
    }

    pub fn tensor(&self, role: &str) -> Option<&PlannedTensor> {
        self.tensors.iter().find(|t| t.request.role == role)
    }

    /// Chunk names and their final lengths.
    pub fn chunks(&self) -> &[(String, u64)] {
        &self.chunks
    }

    pub fn payload_bytes(&self) -> u64 {
        self.chunks.iter().map(|(_, len)| *len).sum()
    }

    /// A digest over everything the output layout is: role, shape, precision,
    /// descriptor, length, chunk and offset, each length-prefixed.
    ///
    /// Part of the run binding, so a resume whose plan has changed in any of
    /// those respects is refused rather than continued into a mixed artifact.
    /// Length-prefixed for the reason [`moxie_format::manifest::artifact_identity`]
    /// is: bare delimiters are legal inside roles, so concatenation has to be
    /// injective on its own.
    pub fn digest(&self) -> String {
        let mut h = StreamingSha256::new();
        let mut field = |bytes: &[u8]| {
            // A closure over one hasher, so no caller can forget the length.
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        };
        field(b"output-plan-v1");
        for t in &self.tensors {
            field(t.request.role.as_bytes());
            field(&t.request.shape.len().to_le_bytes());
            for d in &t.request.shape {
                field(&d.to_le_bytes());
            }
            field(t.request.precision.name().as_bytes());
            match &t.request.affine {
                None => field(b"dense"),
                Some(a) => {
                    // `Debug` of a fieldless enum is its variant name. It feeds
                    // a **binding**, not an identity anyone stores: the only
                    // consequence of a rendering changing between builds is
                    // that a resume refuses and redoes the work, and the
                    // binding's `converter` field -- which carries this crate's
                    // version -- already refuses across builds. The artifact's
                    // own identity is `manifest::artifact_identity`, which
                    // spells every field out.
                    field(b"affine");
                    field(format!("{:?}", a.group_rule).as_bytes());
                    field(format!("{:?}", a.scale_dtype).as_bytes());
                    field(format!("{:?}", a.zero_point).as_bytes());
                    match &a.group_index {
                        None => field(b"contiguous"),
                        Some(map) => {
                            field(b"group-index");
                            for g in map {
                                field(&g.to_le_bytes());
                            }
                        }
                    }
                }
            }
            field(&t.request.length.to_le_bytes());
            field(&t.request.alignment.to_le_bytes());
            field(t.chunk.as_bytes());
            field(&t.offset.to_le_bytes());
            field(&t.logical_order.to_le_bytes());
        }
        h.finalize_hex()
    }
}
