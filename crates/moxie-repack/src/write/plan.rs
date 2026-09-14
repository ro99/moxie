//! The output plan: which physical tensor goes in which shard, decided before
//! a byte is written.
//!
//! [ADR 0025] makes a canonical artifact one or more conforming safetensors
//! shards plus a manifest. A logical tensor becomes one, two or three physical
//! tensors, and **a component never spans shards**: a component larger than the
//! admitted shard size is a refusal, not a tensor split across files.
//!
//! Every offset in a safetensors shard depends on the exact header, so the
//! header is built here -- from the complete component list, before the run
//! starts -- and every later write goes to an offset this plan fixed. That is
//! also what lets a resumed run write into the same places.
//!
//! [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md

use std::collections::BTreeMap;

use moxie_format::canonical::{Component, ComponentKind};
use moxie_format::manifest::{AffineFields, TensorPrecision};
use moxie_format::safetensors::{self, PlannedTensor as ShardTensor, ShardLayout};
use moxie_format::sha256::StreamingSha256;
use moxie_types::{Error, Result};

use crate::write::WriteBudget;

fn invalid(detail: String) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// One tensor a caller wants published, with the components it becomes.
#[derive(Debug, Clone)]
pub struct TensorRequest {
    pub role: String,
    pub shape: Vec<u64>,
    pub precision: TensorPrecision,
    /// Present exactly for affine precisions; the manifest validator refuses a
    /// disagreement either way.
    pub affine: Option<AffineFields>,
    /// The physical tensors this becomes, in canonical order.
    pub components: Vec<Component>,
}

impl TensorRequest {
    pub fn payload_bytes(&self) -> u64 {
        self.components.iter().map(|c| c.len).sum()
    }
}

/// One component, placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedComponent {
    pub role: String,
    pub kind: ComponentKind,
    /// The tensor's name inside its shard.
    pub name: String,
    /// The shard file.
    pub file: String,
    /// Absolute offset in that file, past the header.
    pub file_offset: u64,
    pub len: u64,
}

/// One shard: its name, its exact header bytes, and what it holds.
#[derive(Debug, Clone)]
pub struct ShardPlan {
    pub file: String,
    pub layout: ShardLayout,
}

impl ShardPlan {
    /// Total file length: header plus every payload.
    pub fn file_len(&self) -> u64 {
        self.layout.file_len()
    }
}

/// Where every component goes.
#[derive(Debug, Clone)]
pub struct OutputPlan {
    components: Vec<PlacedComponent>,
    shards: Vec<ShardPlan>,
    requests: Vec<TensorRequest>,
}

impl OutputPlan {
    /// Assign components to shards and fix every offset.
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
        // Group components into shards first; the names need the total, and
        // the offsets need the headers, which need the names.
        let mut groups: Vec<Vec<(String, Component)>> = Vec::new();
        let mut current: Vec<(String, Component)> = Vec::new();
        let mut used: u64 = 0;
        for request in &requests {
            if request.components.is_empty() {
                return Err(invalid(format!(
                    "tensor '{}' has no components: nothing to publish",
                    request.role
                )));
            }
            for component in &request.components {
                if component.len == 0 {
                    return Err(invalid(format!(
                        "tensor '{}' component '{}' is empty",
                        request.role,
                        component.kind.name()
                    )));
                }
                // A component that cannot fit one shard at all is refused
                // here, naming the limit, rather than split across files.
                // `HEADER_ALLOWANCE` keeps room for the header the shard will
                // need; the exact size is known only once its tensors are, and
                // a plan that ignored it could exceed the budget by a header.
                if component.len + HEADER_ALLOWANCE > budget.chunk_file_bytes() {
                    return Err(invalid(format!(
                        "tensor '{}' component '{}' needs {} byte(s) and the admitted shard limit \
                         is {}: a larger component requires a larger admitted shard size, never a \
                         component split across shards",
                        request.role,
                        component.kind.name(),
                        component.len,
                        budget.chunk_file_bytes()
                    )));
                }
                if !current.is_empty()
                    && used + component.len + HEADER_ALLOWANCE > budget.chunk_file_bytes()
                {
                    groups.push(std::mem::take(&mut current));
                    used = 0;
                }
                used += component.len;
                current.push((request.role.clone(), component.clone()));
            }
        }
        if !current.is_empty() {
            groups.push(current);
        }

        // Now the names, and then the headers that fix every offset.
        let total = groups.len();
        let mut shards = Vec::with_capacity(total);
        let mut components = Vec::new();
        for (index, group) in groups.into_iter().enumerate() {
            let file = shard_name(index + 1, total);
            let planned: Vec<ShardTensor> = group
                .iter()
                .map(|(_, c)| ShardTensor {
                    name: c.name.clone(),
                    dtype: c.dtype,
                    shape: c.shape.clone(),
                    len: c.len,
                })
                .collect();
            let mut metadata = BTreeMap::new();
            metadata.insert(
                moxie_format::canonical::SCHEMA_TAG_KEY.to_string(),
                moxie_format::canonical::SCHEMA_TAG.to_string(),
            );
            let layout = safetensors::build_shard(&planned, &metadata)?;
            if layout.file_len() > budget.chunk_file_bytes() {
                return Err(invalid(format!(
                    "shard '{file}' would be {} byte(s), above the admitted shard limit of {}",
                    layout.file_len(),
                    budget.chunk_file_bytes()
                )));
            }
            for (role, component) in group {
                let placed = layout.tensor(&component.name).ok_or_else(|| {
                    invalid(format!(
                        "component '{}' was planned into '{file}' but the header does not carry it",
                        component.name
                    ))
                })?;
                components.push(PlacedComponent {
                    role,
                    kind: component.kind,
                    name: component.name.clone(),
                    file: file.clone(),
                    file_offset: placed.file_offset,
                    len: placed.len(),
                });
            }
            shards.push(ShardPlan { file, layout });
        }

        let payload: u64 = shards.iter().map(|s| s.file_len()).sum();
        // The disk budget covers the **whole** plan: shards, the journal and
        // the staged manifest.
        let total_disk = payload
            .checked_add(overhead_bytes)
            .ok_or_else(|| invalid("the disk plan overflows".into()))?;
        if total_disk > budget.disk_bytes() {
            return Err(invalid(format!(
                "this selection needs {total_disk} byte(s) of disk -- {payload} of shards plus at \
                 most {overhead_bytes} of journal and staged manifest -- above the admitted disk \
                 budget of {}",
                budget.disk_bytes()
            )));
        }
        Ok(Self {
            components,
            shards,
            requests,
        })
    }

    pub fn requests(&self) -> &[TensorRequest] {
        &self.requests
    }

    pub fn components(&self) -> &[PlacedComponent] {
        &self.components
    }

    pub fn shards(&self) -> &[ShardPlan] {
        &self.shards
    }

    pub fn component(&self, name: &str) -> Option<&PlacedComponent> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Every component of one role, in canonical order.
    pub fn components_of(&self, role: &str) -> impl Iterator<Item = &PlacedComponent> {
        self.components.iter().filter(move |c| c.role == role)
    }

    pub fn payload_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.file_len()).sum()
    }

    /// A digest over everything the output layout is: every role, its
    /// components, their names, dtypes, shapes, shards and offsets, each
    /// length-prefixed so concatenation is injective.
    pub fn digest(&self) -> String {
        let mut h = StreamingSha256::new();
        let mut field = |bytes: &[u8]| {
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        };
        field(b"output-plan-v2");
        for r in &self.requests {
            field(r.role.as_bytes());
            field(&r.shape.len().to_le_bytes());
            for d in &r.shape {
                field(&d.to_le_bytes());
            }
            field(r.precision.name().as_bytes());
            match &r.affine {
                None => field(b"dense"),
                Some(a) => {
                    // `Debug` of a fieldless enum is its variant name, and this
                    // feeds a resume binding rather than an identity anyone
                    // stores: a rendering that changed between builds costs a
                    // redone run, and the binding's converter field already
                    // refuses across builds.
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
        }
        for c in &self.components {
            field(c.role.as_bytes());
            field(c.kind.name().as_bytes());
            field(c.name.as_bytes());
            field(c.file.as_bytes());
            field(&c.file_offset.to_le_bytes());
            field(&c.len.to_le_bytes());
        }
        for s in &self.shards {
            field(s.file.as_bytes());
            field(s.layout.header_bytes());
        }
        h.finalize_hex()
    }
}

/// Room reserved for a shard's header when components are grouped.
///
/// The exact header is known only once a shard's tensor list is, and the list
/// is what grouping decides -- so grouping reserves this much and the built
/// layout is checked against the budget afterwards. Sixty-four kibibytes holds
/// a JSON header for hundreds of components.
pub const HEADER_ALLOWANCE: u64 = 64 * 1024;

/// `model-00001-of-00003.safetensors`, the convention the ecosystem reads.
pub fn shard_name(index: usize, total: usize) -> String {
    format!("model-{index:05}-of-{total:05}.safetensors")
}
