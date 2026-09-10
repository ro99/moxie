//! Pure graph-to-resource lowering.
//!
//! This crate makes no allocation and calls no backend. It binds a validated
//! graph's symbolic row dimension, derives exact tensor bytes and liveness, and
//! assigns non-overlapping logical values to deterministic physical slots. The
//! executor may then admit and materialize that immutable candidate.
//!
//! Semantic kernel selection is deliberately absent. A resource plan is the
//! prerequisite for execution, not evidence that an operation kernel exists.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

pub use moxie_graph::{Graph, GraphId, ValueId, ValueRole};
use moxie_graph::{GraphSignature, StateEffect, TensorSpec};
use moxie_types::{DeviceUuid, Error, Precision, Result, SymbolTable, TensorLayout};

const DEVICE_ALIGNMENT: u64 = 256;

/// One process-local resource-plan identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanId(u64);

impl PlanId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for PlanId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PlanId({})", self.0)
    }
}

#[derive(Debug)]
struct PlanIdAllocator {
    next: AtomicU64,
}

impl PlanIdAllocator {
    const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    fn allocate(&self) -> Result<PlanId> {
        let mut current = self.next.load(Ordering::Relaxed);
        loop {
            if current == 0 || current == u64::MAX {
                return Err(invalid(
                    "plan_id",
                    "process plan identity space is exhausted",
                ));
            }
            match self.next.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(PlanId(current)),
                Err(observed) => current = observed,
            }
        }
    }
}

static PLAN_IDS: PlanIdAllocator = PlanIdAllocator::new(1);

/// Which one-user execution phase this row bucket represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Prefill,
    Decode,
    Verify,
    Entropy,
}

impl Phase {
    pub const fn name(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
            Phase::Verify => "verify",
            Phase::Entropy => "entropy",
        }
    }
}

/// Exact row/context bucket lowered by this resource plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceWorkload {
    pub phase: Phase,
    pub rows: u64,
    pub visible_tokens: u64,
    pub branch_rows: u64,
    pub output: ValueId,
    pub device: DeviceUuid,
}

/// Inclusive node-stage lifetime. The terminal-output stage is `node_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub first: u32,
    pub last: u32,
}

/// One external per-step input. It receives no device allocation in this task.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalInput {
    pub value: ValueId,
    pub shape: Vec<u64>,
    pub role: ValueRole,
}

/// One external immutable weight requirement.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalWeight {
    pub value: ValueId,
    pub shape: Vec<u64>,
    pub role: ValueRole,
    pub required_bytes: u64,
}

/// One logical activation backed by a physical arena slot.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaTensor {
    pub value: ValueId,
    pub shape: Vec<u64>,
    pub role: ValueRole,
    pub layout: TensorLayout,
    pub slot: u32,
    pub offset: u64,
    pub bytes: u64,
    pub live: LiveRange,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueBinding {
    ExternalInput(ExternalInput),
    ExternalWeight(ExternalWeight),
    ArenaTensor(ArenaTensor),
}

impl ValueBinding {
    pub const fn value(&self) -> ValueId {
        match self {
            ValueBinding::ExternalInput(v) => v.value,
            ValueBinding::ExternalWeight(v) => v.value,
            ValueBinding::ArenaTensor(v) => v.value,
        }
    }
}

/// One simultaneously allocated physical range in the activation arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaSlot {
    pub id: u32,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
}

/// Pure, immutable candidate. It owns no reservation and no physical bytes.
#[derive(Debug)]
#[must_use = "a plan candidate must be admitted or explicitly discarded"]
pub struct PlanCandidate {
    id: PlanId,
    graph_id: GraphId,
    graph_signature: GraphSignature,
    workload: ResourceWorkload,
    stages: Vec<String>,
    bindings: Vec<ValueBinding>,
    slots: Vec<ArenaSlot>,
    activation_arena_bytes: u64,
    weight_bytes: u64,
}

impl PlanCandidate {
    pub const fn id(&self) -> PlanId {
        self.id
    }

    pub const fn graph_id(&self) -> GraphId {
        self.graph_id
    }

    pub const fn workload(&self) -> ResourceWorkload {
        self.workload
    }

    pub fn stages(&self) -> &[String] {
        &self.stages
    }

    pub fn bindings(&self) -> &[ValueBinding] {
        &self.bindings
    }

    pub fn binding(&self, value: ValueId) -> Option<&ValueBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.value() == value)
    }

    pub fn slots(&self) -> &[ArenaSlot] {
        &self.slots
    }

    pub const fn activation_arena_bytes(&self) -> u64 {
        self.activation_arena_bytes
    }

    pub const fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Identity and exact structure must both agree.
    pub fn matches_graph(&self, graph: &Graph) -> bool {
        self.graph_id == graph.id() && self.graph_signature == graph.signature()
    }
}

#[derive(Debug, Clone)]
struct PendingTensor {
    value: ValueId,
    shape: Vec<u64>,
    role: ValueRole,
    bytes: u64,
    live: LiveRange,
    slot: u32,
}

#[derive(Debug)]
struct PendingSlot {
    bytes: u64,
    alignment: u64,
    available_after: u32,
}

/// Lower one validated stateless graph to an exact resource candidate.
pub fn lower(graph: &Graph, workload: ResourceWorkload) -> Result<PlanCandidate> {
    lower_with_ids(graph, workload, &PLAN_IDS)
}

fn lower_with_ids(
    graph: &Graph,
    workload: ResourceWorkload,
    ids: &PlanIdAllocator,
) -> Result<PlanCandidate> {
    validate_workload(graph, workload)?;

    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), workload.rows);
    let concrete: Vec<Vec<u64>> = graph
        .values()
        .iter()
        .map(|spec| concrete_shape(spec, &symbols))
        .collect::<Result<_>>()?;

    let node_count = u32::try_from(graph.nodes().len()).map_err(|_| {
        invalid(
            "graph",
            "the graph has more nodes than the plan stage index can represent",
        )
    })?;
    let mut producers = BTreeMap::<ValueId, u32>::new();
    let mut consumers = BTreeMap::<ValueId, u32>::new();
    for (stage, node) in graph.nodes().iter().enumerate() {
        let stage = stage as u32;
        producers.insert(node.output, stage);
        for input in &node.inputs {
            consumers
                .entry(*input)
                .and_modify(|last| *last = (*last).max(stage))
                .or_insert(stage);
        }
        let workspace = node.contract.workspace_upper_bound.eval(&symbols)?;
        if workspace != 0 {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: "resource-only lowering cannot treat a declared workspace as zero".into(),
            });
        }
    }

    if !producers.contains_key(&graph.output()) {
        return Err(invalid(
            "output",
            "the selected output must be produced by a graph node",
        ));
    }

    let inputs: BTreeSet<ValueId> = graph.inputs().iter().copied().collect();
    let weights: BTreeSet<ValueId> = graph.weights().iter().copied().collect();
    let mut pending = Vec::new();
    let mut bindings = Vec::with_capacity(graph.value_count());
    let mut weight_bytes = 0u64;

    for (index, (spec, shape)) in graph.values().iter().zip(&concrete).enumerate() {
        let value = ValueId(index as u32);
        let shape = shape.clone();
        if inputs.contains(&value) {
            bindings.push(ValueBinding::ExternalInput(ExternalInput {
                value,
                shape,
                role: spec.role,
            }));
            continue;
        }
        if weights.contains(&value) {
            let required_bytes = tensor_bytes(spec.role, &shape)?;
            weight_bytes = weight_bytes
                .checked_add(required_bytes)
                .ok_or_else(|| invalid("weights", "weight byte total overflowed"))?;
            bindings.push(ValueBinding::ExternalWeight(ExternalWeight {
                value,
                shape,
                role: spec.role,
                required_bytes,
            }));
            continue;
        }
        let Some(first) = producers.get(&value).copied() else {
            return Err(invalid(
                "value",
                format!("value {} is neither external nor produced", value.0),
            ));
        };
        let last = if value == graph.output() {
            node_count
        } else if let Some(last) = consumers.get(&value).copied() {
            last
        } else {
            return Err(invalid(
                "graph",
                format!("node output {} is dead and has no consumer", value.0),
            ));
        };
        let bytes = tensor_bytes(spec.role, &shape)?;
        pending.push(PendingTensor {
            value,
            shape,
            role: spec.role,
            bytes,
            live: LiveRange { first, last },
            slot: u32::MAX,
        });
    }

    let mut slots: Vec<PendingSlot> = Vec::new();
    for tensor in &mut pending {
        let reusable = slots
            .iter()
            .position(|slot| slot.available_after < tensor.live.first);
        let slot_index = reusable.unwrap_or(slots.len());
        if slot_index == slots.len() {
            slots.push(PendingSlot {
                bytes: tensor.bytes,
                alignment: DEVICE_ALIGNMENT,
                available_after: tensor.live.last,
            });
        } else {
            let slot = &mut slots[slot_index];
            slot.bytes = slot.bytes.max(tensor.bytes);
            slot.alignment = slot.alignment.max(DEVICE_ALIGNMENT);
            slot.available_after = tensor.live.last;
        }
        tensor.slot = u32::try_from(slot_index)
            .map_err(|_| invalid("slots", "slot identity does not fit u32"))?;
    }

    let mut cursor = 0u64;
    let mut planned_slots = Vec::with_capacity(slots.len());
    for (index, slot) in slots.into_iter().enumerate() {
        let offset = align_up(cursor, slot.alignment)?;
        cursor = offset
            .checked_add(slot.bytes)
            .ok_or_else(|| invalid("arena", "slot extent overflowed"))?;
        planned_slots.push(ArenaSlot {
            id: index as u32,
            offset,
            bytes: slot.bytes,
            alignment: slot.alignment,
        });
    }
    let activation_arena_bytes = align_up(cursor, DEVICE_ALIGNMENT)?;

    for tensor in pending {
        let slot = &planned_slots[tensor.slot as usize];
        bindings.push(ValueBinding::ArenaTensor(ArenaTensor {
            value: tensor.value,
            shape: tensor.shape,
            role: tensor.role,
            layout: TensorLayout::ContiguousRowMajorV1,
            slot: tensor.slot,
            offset: slot.offset,
            bytes: tensor.bytes,
            live: tensor.live,
        }));
    }
    bindings.sort_by_key(ValueBinding::value);

    let mut stages: Vec<String> = graph
        .nodes()
        .iter()
        .map(|node| format!("node-{}-{}", node.id.0, node.params.op().name()))
        .collect();
    stages.push("terminal-output".into());

    let id = ids.allocate()?;
    Ok(PlanCandidate {
        id,
        graph_id: graph.id(),
        graph_signature: graph.signature(),
        workload,
        stages,
        bindings,
        slots: planned_slots,
        activation_arena_bytes,
        weight_bytes,
    })
}

fn validate_workload(graph: &Graph, workload: ResourceWorkload) -> Result<()> {
    if workload.rows == 0 {
        return Err(invalid("rows", "a resource plan needs at least one row"));
    }
    if workload.visible_tokens == 0 {
        return Err(invalid(
            "visible_tokens",
            "visible history is explicit and nonzero",
        ));
    }
    if workload.branch_rows == 0 {
        return Err(invalid("branch_rows", "a resource plan needs branch rows"));
    }
    if workload.device.to_bytes() == [0; 16] {
        return Err(invalid("device", "the nil UUID is not a device identity"));
    }
    if workload.output != graph.output() {
        return Err(invalid(
            "output",
            format!(
                "workload selected value {}, graph output is {}",
                workload.output.0,
                graph.output().0
            ),
        ));
    }
    match workload.phase {
        Phase::Prefill => {}
        Phase::Decode if workload.rows == 1 => {}
        Phase::Decode => {
            return Err(invalid("rows", "a decode bucket contains exactly one row"));
        }
        Phase::Verify | Phase::Entropy => {
            return Err(Error::Unsupported {
                capability: "branch_resource_plan",
                reason: format!(
                    "{} needs branch state and admission not implemented in this slice",
                    workload.phase.name()
                ),
            });
        }
    }
    if workload.branch_rows != workload.rows {
        return Err(invalid(
            "branch_rows",
            "prefill/decode resource plans require branch_rows == rows",
        ));
    }
    if let Some((node, effect)) = graph
        .state_effects()
        .into_iter()
        .find(|(_, effect)| *effect != StateEffect::None)
    {
        return Err(Error::Unsupported {
            capability: "stateful_resource_plan",
            reason: format!("node {} has state effect {effect:?}", node.0),
        });
    }
    Ok(())
}

fn concrete_shape(spec: &TensorSpec, symbols: &SymbolTable) -> Result<Vec<u64>> {
    let shape = spec.extent(symbols)?;
    if shape.contains(&0) {
        return Err(invalid("shape", "tensor extents must be nonzero"));
    }
    Ok(shape)
}

fn tensor_bytes(role: ValueRole, shape: &[u64]) -> Result<u64> {
    let elements = shape.iter().try_fold(1u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid("shape", "tensor element count overflowed"))
    })?;
    let precision = match role {
        ValueRole::Activation(precision) => precision.get(),
        ValueRole::Weight(precision) => {
            let precision = precision.get();
            if precision.is_integer() {
                return Err(Error::Unsupported {
                    capability: "packed_weight_resource_plan",
                    reason: "integer weight bytes require affine scale/zero-point metadata absent from the graph"
                        .into(),
                });
            }
            precision
        }
        ValueRole::Index => {
            return Err(invalid(
                "role",
                "index values are external and have no implicit element width",
            ));
        }
    };
    let bytes_per_element = match precision {
        Precision::Bf16 | Precision::F16 => 2,
        Precision::F32 => 4,
        Precision::Int4 | Precision::Int8 => unreachable!("integer weights refused above"),
    };
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| invalid("shape", "tensor byte count overflowed"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .ok_or_else(|| invalid("alignment", "aligned byte offset overflowed"))
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_graph::{
        GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
        Visibility,
    };
    use moxie_types::{ActivationPrecision, Dim, Precision, SymbolId, WeightPrecision};

    const ORACLE: OracleId = OracleId("plan-test");
    const ROWS: SymbolId = SymbolId(44);

    fn uuid() -> DeviceUuid {
        DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000011").unwrap()
    }

    fn registry() -> OracleRegistry {
        let mut registry = OracleRegistry::new();
        for op in [Op::Linear, Op::Residual, Op::VocabProjection, Op::Attention] {
            registry
                .register(
                    op,
                    ORACLE,
                    OracleEvidence {
                        implementation: "moxie_plan::tests",
                        test_module: "moxie_plan::tests",
                    },
                )
                .unwrap();
        }
        registry
    }

    fn activation(rows: Dim, width: u64) -> TensorSpec {
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![rows, Dim::constant(width)],
        )
    }

    fn weight(rows: u64, columns: u64) -> TensorSpec {
        TensorSpec::new(
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            vec![Dim::constant(rows), Dim::constant(columns)],
        )
    }

    fn linear_chain(width: u64, fan_in: bool) -> Graph {
        let rows = Dim::symbol(ROWS);
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let input = builder.input("x", activation(rows, width));
        let w1 = builder.weight("w1", weight(width, width)).unwrap();
        let first = builder
            .node(
                OpParams::Linear {
                    in_features: width,
                    out_features: width,
                    bias: false,
                },
                &[input, w1],
            )
            .unwrap();
        let w2 = builder.weight("w2", weight(width, width)).unwrap();
        let second = builder
            .node(
                OpParams::Linear {
                    in_features: width,
                    out_features: width,
                    bias: false,
                },
                &[first, w2],
            )
            .unwrap();
        let output = if fan_in {
            builder.node(OpParams::Residual, &[first, second]).unwrap()
        } else {
            let w3 = builder.weight("w3", weight(width, width)).unwrap();
            builder
                .node(
                    OpParams::Linear {
                        in_features: width,
                        out_features: width,
                        bias: false,
                    },
                    &[second, w3],
                )
                .unwrap()
        };
        builder.finish(output, &registry()).unwrap()
    }

    fn workload(graph: &Graph, rows: u64) -> ResourceWorkload {
        ResourceWorkload {
            phase: if rows == 1 {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            rows,
            visible_tokens: rows.max(17),
            branch_rows: rows,
            output: graph.output(),
            device: uuid(),
        }
    }

    #[test]
    fn derives_shapes_bytes_liveness_and_fan_in_slots() {
        let graph = linear_chain(8, true);
        let plan = lower(&graph, workload(&graph, 3)).unwrap();
        assert_eq!(plan.activation_arena_bytes(), 3 * DEVICE_ALIGNMENT);
        assert_eq!(plan.weight_bytes(), 2 * 8 * 8 * 2);
        assert_eq!(plan.slots().len(), 3);
        let tensors: Vec<_> = plan
            .bindings()
            .iter()
            .filter_map(|binding| match binding {
                ValueBinding::ArenaTensor(tensor) => Some(tensor),
                _ => None,
            })
            .collect();
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors[0].shape, vec![3, 8]);
        assert_eq!(tensors[0].bytes, 48);
        assert_eq!(tensors[0].live, LiveRange { first: 0, last: 2 });
        assert_eq!(tensors[2].live, LiveRange { first: 2, last: 3 });
        assert_eq!(tensors[0].layout, TensorLayout::ContiguousRowMajorV1);
    }

    #[test]
    fn reuses_only_a_strictly_ended_slot_and_is_deterministic() {
        let graph = linear_chain(16, false);
        let first = lower(&graph, workload(&graph, 2)).unwrap();
        let second = lower(&graph, workload(&graph, 2)).unwrap();
        assert_eq!(first.slots(), second.slots());
        assert_eq!(first.slots().len(), 2);
        let tensors: Vec<_> = first
            .bindings()
            .iter()
            .filter_map(|binding| match binding {
                ValueBinding::ArenaTensor(tensor) => Some(tensor),
                _ => None,
            })
            .collect();
        assert_eq!(tensors[0].slot, tensors[2].slot);
        assert_ne!(tensors[0].slot, tensors[1].slot);
        assert!(tensors[0].live.last < tensors[2].live.first);
        assert_eq!(first.activation_arena_bytes(), 2 * DEVICE_ALIGNMENT);
    }

    #[test]
    fn f32_output_bytes_and_alignment_padding_are_exact() {
        let rows = Dim::symbol(ROWS);
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let input = builder.input("x", activation(rows, 5));
        let weight = builder.weight("vocab", weight(7, 5)).unwrap();
        let output = builder
            .node(
                OpParams::VocabProjection {
                    vocab: 7,
                    hidden: 5,
                },
                &[input, weight],
            )
            .unwrap();
        let graph = builder.finish(output, &registry()).unwrap();
        let plan = lower(&graph, workload(&graph, 3)).unwrap();
        let ValueBinding::ArenaTensor(output) = plan.binding(output).unwrap() else {
            panic!("graph output must use the activation arena");
        };
        assert_eq!(output.shape, [3, 7]);
        assert_eq!(output.role.precision(), Some(Precision::F32));
        assert_eq!(output.bytes, 3 * 7 * 4);
        assert_eq!(output.offset, 0);
        assert_eq!(plan.activation_arena_bytes(), DEVICE_ALIGNMENT);
        assert_eq!(plan.weight_bytes(), 7 * 5 * 2);
    }

    #[test]
    fn rows_visible_history_phase_and_device_are_independent_checks() {
        let graph = linear_chain(4, false);
        let mut request = workload(&graph, 2);
        request.visible_tokens = 32_768;
        assert_eq!(lower(&graph, request).unwrap().workload().rows, 2);

        request.phase = Phase::Decode;
        assert!(lower(&graph, request).is_err());
        request.phase = Phase::Entropy;
        assert_eq!(lower(&graph, request).unwrap_err().kind(), "unsupported");
        request.phase = Phase::Prefill;
        request.branch_rows = 1;
        assert!(lower(&graph, request).is_err());
        request.branch_rows = 2;
        request.device = DeviceUuid::from_bytes([0; 16]);
        assert!(lower(&graph, request).is_err());
    }

    #[test]
    fn malformed_and_stateful_requests_fail_before_identity_or_bytes_exist() {
        let graph = linear_chain(4, false);
        let base = workload(&graph, 2);
        for malformed in [
            ResourceWorkload { rows: 0, ..base },
            ResourceWorkload {
                visible_tokens: 0,
                ..base
            },
            ResourceWorkload {
                branch_rows: 0,
                ..base
            },
            ResourceWorkload {
                output: ValueId(u32::MAX),
                ..base
            },
        ] {
            assert_eq!(
                lower(&graph, malformed).unwrap_err().kind(),
                "invalid_request"
            );
        }
        assert_eq!(
            lower(
                &graph,
                ResourceWorkload {
                    phase: Phase::Verify,
                    ..base
                }
            )
            .unwrap_err()
            .kind(),
            "unsupported"
        );
        assert!(
            tensor_bytes(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                &[u64::MAX, 2]
            )
            .is_err()
        );
        assert!(align_up(u64::MAX, DEVICE_ALIGNMENT).is_err());

        let rows = Dim::symbol(ROWS);
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let q = builder.input("q", activation(rows.clone(), 4));
        let k = builder.input("k", activation(rows.clone(), 4));
        let v = builder.input("v", activation(rows.clone(), 4));
        let positions = builder.input("positions", TensorSpec::new(ValueRole::Index, vec![rows]));
        let output = builder
            .node(
                OpParams::Attention {
                    heads: 1,
                    head_dim: 4,
                    visibility: Visibility::Causal,
                    layer: 0,
                },
                &[q, k, v, positions],
            )
            .unwrap();
        let stateful = builder.finish(output, &registry()).unwrap();
        assert_eq!(
            lower(&stateful, workload(&stateful, 2)).unwrap_err().kind(),
            "unsupported"
        );
    }

    #[test]
    fn graph_identity_and_exact_structure_are_both_bound() {
        let graph = linear_chain(4, false);
        let mut plan = lower(&graph, workload(&graph, 1)).unwrap();
        assert!(plan.matches_graph(&graph));
        assert!(plan.matches_graph(&graph.clone()));
        let other = linear_chain(4, false);
        assert!(!plan.matches_graph(&other));

        plan.graph_signature = linear_chain(8, false).signature();
        assert!(!plan.matches_graph(&graph));
    }

    #[test]
    fn plan_identity_exhaustion_refuses_without_wrapping() {
        let graph = linear_chain(4, false);
        let ids = PlanIdAllocator::new(u64::MAX - 1);
        let last = lower_with_ids(&graph, workload(&graph, 1), &ids).unwrap();
        assert_eq!(last.id(), PlanId(u64::MAX - 1));
        assert_eq!(
            lower_with_ids(&graph, workload(&graph, 1), &ids)
                .unwrap_err()
                .kind(),
            "invalid_request"
        );
    }
}
