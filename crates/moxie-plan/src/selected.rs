//! Pure semantic-kernel selection and exact combined-arena lowering.

use moxie_graph::{NodeId, Op, ValueId, ValueRole};
use moxie_types::{
    DeviceCapability, Error, KernelCatalogue, KernelOperand, SemanticKernelDescriptor,
    SemanticKernelOp, TensorLayout,
};

use crate::{Graph, PlanCandidate, ResourceWorkload, ValueBinding, lower};

const ALIGNMENT: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageRegion {
    Weights,
    Activations,
    Workspace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedNode {
    pub node: NodeId,
    pub descriptor: SemanticKernelDescriptor,
    pub workspace_logical_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedValue {
    pub value: ValueId,
    pub role: ValueRole,
    pub shape: Vec<u64>,
    pub region: StorageRegion,
    pub slot: u32,
    pub offset: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub first_stage: u32,
    pub last_stage: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedWorkspace {
    pub node: NodeId,
    pub offset: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub first_stage: u32,
    pub last_stage: u32,
}

/// Fully selected pure candidate. It still owns no reservation or device byte.
#[derive(Debug)]
pub struct SelectedPlanCandidate {
    base: PlanCandidate,
    device_sm: (u32, u32),
    catalogue_digest: [u8; 32],
    nodes: Vec<SelectedNode>,
    values: Vec<PlannedValue>,
    workspace: PlannedWorkspace,
    weight_region_bytes: u64,
    activation_region_bytes: u64,
    workspace_region_bytes: u64,
    combined_arena_bytes: u64,
    stages: Vec<String>,
}

impl SelectedPlanCandidate {
    pub fn base(&self) -> &PlanCandidate {
        &self.base
    }
    pub const fn workload(&self) -> ResourceWorkload {
        self.base.workload()
    }
    pub const fn catalogue_digest(&self) -> [u8; 32] {
        self.catalogue_digest
    }
    pub const fn device_sm(&self) -> (u32, u32) {
        self.device_sm
    }
    pub fn nodes(&self) -> &[SelectedNode] {
        &self.nodes
    }
    pub fn values(&self) -> &[PlannedValue] {
        &self.values
    }
    pub fn value(&self, id: ValueId) -> Option<&PlannedValue> {
        self.values.iter().find(|value| value.value == id)
    }
    pub const fn workspace(&self) -> &PlannedWorkspace {
        &self.workspace
    }
    pub const fn weight_region_bytes(&self) -> u64 {
        self.weight_region_bytes
    }
    pub const fn activation_region_bytes(&self) -> u64 {
        self.activation_region_bytes
    }
    pub const fn workspace_region_bytes(&self) -> u64 {
        self.workspace_region_bytes
    }
    pub const fn combined_arena_bytes(&self) -> u64 {
        self.combined_arena_bytes
    }
    pub fn stages(&self) -> &[String] {
        &self.stages
    }
    pub fn matches(
        &self,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
    ) -> bool {
        self.base.matches_graph(graph)
            && self.workload().device == capability.uuid
            && self.device_sm == (capability.compute_major, capability.compute_minor)
            && self.catalogue_digest == catalogue.digest()
    }
}

/// Select every node before allocating a plan identity or resource, then lower
/// exact weight/activation/workspace regions into one physical arena.
pub fn lower_selected(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
) -> Result<SelectedPlanCandidate, Error> {
    if capability.uuid != workload.device {
        return Err(invalid(
            "device",
            "workload UUID and measured capability differ",
        ));
    }
    let expected = [Op::Linear, Op::RmsNorm, Op::Residual];
    if graph.nodes().len() != expected.len()
        || graph
            .nodes()
            .iter()
            .zip(expected)
            .any(|(node, op)| node.params.op() != op)
    {
        return Err(Error::UnsupportedKernel {
            operation: "graph",
            detail: "task 0012 selects exactly Linear -> RmsNorm -> Residual".into(),
        });
    }
    let linear = &graph.nodes()[0];
    let rms = &graph.nodes()[1];
    let residual = &graph.nodes()[2];
    let exact_edges = graph.inputs().len() == 1
        && graph.weights().len() == 2
        && linear.inputs.as_slice() == [graph.inputs()[0], graph.weights()[0]]
        && rms.inputs.as_slice() == [linear.output, graph.weights()[1]]
        && residual.inputs.as_slice() == [graph.inputs()[0], rms.output]
        && graph.output() == residual.output;
    if !exact_edges {
        return Err(Error::UnsupportedKernel {
            operation: "graph",
            detail: "task 0012 requires exact x,W -> h; h,gain -> n; x,n -> y edges".into(),
        });
    }

    let mut selected = Vec::with_capacity(3);
    for node in graph.nodes() {
        let operation = semantic(node.params.op()).expect("exact chain checked above");
        let roles: Vec<_> = node
            .inputs
            .iter()
            .map(|value| operand(graph.values()[value.0 as usize].role))
            .collect::<Result<_, _>>()?;
        let (input, output) = node_shape(node, workload.rows)?;
        let matches: Vec<_> = catalogue
            .descriptors()
            .iter()
            .filter(|descriptor| {
                descriptor.operation == operation
                    && descriptor.inputs == roles
                    && descriptor.output == node.contract.output
                    && descriptor.accumulation == node.contract.accumulation
                    && descriptor.rounding == moxie_types::RoundingProfile::FinalBf16Rne
                    && descriptor.layout == TensorLayout::ContiguousRowMajorV1
                    && descriptor.sm.major == capability.compute_major
                    && descriptor.sm.minor == capability.compute_minor
                    && workload.rows <= descriptor.shape.max_rows
                    && input <= descriptor.shape.max_input
                    && output <= descriptor.shape.max_output
            })
            .collect();
        if matches.len() != 1 {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: format!(
                    "expected exactly one descriptor for rows={}, input={}, output={}, sm_{}{}; found {}",
                    workload.rows,
                    input,
                    output,
                    capability.compute_major,
                    capability.compute_minor,
                    matches.len()
                ),
            });
        }
        let descriptor = (*matches[0]).clone();
        if descriptor.abi_version != 1 || descriptor.symbols.is_empty() {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: "descriptor ABI or ordered symbol set is unsupported".into(),
            });
        }
        let required_workspace = if operation == SemanticKernelOp::RmsNorm {
            moxie_types::WorkspaceExpression::RowsTimesF32
        } else {
            moxie_types::WorkspaceExpression::Zero
        };
        if descriptor.workspace != required_workspace {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail:
                    "descriptor workspace expression differs from the closed operation contract"
                        .into(),
            });
        }
        let workspace_logical_bytes = descriptor
            .workspace
            .evaluate(workload.rows)
            .ok_or_else(|| invalid("workspace", "selected workspace expression overflowed"))?;
        if operation == SemanticKernelOp::RmsNorm && workspace_logical_bytes == 0 {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: "RMSNorm requires its FP32 row-sum workspace".into(),
            });
        }
        selected.push(SelectedNode {
            node: node.id,
            descriptor,
            workspace_logical_bytes,
        });
    }
    let image = selected[0].descriptor.image_sha256;
    if selected
        .iter()
        .any(|node| node.descriptor.image_sha256 != image)
    {
        return Err(Error::UnsupportedKernel {
            operation: "graph",
            detail: "the initial chain must resolve from one trusted image".into(),
        });
    }

    // Base lowering supplies checked concrete shapes and byte extents. It is
    // deliberately called only after every semantic node was selected.
    let base = lower(graph, workload)?;
    let mut values = Vec::new();
    let mut weight_cursor = 0u64;
    for binding in base.bindings() {
        if let ValueBinding::ExternalWeight(weight) = binding {
            let physical = align_up(weight.required_bytes)?;
            let offset = weight_cursor;
            weight_cursor = checked_add(weight_cursor, physical, "weight region")?;
            values.push(PlannedValue {
                value: weight.value,
                role: weight.role,
                shape: weight.shape.clone(),
                region: StorageRegion::Weights,
                slot: values.len() as u32,
                offset,
                logical_bytes: weight.required_bytes,
                physical_bytes: physical,
                first_stage: 0,
                last_stage: 4,
            });
        }
    }
    let weight_region_bytes = weight_cursor;

    #[derive(Debug)]
    struct Pending {
        value: ValueId,
        role: ValueRole,
        shape: Vec<u64>,
        bytes: u64,
        first: u32,
        last: u32,
        slot: usize,
    }
    let mut pending = Vec::new();
    for binding in base.bindings() {
        match binding {
            ValueBinding::ExternalInput(input) => pending.push(Pending {
                value: input.value,
                role: input.role,
                shape: input.shape.clone(),
                bytes: input.required_bytes,
                first: 0,
                last: 3,
                slot: usize::MAX,
            }),
            ValueBinding::ArenaTensor(tensor) => {
                let (first, last) = match tensor.value {
                    value if value == graph.nodes()[0].output => (0, 2),
                    value if value == graph.nodes()[1].output => (2, 3),
                    value if value == graph.nodes()[2].output => (3, 4),
                    _ => {
                        return Err(invalid(
                            "graph",
                            "unexpected produced value in selected chain",
                        ));
                    }
                };
                pending.push(Pending {
                    value: tensor.value,
                    role: tensor.role,
                    shape: tensor.shape.clone(),
                    bytes: tensor.bytes,
                    first,
                    last,
                    slot: usize::MAX,
                });
            }
            ValueBinding::ExternalWeight(_) => {}
        }
    }
    pending.sort_by_key(|value| (value.first, value.value));
    let mut slots: Vec<(u64, u32)> = Vec::new();
    for value in &mut pending {
        let slot = slots
            .iter()
            .position(|(_, available)| *available < value.first)
            .unwrap_or(slots.len());
        if slot == slots.len() {
            slots.push((value.bytes, value.last));
        } else {
            slots[slot].0 = slots[slot].0.max(value.bytes);
            slots[slot].1 = value.last;
        }
        value.slot = slot;
    }
    let mut activation_offsets = Vec::new();
    let mut activation_cursor = 0u64;
    for (bytes, _) in &slots {
        activation_offsets.push(activation_cursor);
        activation_cursor = checked_add(activation_cursor, align_up(*bytes)?, "activation region")?;
    }
    let activation_region_bytes = activation_cursor;
    for value in pending {
        let physical = align_up(slots[value.slot].0)?;
        values.push(PlannedValue {
            value: value.value,
            role: value.role,
            shape: value.shape,
            region: StorageRegion::Activations,
            slot: value.slot as u32,
            offset: checked_add(
                weight_region_bytes,
                activation_offsets[value.slot],
                "activation offset",
            )?,
            logical_bytes: value.bytes,
            physical_bytes: physical,
            first_stage: value.first,
            last_stage: value.last,
        });
    }
    values.sort_by_key(|value| value.value);

    let workspace_logical_bytes = selected
        .iter()
        .map(|node| node.workspace_logical_bytes)
        .max()
        .unwrap_or(0);
    let workspace_region_bytes = align_up(workspace_logical_bytes)?;
    let workspace_offset = checked_add(
        weight_region_bytes,
        activation_region_bytes,
        "workspace offset",
    )?;
    let combined_arena_bytes =
        checked_add(workspace_offset, workspace_region_bytes, "combined arena")?;
    let workspace = PlannedWorkspace {
        node: graph.nodes()[1].id,
        offset: workspace_offset,
        logical_bytes: workspace_logical_bytes,
        physical_bytes: workspace_region_bytes,
        first_stage: 1,
        last_stage: 2,
    };
    let stages = [
        "linear",
        "rms-reduce",
        "rms-apply",
        "residual",
        "terminal-output",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    Ok(SelectedPlanCandidate {
        base,
        device_sm: (capability.compute_major, capability.compute_minor),
        catalogue_digest: catalogue.digest(),
        nodes: selected,
        values,
        workspace,
        weight_region_bytes,
        activation_region_bytes,
        workspace_region_bytes,
        combined_arena_bytes,
        stages,
    })
}

fn semantic(op: Op) -> Option<SemanticKernelOp> {
    match op {
        Op::Linear => Some(SemanticKernelOp::Linear),
        Op::RmsNorm => Some(SemanticKernelOp::RmsNorm),
        Op::Residual => Some(SemanticKernelOp::Residual),
        _ => None,
    }
}

fn operand(role: ValueRole) -> Result<KernelOperand, Error> {
    match role {
        ValueRole::Activation(value) => Ok(KernelOperand::Activation(value)),
        ValueRole::Weight(value) => Ok(KernelOperand::Weight(value)),
        ValueRole::Index(_) => Err(Error::UnsupportedKernel {
            operation: "index",
            detail: "the BF16 chain has no index operand".into(),
        }),
        ValueRole::Route { .. } => Err(Error::UnsupportedKernel {
            operation: "route",
            detail: "the BF16 chain has no routed operand".into(),
        }),
    }
}

fn node_shape(node: &moxie_graph::Node, rows: u64) -> Result<(u64, u64), Error> {
    match node.params {
        moxie_graph::OpParams::Linear {
            in_features,
            out_features,
            bias,
        } => {
            if bias {
                return Err(Error::UnsupportedKernel {
                    operation: "linear",
                    detail: "the selected descriptor has no bias".into(),
                });
            }
            Ok((in_features, out_features))
        }
        moxie_graph::OpParams::RmsNorm { hidden, group, .. } => {
            // The qualified device norm reduces over the whole row. A grouped
            // norm is a different reduction, and running it on this kernel
            // would silently mix every group's magnitude together.
            if group != 1 {
                return Err(Error::UnsupportedKernel {
                    operation: "rms_norm",
                    detail: format!("no qualified device kernel normalizes {group} groups"),
                });
            }
            Ok((hidden, hidden))
        }
        moxie_graph::OpParams::Residual { scale } => {
            // The qualified device residual adds and rounds; it does not scale.
            // A scaled residual is a different kernel, and until one is
            // qualified this must refuse rather than drop the factor -- which
            // would be a silent numerical change, not a fallback.
            if scale != 1.0 {
                return Err(Error::UnsupportedKernel {
                    operation: "residual",
                    detail: format!(
                        "no qualified device kernel applies a residual scale; got {scale}"
                    ),
                });
            }
            if rows == 0 {
                return Err(invalid("rows", "residual rows must be nonzero"));
            }
            Ok((1, 1))
        }
        _ => Err(Error::UnsupportedKernel {
            operation: node.params.op().name(),
            detail: "operation is outside the selected package".into(),
        }),
    }
}

fn align_up(value: u64) -> Result<u64, Error> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|sum| sum & !(ALIGNMENT - 1))
        .ok_or_else(|| invalid("alignment", "physical slot alignment overflowed"))
}

fn checked_add(a: u64, b: u64, field: &'static str) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid(field, "byte extent overflowed"))
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use moxie_graph::{
        GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
    };
    use moxie_types::{
        AccumulationPolicy, ActivationPrecision, DeviceUuid, Dim, KernelCatalogue, KernelId,
        KernelOperand, KernelShapeBounds, KernelSymbol, Precision, RoundingProfile,
        SemanticKernelDescriptor, SemanticKernelOp, SmVersion, SymbolId, WeightPrecision,
        WorkspaceExpression,
    };

    use super::*;
    use crate::Phase;

    const ROWS: SymbolId = SymbolId(1200);
    const ORACLE: OracleId = OracleId("selected-plan-tests");

    fn graph(hidden: u64, eps: f32) -> Graph {
        graph_edges(hidden, eps, true)
    }

    fn graph_edges(hidden: u64, eps: f32, exact_residual: bool) -> Graph {
        let mut registry = OracleRegistry::new();
        for op in [Op::Linear, Op::RmsNorm, Op::Residual] {
            registry
                .register(
                    op,
                    ORACLE,
                    OracleEvidence {
                        implementation: "moxie_plan::selected::tests",
                        test_module: "moxie_plan::selected::tests",
                    },
                )
                .unwrap();
        }
        let activation = |shape| {
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        let weight = |shape| {
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let x = builder.input(
            "x",
            activation(vec![Dim::symbol(ROWS), Dim::constant(hidden)]),
        );
        let w = builder
            .weight(
                "weight",
                weight(vec![Dim::constant(hidden), Dim::constant(hidden)]),
            )
            .unwrap();
        let h = builder
            .node(
                OpParams::Linear {
                    in_features: hidden,
                    out_features: hidden,
                    bias: false,
                },
                &[x, w],
            )
            .unwrap();
        let gain = builder
            .weight("gain", weight(vec![Dim::constant(hidden)]))
            .unwrap();
        let n = builder
            .node(
                OpParams::RmsNorm {
                    hidden,
                    eps,
                    group: 1,
                },
                &[h, gain],
            )
            .unwrap();
        let residual_left = if exact_residual { x } else { h };
        let y = builder
            .node(OpParams::Residual { scale: 1.0 }, &[residual_left, n])
            .unwrap();
        builder.finish(y, &registry).unwrap()
    }

    fn biased_graph(hidden: u64) -> Graph {
        let mut registry = OracleRegistry::new();
        for op in [Op::Linear, Op::RmsNorm, Op::Residual] {
            registry
                .register(
                    op,
                    ORACLE,
                    OracleEvidence {
                        implementation: "moxie_plan::selected::tests",
                        test_module: "moxie_plan::selected::tests",
                    },
                )
                .unwrap();
        }
        let activation = |shape| {
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        let weight = |shape| {
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let x = builder.input(
            "x",
            activation(vec![Dim::symbol(ROWS), Dim::constant(hidden)]),
        );
        let w = builder
            .weight(
                "weight",
                weight(vec![Dim::constant(hidden), Dim::constant(hidden)]),
            )
            .unwrap();
        let bias = builder
            .weight("bias", weight(vec![Dim::constant(hidden)]))
            .unwrap();
        let h = builder
            .node(
                OpParams::Linear {
                    in_features: hidden,
                    out_features: hidden,
                    bias: true,
                },
                &[x, w, bias],
            )
            .unwrap();
        let gain = builder
            .weight("gain", weight(vec![Dim::constant(hidden)]))
            .unwrap();
        let n = builder
            .node(
                OpParams::RmsNorm {
                    hidden,
                    eps: 1e-5,
                    group: 1,
                },
                &[h, gain],
            )
            .unwrap();
        let y = builder
            .node(OpParams::Residual { scale: 1.0 }, &[x, n])
            .unwrap();
        builder.finish(y, &registry).unwrap()
    }

    fn descriptor(op: SemanticKernelOp, sm: SmVersion) -> SemanticKernelDescriptor {
        let inputs = match op {
            SemanticKernelOp::Linear | SemanticKernelOp::RmsNorm => vec![
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
            ],
            SemanticKernelOp::Residual => vec![
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
            ],
            // Not selectable by `lower_selected`, which is task 0012's exact
            // three-node chain. The grouped expert operand list is task 0021's
            // and is built where that plan is built.
            SemanticKernelOp::ExpertMlp(_) => vec![
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::RouteIndex,
                KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
                KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
            ],
        };
        SemanticKernelDescriptor {
            id: KernelId(format!("{}-{}", op.name(), sm.name())),
            abi_version: 1,
            operation: op,
            inputs,
            output: ActivationPrecision::expect(Precision::Bf16),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            rounding: RoundingProfile::FinalBf16Rne,
            layout: TensorLayout::ContiguousRowMajorV1,
            shape: KernelShapeBounds {
                max_rows: 64,
                max_input: 1024,
                max_output: 1024,
            },
            sm,
            workspace: if op == SemanticKernelOp::RmsNorm {
                WorkspaceExpression::RowsTimesF32
            } else {
                WorkspaceExpression::Zero
            },
            image_sha256: [7; 32],
            symbols: vec![KernelSymbol(op.name().to_string())],
        }
    }

    fn catalogue(sm: SmVersion) -> KernelCatalogue {
        KernelCatalogue::new(
            [
                SemanticKernelOp::Linear,
                SemanticKernelOp::RmsNorm,
                SemanticKernelOp::Residual,
            ]
            .into_iter()
            .map(|op| descriptor(op, sm))
            .collect(),
        )
        .unwrap()
    }

    fn capability(sm: SmVersion) -> DeviceCapability {
        DeviceCapability {
            ordinal: 99,
            uuid: DeviceUuid::from_bytes([9; 16]),
            name: "synthetic".into(),
            compute_major: sm.major,
            compute_minor: sm.minor,
            total_memory_bytes: 4096,
            multiprocessor_count: 1,
            pci_bus_id: "synthetic".into(),
            peer_access: Vec::new(),
        }
    }

    fn workload(graph: &Graph, rows: u64) -> ResourceWorkload {
        ResourceWorkload {
            phase: if rows == 1 {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            rows,
            visible_tokens: if rows == 5 { 32_768 } else { 1 },
            branch_rows: rows,
            output: graph.output(),
            device: DeviceUuid::from_bytes([9; 16]),
        }
    }

    #[test]
    fn exact_regions_for_both_required_shapes_and_both_sms() {
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            for (rows, hidden, expected_weights, expected_total) in
                [(1, 8, 512, 1536), (5, 17, 1024, 2048)]
            {
                let graph = graph(hidden, if hidden == 8 { 3.5 } else { 1e-5 });
                let plan = lower_selected(
                    &graph,
                    workload(&graph, rows),
                    &capability(sm),
                    &catalogue(sm),
                )
                .unwrap();
                assert_eq!(plan.nodes().len(), 3);
                assert_eq!(plan.weight_region_bytes(), expected_weights);
                assert_eq!(plan.activation_region_bytes(), 768);
                assert_eq!(plan.workspace().logical_bytes, rows * 4);
                assert_eq!(plan.workspace_region_bytes(), 256);
                assert_eq!(plan.combined_arena_bytes(), expected_total);
                let x = plan.value(graph.inputs()[0]).unwrap();
                let h = plan.value(graph.nodes()[0].output).unwrap();
                let n = plan.value(graph.nodes()[1].output).unwrap();
                assert_ne!(x.slot, h.slot);
                assert_ne!(x.slot, n.slot);
                assert_ne!(h.slot, n.slot);
                assert_eq!(plan.value(graph.output()).unwrap().slot, h.slot);
            }
        }
    }

    #[test]
    fn selection_refuses_missing_duplicate_shape_sm_bias_and_zero_workspace() {
        let selected_graph = graph(8, 3.5);
        let cap = capability(SmVersion::SM86);
        let mut descriptors = catalogue(SmVersion::SM86).descriptors().to_vec();
        descriptors.retain(|value| value.operation != SemanticKernelOp::Residual);
        assert_eq!(
            lower_selected(
                &selected_graph,
                workload(&selected_graph, 1),
                &cap,
                &KernelCatalogue::new(descriptors).unwrap()
            )
            .unwrap_err()
            .kind(),
            "unsupported_kernel"
        );
        let wrong_sm = catalogue(SmVersion::SM120);
        assert_eq!(
            lower_selected(
                &selected_graph,
                workload(&selected_graph, 1),
                &cap,
                &wrong_sm,
            )
            .unwrap_err()
            .kind(),
            "unsupported_kernel"
        );

        let big = graph(1025, 1e-5);
        assert_eq!(
            lower_selected(&big, workload(&big, 1), &cap, &catalogue(SmVersion::SM86))
                .unwrap_err()
                .kind(),
            "unsupported_kernel"
        );

        let mut zero = catalogue(SmVersion::SM86).descriptors().to_vec();
        zero.iter_mut()
            .find(|value| value.operation == SemanticKernelOp::RmsNorm)
            .unwrap()
            .workspace = WorkspaceExpression::Zero;
        assert_eq!(
            lower_selected(
                &selected_graph,
                workload(&selected_graph, 1),
                &cap,
                &KernelCatalogue::new(zero).unwrap()
            )
            .unwrap_err()
            .kind(),
            "unsupported_kernel"
        );

        let mut unexpected = catalogue(SmVersion::SM86).descriptors().to_vec();
        unexpected
            .iter_mut()
            .find(|value| value.operation == SemanticKernelOp::Linear)
            .unwrap()
            .workspace = WorkspaceExpression::RowsTimesF32;
        assert_eq!(
            lower_selected(
                &selected_graph,
                workload(&selected_graph, 1),
                &cap,
                &KernelCatalogue::new(unexpected).unwrap()
            )
            .unwrap_err()
            .kind(),
            "unsupported_kernel"
        );

        let mut duplicate = catalogue(SmVersion::SM86).descriptors().to_vec();
        let mut second = duplicate[0].clone();
        second.id = KernelId("other-id-same-key".into());
        duplicate.push(second);
        assert_eq!(
            KernelCatalogue::new(duplicate).unwrap_err().kind(),
            "invalid_artifact"
        );
    }

    #[test]
    fn catalogue_digest_is_order_independent_and_changes_with_identity() {
        let first = catalogue(SmVersion::SM86);
        let mut reversed = first.descriptors().to_vec();
        reversed.reverse();
        assert_eq!(
            first.digest(),
            KernelCatalogue::new(reversed).unwrap().digest()
        );
        let mut changed = first.descriptors().to_vec();
        changed[0].image_sha256[0] ^= 1;
        assert_ne!(
            first.digest(),
            KernelCatalogue::new(changed).unwrap().digest()
        );
    }

    #[test]
    fn every_selection_key_and_candidate_binding_bites() {
        let selected_graph = graph(8, 3.5);
        let cap = capability(SmVersion::SM86);
        let catalogue = catalogue(SmVersion::SM86);
        let plan = lower_selected(
            &selected_graph,
            workload(&selected_graph, 1),
            &cap,
            &catalogue,
        )
        .unwrap();
        let ids: std::collections::BTreeSet<_> = plan
            .nodes()
            .iter()
            .map(|node| node.descriptor.id.clone())
            .collect();
        assert_eq!(ids.len(), 3, "each semantic node has its own identity");

        let other_graph = graph(8, 3.5);
        assert!(!plan.matches(&other_graph, &cap, &catalogue));
        let mut other_cap = cap.clone();
        other_cap.uuid = DeviceUuid::from_bytes([8; 16]);
        assert!(!plan.matches(&selected_graph, &other_cap, &catalogue));
        let mut changed = catalogue.descriptors().to_vec();
        changed[0].symbols[0].0.push_str("-changed");
        let changed = KernelCatalogue::new(changed).unwrap();
        assert!(!plan.matches(&selected_graph, &cap, &changed));

        for mutate in [
            0usize, // operand role/precision
            1,      // accumulation profile
            2,      // output precision
            3,      // ABI
            4,      // empty symbols
        ] {
            let mut descriptors = catalogue.descriptors().to_vec();
            let linear = descriptors
                .iter_mut()
                .find(|value| value.operation == SemanticKernelOp::Linear)
                .unwrap();
            match mutate {
                0 => {
                    linear.inputs[0] =
                        KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16));
                }
                1 => linear.accumulation = AccumulationPolicy::F32,
                2 => linear.output = ActivationPrecision::expect(Precision::F32),
                3 => linear.abi_version = 2,
                4 => linear.symbols.clear(),
                _ => unreachable!(),
            }
            assert_eq!(
                lower_selected(
                    &selected_graph,
                    workload(&selected_graph, 1),
                    &cap,
                    &KernelCatalogue::new(descriptors).unwrap()
                )
                .unwrap_err()
                .kind(),
                "unsupported_kernel"
            );
        }

        let biased = biased_graph(8);
        assert_eq!(
            lower_selected(&biased, workload(&biased, 1), &cap, &catalogue)
                .unwrap_err()
                .kind(),
            "unsupported_kernel"
        );

        let wrong_edges = graph_edges(8, 3.5, false);
        assert_eq!(
            lower_selected(&wrong_edges, workload(&wrong_edges, 1), &cap, &catalogue)
                .unwrap_err()
                .kind(),
            "unsupported_kernel"
        );
    }

    /// A graph whose feed-forward is routed, for the refusal below.
    fn routed_graph(hidden: u64) -> Graph {
        let mut registry = OracleRegistry::new();
        for op in [Op::Linear, Op::Route, Op::ExpertMlp, Op::Combine] {
            registry
                .register(
                    op,
                    ORACLE,
                    OracleEvidence {
                        implementation: "moxie_plan::selected::tests",
                        test_module: "moxie_plan::selected::tests",
                    },
                )
                .unwrap();
        }
        let activation = |shape| {
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        let weight = |shape| {
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                shape,
            )
        };
        const EXPERTS: u64 = 2;
        const TOP_K: u64 = 1;
        const INTERMEDIATE: u64 = 3;
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let x = builder.input(
            "x",
            activation(vec![Dim::symbol(ROWS), Dim::constant(hidden)]),
        );
        let gain = builder
            .weight("router gain", weight(vec![Dim::constant(hidden)]))
            .unwrap();
        let proj = builder
            .weight(
                "router projection",
                weight(vec![Dim::constant(EXPERTS), Dim::constant(hidden)]),
            )
            .unwrap();
        let route = builder
            .node(
                OpParams::Route {
                    hidden,
                    experts: EXPERTS,
                    top_k: TOP_K,
                    eps: 1e-6,
                    input_scale: 1.0,
                    per_expert_scale: false,
                },
                &[x, gain, proj],
            )
            .unwrap();
        let gate_up = builder
            .weight(
                "fused gate/up",
                weight(vec![
                    Dim::constant(EXPERTS),
                    Dim::constant(2 * INTERMEDIATE),
                    Dim::constant(hidden),
                ]),
            )
            .unwrap();
        let down = builder
            .weight(
                "fused down",
                weight(vec![
                    Dim::constant(EXPERTS),
                    Dim::constant(hidden),
                    Dim::constant(INTERMEDIATE),
                ]),
            )
            .unwrap();
        let slots = builder
            .node(
                OpParams::ExpertMlp {
                    hidden,
                    intermediate: INTERMEDIATE,
                    experts: EXPERTS,
                    top_k: TOP_K,
                    activation: moxie_graph::ExpertActivation::GeGlu,
                },
                &[x, route, gate_up, down],
            )
            .unwrap();
        let y = builder
            .node(
                OpParams::Combine {
                    hidden,
                    top_k: TOP_K,
                    order: moxie_graph::CombineOrder::AscendingExpertId,
                },
                &[route, slots],
            )
            .unwrap();
        builder.finish(y, &registry).unwrap()
    }

    #[test]
    fn the_selected_chain_refuses_routed_operations() {
        // Task 0019 adds routing to the shared catalogue and **no** device
        // kernel for it. The qualified BF16 chain must say so rather than
        // acquire a routed path by falling through: a routed step that
        // "succeeded" on a chain with no expert dispatch would be a silent
        // wrong answer, not a fallback.
        let graph = routed_graph(8);
        let cap = capability(SmVersion::SM86);
        let catalogue = catalogue(SmVersion::SM86);
        let error = lower_selected(&graph, workload(&graph, 1), &cap, &catalogue).unwrap_err();
        assert_eq!(error.kind(), "unsupported_kernel", "{error}");

        // And the partition rules stay closed: routing is replicated by
        // requirement, expert compute and combination are undetermined until
        // M5 decides where an expert lives.
        assert_eq!(
            OpParams::Route {
                hidden: 8,
                experts: 2,
                top_k: 1,
                eps: 1e-6,
                input_scale: 1.0,
                per_expert_scale: false,
            }
            .partition_rule(),
            moxie_graph::PartitionRule::Replicated
        );
        for params in [
            OpParams::ExpertMlp {
                hidden: 8,
                intermediate: 3,
                experts: 2,
                top_k: 1,
                activation: moxie_graph::ExpertActivation::GeGlu,
            },
            OpParams::Combine {
                hidden: 8,
                top_k: 1,
                order: moxie_graph::CombineOrder::AscendingExpertId,
            },
        ] {
            assert_eq!(
                params.partition_rule(),
                moxie_graph::PartitionRule::NotDetermined,
                "{} must fail closed until M5",
                params.op().name()
            );
        }
    }

    #[test]
    fn workspace_overflow_is_pure_and_precedes_plan_identity() {
        let graph = graph(8, 3.5);
        let cap = capability(SmVersion::SM86);
        let mut descriptors = catalogue(SmVersion::SM86).descriptors().to_vec();
        for descriptor in &mut descriptors {
            descriptor.shape.max_rows = u64::MAX;
        }
        assert_eq!(
            lower_selected(
                &graph,
                ResourceWorkload {
                    rows: u64::MAX,
                    branch_rows: u64::MAX,
                    visible_tokens: u64::MAX,
                    ..workload(&graph, 1)
                },
                &cap,
                &KernelCatalogue::new(descriptors).unwrap(),
            )
            .unwrap_err()
            .kind(),
            "invalid_request"
        );
    }
}
