//! Pure semantic-kernel selection and exact combined-arena lowering.

use std::collections::{BTreeMap, BTreeSet};

use moxie_graph::{
    CombineReductionOrder, ExpertOwnership, LinearReductionOrder, NodeId, Op, OpParams, ValueId,
    ValueRole,
};
use moxie_types::{
    AccumulationPolicy, DeviceCapability, Error, KernelCatalogue, KernelOperand,
    SemanticKernelDescriptor, SemanticKernelOp, TensorLayout, WeightPrecision,
};

use crate::expert::WeightFormat;
use crate::{Graph, PlanCandidate, ResourceWorkload, ValueBinding, lower};
use crate::{HostExpertJoin, HostExpertLowering};

const ALIGNMENT: u64 = 256;
const BLAS_WORKSPACE_BYTES: u64 = 32 * 1024 * 1024;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedPackage {
    Chain,
    Attention,
    Dense,
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
    package: SelectedPackage,
    host_workspace_bytes: u64,
    rope_table_offsets: BTreeMap<(u64, u64, u32), u64>,
    blas_workspace_offset: Option<u64>,
    linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: BTreeMap<NodeId, CombineReductionOrder>,
    expert_ownership: BTreeMap<NodeId, ExpertOwnership>,
    host_expert_joins: BTreeMap<NodeId, HostExpertJoin>,
    weight_formats: BTreeMap<ValueId, WeightFormat>,
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
    pub const fn is_dense(&self) -> bool {
        matches!(self.package, SelectedPackage::Dense)
    }
    pub const fn host_workspace_bytes(&self) -> u64 {
        self.host_workspace_bytes
    }
    pub fn rope_table_offsets(&self) -> &BTreeMap<(u64, u64, u32), u64> {
        &self.rope_table_offsets
    }
    pub const fn blas_workspace_offset(&self) -> Option<u64> {
        self.blas_workspace_offset
    }
    pub const fn blas_workspace_bytes(&self) -> Option<u64> {
        if self.blas_workspace_offset.is_some() {
            Some(BLAS_WORKSPACE_BYTES)
        } else {
            None
        }
    }
    pub fn linear_orders(&self) -> &BTreeMap<NodeId, LinearReductionOrder> {
        &self.linear_orders
    }
    pub fn combine_orders(&self) -> &BTreeMap<NodeId, CombineReductionOrder> {
        &self.combine_orders
    }
    pub fn expert_ownership(&self) -> &BTreeMap<NodeId, ExpertOwnership> {
        &self.expert_ownership
    }
    pub fn host_expert_joins(&self) -> &BTreeMap<NodeId, HostExpertJoin> {
        &self.host_expert_joins
    }
    pub fn weight_formats(&self) -> &BTreeMap<ValueId, WeightFormat> {
        &self.weight_formats
    }
    pub fn is_paged_attention(&self) -> bool {
        matches!(self.nodes.as_slice(), [node] if node.descriptor.operation == SemanticKernelOp::PagedAttention)
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

/// Split a prefill into greedy chunks from a validated ascending bucket set.
pub fn prefill_chunks(rows: u64, buckets: &[u64]) -> Result<Vec<u64>, Error> {
    if buckets.is_empty()
        || buckets.contains(&0)
        || !buckets.contains(&1)
        || buckets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "buckets",
            "buckets must be nonzero, strictly ascending, and contain one",
        ));
    }
    if rows == 0 {
        return Err(invalid("rows", "prefill rows must be nonzero"));
    }

    let mut chunks = Vec::new();
    let mut remaining = rows;
    while remaining != 0 {
        let chunk = buckets
            .iter()
            .rev()
            .find(|&&bucket| bucket <= remaining)
            .copied()
            .ok_or_else(|| invalid("buckets", "no bucket fits the remaining rows"))?;
        chunks
            .try_reserve(1)
            .map_err(|_| invalid("buckets", "chunk list could not be reserved"))?;
        chunks.push(chunk);
        remaining -= chunk;
    }
    Ok(chunks)
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
    if matches!(graph.nodes(), [node] if matches!(node.params, moxie_graph::OpParams::Attention { .. }))
    {
        return lower_attention(graph, workload, capability, catalogue);
    }
    let expected = [Op::Linear, Op::RmsNorm, Op::Residual];
    if graph.nodes().len() != expected.len()
        || graph
            .nodes()
            .iter()
            .zip(expected)
            .any(|(node, op)| node.params.op() != op)
    {
        return lower_dense(graph, workload, capability, catalogue);
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
        return lower_dense(graph, workload, capability, catalogue);
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
        package: SelectedPackage::Chain,
        host_workspace_bytes: 0,
        rope_table_offsets: BTreeMap::new(),
        blas_workspace_offset: None,
        linear_orders: BTreeMap::new(),
        combine_orders: BTreeMap::new(),
        expert_ownership: BTreeMap::new(),
        host_expert_joins: BTreeMap::new(),
        weight_formats: BTreeMap::new(),
    })
}

/// Lower a dense graph whose row-parallel `Linear`s carry a declared
/// reduction order (ADR 0036). An order without a slice is the single-device
/// reference: the split-aware kernel sums every declared block and rounds
/// once. An order with a slice is one rank's stage: the partial kernel writes
/// FP32 for the tensor-parallel reduce to combine. A stage need not contain
/// the whole graph.
pub fn lower_selected_ordered(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
    expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
) -> Result<SelectedPlanCandidate, Error> {
    if capability.uuid != workload.device {
        return Err(invalid(
            "device",
            "workload UUID and measured capability differ",
        ));
    }
    for (id, order) in orders {
        let Some(moxie_graph::OpParams::Linear {
            in_features,
            bias: false,
            ..
        }) = graph
            .nodes()
            .get(id.0 as usize)
            .filter(|node| node.id == *id)
            .map(|node| &node.params)
        else {
            return Err(invalid(
                "linear_order",
                "an order names no unbiased Linear of this graph",
            ));
        };
        let blocks = u64::from(order.blocks);
        // The interpreter's rule (`validate_linear_slice`): a slice is one
        // whole declared block of the full input axis.
        let fits = blocks > 0
            && match order.slice {
                None => in_features.is_multiple_of(blocks),
                Some(slice) => {
                    slice.width == *in_features
                        && slice.width > 0
                        && slice.full_width.is_multiple_of(blocks)
                        && slice.width == slice.full_width / blocks
                        && slice.first <= slice.full_width - slice.width
                        && slice.first.is_multiple_of(slice.width)
                }
            };
        if !fits {
            return Err(invalid(
                "linear_order",
                "the declared order does not divide this Linear's input axis",
            ));
        }
    }
    for (id, order) in combine_orders {
        if !matches!(
            graph
                .nodes()
                .get(id.0 as usize)
                .filter(|node| node.id == *id)
                .map(|node| &node.params),
            Some(OpParams::Combine { .. })
        ) || order.groups == 0
            || order.owned.is_some_and(|owned| owned >= order.groups)
        {
            return Err(invalid(
                "combine_order",
                "combine order names no Combine or has an invalid owner group",
            ));
        }
    }
    for (id, ownership) in expert_ownership {
        if !matches!(
            graph
                .nodes()
                .get(id.0 as usize)
                .filter(|node| node.id == *id)
                .map(|node| &node.params),
            Some(OpParams::ExpertMlp { .. })
        ) || ownership.groups == 0
            || ownership.owned >= ownership.groups
        {
            return Err(invalid(
                "expert_ownership",
                "expert ownership names no ExpertMlp or has an invalid owner group",
            ));
        }
    }
    lower_dense_mode(
        graph,
        workload,
        capability,
        catalogue,
        false,
        orders,
        combine_orders,
        expert_ownership,
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
}

/// Select the device half of a host-owned expert graph and its ordered host join.
pub fn lower_selected_host_experts(
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
    lowering: &HostExpertLowering,
) -> Result<SelectedPlanCandidate, Error> {
    if capability.uuid != workload.device {
        return Err(invalid(
            "device",
            "workload UUID and measured capability differ",
        ));
    }
    let graph = &lowering.device_stage().graph;
    let part = lowering.device();
    for (combine_id, join) in lowering.joins() {
        let Some(combine) = graph
            .nodes()
            .get(combine_id.0 as usize)
            .filter(|node| node.id == *combine_id)
            .filter(|node| matches!(node.params, OpParams::Combine { .. }))
        else {
            return Err(invalid("host_experts", "a host join names no Combine node"));
        };
        if part.combine_orders.get(combine_id)
            != Some(&CombineReductionOrder {
                groups: 2,
                owned: Some(0),
            })
        {
            return Err(invalid(
                "host_experts",
                "a host join disagrees with its device combine order",
            ));
        }
        let Some(expert_node) = combine
            .inputs
            .get(1)
            .and_then(|slots| graph.producer(*slots))
        else {
            return Err(invalid(
                "host_experts",
                "a host join has no ExpertMlp producer",
            ));
        };
        let OpParams::ExpertMlp { experts, .. } = expert_node.params else {
            return Err(invalid(
                "host_experts",
                "a host join's slots are not produced by ExpertMlp",
            ));
        };
        if experts != u64::from(join.host_experts())
            || experts != u64::from(join.first_host_expert())
        {
            return Err(invalid(
                "host_experts",
                "a host join disagrees with its local ExpertMlp extent",
            ));
        }
    }
    lower_dense_mode(
        graph,
        workload,
        capability,
        catalogue,
        true,
        &BTreeMap::new(),
        &part.combine_orders,
        &part.expert_ownership,
        lowering.joins(),
        &BTreeMap::new(),
    )
}

fn lower_attention(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
) -> Result<SelectedPlanCandidate, Error> {
    let node = &graph.nodes()[0];
    let moxie_graph::OpParams::Attention {
        heads,
        kv_heads: _,
        head_dim,
        ..
    } = node.params
    else {
        unreachable!("caller selected an attention node")
    };
    if !graph.weights().is_empty()
        || graph.inputs().len() != 4
        || node.inputs.as_slice() != graph.inputs()
        || node.output != graph.output()
    {
        return Err(Error::UnsupportedKernel {
            operation: "graph",
            detail: "paged attention requires exact query/key/value/position inputs".into(),
        });
    }

    let bf16 = KernelOperand::Activation(moxie_types::ActivationPrecision::expect(
        moxie_types::Precision::Bf16,
    ));
    let operands = [bf16, bf16, bf16, KernelOperand::PageIndex];
    let mut selected = None;
    let mut matched = 0usize;
    for descriptor in catalogue.descriptors().iter().filter(|descriptor| {
        descriptor.operation == SemanticKernelOp::PagedAttention
            && descriptor.inputs.as_slice() == operands
            && descriptor.output == node.contract.output
            && descriptor.accumulation == node.contract.accumulation
            && descriptor.rounding == moxie_types::RoundingProfile::FinalBf16Rne
            && descriptor.layout == TensorLayout::ContiguousRowMajorV1
            && descriptor.workspace == moxie_types::WorkspaceExpression::Zero
            && descriptor.sm.major == capability.compute_major
            && descriptor.sm.minor == capability.compute_minor
            && workload.rows <= descriptor.shape.max_rows
            && head_dim <= descriptor.shape.max_input
            && head_dim <= descriptor.shape.max_output
            && descriptor.symbols.len() == 1
    }) {
        matched += 1;
        selected.get_or_insert(descriptor);
    }
    if matched != 1 {
        return Err(Error::UnsupportedKernel {
            operation: "paged_attention",
            detail: format!(
                "expected one descriptor for {} row(s), {heads} head(s) of width {head_dim} on sm_{}{}; found {}",
                workload.rows, capability.compute_major, capability.compute_minor, matched
            ),
        });
    }
    let descriptor = selected.expect("one selected descriptor").clone();
    if descriptor.abi_version != 1 {
        return Err(Error::UnsupportedKernel {
            operation: "paged_attention",
            detail: "the selected descriptor ABI is unsupported".into(),
        });
    }

    let base = lower(graph, workload)?;
    let query = match base.binding(node.inputs[0]) {
        Some(ValueBinding::ExternalInput(value)) => value,
        _ => return Err(invalid("query", "attention query is not an external input")),
    };
    let output = match base.binding(node.output) {
        Some(ValueBinding::ArenaTensor(value)) => value,
        _ => return Err(invalid("output", "attention output has no arena tensor")),
    };
    let query_physical = align_up(query.required_bytes)?;
    let output_physical = align_up(output.bytes)?;
    let activation_region_bytes = checked_add(
        query_physical,
        output_physical,
        "attention activation region",
    )?;
    let values = vec![
        PlannedValue {
            value: query.value,
            role: query.role,
            shape: query.shape.clone(),
            region: StorageRegion::Activations,
            slot: 0,
            offset: 0,
            logical_bytes: query.required_bytes,
            physical_bytes: query_physical,
            first_stage: 0,
            last_stage: 0,
        },
        PlannedValue {
            value: output.value,
            role: output.role,
            shape: output.shape.clone(),
            region: StorageRegion::Activations,
            slot: 1,
            offset: query_physical,
            logical_bytes: output.bytes,
            physical_bytes: output_physical,
            first_stage: 0,
            last_stage: 1,
        },
    ];
    Ok(SelectedPlanCandidate {
        base,
        device_sm: (capability.compute_major, capability.compute_minor),
        catalogue_digest: catalogue.digest(),
        nodes: vec![SelectedNode {
            node: node.id,
            descriptor,
            workspace_logical_bytes: 0,
        }],
        values,
        workspace: PlannedWorkspace {
            node: node.id,
            offset: activation_region_bytes,
            logical_bytes: 0,
            physical_bytes: 0,
            first_stage: 0,
            last_stage: 0,
        },
        weight_region_bytes: 0,
        activation_region_bytes,
        workspace_region_bytes: 0,
        combined_arena_bytes: activation_region_bytes,
        stages: vec!["attention".into(), "terminal-output".into()],
        package: SelectedPackage::Attention,
        host_workspace_bytes: 0,
        rope_table_offsets: BTreeMap::new(),
        blas_workspace_offset: None,
        linear_orders: BTreeMap::new(),
        combine_orders: BTreeMap::new(),
        expert_ownership: BTreeMap::new(),
        host_expert_joins: BTreeMap::new(),
        weight_formats: BTreeMap::new(),
    })
}

/// Lower a complete dense graph with sidecar affine storage formats for its
/// selected linear weights. The graph continues to describe logical BF16
/// weights; formats affect kernel selection and the admitted byte extents.
pub fn lower_selected_with_formats(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
    formats: &BTreeMap<ValueId, WeightFormat>,
) -> Result<SelectedPlanCandidate, Error> {
    if capability.uuid != workload.device {
        return Err(invalid(
            "device",
            "workload UUID and measured capability differ",
        ));
    }
    lower_dense_mode(
        graph,
        workload,
        capability,
        catalogue,
        true,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        formats,
    )
}

fn lower_dense(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
) -> Result<SelectedPlanCandidate, Error> {
    lower_dense_mode(
        graph,
        workload,
        capability,
        catalogue,
        true,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
}

fn check_routed_edges(
    graph: &Graph,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
    expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
) -> Result<(), Error> {
    let unsupported = |detail| Error::UnsupportedKernel {
        operation: "route",
        detail,
    };
    for node in graph.nodes() {
        match node.params {
            OpParams::ExpertMlp { experts, top_k, .. } => {
                let Some(route_value) = node.inputs.get(1).copied() else {
                    return Err(unsupported("ExpertMlp is missing input[1] route".into()));
                };
                let Some(route_node) = graph.producer(route_value) else {
                    return Err(unsupported(
                        "ExpertMlp input[1] has no Route producer".into(),
                    ));
                };
                let OpParams::Route {
                    experts: route_experts,
                    top_k: route_top_k,
                    ..
                } = route_node.params
                else {
                    return Err(unsupported(
                        "ExpertMlp input[1] is not produced by Route".into(),
                    ));
                };
                let groups = expert_ownership
                    .get(&node.id)
                    .map_or(1, |ownership| u64::from(ownership.groups));
                if experts.checked_mul(groups) != Some(route_experts) || top_k != route_top_k {
                    return Err(unsupported(format!(
                        "ExpertMlp declares experts={experts} across {groups} owner group(s), top_k={top_k}; its Route producer declares experts={route_experts}, top_k={route_top_k}"
                    )));
                }
            }
            OpParams::Combine { top_k, .. } => {
                let (Some(route_value), Some(slots_value)) =
                    (node.inputs.first().copied(), node.inputs.get(1).copied())
                else {
                    return Err(unsupported(
                        "Combine is missing its route or slots input".into(),
                    ));
                };
                let Some(route_node) = graph.producer(route_value) else {
                    return Err(unsupported("Combine input[0] has no Route producer".into()));
                };
                match route_node.params {
                    OpParams::Route {
                        top_k: route_top_k, ..
                    } if route_top_k == top_k => {}
                    OpParams::Route {
                        top_k: route_top_k, ..
                    } => {
                        return Err(unsupported(format!(
                            "Combine declares top_k={top_k}; its Route producer declares top_k={route_top_k}"
                        )));
                    }
                    _ => {
                        return Err(unsupported(
                            "Combine input[0] is not produced by Route".into(),
                        ));
                    }
                }
                let Some(expert_node) = graph.producer(slots_value) else {
                    return Err(unsupported(
                        "Combine input[1] has no ExpertMlp producer".into(),
                    ));
                };
                let OpParams::ExpertMlp { experts, .. } = expert_node.params else {
                    return Err(unsupported(
                        "Combine input[1] is not produced by ExpertMlp".into(),
                    ));
                };
                if expert_node.inputs.get(1) != Some(&route_value) {
                    return Err(unsupported(
                        "Combine slots were produced from a different Route input".into(),
                    ));
                }
                let order = combine_orders.get(&node.id);
                let own = expert_ownership.get(&expert_node.id);
                let ownership_matches = match (order, own) {
                    (None, None) => true,
                    (Some(order), None) if order.owned.is_none() => {
                        order.groups > 0 && experts.is_multiple_of(u64::from(order.groups))
                    }
                    (Some(order), Some(own)) => {
                        order.owned == Some(own.owned) && order.groups == own.groups
                    }
                    _ => false,
                };
                if !ownership_matches {
                    return Err(unsupported(
                        "Combine reduction order disagrees with its ExpertMlp ownership".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // These are the graph's separate declared order maps.
fn lower_dense_mode(
    graph: &Graph,
    workload: ResourceWorkload,
    capability: &DeviceCapability,
    catalogue: &KernelCatalogue,
    require_complete_graph: bool,
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
    expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
    joins: &BTreeMap<NodeId, HostExpertJoin>,
    weight_formats: &BTreeMap<ValueId, WeightFormat>,
) -> Result<SelectedPlanCandidate, Error> {
    check_routed_edges(graph, combine_orders, expert_ownership)?;
    if require_complete_graph
        && (graph.nodes().is_empty()
            || !graph
                .nodes()
                .iter()
                .any(|node| node.params.op() == Op::Embedding)
            || !graph
                .nodes()
                .iter()
                .any(|node| node.params.op() == Op::VocabProjection)
            || !graph
                .nodes()
                .iter()
                .any(|node| node.params.op() == Op::Attention))
    {
        return Err(Error::UnsupportedKernel {
            operation: "graph",
            detail:
                "selected dense package requires Embedding, paged Attention and VocabProjection"
                    .into(),
        });
    }

    let mut formatted_weight_bytes = BTreeMap::new();
    for (&value, format) in weight_formats {
        if !graph.weights().contains(&value) {
            return Err(invalid(
                "weight_formats",
                "a format names a value that is not a graph weight",
            ));
        }
        if !matches!(
            graph.spec(value).map(|spec| spec.role),
            Some(ValueRole::Weight(precision))
                if precision == WeightPrecision::expect(moxie_types::Precision::Bf16)
        ) {
            return Err(invalid(
                "weight_formats",
                "formatted graph weights must have logical BF16 role",
            ));
        }
        if !matches!(format, WeightFormat::Affine { .. }) {
            return Err(invalid(
                "weight_formats",
                "a BF16 storage format is meaningless in the sidecar map",
            ));
        }
        let mut consumers = graph
            .nodes()
            .iter()
            .filter(|node| node.inputs.contains(&value));
        let Some(consumer) = consumers.next() else {
            return Err(invalid(
                "weight_formats",
                "a formatted weight must be consumed by one unbiased Linear",
            ));
        };
        if consumers.next().is_some() || consumer.inputs.get(1) != Some(&value) {
            return Err(invalid(
                "weight_formats",
                "a formatted weight must be consumed by one unbiased Linear as input 1",
            ));
        }
        let OpParams::Linear {
            in_features,
            out_features,
            bias: false,
        } = &consumer.params
        else {
            return Err(invalid(
                "weight_formats",
                "a formatted weight must be consumed by one unbiased Linear as input 1",
            ));
        };
        let Some(sections) = format.sections(*out_features, *in_features) else {
            return Err(invalid(
                "weight_formats",
                "the affine format is invalid for this Linear geometry",
            ));
        };
        formatted_weight_bytes.insert(value, sections.bytes);
    }

    let mut selected = Vec::new();
    let mut workspace_shared_bytes = 0u64;
    let mut host_workspace_bytes = 0u64;
    let mut rope_host_bytes = 0u64;
    let mut rope_table_bytes = BTreeMap::new();
    for node in graph.nodes() {
        let mut operation = dense_semantic(node)?;
        // `lower_selected_ordered` validated every order against its node.
        if let Some(order) = orders.get(&node.id) {
            operation = if order.slice.is_some() {
                SemanticKernelOp::LinearPartial
            } else {
                SemanticKernelOp::LinearSplit
            };
        }
        if joins.contains_key(&node.id) {
            operation = SemanticKernelOp::CombineHostJoin;
        } else if combine_orders
            .get(&node.id)
            .is_some_and(|order| order.owned.is_some())
        {
            operation = SemanticKernelOp::CombinePartial;
        }
        let mut roles = dense_operands(operation, node, graph)?;
        if matches!(node.params, OpParams::Linear { .. })
            && let Some(format) = node
                .inputs
                .get(1)
                .and_then(|weight| weight_formats.get(weight))
        {
            roles[1] = KernelOperand::Weight(WeightPrecision::expect(format.precision()));
        }
        let (input, output) = dense_shape(node)?;
        let matches: Vec<_> = catalogue
            .descriptors()
            .iter()
            .filter(|descriptor| {
                descriptor.operation == operation
                    && descriptor.inputs == roles
                    && descriptor.output
                        == if matches!(
                            operation,
                            SemanticKernelOp::LinearPartial
                                | SemanticKernelOp::CombinePartial
                                | SemanticKernelOp::Route
                        ) {
                            moxie_types::ActivationPrecision::expect(moxie_types::Precision::F32)
                        } else {
                            node.contract.output
                        }
                    && (descriptor.accumulation == node.contract.accumulation
                        || (operation == SemanticKernelOp::Linear
                            && node.contract.accumulation == AccumulationPolicy::Bf16InF32Acc
                            && descriptor.accumulation
                                == AccumulationPolicy::Bf16InF32AccUnordered))
                    && descriptor.rounding
                        == if matches!(
                            operation,
                            SemanticKernelOp::VocabProjection
                                | SemanticKernelOp::LinearPartial
                                | SemanticKernelOp::CombinePartial
                                | SemanticKernelOp::Route
                        ) {
                            moxie_types::RoundingProfile::Unrounded
                        } else {
                            moxie_types::RoundingProfile::FinalBf16Rne
                        }
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
                    "expected exactly one dense descriptor for rows={}, input={}, output={}, sm_{}{}; found {}",
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
        if descriptor.symbols.is_empty() {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: "dense descriptor has no ordered symbols".into(),
            });
        }
        let (workspace, workspace_bytes, host_bytes) =
            dense_workspace(graph, node, operation, workload.rows, joins)?;
        if descriptor.workspace != workspace {
            return Err(Error::UnsupportedKernel {
                operation: node.params.op().name(),
                detail: "dense descriptor workspace differs from the operation contract".into(),
            });
        }
        match node.params {
            OpParams::Rope {
                rotary_dim,
                frequency_dim,
                base,
                ..
            } => {
                let key = (rotary_dim, frequency_dim, base.to_bits());
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    rope_table_bytes.entry(key)
                {
                    entry.insert(workspace_bytes);
                    rope_host_bytes = rope_host_bytes
                        .checked_add(host_bytes)
                        .ok_or_else(|| invalid("host_workspace", "RoPE tables overflowed"))?;
                }
            }
            _ => {
                workspace_shared_bytes = workspace_shared_bytes.max(workspace_bytes);
                host_workspace_bytes = host_workspace_bytes.max(host_bytes);
            }
        }
        selected.push(SelectedNode {
            node: node.id,
            descriptor,
            workspace_logical_bytes: workspace_bytes,
        });
    }
    host_workspace_bytes = host_workspace_bytes
        .checked_add(rope_host_bytes)
        .ok_or_else(|| invalid("host_workspace", "dense host workspace overflowed"))?;
    let mut rope_table_offsets = BTreeMap::new();
    let mut workspace_logical_bytes = align_up(workspace_shared_bytes)?;
    for (key, table_bytes) in rope_table_bytes {
        rope_table_offsets.insert(key, workspace_logical_bytes);
        workspace_logical_bytes =
            checked_add(workspace_logical_bytes, align_up(table_bytes)?, "workspace")?;
    }
    let has_cublas_node = selected.iter().any(|node| {
        node.descriptor
            .symbols
            .iter()
            .any(|symbol| symbol.0.starts_with("cublas:"))
    });
    let blas_workspace_offset = if has_cublas_node {
        let offset = align_up(workspace_logical_bytes)?;
        workspace_logical_bytes = checked_add(offset, BLAS_WORKSPACE_BYTES, "cuBLAS workspace")?;
        Some(offset)
    } else {
        None
    };

    let base = lower(graph, workload)?;
    let last_stage = u32::try_from(graph.nodes().len())
        .map_err(|_| invalid("stages", "dense graph stage count exceeds u32"))?;
    let mut values = Vec::new();
    let mut weight_cursor = 0u64;
    let partial_outputs: BTreeSet<ValueId> = graph
        .nodes()
        .iter()
        .filter(|node| {
            orders
                .get(&node.id)
                .is_some_and(|order| order.slice.is_some())
                || (combine_orders
                    .get(&node.id)
                    .is_some_and(|order| order.owned.is_some())
                    && !joins.contains_key(&node.id))
        })
        .map(|node| node.output)
        .collect();
    for binding in base.bindings() {
        if let ValueBinding::ExternalWeight(weight) = binding {
            let logical_bytes = formatted_weight_bytes
                .get(&weight.value)
                .copied()
                .unwrap_or(weight.required_bytes);
            let physical = align_up(logical_bytes)?;
            let offset = weight_cursor;
            weight_cursor = checked_add(weight_cursor, physical, "weight region")?;
            values.push(PlannedValue {
                value: weight.value,
                role: weight.role,
                shape: weight.shape.clone(),
                region: StorageRegion::Weights,
                slot: values.len() as u32,
                offset,
                logical_bytes,
                physical_bytes: physical,
                first_stage: 0,
                last_stage,
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
                last: last_stage,
                slot: usize::MAX,
            }),
            ValueBinding::ArenaTensor(tensor) => {
                let role = if partial_outputs.contains(&tensor.value) {
                    ValueRole::Activation(moxie_types::ActivationPrecision::expect(
                        moxie_types::Precision::F32,
                    ))
                } else {
                    tensor.role
                };
                let bytes = if partial_outputs.contains(&tensor.value) {
                    tensor
                        .shape
                        .iter()
                        .try_fold(1u64, |total, extent| total.checked_mul(*extent))
                        .and_then(|elements| elements.checked_mul(4))
                        .ok_or_else(|| invalid("partial_linear", "partial extent overflowed"))?
                } else {
                    tensor.bytes
                };
                pending.push(Pending {
                    value: tensor.value,
                    role,
                    shape: tensor.shape.clone(),
                    bytes,
                    first: tensor.live.first,
                    last: tensor.live.last,
                    slot: usize::MAX,
                })
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

    let workspace_region_bytes = align_up(workspace_logical_bytes)?;
    let workspace_offset = checked_add(
        weight_region_bytes,
        activation_region_bytes,
        "workspace offset",
    )?;
    let combined_arena_bytes =
        checked_add(workspace_offset, workspace_region_bytes, "combined arena")?;
    let workspace_node = selected
        .iter()
        .find(|node| node.workspace_logical_bytes != 0)
        .map(|node| node.node)
        .unwrap_or(graph.nodes()[0].id);
    Ok(SelectedPlanCandidate {
        base,
        device_sm: (capability.compute_major, capability.compute_minor),
        catalogue_digest: catalogue.digest(),
        nodes: selected,
        values,
        workspace: PlannedWorkspace {
            node: workspace_node,
            offset: workspace_offset,
            logical_bytes: workspace_logical_bytes,
            physical_bytes: workspace_region_bytes,
            first_stage: 0,
            last_stage,
        },
        weight_region_bytes,
        activation_region_bytes,
        workspace_region_bytes,
        combined_arena_bytes,
        stages: graph
            .nodes()
            .iter()
            .map(|node| format!("node-{}-{}", node.id.0, node.params.op().name()))
            .chain(std::iter::once("terminal-output".into()))
            .collect(),
        package: SelectedPackage::Dense,
        host_workspace_bytes,
        rope_table_offsets,
        blas_workspace_offset,
        linear_orders: orders.clone(),
        combine_orders: combine_orders.clone(),
        expert_ownership: expert_ownership.clone(),
        host_expert_joins: joins.clone(),
        weight_formats: weight_formats.clone(),
    })
}

fn dense_semantic(node: &moxie_graph::Node) -> Result<SemanticKernelOp, Error> {
    match node.params {
        moxie_graph::OpParams::Embedding { .. } => Ok(SemanticKernelOp::Embedding),
        moxie_graph::OpParams::Linear { bias, .. } if !bias => Ok(SemanticKernelOp::Linear),
        moxie_graph::OpParams::Linear { .. } => Err(Error::UnsupportedKernel {
            operation: "linear",
            detail: "the dense BF16 package has no biased linear kernel".into(),
        }),
        moxie_graph::OpParams::RmsNorm { group: 1, .. } => Ok(SemanticKernelOp::RmsNorm),
        moxie_graph::OpParams::RmsNorm { .. } => Ok(SemanticKernelOp::GroupedRmsNorm),
        moxie_graph::OpParams::Rope { .. } => Ok(SemanticKernelOp::Rope),
        moxie_graph::OpParams::Attention { .. } => Ok(SemanticKernelOp::PagedAttention),
        moxie_graph::OpParams::GeGlu { .. } => Ok(SemanticKernelOp::GeGlu),
        moxie_graph::OpParams::Residual { scale: 1.0 } => Ok(SemanticKernelOp::Residual),
        moxie_graph::OpParams::Residual { .. } => Ok(SemanticKernelOp::ScaledResidual),
        moxie_graph::OpParams::VocabProjection { .. } => Ok(SemanticKernelOp::VocabProjection),
        moxie_graph::OpParams::Route {
            input: moxie_graph::RouterInput::Normalized { .. },
            score: moxie_graph::RouteScore::Softmax,
            per_expert_scale: true,
            selection_bias: false,
            coefficient: moxie_graph::RouteCoefficient::Fp32,
            ..
        } => Ok(SemanticKernelOp::Route),
        moxie_graph::OpParams::Route {
            input,
            score,
            per_expert_scale,
            selection_bias,
            coefficient,
            ..
        } => Err(Error::UnsupportedKernel {
            operation: "route",
            detail: format!(
                "unsupported input={input:?}, score={score:?}, per_expert_scale={per_expert_scale}, selection_bias={selection_bias}, coefficient={coefficient:?}"
            ),
        }),
        moxie_graph::OpParams::ExpertMlp {
            activation: moxie_graph::ExpertActivation::GeGlu,
            ..
        } => Ok(SemanticKernelOp::ExpertMlp(
            moxie_types::GateTransform::GeluTanh,
        )),
        moxie_graph::OpParams::ExpertMlp { activation, .. } => Err(Error::UnsupportedKernel {
            operation: "expert_mlp",
            detail: format!("unsupported activation={activation:?}"),
        }),
        moxie_graph::OpParams::Combine {
            order: moxie_graph::CombineOrder::AscendingExpertId,
            ..
        } => Ok(SemanticKernelOp::Combine),
        moxie_graph::OpParams::Combine { order, .. } => Err(Error::UnsupportedKernel {
            operation: "combine",
            detail: format!("unsupported order={order:?}"),
        }),
        _ => Err(Error::UnsupportedKernel {
            operation: node.params.op().name(),
            detail: "operation is outside the reduced dense device package".into(),
        }),
    }
}

fn dense_operands(
    operation: SemanticKernelOp,
    node: &moxie_graph::Node,
    graph: &Graph,
) -> Result<Vec<KernelOperand>, Error> {
    if operation == SemanticKernelOp::PagedAttention {
        return Ok(vec![
            KernelOperand::Activation(moxie_types::ActivationPrecision::expect(
                moxie_types::Precision::Bf16,
            )),
            KernelOperand::Activation(moxie_types::ActivationPrecision::expect(
                moxie_types::Precision::Bf16,
            )),
            KernelOperand::Activation(moxie_types::ActivationPrecision::expect(
                moxie_types::Precision::Bf16,
            )),
            KernelOperand::PageIndex,
        ]);
    }
    node.inputs
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let role = graph.values()[value.0 as usize].role;
            match role {
                ValueRole::Index(_) => {
                    Ok(if operation == SemanticKernelOp::Embedding && index == 0 {
                        KernelOperand::TokenIndex
                    } else {
                        KernelOperand::PositionIndex
                    })
                }
                ValueRole::Route { .. } => Ok(KernelOperand::RouteIndex),
                other => operand(other),
            }
        })
        .collect()
}

fn dense_shape(node: &moxie_graph::Node) -> Result<(u64, u64), Error> {
    let product = |a: u64, b: u64| {
        a.checked_mul(b)
            .ok_or_else(|| invalid("shape", "dense kernel shape overflowed"))
    };
    match node.params {
        moxie_graph::OpParams::Embedding { hidden, .. }
        | moxie_graph::OpParams::RmsNorm { hidden, .. }
        | moxie_graph::OpParams::VocabProjection { hidden, .. } => match node.params {
            moxie_graph::OpParams::VocabProjection { vocab, hidden, .. } => Ok((hidden, vocab)),
            _ => Ok((hidden, hidden)),
        },
        moxie_graph::OpParams::Linear {
            in_features,
            out_features,
            ..
        } => Ok((in_features, out_features)),
        moxie_graph::OpParams::Rope {
            heads, head_dim, ..
        } => {
            let width = product(heads, head_dim)?;
            Ok((width, width))
        }
        moxie_graph::OpParams::Attention {
            heads, head_dim, ..
        } => {
            let width = product(heads, head_dim)?;
            Ok((head_dim, width))
        }
        moxie_graph::OpParams::GeGlu { width } => Ok((width, width)),
        moxie_graph::OpParams::Residual { .. } => Ok((1, 1)),
        moxie_graph::OpParams::Route {
            hidden, experts, ..
        } => Ok((hidden, experts)),
        moxie_graph::OpParams::ExpertMlp {
            hidden,
            intermediate,
            ..
        } => Ok((hidden, intermediate)),
        moxie_graph::OpParams::Combine { hidden, .. } => Ok((hidden, hidden)),
        _ => Err(Error::UnsupportedKernel {
            operation: node.params.op().name(),
            detail: "operation has no dense descriptor shape".into(),
        }),
    }
}

fn dense_workspace(
    graph: &Graph,
    node: &moxie_graph::Node,
    operation: SemanticKernelOp,
    rows: u64,
    joins: &BTreeMap<NodeId, HostExpertJoin>,
) -> Result<(moxie_types::WorkspaceExpression, u64, u64), Error> {
    match operation {
        SemanticKernelOp::RmsNorm => {
            let bytes = rows
                .checked_mul(4)
                .ok_or_else(|| invalid("workspace", "RMS workspace overflowed"))?;
            Ok((moxie_types::WorkspaceExpression::RowsTimesF32, bytes, 0))
        }
        SemanticKernelOp::Rope => {
            let moxie_graph::OpParams::Rope { rotary_dim, .. } = node.params else {
                unreachable!("dense Rope operation")
            };
            let pairs = rotary_dim / 2;
            let bytes = rows
                .checked_mul(pairs)
                .and_then(|v| v.checked_mul(8))
                .ok_or_else(|| invalid("workspace", "RoPE angle table overflowed"))?;
            Ok((
                moxie_types::WorkspaceExpression::RowsTimesRopeAnglesF32,
                bytes,
                bytes,
            ))
        }
        SemanticKernelOp::ExpertMlp(_) => {
            let moxie_graph::OpParams::ExpertMlp {
                intermediate,
                top_k,
                ..
            } = node.params
            else {
                unreachable!("dense ExpertMlp operation")
            };
            let bytes = rows
                .checked_mul(top_k)
                .and_then(|v| v.checked_mul(intermediate))
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| invalid("workspace", "expert workspace overflowed"))?;
            Ok((
                moxie_types::WorkspaceExpression::RowsTimesIntermediateF32,
                bytes,
                0,
            ))
        }
        SemanticKernelOp::CombineHostJoin => {
            let moxie_graph::OpParams::Combine { hidden, top_k, .. } = node.params else {
                unreachable!("host join descriptor belongs to Combine")
            };
            let join = joins
                .get(&node.id)
                .ok_or_else(|| invalid("host_experts", "host join has no owner metadata"))?;
            let expert = node
                .inputs
                .get(1)
                .and_then(|slots| graph.producer(*slots))
                .ok_or_else(|| invalid("host_experts", "host join has no ExpertMlp producer"))?;
            let moxie_graph::OpParams::ExpertMlp { intermediate, .. } = expert.params else {
                return Err(invalid(
                    "host_experts",
                    "host join slots are not produced by ExpertMlp",
                ));
            };
            if join.host_experts() == 0 {
                return Err(invalid("host_experts", "host owner group is empty"));
            }
            let device_bytes = rows
                .checked_mul(hidden)
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| bytes.checked_mul(4))
                .ok_or_else(|| invalid("workspace", "host join device workspace overflowed"))?;
            let routed_input = rows
                .checked_mul(hidden)
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| invalid("host_workspace", "host input extent overflowed"))?;
            let route = rows
                .checked_mul(top_k)
                .and_then(|bytes| bytes.checked_mul(8))
                .ok_or_else(|| invalid("host_workspace", "host route extent overflowed"))?;
            let host_slots = rows
                .checked_mul(top_k)
                .and_then(|bytes| bytes.checked_mul(hidden))
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| invalid("host_workspace", "host slot extent overflowed"))?;
            let host_partial = rows
                .checked_mul(hidden)
                .and_then(|bytes| bytes.checked_mul(4))
                .ok_or_else(|| invalid("host_workspace", "host partial extent overflowed"))?;
            // `lanes(intermediate)` clamps to `intermediate`, so the CPU
            // ExpertShape workspace is intermediate + 2 * intermediate floats.
            let cpu_workspace = intermediate
                .checked_mul(2)
                .and_then(|tiles| intermediate.checked_add(tiles))
                .and_then(|floats| floats.checked_mul(4))
                .ok_or_else(|| invalid("host_workspace", "CPU expert workspace overflowed"))?;
            let host_bytes = [routed_input, route, host_slots, host_partial, cpu_workspace]
                .into_iter()
                .try_fold(0u64, |total, bytes| total.checked_add(bytes))
                .ok_or_else(|| invalid("host_workspace", "host join workspace overflowed"))?;
            Ok((
                moxie_types::WorkspaceExpression::RowsTimesHiddenTimesTwoF32,
                device_bytes,
                host_bytes,
            ))
        }
        _ => Ok((moxie_types::WorkspaceExpression::Zero, 0, 0)),
    }
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

    #[test]
    fn prefill_chunks_greedily_uses_valid_buckets_and_refuses_bad_input() {
        for (rows, buckets, expected) in [
            (13, &[1, 2, 4, 8][..], &[8, 4, 1][..]),
            (8, &[1, 2, 4, 8][..], &[8][..]),
            (3, &[1, 4][..], &[1, 1, 1][..]),
        ] {
            assert_eq!(prefill_chunks(rows, buckets), Ok(expected.to_vec()));
        }

        for (rows, buckets, field) in [
            (13, &[][..], "buckets"),
            (13, &[1, 4, 2][..], "buckets"),
            (13, &[2, 4][..], "buckets"),
            (13, &[0, 1][..], "buckets"),
            (0, &[1][..], "rows"),
        ] {
            assert!(matches!(
                prefill_chunks(rows, buckets),
                Err(Error::InvalidRequest { field: actual, .. }) if actual == field
            ));
        }
    }

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
            SemanticKernelOp::Linear
            | SemanticKernelOp::LinearSplit
            | SemanticKernelOp::LinearPartial
            | SemanticKernelOp::RmsNorm => vec![
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
            // Task 0037's operand list, built where that launch is built for
            // the same reason: query activations, the two paged payloads, and
            // the page table that says where a logical page physically is.
            SemanticKernelOp::PagedAttention => vec![
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::PageIndex,
            ],
            SemanticKernelOp::CombineHostJoin => vec![
                KernelOperand::RouteIndex,
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
            ],
            SemanticKernelOp::Embedding
            | SemanticKernelOp::GroupedRmsNorm
            | SemanticKernelOp::Rope
            | SemanticKernelOp::GeGlu
            | SemanticKernelOp::ScaledResidual
            | SemanticKernelOp::VocabProjection
            | SemanticKernelOp::Route
            | SemanticKernelOp::Combine
            | SemanticKernelOp::CombinePartial => Vec::new(),
        };
        SemanticKernelDescriptor {
            id: KernelId(format!("{}-{}", op.name(), sm.name())),
            abi_version: 1,
            operation: op,
            inputs,
            output: ActivationPrecision::expect(if op == SemanticKernelOp::CombinePartial {
                Precision::F32
            } else {
                Precision::Bf16
            }),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            rounding: if matches!(
                op,
                SemanticKernelOp::LinearPartial | SemanticKernelOp::CombinePartial
            ) {
                RoundingProfile::Unrounded
            } else {
                RoundingProfile::FinalBf16Rne
            },
            layout: TensorLayout::ContiguousRowMajorV1,
            shape: KernelShapeBounds {
                max_rows: 64,
                max_input: 1024,
                max_output: 1024,
            },
            sm,
            workspace: match op {
                SemanticKernelOp::RmsNorm => WorkspaceExpression::RowsTimesF32,
                SemanticKernelOp::CombineHostJoin => {
                    WorkspaceExpression::RowsTimesHiddenTimesTwoF32
                }
                _ => WorkspaceExpression::Zero,
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
            // The real limits on every NVIDIA architecture to date.
            max_grid: (2_147_483_647, 65_535, 65_535),
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
            paged_state_capacity: None,
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
    fn a_declared_order_must_fit_an_existing_unbiased_linear() {
        let selected_graph = graph(8, 3.5);
        let cap = capability(SmVersion::SM86);
        let linear = selected_graph
            .nodes()
            .iter()
            .find(|node| node.params.op() == Op::Linear)
            .unwrap()
            .id;
        let other = selected_graph
            .nodes()
            .iter()
            .find(|node| node.params.op() != Op::Linear)
            .unwrap()
            .id;
        let order = |blocks, slice| LinearReductionOrder { blocks, slice };
        let slice = |first, width, full_width| {
            Some(moxie_graph::LinearInputSlice {
                first,
                width,
                full_width,
            })
        };
        for (id, declared) in [
            (linear, order(0, None)),
            (linear, order(3, None)),
            (linear, order(2, slice(0, 4, 8))),
            (linear, order(2, slice(12, 8, 16))),
            (linear, order(2, slice(1, 8, 16))),
            (other, order(1, None)),
            (NodeId(999), order(1, None)),
        ] {
            let refused = lower_selected_ordered(
                &selected_graph,
                workload(&selected_graph, 1),
                &cap,
                &catalogue(SmVersion::SM86),
                &BTreeMap::from([(id, declared)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    refused,
                    Error::InvalidRequest {
                        field: "linear_order",
                        ..
                    }
                ),
                "{id:?} {declared:?}: {refused}"
            );
        }
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
    fn routed_graph(hidden: u64, route_experts: u64, expert_experts: u64) -> Graph {
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
                weight(vec![Dim::constant(route_experts), Dim::constant(hidden)]),
            )
            .unwrap();
        let route = builder
            .node(
                OpParams::Route {
                    hidden,
                    experts: route_experts,
                    top_k: TOP_K,
                    input: moxie_graph::RouterInput::Normalized {
                        eps: 1e-6,
                        input_scale: 1.0,
                    },
                    score: moxie_graph::RouteScore::Softmax,
                    per_expert_scale: false,
                    selection_bias: false,
                    coefficient: moxie_graph::RouteCoefficient::Fp32,
                },
                &[x, proj, gain],
            )
            .unwrap();
        let gate_up = builder
            .weight(
                "fused gate/up",
                weight(vec![
                    Dim::constant(expert_experts),
                    Dim::constant(2 * INTERMEDIATE),
                    Dim::constant(hidden),
                ]),
            )
            .unwrap();
        let down = builder
            .weight(
                "fused down",
                weight(vec![
                    Dim::constant(expert_experts),
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
                    experts: expert_experts,
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
                    output_scale: 1.0,
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
        let graph = routed_graph(8, 2, 2);
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
                input: moxie_graph::RouterInput::Raw,
                score: moxie_graph::RouteScore::Sigmoid,
                per_expert_scale: false,
                selection_bias: false,
                coefficient: moxie_graph::RouteCoefficient::Bf16,
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
                output_scale: 1.0,
                hidden: 8,
                top_k: 1,
                order: moxie_graph::CombineOrder::AscendingExpertId,
            },
        ] {
            assert_eq!(
                params.partition_rule(),
                moxie_graph::PartitionRule::ExpertOwnerShardable,
                "{} must fail closed until M5",
                params.op().name()
            );
        }
    }

    #[test]
    fn routed_edges_must_agree_on_the_route() {
        let graph = routed_graph(8, 5, 4);
        let cap = capability(SmVersion::SM86);
        let catalogue = catalogue(SmVersion::SM86);
        let error = lower_selected(&graph, workload(&graph, 1), &cap, &catalogue).unwrap_err();
        match error {
            Error::UnsupportedKernel { operation, detail } => {
                assert_eq!(operation, "route");
                assert!(detail.contains("experts=4") && detail.contains("experts=5"));
            }
            other => panic!("expected a typed route refusal, got {other:?}"),
        }

        let graph = routed_graph(8, 8, 4);
        let expert = graph
            .nodes()
            .iter()
            .find(|node| matches!(node.params, OpParams::ExpertMlp { .. }))
            .unwrap();
        let combine = graph
            .nodes()
            .iter()
            .find(|node| matches!(node.params, OpParams::Combine { .. }))
            .unwrap();
        let ownership = BTreeMap::from([(
            expert.id,
            ExpertOwnership {
                groups: 2,
                owned: 0,
            },
        )]);
        let combine_orders = BTreeMap::from([(
            combine.id,
            CombineReductionOrder {
                groups: 2,
                owned: Some(1),
            },
        )]);
        assert!(matches!(
            lower_selected_ordered(
                &graph,
                workload(&graph, 1),
                &cap,
                &catalogue,
                &BTreeMap::new(),
                &combine_orders,
                &ownership,
            ),
            Err(Error::UnsupportedKernel {
                operation: "route",
                ..
            })
        ));
    }

    #[test]
    fn attention_selects_one_kernel_and_only_query_output_device_slots() {
        let mut registry = OracleRegistry::new();
        registry
            .register(
                Op::Attention,
                ORACLE,
                OracleEvidence {
                    implementation: "moxie_plan::selected::tests",
                    test_module: "moxie_plan::selected::tests",
                },
            )
            .unwrap();
        let activation = |width| {
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                vec![Dim::symbol(ROWS), Dim::constant(width)],
            )
        };
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let query = builder.input("query", activation(128));
        let keys = builder.input("keys", activation(64));
        let values = builder.input("values", activation(64));
        let positions = builder.input(
            "positions",
            TensorSpec::new(
                ValueRole::Index(moxie_graph::IndexEncoding::U64),
                vec![Dim::symbol(ROWS)],
            ),
        );
        let output = builder
            .node(
                OpParams::Attention {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 64,
                    scale: crate::reciprocal_sqrt_scale(64),
                    visibility: crate::Visibility::Causal,
                    layer: 0,
                },
                &[query, keys, values, positions],
            )
            .unwrap();
        let graph = builder.finish(output, &registry).unwrap();
        let sm = SmVersion::SM86;
        let cap = capability(sm);
        let catalogue =
            KernelCatalogue::new(vec![descriptor(SemanticKernelOp::PagedAttention, sm)]).unwrap();
        let plan = lower_selected(&graph, workload(&graph, 1), &cap, &catalogue).unwrap();

        assert!(plan.is_paged_attention());
        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.values().len(), 2);
        assert_eq!(plan.values()[0].value, query);
        assert_eq!(plan.values()[1].value, output);
        assert_eq!(plan.activation_region_bytes(), 512);
        assert_eq!(plan.weight_region_bytes(), 0);
        assert_eq!(plan.workspace_region_bytes(), 0);
        assert_eq!(plan.stages(), &["attention", "terminal-output"]);
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
