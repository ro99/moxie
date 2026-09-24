//! Host tensor-parallel lowering of a dense graph (tasks 0056 and 0058).
//!
//! Attention heads remain local and are gathered. Dense MLP projections and
//! attention output projections use the same compact description for their
//! input-axis slice and are reduced in the declared rank order. Vocabulary
//! columns are local and gathered as FP32. This module only describes those
//! boundaries; it allocates no tensor data and owns no collective.

use std::collections::BTreeMap;
use std::ops::Range;

use moxie_graph::{
    CombineReductionOrder, ExpertOwnership, Graph, GraphBuilder, LinearInputSlice,
    LinearReductionOrder, NodeId, Op, OpParams, OracleId, OracleRegistry, PartitionRule, ValueId,
};
use moxie_types::{Dim, Error};

/// Consecutive nodes that run the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// Every rank runs these nodes with the graph's own parameters.
    Replicated(Range<usize>),
    /// Every rank runs these nodes with its own [`RankPart`] parameters, then
    /// applies the declared join to the stage output.
    Local { nodes: Range<usize>, join: Join },
}

/// The activation boundary after a local stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Join {
    /// Concatenate rank-local columns in ascending rank order.
    Gather { output: ValueId },
    /// Add rank-local FP32 partials in ascending rank order, then round once at
    /// the consumer's declared BF16 boundary.
    Reduce { output: ValueId },
}

/// One rank's parameter and compact-addressing view of the graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RankPart {
    /// Rewritten parameters for every local node.
    pub params: BTreeMap<NodeId, OpParams>,
    /// Contiguous outer-axis range of a row-major weight this rank owns.
    pub rows: BTreeMap<ValueId, Range<u64>>,
    /// An input-axis slice keyed by the weight itself, for fused operations
    /// whose activation input is replicated.
    pub weight_slices: BTreeMap<ValueId, LinearInputSlice>,
    /// One compact contiguous input-axis slice used in every row of a weight
    /// and in the corresponding activation.
    pub slices: BTreeMap<ValueId, LinearInputSlice>,
    /// The local view of each input-axis-split linear's declared order.
    pub linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
    /// Rank-local ordered reductions for `Combine` nodes.
    pub combine_orders: BTreeMap<NodeId, CombineReductionOrder>,
    /// Rank-local whole-expert ownership for `ExpertMlp` nodes.
    pub expert_ownership: BTreeMap<NodeId, ExpertOwnership>,
}

/// One boundary value read by a production stage graph. `original` is the
/// value in the full graph; `local` is the value the rebuilt graph binds.
/// `slice` is present only for a compact row-parallel activation view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageRead {
    pub original: ValueId,
    pub local: ValueId,
    pub slice: Option<LinearInputSlice>,
}

/// One weight binding in a stage graph, with the compact view applied by the
/// rank part. The composition root uses this to materialize the corresponding
/// host bytes without duplicating the lowering decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageWeight {
    pub original: ValueId,
    pub local: ValueId,
    pub rows: Option<Range<u64>>,
    pub slice: Option<LinearInputSlice>,
}

/// A standalone graph for one lowered stage, plus the small remapping table a
/// device consumer needs to bind its original inputs and publish its outputs.
/// Construction stays in `moxie-plan`; a composition root supplies the oracle
/// registry and owns the concrete host/device bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct StageGraph {
    pub graph: Graph,
    pub reads: Vec<StageRead>,
    pub weights: Vec<StageWeight>,
    pub produces: Vec<(ValueId, ValueId)>,
    pub linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
    pub combine_orders: BTreeMap<NodeId, CombineReductionOrder>,
    pub expert_ownership: BTreeMap<NodeId, ExpertOwnership>,
    /// Stage-local attention nodes use layer zero so the standalone graph's
    /// state schema remains valid; the device consumer restores this original
    /// layer when it appends to the rank's full KV authority.
    pub state_layers: BTreeMap<NodeId, u32>,
}

/// Rebuild a consecutive stage as a standalone validated graph.
pub fn build_stage_graph(
    graph: &Graph,
    part: Option<&RankPart>,
    nodes: Range<usize>,
    output: Option<ValueId>,
    oracle: OracleId,
    oracles: &OracleRegistry,
) -> moxie_types::Result<StageGraph> {
    if nodes.start >= nodes.end || nodes.end > graph.nodes().len() {
        return Err(Error::InvalidRequest {
            field: "stage",
            detail: "stage node range is empty or outside the graph".into(),
        });
    }
    let mut builder = GraphBuilder::new(oracle, graph.rows_symbol());
    let mut map = BTreeMap::new();
    let mut reads = Vec::new();
    let mut weights = Vec::new();
    let mut produces = Vec::new();
    let mut linear_orders = BTreeMap::new();
    let mut combine_orders = BTreeMap::new();
    let mut expert_ownership = BTreeMap::new();
    let mut state_layers = BTreeMap::new();
    let mut last = None;

    for (local_index, node) in graph.nodes()[nodes.clone()].iter().enumerate() {
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for &input in &node.inputs {
            let local = if let Some(&local) = map.get(&input) {
                local
            } else {
                let name = graph.name(input).unwrap_or("value");
                let slice = part.and_then(|p| {
                    if graph.weights().contains(&input) {
                        p.weight_slices
                            .get(&input)
                            .or_else(|| p.slices.get(&node.inputs[0]))
                            .copied()
                    } else {
                        p.slices.get(&input).copied()
                    }
                });
                let mut spec = graph
                    .spec(input)
                    .cloned()
                    .ok_or_else(|| invalid("stage", "stage input has no tensor spec"))?;
                if let Some(rows) = part.and_then(|p| p.rows.get(&input)) {
                    if spec.shape.is_empty() {
                        return Err(invalid("stage", "outer-axis-sharded weight is rank zero"));
                    }
                    spec.shape[0] = Dim::constant(rows.end - rows.start);
                } else if let Some(slice) = slice {
                    if spec.shape.len() != 2 {
                        return Err(invalid("stage", "input-sharded tensor is not rank two"));
                    }
                    spec.shape[1] = Dim::constant(slice.width);
                }
                let local = if graph.weights().contains(&input) {
                    let local = builder.weight(name, spec)?;
                    weights.push(StageWeight {
                        original: input,
                        local,
                        rows: part.and_then(|p| p.rows.get(&input)).cloned(),
                        slice,
                    });
                    local
                } else {
                    let local = builder.input(name, spec);
                    reads.push(StageRead {
                        original: input,
                        local,
                        slice,
                    });
                    local
                };
                map.insert(input, local);
                local
            };
            inputs.push(local);
        }
        let mut params = part
            .and_then(|p| p.params.get(&node.id))
            .cloned()
            .unwrap_or_else(|| node.params.clone());
        if let OpParams::Attention { layer, .. } = &params {
            state_layers.insert(NodeId(local_index as u32), *layer);
            if let OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
                visibility,
                ..
            } = params
            {
                params = OpParams::Attention {
                    heads,
                    kv_heads,
                    head_dim,
                    scale,
                    visibility,
                    layer: 0,
                };
            }
        } else if let OpParams::MlaAttention { descriptor } = &params {
            state_layers.insert(NodeId(local_index as u32), descriptor.layer);
            if let Some(part) = part
                && let Some(order) = part.combine_orders.get(&node.id).copied()
                && let Some(owner) = order.owned
            {
                let OpParams::MlaAttention { descriptor: source } = &node.params else {
                    return Err(invalid("stage", "MLA rank parameters lack a source node"));
                };
                let Some((slice, q_rows, kv_rows)) = part
                    .weight_slices
                    .get(&node.inputs[8])
                    .zip(part.rows.get(&node.inputs[4]))
                    .zip(part.rows.get(&node.inputs[7]))
                    .map(|((slice, q_rows), kv_rows)| (slice, q_rows, kv_rows))
                else {
                    return Err(invalid("stage", "MLA owner is missing its weight slices"));
                };
                let hp = descriptor.heads;
                let v = source.v_head_dim;
                let first_head = u64::from(owner).checked_mul(hp);
                let end_head = u64::from(owner)
                    .checked_add(1)
                    .and_then(|owner| owner.checked_mul(hp));
                let row_range = |width: Option<u64>| {
                    first_head
                        .zip(end_head)
                        .zip(width)
                        .and_then(|((first, end), width)| {
                            Some((first.checked_mul(width)?, end.checked_mul(width)?))
                        })
                };
                let q_width = source.qk_nope_head_dim.checked_add(source.qk_rope_head_dim);
                let kv_width = source.qk_nope_head_dim.checked_add(v);
                if owner >= order.groups
                    || hp.checked_mul(u64::from(order.groups)) != Some(source.heads)
                    || source.heads.checked_mul(v) != Some(slice.full_width)
                    || hp.checked_mul(v) != Some(slice.width)
                    || first_head.and_then(|head| head.checked_mul(v)) != Some(slice.first)
                    || !row_range(q_width).is_some_and(|(start, end)| *q_rows == (start..end))
                    || !row_range(kv_width).is_some_and(|(start, end)| *kv_rows == (start..end))
                {
                    return Err(invalid(
                        "stage",
                        "MLA owner disagrees with its weight slices",
                    ));
                }
            }
            let mut descriptor = *descriptor;
            descriptor.layer = 0;
            params = OpParams::MlaAttention { descriptor };
        }
        let local_output = builder.node(params, &inputs)?;
        if let Some(order) = part.and_then(|p| p.linear_orders.get(&node.id)).copied() {
            linear_orders.insert(NodeId(local_index as u32), order);
        }
        if let Some(order) = part.and_then(|p| p.combine_orders.get(&node.id)).copied() {
            combine_orders.insert(NodeId(local_index as u32), order);
        }
        if let Some(ownership) = part.and_then(|p| p.expert_ownership.get(&node.id)).copied() {
            expert_ownership.insert(NodeId(local_index as u32), ownership);
        }
        map.insert(node.output, local_output);
        produces.push((node.output, local_output));
        last = Some(local_output);
    }
    let selected_output = output
        .map(|original| {
            map.get(&original)
                .copied()
                .ok_or_else(|| invalid("stage", "stage output is not produced in the stage"))
        })
        .transpose()?
        .or(last)
        .ok_or_else(|| invalid("stage", "stage did not produce an output"))?;
    Ok(StageGraph {
        graph: builder.finish(selected_output, oracles)?,
        reads,
        weights,
        produces,
        linear_orders,
        combine_orders,
        expert_ownership,
        state_layers,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorParallelLowering {
    pub stages: Vec<Stage>,
    pub ranks: Vec<RankPart>,
    /// Node-keyed declaration consumed by the full host reference path.
    pub linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
    /// Reference declaration for the ordered cross-owner `Combine` reduction.
    pub combine_orders: BTreeMap<NodeId, CombineReductionOrder>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelRefused {
    NoRanks,
    /// The query heads do not divide over the ranks.
    Heads {
        layer: u32,
        heads: u64,
        ranks: u32,
    },
    /// The key/value heads neither divide over the ranks nor divide them.
    KvHeads {
        layer: u32,
        kv_heads: u64,
        ranks: u32,
    },
    /// An output or input dimension cannot be evenly owned by the ranks.
    Dimension {
        node: NodeId,
        axis: &'static str,
        value: u64,
        ranks: u32,
    },
    /// A biased input-axis projection cannot apply its bias exactly once in a
    /// rank-local partial stage.
    BiasedRow {
        node: NodeId,
    },
    /// An operation this slice does not lower.
    Op {
        node: NodeId,
        op: Op,
    },
    /// A head chain or dense local pattern this lowering cannot follow.
    HeadChain {
        node: NodeId,
    },
    /// An expert slot tensor must feed exactly one adjacent `Combine` node.
    ExpertSlots {
        node: NodeId,
    },
    /// A routed combine with non-unit output scale needs a scaled join that this
    /// slice does not declare.
    ScaledCombine {
        node: NodeId,
    },
}

impl core::fmt::Display for TensorParallelRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoRanks => write!(f, "tensor parallelism needs at least one rank"),
            Self::Heads {
                layer,
                heads,
                ranks,
            } => write!(
                f,
                "layer {layer}: {heads} query heads do not divide over {ranks} ranks"
            ),
            Self::KvHeads {
                layer,
                kv_heads,
                ranks,
            } => write!(
                f,
                "layer {layer}: {kv_heads} key/value heads neither divide over nor divide \
                 {ranks} ranks"
            ),
            Self::Dimension {
                node,
                axis,
                value,
                ranks,
            } => write!(
                f,
                "node {}: {axis} dimension {value} does not divide over {ranks} ranks",
                node.0
            ),
            Self::BiasedRow { node } => write!(
                f,
                "node {}: a biased input-axis linear cannot form one exact partial per rank",
                node.0
            ),
            Self::Op { node, op } => write!(
                f,
                "node {}: {} is not lowered to tensor parallelism",
                node.0,
                op.name()
            ),
            Self::HeadChain { node } => write!(
                f,
                "node {}: not part of a legal tensor-parallel local stage",
                node.0
            ),
            Self::ExpertSlots { node } => write!(
                f,
                "node {}: expert slots must feed exactly one adjacent combine",
                node.0
            ),
            Self::ScaledCombine { node } => write!(
                f,
                "node {}: routed combine output scale must be 1.0 for this lowering",
                node.0
            ),
        }
    }
}

impl std::error::Error for TensorParallelRefused {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Query,
    Kv,
}

#[derive(Debug, Clone, Copy)]
struct Heads {
    first: u64,
    count: u64,
}

#[derive(Debug, Clone, Copy)]
struct Mlp {
    gate: usize,
    up: usize,
    activation: usize,
    down: usize,
    width: u64,
    geglu: bool,
}

#[derive(Debug, Clone, Copy)]
struct ExpertStage {
    mlp: usize,
    combine: usize,
    experts: u64,
}

#[derive(Debug, Clone, Copy)]
struct Span {
    end: usize,
    join: Join,
}

/// Lower `graph` over `ranks` ranks.
pub fn lower_tensor_parallel(
    graph: &Graph,
    ranks: u32,
) -> Result<TensorParallelLowering, TensorParallelRefused> {
    if ranks == 0 {
        return Err(TensorParallelRefused::NoRanks);
    }
    let nodes = graph.nodes();
    let producer: BTreeMap<ValueId, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.output, i))
        .collect();

    // Node index -> (attention node index, axis) for every head-local node.
    let mut chain: BTreeMap<usize, (usize, Axis)> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        match node.params {
            OpParams::MlaAttention { .. } => continue,
            OpParams::Route { .. } => continue,
            _ if node.params.partition_rule() == PartitionRule::NotDetermined => {
                return Err(refuse_op(node));
            }
            OpParams::Attention { .. } => {}
            _ => continue,
        }
        chain.insert(index, (index, Axis::Query));
        for (slot, axis) in [(0, Axis::Query), (1, Axis::Kv), (2, Axis::Kv)] {
            let mut value = node.inputs[slot];
            loop {
                let at = *producer
                    .get(&value)
                    .ok_or(TensorParallelRefused::HeadChain { node: node.id })?;
                if chain
                    .insert(at, (index, axis))
                    .is_some_and(|seen| seen != (index, axis))
                {
                    return Err(TensorParallelRefused::HeadChain { node: nodes[at].id });
                }
                match nodes[at].params {
                    OpParams::Rope { .. } | OpParams::RmsNorm { .. } => value = nodes[at].inputs[0],
                    OpParams::Linear { .. } => break,
                    _ => return Err(TensorParallelRefused::HeadChain { node: nodes[at].id }),
                }
            }
        }
    }

    let mut spans = BTreeMap::new();
    let mut claimed = vec![false; nodes.len()];
    let mut attention_indices = BTreeMap::new();
    for &(attention, _) in chain.values() {
        attention_indices.insert(attention, ());
    }
    for &attention in attention_indices.keys() {
        let members: Vec<usize> = chain
            .iter()
            .filter_map(|(index, (owner, _))| (*owner == attention).then_some(*index))
            .collect();
        let start = *members.first().expect("attention has a chain");
        let end = members.last().copied().expect("attention has a chain") + 1;
        if end - start != members.len()
            || nodes[end - 1].params.op() != Op::Attention
            || members.iter().any(|index| claimed[*index])
        {
            return Err(TensorParallelRefused::HeadChain {
                node: nodes[start].id,
            });
        }
        for claimed in claimed.iter_mut().take(end).skip(start) {
            *claimed = true;
        }
        spans.insert(
            start,
            Span {
                end,
                join: Join::Gather {
                    output: nodes[attention].output,
                },
            },
        );
    }

    let mut mlps = Vec::new();
    for (activation, node) in nodes.iter().enumerate() {
        let (width, geglu) = match node.params {
            OpParams::GeGlu { width } => (width, true),
            OpParams::SwiGlu { width } => (width, false),
            _ => continue,
        };
        let Some(&gate) = producer.get(&node.inputs[0]) else {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        };
        let Some(&up) = producer.get(&node.inputs[1]) else {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        };
        let Some(down) = nodes.iter().enumerate().find_map(|(index, candidate)| {
            candidate.inputs.contains(&node.output).then_some(index)
        }) else {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        };
        let (
            OpParams::Linear {
                in_features: gate_in,
                out_features: gate_out,
                bias: gate_bias,
            },
            OpParams::Linear {
                in_features: up_in,
                out_features: up_out,
                bias: up_bias,
            },
            OpParams::Linear {
                in_features: down_in,
                bias: down_bias,
                ..
            },
        ) = (
            nodes[gate].params.clone(),
            nodes[up].params.clone(),
            nodes[down].params.clone(),
        )
        else {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        };
        if gate_in != up_in
            || gate_out != width
            || up_out != width
            || down_in != width
            || gate_bias
            || up_bias
            || down_bias
            || nodes[gate].inputs[0] != nodes[up].inputs[0]
        {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        let start = gate.min(up).min(activation).min(down);
        let end = gate.max(up).max(activation).max(down) + 1;
        if end - start != 4 || (start..end).any(|i| claimed[i]) {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        for claimed in claimed.iter_mut().take(end).skip(start) {
            *claimed = true;
        }
        spans.insert(
            start,
            Span {
                end,
                join: Join::Reduce {
                    output: nodes[down].output,
                },
            },
        );
        mlps.push(Mlp {
            gate,
            up,
            activation,
            down,
            width,
            geglu,
        });
    }

    // Route, ExpertMlp and their consuming Combine form one rank-local stage.
    // Every rank runs the unchanged Route over identical inputs and weights.
    let mut expert_stages = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let OpParams::ExpertMlp { experts, .. } = node.params else {
            continue;
        };
        let route_index = producer
            .get(&node.inputs[1])
            .copied()
            .ok_or(TensorParallelRefused::ExpertSlots { node: node.id })?;
        if route_index >= index || !matches!(nodes[route_index].params, OpParams::Route { .. }) {
            return Err(TensorParallelRefused::ExpertSlots { node: node.id });
        }
        let consumers: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(consumer, candidate)| {
                candidate.inputs.contains(&node.output).then_some(consumer)
            })
            .collect();
        if consumers.len() != 1 {
            return Err(TensorParallelRefused::ExpertSlots { node: node.id });
        }
        let combine_index = consumers[0];
        let OpParams::Combine { output_scale, .. } = nodes[combine_index].params else {
            return Err(TensorParallelRefused::ExpertSlots { node: node.id });
        };
        if combine_index != index + 1 || nodes[combine_index].inputs[0] != node.inputs[1] {
            return Err(TensorParallelRefused::ExpertSlots { node: node.id });
        }
        if output_scale != 1.0 {
            return Err(TensorParallelRefused::ScaledCombine {
                node: nodes[combine_index].id,
            });
        }
        if !experts.is_multiple_of(u64::from(ranks)) {
            return Err(TensorParallelRefused::Dimension {
                node: node.id,
                axis: "experts",
                value: experts,
                ranks,
            });
        }
        let start = route_index;
        let end = combine_index + 1;
        if (start..end).any(|i| claimed[i]) {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        for claimed in claimed.iter_mut().take(end).skip(start) {
            *claimed = true;
        }
        let combine = &nodes[combine_index];
        if spans
            .insert(
                start,
                Span {
                    end,
                    join: Join::Reduce {
                        output: combine.output,
                    },
                },
            )
            .is_some()
        {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        expert_stages.push(ExpertStage {
            mlp: index,
            combine: combine_index,
            experts,
        });
    }
    for (index, node) in nodes.iter().enumerate() {
        if matches!(node.params, OpParams::Combine { .. })
            && !expert_stages.iter().any(|stage| stage.combine == index)
        {
            return Err(refuse_op(node));
        }
    }

    let mut output_linears = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let OpParams::Linear {
            in_features,
            out_features,
            bias,
        } = node.params
        else {
            continue;
        };
        let Some(&input_producer) = producer.get(&node.inputs[0]) else {
            continue;
        };
        if !matches!(nodes[input_producer].params, OpParams::Attention { .. }) {
            continue;
        }
        if claimed[index] {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        if bias {
            return Err(TensorParallelRefused::BiasedRow { node: node.id });
        }
        claimed[index] = true;
        spans.insert(
            index,
            Span {
                end: index + 1,
                join: Join::Reduce {
                    output: node.output,
                },
            },
        );
        output_linears.push((index, in_features, out_features));
    }

    let mut vocab = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if let OpParams::VocabProjection {
            vocab: width,
            hidden,
            ..
        } = node.params
        {
            if claimed[index] {
                return Err(TensorParallelRefused::HeadChain { node: node.id });
            }
            claimed[index] = true;
            spans.insert(
                index,
                Span {
                    end: index + 1,
                    join: Join::Gather {
                        output: node.output,
                    },
                },
            );
            vocab.push((index, width, hidden));
        }
    }

    let mut mla_nodes = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let OpParams::MlaAttention { descriptor } = node.params else {
            continue;
        };
        if descriptor.heads % u64::from(ranks) != 0 {
            return Err(TensorParallelRefused::Heads {
                layer: descriptor.layer,
                heads: descriptor.heads,
                ranks,
            });
        }
        if claimed[index] {
            return Err(TensorParallelRefused::HeadChain { node: node.id });
        }
        claimed[index] = true;
        spans.insert(
            index,
            Span {
                end: index + 1,
                join: Join::Reduce {
                    output: node.output,
                },
            },
        );
        mla_nodes.push((index, descriptor));
    }

    // A value produced by a local stage may leave only through that stage's
    // declared join, whether the consumer is another stage or the graph
    // boundary itself.
    for (&start, span) in &spans {
        let local = start..span.end;
        let join = match span.join {
            Join::Gather { output } | Join::Reduce { output } => output,
        };
        for index in local.clone() {
            let output = nodes[index].output;
            if output == join {
                continue;
            }
            let escapes = output == graph.output()
                || nodes.iter().enumerate().any(|(consumer, node)| {
                    !local.contains(&consumer) && node.inputs.contains(&output)
                });
            if escapes {
                return Err(TensorParallelRefused::HeadChain {
                    node: nodes[index].id,
                });
            }
        }
    }

    let mut stages = Vec::new();
    let mut start = 0;
    while start < nodes.len() {
        if let Some(span) = spans.get(&start) {
            stages.push(Stage::Local {
                nodes: start..span.end,
                join: span.join,
            });
            start = span.end;
        } else {
            let end = spans.keys().find(|&&candidate| candidate > start).copied();
            let end = end.unwrap_or(nodes.len());
            stages.push(Stage::Replicated(start..end));
            start = end;
        }
    }

    let r = u64::from(ranks);
    let mut parts = vec![RankPart::default(); ranks as usize];
    let mut linear_orders = BTreeMap::new();
    let mut combine_orders = BTreeMap::new();

    for stage in expert_stages {
        let expert = &nodes[stage.mlp];
        let combine = &nodes[stage.combine];
        let experts_per_rank = stage.experts / r;
        let OpParams::ExpertMlp {
            hidden,
            intermediate,
            top_k,
            activation,
            ..
        } = expert.params
        else {
            unreachable!("expert stages are indexed by ExpertMlp");
        };
        combine_orders.insert(
            combine.id,
            CombineReductionOrder {
                groups: ranks,
                owned: None,
            },
        );
        for (rank, part) in parts.iter_mut().enumerate() {
            let owned = rank as u32;
            let first = u64::from(owned) * experts_per_rank;
            let rows = first..first + experts_per_rank;
            for weight in [expert.inputs[2], expert.inputs[3]] {
                if part
                    .rows
                    .insert(weight, rows.clone())
                    .is_some_and(|seen| seen != rows)
                {
                    return Err(TensorParallelRefused::HeadChain { node: expert.id });
                }
            }
            part.params.insert(
                expert.id,
                OpParams::ExpertMlp {
                    hidden,
                    intermediate,
                    experts: experts_per_rank,
                    top_k,
                    activation,
                },
            );
            part.expert_ownership.insert(
                expert.id,
                ExpertOwnership {
                    groups: ranks,
                    owned,
                },
            );
            part.combine_orders.insert(
                combine.id,
                CombineReductionOrder {
                    groups: ranks,
                    owned: Some(owned),
                },
            );
        }
    }

    // MLA keeps the shared latent replicated and splits complete heads plus
    // each rank's matching `o_proj` input columns.
    for (index, descriptor) in mla_nodes {
        let node = &nodes[index];
        let heads_per_rank = descriptor.heads / r;
        let query_head = descriptor.qk_nope_head_dim + descriptor.qk_rope_head_dim;
        let kv_head = descriptor.qk_nope_head_dim + descriptor.v_head_dim;
        let full_width = descriptor.heads * descriptor.v_head_dim;
        combine_orders.insert(
            node.id,
            CombineReductionOrder {
                groups: ranks,
                owned: None,
            },
        );
        for (rank, part) in parts.iter_mut().enumerate() {
            let first_head = rank as u64 * heads_per_rank;
            let q_rows = first_head * query_head..(first_head + heads_per_rank) * query_head;
            let kv_rows = first_head * kv_head..(first_head + heads_per_rank) * kv_head;
            for (weight, rows) in [(node.inputs[4], q_rows), (node.inputs[7], kv_rows)] {
                if part
                    .rows
                    .insert(weight, rows.clone())
                    .is_some_and(|seen| seen != rows)
                {
                    return Err(TensorParallelRefused::HeadChain { node: node.id });
                }
            }
            let slice = LinearInputSlice {
                first: first_head * descriptor.v_head_dim,
                width: heads_per_rank * descriptor.v_head_dim,
                full_width,
            };
            if part
                .weight_slices
                .insert(node.inputs[8], slice)
                .is_some_and(|seen| seen != slice)
            {
                return Err(TensorParallelRefused::HeadChain { node: node.id });
            }
            let mut local_descriptor = descriptor;
            local_descriptor.heads = heads_per_rank;
            part.params.insert(
                node.id,
                OpParams::MlaAttention {
                    descriptor: local_descriptor,
                },
            );
            part.combine_orders.insert(
                node.id,
                CombineReductionOrder {
                    groups: ranks,
                    owned: Some(rank as u32),
                },
            );
        }
    }

    // Attention head-local parameters.
    for (&index, &(attention, axis)) in &chain {
        let OpParams::Attention {
            heads,
            kv_heads,
            head_dim,
            layer,
            ..
        } = nodes[attention].params
        else {
            unreachable!("every chain is keyed by its attention node");
        };
        if heads % r != 0 {
            return Err(TensorParallelRefused::Heads {
                layer,
                heads,
                ranks,
            });
        }
        if kv_heads % r != 0 && r % kv_heads != 0 {
            return Err(TensorParallelRefused::KvHeads {
                layer,
                kv_heads,
                ranks,
            });
        }
        let node = &nodes[index];
        let global = match axis {
            Axis::Query => heads,
            Axis::Kv => kv_heads,
        };
        for (rank, part) in parts.iter_mut().enumerate() {
            let rank = rank as u64;
            let query = Heads {
                first: rank * (heads / r),
                count: heads / r,
            };
            let kv = if kv_heads % r == 0 {
                Heads {
                    first: rank * (kv_heads / r),
                    count: kv_heads / r,
                }
            } else {
                Heads {
                    first: rank * kv_heads / r,
                    count: 1,
                }
            };
            let local = match axis {
                Axis::Query => query,
                Axis::Kv => kv,
            };
            let params = match node.params.clone() {
                OpParams::Attention {
                    scale, visibility, ..
                } => OpParams::Attention {
                    heads: query.count,
                    kv_heads: kv.count,
                    head_dim,
                    scale,
                    visibility,
                    layer,
                },
                OpParams::Rope {
                    heads: h,
                    head_dim,
                    rotary_dim,
                    frequency_dim,
                    base,
                    layout,
                } if h == global => OpParams::Rope {
                    heads: local.count,
                    head_dim,
                    rotary_dim,
                    frequency_dim,
                    base,
                    layout,
                },
                OpParams::RmsNorm { hidden, group, eps } if group == global => OpParams::RmsNorm {
                    hidden: hidden / global * local.count,
                    group: local.count,
                    eps,
                },
                OpParams::Linear {
                    in_features,
                    out_features,
                    bias: false,
                } if out_features == global * head_dim => {
                    let rows = local.first * head_dim..(local.first + local.count) * head_dim;
                    let weight = node.inputs[1];
                    if part
                        .rows
                        .insert(weight, rows.clone())
                        .is_some_and(|seen| seen != rows)
                    {
                        return Err(TensorParallelRefused::HeadChain { node: node.id });
                    }
                    OpParams::Linear {
                        in_features,
                        out_features: local.count * head_dim,
                        bias: false,
                    }
                }
                _ => return Err(TensorParallelRefused::HeadChain { node: node.id }),
            };
            part.params.insert(node.id, params);
        }
    }

    // Dense MLP local stage: gate/up and either GLU variant are output-column
    // local; down consumes the compact local activation and contributes a
    // reduction.
    for mlp in mlps {
        let gate = &nodes[mlp.gate];
        let up = &nodes[mlp.up];
        let down = &nodes[mlp.down];
        let (gate_in, gate_out) = linear_shape(gate);
        let (up_in, up_out) = linear_shape(up);
        let (down_in, down_out) = linear_shape(down);
        if gate_in != up_in || gate_out != mlp.width || up_out != mlp.width || down_in != mlp.width
        {
            return Err(TensorParallelRefused::HeadChain { node: down.id });
        }
        if mlp.width % r != 0 {
            return Err(TensorParallelRefused::Dimension {
                node: gate.id,
                axis: "output",
                value: mlp.width,
                ranks,
            });
        }
        let local = mlp.width / r;
        linear_orders.insert(
            down.id,
            LinearReductionOrder {
                blocks: ranks,
                slice: None,
            },
        );
        for (rank, part) in parts.iter_mut().enumerate() {
            let rank = rank as u64;
            let first = rank * local;
            let rows = first..first + local;
            for node in [gate, up] {
                let weight = node.inputs[1];
                if part
                    .rows
                    .insert(weight, rows.clone())
                    .is_some_and(|seen| seen != rows)
                {
                    return Err(TensorParallelRefused::HeadChain { node: node.id });
                }
            }
            part.params.insert(
                gate.id,
                OpParams::Linear {
                    in_features: gate_in,
                    out_features: local,
                    bias: false,
                },
            );
            part.params.insert(
                up.id,
                OpParams::Linear {
                    in_features: up_in,
                    out_features: local,
                    bias: false,
                },
            );
            let activation = if mlp.geglu {
                OpParams::GeGlu { width: local }
            } else {
                OpParams::SwiGlu { width: local }
            };
            part.params.insert(nodes[mlp.activation].id, activation);
            part.params.insert(
                down.id,
                OpParams::Linear {
                    in_features: local,
                    out_features: down_out,
                    bias: false,
                },
            );
            let slice = LinearInputSlice {
                first,
                width: local,
                full_width: mlp.width,
            };
            part.slices.insert(nodes[mlp.activation].output, slice);
            part.linear_orders.insert(
                down.id,
                LinearReductionOrder {
                    blocks: ranks,
                    slice: Some(slice),
                },
            );
        }
    }

    // Attention output projections use the same row-parallel path.
    for (index, in_features, out_features) in output_linears {
        let node = &nodes[index];
        if in_features % r != 0 {
            return Err(TensorParallelRefused::Dimension {
                node: node.id,
                axis: "input",
                value: in_features,
                ranks,
            });
        }
        let local = in_features / r;
        linear_orders.insert(
            node.id,
            LinearReductionOrder {
                blocks: ranks,
                slice: None,
            },
        );
        for (rank, part) in parts.iter_mut().enumerate() {
            let slice = LinearInputSlice {
                first: rank as u64 * local,
                width: local,
                full_width: in_features,
            };
            part.slices.insert(node.inputs[0], slice);
            part.params.insert(
                node.id,
                OpParams::Linear {
                    in_features: local,
                    out_features,
                    bias: false,
                },
            );
            part.linear_orders.insert(
                node.id,
                LinearReductionOrder {
                    blocks: ranks,
                    slice: Some(slice),
                },
            );
        }
    }

    // Vocabulary output columns are local and retain FP32 until the gather.
    for (index, width, hidden) in vocab {
        let node = &nodes[index];
        if width % r != 0 {
            return Err(TensorParallelRefused::Dimension {
                node: node.id,
                axis: "output",
                value: width,
                ranks,
            });
        }
        let local = width / r;
        let softcap = match node.params {
            OpParams::VocabProjection { softcap, .. } => softcap,
            _ => unreachable!(),
        };
        for (rank, part) in parts.iter_mut().enumerate() {
            let first = rank as u64 * local;
            let end = first + local;
            let rows = first..end;
            let weight = node.inputs[1];
            if part
                .rows
                .insert(weight, rows.clone())
                .is_some_and(|seen| seen != rows)
            {
                return Err(TensorParallelRefused::HeadChain { node: node.id });
            }
            part.params.insert(
                node.id,
                OpParams::VocabProjection {
                    vocab: local,
                    hidden,
                    softcap,
                },
            );
        }
    }

    Ok(TensorParallelLowering {
        stages,
        ranks: parts,
        linear_orders,
        combine_orders,
    })
}

fn linear_shape(node: &moxie_graph::Node) -> (u64, u64) {
    match node.params {
        OpParams::Linear {
            in_features,
            out_features,
            ..
        } => (in_features, out_features),
        _ => unreachable!("the MLP matcher only stores linear nodes"),
    }
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

fn refuse_op(node: &moxie_graph::Node) -> TensorParallelRefused {
    TensorParallelRefused::Op {
        node: node.id,
        op: node.params.op(),
    }
}
