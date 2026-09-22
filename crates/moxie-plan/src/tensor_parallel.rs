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
    Graph, LinearInputSlice, LinearReductionOrder, NodeId, Op, OpParams, PartitionRule, ValueId,
};

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
    /// Contiguous rows of a row-major `[out, in]` weight this rank owns.
    pub rows: BTreeMap<ValueId, Range<u64>>,
    /// One compact contiguous input-axis slice used in every row of a weight
    /// and in the corresponding activation.
    pub slices: BTreeMap<ValueId, LinearInputSlice>,
    /// The local view of each input-axis-split linear's declared order.
    pub linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorParallelLowering {
    pub stages: Vec<Stage>,
    pub ranks: Vec<RankPart>,
    /// Node-keyed declaration consumed by the full host reference path.
    pub linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
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
            OpParams::MlaAttention { .. } | OpParams::Route { .. } => {
                return Err(refuse_op(node));
            }
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

fn refuse_op(node: &moxie_graph::Node) -> TensorParallelRefused {
    TensorParallelRefused::Op {
        node: node.id,
        op: node.params.op(),
    }
}
