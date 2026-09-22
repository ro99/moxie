//! Host tensor-parallel lowering of the attention sublayer (task 0056, M5.2).
//!
//! Each attention layer's head chain -- the `q`/`k`/`v` projections, their
//! per-head norms and rotations, and the attention itself -- is split over
//! ranks by whole heads. Everything else runs replicated. The per-rank
//! attention outputs are combined by one concatenation in global head order,
//! so no per-head arithmetic changes and the split result must be bit-identical
//! to the unsplit graph.
//!
//! Query heads split contiguously. Key/value heads split contiguously when the
//! rank count divides them. When the ranks outnumber them, each rank holds one
//! key/value head, replicated across the ranks that share it. Either way, query
//! head `h` still reads key/value head `h / (heads / kv_heads)`. Any other
//! combination is refused.
//!
//! This is a pure description. It allocates nothing and moves no bytes. The
//! executor owns collectives; this says where they go.

use std::collections::BTreeMap;
use std::ops::Range;

use moxie_graph::{Graph, NodeId, Op, OpParams, PartitionRule, ValueId};

/// Consecutive nodes, by index into [`Graph::nodes`], that run the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// Every rank runs these nodes with the graph's own parameters.
    Replicated(Range<usize>),
    /// Every rank runs these nodes with its own [`RankPart`] parameters. The
    /// ranks' `gather` values, concatenated along columns in rank order, are
    /// the unsplit graph's `gather` value.
    HeadLocal {
        nodes: Range<usize>,
        gather: ValueId,
    },
}

/// What one rank executes differently from the unsplit graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RankPart {
    /// Rewritten parameters for every head-local node.
    pub params: BTreeMap<NodeId, OpParams>,
    /// The logical rows of a row-major `[out, in]` weight this rank holds. A
    /// weight absent here is held whole.
    pub rows: BTreeMap<ValueId, Range<u64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorParallelLowering {
    pub stages: Vec<Stage>,
    pub ranks: Vec<RankPart>,
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
    /// An operation this slice does not lower.
    Op {
        node: NodeId,
        op: Op,
    },
    /// A head chain this lowering cannot follow: an unexpected producer, a
    /// width that is not head-aligned, a biased projection, a head-local value
    /// used outside its chain, or a chain that is not contiguous.
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
            } => {
                write!(
                    f,
                    "layer {layer}: {heads} query heads do not divide over {ranks} ranks"
                )
            }
            Self::KvHeads {
                layer,
                kv_heads,
                ranks,
            } => write!(
                f,
                "layer {layer}: {kv_heads} key/value heads neither divide over nor divide \
                 {ranks} ranks"
            ),
            Self::Op { node, op } => {
                write!(
                    f,
                    "node {}: {} is not lowered to tensor parallelism",
                    node.0,
                    op.name()
                )
            }
            Self::HeadChain { node } => {
                write!(
                    f,
                    "node {}: not part of a head-aligned attention chain",
                    node.0
                )
            }
        }
    }
}

impl std::error::Error for TensorParallelRefused {}

/// Which head axis a chain node carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Query,
    Kv,
}

/// One rank's contiguous run of heads on one axis.
#[derive(Debug, Clone, Copy)]
struct Heads {
    first: u64,
    count: u64,
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
                    // Between a projection and attention, a norm is per-head
                    // even when one head makes its group 1.
                    OpParams::Rope { .. } | OpParams::RmsNorm { .. } => value = nodes[at].inputs[0],
                    OpParams::Linear { .. } => break,
                    _ => return Err(TensorParallelRefused::HeadChain { node: nodes[at].id }),
                }
            }
        }
    }

    // Only the attention output may leave its chain.
    for (index, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            if let Some(&from) = producer.get(input)
                && let Some(&(attention, _)) = chain.get(&from)
                && from != attention
                && chain.get(&index).map(|c| c.0) != Some(attention)
            {
                return Err(TensorParallelRefused::HeadChain { node: node.id });
            }
        }
    }

    let mut stages = Vec::new();
    let mut start = 0;
    while start < nodes.len() {
        let owner = chain.get(&start).map(|c| c.0);
        let end = (start..nodes.len())
            .find(|i| chain.get(i).map(|c| c.0) != owner)
            .unwrap_or(nodes.len());
        stages.push(match owner {
            None => Stage::Replicated(start..end),
            // A chain ends at its attention node, and all of it is in this run.
            Some(attention) => {
                let whole = chain.values().filter(|c| c.0 == attention).count();
                if attention != end - 1 || end - start != whole {
                    return Err(TensorParallelRefused::HeadChain {
                        node: nodes[start].id,
                    });
                }
                Stage::HeadLocal {
                    nodes: start..end,
                    gather: nodes[attention].output,
                }
            }
        });
        start = end;
    }

    let mut parts = vec![RankPart::default(); ranks as usize];
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
        let r = u64::from(ranks);
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
        let refuse = TensorParallelRefused::HeadChain { node: node.id };
        let global = match axis {
            Axis::Query => heads,
            Axis::Kv => kv_heads,
        };
        for (rank, part) in parts.iter_mut().enumerate() {
            let rank = rank as u64;
            let q = Heads {
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
                Axis::Query => q,
                Axis::Kv => kv,
            };
            let params = match node.params.clone() {
                OpParams::Attention {
                    scale, visibility, ..
                } => OpParams::Attention {
                    heads: q.count,
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
                        return Err(refuse);
                    }
                    OpParams::Linear {
                        in_features,
                        out_features: local.count * head_dim,
                        bias: false,
                    }
                }
                _ => return Err(refuse),
            };
            part.params.insert(node.id, params);
        }
    }

    Ok(TensorParallelLowering {
        stages,
        ranks: parts,
    })
}

fn refuse_op(node: &moxie_graph::Node) -> TensorParallelRefused {
    TensorParallelRefused::Op {
        node: node.id,
        op: node.params.op(),
    }
}
