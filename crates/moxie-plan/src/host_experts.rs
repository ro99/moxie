//! Host-owned expert lowering beside one device owner group.

use std::collections::BTreeMap;

use moxie_graph::{
    CombineReductionOrder, ExpertOwnership, Graph, NodeId, OpParams, OracleId, OracleRegistry,
};

use crate::{RankPart, StageGraph, TensorParallelRefused, build_stage_graph};

/// Two expert-owner groups: group 0 on the device, group 1 on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostExpertJoin {
    first_host_expert: u32,
    host_experts: u32,
}

impl HostExpertJoin {
    pub const fn first_host_expert(&self) -> u32 {
        self.first_host_expert
    }

    pub const fn host_experts(&self) -> u32 {
        self.host_experts
    }
}

/// Checked device ownership and host joins for a routed graph.
#[derive(Debug, Clone, PartialEq)]
pub struct HostExpertLowering {
    device: RankPart,
    device_stage: StageGraph,
    reference_orders: BTreeMap<NodeId, CombineReductionOrder>,
    joins: BTreeMap<NodeId, HostExpertJoin>,
}

impl HostExpertLowering {
    pub fn device(&self) -> &RankPart {
        &self.device
    }

    pub fn device_stage(&self) -> &StageGraph {
        &self.device_stage
    }

    pub fn reference_orders(&self) -> &BTreeMap<NodeId, CombineReductionOrder> {
        &self.reference_orders
    }

    pub fn joins(&self) -> &BTreeMap<NodeId, HostExpertJoin> {
        &self.joins
    }
}

/// Lower each legal routed block into a device-owned first half and a host join.
pub fn lower_host_experts(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
) -> Result<HostExpertLowering, TensorParallelRefused> {
    let nodes = graph.nodes();
    let first = nodes
        .first()
        .expect("a validated graph contains at least one node");
    let mut device = RankPart::default();
    let mut reference_orders = BTreeMap::new();
    let mut joins = BTreeMap::new();
    let mut lowered = 0usize;

    for (index, node) in nodes.iter().enumerate() {
        let OpParams::ExpertMlp {
            hidden,
            intermediate,
            experts,
            top_k,
            activation,
        } = node.params
        else {
            continue;
        };
        let route_value = node
            .inputs
            .get(1)
            .copied()
            .ok_or(TensorParallelRefused::ExpertSlots { node: node.id })?;
        let route_index = nodes
            .iter()
            .position(|candidate| candidate.output == route_value)
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
        if combine_index != index + 1 || nodes[combine_index].inputs[0] != route_value {
            return Err(TensorParallelRefused::ExpertSlots { node: node.id });
        }
        if output_scale != 1.0 {
            return Err(TensorParallelRefused::ScaledCombine {
                node: nodes[combine_index].id,
            });
        }
        if !experts.is_multiple_of(2) {
            return Err(TensorParallelRefused::Dimension {
                node: node.id,
                axis: "experts",
                value: experts,
                ranks: 2,
            });
        }
        let half = experts / 2;
        let half_u32 = u32::try_from(half).map_err(|_| TensorParallelRefused::Dimension {
            node: node.id,
            axis: "experts",
            value: experts,
            ranks: 2,
        })?;
        let combine = nodes[combine_index].id;
        let join = HostExpertJoin {
            first_host_expert: half_u32,
            host_experts: half_u32,
        };
        let order = CombineReductionOrder {
            groups: 2,
            owned: Some(0),
        };
        device.params.insert(
            node.id,
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts: half,
                top_k,
                activation,
            },
        );
        for weight in [node.inputs[2], node.inputs[3]] {
            device.rows.insert(weight, 0..half);
        }
        device.expert_ownership.insert(
            node.id,
            ExpertOwnership {
                groups: 2,
                owned: 0,
            },
        );
        device.combine_orders.insert(combine, order);
        reference_orders.insert(
            combine,
            CombineReductionOrder {
                groups: 2,
                owned: None,
            },
        );
        joins.insert(combine, join);
        lowered += 1;
    }

    if lowered == 0 {
        return Err(TensorParallelRefused::Op {
            node: first.id,
            op: first.params.op(),
        });
    }

    let device_stage = build_stage_graph(
        graph,
        Some(&device),
        0..nodes.len(),
        Some(graph.output()),
        oracle,
        oracles,
    )
    .map_err(|error| TensorParallelRefused::StageGraph { error })?;

    Ok(HostExpertLowering {
        device,
        device_stage,
        reference_orders,
        joins,
    })
}
