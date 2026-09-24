use moxie_graph::OracleRegistry;
use moxie_models::gemma4::{Fraction, Gemma4Text, TextConfig, embedding_scale};
use moxie_plan::{
    CandidateKind, DeviceCost, Endpoint, LinkCost, TopologyCosts, UserWorkload, Verdict,
    compare_plans,
};
use moxie_types::{DeviceUuid, SymbolId};

fn devices() -> [DeviceUuid; 3] {
    [
        DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap(),
        DeviceUuid::parse("GPU-81fe4578-59b2-37c4-421e-287cdac78704").unwrap(),
        DeviceUuid::parse("GPU-97fe4889-4874-a378-198e-955d2e72c4a3").unwrap(),
    ]
}

fn graph() -> (moxie_graph::Graph, OracleRegistry) {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let graph = Gemma4Text::reduced(
        TextConfig {
            hidden: 32,
            layers: 6,
            heads: 4,
            local_kv_heads: 2,
            local_head_dim: 8,
            global_kv_heads: 2,
            global_head_dim: 8,
            intermediate: 48,
            vocab: 11,
            global_stride: 1,
            sliding_window: 16,
            rms_eps: 1e-6,
            sliding_rope_theta: 10_000.0,
            global_rope_theta: 1_000_000.0,
            global_partial_rotary: Fraction::QUARTER,
            final_logit_softcap: 30.0,
            layer_scalars: vec![1.0; 6],
            embedding_scale: embedding_scale(32),
            max_trained_position: 20_000,
            moe: None,
        },
        "plan-comparison-fixture",
    )
    .unwrap()
    .compose(&oracles, SymbolId(0))
    .unwrap()
    .graph;
    (graph, oracles)
}

fn link(from: Endpoint, to: Endpoint, bandwidth_gbps: f64, latency_us: f64) -> LinkCost {
    LinkCost {
        from,
        to,
        latency_us,
        bandwidth_gbps,
        concurrent_gbps: bandwidth_gbps,
    }
}

fn costs() -> TopologyCosts {
    let [a, b, c] = devices();
    let mut links = Vec::new();
    for device in [a, b, c] {
        links.push(link(Endpoint::Device(device), Endpoint::Host, 32.0, 8.0));
        links.push(link(Endpoint::Host, Endpoint::Device(device), 32.0, 8.0));
    }
    links.extend([
        link(Endpoint::Device(a), Endpoint::Device(b), 800.0, 5.0),
        link(Endpoint::Device(b), Endpoint::Device(a), 800.0, 5.0),
    ]);
    TopologyCosts {
        devices: vec![
            DeviceCost {
                device: a,
                memory_gbps: 800.0,
                usable_bytes: 3_000_000,
            },
            DeviceCost {
                device: b,
                memory_gbps: 800.0,
                usable_bytes: 3_000_000,
            },
            DeviceCost {
                device: c,
                memory_gbps: 380.0,
                usable_bytes: 3_000_000,
            },
        ],
        links,
    }
}

fn tie_costs() -> TopologyCosts {
    let [a, b, _] = devices();
    TopologyCosts {
        devices: [a, b]
            .map(|device| DeviceCost {
                device,
                memory_gbps: 800.0,
                usable_bytes: 3_000_000,
            })
            .into(),
        links: [a, b]
            .into_iter()
            .flat_map(|device| {
                [
                    link(Endpoint::Device(device), Endpoint::Host, 32.0, 8.0),
                    link(Endpoint::Host, Endpoint::Device(device), 32.0, 8.0),
                ]
            })
            .collect(),
    }
}

#[test]
fn the_ranking_is_deterministic_and_input_order_free() {
    let (graph, oracles) = graph();
    let costs = costs();
    let first = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &costs,
    )
    .unwrap();
    assert!(first.iter().any(|(kind, verdict)| {
        matches!(kind, CandidateKind::Tp2 { .. })
            && matches!(verdict, Verdict::Rejected { reason } if reason.contains("output") && reason.contains("11"))
    }));
    assert!(first.iter().any(|(kind, verdict)| {
        matches!(kind, CandidateKind::Pipeline { stages } if stages[0].0.len() == 2)
            && matches!(verdict, Verdict::Ranked { .. })
    }));
    let mut shuffled = costs.clone();
    shuffled.devices.reverse();
    shuffled.links.reverse();
    let second = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &shuffled,
    )
    .unwrap();
    assert_eq!(first, second);

    let ties = tie_costs();
    let ordered = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &ties,
    )
    .unwrap();
    let mut reverse_ties = ties;
    reverse_ties.devices.reverse();
    reverse_ties.links.reverse();
    assert_eq!(
        ordered,
        compare_plans(
            &graph,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            UserWorkload {
                prompt_tokens: 8,
                generated_tokens: 2
            },
            &reverse_ties,
        )
        .unwrap()
    );
    let tied_singles: Vec<_> = ordered
        .iter()
        .filter(|(_, verdict)| matches!(verdict, Verdict::Ranked { .. }))
        .filter_map(|(kind, _)| matches!(kind, CandidateKind::Single { .. }).then_some(kind))
        .collect();
    assert_eq!(tied_singles.len(), 2);
    assert_eq!(
        tied_singles[0],
        &CandidateKind::Single {
            device: devices()[0]
        }
    );
}

#[test]
fn the_joint_workload_changes_the_winner() {
    let (graph, oracles) = graph();
    let costs = costs();
    let short = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &costs,
    )
    .unwrap();
    let long = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 10_000,
            generated_tokens: 1,
        },
        &costs,
    )
    .unwrap();
    assert!(matches!(short[0].0, CandidateKind::Single { .. }));
    assert!(matches!(
        long[0].0,
        CandidateKind::Tp2 { .. } | CandidateKind::Pipeline { .. }
    ));
    assert!(long
        .iter()
        .filter(|(kind, _)| matches!(kind, CandidateKind::Single { .. }))
        .all(|(_, verdict)| matches!(verdict, Verdict::Rejected { reason } if reason.contains("needs"))));
}

#[test]
fn no_tp2_without_both_peer_links() {
    let (graph, oracles) = graph();
    let [a, b, _] = devices();
    let mut costs = costs();
    costs
        .links
        .retain(|link| !(link.from == Endpoint::Device(b) && link.to == Endpoint::Device(a)));
    let plans = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &costs,
    )
    .unwrap();
    assert!(plans.iter().all(|(kind, _)| match kind {
        CandidateKind::Tp2 { .. } => false,
        CandidateKind::Pipeline { stages } => stages.iter().all(|(devices, _)| devices.len() < 2),
        _ => true,
    }));
}
