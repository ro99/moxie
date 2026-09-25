use moxie_graph::{OpParams, OracleRegistry, Visibility};
use moxie_models::gemma4::{Fraction, Gemma4Text, TextConfig, embedding_scale};
use moxie_plan::{
    CandidateKind, DeviceCost, Endpoint, LinkCost, PairVerdict, TopologyCosts, UserWorkload,
    Verdict, compare_phase_pairs, compare_plans,
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
                linear_tflops: 1e9,
                usable_bytes: 3_000_000,
            },
            DeviceCost {
                device: b,
                memory_gbps: 800.0,
                linear_tflops: 1e9,
                usable_bytes: 3_000_000,
            },
            DeviceCost {
                device: c,
                memory_gbps: 380.0,
                linear_tflops: 1e9,
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
                linear_tflops: 1e9,
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

#[test]
fn split_phase_pairs_price_the_kv_move() {
    let (graph, oracles) = graph();
    let costs = costs();
    let workload = UserWorkload {
        prompt_tokens: 8,
        generated_tokens: 2,
    };
    let plans = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        workload,
        &costs,
    )
    .unwrap();
    let pairs = compare_phase_pairs(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        workload,
        &costs,
    )
    .unwrap();

    for (phase, verdict) in &pairs {
        if phase.prefill != phase.decode {
            continue;
        }
        let PairVerdict::Ranked { estimate, .. } = verdict else {
            panic!("same-placement pair was not ranked: {phase:?}");
        };
        let (
            _,
            Verdict::Ranked {
                estimate: plan_estimate,
                ..
            },
        ) = plans
            .iter()
            .find(|(kind, _)| kind == &phase.prefill)
            .expect("same-placement pair has an estimable plan")
        else {
            panic!("same-placement pair is not estimated by compare_plans: {phase:?}");
        };
        assert_eq!(estimate.device_bytes, plan_estimate.device_bytes);
        assert_eq!(estimate.prefill_ms, plan_estimate.prefill_ms);
        assert_eq!(estimate.first_decode_ms, plan_estimate.first_decode_ms);
        assert_eq!(estimate.total_ms, plan_estimate.total_ms);
        assert_eq!(estimate.transition_bytes, 0);
        assert_eq!(estimate.transition_ms, 0.0);
        assert!(!estimate.transition_via_host);
    }

    let [a, b, c] = devices();
    let single = |device| CandidateKind::Single { device };
    let pair_for = |prefill: CandidateKind, decode: CandidateKind| {
        pairs
            .iter()
            .find(|(phase, _)| phase.prefill == prefill && phase.decode == decode)
            .map(|(_, verdict)| verdict)
            .expect("the requested estimable pair exists")
    };
    let PairVerdict::Ranked { estimate, .. } = pair_for(single(a), single(b)) else {
        panic!("direct-link pair must be ranked");
    };
    assert_eq!(estimate.transition_bytes, 3_072);
    assert!(!estimate.transition_via_host);

    let PairVerdict::Ranked { estimate, .. } = pair_for(single(a), single(c)) else {
        panic!("host-routed pair must be ranked");
    };
    assert_eq!(estimate.transition_bytes, 3_072);
    assert!(estimate.transition_via_host);

    let partially_shared = pairs.iter().find_map(|(phase, verdict)| {
        if phase.prefill != single(a)
            || !matches!(phase.decode, CandidateKind::Pipeline { ref stages }
                if stages.iter().any(|(devices, _)| devices.contains(&a)))
        {
            return None;
        }
        match verdict {
            PairVerdict::Ranked { estimate, .. } => Some(estimate),
            PairVerdict::Rejected { .. } => None,
        }
    });
    let partially_shared = partially_shared.expect("a split pair keeps part of KV on the source");
    assert!(partially_shared.transition_bytes > 0);
    assert!(partially_shared.transition_bytes < 3_072);

    let mut reversed_costs = costs;
    reversed_costs.devices.reverse();
    reversed_costs.links.reverse();
    assert_eq!(
        pairs,
        compare_phase_pairs(
            &graph,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            workload,
            &reversed_costs,
        )
        .unwrap()
    );
}

#[test]
fn compute_term_prices_long_prefill() {
    const PROMPT: u64 = 10_000;
    const LOW_TFLOPS: f64 = 1e-6;

    let (graph, oracles) = graph();
    let baseline = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: 8,
            generated_tokens: 2,
        },
        &costs(),
    )
    .unwrap();
    let memory_only = [
        (0.000_114_6, 0.000_114_6, 0.000_344_28),
        (0.000_114_6, 0.000_114_6, 0.000_344_28),
        (0.000_241_263_158, 0.000_241_263_158, 0.000_724_8),
        (0.032_221_347_368, 0.032_165_347_368, 0.096_552_698_947),
        (0.032_221_347_368, 0.032_165_347_368, 0.096_552_698_947),
        (0.032_222_32, 0.032_166_32, 0.096_555_616_842),
        (0.032_222_32, 0.032_166_32, 0.096_555_616_842),
        (0.032_222_408_421, 0.032_166_408_421, 0.096_555_882_105),
        (0.032_222_408_421, 0.032_166_408_421, 0.096_555_882_105),
        (0.076_166_088_421, 0.076_126_888_421, 0.228_420_362_105),
    ];
    let mut ranked = 0;
    for (_, verdict) in baseline {
        if let Verdict::Ranked { estimate, .. } = verdict {
            let (prefill, first_decode, total) = memory_only[ranked];
            assert!((estimate.prefill_ms - prefill).abs() < 1e-12);
            assert!((estimate.first_decode_ms - first_decode).abs() < 1e-12);
            assert!((estimate.total_ms - total).abs() < 1e-12);
            ranked += 1;
        }
    }
    assert_eq!(ranked, memory_only.len());

    let mut slow_costs = costs();
    for device in &mut slow_costs.devices {
        device.linear_tflops = LOW_TFLOPS;
        device.usable_bytes = 100_000_000;
    }
    let estimates = compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens: PROMPT,
            generated_tokens: 0,
        },
        &slow_costs,
    )
    .unwrap();
    let device = devices()[0];
    let estimate = estimates
        .iter()
        .find_map(|(kind, verdict)| match (kind, verdict) {
            (CandidateKind::Single { device: candidate }, Verdict::Ranked { estimate, .. })
                if *candidate == device =>
            {
                Some(estimate)
            }
            _ => None,
        })
        .expect("the fixture fits on one device");

    let linear_flops_per_row: u64 = graph
        .nodes()
        .iter()
        .map(|node| match node.params {
            OpParams::Linear {
                in_features,
                out_features,
                ..
            } => 2 * in_features * out_features,
            OpParams::VocabProjection { vocab, hidden, .. } => 2 * vocab * hidden,
            OpParams::ExpertMlp { .. } => unreachable!("the fixture is dense"),
            _ => 0,
        })
        .sum();
    let attention_flops: u64 = graph
        .nodes()
        .iter()
        .map(|node| match node.params {
            OpParams::Attention {
                heads,
                head_dim,
                visibility,
                ..
            } => {
                let keys: u64 = (1..=PROMPT)
                    .map(|position| match visibility {
                        Visibility::Causal => position,
                        Visibility::SlidingWindow { window } => position.min(window),
                    })
                    .sum();
                4 * heads * head_dim * keys
            }
            _ => 0,
        })
        .sum();
    let expected_ms = (PROMPT * linear_flops_per_row + attention_flops) as f64 / (LOW_TFLOPS * 1e9);
    assert!(
        (estimate.prefill_ms - expected_ms).abs() <= expected_ms * 1e-12,
        "long prefill should be compute-limited at {LOW_TFLOPS} TFLOP/s: actual={}, expected={expected_ms}",
        estimate.prefill_ms
    );
}
