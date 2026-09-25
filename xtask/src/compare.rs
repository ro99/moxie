//! `cargo xtask-cuda compare-plans` -- print measured-cost plan estimates.

use moxie_graph::OracleRegistry;
use moxie_models::gemma4::{Fraction, Gemma4Text, TextConfig, embedding_scale};
use moxie_plan::{
    CandidateKind, Estimate, PairVerdict, UserWorkload, Verdict, compare_phase_pairs, compare_plans,
};
use moxie_types::{DeviceUuid, SymbolId};

pub fn run(
    costs: Option<&str>,
    model: Option<&str>,
    prompt: Option<&str>,
    generate: Option<&str>,
) -> i32 {
    let (Some(costs), Some(model), Some(prompt), Some(generate)) = (costs, model, prompt, generate)
    else {
        eprintln!("compare-plans requires --costs, --model, --prompt and --generate");
        return 2;
    };
    let parse = |name: &str, value: &str| match value.parse::<u64>() {
        Ok(value) => Some(value),
        Err(error) => {
            eprintln!("invalid {name} {value:?}: {error}");
            None
        }
    };
    let Some(prompt_tokens) = parse("--prompt", prompt) else {
        return 2;
    };
    let Some(generated_tokens) = parse("--generate", generate) else {
        return 2;
    };
    let layers = match model {
        "gemma-dense-fits" => 50,
        "gemma-dense-large" => 265,
        _ => {
            eprintln!("unknown model {model:?}; expected gemma-dense-fits or gemma-dense-large");
            return 2;
        }
    };
    let topology = match super::probe::read_costs(costs) {
        Ok(costs) => costs,
        Err(error) => {
            eprintln!("cannot read costs {costs}: {error}");
            return 1;
        }
    };
    let mut oracles = OracleRegistry::new();
    if let Err(error) = moxie_oracles::register(&mut oracles) {
        eprintln!("cannot register host graph oracles: {error}");
        return 1;
    }
    let text = match Gemma4Text::reduced(dense_config(layers), "plan-comparison") {
        Ok(text) => text,
        Err(error) => {
            eprintln!("invalid {model} configuration: {error}");
            return 1;
        }
    };
    let graph = match text.compose(&oracles, SymbolId(0)) {
        Ok(composition) => composition.graph,
        Err(error) => {
            eprintln!("cannot compose {model}: {error}");
            return 1;
        }
    };
    let plans = match compare_plans(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens,
            generated_tokens,
        },
        &topology,
    ) {
        Ok(plans) => plans,
        Err(error) => {
            eprintln!("cannot compare plans: {error}");
            return 1;
        }
    };

    println!("{model}: prompt={prompt_tokens}, generate={generated_tokens}");
    println!(
        "| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |"
    );
    println!("|---:|---|---|---:|---:|---:|---|");
    for (kind, verdict) in plans {
        match verdict {
            Verdict::Ranked { rank, estimate } => println!(
                "| {rank} | {} | {} | {:.3} | {:.3} | {:.3} | |",
                candidate_name(&kind),
                device_bytes(&estimate),
                estimate.prefill_ms,
                estimate.first_decode_ms,
                estimate.total_ms / 1_000.0
            ),
            Verdict::Rejected { reason } => println!(
                "| — | {} | | | | | {} |",
                candidate_name(&kind),
                reason.replace('|', "\\|")
            ),
        }
    }

    let phase_pairs = match compare_phase_pairs(
        &graph,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        UserWorkload {
            prompt_tokens,
            generated_tokens,
        },
        &topology,
    ) {
        Ok(pairs) => pairs,
        Err(error) => {
            eprintln!("cannot compare phase pairs: {error}");
            return 1;
        }
    };
    println!("phase pairs");
    println!(
        "| Rank | Prefill candidate | Decode candidate | Transition MB | Via host | Prefill ms | First decode ms | Total s |"
    );
    println!("|---:|---|---|---:|---|---:|---:|---:|");
    let mut best_same_placement = None;
    for (pair, verdict) in phase_pairs {
        let PairVerdict::Ranked { rank, estimate } = verdict else {
            continue;
        };
        if pair.prefill == pair.decode && best_same_placement.is_none() {
            best_same_placement = Some((rank, candidate_name(&pair.prefill)));
        }
        if rank > 10 {
            continue;
        }
        println!(
            "| {rank} | {} | {} | {:.3} | {} | {:.3} | {:.3} | {:.3} |",
            candidate_name(&pair.prefill),
            candidate_name(&pair.decode),
            estimate.transition_bytes as f64 / 1_000_000.0,
            if estimate.transition_via_host {
                "yes"
            } else {
                "no"
            },
            estimate.prefill_ms,
            estimate.first_decode_ms,
            estimate.total_ms / 1_000.0,
        );
    }
    if let Some((rank, candidate)) = best_same_placement {
        println!("best same-placement pair: rank {rank} ({candidate})");
    } else {
        println!("best same-placement pair: none");
    }
    0
}

fn dense_config(layers: u32) -> TextConfig {
    TextConfig {
        hidden: 2048,
        layers,
        heads: 16,
        local_kv_heads: 4,
        local_head_dim: 128,
        global_kv_heads: 4,
        global_head_dim: 128,
        intermediate: 8192,
        vocab: 32_000,
        global_stride: 6,
        sliding_window: 4096,
        rms_eps: 1e-6,
        sliding_rope_theta: 10_000.0,
        global_rope_theta: 1_000_000.0,
        global_partial_rotary: Fraction::QUARTER,
        final_logit_softcap: 30.0,
        layer_scalars: vec![1.0; layers as usize],
        embedding_scale: embedding_scale(2048),
        max_trained_position: 262_144,
        moe: None,
    }
}

fn candidate_name(candidate: &CandidateKind) -> String {
    let uuid = |device: DeviceUuid| {
        let uuid = device.to_string();
        uuid[4..12].to_owned()
    };
    match candidate {
        CandidateKind::Single { device } => format!("single+{}", uuid(*device)),
        CandidateKind::Tp2 { devices } => {
            format!("tp2+{}+{}", uuid(devices[0]), uuid(devices[1]))
        }
        CandidateKind::Pipeline { stages } => format!(
            "pipeline+{}",
            stages
                .iter()
                .map(|(devices, (start, end))| format!(
                    "{}@{start}..{end}",
                    devices
                        .iter()
                        .map(|device| uuid(*device))
                        .collect::<Vec<_>>()
                        .join("+")
                ))
                .collect::<Vec<_>>()
                .join("/")
        ),
        CandidateKind::HostExperts { device } => format!("host-experts+{}", uuid(*device)),
    }
}

fn device_bytes(estimate: &Estimate) -> String {
    estimate
        .device_bytes
        .iter()
        .map(|(device, bytes)| {
            let uuid = device.to_string();
            format!(
                "{}={:.3}",
                &uuid[4..12],
                *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}
