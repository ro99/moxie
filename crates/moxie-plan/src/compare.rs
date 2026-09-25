//! Deterministic, measured-cost comparisons of single, TP2 and pipeline plans.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use moxie_graph::{Graph, OpParams, OracleId, OracleRegistry, Visibility};
use moxie_types::{DeviceUuid, Error, Result, SymbolTable};

use crate::tensor_parallel::kv_head_range;
use crate::{
    Endpoint, Join, Stage, TopologyCosts, build_stage_graph, lower_pipeline, lower_tensor_parallel,
    value_bytes,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserWorkload {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateKind {
    Single {
        device: DeviceUuid,
    },
    Tp2 {
        devices: [DeviceUuid; 2],
    },
    /// One stage per entry; a stage is one device, or two for a TP2 stage.
    Pipeline {
        stages: Vec<(Vec<DeviceUuid>, (usize, usize))>,
    },
    /// Host-owned experts beside one device, which have no measured compute path.
    HostExperts {
        device: DeviceUuid,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    /// Resident-byte fit: weights plus KV at full context, excluding activation
    /// and peer scratch buffers; this is not full execution admission.
    pub device_bytes: Vec<(DeviceUuid, u64)>,
    pub prefill_ms: f64,
    /// Decode latency at `context = prompt_tokens`.
    pub first_decode_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Ranked { rank: usize, estimate: Estimate },
    Rejected { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhasePair {
    pub prefill: CandidateKind,
    pub decode: CandidateKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairEstimate {
    pub device_bytes: Vec<(DeviceUuid, u64)>,
    pub prefill_ms: f64,
    pub transition_bytes: u64,
    pub transition_ms: f64,
    pub transition_via_host: bool,
    pub first_decode_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PairVerdict {
    Ranked { rank: usize, estimate: PairEstimate },
    Rejected { reason: String },
}

pub fn compare_plans(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
    workload: UserWorkload,
    costs: &TopologyCosts,
) -> Result<Vec<(CandidateKind, Verdict)>> {
    Ok(compare_candidates(graph, oracle, oracles, workload, costs)?
        .into_iter()
        .map(|candidate| (candidate.kind, candidate.verdict))
        .collect())
}

fn compare_candidates(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
    workload: UserWorkload,
    costs: &TopologyCosts,
) -> Result<Vec<EvaluatedCandidate>> {
    let full_context = workload
        .prompt_tokens
        .checked_add(workload.generated_tokens)
        .ok_or_else(|| invalid("prompt plus generated token count overflowed"))?;
    if graph.nodes().is_empty() || costs.devices.is_empty() {
        return Err(invalid(
            "a nonempty graph and measured devices are required",
        ));
    }
    let devices = sorted_devices(costs)?;
    let mut pairs = Vec::new();
    for i in 0..devices.len() {
        for j in i + 1..devices.len() {
            let pair = [devices[i], devices[j]];
            if has_both_peer_links(costs, pair[0], pair[1]) {
                pairs.push(pair);
            }
        }
    }
    let tp = lower_tensor_parallel(graph, 2);

    let legal_cuts: Vec<_> = (1..graph.nodes().len())
        .filter(|&cut| lower_pipeline(graph, &[cut]).is_ok())
        .collect();
    if legal_cuts.is_empty() {
        return Err(invalid("the graph has no legal pipeline cut"));
    }
    let prefix_bytes = prefix_weight_bytes(graph)?;

    let mut work = Vec::<(CandidateKind, Option<String>)>::new();
    for &device in &devices {
        work.push((CandidateKind::Single { device }, None));
    }
    for pair in &pairs {
        let kind = CandidateKind::Tp2 { devices: *pair };
        match &tp {
            Ok(_) => work.push((kind, None)),
            Err(error) => work.push((kind, Some(error.to_string()))),
        }
    }
    for ordering in permutations(&devices) {
        push_pipeline(
            &mut work,
            graph,
            &ordering
                .iter()
                .map(|device| vec![*device])
                .collect::<Vec<_>>(),
            &legal_cuts,
            &prefix_bytes,
            costs,
        )?;
    }
    for pair in &pairs {
        let rest: Vec<_> = devices
            .iter()
            .copied()
            .filter(|device| !pair.contains(device))
            .collect();
        for ordering in permutations(&rest) {
            let mut groups = vec![pair.to_vec()];
            groups.extend(ordering.into_iter().map(|device| vec![device]));
            push_pipeline(&mut work, graph, &groups, &legal_cuts, &prefix_bytes, costs)?;
        }
    }
    if graph
        .nodes()
        .iter()
        .any(|node| matches!(node.params, OpParams::ExpertMlp { .. }))
    {
        for &device in &devices {
            work.push((
                CandidateKind::HostExperts { device },
                Some("host-owned experts: no measured host compute cost (M6)".into()),
            ));
        }
    }

    let mut ranked = Vec::new();
    let mut rejected = Vec::new();
    for (kind, refusal) in work {
        if let Some(reason) = refusal {
            rejected.push((kind, Verdict::Rejected { reason }));
            continue;
        }
        match estimate(graph, oracle, oracles, workload, full_context, costs, &kind)? {
            Ok((estimate, meters)) => ranked.push((kind, estimate, meters)),
            Err(reason) => rejected.push((kind, Verdict::Rejected { reason })),
        }
    }
    ranked.sort_by(|(left_kind, left, _), (right_kind, right, _)| {
        left.total_ms
            .total_cmp(&right.total_ms)
            .then_with(|| left_kind.cmp(right_kind))
    });
    let mut output: Vec<_> = ranked
        .into_iter()
        .enumerate()
        .map(|(index, (kind, estimate, meters))| EvaluatedCandidate {
            kind,
            verdict: Verdict::Ranked {
                rank: index + 1,
                estimate,
            },
            meters: Some(meters),
        })
        .collect();
    output.extend(
        rejected
            .into_iter()
            .map(|(kind, verdict)| EvaluatedCandidate {
                kind,
                verdict,
                meters: None,
            }),
    );
    Ok(output)
}

struct EstimableCandidate {
    kind: CandidateKind,
    estimate: Estimate,
    meters: Vec<RankMeter>,
}

/// Rank prefill/decode placement pairs for one turn.
///
/// A multi-turn pair also moves the new history back to the prefill placement
/// each turn, which is not estimated. ponytail: add a turn sequence before
/// pricing those return transfers.
pub fn compare_phase_pairs(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
    workload: UserWorkload,
    costs: &TopologyCosts,
) -> Result<Vec<(PhasePair, PairVerdict)>> {
    let candidates = compare_candidates(graph, oracle, oracles, workload, costs)?;
    let candidates: Vec<_> = candidates
        .into_iter()
        .filter_map(
            |EvaluatedCandidate {
                 kind,
                 verdict,
                 meters,
             }| match (verdict, meters) {
                (Verdict::Ranked { estimate, .. }, Some(meters)) => Some(EstimableCandidate {
                    kind,
                    estimate,
                    meters,
                }),
                _ => None,
            },
        )
        .collect();
    let full_context = workload
        .prompt_tokens
        .checked_add(workload.generated_tokens)
        .ok_or_else(|| invalid("prompt plus generated token count overflowed"))?;
    let mut ranked = Vec::new();
    let mut rejected = Vec::new();
    for prefill in &candidates {
        for decode in &candidates {
            let pair = PhasePair {
                prefill: prefill.kind.clone(),
                decode: decode.kind.clone(),
            };
            if prefill.kind == decode.kind {
                ranked.push((
                    pair,
                    PairEstimate {
                        device_bytes: prefill.estimate.device_bytes.clone(),
                        prefill_ms: prefill.estimate.prefill_ms,
                        transition_bytes: 0,
                        transition_ms: 0.0,
                        transition_via_host: false,
                        first_decode_ms: prefill.estimate.first_decode_ms,
                        total_ms: prefill.estimate.total_ms,
                    },
                ));
                continue;
            }
            match pair_estimate(prefill, decode, workload.prompt_tokens, full_context, costs)? {
                Ok(estimate) => ranked.push((pair, estimate)),
                Err(reason) => rejected.push((pair, reason)),
            }
        }
    }
    ranked.sort_by(|(left_pair, left), (right_pair, right)| {
        left.total_ms
            .total_cmp(&right.total_ms)
            .then_with(|| left_pair.prefill.cmp(&right_pair.prefill))
            .then_with(|| left_pair.decode.cmp(&right_pair.decode))
    });
    let mut output: Vec<_> = ranked
        .into_iter()
        .enumerate()
        .map(|(index, (pair, estimate))| {
            (
                pair,
                PairVerdict::Ranked {
                    rank: index + 1,
                    estimate,
                },
            )
        })
        .collect();
    output.extend(
        rejected
            .into_iter()
            .map(|(pair, reason)| (pair, PairVerdict::Rejected { reason })),
    );
    Ok(output)
}

#[derive(Default)]
struct DeviceKvMeter {
    weight_bytes: u64,
    kv: Vec<KvRead>,
}

fn aggregate_meters(meters: &[RankMeter]) -> Result<BTreeMap<DeviceUuid, DeviceKvMeter>> {
    let mut devices = BTreeMap::<DeviceUuid, DeviceKvMeter>::new();
    for meter in meters {
        let device = devices.entry(meter.device).or_default();
        device.weight_bytes = device
            .weight_bytes
            .checked_add(meter.weight_bytes)
            .ok_or_else(|| invalid("phase-pair weight-byte total overflowed"))?;
        device.kv.extend_from_slice(&meter.kv);
    }
    Ok(devices)
}

fn pair_estimate(
    prefill: &EstimableCandidate,
    decode: &EstimableCandidate,
    prompt_tokens: u64,
    full_context: u64,
    costs: &TopologyCosts,
) -> Result<std::result::Result<PairEstimate, String>> {
    let prefill_meters = aggregate_meters(&prefill.meters)?;
    let decode_meters = aggregate_meters(&decode.meters)?;
    let devices: BTreeSet<_> = prefill_meters
        .keys()
        .chain(decode_meters.keys())
        .copied()
        .collect();
    let mut device_bytes = Vec::with_capacity(devices.len());
    for device in devices {
        let prefill_meter = prefill_meters.get(&device);
        let decode_meter = decode_meters.get(&device);
        let prefill_kv_bytes = kv_storage(
            prefill_meter.map_or(&[], |meter| meter.kv.as_slice()),
            prompt_tokens,
        )?;
        let decode_kv_bytes = kv_storage(
            decode_meter.map_or(&[], |meter| meter.kv.as_slice()),
            full_context,
        )?;
        let bytes = prefill_meter
            .map_or(0, |meter| meter.weight_bytes)
            .checked_add(decode_meter.map_or(0, |meter| meter.weight_bytes))
            .and_then(|bytes| bytes.checked_add(prefill_kv_bytes))
            .and_then(|bytes| bytes.checked_add(decode_kv_bytes))
            .ok_or_else(|| invalid("phase-pair resident-byte total overflowed"))?;
        let usable = costs.device(device).map_or(0, |cost| cost.usable_bytes);
        if bytes > usable {
            return Ok(Err(format!(
                "resident-byte fit needs {bytes} bytes on {device}, which has {usable} usable"
            )));
        }
        device_bytes.push((device, bytes));
    }

    let (transition_bytes, transition_ms, transition_via_host) =
        match transition_cost(&prefill_meters, &decode_meters, prompt_tokens, costs)? {
            Ok(cost) => cost,
            Err(reason) => return Ok(Err(reason)),
        };
    let first_decode_ms = decode.estimate.first_decode_ms;
    let total_ms = prefill.estimate.prefill_ms
        + transition_ms
        + (decode.estimate.total_ms - decode.estimate.prefill_ms);
    if !total_ms.is_finite() {
        return Err(invalid("phase-pair time estimate is not finite"));
    }
    Ok(Ok(PairEstimate {
        device_bytes,
        prefill_ms: prefill.estimate.prefill_ms,
        transition_bytes,
        transition_ms,
        transition_via_host,
        first_decode_ms,
        total_ms,
    }))
}

fn transition_cost(
    prefill: &BTreeMap<DeviceUuid, DeviceKvMeter>,
    decode: &BTreeMap<DeviceUuid, DeviceKvMeter>,
    prompt_tokens: u64,
    costs: &TopologyCosts,
) -> Result<std::result::Result<(u64, f64, bool), String>> {
    let mut transfers = BTreeMap::<(DeviceUuid, DeviceUuid), u64>::new();
    let mut total_bytes = 0_u64;
    for (&destination, destination_meter) in decode {
        for read in &destination_meter.kv {
            let head_count = read
                .heads
                .end
                .checked_sub(read.heads.start)
                .filter(|count| *count != 0)
                .ok_or_else(|| invalid("phase-pair KV head range is empty or reversed"))?;
            if read.bytes_per_row % head_count != 0 {
                return Err(invalid("KV bytes per row do not divide evenly by heads"));
            }
            let bytes_per_head = read.bytes_per_row / head_count;
            let visible_rows = read
                .window
                .map_or(prompt_tokens, |window| prompt_tokens.min(window));
            let bytes = bytes_per_head
                .checked_mul(visible_rows)
                .ok_or_else(|| invalid("phase-pair transition-byte count overflowed"))?;
            for head in read.heads.clone() {
                let stored_here = prefill.get(&destination).is_some_and(|meter| {
                    meter
                        .kv
                        .iter()
                        .any(|source| source.layer == read.layer && source.heads.contains(&head))
                });
                if stored_here {
                    continue;
                }
                let Some(source) = prefill
                    .iter()
                    .filter(|(_, meter)| {
                        meter.kv.iter().any(|source| {
                            source.layer == read.layer && source.heads.contains(&head)
                        })
                    })
                    .map(|(&device, _)| device)
                    .min()
                else {
                    return Ok(Err(format!(
                        "no prefill placement stores KV head {head} for layer {}",
                        read.layer
                    )));
                };
                total_bytes = total_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("phase-pair transition-byte total overflowed"))?;
                let transfer = transfers.entry((source, destination)).or_default();
                *transfer = transfer
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("phase-pair grouped transition bytes overflowed"))?;
            }
        }
    }

    let mut total_ms = 0.0;
    let mut via_host = false;
    for ((source, destination), bytes) in transfers {
        let copy_ms = |link: &crate::LinkCost| {
            link.latency_us / 1_000.0 + bytes as f64 / (link.bandwidth_gbps * 1_000_000.0)
        };
        let time = if let Some(link) =
            costs.link(Endpoint::Device(source), Endpoint::Device(destination))
        {
            copy_ms(link)
        } else {
            let Some(download) = costs.link(Endpoint::Device(source), Endpoint::Host) else {
                return Ok(Err(format!("no measured device-to-host path for {source}")));
            };
            let Some(upload) = costs.link(Endpoint::Host, Endpoint::Device(destination)) else {
                return Ok(Err(format!(
                    "no measured host-to-device path for {destination}"
                )));
            };
            via_host = true;
            copy_ms(download) + copy_ms(upload)
        };
        total_ms += time;
    }
    if !total_ms.is_finite() {
        return Err(invalid("phase-pair transition time is not finite"));
    }
    Ok(Ok((total_bytes, total_ms, via_host)))
}

struct PlanStage {
    devices: Vec<DeviceUuid>,
    nodes: Range<usize>,
    tp2: bool,
}

#[derive(Clone)]
struct KvRead {
    layer: u32,
    heads: Range<u64>,
    bytes_per_row: u64,
    head_dim: u64,
    query_heads: u64,
    window: Option<u64>,
}

struct RankMeter {
    device: DeviceUuid,
    weight_bytes: u64,
    memory_gbps: f64,
    linear_flops_per_row: u64,
    tflops: f64,
    kv: Vec<KvRead>,
}

struct EvaluatedCandidate {
    kind: CandidateKind,
    verdict: Verdict,
    meters: Option<Vec<RankMeter>>,
}

fn sorted_devices(costs: &TopologyCosts) -> Result<Vec<DeviceUuid>> {
    let mut devices: Vec<_> = costs.devices.iter().map(|cost| cost.device).collect();
    devices.sort_unstable();
    if devices.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("topology contains a duplicate device UUID"));
    }
    if costs.devices.iter().any(|cost| {
        !cost.memory_gbps.is_finite()
            || cost.memory_gbps <= 0.0
            || !cost.linear_tflops.is_finite()
            || cost.linear_tflops <= 0.0
            || cost.usable_bytes == 0
    }) {
        return Err(invalid("device rates and usable bytes must be positive"));
    }
    let mut links = BTreeSet::new();
    for link in &costs.links {
        if !links.insert((link.from, link.to)) {
            return Err(invalid("topology contains a duplicate directed link"));
        }
        if !link.latency_us.is_finite()
            || link.latency_us < 0.0
            || !link.bandwidth_gbps.is_finite()
            || link.bandwidth_gbps <= 0.0
            || !link.concurrent_gbps.is_finite()
            || link.concurrent_gbps <= 0.0
        {
            return Err(invalid(
                "link latency and rates must be finite and positive",
            ));
        }
    }
    Ok(devices)
}

fn has_both_peer_links(costs: &TopologyCosts, a: DeviceUuid, b: DeviceUuid) -> bool {
    costs
        .link(Endpoint::Device(a), Endpoint::Device(b))
        .is_some()
        && costs
            .link(Endpoint::Device(b), Endpoint::Device(a))
            .is_some()
}

fn permutations(values: &[DeviceUuid]) -> Vec<Vec<DeviceUuid>> {
    fn visit(values: &mut [DeviceUuid], index: usize, out: &mut Vec<Vec<DeviceUuid>>) {
        if index == values.len() {
            out.push(values.to_vec());
            return;
        }
        for next in index..values.len() {
            values.swap(index, next);
            visit(values, index + 1, out);
            values.swap(index, next);
        }
    }
    let mut out = Vec::new();
    visit(&mut values.to_vec(), 0, &mut out);
    out
}

fn push_pipeline(
    out: &mut Vec<(CandidateKind, Option<String>)>,
    graph: &Graph,
    groups: &[Vec<DeviceUuid>],
    legal: &[usize],
    prefix_bytes: &[u64],
    costs: &TopologyCosts,
) -> Result<()> {
    let Some(ranges) = balanced_ranges(graph, groups, legal, prefix_bytes, costs)? else {
        return Ok(());
    };
    let cuts: Vec<_> = ranges
        .iter()
        .take(ranges.len() - 1)
        .map(|range| range.end)
        .collect();
    let kind = pipeline_kind(groups, &ranges);
    out.push((
        kind,
        lower_pipeline(graph, &cuts)
            .err()
            .map(|error| error.to_string()),
    ));
    Ok(())
}

fn pipeline_kind(groups: &[Vec<DeviceUuid>], ranges: &[Range<usize>]) -> CandidateKind {
    CandidateKind::Pipeline {
        stages: groups
            .iter()
            .zip(ranges)
            .map(|(devices, range)| (devices.clone(), (range.start, range.end)))
            .collect(),
    }
}

fn balanced_ranges(
    graph: &Graph,
    groups: &[Vec<DeviceUuid>],
    legal: &[usize],
    prefix_bytes: &[u64],
    costs: &TopologyCosts,
) -> Result<Option<Vec<Range<usize>>>> {
    if legal.len() + 1 < groups.len() {
        return Ok(None);
    }
    let total_bytes = *prefix_bytes.last().unwrap_or(&0);
    let total_capacity = groups.iter().flatten().try_fold(0_u64, |sum, device| {
        sum.checked_add(costs.device(*device).unwrap().usable_bytes)
            .ok_or_else(|| invalid("device-capacity total overflowed"))
    })?;
    let mut previous = 0;
    let mut cuts = Vec::with_capacity(groups.len().saturating_sub(1));
    let mut prior_capacity = 0_u64;
    for (stage, devices) in groups.iter().take(groups.len() - 1).enumerate() {
        prior_capacity = devices.iter().try_fold(prior_capacity, |sum, device| {
            sum.checked_add(costs.device(*device).unwrap().usable_bytes)
                .ok_or_else(|| invalid("cumulative stage capacity overflowed"))
        })?;
        let target = u128::from(total_bytes) * u128::from(prior_capacity);
        let denominator = u128::from(total_capacity);
        let remaining_cuts = groups.len() - stage - 2;
        let Some((_, cut)) = legal
            .iter()
            .copied()
            .take(legal.len().saturating_sub(remaining_cuts))
            .filter(|cut| *cut > previous)
            .map(|cut| {
                let bytes = prefix_bytes[cut];
                let scaled = u128::from(bytes) * denominator;
                (scaled.abs_diff(target), cut)
            })
            .min_by_key(|(distance, cut)| (*distance, *cut))
        else {
            return Ok(None);
        };
        // ponytail: nearest capacity-weighted prefix; exhaustive cut search waits for measured stage costs.
        cuts.push(cut);
        previous = cut;
    }
    let mut start = 0;
    let ranges = cuts
        .into_iter()
        .chain(std::iter::once(graph.nodes().len()))
        .map(|end| {
            let range = start..end;
            start = end;
            range
        })
        .collect();
    Ok(Some(ranges))
}

fn prefix_weight_bytes(graph: &Graph) -> Result<Vec<u64>> {
    let mut seen = BTreeSet::new();
    let mut total = 0_u64;
    let mut prefixes = Vec::with_capacity(graph.nodes().len() + 1);
    prefixes.push(total);
    for node in graph.nodes() {
        for &weight in &node.inputs {
            if graph.weights().contains(&weight) && seen.insert(weight) {
                total = total
                    .checked_add(value_bytes(graph, weight, 1)?)
                    .ok_or_else(|| invalid("weight-byte prefix overflowed"))?;
            }
        }
        prefixes.push(total);
    }
    Ok(prefixes)
}

fn plan_stages(kind: &CandidateKind, node_count: usize) -> Vec<PlanStage> {
    let full = || 0..node_count;
    match kind {
        CandidateKind::Single { device } => vec![plan_stage(vec![*device], full())],
        CandidateKind::Tp2 { devices } => vec![plan_stage(devices.to_vec(), full())],
        CandidateKind::Pipeline { stages } => stages
            .iter()
            .map(|(devices, (start, end))| plan_stage(devices.clone(), *start..*end))
            .collect(),
        CandidateKind::HostExperts { .. } => unreachable!("host candidates are rejected first"),
    }
}

fn plan_stage(devices: Vec<DeviceUuid>, nodes: Range<usize>) -> PlanStage {
    PlanStage {
        tp2: devices.len() == 2,
        devices,
        nodes,
    }
}

fn estimate(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
    workload: UserWorkload,
    full_context: u64,
    costs: &TopologyCosts,
    kind: &CandidateKind,
) -> Result<std::result::Result<(Estimate, Vec<RankMeter>), String>> {
    let stages = plan_stages(kind, graph.nodes().len());
    let mut meters = Vec::with_capacity(stages.len());
    let mut device_bytes = std::collections::BTreeMap::<DeviceUuid, u64>::new();
    for stage in &stages {
        let output = graph.nodes()[stage.nodes.end - 1].output;
        let stage_graph = match build_stage_graph(
            graph,
            None,
            stage.nodes.clone(),
            Some(output),
            oracle,
            oracles,
        ) {
            Ok(stage_graph) => stage_graph,
            Err(error) => return Ok(Err(error.to_string())),
        };
        let meter = if stage.tp2 {
            let lowering = match lower_tensor_parallel(&stage_graph.graph, 2) {
                Ok(lowering) => lowering,
                Err(error) => return Ok(Err(error.to_string())),
            };
            match tensor_parallel_meter(
                &stage_graph,
                &lowering,
                &stage.devices,
                oracle,
                oracles,
                costs,
            ) {
                Ok(meter) => meter,
                Err(reason) => return Ok(Err(reason)),
            }
        } else {
            let mut rank = stage_meter(&stage_graph.graph, stage.devices[0], costs)?;
            if let Err(reason) = remap_kv_layers(&mut rank, &stage_graph, None) {
                return Ok(Err(reason));
            }
            StageMeter {
                devices: stage.devices.clone(),
                meters: vec![rank],
                collectives: Vec::new(),
            }
        };
        for rank in &meter.meters {
            let bytes = rank
                .weight_bytes
                .checked_add(kv_storage(&rank.kv, full_context)?)
                .ok_or_else(|| invalid("per-device resident-byte total overflowed"))?;
            let total = device_bytes.entry(rank.device).or_default();
            *total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("per-device staged-byte total overflowed"))?;
        }
        meters.push(meter);
    }
    for (&device, &bytes) in &device_bytes {
        let usable = costs
            .device(device)
            .map(|cost| cost.usable_bytes)
            .unwrap_or(0);
        if bytes > usable {
            return Ok(Err(format!(
                "resident-byte fit needs {bytes} bytes on {device}, which has {usable} usable"
            )));
        }
    }

    let compute_ms = meters.iter().try_fold(0.0, |sum, stage| {
        Ok::<_, Error>(sum + stage_time(stage, workload.prompt_tokens, workload.prompt_tokens)?)
    })?;
    let first_decode_compute_ms = meters.iter().try_fold(0.0, |sum, stage| {
        Ok::<_, Error>(sum + stage_time(stage, 1, workload.prompt_tokens)?)
    })?;
    let total_decode_ms = meters.iter().try_fold(0.0, |sum, stage| {
        Ok::<_, Error>(
            sum + sum_stage_decode(stage, workload.prompt_tokens, workload.generated_tokens)?,
        )
    })?;
    let prefill_transfer = match transfer_time(graph, &stages, costs, workload.prompt_tokens) {
        Ok(time) => time,
        Err(reason) => return Ok(Err(reason)),
    };
    let decode_transfer = match transfer_time(graph, &stages, costs, 1) {
        Ok(time) => time,
        Err(reason) => return Ok(Err(reason)),
    };
    let first_collectives: f64 = meters
        .iter()
        .map(|stage| collective_time(stage, costs, 1))
        .sum::<std::result::Result<_, _>>()
        .map_err(invalid)?;
    let prefill_collectives: f64 = meters
        .iter()
        .map(|stage| collective_time(stage, costs, workload.prompt_tokens))
        .sum::<std::result::Result<_, _>>()
        .map_err(invalid)?;
    let decode_copy_ms = decode_transfer + first_collectives;
    let prefill_ms = compute_ms + prefill_transfer + prefill_collectives;
    let first_decode_ms = first_decode_compute_ms + decode_copy_ms;
    let total_ms = prefill_ms + total_decode_ms + decode_copy_ms * workload.generated_tokens as f64;
    if !prefill_ms.is_finite() || !first_decode_ms.is_finite() || !total_ms.is_finite() {
        return Err(invalid("estimated time is not finite"));
    }
    let rank_meters = meters.into_iter().flat_map(|stage| stage.meters).collect();
    Ok(Ok((
        Estimate {
            device_bytes: device_bytes.into_iter().collect(),
            prefill_ms,
            first_decode_ms,
            total_ms,
        },
        rank_meters,
    )))
}

struct StageMeter {
    devices: Vec<DeviceUuid>,
    meters: Vec<RankMeter>,
    collectives: Vec<u64>,
}

fn tensor_parallel_meter(
    stage: &crate::StageGraph,
    lowering: &crate::TensorParallelLowering,
    devices: &[DeviceUuid],
    oracle: OracleId,
    oracles: &OracleRegistry,
    costs: &TopologyCosts,
) -> std::result::Result<StageMeter, String> {
    let mut meters: Vec<_> = devices
        .iter()
        .map(|&device| RankMeter {
            device,
            weight_bytes: 0,
            memory_gbps: costs.device(device).unwrap().memory_gbps,
            linear_flops_per_row: 0,
            tflops: costs.device(device).unwrap().linear_tflops,
            kv: Vec::new(),
        })
        .collect();
    for declared in &lowering.stages {
        match declared {
            Stage::Replicated(nodes) => {
                let output = stage.graph.nodes()[nodes.end - 1].output;
                let local = build_stage_graph(
                    &stage.graph,
                    None,
                    nodes.clone(),
                    Some(output),
                    oracle,
                    oracles,
                )
                .map_err(|error| error.to_string())?;
                let mut meter = stage_meter(&local.graph, devices[0], costs)
                    .map_err(|error| error.to_string())?;
                remap_kv_layers(&mut meter, &local, Some(stage))?;
                for rank in &mut meters {
                    add_meter(rank, &meter)?;
                }
            }
            Stage::Local { nodes, join } => {
                let output = match join {
                    Join::Gather { output } | Join::Reduce { output } => *output,
                };
                for (rank_index, ((device, part), rank)) in devices
                    .iter()
                    .zip(&lowering.ranks)
                    .zip(&mut meters)
                    .enumerate()
                {
                    let local = build_stage_graph(
                        &stage.graph,
                        Some(part),
                        nodes.clone(),
                        Some(output),
                        oracle,
                        oracles,
                    )
                    .map_err(|error| error.to_string())?;
                    let mut local_meter =
                        stage_meter(&local.graph, *device, costs).map_err(|e| e.to_string())?;
                    for read in &mut local_meter.kv {
                        let (global_layer, global_kv_heads) =
                            kv_read_metadata(&local, Some(stage), read.layer)?;
                        read.layer = global_layer;
                        read.heads = kv_head_range(global_kv_heads, 2, rank_index as u64);
                    }
                    add_meter(rank, &local_meter)?;
                }
            }
        }
    }
    let collectives = collective_widths(&stage.graph, lowering)?;
    Ok(StageMeter {
        devices: devices.to_vec(),
        meters,
        collectives,
    })
}

fn add_meter(target: &mut RankMeter, source: &RankMeter) -> std::result::Result<(), String> {
    target.weight_bytes = target
        .weight_bytes
        .checked_add(source.weight_bytes)
        .ok_or_else(|| "rank-local weight-byte total overflowed".to_string())?;
    target.linear_flops_per_row = target
        .linear_flops_per_row
        .checked_add(source.linear_flops_per_row)
        .ok_or_else(|| "rank-local FLOP total overflowed".to_string())?;
    target.kv.extend_from_slice(&source.kv);
    Ok(())
}

fn stage_meter(graph: &Graph, device: DeviceUuid, costs: &TopologyCosts) -> Result<RankMeter> {
    let weight_bytes = graph.weights().iter().try_fold(0_u64, |sum, &weight| {
        sum.checked_add(value_bytes(graph, weight, 1)?)
            .ok_or_else(|| invalid("stage weight-byte total overflowed"))
    })?;
    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), 1);
    let width_of = |value| -> Result<u64> {
        graph
            .spec(value)
            .ok_or_else(|| invalid("activation has no width dimension"))?
            .extent(&symbols)?
            .last()
            .copied()
            .ok_or_else(|| invalid("activation has no width dimension"))
    };
    let linear_flops_per_row = graph.nodes().iter().try_fold(0_u64, |sum, node| {
        let flops = match node.params {
            OpParams::Linear {
                in_features,
                out_features,
                ..
            } => in_features
                .checked_mul(out_features)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| invalid("linear FLOP count overflowed"))?,
            OpParams::VocabProjection { vocab, hidden, .. } => vocab
                .checked_mul(hidden)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| invalid("vocabulary projection FLOP count overflowed"))?,
            OpParams::ExpertMlp { experts, top_k, .. } => {
                if experts == 0 {
                    return Err(invalid("expert count is zero while metering FLOPs"));
                }
                let expert_weight_elements = node
                    .inputs
                    .iter()
                    .copied()
                    .filter(|input| graph.weights().contains(input))
                    .try_fold(0_u64, |total, input| {
                        let shape = graph
                            .spec(input)
                            .ok_or_else(|| invalid("expert weight has no tensor shape"))?
                            .extent(&symbols)?;
                        let elements = shape.into_iter().try_fold(1_u64, |product, extent| {
                            product
                                .checked_mul(extent)
                                .ok_or_else(|| invalid("expert weight element count overflowed"))
                        })?;
                        total
                            .checked_add(elements)
                            .ok_or_else(|| invalid("expert weight element total overflowed"))
                    })?;
                let per_expert = expert_weight_elements
                    .checked_div(experts)
                    .filter(|_| expert_weight_elements % experts == 0)
                    .ok_or_else(|| invalid("expert weights do not divide evenly by experts"))?;
                per_expert
                    .checked_mul(top_k)
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| invalid("expert FLOP count overflowed"))?
            }
            _ => 0,
        };
        sum.checked_add(flops)
            .ok_or_else(|| invalid("linear FLOPs per row overflowed"))
    })?;
    let kv = graph
        .nodes()
        .iter()
        .filter_map(|node| match node.params {
            OpParams::Attention {
                heads,
                head_dim,
                visibility,
                layer,
                kv_heads,
                ..
            } => Some((node, visibility, layer, kv_heads, heads, head_dim)),
            _ => None,
        })
        .map(
            |(node, visibility, layer, kv_heads, query_heads, head_dim)| {
                let width = width_of(node.inputs[1])?
                    .checked_add(width_of(node.inputs[2])?)
                    .and_then(|width| width.checked_mul(2))
                    .ok_or_else(|| invalid("KV bytes per row overflowed"))?;
                Ok(KvRead {
                    layer,
                    heads: 0..kv_heads,
                    bytes_per_row: width,
                    head_dim,
                    query_heads,
                    window: match visibility {
                        Visibility::SlidingWindow { window } => Some(window),
                        Visibility::Causal => None,
                    },
                })
            },
        )
        .collect::<Result<Vec<_>>>()?;
    let device_cost = costs.device(device).unwrap();
    Ok(RankMeter {
        device,
        weight_bytes,
        memory_gbps: device_cost.memory_gbps,
        linear_flops_per_row,
        tflops: device_cost.linear_tflops,
        kv,
    })
}

fn remap_kv_layers(
    meter: &mut RankMeter,
    stage: &crate::StageGraph,
    parent: Option<&crate::StageGraph>,
) -> std::result::Result<(), String> {
    for read in &mut meter.kv {
        read.layer = kv_read_metadata(stage, parent, read.layer)?.0;
    }
    Ok(())
}

fn kv_read_metadata(
    stage: &crate::StageGraph,
    parent: Option<&crate::StageGraph>,
    local_layer: u32,
) -> std::result::Result<(u32, u64), String> {
    let local_node = stage
        .graph
        .nodes()
        .iter()
        .find(
            |node| matches!(node.params, OpParams::Attention { layer, .. } if layer == local_layer),
        )
        .ok_or_else(|| format!("attention layer {local_layer} is missing from its stage graph"))?;
    let source_layer = stage
        .state_layers
        .get(&local_node.id)
        .copied()
        .ok_or_else(|| format!("attention layer {local_layer} has no source-layer mapping"))?;
    let (global_layer, kv_heads) = if let Some(parent) = parent {
        let parent_node = parent
            .graph
            .nodes()
            .iter()
            .find(|node| {
                matches!(node.params, OpParams::Attention { layer, .. } if layer == source_layer)
            })
            .ok_or_else(|| {
                format!("attention layer {source_layer} is missing from its parent stage graph")
            })?;
        let global_layer = parent
            .state_layers
            .get(&parent_node.id)
            .copied()
            .ok_or_else(|| format!("attention layer {source_layer} has no global-layer mapping"))?;
        let OpParams::Attention { kv_heads, .. } = parent_node.params else {
            unreachable!("the matched parent node is attention");
        };
        (global_layer, kv_heads)
    } else {
        let OpParams::Attention { kv_heads, .. } = local_node.params else {
            unreachable!("the matched stage node is attention");
        };
        (source_layer, kv_heads)
    };
    Ok((global_layer, kv_heads))
}

fn kv_storage(kv: &[KvRead], context: u64) -> Result<u64> {
    kv.iter().try_fold(0_u64, |sum, read| {
        let visible = read.window.map_or(context, |window| context.min(window));
        sum.checked_add(
            read.bytes_per_row
                .checked_mul(visible)
                .ok_or_else(|| invalid("KV resident-byte count overflowed"))?,
        )
        .ok_or_else(|| invalid("KV resident-byte total overflowed"))
    })
}

fn kv_read(kv: &[KvRead], context: u64) -> f64 {
    kv.iter()
        .map(|read| {
            read.bytes_per_row as f64 * read.window.map_or(context, |w| context.min(w)) as f64
        })
        .sum()
}

fn rank_time(rank: &RankMeter, rows: u64, context: u64) -> Result<f64> {
    let memory_ms =
        (rank.weight_bytes as f64 + kv_read(&rank.kv, context)) / (rank.memory_gbps * 1_000_000.0);
    let linear_flops = rows
        .checked_mul(rank.linear_flops_per_row)
        .ok_or_else(|| invalid("step linear FLOP count overflowed"))?;
    let attention_flops = rank.kv.iter().try_fold(0_u64, |sum, read| {
        let keys = sum_capped_keys(rows, context, read.window)?;
        let flops = read
            .query_heads
            .checked_mul(read.head_dim)
            .and_then(|value| value.checked_mul(keys))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| invalid("attention FLOP count overflowed"))?;
        sum.checked_add(flops)
            .ok_or_else(|| invalid("attention FLOP total overflowed"))
    })?;
    let flops = linear_flops
        .checked_add(attention_flops)
        .ok_or_else(|| invalid("step FLOP total overflowed"))?;
    let compute_ms = flops as f64 / (rank.tflops * 1e9);
    if !memory_ms.is_finite() || !compute_ms.is_finite() {
        return Err(invalid("step time estimate is not finite"));
    }
    // ponytail: max assumes perfect overlap between memory traffic and compute.
    Ok(memory_ms.max(compute_ms))
}

fn sum_capped_keys(rows: u64, context: u64, window: Option<u64>) -> Result<u64> {
    if rows == 0 {
        return Ok(0);
    }
    let first = context
        .checked_sub(rows)
        .and_then(|position| position.checked_add(1))
        .ok_or_else(|| invalid("attention row range is outside its context"))?;
    let Some(window) = window else {
        return sum_positions(first, context);
    };
    let uncapped_end = context.min(window);
    let uncapped = if first <= uncapped_end {
        sum_positions(first, uncapped_end)?
    } else {
        0
    };
    let capped = if context > window {
        let capped_first = first.max(
            window
                .checked_add(1)
                .ok_or_else(|| invalid("attention window position overflowed"))?,
        );
        let count = context
            .checked_sub(capped_first)
            .and_then(|positions| positions.checked_add(1))
            .ok_or_else(|| invalid("attention capped row count overflowed"))?;
        window
            .checked_mul(count)
            .ok_or_else(|| invalid("attention capped key total overflowed"))?
    } else {
        0
    };
    uncapped
        .checked_add(capped)
        .ok_or_else(|| invalid("attention key total overflowed"))
}

fn sum_positions(first: u64, last: u64) -> Result<u64> {
    if first > last {
        return Ok(0);
    }
    let count = u128::from(
        last.checked_sub(first)
            .and_then(|distance| distance.checked_add(1))
            .ok_or_else(|| invalid("attention row count overflowed"))?,
    );
    let endpoints = u128::from(first) + u128::from(last);
    let (left, right) = if count.is_multiple_of(2) {
        (count / 2, endpoints)
    } else {
        (count, endpoints / 2)
    };
    let sum = left
        .checked_mul(right)
        .ok_or_else(|| invalid("attention key sum overflowed"))?;
    u64::try_from(sum).map_err(|_| invalid("attention key sum exceeds u64"))
}

fn stage_time(stage: &StageMeter, rows: u64, context: u64) -> Result<f64> {
    stage.meters.iter().try_fold(0.0_f64, |maximum, rank| {
        Ok(maximum.max(rank_time(rank, rows, context)?))
    })
}

fn sum_stage_decode(stage: &StageMeter, first: u64, count: u64) -> Result<f64> {
    // ponytail: generated-token counts are small; direct summation is clearer.
    (0..count).try_fold(0.0, |sum, step| {
        let context = first
            .checked_add(step)
            .ok_or_else(|| invalid("decode context overflowed"))?;
        let total = sum + stage_time(stage, 1, context)?;
        if total.is_finite() {
            Ok(total)
        } else {
            Err(invalid("total decode estimate is not finite"))
        }
    })
}

fn collective_widths(
    graph: &Graph,
    lowering: &crate::TensorParallelLowering,
) -> std::result::Result<Vec<u64>, String> {
    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), 1);
    lowering
        .stages
        .iter()
        .filter_map(|stage| match stage {
            Stage::Local { join, .. } => Some(join),
            Stage::Replicated(_) => None,
        })
        .map(|join| {
            let output = match join {
                Join::Gather { output } | Join::Reduce { output } => *output,
            };
            let width = graph
                .spec(output)
                .ok_or_else(|| "activation has no width dimension".to_string())?
                .extent(&symbols)
                .map_err(|error| error.to_string())?
                .last()
                .copied()
                .ok_or_else(|| "activation has no width dimension".to_string())?;
            Ok(match join {
                Join::Gather { .. } if width % 2 == 0 => width / 2,
                Join::Gather { .. } => return Err("gather width does not divide over TP2".into()),
                Join::Reduce { .. } => width,
            })
        })
        .collect()
}

fn collective_time(
    stage: &StageMeter,
    costs: &TopologyCosts,
    rows: u64,
) -> std::result::Result<f64, String> {
    if stage.collectives.is_empty() {
        return Ok(0.0);
    }
    let [a, b] = stage.devices.as_slice() else {
        return Err("TP2 collective requires exactly two devices".into());
    };
    let link = |from, to| {
        costs
            .link(Endpoint::Device(from), Endpoint::Device(to))
            .ok_or_else(|| format!("no measured peer path from {from} to {to}"))
    };
    let ab = link(*a, *b)?;
    let ba = link(*b, *a)?;
    stage.collectives.iter().try_fold(0.0, |total, width| {
        let bytes = rows
            .checked_mul(*width)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "TP2 collective extent overflowed".to_string())?;
        let one_way = |link: &crate::LinkCost| {
            link.latency_us / 1_000.0 + bytes as f64 / (link.concurrent_gbps * 1_000_000.0)
        };
        Ok(total + one_way(ab).max(one_way(ba)))
    })
}

fn transfer_time(
    graph: &Graph,
    stages: &[PlanStage],
    costs: &TopologyCosts,
    rows: u64,
) -> std::result::Result<f64, String> {
    let mut total = 0.0;
    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), 1);
    for (source, destination) in stages.windows(2).map(|pair| (&pair[0], &pair[1])) {
        let (src, dst) = (source.devices[0], destination.devices[0]);
        let output = graph.nodes()[source.nodes.end - 1].output;
        let width = graph
            .spec(output)
            .ok_or_else(|| "activation has no width dimension".to_string())?
            .extent(&symbols)
            .map_err(|error| error.to_string())?
            .last()
            .copied()
            .ok_or_else(|| "activation has no width dimension".to_string())?;
        let bytes = rows
            .checked_mul(width)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| "pipeline handoff extent overflowed".to_string())?;
        let upload = costs
            .link(Endpoint::Host, Endpoint::Device(dst))
            .ok_or_else(|| format!("no measured host-to-device path for {dst}"))?;
        let download = costs
            .link(Endpoint::Device(src), Endpoint::Host)
            .ok_or_else(|| format!("no measured device-to-host path for {src}"))?;
        let copy_ms = |link: &crate::LinkCost| {
            link.latency_us / 1_000.0 + bytes as f64 / (link.bandwidth_gbps * 1_000_000.0)
        };
        total += copy_ms(download) + copy_ms(upload);
    }
    Ok(total)
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field: "plan_comparison",
        detail: detail.into(),
    }
}
