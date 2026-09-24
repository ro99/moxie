//! Deterministic, measured-cost comparisons of single, TP2 and pipeline plans.

use std::{collections::BTreeSet, ops::Range};

use moxie_graph::{Graph, OpParams, OracleId, OracleRegistry, Visibility};
use moxie_types::{DeviceUuid, Error, Result, SymbolTable};

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

pub fn compare_plans(
    graph: &Graph,
    oracle: OracleId,
    oracles: &OracleRegistry,
    workload: UserWorkload,
    costs: &TopologyCosts,
) -> Result<Vec<(CandidateKind, Verdict)>> {
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
            Ok(estimate) => ranked.push((kind, estimate)),
            Err(reason) => rejected.push((kind, Verdict::Rejected { reason })),
        }
    }
    ranked.sort_by(|(left_kind, left), (right_kind, right)| {
        left.total_ms
            .total_cmp(&right.total_ms)
            .then_with(|| left_kind.cmp(right_kind))
    });
    let mut output: Vec<_> = ranked
        .into_iter()
        .enumerate()
        .map(|(index, (kind, estimate))| {
            (
                kind,
                Verdict::Ranked {
                    rank: index + 1,
                    estimate,
                },
            )
        })
        .collect();
    output.extend(rejected);
    Ok(output)
}

struct PlanStage {
    devices: Vec<DeviceUuid>,
    nodes: Range<usize>,
    tp2: bool,
}

#[derive(Clone, Copy)]
struct KvRead {
    bytes_per_row: u64,
    window: Option<u64>,
}

struct RankMeter {
    device: DeviceUuid,
    weight_bytes: u64,
    memory_gbps: f64,
    kv: Vec<KvRead>,
}

fn sorted_devices(costs: &TopologyCosts) -> Result<Vec<DeviceUuid>> {
    let mut devices: Vec<_> = costs.devices.iter().map(|cost| cost.device).collect();
    devices.sort_unstable();
    if devices.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("topology contains a duplicate device UUID"));
    }
    if costs.devices.iter().any(|cost| {
        !cost.memory_gbps.is_finite() || cost.memory_gbps <= 0.0 || cost.usable_bytes == 0
    }) {
        return Err(invalid(
            "device memory rates and usable bytes must be positive",
        ));
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
) -> Result<std::result::Result<Estimate, String>> {
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
            StageMeter {
                devices: stage.devices.clone(),
                meters: vec![stage_meter(&stage_graph.graph, stage.devices[0], costs)?],
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

    let compute_ms: f64 = meters
        .iter()
        .map(|stage| stage_time(stage, workload.prompt_tokens))
        .sum();
    let total_decode_ms: f64 = meters
        .iter()
        .map(|stage| sum_stage_decode(stage, workload.prompt_tokens, workload.generated_tokens))
        .sum();
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
    let first_decode_ms = compute_ms + decode_copy_ms;
    let total_ms = prefill_ms + total_decode_ms + decode_copy_ms * workload.generated_tokens as f64;
    if !prefill_ms.is_finite() || !first_decode_ms.is_finite() || !total_ms.is_finite() {
        return Err(invalid("estimated time is not finite"));
    }
    Ok(Ok(Estimate {
        device_bytes: device_bytes.into_iter().collect(),
        prefill_ms,
        first_decode_ms,
        total_ms,
    }))
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
                let meter = stage_meter(&local.graph, devices[0], costs)
                    .map_err(|error| error.to_string())?;
                for rank in &mut meters {
                    add_meter(rank, &meter)?;
                }
            }
            Stage::Local { nodes, join } => {
                let output = match join {
                    Join::Gather { output } | Join::Reduce { output } => *output,
                };
                for ((device, part), rank) in devices.iter().zip(&lowering.ranks).zip(&mut meters) {
                    let local = build_stage_graph(
                        &stage.graph,
                        Some(part),
                        nodes.clone(),
                        Some(output),
                        oracle,
                        oracles,
                    )
                    .map_err(|error| error.to_string())?;
                    add_meter(
                        rank,
                        &stage_meter(&local.graph, *device, costs).map_err(|e| e.to_string())?,
                    )?;
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
    let kv = graph
        .nodes()
        .iter()
        .filter_map(|node| match node.params {
            OpParams::Attention { visibility, .. } => Some((node, visibility)),
            _ => None,
        })
        .map(|(node, visibility)| {
            let width = width_of(node.inputs[1])?
                .checked_add(width_of(node.inputs[2])?)
                .and_then(|width| width.checked_mul(2))
                .ok_or_else(|| invalid("KV bytes per row overflowed"))?;
            Ok(KvRead {
                bytes_per_row: width,
                window: match visibility {
                    Visibility::SlidingWindow { window } => Some(window),
                    Visibility::Causal => None,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RankMeter {
        device,
        weight_bytes,
        memory_gbps: costs.device(device).unwrap().memory_gbps,
        kv,
    })
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

fn rank_time(rank: &RankMeter, context: u64) -> f64 {
    (rank.weight_bytes as f64 + kv_read(&rank.kv, context)) / (rank.memory_gbps * 1_000_000.0)
}

fn stage_time(stage: &StageMeter, context: u64) -> f64 {
    stage
        .meters
        .iter()
        .map(|rank| rank_time(rank, context))
        .fold(0.0, f64::max)
}

fn sum_stage_decode(stage: &StageMeter, first: u64, count: u64) -> f64 {
    // ponytail: generated-token counts are small; direct summation is clearer.
    (0..count).map(|step| stage_time(stage, first + step)).sum()
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
