//! Task 0062's per-rank-thread TP2 device gate.
//!
//! The fixture is deliberately small and synthetic.  It runs the full dense
//! graph on one 3090 through the split-aware ordinary linear kernel, then runs
//! the lowered stage graph on persistent workers that own the 3090 pair.
#![cfg(all(feature = "paged-attention-binding", feature = "nccl"))]

use core::ffi::{c_int, c_void};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::paged_attention::device::commit_paged_state;
use moxie_executor::{
    DenseGraphStep, DenseRankWorkerConfig, DenseRankWorkers, PageGeometry, PagedAttentionRun,
    PipelineStageWorker, PipelineWorkers, SelectedReservedPlan, SoloRankWorker,
    SoloRankWorkerConfig, Staging,
};
use moxie_format::bf16::{bf16_bits_to_f32, f32_to_bf16_bits};
use moxie_graph::{
    Bindings, CombineReductionOrder, ExpertOwnership, Graph, LinearReductionOrder, NodeId,
    OracleRegistry, ValueId, ValueRole,
};
use moxie_interp::{Cancel, Interpreter, KvCache};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{
    Phase, PipelineLowering, ResourceWorkload, StageGraph, StageWeight, TensorParallelLowering,
    build_stage_graph, lower_pipeline, lower_selected_ordered, lower_tensor_parallel,
};
use moxie_state::{DeviceKvSequence, KvGeometry, LayerKv, Retention, SequenceState, StateKind};
use moxie_types::{
    DeviceCapability, DeviceUuid, Dim, Error, Precision, RankId, SemanticKernelOp, TensorLayout,
};

const DEADLINE: Duration = Duration::from_secs(20);
const RANK_UUIDS: [&str; 2] = [
    "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9",
    "GPU-81fe4578-59b2-37c4-421e-287cdac78704",
];
static DEVICE_TEST: Mutex<()> = Mutex::new(());

// Driver faults, by interposing two CUDA driver symbols in this test binary
// (0059's pattern). Armed from the binding callback, so they land mid-step.
const NO_FAULT: u8 = 0;
/// The next kernel launch is refused as an invalid value (N2).
const LAUNCH_FAULT: u8 = 1;
/// The next paged-attention kernel launch is refused, quarantining its run
/// with the stage's query and output ranges (R3-2).
const ATTENTION_LAUNCH_FAULT: u8 = 2;
/// The next host-to-device copy fails.
const UPLOAD_FAULT: u8 = 3;
/// The next device-to-host copy (an attention stage reading its keys back
/// before the append) arms `UPLOAD_FAULT`. The append's first upload is then
/// the page-table publication, whose failure holds the run's admitted
/// upload buffer (R5-1).
const TABLE_UPLOAD_FAULT: u8 = 4;

static DRIVER_FAULT: AtomicU8 = AtomicU8::new(NO_FAULT);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const core::ffi::c_char) -> *mut c_void;
}

fn real(symbol: &core::ffi::CStr) -> *mut c_void {
    // SAFETY: RTLD_NEXT resolves the real CUDA symbol after this executable.
    let found = unsafe { dlsym((-1isize) as *mut c_void, symbol.as_ptr()) };
    assert!(!found.is_null(), "missing real {symbol:?}");
    found
}

/// One-shot: consume `fault` if it is armed.
fn fires(fault: u8) -> bool {
    DRIVER_FAULT
        .compare_exchange(fault, NO_FAULT, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

type Launch = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn cuLaunchKernel(
    f: *mut c_void,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    by: u32,
    bz: u32,
    shared: u32,
    stream: *mut c_void,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> c_int {
    if fires(LAUNCH_FAULT) || (is_attention(f) && fires(ATTENTION_LAUNCH_FAULT)) {
        return 1;
    }
    // SAFETY: as above.
    unsafe {
        let launch: Launch = std::mem::transmute(real(c"cuLaunchKernel"));
        launch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra)
    }
}

/// Whether a launched function is the paged-attention kernel, by the
/// driver's own name for it.
fn is_attention(function: *mut c_void) -> bool {
    type Name = unsafe extern "C" fn(*mut *const core::ffi::c_char, *mut c_void) -> c_int;
    let mut name = core::ptr::null();
    // SAFETY: `cuFuncGetName`'s ABI; the returned name is owned by the driver.
    unsafe {
        let get: Name = std::mem::transmute(real(c"cuFuncGetName"));
        get(&mut name, function) == 0
            && core::ffi::CStr::from_ptr(name).to_bytes()
                == moxie_kernels::PAGED_ATTENTION.as_bytes()
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    destination: u64,
    source: *const c_void,
    bytes: usize,
    stream: *mut c_void,
) -> c_int {
    if fires(UPLOAD_FAULT) {
        return 1;
    }
    type Copy = unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> c_int;
    // SAFETY: as above.
    unsafe {
        let copy: Copy = std::mem::transmute(real(c"cuMemcpyHtoDAsync_v2"));
        copy(destination, source, bytes, stream)
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyDtoH_v2(destination: *mut c_void, source: u64, bytes: usize) -> c_int {
    let _ = DRIVER_FAULT.compare_exchange(
        TABLE_UPLOAD_FAULT,
        UPLOAD_FAULT,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    // SAFETY: as above.
    unsafe {
        let copy: unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int =
            std::mem::transmute(real(c"cuMemcpyDtoH_v2"));
        copy(destination, source, bytes)
    }
}

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture(
    routed: bool,
) -> (
    moxie_cli::fixture::Fixture,
    moxie_models::gemma4::TextConfig,
) {
    let mut config = moxie_cli::gemma::Shape::A.config();
    config.heads = 8;
    config.local_kv_heads = 4;
    config.global_kv_heads = 1;
    config.vocab = 12;
    if routed {
        let mut moe = moxie_cli::gemma::Shape::C
            .config()
            .moe
            .expect("Shape C MoE config");
        moe.experts = 8;
        config.moe = Some(moe);
    }
    let mut fixture = moxie_cli::gemma::build_with_config(config.clone()).expect("TP2 fixture");
    if routed {
        for weight in fixture.graph.weights() {
            let Some(name) = fixture.graph.name(*weight) else {
                continue;
            };
            let tensor = fixture
                .weights
                .get(*weight)
                .expect("router weight")
                .as_float()
                .expect("router BF16 weight");
            let shape = tensor.shape().to_vec();
            let len = tensor.data().len();
            if name.starts_with("router_proj.") {
                let width = shape[1];
                let mut values = vec![0.0; len];
                values[0] = 1.0;
                values[4 * width] = -1.0;
                fixture.weights.set(
                    *weight,
                    Value::Float(HostTensor::bf16(values, shape).expect("router projection")),
                );
            } else if name.starts_with("router_scale.") {
                fixture.weights.set(
                    *weight,
                    Value::Float(HostTensor::bf16(vec![1.0; len], shape).expect("router scale")),
                );
            }
        }
    }
    (fixture, config)
}

fn pair_ordinals() -> [u32; 2] {
    let wanted = RANK_UUIDS.map(|uuid| DeviceUuid::parse(uuid).expect("3090 UUID"));
    let mut found = [None, None];
    for ordinal in 0..device_count().expect("enumerate CUDA devices") {
        let capability = query_device(ordinal).expect("query CUDA device");
        for (rank, uuid) in wanted.iter().enumerate() {
            if capability.uuid == *uuid {
                found[rank] = Some(ordinal);
            }
        }
    }
    [
        found[0].expect("first declared 3090 is visible"),
        found[1].expect("second declared 3090 is visible"),
    ]
}

fn concrete_shape(graph: &Graph, value: ValueId, rows: usize) -> Vec<u64> {
    graph
        .spec(value)
        .expect("tensor spec")
        .shape
        .iter()
        .map(|dim| match dim {
            Dim::Const(value) => *value,
            Dim::Symbol(_) => rows as u64,
            other => panic!("unexpected symbolic dimension {other:?}"),
        })
        .collect()
}

fn encode_value(value: &Value) -> Vec<u8> {
    let mut bytes: Vec<u8> = match value {
        Value::Index(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Value::Float(tensor) => tensor
            .data()
            .iter()
            .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
            .collect(),
        Value::Route(_) => panic!("dense TP fixture has no route input"),
    };
    bytes.shrink_to_fit();
    bytes
}

fn owned_full_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<moxie_executor::OwnedBinding> {
    let mut bindings = Vec::new();
    for value in fixture.graph.inputs() {
        let source = if *value == fixture.tokens {
            Value::Index(tokens.to_vec())
        } else if *value == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            panic!("unexpected dense input {}", value.0)
        };
        bindings.push(moxie_executor::OwnedBinding {
            value: *value,
            role: fixture.graph.spec(*value).expect("input spec").role,
            shape: concrete_shape(&fixture.graph, *value, tokens.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    for value in fixture.graph.weights() {
        bindings.push(moxie_executor::OwnedBinding {
            value: *value,
            role: fixture.graph.spec(*value).expect("weight spec").role,
            shape: concrete_shape(&fixture.graph, *value, tokens.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(fixture.weights.get(*value).expect("weight binding")),
        });
    }
    bindings
}

fn host_logits(
    fixture: &moxie_cli::fixture::Fixture,
    tokens: &[u64],
    positions: &[u64],
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
) -> Vec<u32> {
    let mut state = SequenceState::new([StateKind::KvPages]);
    let mut cache = KvCache::for_branch(
        fixture.graph.attention_layers().len(),
        &state,
        moxie_state::ROOT,
    )
    .expect("host KV cache");
    let mut bindings = fixture.weights.clone();
    bindings.set(fixture.tokens, Value::Index(tokens.to_vec()));
    bindings.set(fixture.positions, Value::Index(positions.to_vec()));
    let output = Interpreter::new()
        .run_with_partition_orders(
            &fixture.graph,
            &bindings,
            &mut state,
            moxie_state::ROOT,
            &mut cache,
            &Cancel::never(),
            orders,
            combine_orders,
            &BTreeMap::new(),
        )
        .expect("host declared split")
        .logits;
    output.data().iter().map(|value| value.to_bits()).collect()
}

fn host_route_ids(
    fixture: &moxie_cli::fixture::Fixture,
    tokens: &[u64],
    positions: &[u64],
    linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
) -> Vec<Vec<u32>> {
    let node = fixture
        .graph
        .nodes()
        .iter()
        .find(|node| matches!(node.params, moxie_graph::OpParams::Route { .. }))
        .expect("routed fixture has a Route node");
    let moxie_graph::OpParams::Route {
        hidden,
        experts,
        top_k,
        input,
        score,
        per_expert_scale,
        coefficient,
        selection_bias,
    } = node.params
    else {
        unreachable!();
    };
    let x = prefix_activation(
        fixture,
        node.id.0 as usize,
        node.inputs[0],
        tokens,
        positions,
        linear_orders,
    );
    let weights = |index: usize| {
        fixture
            .weights
            .get(node.inputs[index])
            .expect("router weight")
            .as_float()
            .expect("BF16 router weight")
    };
    let projection = weights(1);
    let gain = weights(2);
    let per_expert = weights(3);
    let selection_bias = selection_bias.then(|| weights(4));
    let spec = moxie_oracles::route::RouterSpec {
        experts: experts as usize,
        top_k: top_k as usize,
        input,
        score,
        coefficient,
    };
    (0..tokens.len())
        .map(|row| {
            let start = row * hidden as usize;
            let activation = &x[start..start + hidden as usize];
            moxie_oracles::route::router_route_row(
                activation,
                matches!(input, moxie_graph::RouterInput::Normalized { .. }).then_some(gain.data()),
                projection.data(),
                per_expert_scale.then_some(per_expert.data()),
                selection_bias.as_ref().map(|scale| scale.data()),
                spec,
            )
            .expect("host router row")
            .experts
        })
        .collect()
}

fn assert_expert_owner_resource(
    fixture: &moxie_cli::fixture::Fixture,
    lowering: &TensorParallelLowering,
    capabilities: &[DeviceCapability; 2],
) {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("oracles");
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let rows = 5;
    let reference_workload = ResourceWorkload {
        phase: Phase::Prefill,
        rows,
        visible_tokens: rows,
        branch_rows: rows,
        output: fixture.graph.output(),
        device: capabilities[0].uuid,
        paged_state_capacity: None,
    };
    let reference_candidate = lower_selected_ordered(
        &fixture.graph,
        reference_workload,
        &capabilities[0],
        &catalogue,
        &lowering.linear_orders,
        &lowering.combine_orders,
        &BTreeMap::new(),
    )
    .expect("full-graph reference plan");
    for stage in &lowering.stages {
        let moxie_plan::Stage::Local { nodes, join } = stage else {
            continue;
        };
        let Some(expert) = fixture.graph.nodes()[nodes.clone()]
            .iter()
            .find(|node| matches!(node.params, moxie_graph::OpParams::ExpertMlp { .. }))
        else {
            continue;
        };
        let output = match join {
            moxie_plan::Join::Gather { output } | moxie_plan::Join::Reduce { output } => *output,
        };
        for (rank, capability) in capabilities.iter().enumerate() {
            let local = build_stage_graph(
                &fixture.graph,
                Some(&lowering.ranks[rank]),
                nodes.clone(),
                Some(output),
                moxie_oracles::HOST_REFERENCE,
                &oracles,
            )
            .expect("rank expert stage");
            let candidate = lower_selected_ordered(
                &local.graph,
                ResourceWorkload {
                    device: capability.uuid,
                    output: local.graph.output(),
                    ..reference_workload
                },
                capability,
                &catalogue,
                &local.linear_orders,
                &local.combine_orders,
                &local.expert_ownership,
            )
            .expect("rank expert-stage plan");
            for original in [expert.inputs[2], expert.inputs[3]] {
                let local_value = local
                    .weights
                    .iter()
                    .find(|weight| weight.original == original)
                    .expect("rank sharded expert weight")
                    .local;
                assert_eq!(
                    candidate
                        .value(local_value)
                        .expect("rank weight bytes")
                        .logical_bytes
                        * 2,
                    reference_candidate
                        .value(original)
                        .expect("reference weight bytes")
                        .logical_bytes,
                    "rank {rank} owns half of expert weight {}",
                    fixture.graph.name(original).unwrap_or("<unnamed>")
                );
            }
        }
    }
}

fn one_block_orders(
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
) -> BTreeMap<NodeId, LinearReductionOrder> {
    orders
        .keys()
        .copied()
        .map(|node| {
            (
                node,
                LinearReductionOrder {
                    blocks: 1,
                    slice: None,
                },
            )
        })
        .collect()
}

fn prefix_activation(
    fixture: &moxie_cli::fixture::Fixture,
    end: usize,
    output: ValueId,
    tokens: &[u64],
    positions: &[u64],
    linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
) -> Vec<f32> {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("prefix oracle");
    let stage = build_stage_graph(
        &fixture.graph,
        None,
        0..end,
        Some(output),
        moxie_oracles::HOST_REFERENCE,
        &oracles,
    )
    .expect("prefix graph");
    let prefix_orders = linear_orders
        .iter()
        .filter(|(id, _)| (id.0 as usize) < end)
        .map(|(id, order)| (*id, *order))
        .collect();
    let mut bindings = Bindings::new();
    for read in &stage.reads {
        let value = if read.original == fixture.tokens {
            Value::Index(tokens.to_vec())
        } else if read.original == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            panic!("prefix has an unexpected external activation");
        };
        bindings.set(read.local, value);
    }
    for weight in &stage.weights {
        bindings.set(weight.local, stage_weight_value(fixture, weight));
    }
    let mut state = SequenceState::new([StateKind::KvPages]);
    let mut cache = KvCache::for_branch(
        stage.graph.attention_layers().len(),
        &state,
        moxie_state::ROOT,
    )
    .expect("prefix cache");
    Interpreter::new()
        .run_with_partition_orders(
            &stage.graph,
            &bindings,
            &mut state,
            moxie_state::ROOT,
            &mut cache,
            &Cancel::never(),
            &prefix_orders,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .expect("prefix activation")
        .logits
        .data()
        .to_vec()
}

/// Rewrite two output rows of the first row-parallel linear so that, for
/// token 0, a wrong summation order changes a whole value. Every product is
/// exact in FP32 and every large one is at least 2^26, so a 1 added after it
/// is absorbed. Model code is unchanged.
///
/// - Row 0, across blocks: block 0 holds `+P`, block 1 holds `-P` then
///   `≈ 1`. S=1 sums to 1; S=2 absorbs the 1 into block 1's partial and sums
///   to 0.
/// - Row 1, within block 0: `+Q`, `-Q`, `≈ 1` in ascending `k`. Ascending
///   sums to 1; descending sums to 0.
fn order_sensitive_fixture(
    routed: bool,
) -> (
    moxie_cli::fixture::Fixture,
    TensorParallelLowering,
    moxie_models::gemma4::TextConfig,
) {
    let (base, config) = fixture(routed);
    let lowering = lower_tensor_parallel(&base.graph, 2).expect("TP2 lowering");
    let node = &base.graph.nodes()[lowering
        .linear_orders
        .keys()
        .next()
        .expect("a row-parallel linear")
        .0 as usize];
    let weight_id = node.inputs[1];
    let original = base
        .weights
        .get(weight_id)
        .expect("row-parallel weight")
        .as_float()
        .expect("BF16 weight");
    let (rows, columns) = (original.rows(), original.cols());
    let x = prefix_activation(
        &base,
        node.id.0 as usize,
        node.inputs[0],
        &[1],
        &[0],
        &BTreeMap::new(),
    );
    let block = columns / 2;
    let nonzero = |range: std::ops::Range<usize>| range.into_iter().filter(|k| x[*k] != 0.0);
    let first: Vec<usize> = nonzero(0..block).take(3).collect();
    let second: Vec<usize> = nonzero(block..columns).take(2).collect();
    assert!(
        first.len() == 3 && second.len() == 2,
        "enough nonzero inputs"
    );
    let mut data = original.data().to_vec();
    // Row `row` gets `+P` at `a`, `-P` at `b` and `≈ 1` at `c`.
    let mut cancel = |row: usize, a: usize, b: usize, c: usize| {
        let scale = 2f32.powi(26 - (x[a] * x[b]).abs().log2().floor() as i32);
        let weights = &mut data[row * columns..(row + 1) * columns];
        weights.fill(0.0);
        weights[a] = x[b] * scale;
        weights[b] = -x[a] * scale;
        weights[c] = bf16_bits_to_f32(f32_to_bf16_bits(1.0 / x[c]));
    };
    cancel(0, first[0], second[0], second[1]);
    cancel(1, first[0], first[1], first[2]);
    let mut weights = base.weights.clone();
    weights.set(
        weight_id,
        Value::Float(HostTensor::bf16(data, vec![rows, columns]).expect("fixture weight")),
    );
    let fixture = moxie_cli::fixture::Fixture { weights, ..base };
    (fixture, lowering, config)
}

fn geometry(
    config: &moxie_models::gemma4::TextConfig,
    page_tokens: usize,
    max_tokens: usize,
    tentative_rows: usize,
) -> KvGeometry {
    geometry_for_layers(
        config,
        0..config.layers,
        page_tokens,
        max_tokens,
        tentative_rows,
    )
}

fn geometry_for_layers(
    config: &moxie_models::gemma4::TextConfig,
    layers: impl IntoIterator<Item = u32>,
    page_tokens: usize,
    max_tokens: usize,
    tentative_rows: usize,
) -> KvGeometry {
    KvGeometry {
        layers: layers
            .into_iter()
            .map(|layer| {
                let geometry = config.layer_geometry(layer);
                LayerKv {
                    kv_heads: geometry.kv_heads as usize,
                    key_dim: geometry.head_dim as usize,
                    value_dim: geometry.head_dim as usize,
                    retention: match geometry.window {
                        Some(window) => Retention::Window {
                            window: window as usize,
                        },
                        None => Retention::All,
                    },
                }
            })
            .collect(),
        precision: Precision::Bf16,
        page_tokens,
        max_tokens,
        tentative_rows,
    }
}

fn local_config(config: &moxie_models::gemma4::TextConfig) -> moxie_models::gemma4::TextConfig {
    let mut local = config.clone();
    local.heads /= 2;
    local.local_kv_heads /= 2;
    // The global fixture has one KV head.  The lowering replicates it on both
    // ranks, so the local state still has one global KV head.
    local
}

fn measured_ledger(context: &RankContext) -> Ledger {
    let device = CapacitySnapshot::measured(&context.measure().expect("measure GPU"), 1 << 20)
        .expect("device capacity");
    let host = CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
        .expect("host capacity");
    Ledger::new([device, host]).expect("ledger")
}

fn paged_descriptor(capability: &DeviceCapability) -> moxie_types::SemanticKernelDescriptor {
    moxie_kernels::paged_attention_catalogue()
        .descriptors()
        .iter()
        .find(|descriptor| {
            descriptor.sm.major == capability.compute_major
                && descriptor.sm.minor == capability.compute_minor
        })
        .cloned()
        .expect("paged descriptor")
}

fn admit_runs<'ctx>(
    ledger: &mut Ledger,
    context: &'ctx RankContext,
    config: &moxie_models::gemma4::TextConfig,
    state: &DeviceKvSequence,
    max_rows: u64,
) -> Vec<PagedAttentionRun<'ctx>> {
    let descriptor = paged_descriptor(context.capability());
    let lineage_capacity = u64::try_from(state.geometry().expect("state geometry").max_tokens)
        .expect("lineage capacity fits u64")
        .checked_add(1)
        .expect("lineage entry count fits u64");
    (0..config.layers)
        .map(|layer| {
            let declared = config.layer_geometry(layer);
            let layout = state.layout(layer as usize).expect("state layout");
            let geometry = PageGeometry {
                kv_heads: declared.kv_heads,
                head_dim: declared.head_dim,
                page_tokens: state.geometry().expect("state geometry").page_tokens as u64,
                pages: layout.pages,
            };
            let admit = if layer == 0 {
                PagedAttentionRun::admit_for_sequence(
                    ledger,
                    context,
                    descriptor.clone(),
                    geometry,
                    config.heads,
                    max_rows,
                    lineage_capacity,
                    Staging::DeviceHandles,
                )
            } else {
                PagedAttentionRun::admit(
                    ledger,
                    context,
                    descriptor.clone(),
                    geometry,
                    config.heads,
                    max_rows,
                    Staging::DeviceHandles,
                )
            };
            admit
                .map_err(|refused| refused.error)
                .expect("paged attention admission")
        })
        .collect()
}

fn stage_weight_value(
    fixture: &moxie_cli::fixture::Fixture,
    weight: &moxie_plan::StageWeight,
) -> Value {
    let whole = fixture
        .weights
        .get(weight.original)
        .expect("stage weight")
        .as_float()
        .expect("BF16 stage weight");
    if let Some(rows) = &weight.rows {
        let row_elements = whole.shape()[1..].iter().product::<usize>();
        let mut shape = whole.shape().to_vec();
        shape[0] = (rows.end - rows.start) as usize;
        return Value::Float(
            HostTensor::bf16(
                whole.data()[rows.start as usize * row_elements..rows.end as usize * row_elements]
                    .to_vec(),
                shape,
            )
            .expect("row-shard weight"),
        );
    }
    if let Some(slice) = weight.slice {
        let mut data = Vec::with_capacity(whole.rows() * slice.width as usize);
        for row in 0..whole.rows() {
            let start = row * whole.cols() + slice.first as usize;
            let end = start + slice.width as usize;
            data.extend_from_slice(&whole.data()[start..end]);
        }
        return Value::Float(
            HostTensor::bf16(data, vec![whole.rows(), slice.width as usize])
                .expect("input-shard weight"),
        );
    }
    Value::Float(whole.clone())
}

fn stage_host_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    stage: &StageGraph,
    sub_stage: Option<&StageGraph>,
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<moxie_executor::OwnedBinding> {
    let rows = tokens.len();
    let graph = sub_stage.map_or(&stage.graph, |sub_stage| &sub_stage.graph);
    let reads = sub_stage.map_or(stage.reads.as_slice(), |sub_stage| {
        sub_stage.reads.as_slice()
    });
    let weights = sub_stage.map_or(stage.weights.as_slice(), |sub_stage| {
        sub_stage.weights.as_slice()
    });
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    for read in reads {
        let original = if sub_stage.is_some() {
            let Some(parent) = stage
                .reads
                .iter()
                .find(|parent| parent.local == read.original)
            else {
                continue;
            };
            parent.original
        } else {
            read.original
        };
        let source = if original == fixture.tokens {
            Value::Index(tokens.to_vec())
        } else if original == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            continue;
        };
        if !seen.insert(read.local) {
            continue;
        }
        bindings.push(moxie_executor::OwnedBinding {
            value: read.local,
            role: graph.spec(read.local).expect("stage input spec").role,
            shape: concrete_shape(graph, read.local, rows),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    for weight in weights {
        let mapped;
        let weight = if sub_stage.is_some() {
            let parent = stage
                .weights
                .iter()
                .find(|parent| parent.local == weight.original)
                .expect("TP weight maps through pipeline stage graph");
            mapped = StageWeight {
                original: parent.original,
                local: weight.local,
                rows: weight.rows.clone().or_else(|| parent.rows.clone()),
                slice: weight.slice.or(parent.slice),
            };
            &mapped
        } else {
            weight
        };
        bindings.push(moxie_executor::OwnedBinding {
            value: weight.local,
            role: graph.spec(weight.local).expect("stage weight spec").role,
            shape: concrete_shape(graph, weight.local, rows),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&stage_weight_value(fixture, weight)),
        });
    }
    bindings
}

fn weight_binding_bytes(bindings: &[moxie_executor::OwnedBinding]) -> u64 {
    bindings
        .iter()
        .filter(|binding| matches!(binding.role, ValueRole::Weight(_)))
        .map(|binding| binding.bytes.len() as u64)
        .sum()
}

fn first_node_of_layer(graph: &Graph, layer: usize) -> usize {
    let suffix = format!(".{layer}");
    graph
        .nodes()
        .iter()
        .position(|node| {
            node.inputs.iter().any(|input| {
                graph.weights().contains(input)
                    && graph
                        .name(*input)
                        .is_some_and(|name| name.ends_with(&suffix))
            })
        })
        .expect("layer has a weight-consuming node")
}

fn pipeline_stage_graphs(
    fixture: &moxie_cli::fixture::Fixture,
    lowering: &PipelineLowering,
) -> Vec<StageGraph> {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("register pipeline oracles");
    lowering
        .stages()
        .iter()
        .enumerate()
        .map(|(stage, nodes)| {
            let output = lowering
                .handoffs()
                .get(stage)
                .copied()
                .unwrap_or_else(|| fixture.graph.output());
            build_stage_graph(
                &fixture.graph,
                None,
                nodes.clone(),
                Some(output),
                moxie_oracles::HOST_REFERENCE,
                &oracles,
            )
            .expect("build pipeline stage graph")
        })
        .collect()
}

fn host_sequence(
    fixture: &moxie_cli::fixture::Fixture,
    ranges: &[Range<u64>],
    linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
) -> Vec<Vec<u32>> {
    let mut state = SequenceState::new([StateKind::KvPages]);
    let mut cache = KvCache::for_branch(
        fixture.graph.attention_layers().len(),
        &state,
        moxie_state::ROOT,
    )
    .expect("host sequence cache");
    ranges
        .iter()
        .map(|rows| {
            let tokens: Vec<_> = rows.clone().map(|row| row % 12).collect();
            let positions: Vec<_> = rows.clone().collect();
            let mut bindings = fixture.weights.clone();
            bindings.set(fixture.tokens, Value::Index(tokens));
            bindings.set(fixture.positions, Value::Index(positions));
            Interpreter::new()
                .run_with_partition_orders(
                    &fixture.graph,
                    &bindings,
                    &mut state,
                    moxie_state::ROOT,
                    &mut cache,
                    &Cancel::never(),
                    linear_orders,
                    combine_orders,
                    &BTreeMap::new(),
                )
                .expect("host sequential reference")
                .logits
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        })
        .collect()
}

fn assert_logits(label: &str, actual: &[u8], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len() * 4, "{label} byte extent");
    for (index, (bytes, expected)) in actual.chunks_exact(4).zip(expected).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().expect("FP32 logit"));
        let expected = f32::from_bits(*expected);
        assert!(
            (actual - expected).abs() <= bf16_ulp(expected),
            "{label} element {index}: {actual} vs {expected}"
        );
    }
}

fn median_millis(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    (samples[3] + samples[4]) / 2.0
}

type CombinedContext<'a> = (
    &'a Graph,
    &'a PipelineLowering,
    &'a moxie_types::KernelCatalogue,
    &'a OracleRegistry,
);

fn execute_combined<'w>(
    workers: &'w mut PipelineWorkers,
    (graph, lowering, catalogue, oracles): CombinedContext<'_>,
    ranges: &[Range<u64>],
    bindings: &mut dyn FnMut(
        moxie_executor::StageBindings<'_>,
    ) -> moxie_types::Result<Vec<moxie_executor::OwnedBinding>>,
    cancel: &AtomicBool,
) -> moxie_types::Result<moxie_executor::PipelineStep<'w>> {
    workers.execute(
        graph,
        lowering,
        moxie_oracles::HOST_REFERENCE,
        oracles,
        catalogue,
        ranges,
        bindings,
        cancel,
    )
}

fn solo_worker_step(
    worker: &mut SoloRankWorker,
    fixture: &moxie_cli::fixture::Fixture,
    catalogue: &moxie_types::KernelCatalogue,
    capability: &DeviceCapability,
    rows: Range<u64>,
) -> Vec<u8> {
    let tokens: Vec<_> = rows.clone().map(|row| row % 12).collect();
    let positions: Vec<_> = rows.clone().collect();
    let output = worker
        .step(
            fixture.graph.clone(),
            catalogue.clone(),
            owned_full_bindings(fixture, &tokens, &positions, capability),
            rows.end - rows.start,
            rows.end,
        )
        .expect("one-GPU dense step");
    worker.prepare_commit().expect("one-GPU prepare");
    worker.apply_commit().expect("one-GPU apply");
    output
}

/// One GPU's stream, ledger, KV authority and attention runs.
struct Rank<'c> {
    ctx: &'c RankContext,
    capability: DeviceCapability,
    stream: Stream<'c>,
    ledger: Ledger,
    state: DeviceKvSequence,
    runs: Vec<PagedAttentionRun<'c>>,
}

impl<'c> Rank<'c> {
    fn new(ctx: &'c RankContext, config: &moxie_models::gemma4::TextConfig) -> Self {
        let mut ledger = measured_ledger(ctx);
        let state = DeviceKvSequence::new(geometry(config, 4, 64, 6)).expect("KV state");
        let runs = admit_runs(&mut ledger, ctx, config, &state, 5);
        Self {
            ctx,
            capability: query_device(ctx.ordinal()).expect("capability"),
            stream: Stream::new(ctx).expect("stream"),
            ledger,
            state,
            runs,
        }
    }

    fn close(mut self) {
        for run in self.runs.drain(..) {
            run.close(&mut self.ledger).expect("close run");
        }
        assert!(
            self.ledger.outstanding().is_empty(),
            "every reservation is returned"
        );
    }
}

/// The one-3090 reference: the ordinary dense path with the split-aware
/// linear honouring the declared S (C1). It never selects the TP partial.
fn reference_step(
    fixture: &moxie_cli::fixture::Fixture,
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
    rank: &mut Rank<'_>,
    tokens: &[u64],
    positions: &[u64],
) -> Vec<u8> {
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let workload = ResourceWorkload {
        phase: if tokens.len() == 1 {
            Phase::Decode
        } else {
            Phase::Prefill
        },
        rows: tokens.len() as u64,
        visible_tokens: positions.last().copied().unwrap_or(0) + 1,
        branch_rows: tokens.len() as u64,
        output: fixture.graph.output(),
        device: rank.capability.uuid,
        paged_state_capacity: None,
    };
    let candidate = lower_selected_ordered(
        &fixture.graph,
        workload,
        &rank.capability,
        &catalogue,
        orders,
        combine_orders,
        &BTreeMap::<NodeId, ExpertOwnership>::new(),
    )
    .expect("split-aware reference selection");
    assert!(
        candidate
            .nodes()
            .iter()
            .all(|node| node.descriptor.operation != SemanticKernelOp::LinearPartial),
        "the reference must not share the TP partial kernel"
    );
    let plan = SelectedReservedPlan::admit(
        candidate,
        &fixture.graph,
        &rank.capability,
        &catalogue,
        &mut rank.ledger,
        rank.ctx,
    )
    .map_err(|_| "reference admission")
    .unwrap();
    let transaction = rank.state.begin().expect("reference transaction");
    let result = plan
        .execute_dense(DenseGraphStep {
            graph: &fixture.graph,
            capability: &rank.capability,
            catalogue: &catalogue,
            ctx: rank.ctx,
            stream: &rank.stream,
            state: &mut rank.state,
            transaction,
            runs: &mut rank.runs,
            bindings: owned_full_bindings(fixture, tokens, positions, &rank.capability),
            host_experts: &[],
        })
        .map_err(|refused| refused.error)
        .expect("reference execution")
        .finish()
        .map_err(|refused| refused.error)
        .expect("reference finish");
    commit_paged_state(
        &mut rank.state,
        transaction,
        0,
        &mut rank.runs,
        &rank.stream,
    )
    .expect("reference commit");
    result
        .plan
        .close(&mut rank.ledger)
        .map_err(|refused| refused.error)
        .expect("reference plan close");
    result.output
}

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    /// Rank 1 refuses its layer-1 attention stage, after layer 0 appended KV
    /// on both ranks.
    Rank,
    /// Cancellation is requested at the layer-1 stage handoff.
    Cancel,
    /// The collective after that stage is refused.
    Collective,
    /// Rank 1 reports a post-prepare status failure through NCCL.
    JoinStatus,
    /// Rank 1's commit refuses while preparing, after rank 0 prepared (C2).
    Commit,
    /// The executed step is dropped uncommitted (F11).
    Drop,
    /// Rank 1's stage launch is refused after its uploads (N2): recoverable.
    Launch,
    /// Rank 1's paged-attention launch is refused (R3-2): recoverable.
    AttentionLaunch,
    /// A row count whose aligned boundary size overflows (R3-1): refused
    /// before either transaction begins.
    Oversized,
    /// Rank 1's in-step page-table upload fails (R5-1): recoverable, with
    /// the run's admitted upload buffer restored.
    TableUpload,
    /// Rank 1 stalls before the first stage rendezvous after Begin.
    Stall,
}

fn tp_worker_step(
    workers: &mut DenseRankWorkers,
    fixture: &moxie_cli::fixture::Fixture,
    lowering: &TensorParallelLowering,
    capabilities: &[DeviceCapability; 2],
    tokens: &[u64],
    positions: &[u64],
    fault: Fault,
) -> Result<Vec<u8>, Error> {
    if fault == Fault::Collective {
        workers.inject_collective_mismatch_once();
    }
    if fault == Fault::Commit {
        workers.refuse_next_commit_prepare(1);
    }
    if fault == Fault::Stall {
        workers.stall_before_next_rendezvous(1);
    }
    if fault == Fault::JoinStatus {
        workers.fail_next_join_after_prepare(1);
    }
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("oracles");
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let cancel = AtomicBool::new(false);
    let mut bindings = |rank: usize, stage: &StageGraph| {
        if rank == 1 && stage.state_layers.values().next() == Some(&1) {
            match fault {
                Fault::Rank => {
                    return Err(Error::InvalidRequest {
                        field: "fault",
                        detail: "injected rank-1 stage failure".into(),
                    });
                }
                Fault::Cancel => cancel.store(true, Ordering::Release),
                Fault::Collective => {}
                Fault::Launch => DRIVER_FAULT.store(LAUNCH_FAULT, Ordering::SeqCst),
                Fault::TableUpload => DRIVER_FAULT.store(TABLE_UPLOAD_FAULT, Ordering::SeqCst),
                Fault::AttentionLaunch => {
                    DRIVER_FAULT.store(ATTENTION_LAUNCH_FAULT, Ordering::SeqCst)
                }
                Fault::None
                | Fault::Commit
                | Fault::Drop
                | Fault::JoinStatus
                | Fault::Oversized
                | Fault::Stall => {}
            }
        }
        Ok(stage_host_bindings(
            fixture,
            stage,
            None,
            tokens,
            positions,
            &capabilities[rank],
        ))
    };
    let step = workers.execute_dense(
        &fixture.graph,
        lowering,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
        &catalogue,
        if fault == Fault::Oversized {
            192_153_584_101_141_162
        } else {
            tokens.len() as u64
        },
        positions.last().copied().unwrap_or(0) + 1,
        &mut bindings,
        &cancel,
    )?;
    if fault == Fault::Drop {
        drop(step);
        return Err(Error::Cancelled { at: "dropped" });
    }
    let sampled = step.logits().to_vec();
    let committed = step.commit()?;
    assert_eq!(committed, sampled, "commit returns the sampled logits");
    Ok(committed)
}

fn tp2_worker_gate(routed: bool) {
    let _guard = one_at_a_time();
    let (fixture, lowering, config) = order_sensitive_fixture(routed);
    // Token 9 routes both selected experts to rank 0 for the routed decode.
    let decode_token = if routed { 9 } else { 6 };
    let tokens = [1, 4, 7, 2, 9];
    let positions = [0, 1, 2, 3, 4];
    let host_s2 = host_logits(
        &fixture,
        &tokens,
        &positions,
        &lowering.linear_orders,
        &lowering.combine_orders,
    );
    assert_ne!(
        host_logits(
            &fixture,
            &tokens,
            &positions,
            &one_block_orders(&lowering.linear_orders),
            &BTreeMap::new(),
        ),
        host_s2,
        "the fixture must make the declared S=2 order load-bearing"
    );
    let ordinals = pair_ordinals();
    let local = local_config(&config);
    let (
        reference_prefill,
        reference_decode,
        reference_after_rank_failure,
        reference_after_cancel,
        reference_after_mismatch,
        reference_after_remaining_faults,
    ) = {
        let context = RankContext::acquire(RankId(61_000), ordinals[0]).expect("reference context");
        let mut reference = Rank::new(&context, &config);
        let prefill = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &tokens,
            &positions,
        );
        let decode = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[decode_token],
            &[5],
        );
        let after_rank_failure = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[7],
            &[6],
        );
        let after_cancel = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[8],
            &[7],
        );
        let after_mismatch = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[9],
            &[8],
        );
        let after_remaining_faults = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[10],
            &[9],
        );
        for (word, expected) in prefill.chunks_exact(4).zip(&host_s2) {
            let (actual, expected) = (
                f32::from_le_bytes(word.try_into().unwrap()),
                f32::from_bits(*expected),
            );
            assert!(
                (actual - expected).abs() <= bf16_ulp(expected),
                "reference {actual} vs host {expected}"
            );
        }
        reference.close();
        (
            prefill,
            decode,
            after_rank_failure,
            after_cancel,
            after_mismatch,
            after_remaining_faults,
        )
    };
    let host_capacity =
        CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
            .expect("host capacity");
    let mut workers = DenseRankWorkers::spawn(DenseRankWorkerConfig {
        ranks: [RankId(61_001), RankId(61_002)],
        ordinals,
        geometry: geometry(&local, 4, 64, 6),
        heads: local.heads,
        max_rows: 5,
        host_capacity,
        deadline: DEADLINE,
    })
    .expect("persistent rank workers");
    let capabilities = [
        query_device(ordinals[0]).expect("rank 0 capability"),
        query_device(ordinals[1]).expect("rank 1 capability"),
    ];
    println!(
        "TP2 device pair UUIDs: {} {}",
        capabilities[0].uuid, capabilities[1].uuid
    );
    if routed {
        let owners = |row: &[u32]| row.iter().map(|expert| expert / 4).collect::<Vec<_>>();
        let routes = host_route_ids(&fixture, &tokens, &positions, &lowering.linear_orders);
        assert!(
            routes
                .iter()
                .any(|row| owners(row).windows(2).all(|pair| pair[0] == pair[1])),
            "prefill must contain a same-owner route: {routes:?}"
        );
        assert!(
            routes
                .iter()
                .any(|row| owners(row).windows(2).any(|pair| pair[0] != pair[1])),
            "prefill must contain a cross-owner route: {routes:?}"
        );
        let mut decode_tokens = tokens.to_vec();
        decode_tokens.push(decode_token);
        let mut decode_positions = positions.to_vec();
        decode_positions.push(5);
        let decode_routes = host_route_ids(
            &fixture,
            &decode_tokens,
            &decode_positions,
            &lowering.linear_orders,
        );
        assert!(
            owners(decode_routes.last().expect("decode route row"))
                .windows(2)
                .all(|pair| pair[0] == pair[1]),
            "decode [{decode_token}]/[5] must route to one owner: {decode_routes:?}"
        );
        assert_expert_owner_resource(&fixture, &lowering, &capabilities);
    }
    let prefill = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &tokens,
        &positions,
        Fault::None,
    )
    .expect("worker TP2 prefill");
    assert_eq!(
        prefill, reference_prefill,
        "worker prefill is bit-identical"
    );
    let decode = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[decode_token],
        &[5],
        Fault::None,
    )
    .expect("worker TP2 decode");
    assert_eq!(decode, reference_decode, "worker decode is bit-identical");

    let rank_error = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[7],
        &[6],
        Fault::Rank,
    )
    .expect_err("an injected rank failure refuses");
    assert!(matches!(rank_error, Error::InvalidRequest { .. }));
    let after_rank_failure = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[7],
        &[6],
        Fault::None,
    )
    .expect("clean step immediately after rank refusal");
    assert_eq!(after_rank_failure, reference_after_rank_failure);
    let before_cancel = workers.stats().expect("worker stats before cancellation");
    let cancelled = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[8],
        &[7],
        Fault::Cancel,
    )
    .expect_err("stage-boundary cancellation refuses the step");
    assert!(
        matches!(
            cancelled,
            Error::Cancelled {
                at: "tensor-parallel stage"
            }
        ),
        "expected stage cancellation, got {cancelled}"
    );
    assert_eq!(
        workers.stats().expect("worker stats after cancellation"),
        before_cancel,
        "cancellation leaves frontiers and reservations unchanged"
    );
    let after_cancel = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[8],
        &[7],
        Fault::None,
    )
    .expect("clean step immediately after cancellation");
    assert_eq!(after_cancel, reference_after_cancel);
    let mismatch = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[9],
        &[8],
        Fault::Collective,
    )
    .expect_err("the unequal rank declaration is refused");
    assert!(
        matches!(
            &mismatch,
            Error::InvalidRequest { field: "collective", detail }
                if detail.contains("rendezvous declarations differ")
        ),
        "the rendezvous must reject the unequal declaration: {mismatch}"
    );
    let after_mismatch = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[9],
        &[8],
        Fault::None,
    )
    .expect("clean step immediately after declaration mismatch");
    assert_eq!(after_mismatch, reference_after_mismatch);
    let before = workers.stats().expect("stats after mismatch recovery");

    for fault in [
        Fault::JoinStatus,
        Fault::Launch,
        Fault::AttentionLaunch,
        Fault::Oversized,
        Fault::TableUpload,
        Fault::Commit,
        Fault::Drop,
    ] {
        tp_worker_step(
            &mut workers,
            &fixture,
            &lowering,
            &capabilities,
            &[10],
            &[9],
            fault,
        )
        .expect_err("an injected rank failure refuses");
        assert_eq!(
            workers.stats().expect("recoverable worker stats"),
            before,
            "a recoverable refusal returns both frontiers and reservations"
        );
    }
    let recovered = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[10],
        &[9],
        Fault::None,
    )
    .expect("clean step after recoverable refusals");
    assert_eq!(
        recovered, reference_after_remaining_faults,
        "recovered step is bit-identical"
    );

    if !routed {
        workers
            .close()
            .expect("close dense workers before routed run");
        return;
    }

    let before_stall = workers.stats().expect("worker stats before stall");
    let published_before_stall = workers.published_frontiers();
    assert_eq!(
        published_before_stall,
        [before_stall[0].0, before_stall[1].0],
        "published frontier mirror agrees with stats before stall"
    );
    let committed_before_stall = workers.committed_frontiers();
    assert_eq!(
        committed_before_stall,
        [before_stall[0].1, before_stall[1].1],
        "committed frontier mirror agrees with stats before stall"
    );
    workers.set_deadline_for_test(Duration::from_millis(500));
    let lost = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[11],
        &[10],
        Fault::Stall,
    )
    .expect_err("a stalled rank returns no logits");
    assert!(matches!(lost, Error::DeviceLost { .. }), "{lost}");
    assert_eq!(
        workers.committed_frontiers(),
        committed_before_stall,
        "stall leaves both committed KV frontiers unchanged"
    );
    assert_eq!(
        workers.published_frontiers(),
        published_before_stall,
        "stall leaves both published KV frontiers unchanged"
    );
    assert!(
        workers.stats().is_err_and(|error| error == lost),
        "the next same-group call reports the sticky loss"
    );
}

fn bf16_ulp(value: f32) -> f32 {
    let exponent = (value.abs().to_bits() >> 23) & 0xff;
    match exponent {
        0 => f32::from_bits(1 << 16),
        1..=7 => f32::from_bits(1 << (15 + exponent)),
        _ => f32::from_bits((exponent - 7) << 23),
    }
}

#[test]
fn nccl_status_poisons_both_ranks() {
    let _guard = one_at_a_time();
    let (fixture, lowering, config) = order_sensitive_fixture(false);
    let ordinals = pair_ordinals();
    let local = local_config(&config);
    let (reference_prefill, reference_after_refusal) = {
        let context = RankContext::acquire(RankId(61_003), ordinals[0]).expect("reference context");
        let mut reference = Rank::new(&context, &config);
        let prefill = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[1, 4],
            &[0, 1],
        );
        let after_refusal = reference_step(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &mut reference,
            &[7],
            &[2],
        );
        reference.close();
        (prefill, after_refusal)
    };
    let host_capacity =
        CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
            .expect("host capacity");
    let mut workers = DenseRankWorkers::spawn(DenseRankWorkerConfig {
        ranks: [RankId(61_004), RankId(61_005)],
        ordinals,
        geometry: geometry(&local, 4, 64, 6),
        heads: local.heads,
        max_rows: 5,
        host_capacity,
        deadline: DEADLINE,
    })
    .expect("persistent rank workers");
    let capabilities = [
        query_device(ordinals[0]).expect("rank 0 capability"),
        query_device(ordinals[1]).expect("rank 1 capability"),
    ];
    let prefill = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[1, 4],
        &[0, 1],
        Fault::None,
    )
    .expect("prefill before injected status failure");
    assert_eq!(prefill, reference_prefill);
    let before = workers
        .stats()
        .expect("stats before injected status failure");
    let refused = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[7],
        &[2],
        Fault::JoinStatus,
    )
    .expect_err("the post-prepare status failure refuses both ranks");
    assert!(
        matches!(
            &refused,
            Error::InvalidRequest { field: "collective", detail }
                if detail == "a rank failed after join preparation"
        ),
        "both ranks report the same typed status error: {refused}"
    );
    assert_eq!(
        workers
            .stats()
            .expect("stats after injected status failure"),
        before,
        "the failed transactions leave published and committed state unchanged"
    );
    let recovered = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &[7],
        &[2],
        Fault::None,
    )
    .expect("same group recovers after the status failure");
    assert_eq!(recovered, reference_after_refusal);
    workers.close().expect("close workers after recovery");
}

#[test]
fn tp2_dense_prefill_decode_is_exact_and_rank_step_is_atomic() {
    // The stall case pins the pair's claims for this process, so run it last.
    combined_tp2_tp1_pipeline_matches_host_and_reports_capacity_latency();
    tp2_worker_gate(false);
    tp2_worker_gate(true);
}

fn combined_tp2_tp1_pipeline_matches_host_and_reports_capacity_latency() {
    let _guard = one_at_a_time();
    let visible = match device_count() {
        Ok(count) => (0..count)
            .filter_map(|ordinal| query_device(ordinal).ok())
            .collect::<Vec<_>>(),
        Err(error) => {
            eprintln!("SKIP combined TP2+TP1 pipeline: {error}");
            return;
        }
    };
    let wanted = RANK_UUIDS.map(|uuid| DeviceUuid::parse(uuid).expect("3090 UUID"));
    if wanted
        .iter()
        .any(|uuid| !visible.iter().any(|device| device.uuid == *uuid))
    {
        eprintln!("SKIP combined TP2+TP1 pipeline: the qualified 3090 pair is not visible");
        return;
    }
    let Some(gpu5060) = visible
        .iter()
        .find(|device| device.name.contains("5060 Ti"))
        .cloned()
    else {
        eprintln!("SKIP combined TP2+TP1 pipeline: an RTX 5060 Ti is not visible");
        return;
    };
    let pair_ordinals = pair_ordinals();
    let pair_capabilities = pair_ordinals
        .map(|ordinal| query_device(ordinal).expect("query 3090 pair capability after UUID check"));
    let devices = [
        pair_capabilities[0].clone(),
        pair_capabilities[1].clone(),
        gpu5060,
    ];
    let (fixture, config) = fixture(false);
    let combined_lowering =
        lower_pipeline(&fixture.graph, &[first_node_of_layer(&fixture.graph, 3)])
            .expect("lower Shape A into TP2 then TP1 stages");
    let combined_stages = pipeline_stage_graphs(&fixture, &combined_lowering);
    assert_eq!(combined_stages.len(), 2);
    assert_eq!(
        combined_stages[0]
            .state_layers
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [0, 1, 2],
        "the pair's KV geometry is stage-local"
    );
    assert_eq!(
        combined_stages[1]
            .state_layers
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [3, 4, 5],
        "the solo stage owns layers 3 through 5"
    );
    lower_tensor_parallel(&combined_stages[0].graph, 2)
        .unwrap_or_else(|error| panic!("STOP: TP2 lowering refused the stage-0 graph: {error}"));
    let whole_lowering = lower_tensor_parallel(&fixture.graph, 2).expect("whole graph TP2 order");
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let host_ranges = [0..5, 5..6, 6..7];
    let host = host_sequence(
        &fixture,
        &host_ranges,
        &whole_lowering.linear_orders,
        &whole_lowering.combine_orders,
    );

    let host_capacity = CapacitySnapshot::measured_host(
        &moxie_host::read().expect("measure host capacity"),
        1 << 20,
    )
    .expect("host capacity snapshot");
    let local = local_config(&config);
    let pair_config = DenseRankWorkerConfig {
        ranks: [RankId(72_001), RankId(72_002)],
        ordinals: pair_ordinals,
        geometry: geometry_for_layers(
            &local,
            combined_stages[0].state_layers.values().copied(),
            4,
            64,
            5,
        ),
        heads: local.heads,
        max_rows: 5,
        host_capacity: host_capacity.clone(),
        deadline: Duration::from_secs(60),
    };
    let solo_config = SoloRankWorkerConfig {
        rank: RankId(72_003),
        ordinal: devices[2].ordinal,
        geometry: geometry_for_layers(
            &config,
            combined_stages[1].state_layers.values().copied(),
            4,
            64,
            5,
        ),
        heads: config.heads,
        max_rows: 5,
        host_capacity: host_capacity.clone(),
        deadline: Duration::from_secs(60),
    };

    let pair = DenseRankWorkers::spawn(pair_config.clone()).expect("spawn combined TP2 pair");
    let solo = SoloRankWorker::spawn(solo_config.clone()).expect("spawn combined TP1 stage");
    let mut workers = PipelineWorkers::new(vec![
        PipelineStageWorker::Pair(pair),
        PipelineStageWorker::Solo(solo),
    ])
    .expect("create TP2+TP1 pipeline");
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("register combined pipeline oracles");
    let combined_context = (&fixture.graph, &combined_lowering, &catalogue, &oracles);
    let combined_binding_bytes = RefCell::new(BTreeMap::<DeviceUuid, u64>::new());
    let record_binding_bytes = AtomicBool::new(false);
    let cancel_rank1_layer1 = AtomicBool::new(false);
    let cancel = AtomicBool::new(false);
    let mut pipeline_bindings = |request: moxie_executor::StageBindings<'_>| {
        let rows = request.rows;
        if cancel_rank1_layer1.load(Ordering::SeqCst)
            && request.rank.is_some_and(|(rank, stage)| {
                rank == 1 && stage.state_layers.values().next() == Some(&1)
            })
        {
            cancel.store(true, Ordering::SeqCst);
        }
        let tokens: Vec<_> = rows.clone().map(|row| row % 12).collect();
        let positions: Vec<_> = rows.clone().collect();
        let (device, bindings) = if let Some((rank, sub_stage)) = request.rank {
            let device = &devices[rank];
            (
                device.uuid,
                stage_host_bindings(
                    &fixture,
                    request.graph,
                    Some(sub_stage),
                    &tokens,
                    &positions,
                    device,
                ),
            )
        } else {
            let device = &devices[request.stage + 1];
            (
                device.uuid,
                stage_host_bindings(&fixture, request.graph, None, &tokens, &positions, device),
            )
        };
        if record_binding_bytes.load(Ordering::SeqCst) {
            *combined_binding_bytes
                .borrow_mut()
                .entry(device)
                .or_default() += weight_binding_bytes(&bindings);
        }
        Ok(bindings)
    };
    let spawned = workers
        .stats()
        .expect("stats after combined pipeline spawn");
    assert_eq!(spawned.len(), 1, "stats report the solo stage only");
    let frontiers = workers.pair_frontiers().expect("pair frontiers");
    assert_eq!(frontiers, ([0, 0], [0, 0]));

    let mut not_called = false;
    let refusal = workers
        .execute(
            &fixture.graph,
            &combined_lowering,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            &catalogue,
            &[0..2, 2..5],
            &mut |_| {
                not_called = true;
                Ok(Vec::new())
            },
            &AtomicBool::new(false),
        )
        .expect_err("a pair refuses multiple microbatches");
    assert!(matches!(
        refusal,
        Error::InvalidRequest {
            field: "microbatches",
            ..
        }
    ));
    assert!(
        !not_called,
        "microbatch refusal precedes all worker commands"
    );
    assert_eq!(
        workers.stats().expect("stats after microbatch refusal"),
        spawned
    );
    assert_eq!(workers.pair_frontiers(), Some(frontiers));

    record_binding_bytes.store(true, Ordering::SeqCst);
    let first_clean = execute_combined(
        &mut workers,
        combined_context,
        &host_ranges[..1],
        &mut pipeline_bindings,
        &cancel,
    )
    .expect("first clean combined prefill");
    record_binding_bytes.store(false, Ordering::SeqCst);
    let first_clean_logits = first_clean.logits().to_vec();
    assert_logits("combined initial prefill", &first_clean_logits, &host[0]);
    assert_eq!(first_clean.handoff_bytes(), &[5 * config.hidden * 2]);
    drop(first_clean);
    assert_eq!(workers.stats().expect("stats after clean abort"), spawned);
    assert_eq!(workers.pair_frontiers(), Some(frontiers));

    #[derive(Clone, Copy)]
    enum Fault {
        SoloStep,
        Cancel,
        PairCommit,
        SoloPrepare,
    }
    for fault in [
        Fault::SoloStep,
        Fault::Cancel,
        Fault::PairCommit,
        Fault::SoloPrepare,
    ] {
        cancel.store(false, Ordering::SeqCst);
        let error = match fault {
            Fault::SoloStep => {
                workers
                    .fail_next_step(1)
                    .expect("inject stage-one step refusal");
                execute_combined(
                    &mut workers,
                    combined_context,
                    &host_ranges[..1],
                    &mut pipeline_bindings,
                    &cancel,
                )
                .expect_err("solo step refuses after pair execution")
            }
            Fault::Cancel => {
                cancel_rank1_layer1.store(true, Ordering::SeqCst);
                let result = execute_combined(
                    &mut workers,
                    combined_context,
                    &host_ranges[..1],
                    &mut pipeline_bindings,
                    &cancel,
                )
                .expect_err("rank-1 layer-1 cancellation refuses the pair step");
                cancel_rank1_layer1.store(false, Ordering::SeqCst);
                result
            }
            Fault::PairCommit => {
                workers
                    .refuse_next_pair_commit(1)
                    .expect("inject rank-one pair commit refusal");
                let step = execute_combined(
                    &mut workers,
                    combined_context,
                    &host_ranges[..1],
                    &mut pipeline_bindings,
                    &cancel,
                )
                .expect("pair and solo execute before pair commit refusal");
                assert_eq!(step.logits(), first_clean_logits);
                step.commit()
                    .expect_err("pair commit refusal aborts the pipeline")
            }
            Fault::SoloPrepare => {
                workers
                    .refuse_next_prepare(1)
                    .expect("inject solo prepare refusal");
                let step = execute_combined(
                    &mut workers,
                    combined_context,
                    &host_ranges[..1],
                    &mut pipeline_bindings,
                    &cancel,
                )
                .expect("pair and solo execute before solo prepare refusal");
                assert_eq!(step.logits(), first_clean_logits);
                step.commit()
                    .expect_err("solo prepare refusal drops the pair step")
            }
        };
        match fault {
            Fault::Cancel => assert!(matches!(error, Error::Cancelled { .. }), "{error}"),
            _ => assert!(
                matches!(error, Error::InvalidRequest { field: "fault", .. }),
                "{error}"
            ),
        }
        cancel.store(false, Ordering::SeqCst);
        assert_eq!(
            workers.stats().expect("stats after pipeline fault"),
            spawned
        );
        assert_eq!(workers.pair_frontiers(), Some(frontiers));
        let retry = execute_combined(
            &mut workers,
            combined_context,
            &host_ranges[..1],
            &mut pipeline_bindings,
            &cancel,
        )
        .expect("clean retry after combined pipeline fault");
        assert_eq!(retry.logits(), first_clean_logits);
        drop(retry);
        assert_eq!(workers.stats().expect("stats after retry abort"), spawned);
        assert_eq!(workers.pair_frontiers(), Some(frontiers));
    }

    let prefill = execute_combined(
        &mut workers,
        combined_context,
        &host_ranges[..1],
        &mut pipeline_bindings,
        &cancel,
    )
    .expect("combined prefill");
    let prefill_logits = prefill.commit().expect("commit combined prefill");
    assert_logits("combined prefill", &prefill_logits, &host[0]);
    assert_eq!(workers.pair_frontiers().expect("pair frontiers").0, [5, 5]);
    assert_eq!(
        workers.stats().expect("stats after prefill")[0].2,
        spawned[0].2
    );

    for (rows, expected) in host_ranges[1..].iter().zip(&host[1..]) {
        let step = execute_combined(
            &mut workers,
            combined_context,
            std::slice::from_ref(rows),
            &mut pipeline_bindings,
            &cancel,
        )
        .expect("combined decode");
        let handoff = config.hidden * 2;
        assert_eq!(step.handoff_bytes(), &[handoff]);
        let logits = step.commit().expect("commit combined decode");
        assert_logits("combined decode", &logits, expected);
        assert_eq!(
            workers.stats().expect("stats after decode")[0].2,
            spawned[0].2
        );
    }
    assert_eq!(workers.pair_frontiers().expect("pair frontiers").0, [7, 7]);

    let mut combined_latency = Vec::with_capacity(8);
    for row in 7..15 {
        let start = Instant::now();
        let rows = row..row + 1;
        let step = execute_combined(
            &mut workers,
            combined_context,
            std::slice::from_ref(&rows),
            &mut pipeline_bindings,
            &cancel,
        )
        .expect("timed combined decode");
        step.commit().expect("commit timed combined decode");
        combined_latency.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(
            workers.stats().expect("stats after timed decode")[0].2,
            spawned[0].2
        );
    }
    let tokens: Vec<_> = (0..5).collect();
    let positions: Vec<_> = (0..5).collect();
    let mut single = BTreeMap::new();
    single.insert(
        pair_capabilities[0].uuid,
        weight_binding_bytes(&owned_full_bindings(
            &fixture,
            &tokens,
            &positions,
            &pair_capabilities[0],
        )),
    );
    let pp_lowering = lower_pipeline(
        &fixture.graph,
        &[
            first_node_of_layer(&fixture.graph, 1),
            first_node_of_layer(&fixture.graph, 4),
        ],
    )
    .expect("lower three-stage PP layout");
    let pp_stages = pipeline_stage_graphs(&fixture, &pp_lowering);
    let mut pci_pair = pair_capabilities.clone();
    pci_pair.sort_by(|left, right| left.pci_bus_id.cmp(&right.pci_bus_id));
    let mut pp = BTreeMap::new();
    for (device, stage) in [
        (&pci_pair[0], &pp_stages[0]),
        (&pci_pair[1], &pp_stages[1]),
        (&devices[2], &pp_stages[2]),
    ] {
        pp.insert(
            device.uuid,
            weight_binding_bytes(&stage_host_bindings(
                &fixture, stage, None, &tokens, &positions, device,
            )),
        );
    }
    workers
        .close()
        .expect("close combined pipeline before one-GPU run");

    let mut solo = SoloRankWorker::spawn(SoloRankWorkerConfig {
        rank: RankId(72_010),
        ordinal: pair_ordinals[0],
        geometry: geometry(&config, 4, 64, 5),
        heads: config.heads,
        max_rows: 5,
        host_capacity: host_capacity.clone(),
        deadline: Duration::from_secs(60),
    })
    .expect("spawn one-GPU whole-graph reference worker");
    solo_worker_step(&mut solo, &fixture, &catalogue, &pair_capabilities[0], 0..5);
    solo_worker_step(&mut solo, &fixture, &catalogue, &pair_capabilities[0], 5..6);
    solo_worker_step(&mut solo, &fixture, &catalogue, &pair_capabilities[0], 6..7);
    let mut solo_latency = Vec::with_capacity(8);
    for row in 7..15 {
        let start = Instant::now();
        solo_worker_step(
            &mut solo,
            &fixture,
            &catalogue,
            &pair_capabilities[0],
            row..row + 1,
        );
        solo_latency.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    solo.close().expect("close one-GPU reference");

    println!(
        "Shape A fixture: weight-role binding bytes per GPU\nGPU | one 3090 whole graph | combined TP2+TP1 | three-stage PP"
    );
    for device in &devices {
        println!(
            "{} {} | {} | {} | {}",
            device.name,
            device.uuid,
            single.get(&device.uuid).copied().unwrap_or(0),
            combined_binding_bytes
                .borrow()
                .get(&device.uuid)
                .copied()
                .unwrap_or(0),
            pp.get(&device.uuid).copied().unwrap_or(0),
        );
    }
    println!(
        "median decode ms (8 committed steps): one 3090 {:.3}, combined {:.3}",
        median_millis(solo_latency),
        median_millis(combined_latency),
    );

    let misplaced_pair =
        DenseRankWorkers::spawn(pair_config).expect("spawn placement refusal pair");
    let misplaced_solo = SoloRankWorker::spawn(solo_config).expect("spawn placement refusal solo");
    assert!(matches!(
        PipelineWorkers::new(vec![
            PipelineStageWorker::Solo(misplaced_solo),
            PipelineStageWorker::Pair(misplaced_pair),
        ]),
        Err(Error::InvalidRequest {
            field: "pipeline",
            ..
        })
    ));
}
