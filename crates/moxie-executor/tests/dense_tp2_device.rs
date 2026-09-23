//! Task 0060's TP2 device gate.
//!
//! The fixture is deliberately small and synthetic.  It runs the full dense
//! graph on one 3090 through the split-aware ordinary linear kernel, then runs
//! the lowered stage graph on both 3090s and joins the rank-local boundaries.
#![cfg(feature = "paged-attention-binding")]

use core::ffi::{c_int, c_void};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::paged_attention::device::{
    PagedKvRows, append_paged_layer, commit_paged_state,
};
use moxie_executor::{
    DenseGraphStep, DenseRank, DenseTensorParallelStep, PageGeometry, PagedAttentionRun, RankGroup,
    SelectedReservedPlan, Staging,
};
use moxie_format::bf16::{bf16_bits_to_f32, f32_to_bf16_bits};
use moxie_graph::{Bindings, Graph, LinearReductionOrder, NodeId, OracleRegistry, ValueId};
use moxie_interp::{Cancel, Interpreter, KvCache};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{
    Phase, ResourceWorkload, StageGraph, TensorParallelLowering, build_stage_graph,
    lower_selected_ordered, lower_tensor_parallel,
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
/// The second peer copy after arming fails: a collective fails after its
/// first copy is enqueued (F2).
const PEER_COPY_FAULT: u8 = 1;
/// The first peer copy after arming arms `EVENT_QUERY_FAULT` (F5).
const DRAIN_FAULT: u8 = 2;
/// The next event query fails: rank 0's drain cannot be observed.
const EVENT_QUERY_FAULT: u8 = 3;
/// The next kernel launch is refused as an invalid value (N2).
const LAUNCH_FAULT: u8 = 4;
/// The next device free fails, so a plan cannot close (N1).
const FREE_FAULT: u8 = 5;
/// The next paged-attention kernel launch is refused, quarantining its run
/// with the stage's query and output ranges (R3-2).
const ATTENTION_LAUNCH_FAULT: u8 = 6;
/// The next host-to-device copy fails.
const UPLOAD_FAULT: u8 = 7;
/// The next device-to-host copy (an attention stage reading its keys back
/// before the append) arms `UPLOAD_FAULT`. The append's first upload is then
/// the page-table publication, whose failure holds the run's admitted
/// upload buffer (R5-1).
const TABLE_UPLOAD_FAULT: u8 = 8;

static DRIVER_FAULT: AtomicU8 = AtomicU8::new(NO_FAULT);
static PEER_COPIES: AtomicU32 = AtomicU32::new(0);

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

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyPeerAsync(
    dst: u64,
    dst_ctx: *mut c_void,
    src: u64,
    src_ctx: *mut c_void,
    bytes: usize,
    stream: *mut c_void,
) -> c_int {
    match DRIVER_FAULT.load(Ordering::SeqCst) {
        PEER_COPY_FAULT if PEER_COPIES.fetch_add(1, Ordering::SeqCst) == 1 => {
            DRIVER_FAULT.store(NO_FAULT, Ordering::SeqCst);
            return 1;
        }
        DRAIN_FAULT => DRIVER_FAULT.store(EVENT_QUERY_FAULT, Ordering::SeqCst),
        _ => {}
    }
    type Copy =
        unsafe extern "C" fn(u64, *mut c_void, u64, *mut c_void, usize, *mut c_void) -> c_int;
    // SAFETY: the resolved symbol has this CUDA ABI; arguments pass unchanged.
    unsafe {
        let copy: Copy = std::mem::transmute(real(c"cuMemcpyPeerAsync"));
        copy(dst, dst_ctx, src, src_ctx, bytes, stream)
    }
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

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemFree_v2(pointer: u64) -> c_int {
    if fires(FREE_FAULT) {
        return 1;
    }
    // SAFETY: as above.
    unsafe {
        let free: unsafe extern "C" fn(u64) -> c_int = std::mem::transmute(real(c"cuMemFree_v2"));
        free(pointer)
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventQuery(event: *mut c_void) -> c_int {
    if fires(EVENT_QUERY_FAULT) {
        return 700;
    }
    // SAFETY: as above.
    unsafe {
        let query: unsafe extern "C" fn(*mut c_void) -> c_int =
            std::mem::transmute(real(c"cuEventQuery"));
        query(event)
    }
}

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture() -> (
    moxie_cli::fixture::Fixture,
    moxie_models::gemma4::TextConfig,
) {
    let mut config = moxie_cli::gemma::Shape::A.config();
    config.heads = 8;
    config.local_kv_heads = 4;
    config.global_kv_heads = 1;
    config.vocab = 12;
    (
        moxie_cli::gemma::build_with_config(config.clone()).expect("TP2 fixture"),
        config,
    )
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
        .run_with_linear_orders(
            &fixture.graph,
            &bindings,
            &mut state,
            moxie_state::ROOT,
            &mut cache,
            &Cancel::never(),
            orders,
        )
        .expect("host declared split")
        .logits;
    output.data().iter().map(|value| value.to_bits()).collect()
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
        .run(
            &stage.graph,
            &bindings,
            &mut state,
            moxie_state::ROOT,
            &mut cache,
            &Cancel::never(),
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
fn order_sensitive_fixture() -> (
    moxie_cli::fixture::Fixture,
    TensorParallelLowering,
    moxie_models::gemma4::TextConfig,
) {
    let (base, config) = fixture();
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
    let x = prefix_activation(&base, node.id.0 as usize, node.inputs[0], &[1], &[0]);
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
    KvGeometry {
        layers: (0..config.layers)
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
    (0..config.layers)
        .map(|layer| {
            let declared = config.layer_geometry(layer);
            let layout = state.layout(layer as usize).expect("state layout");
            PagedAttentionRun::admit(
                ledger,
                context,
                descriptor.clone(),
                PageGeometry {
                    kv_heads: declared.kv_heads,
                    head_dim: declared.head_dim,
                    page_tokens: state.geometry().expect("state geometry").page_tokens as u64,
                    pages: layout.pages,
                },
                config.heads,
                max_rows,
                Staging::DeviceHandles,
            )
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
        let columns = whole.cols();
        return Value::Float(
            HostTensor::bf16(
                whole.data()[rows.start as usize * columns..rows.end as usize * columns].to_vec(),
                vec![(rows.end - rows.start) as usize, columns],
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
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<moxie_executor::OwnedBinding> {
    let rows = tokens.len();
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    for read in &stage.reads {
        let source = if read.original == fixture.tokens {
            Value::Index(tokens.to_vec())
        } else if read.original == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            continue;
        };
        if !seen.insert(read.local) {
            continue;
        }
        bindings.push(moxie_executor::OwnedBinding {
            value: read.local,
            role: stage.graph.spec(read.local).expect("stage input spec").role,
            shape: concrete_shape(&stage.graph, read.local, rows),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    for weight in &stage.weights {
        bindings.push(moxie_executor::OwnedBinding {
            value: weight.local,
            role: stage
                .graph
                .spec(weight.local)
                .expect("stage weight spec")
                .role,
            shape: concrete_shape(&stage.graph, weight.local, rows),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&stage_weight_value(fixture, weight)),
        });
    }
    bindings
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

    fn split(&mut self) -> (DenseRank<'_, 'c>, &mut [PagedAttentionRun<'c>]) {
        (
            DenseRank {
                ctx: self.ctx,
                capability: &self.capability,
                stream: &self.stream,
                ledger: &mut self.ledger,
                state: &mut self.state,
            },
            &mut self.runs,
        )
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
    /// The collective after that stage is refused.
    Collective,
    /// A collective fails after its first peer copy is enqueued (F2).
    PeerCopy,
    /// Rank 1's commit refuses while preparing, after rank 0 prepared (C2).
    Commit,
    /// Rank 0's drain cannot be observed (F5): the group is lost.
    Drain,
    /// The executed step is dropped uncommitted (F11).
    Drop,
    /// Rank 1's stage launch is refused after its uploads (N2): recoverable.
    Launch,
    /// A plan cannot close after the drain (N1): the group is lost.
    Close,
    /// Rank 1's paged-attention launch is refused (R3-2): recoverable.
    AttentionLaunch,
    /// A row count whose aligned boundary size overflows (R3-1): refused
    /// before either transaction begins.
    Oversized,
    /// Rank 1's in-step page-table upload fails (R5-1): recoverable, with
    /// the run's admitted upload buffer restored.
    TableUpload,
}

fn tp_step<'c>(
    group: &mut RankGroup<'c>,
    ranks: &mut [Rank<'c>; 2],
    fixture: &moxie_cli::fixture::Fixture,
    lowering: &TensorParallelLowering,
    tokens: &[u64],
    positions: &[u64],
    fault: Fault,
) -> Result<Vec<u8>, Error> {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("oracles");
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let cancel = AtomicBool::new(false);
    let capabilities = [ranks[0].capability.clone(), ranks[1].capability.clone()];
    let mut bindings = |rank: usize, stage: &StageGraph| {
        if rank == 1 && stage.state_layers.values().next() == Some(&1) {
            match fault {
                Fault::Rank => {
                    return Err(Error::InvalidRequest {
                        field: "fault",
                        detail: "injected rank-1 stage failure".into(),
                    });
                }
                Fault::Collective => cancel.store(true, Ordering::Release),
                Fault::PeerCopy => {
                    PEER_COPIES.store(0, Ordering::SeqCst);
                    DRIVER_FAULT.store(PEER_COPY_FAULT, Ordering::SeqCst);
                }
                Fault::Drain => DRIVER_FAULT.store(DRAIN_FAULT, Ordering::SeqCst),
                Fault::Launch => DRIVER_FAULT.store(LAUNCH_FAULT, Ordering::SeqCst),
                Fault::Close => DRIVER_FAULT.store(FREE_FAULT, Ordering::SeqCst),
                Fault::TableUpload => DRIVER_FAULT.store(TABLE_UPLOAD_FAULT, Ordering::SeqCst),
                Fault::AttentionLaunch => {
                    DRIVER_FAULT.store(ATTENTION_LAUNCH_FAULT, Ordering::SeqCst)
                }
                Fault::None | Fault::Commit | Fault::Drop | Fault::Oversized => {}
            }
        }
        Ok(stage_host_bindings(
            fixture,
            stage,
            tokens,
            positions,
            &capabilities[rank],
        ))
    };
    let [r0, r1] = ranks;
    let ((d0, runs0), (d1, runs1)) = (r0.split(), r1.split());
    let step = group.execute_dense(
        [d0, d1],
        [&mut *runs0, &mut *runs1],
        DenseTensorParallelStep {
            graph: &fixture.graph,
            lowering,
            oracle: moxie_oracles::HOST_REFERENCE,
            oracles: &oracles,
            catalogue: &catalogue,
            rows: if fault == Fault::Oversized {
                192_153_584_101_141_162
            } else {
                tokens.len() as u64
            },
            visible_tokens: positions.last().copied().unwrap_or(0) + 1,
            bindings: &mut bindings,
            cancel: &cancel,
        },
    )?;
    if fault == Fault::Drop {
        drop(step);
        return Err(Error::Cancelled { at: "dropped" });
    }
    // One run short of the layer count: rank 1's prepare refuses, after
    // rank 0 has prepared.
    let runs1 = if fault == Fault::Commit {
        &mut runs1[..1]
    } else {
        runs1
    };
    // The caller samples from the logits before deciding to commit.
    let sampled = step.logits().to_vec();
    let committed = step.commit([runs0, runs1])?;
    assert_eq!(
        committed, sampled,
        "commit returns the logits it was sampled from"
    );
    Ok(committed)
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
fn tp2_dense_prefill_decode_is_exact_and_rank_step_is_atomic() {
    let _guard = one_at_a_time();
    let (fixture, lowering, config) = order_sensitive_fixture();
    let (tokens, positions) = ([1, 4, 7, 2, 9], [0, 1, 2, 3, 4]);
    let host_s2 = host_logits(&fixture, &tokens, &positions, &lowering.linear_orders);
    assert_ne!(
        host_logits(
            &fixture,
            &tokens,
            &positions,
            &one_block_orders(&lowering.linear_orders)
        ),
        host_s2,
        "the fixture must make the declared S=2 order load-bearing"
    );

    let ordinals = pair_ordinals();
    let contexts = [
        RankContext::acquire(RankId(60_000), ordinals[0]).expect("rank 0 context"),
        RankContext::acquire(RankId(60_001), ordinals[1]).expect("rank 1 context"),
    ];
    let mut group = RankGroup::form([&contexts[0], &contexts[1]], DEADLINE).expect("rank group");
    eprintln!(
        "task0060 TP2 pair: {} + {}",
        group.uuids()[0],
        group.uuids()[1]
    );
    let local = local_config(&config);
    let mut ranks = [
        Rank::new(&contexts[0], &local),
        Rank::new(&contexts[1], &local),
    ];
    let mut reference = Rank::new(&contexts[0], &config);

    // C1: the reference agrees with task 0058's host interpreter at S=2
    // within task 0012's one-BF16-ULP gate.
    let reference_prefill = reference_step(
        &fixture,
        &lowering.linear_orders,
        &mut reference,
        &tokens,
        &positions,
    );
    for (word, expected) in reference_prefill.chunks_exact(4).zip(&host_s2) {
        let (actual, expected) = (
            f32::from_le_bytes(word.try_into().unwrap()),
            f32::from_bits(*expected),
        );
        assert!(
            (actual - expected).abs() <= bf16_ulp(expected),
            "reference {actual} vs host {expected}"
        );
    }

    let prefill = tp_step(
        &mut group,
        &mut ranks,
        &fixture,
        &lowering,
        &tokens,
        &positions,
        Fault::None,
    )
    .expect("TP2 prefill");
    assert_eq!(prefill, reference_prefill, "TP2 prefill is bit-identical");
    let (decode, decode_positions) = ([6], [5]);
    let reference_decode = reference_step(
        &fixture,
        &lowering.linear_orders,
        &mut reference,
        &decode,
        &decode_positions,
    );
    let frontier =
        |ranks: &[Rank<'_>; 2]| ranks.each_ref().map(|r| r.state.published_rows().unwrap());
    let reservations =
        |ranks: &[Rank<'_>; 2]| ranks.each_ref().map(|r| r.ledger.outstanding().len());
    let before = frontier(&ranks);
    let balanced = reservations(&ranks);
    for fault in [
        Fault::Rank,
        Fault::Collective,
        Fault::PeerCopy,
        Fault::Launch,
        Fault::AttentionLaunch,
        Fault::Oversized,
        Fault::TableUpload,
        Fault::Commit,
        Fault::Drop,
    ] {
        tp_step(
            &mut group,
            &mut ranks,
            &fixture,
            &lowering,
            &decode,
            &decode_positions,
            fault,
        )
        .expect_err("an injected fault refuses");
        assert_eq!(
            frontier(&ranks),
            before,
            "a refused step publishes no KV on either rank"
        );
        assert_eq!(
            reservations(&ranks),
            balanced,
            "a recoverable refusal returns every reservation"
        );
    }
    let decoded = tp_step(
        &mut group,
        &mut ranks,
        &fixture,
        &lowering,
        &decode,
        &decode_positions,
        Fault::None,
    )
    .expect("TP2 decode");
    assert_eq!(
        decoded, reference_decode,
        "the next clean step is bit-identical"
    );

    // A drain that cannot be observed is terminal: the step is withheld,
    // the group records why, and every later step refuses. The pair's
    // ledgers keep the withheld reservations, so the pair is not closed.
    let lost = tp_step(
        &mut group,
        &mut ranks,
        &fixture,
        &lowering,
        &decode,
        &decode_positions,
        Fault::Drain,
    )
    .expect_err("an unobserved drain refuses");
    assert!(matches!(lost, Error::DeviceLost { .. }), "{lost}");
    assert!(
        tp_step(
            &mut group,
            &mut ranks,
            &fixture,
            &lowering,
            &decode,
            &decode_positions,
            Fault::None,
        )
        .is_err_and(|refused| refused == lost),
        "a lost group refuses with its recorded reason"
    );

    // A step accepts only idle runs. Rank 1's first run is quarantined by a
    // page-table upload that failed on another stream, holding the upload
    // bytes that copy may still read. The step refuses before either
    // transaction begins and leaves the quarantine and held bytes alone.
    let mut group = RankGroup::form([&contexts[0], &contexts[1]], DEADLINE).expect("rank group");
    let mut dirty = [
        Rank::new(&contexts[0], &local),
        Rank::new(&contexts[1], &local),
    ];
    let side = Stream::new(&contexts[1]).expect("side stream");
    {
        let rank = &mut dirty[1];
        let layer = local.layer_geometry(0);
        let bytes = (layer.kv_heads * layer.head_dim * 2) as usize;
        let txn = rank.state.begin().expect("side transaction");
        DRIVER_FAULT.store(UPLOAD_FAULT, Ordering::SeqCst);
        let refused = append_paged_layer(
            &mut rank.state,
            txn,
            0,
            1,
            &mut rank.runs[0],
            &side,
            PagedKvRows {
                keys: vec![0; bytes],
                values: vec![0; bytes],
            },
        )
        .expect_err("the injected page-table upload refuses");
        assert!(
            refused.error.to_string().contains("cuMemcpyHtoD"),
            "{}",
            refused.error
        );
        rank.state.abort(txn).expect("abort the side transaction");
    }
    let (published, idle) = (frontier(&dirty), reservations(&dirty));
    let refused = tp_step(
        &mut group,
        &mut dirty,
        &fixture,
        &lowering,
        &tokens,
        &positions,
        Fault::None,
    )
    .expect_err("a quarantined run is refused");
    assert!(
        matches!(refused, Error::InvalidRequest { field: "runs", .. }),
        "{refused}"
    );
    assert_eq!((frontier(&dirty), reservations(&dirty)), (published, idle));
    for rank in &mut dirty {
        let txn = rank.state.begin().expect("no transaction was left open");
        rank.state.abort(txn).expect("abort");
    }
    let [_, dirty_one] = &mut dirty;
    let run = dirty_one.runs.remove(0);
    assert!(
        run.close(&mut dirty_one.ledger).is_err(),
        "the run is still quarantined"
    );

    // A plan that cannot close after the drain is terminal too, on a fresh
    // pair: the group is lost and the plan's reservation stays withheld.
    let mut ranks = [
        Rank::new(&contexts[0], &local),
        Rank::new(&contexts[1], &local),
    ];
    let idle = reservations(&ranks);
    let lost = tp_step(
        &mut group,
        &mut ranks,
        &fixture,
        &lowering,
        &tokens,
        &positions,
        Fault::Close,
    )
    .expect_err("a plan that cannot close refuses");
    assert!(matches!(lost, Error::DeviceLost { .. }), "{lost}");
    assert_eq!(group.lost(), Some(&lost));
    assert!(
        reservations(&ranks)[0] > idle[0],
        "the unclosable plan is withheld, not dropped"
    );
    reference.close();
}
