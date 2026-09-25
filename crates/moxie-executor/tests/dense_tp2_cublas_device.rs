#![cfg(feature = "cublas")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::paged_attention::device::commit_paged_state;
use moxie_executor::{
    DenseGraphStep, DenseRankWorkerConfig, DenseRankWorkers, OwnedBinding, PageGeometry,
    PagedAttentionRun, SelectedReservedPlan, Staging,
};
use moxie_format::bf16::{bf16_bits_to_f32, f32_to_bf16_bits};
use moxie_graph::{
    Bindings, CombineReductionOrder, ExpertOwnership, Graph, LinearReductionOrder, NodeId,
    OracleRegistry, ValueId,
};
use moxie_interp::{Cancel, Interpreter, KvCache};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{
    Phase, ResourceWorkload, StageGraph, TensorParallelLowering, build_stage_graph,
    lower_selected_ordered, lower_tensor_parallel,
};
use moxie_state::{DeviceKvSequence, KvGeometry, LayerKv, Retention, SequenceState, StateKind};
use moxie_types::{
    DeviceCapability, DeviceUuid, Dim, KernelCatalogue, Precision, RankId, SemanticKernelOp,
    SmVersion, TensorLayout,
};

const DEADLINE: Duration = Duration::from_secs(20);
const RANK_UUIDS: [&str; 2] = [
    "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9",
    "GPU-81fe4578-59b2-37c4-421e-287cdac78704",
];

fn gemma_shape_a_fixture() -> (
    moxie_cli::fixture::Fixture,
    moxie_models::gemma4::TextConfig,
) {
    let mut config = moxie_cli::gemma::Shape::A.config();
    config.heads = 8;
    config.local_kv_heads = 4;
    config.global_kv_heads = 1;
    config.vocab = 12;
    let fixture = moxie_cli::gemma::build_with_config(config.clone()).expect("TP2 fixture");
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
fn order_sensitive_fixture() -> (
    moxie_cli::fixture::Fixture,
    TensorParallelLowering,
    moxie_models::gemma4::TextConfig,
) {
    let (base, config) = gemma_shape_a_fixture();
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

fn stage_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    stage: &StageGraph,
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<OwnedBinding> {
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
        bindings.push(OwnedBinding {
            value: read.local,
            role: stage.graph.spec(read.local).expect("stage input spec").role,
            shape: concrete_shape(&stage.graph, read.local, rows),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    for weight in &stage.weights {
        bindings.push(OwnedBinding {
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

fn tp_worker_step(
    workers: &mut DenseRankWorkers,
    fixture: &moxie_cli::fixture::Fixture,
    lowering: &TensorParallelLowering,
    capabilities: &[DeviceCapability; 2],
    catalogue: &KernelCatalogue,
    tokens: &[u64],
    positions: &[u64],
) -> Vec<u8> {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("oracles");
    let cancel = AtomicBool::new(false);
    let mut bindings = |rank: usize, stage: &StageGraph| {
        Ok(stage_bindings(
            fixture,
            stage,
            tokens,
            positions,
            &capabilities[rank],
        ))
    };
    let step = workers
        .execute_dense(
            &fixture.graph,
            lowering,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            catalogue,
            tokens.len() as u64,
            positions.last().copied().unwrap_or(0) + 1,
            &mut bindings,
            &cancel,
        )
        .expect("eager unordered TP2 step");
    let sampled = step.logits().to_vec();
    let committed = step.commit().expect("TP2 commit");
    assert_eq!(committed, sampled, "commit returns the sampled logits");
    committed
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

fn reference_step_with_catalogue(
    fixture: &moxie_cli::fixture::Fixture,
    orders: &BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
    catalogue: &KernelCatalogue,
    rank: &mut Rank<'_>,
    tokens: &[u64],
    positions: &[u64],
) -> Vec<u8> {
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
        catalogue,
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
        catalogue,
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
            catalogue,
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

#[test]
#[cfg(feature = "cublas")]
fn tp2_unordered_matches_the_packed_split_reference() {
    let (fixture, lowering, config) = order_sensitive_fixture();
    let ordinals = pair_ordinals();
    let capabilities = [
        query_device(ordinals[0]).expect("first 3090 capability"),
        query_device(ordinals[1]).expect("second 3090 capability"),
    ];
    let catalogue = moxie_kernels::dense_graph_catalogue_unordered(SmVersion::SM86);
    let tokens = [1, 4, 7, 2, 9];
    let positions = [0, 1, 2, 3, 4];
    let expected = {
        let context = RankContext::acquire(RankId(0), ordinals[0])
            .expect("acquire single-device reference context");
        let mut reference = Rank::new(&context, &config);
        let output = reference_step_with_catalogue(
            &fixture,
            &lowering.linear_orders,
            &lowering.combine_orders,
            &catalogue,
            &mut reference,
            &tokens,
            &positions,
        );
        reference.close();
        output
    };
    let host_capacity = CapacitySnapshot::measured_host(
        &moxie_host::read().expect("measure host capacity"),
        1 << 20,
    )
    .expect("host capacity snapshot");
    let local = local_config(&config);
    let mut workers = DenseRankWorkers::spawn(DenseRankWorkerConfig {
        ranks: [RankId(1), RankId(2)],
        ordinals,
        geometry: geometry(&local, 4, 64, 6),
        heads: local.heads,
        max_rows: 5,
        host_capacity,
        deadline: DEADLINE,
    })
    .expect("spawn eager unordered TP2 pair");
    let actual = tp_worker_step(
        &mut workers,
        &fixture,
        &lowering,
        &capabilities,
        &catalogue,
        &tokens,
        &positions,
    );
    assert_eq!(
        actual, expected,
        "unordered TP2 output bytes equal the packed split reference"
    );
    workers.close().expect("close unordered TP2 workers");
}
