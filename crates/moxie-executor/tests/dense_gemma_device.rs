//! Task 0059/0065's dense and routed-device gate.
//!
//! The reduced Gemma composition root stays unchanged.  This test lowers its
//! dense and routed graph through the selected package, runs a multi-row
//! prefill and one decode row through the existing device KV authority, and
//! compares both outputs with the host interpreter on every visible GPU.
#![cfg(feature = "paged-attention-binding")]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::paged_attention::device::commit_paged_state;
use moxie_executor::{
    DenseGraphStep, HostExpertWeights, PageGeometry, PagedAttentionRun, PipelineStageWorker,
    PipelineWorkers, SelectedReservedPlan, SoloRankWorker, SoloRankWorkerConfig, Staging,
};
use moxie_format::bf16::f32_to_bf16_bits;
use moxie_graph::{Graph, GraphBuilder, OpParams, OracleRegistry, ValueId, ValueRole};
use moxie_interp::{Cancel, Interpreter, KvCache};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{
    Phase, PipelineLowering, ResourceWorkload, SelectedPlanCandidate, StageGraph,
    build_stage_graph, lower_host_experts, lower_pipeline, lower_selected,
    lower_selected_host_experts, lower_selected_ordered,
};
use moxie_state::{
    DeviceKvSequence, KvGeometry, LayerKv, ROOT, Retention, SequenceState, StateKind,
};
use moxie_types::{
    DeviceCapability, DeviceUuid, Dim, HostTier, PagePlacement, Precision, RankId, Scope,
    TensorLayout, Tier,
};

static DEVICE_TEST: Mutex<()> = Mutex::new(());
static FAIL_NEXT_STREAM_SYNC: AtomicBool = AtomicBool::new(false);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamSynchronize(stream: *mut c_void) -> c_int {
    if FAIL_NEXT_STREAM_SYNC.swap(false, SeqCst) {
        return 700;
    }
    // SAFETY: RTLD_NEXT resolves the real CUDA ABI symbol after this test
    // executable; the stream handle is forwarded unchanged.
    let symbol = unsafe { dlsym((-1isize) as *mut c_void, c"cuStreamSynchronize".as_ptr()) };
    assert!(!symbol.is_null(), "missing real cuStreamSynchronize");
    // SAFETY: the resolved symbol has the CUDA cuStreamSynchronize ABI.
    let real: unsafe extern "C" fn(*mut c_void) -> c_int = unsafe { std::mem::transmute(symbol) };
    // SAFETY: the function pointer and argument match the CUDA ABI.
    unsafe { real(stream) }
}

struct FailNextStreamSynchronize;

impl FailNextStreamSynchronize {
    fn arm() -> Self {
        FAIL_NEXT_STREAM_SYNC.store(true, SeqCst);
        Self
    }
}

impl Drop for FailNextStreamSynchronize {
    fn drop(&mut self) {
        FAIL_NEXT_STREAM_SYNC.store(false, SeqCst);
    }
}

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
                let layer = config.layer_geometry(layer);
                LayerKv {
                    kv_heads: layer.kv_heads as usize,
                    key_dim: layer.head_dim as usize,
                    value_dim: layer.head_dim as usize,
                    retention: match layer.window {
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

fn measured_ledger(ctx: &RankContext) -> Ledger {
    let device = CapacitySnapshot::measured(&ctx.measure().expect("measure device"), 1 << 20)
        .expect("device capacity");
    let host = CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
        .expect("host capacity");
    Ledger::new([device, host]).expect("ledger")
}

fn pageable_host_bytes(ledger: &Ledger) -> u64 {
    ledger.committed(Scope::Host, Tier::Host(HostTier::Pageable))
}

fn concrete_shape(graph: &Graph, value: ValueId, rows: usize) -> Vec<u64> {
    graph
        .spec(value)
        .expect("graph value spec")
        .shape
        .iter()
        .map(|dim| match dim {
            Dim::Const(value) => *value,
            Dim::Symbol(_) => rows as u64,
            other => panic!("unexpected symbolic fixture dimension {other:?}"),
        })
        .collect()
}

fn encode_value(value: &Value) -> Vec<u8> {
    match value {
        Value::Index(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Value::Float(tensor) => tensor
            .data()
            .iter()
            .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
            .collect(),
        Value::Route(_) => panic!("the dense fixture has no external route"),
    }
}

fn stage_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    stage: Option<&StageGraph>,
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<moxie_executor::OwnedBinding> {
    stage_bindings_omitting(fixture, stage, tokens, positions, capability, None)
}

fn stage_bindings_omitting(
    fixture: &moxie_cli::fixture::Fixture,
    stage: Option<&StageGraph>,
    tokens: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
    omit_read: Option<ValueId>,
) -> Vec<moxie_executor::OwnedBinding> {
    let graph = stage.map_or(&fixture.graph, |stage| &stage.graph);
    let mut bindings = Vec::new();
    let reads: Vec<_> = stage.map_or_else(
        || {
            fixture
                .graph
                .inputs()
                .iter()
                .map(|value| (*value, *value))
                .collect()
        },
        |stage| {
            stage
                .reads
                .iter()
                .map(|read| (read.original, read.local))
                .collect()
        },
    );
    for (original, local) in reads {
        if Some(original) == omit_read {
            continue;
        }
        let source = if original == fixture.tokens {
            Value::Index(tokens.to_vec())
        } else if original == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            panic!("unexpected graph input {}", original.0)
        };
        bindings.push(moxie_executor::OwnedBinding {
            value: local,
            role: graph.spec(local).expect("read spec").role,
            shape: concrete_shape(graph, local, tokens.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    let weights: Vec<_> = stage.map_or_else(
        || {
            fixture
                .graph
                .weights()
                .iter()
                .map(|value| (*value, *value, None))
                .collect()
        },
        |stage| {
            stage
                .weights
                .iter()
                .map(|weight| {
                    assert!(
                        weight.slice.is_none(),
                        "the whole graph has no input-axis weight slices"
                    );
                    (weight.original, weight.local, weight.rows.clone())
                })
                .collect()
        },
    );
    for (original, local, rows) in weights {
        let source = fixture.weights.get(original).expect("weight binding");
        let bytes = if let Some(rows) = rows {
            let tensor = source.as_float().expect("BF16 expert weight");
            let row_elements = tensor.shape()[1..].iter().product::<usize>();
            let start = usize::try_from(rows.start).expect("row start fits usize") * row_elements;
            let end = usize::try_from(rows.end).expect("row end fits usize") * row_elements;
            tensor.data()[start..end]
                .iter()
                .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
                .collect()
        } else {
            encode_value(source)
        };
        bindings.push(moxie_executor::OwnedBinding {
            value: local,
            role: graph.spec(local).expect("weight spec").role,
            shape: concrete_shape(graph, local, tokens.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes,
        });
    }
    bindings
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_decode(
    prefill_candidate: SelectedPlanCandidate,
    decode_candidate: SelectedPlanCandidate,
    graph: &Graph,
    capability: &DeviceCapability,
    catalogue: &moxie_types::KernelCatalogue,
    config: &moxie_models::gemma4::TextConfig,
    context: &RankContext,
    stream: &Stream<'_>,
    prompt_rows: usize,
    prefill_bindings: Vec<moxie_executor::OwnedBinding>,
    decode_bindings: Vec<moxie_executor::OwnedBinding>,
    third_decode: Option<(SelectedPlanCandidate, Vec<moxie_executor::OwnedBinding>)>,
    host_experts: &[HostExpertWeights<'_>],
    check_empty_refusal: bool,
    expected_joins: usize,
) -> (Vec<u8>, Vec<u8>, Option<Vec<u8>>) {
    let mut state =
        DeviceKvSequence::new(geometry(config, 4, 64, prompt_rows)).expect("device state");
    let mut ledger = measured_ledger(context);
    let mut runs = admit_runs(&mut ledger, context, config, &state, prompt_rows as u64);
    let (prefill, decode, third) = {
        let mut execute = |candidate, bindings, check_empty| {
            let plan = SelectedReservedPlan::admit(
                candidate,
                graph,
                capability,
                catalogue,
                &mut ledger,
                context,
            )
            .unwrap_or_else(|refused| panic!("step admission: {refused:?}"));
            let transaction = state.begin().expect("step transaction");
            let (plan, bindings) = if check_empty {
                let refused = plan
                    .execute_dense(DenseGraphStep {
                        graph,
                        capability,
                        catalogue,
                        ctx: context,
                        stream,
                        state: &mut state,
                        transaction,
                        runs: &mut runs,
                        bindings,
                        host_experts: &[],
                    })
                    .expect_err("empty host weights must be refused before launch");
                assert!(
                    matches!(
                        &refused.error,
                        moxie_types::Error::InvalidRequest {
                            field: "host_experts",
                            ..
                        }
                    ) && refused.held.is_none()
                        && refused.plan.is_some(),
                    "empty host weights must produce a pre-launch typed refusal"
                );
                (
                    refused.plan.expect("refusal returns admitted plan"),
                    refused.bindings,
                )
            } else {
                (plan, bindings)
            };
            let result = plan
                .execute_dense(DenseGraphStep {
                    graph,
                    capability,
                    catalogue,
                    ctx: context,
                    stream,
                    state: &mut state,
                    transaction,
                    runs: &mut runs,
                    bindings,
                    host_experts,
                })
                .map_err(|refused| refused.error)
                .expect("step execution")
                .finish()
                .map_err(|refused| refused.error)
                .expect("step finish");
            if !host_experts.is_empty() {
                assert_eq!(
                    result
                        .launch_order
                        .iter()
                        .filter(|name| name.as_str() == "combine-host-join")
                        .count(),
                    expected_joins,
                    "one visible host join per routed layer on {}",
                    capability.uuid
                );
            }
            let output = result.output;
            commit_paged_state(&mut state, transaction, 0, &mut runs, stream).expect("commit step");
            result
                .plan
                .close(&mut ledger)
                .map_err(|refused| refused.error)
                .expect("close step plan");
            output
        };
        let prefill = execute(prefill_candidate, prefill_bindings, check_empty_refusal);
        let decode = execute(decode_candidate, decode_bindings, false);
        let third = third_decode.map(|(candidate, bindings)| execute(candidate, bindings, false));
        (prefill, decode, third)
    };
    for run in runs.drain(..) {
        run.close(&mut ledger)
            .map_err(|refused| refused.error)
            .expect("close attention run");
    }
    assert!(ledger.outstanding().is_empty());
    (prefill, decode, third)
}

fn host_step(
    fixture: &moxie_cli::fixture::Fixture,
    state: &mut SequenceState,
    cache: &mut KvCache,
    tokens: &[u64],
    positions: &[u64],
) -> Vec<f32> {
    let mut bindings = fixture.weights.clone();
    bindings.set(fixture.tokens, Value::Index(tokens.to_vec()));
    bindings.set(fixture.positions, Value::Index(positions.to_vec()));
    Interpreter::new()
        .run(
            &fixture.graph,
            &bindings,
            state,
            moxie_state::ROOT,
            cache,
            &Cancel::never(),
        )
        .expect("host reduced-Gemma step")
        .logits
        .data()
        .to_vec()
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
            .expect("build stage graph for worker geometry")
        })
        .collect()
}

fn spawn_stage_workers(
    stages: &[StageGraph],
    config: &moxie_models::gemma4::TextConfig,
    devices: &[DeviceCapability],
    host_capacity: &CapacitySnapshot,
    rank_seed: &mut u32,
) -> Vec<SoloRankWorker> {
    assert_eq!(stages.len(), devices.len());
    stages
        .iter()
        .zip(devices)
        .map(|(stage, device)| {
            let rank = RankId(*rank_seed);
            *rank_seed += 1;
            SoloRankWorker::spawn(SoloRankWorkerConfig {
                rank,
                ordinal: device.ordinal,
                geometry: geometry_for_layers(
                    config,
                    stage.state_layers.values().copied(),
                    4,
                    64,
                    5,
                ),
                heads: config.heads,
                max_rows: 5,
                host_capacity: host_capacity.clone(),
                deadline: Duration::from_secs(60),
            })
            .unwrap_or_else(|error| panic!("spawn stage worker on {}: {error}", device.uuid))
        })
        .collect()
}

struct PipelineRun<'a> {
    fixture: &'a moxie_cli::fixture::Fixture,
    config: &'a moxie_models::gemma4::TextConfig,
    lowering: &'a PipelineLowering,
    catalogue: &'a moxie_types::KernelCatalogue,
    devices: &'a [DeviceCapability],
}

impl PipelineRun<'_> {
    fn execute<'w>(
        &self,
        workers: &'w mut PipelineWorkers,
        microbatches: &[Range<u64>],
        cancel: &AtomicBool,
        cancel_at: Option<(usize, usize)>,
    ) -> moxie_types::Result<moxie_executor::PipelineStep<'w>> {
        let mut oracles = OracleRegistry::new();
        moxie_oracles::register(&mut oracles).expect("register pipeline oracles");
        let mut make_bindings = |request: moxie_executor::StageBindings<'_>| {
            let stage = request.stage;
            let graph = request.graph;
            let rows = request.rows;
            assert!(
                request.rank.is_none(),
                "solo pipeline has no rank sub-stage"
            );
            if let Some((cancel_stage, cancel_batch)) = cancel_at {
                let batch = microbatches.iter().position(|candidate| *candidate == rows);
                if stage == cancel_stage && batch == Some(cancel_batch) {
                    cancel.store(true, SeqCst);
                }
            }
            let tokens: Vec<_> = rows.clone().map(|row| row % self.config.vocab).collect();
            let positions: Vec<_> = rows.clone().collect();
            let incoming = if stage > 0 {
                Some(self.lowering.handoffs()[stage - 1])
            } else {
                None
            };
            Ok(stage_bindings_omitting(
                self.fixture,
                Some(graph),
                &tokens,
                &positions,
                &self.devices[stage],
                incoming,
            ))
        };
        workers.execute(
            &self.fixture.graph,
            self.lowering,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            self.catalogue,
            microbatches,
            &mut make_bindings,
            cancel,
        )
    }
}

fn assert_pipeline_reservations(
    workers: &mut PipelineWorkers,
    spawned: &[(u64, u64, usize)],
    label: &str,
) {
    let current = workers.stats().expect("pipeline stage stats");
    assert!(
        current
            .iter()
            .zip(spawned)
            .all(|(actual, initial)| actual.2 == initial.2),
        "{label}: outstanding reservations changed"
    );
}

fn host_prefill_and_decodes(
    fixture: &moxie_cli::fixture::Fixture,
    config: &moxie_models::gemma4::TextConfig,
) -> [Vec<f32>; 3] {
    let mut state = SequenceState::new([StateKind::KvPages]);
    let mut cache =
        KvCache::for_branch(config.layers as usize, &state, ROOT).expect("host pipeline cache");
    let tokens: Vec<_> = (0..5).map(|row| row % config.vocab).collect();
    state
        .append_prompt(ROOT, tokens.len() as u64)
        .expect("append host prefill");
    let prefill = host_step(fixture, &mut state, &mut cache, &tokens, &[0, 1, 2, 3, 4]);
    state.append_prompt(ROOT, 1).expect("append host decode");
    let decode_5 = host_step(fixture, &mut state, &mut cache, &[5 % config.vocab], &[5]);
    state
        .append_prompt(ROOT, 1)
        .expect("append host second decode");
    let decode_6 = host_step(fixture, &mut state, &mut cache, &[6 % config.vocab], &[6]);
    [prefill, decode_5, decode_6]
}

fn solo_worker_step(
    worker: &mut SoloRankWorker,
    fixture: &moxie_cli::fixture::Fixture,
    catalogue: &moxie_types::KernelCatalogue,
    capability: &DeviceCapability,
    tokens: &[u64],
    positions: &[u64],
    visible_tokens: u64,
) -> moxie_types::Result<Vec<u8>> {
    worker.step(
        fixture.graph.clone(),
        catalogue.clone(),
        stage_bindings(fixture, None, tokens, positions, capability),
        tokens.len() as u64,
        visible_tokens,
    )
}

fn decode_outputs(
    batches: &[Range<u64>],
    execute: impl FnMut(&Range<u64>) -> Vec<u8>,
) -> Vec<Vec<u8>> {
    batches.iter().map(execute).collect()
}

fn bf16_ulp(value: f32) -> f32 {
    let exponent = (value.abs().to_bits() >> 23) & 0xff;
    match exponent {
        0 => f32::from_bits(1 << 16),
        1..=7 => f32::from_bits(1 << (15 + exponent)),
        _ => f32::from_bits((exponent - 7) << 23),
    }
}

fn assert_logits(label: &str, got: &[u8], want: &[f32]) {
    assert_eq!(got.len(), want.len() * 4, "{label}: output byte extent");
    let mut max_ulp = 0.0f32;
    for (index, expected) in want.iter().copied().enumerate() {
        let offset = index * 4;
        let actual = f32::from_le_bytes(got[offset..offset + 4].try_into().expect("F32 word"));
        assert!(actual.is_finite(), "{label}: nonfinite logit {index}");
        let distance = (actual - expected).abs() / bf16_ulp(expected);
        max_ulp = max_ulp.max(distance);
        assert!(
            distance <= 1.0,
            "{label}: logit {index} is {actual} vs {expected}, {distance} BF16 ULP"
        );
    }
    eprintln!("{label}: max_bf16_ulp={max_ulp:.3}");
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
        .expect("paged attention descriptor for device architecture")
}

fn admit_runs<'ctx>(
    ledger: &mut Ledger,
    context: &'ctx RankContext,
    config: &moxie_models::gemma4::TextConfig,
    state: &DeviceKvSequence,
    max_rows: u64,
) -> Vec<PagedAttentionRun<'ctx>> {
    let descriptor = paged_descriptor(context.capability());
    let lineage_capacity = u64::try_from(state.geometry().expect("device geometry").max_tokens)
        .expect("lineage capacity fits u64")
        .checked_add(1)
        .expect("lineage entry count fits u64");
    (0..config.layers)
        .map(|layer| {
            let declared = config.layer_geometry(layer);
            let layout = state.layout(layer as usize).expect("device layer layout");
            let geometry = PageGeometry {
                kv_heads: declared.kv_heads,
                head_dim: declared.head_dim,
                page_tokens: state.geometry().expect("device geometry").page_tokens as u64,
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
                .expect("paged attention run admission")
        })
        .collect()
}

#[test]
fn reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu() {
    let _guard = one_at_a_time();
    let count = device_count().expect("enumerate CUDA devices");
    assert!(
        count >= 3,
        "requires both 3090s and the 5060 Ti; saw {count}"
    );
    let mut cases: Vec<_> = [
        moxie_cli::gemma::Shape::A,
        moxie_cli::gemma::Shape::B,
        moxie_cli::gemma::Shape::C,
    ]
    .into_iter()
    .map(|shape| {
        (
            shape.name(),
            shape.config(),
            moxie_cli::gemma::build(shape).expect("build Gemma fixture"),
        )
    })
    .collect();
    let mut top3_config = moxie_cli::gemma::Shape::C.config();
    top3_config
        .moe
        .as_mut()
        .expect("Shape C MoE geometry")
        .top_k = 3;
    let top3_fixture = moxie_cli::gemma::build_with_config(top3_config.clone())
        .expect("build top-3 Gemma C fixture");
    let mut tied_fixture = moxie_cli::gemma::build_with_config(top3_config.clone())
        .expect("build tied Gemma C fixture");
    let router_weights: Vec<_> = tied_fixture
        .graph
        .weights()
        .iter()
        .copied()
        .filter(|weight| {
            tied_fixture
                .graph
                .name(*weight)
                .is_some_and(|name| name.starts_with("router_proj."))
        })
        .collect();
    assert!(!router_weights.is_empty(), "Shape C has router projections");
    for weight in router_weights {
        let zero_projection = {
            let tensor = tied_fixture
                .weights
                .get(weight)
                .unwrap()
                .as_float()
                .unwrap();
            HostTensor::bf16(vec![0.0; tensor.data().len()], tensor.shape().to_vec())
                .expect("zero BF16 router projection")
        };
        tied_fixture
            .weights
            .set(weight, Value::Float(zero_projection));
    }
    cases.push(("gemma-c-tied", top3_config.clone(), tied_fixture));
    cases.push(("gemma-c-top3", top3_config, top3_fixture));

    for ordinal in 0..count {
        let context = RankContext::acquire(RankId(59_000 + ordinal), ordinal)
            .expect("acquire GPU rank context");
        let capability = query_device(ordinal).expect("query GPU capability");
        assert_eq!(context.uuid(), capability.uuid);
        let stream = Stream::new(&context).expect("create stream");
        for (label, config, fixture) in &cases {
            let prompt: Vec<u64> = (0..5).map(|row| row % config.vocab).collect();
            let decode = vec![5 % config.vocab];
            let prefill_positions: Vec<u64> = (0..prompt.len() as u64).collect();
            let decode_positions = vec![prompt.len() as u64];

            let mut host_state = SequenceState::new([StateKind::KvPages]);
            let mut host_cache =
                KvCache::for_branch(config.layers as usize, &host_state, moxie_state::ROOT)
                    .expect("host cache");
            let host_prefill = host_step(
                fixture,
                &mut host_state,
                &mut host_cache,
                &prompt,
                &prefill_positions,
            );
            let host_decode = host_step(
                fixture,
                &mut host_state,
                &mut host_cache,
                &decode,
                &decode_positions,
            );

            let kv = geometry(config, 4, 64, prompt.len() + 1);
            let mut device_state = DeviceKvSequence::new(kv).expect("device state");
            let mut ledger = measured_ledger(&context);
            let mut runs = admit_runs(
                &mut ledger,
                &context,
                config,
                &device_state,
                prompt.len() as u64,
            );
            let paged_run_host_bytes = (0..config.layers)
                .map(|layer| {
                    let declared = config.layer_geometry(layer);
                    let layout = device_state
                        .layout(layer as usize)
                        .expect("device layer layout");
                    let page_tokens = device_state
                        .geometry()
                        .expect("device geometry")
                        .page_tokens as u64;
                    let geometry = PageGeometry {
                        kv_heads: declared.kv_heads,
                        head_dim: declared.head_dim,
                        page_tokens,
                        pages: layout.pages,
                    };
                    let view_and_run_table = layout
                        .pages
                        .checked_mul(core::mem::size_of::<u32>() as u64 * 2)
                        .ok_or(moxie_types::Error::Dim(moxie_types::DimError::Overflow))?;
                    let placements = (prompt.len() as u64)
                        .div_ceil(page_tokens)
                        .checked_add(1)
                        .and_then(|capacity| {
                            capacity.checked_mul(core::mem::size_of::<PagePlacement>() as u64)
                        })
                        .ok_or(moxie_types::Error::Dim(moxie_types::DimError::Overflow))?;
                    geometry
                        .page_table_upload_bytes()?
                        .checked_add(view_and_run_table)
                        .and_then(|bytes| bytes.checked_add(placements))
                        .ok_or(moxie_types::Error::Dim(moxie_types::DimError::Overflow))
                })
                .try_fold(0u64, |total, bytes| {
                    let bytes = bytes?;
                    total
                        .checked_add(bytes)
                        .ok_or(moxie_types::Error::Dim(moxie_types::DimError::Overflow))
                })
                .expect("page-table host workspace extent");
            let host_with_paged_runs = pageable_host_bytes(&ledger);
            assert!(
                host_with_paged_runs >= paged_run_host_bytes,
                "each admitted paged run must reserve its upload and append metadata"
            );
            let dense_catalogue = moxie_kernels::dense_graph_catalogue();
            let prefill_workload = ResourceWorkload {
                phase: Phase::Prefill,
                rows: prompt.len() as u64,
                visible_tokens: prompt.len() as u64,
                branch_rows: prompt.len() as u64,
                output: fixture.graph.output(),
                device: capability.uuid,
                paged_state_capacity: None,
            };
            let prefill_candidate = lower_selected(
                &fixture.graph,
                prefill_workload,
                &capability,
                &dense_catalogue,
            )
            .expect("prefill lowering");
            let prefill_host_workspace = prefill_candidate.host_workspace_bytes();
            let host_before_prefill_plan = pageable_host_bytes(&ledger);
            let prefill_plan = SelectedReservedPlan::admit(
                prefill_candidate,
                &fixture.graph,
                &capability,
                &dense_catalogue,
                &mut ledger,
                &context,
            )
            .map_err(|refused| match refused {
                moxie_executor::SelectedAdmitRefused::Invalid { error, .. }
                | moxie_executor::SelectedAdmitRefused::Held { error, .. } => error,
                moxie_executor::SelectedAdmitRefused::Rejected { rejection, .. } => {
                    rejection.into()
                }
            })
            .expect("prefill admission");
            assert!(
                pageable_host_bytes(&ledger) >= host_before_prefill_plan + prefill_host_workspace,
                "prefill K/V staging must be admitted concurrently with page-table workspace"
            );
            let prefill_txn = device_state.begin().expect("prefill transaction");
            let prefill_result = prefill_plan
                .execute_dense(DenseGraphStep {
                    graph: &fixture.graph,
                    capability: &capability,
                    catalogue: &dense_catalogue,
                    ctx: &context,
                    stream: &stream,
                    state: &mut device_state,
                    transaction: prefill_txn,
                    runs: &mut runs,
                    bindings: stage_bindings(
                        fixture,
                        None,
                        &prompt,
                        &prefill_positions,
                        &capability,
                    ),
                    host_experts: &[],
                })
                .map_err(|refused| refused.error)
                .expect("prefill device execution")
                .finish()
                .map_err(|refused| refused.error)
                .expect("prefill completion");
            assert_logits(
                &format!(
                    "{} {} prefill {} sm_{}{}",
                    label,
                    capability.uuid,
                    ordinal,
                    capability.compute_major,
                    capability.compute_minor
                ),
                &prefill_result.output,
                &host_prefill,
            );
            commit_paged_state(&mut device_state, prefill_txn, 0, &mut runs, &stream)
                .expect("commit prefill state");
            prefill_result
                .plan
                .close(&mut ledger)
                .map_err(|refused| refused.error)
                .expect("close prefill plan");
            assert_eq!(
                pageable_host_bytes(&ledger),
                host_with_paged_runs,
                "prefill host workspace must be released before decode admission"
            );

            let decode_workload = ResourceWorkload {
                phase: Phase::Decode,
                rows: 1,
                visible_tokens: prompt.len() as u64 + 1,
                branch_rows: 1,
                output: fixture.graph.output(),
                device: capability.uuid,
                paged_state_capacity: None,
            };
            let decode_candidate = lower_selected(
                &fixture.graph,
                decode_workload,
                &capability,
                &dense_catalogue,
            )
            .expect("decode lowering");
            let decode_host_workspace = decode_candidate.host_workspace_bytes();
            let host_before_decode_plan = pageable_host_bytes(&ledger);
            let decode_plan = SelectedReservedPlan::admit(
                decode_candidate,
                &fixture.graph,
                &capability,
                &dense_catalogue,
                &mut ledger,
                &context,
            )
            .map_err(|refused| match refused {
                moxie_executor::SelectedAdmitRefused::Invalid { error, .. }
                | moxie_executor::SelectedAdmitRefused::Held { error, .. } => error,
                moxie_executor::SelectedAdmitRefused::Rejected { rejection, .. } => {
                    rejection.into()
                }
            })
            .expect("decode admission");
            assert!(
                pageable_host_bytes(&ledger) >= host_before_decode_plan + decode_host_workspace,
                "decode K/V staging must be admitted concurrently with page-table workspace"
            );
            let decode_txn = device_state.begin().expect("decode transaction");
            let decode_result = decode_plan
                .execute_dense(DenseGraphStep {
                    graph: &fixture.graph,
                    capability: &capability,
                    catalogue: &dense_catalogue,
                    ctx: &context,
                    stream: &stream,
                    state: &mut device_state,
                    transaction: decode_txn,
                    runs: &mut runs,
                    bindings: stage_bindings(
                        fixture,
                        None,
                        &decode,
                        &decode_positions,
                        &capability,
                    ),
                    host_experts: &[],
                })
                .map_err(|refused| refused.error)
                .expect("decode device execution")
                .finish()
                .map_err(|refused| refused.error)
                .expect("decode completion");
            assert_logits(
                &format!(
                    "{} {} decode {} sm_{}{}",
                    label,
                    capability.uuid,
                    ordinal,
                    capability.compute_major,
                    capability.compute_minor
                ),
                &decode_result.output,
                &host_decode,
            );
            commit_paged_state(&mut device_state, decode_txn, 0, &mut runs, &stream)
                .expect("commit decode state");
            decode_result
                .plan
                .close(&mut ledger)
                .map_err(|refused| refused.error)
                .expect("close decode plan");
            for run in runs.drain(..) {
                run.close(&mut ledger)
                    .map_err(|refused| refused.error)
                    .expect("close attention run");
            }
            assert!(
                ledger.outstanding().is_empty(),
                "step left ledger reservations"
            );
            eprintln!(
                "PASS dense Gemma {} on UUID {} (SM{}{}) prefill+decode",
                label, capability.uuid, capability.compute_major, capability.compute_minor
            );
        }
    }
}

#[test]
fn solo_rank_worker_matches_the_direct_path_on_every_gpu() {
    let _guard = one_at_a_time();
    let count = device_count().expect("enumerate CUDA devices");
    assert!(count >= 3, "requires all three GPUs; saw {count}");
    let shape = moxie_cli::gemma::Shape::A;
    let config = shape.config();
    let fixture = moxie_cli::gemma::build(shape).expect("build dense Gemma fixture");
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let prompt: Vec<u64> = (0..5).map(|row| row % config.vocab).collect();
    let decode = vec![5 % config.vocab];
    let next_decode = vec![6 % config.vocab];
    let after_next_decode = vec![7 % config.vocab];
    let prompt_positions: Vec<u64> = (0..5).collect();
    let decode_position = [5];
    let next_position = [6];
    let after_next_position = [7];

    for ordinal in 0..count {
        let capability = query_device(ordinal).expect("query GPU capability");
        let workload = |rows, visible_tokens| ResourceWorkload {
            phase: if rows == 1 {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            rows,
            visible_tokens,
            branch_rows: rows,
            output: fixture.graph.output(),
            device: capability.uuid,
            paged_state_capacity: None,
        };
        let prefill_candidate =
            lower_selected(&fixture.graph, workload(5, 5), &capability, &catalogue)
                .expect("prefill lowering");
        let decode_candidate =
            lower_selected(&fixture.graph, workload(1, 6), &capability, &catalogue)
                .expect("decode lowering");
        let third_candidate =
            lower_selected(&fixture.graph, workload(1, 7), &capability, &catalogue)
                .expect("third decode lowering");
        let context = RankContext::acquire(RankId(70_000 + ordinal), ordinal)
            .expect("acquire direct-path context");
        let stream = Stream::new(&context).expect("create direct-path stream");
        let reference = run_prefill_decode(
            prefill_candidate,
            decode_candidate,
            &fixture.graph,
            &capability,
            &catalogue,
            &config,
            &context,
            &stream,
            prompt.len(),
            stage_bindings(&fixture, None, &prompt, &prompt_positions, &capability),
            stage_bindings(&fixture, None, &decode, &decode_position, &capability),
            Some((
                third_candidate,
                stage_bindings(&fixture, None, &next_decode, &next_position, &capability),
            )),
            &[],
            false,
            0,
        );
        drop(stream);
        drop(context);

        let host_capacity =
            CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
                .expect("host capacity");
        let mut worker = SoloRankWorker::spawn(SoloRankWorkerConfig {
            rank: RankId(70_000 + ordinal),
            ordinal,
            geometry: geometry(&config, 4, 64, prompt.len() + 2),
            heads: config.heads,
            max_rows: 5,
            host_capacity,
            deadline: Duration::from_secs(30),
        })
        .expect("start solo rank worker");
        let step =
            |worker: &mut SoloRankWorker, tokens: &[u64], positions: &[u64], rows, visible| {
                debug_assert_eq!(rows, tokens.len() as u64);
                solo_worker_step(
                    worker,
                    &fixture,
                    &catalogue,
                    &capability,
                    tokens,
                    positions,
                    visible,
                )
            };
        let prefill = step(&mut worker, &prompt, &prompt_positions, 5, 5).expect("worker prefill");
        assert_eq!(prefill, reference.0, "prefill bytes on GPU {ordinal}");
        worker.prepare_commit().expect("prepare prefill");
        worker.apply_commit().expect("apply prefill");

        let decoded = step(&mut worker, &decode, &decode_position, 1, 6).expect("worker decode");
        assert_eq!(decoded, reference.1, "decode bytes on GPU {ordinal}");
        worker.prepare_commit().expect("prepare decode");
        worker.apply_commit().expect("apply decode");

        let before_abort = worker.stats().expect("stats before abort");
        let aborted =
            step(&mut worker, &next_decode, &next_position, 1, 7).expect("first abort step");
        assert_eq!(
            aborted.as_slice(),
            reference
                .2
                .as_ref()
                .expect("third direct output")
                .as_slice(),
            "third decode bytes on GPU {ordinal}"
        );
        step(&mut worker, &after_next_decode, &after_next_position, 1, 8)
            .expect("second step shares abort transaction");
        worker.abort().expect("abort both decode rows");
        assert_eq!(
            worker.stats().expect("stats after abort"),
            before_abort,
            "abort restores frontiers and reservations on GPU {ordinal}"
        );
        let retried = step(&mut worker, &next_decode, &next_position, 1, 7)
            .expect("retry decode after abort");
        assert_eq!(
            retried,
            reference.2.expect("third direct output"),
            "retried decode bytes on GPU {ordinal}"
        );
        worker.prepare_commit().expect("prepare retried decode");
        worker.apply_commit().expect("apply retried decode");

        worker.fail_next_step();
        assert!(matches!(
            step(&mut worker, &after_next_decode, &after_next_position, 1, 8),
            Err(moxie_types::Error::InvalidRequest { field: "fault", .. })
        ));
        worker.abort().expect("abort after pre-admission refusal");
        step(&mut worker, &after_next_decode, &after_next_position, 1, 8)
            .expect("clean step after injected refusal");
        worker.close().expect("close worker without reservations");
    }
}

#[test]
fn pipeline_runs_dense_and_routed_gemma_on_three_gpus() {
    let _guard = one_at_a_time();
    let count = match device_count() {
        Ok(count) => count,
        Err(error) => {
            eprintln!("SKIP pipeline Gemma: could not enumerate GPUs: {error}");
            return;
        }
    };
    let mut devices = Vec::new();
    for ordinal in 0..count {
        match query_device(ordinal) {
            Ok(device) => devices.push(device),
            Err(error) => {
                eprintln!("SKIP pipeline Gemma: could not query GPU {ordinal}: {error}");
                return;
            }
        }
    }
    let mut rtx3090s: Vec<_> = devices
        .iter()
        .filter(|device| device.name.contains("3090"))
        .cloned()
        .collect();
    rtx3090s.sort_by(|left, right| left.pci_bus_id.cmp(&right.pci_bus_id));
    let Some(rtx5060) = devices
        .iter()
        .find(|device| device.name.contains("5060 Ti"))
        .cloned()
    else {
        eprintln!("SKIP pipeline Gemma: an RTX 5060 Ti is not visible");
        return;
    };
    if rtx3090s.len() < 2 {
        eprintln!(
            "SKIP pipeline Gemma: need two RTX 3090s and one RTX 5060 Ti; found {} RTX 3090s",
            rtx3090s.len()
        );
        return;
    }
    let pair_devices = [rtx3090s[0].clone(), rtx3090s[1].clone()];
    let three_devices = [
        pair_devices[0].clone(),
        pair_devices[1].clone(),
        rtx5060.clone(),
    ];
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let host_capacity = CapacitySnapshot::measured_host(
        &moxie_host::read().expect("measure host capacity"),
        1 << 20,
    )
    .expect("host capacity snapshot");
    let prefill_batches = [0..2, 2..5];
    let decode_batches = [5..6, 6..7];
    let mut rank_seed = 71_000;

    let mut routed_config = moxie_cli::gemma::Shape::C.config();
    routed_config.moe.as_mut().expect("Shape C MoE").experts = 8;
    routed_config.vocab = 12;
    let dense_config = moxie_cli::gemma::Shape::A.config();
    let dense_fixture =
        moxie_cli::gemma::build(moxie_cli::gemma::Shape::A).expect("build dense Shape A fixture");
    let dense_graph = dense_fixture.graph.clone();
    // Shared activation cuts make the wrong graph yield a distinct valid plan.
    let mismatch_cuts = [14, 41];
    let dense_mismatch_lowering = lower_pipeline(&dense_graph, &mismatch_cuts)
        .expect("lower dense graph at shared mismatch cuts");
    let dense_cuts = [
        first_node_of_layer(&dense_fixture.graph, 1),
        first_node_of_layer(&dense_fixture.graph, 4),
    ];
    let dense_lowering = lower_pipeline(&dense_fixture.graph, &dense_cuts)
        .expect("lower dense Shape A for mismatched-lowering refusal");
    let cases = [
        ("dense", dense_config, dense_fixture),
        (
            "routed",
            routed_config.clone(),
            moxie_cli::gemma::build_with_config(routed_config)
                .expect("build routed Shape C fixture"),
        ),
    ];

    for (label, config, fixture) in cases {
        let cuts = [
            first_node_of_layer(&fixture.graph, 1),
            first_node_of_layer(&fixture.graph, 4),
        ];
        let lowering = if label == "dense" {
            dense_lowering.clone()
        } else {
            lower_pipeline(&fixture.graph, &cuts)
                .unwrap_or_else(|error| panic!("{label} Gemma layer handoff refused: {error}"))
        };
        let stages = pipeline_stage_graphs(&fixture, &lowering);
        assert_eq!(stages.len(), 3);
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.state_layers.values().copied().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            [vec![0], vec![1, 2, 3], vec![4, 5]],
            "each worker's state geometry follows its original attention layers"
        );
        let host = host_prefill_and_decodes(&fixture, &config);
        let host_cancel = AtomicBool::new(false);
        let run = PipelineRun {
            fixture: &fixture,
            config: &config,
            lowering: &lowering,
            catalogue: &catalogue,
            devices: &three_devices,
        };
        let mut workers = PipelineWorkers::new(
            spawn_stage_workers(
                &stages,
                &config,
                &three_devices,
                &host_capacity,
                &mut rank_seed,
            )
            .into_iter()
            .map(PipelineStageWorker::Solo)
            .collect(),
        )
        .expect("create three-stage pipeline");
        let spawned_stats = workers.stats().expect("stats after pipeline spawn");
        if label == "routed" {
            assert_ne!(
                lower_pipeline(&fixture.graph, &mismatch_cuts)
                    .expect("same cuts lower the routed graph"),
                dense_mismatch_lowering,
                "the mismatched graph must produce a distinct valid lowering"
            );
            let mut oracles = OracleRegistry::new();
            moxie_oracles::register(&mut oracles).expect("register mismatch-test oracles");
            let mut called = false;
            let mismatched = workers.execute(
                &fixture.graph,
                &dense_mismatch_lowering,
                moxie_oracles::HOST_REFERENCE,
                &oracles,
                &catalogue,
                &prefill_batches,
                &mut |_| {
                    called = true;
                    Err(moxie_types::Error::InvalidRequest {
                        field: "binding",
                        detail: "mismatched lowering reached bindings".into(),
                    })
                },
                &host_cancel,
            );
            assert!(matches!(
                mismatched,
                Err(moxie_types::Error::InvalidRequest {
                    field: "pipeline",
                    ..
                })
            ));
            drop(mismatched);
            assert!(!called, "invalid lowering must be refused before bindings");
            assert_eq!(
                workers.stats().expect("stats after mismatched lowering"),
                spawned_stats
            );
        }
        let first_clean = run
            .execute(&mut workers, &prefill_batches, &host_cancel, None)
            .expect("first clean pipeline prefill");
        let first_clean_logits = first_clean.logits().to_vec();
        let expected_handoff_bytes: Vec<_> = lowering
            .handoffs()
            .iter()
            .map(|handoff| {
                let ValueRole::Activation(precision) = fixture.graph.spec(*handoff).unwrap().role
                else {
                    panic!("pipeline handoff is not an activation");
                };
                5 * config.hidden * u64::from(precision.get().bits() / 8)
            })
            .collect();
        assert_eq!(first_clean.handoff_bytes(), expected_handoff_bytes);
        assert_logits(
            &format!("{label} first clean pipeline prefill"),
            &first_clean_logits,
            &host[0],
        );
        drop(first_clean);
        assert_eq!(
            workers.stats().expect("stats after clean abort"),
            spawned_stats
        );

        #[derive(Clone, Copy)]
        enum Fault {
            Step,
            Cancel(usize, usize),
            Prepare,
        }
        for fault in [
            Fault::Step,
            Fault::Cancel(1, 0),
            Fault::Prepare,
            Fault::Cancel(2, prefill_batches.len() - 1),
        ] {
            host_cancel.store(false, SeqCst);
            let error = match fault {
                Fault::Step => {
                    workers.fail_next_step(1).expect("inject stage-one failure");
                    run.execute(&mut workers, &prefill_batches, &host_cancel, None)
                        .expect_err("injected stage failure")
                }
                Fault::Cancel(stage, batch) => run
                    .execute(
                        &mut workers,
                        &prefill_batches,
                        &host_cancel,
                        Some((stage, batch)),
                    )
                    .expect_err("injected pipeline cancellation"),
                Fault::Prepare => {
                    workers
                        .refuse_next_prepare(2)
                        .expect("inject stage-two prepare refusal");
                    let step = run
                        .execute(&mut workers, &prefill_batches, &host_cancel, None)
                        .expect("pipeline prefill before prepare refusal");
                    assert_eq!(step.logits(), first_clean_logits);
                    step.commit().expect_err("injected prepare refusal")
                }
            };
            if matches!(fault, Fault::Step | Fault::Prepare) {
                assert!(matches!(
                    error,
                    moxie_types::Error::InvalidRequest { field: "fault", .. }
                ));
            } else {
                assert!(matches!(
                    error,
                    moxie_types::Error::Cancelled {
                        at: "pipeline stage"
                    }
                ));
            }
            host_cancel.store(false, SeqCst);
            assert_eq!(
                workers.stats().expect("stats after pipeline fault"),
                spawned_stats
            );
            let retry = run
                .execute(&mut workers, &prefill_batches, &host_cancel, None)
                .expect("clean retry after pipeline fault");
            assert_eq!(retry.logits(), first_clean_logits);
            drop(retry);
            assert_eq!(
                workers.stats().expect("stats after pipeline retry"),
                spawned_stats
            );
        }

        let prefill = run
            .execute(&mut workers, &prefill_batches, &host_cancel, None)
            .expect("three-GPU microbatched prefill");
        assert_eq!(prefill.handoff_bytes(), expected_handoff_bytes);
        let prefill_logits = prefill.commit().expect("commit three-GPU prefill");
        assert_logits(
            &format!("{label} three-GPU prefill"),
            &prefill_logits,
            &host[0],
        );
        assert_pipeline_reservations(&mut workers, &spawned_stats, "three-stage prefill");

        let three_stage_decodes = decode_outputs(&decode_batches, |rows| {
            let step = run
                .execute(&mut workers, std::slice::from_ref(rows), &host_cancel, None)
                .expect("three-GPU decode");
            let decode_bytes: Vec<_> = expected_handoff_bytes
                .iter()
                .map(|bytes| bytes / 5)
                .collect();
            assert_eq!(step.handoff_bytes(), decode_bytes);
            let logits = step.commit().expect("commit three-GPU decode");
            assert_pipeline_reservations(&mut workers, &spawned_stats, "three-stage decode");
            logits
        });
        for (logits, expected) in three_stage_decodes.iter().zip(&host[1..]) {
            assert_logits(&format!("{label} three-GPU decode"), logits, expected);
        }
        workers.close().expect("close three-stage workers");

        let pair_lowering =
            lower_pipeline(&fixture.graph, &[first_node_of_layer(&fixture.graph, 3)])
                .expect("lower two-stage 3090 pipeline");
        let pair_stages = pipeline_stage_graphs(&fixture, &pair_lowering);
        let pair_run = PipelineRun {
            fixture: &fixture,
            config: &config,
            lowering: &pair_lowering,
            catalogue: &catalogue,
            devices: &pair_devices,
        };
        let mut pair = PipelineWorkers::new(
            spawn_stage_workers(
                &pair_stages,
                &config,
                &pair_devices,
                &host_capacity,
                &mut rank_seed,
            )
            .into_iter()
            .map(PipelineStageWorker::Solo)
            .collect(),
        )
        .expect("create two-stage 3090 pipeline");
        let pair_spawned = pair.stats().expect("stats after pair spawn");
        let pair_prefill = pair_run
            .execute(&mut pair, &prefill_batches, &host_cancel, None)
            .expect("pair microbatched prefill");
        let pair_prefill_handoff = pair_prefill.handoff_bytes().to_vec();
        let pair_prefill_logits = pair_prefill.commit().expect("commit pair prefill");
        assert_eq!(pair_prefill_handoff, expected_handoff_bytes[..1]);
        assert_pipeline_reservations(&mut pair, &pair_spawned, "pair prefill");
        let pair_decodes = decode_outputs(&decode_batches, |rows| {
            let pair_step = pair_run
                .execute(&mut pair, std::slice::from_ref(rows), &host_cancel, None)
                .expect("pair decode");
            assert_eq!(pair_step.handoff_bytes(), &[pair_prefill_handoff[0] / 5]);
            let pair_logits = pair_step.commit().expect("commit pair decode");
            assert_pipeline_reservations(&mut pair, &pair_spawned, "pair decode");
            pair_logits
        });
        pair.close().expect("close two-stage pipeline");

        let mut whole_worker = SoloRankWorker::spawn(SoloRankWorkerConfig {
            rank: RankId(rank_seed),
            ordinal: pair_devices[0].ordinal,
            geometry: geometry(&config, 4, 64, 5),
            heads: config.heads,
            max_rows: 5,
            host_capacity: host_capacity.clone(),
            deadline: Duration::from_secs(60),
        })
        .expect("spawn whole-graph 3090 worker");
        rank_seed += 1;
        let mut whole_prefill = Vec::new();
        for rows in &prefill_batches {
            let tokens: Vec<_> = rows.clone().map(|row| row % config.vocab).collect();
            let positions: Vec<_> = rows.clone().collect();
            whole_prefill.extend(
                solo_worker_step(
                    &mut whole_worker,
                    &fixture,
                    &catalogue,
                    &pair_devices[0],
                    &tokens,
                    &positions,
                    rows.end,
                )
                .expect("whole-graph microbatch"),
            );
        }
        whole_worker
            .prepare_commit()
            .expect("prepare whole prefill");
        whole_worker.apply_commit().expect("apply whole prefill");
        assert_logits(
            &format!("{label} one-GPU microbatched reference"),
            &whole_prefill,
            &host[0],
        );
        assert_eq!(
            pair_prefill_logits, whole_prefill,
            "{label} pair prefill bits"
        );
        let whole_decodes = decode_outputs(&decode_batches, |rows| {
            let tokens: Vec<_> = rows.clone().map(|row| row % config.vocab).collect();
            let positions: Vec<_> = rows.clone().collect();
            let whole_logits = solo_worker_step(
                &mut whole_worker,
                &fixture,
                &catalogue,
                &pair_devices[0],
                &tokens,
                &positions,
                rows.end,
            )
            .expect("whole-graph decode");
            whole_worker.prepare_commit().expect("prepare whole decode");
            whole_worker.apply_commit().expect("apply whole decode");
            whole_logits
        });
        for ((pair, whole), expected) in pair_decodes.iter().zip(&whole_decodes).zip(&host[1..]) {
            assert_eq!(pair, whole, "{label} pair decode bits");
            assert_logits(&format!("{label} one-GPU decode"), whole, expected);
        }
        whole_worker.close().expect("close whole-graph worker");
        eprintln!(
            "PASS pipeline {label} on 3090 pair + 5060 Ti; handoff bytes {:?}",
            expected_handoff_bytes
        );
    }
}

#[test]
fn angle_copy_sync_failure_retains_its_host_source() {
    let _guard = one_at_a_time();
    let context = RankContext::acquire(RankId(59_090), 0).expect("acquire GPU rank context");
    let capability = query_device(0).expect("query GPU capability");
    let stream = Stream::new(&context).expect("create stream");
    let shape = moxie_cli::gemma::Shape::A;
    let config = shape.config();
    let fixture = moxie_cli::gemma::build(shape).expect("build dense fixture");
    let prompt: Vec<u64> = (0..5).map(|row| row % config.vocab).collect();
    let positions: Vec<u64> = (0..prompt.len() as u64).collect();
    let expected_angle_bytes = fixture
        .graph
        .nodes()
        .iter()
        .find_map(|node| match node.params {
            moxie_graph::OpParams::Rope { rotary_dim, .. } => {
                Some(prompt.len() * (rotary_dim as usize / 2) * 8)
            }
            _ => None,
        })
        .expect("dense fixture has a RoPE node");
    let mut device_state =
        DeviceKvSequence::new(geometry(&config, 4, 64, prompt.len())).expect("device state");
    let mut ledger = measured_ledger(&context);
    let mut runs = admit_runs(
        &mut ledger,
        &context,
        &config,
        &device_state,
        prompt.len() as u64,
    );
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let workload = ResourceWorkload {
        phase: Phase::Prefill,
        rows: prompt.len() as u64,
        visible_tokens: prompt.len() as u64,
        branch_rows: prompt.len() as u64,
        output: fixture.graph.output(),
        device: capability.uuid,
        paged_state_capacity: None,
    };
    let candidate = lower_selected(&fixture.graph, workload, &capability, &catalogue)
        .expect("prefill lowering");
    let plan = SelectedReservedPlan::admit(
        candidate,
        &fixture.graph,
        &capability,
        &catalogue,
        &mut ledger,
        &context,
    )
    .map_err(|refused| match refused {
        moxie_executor::SelectedAdmitRefused::Invalid { error, .. }
        | moxie_executor::SelectedAdmitRefused::Held { error, .. } => error,
        moxie_executor::SelectedAdmitRefused::Rejected { rejection, .. } => rejection.into(),
    })
    .expect("prefill admission");
    let transaction = device_state.begin().expect("prefill transaction");
    let _fault = FailNextStreamSynchronize::arm();
    let refused = plan
        .execute_dense(DenseGraphStep {
            graph: &fixture.graph,
            capability: &capability,
            catalogue: &catalogue,
            ctx: &context,
            stream: &stream,
            state: &mut device_state,
            transaction,
            runs: &mut runs,
            bindings: stage_bindings(&fixture, None, &prompt, &positions, &capability),
            host_experts: &[],
        })
        .expect_err("the injected angle-copy synchronization must refuse");
    assert_eq!(
        refused.retained_host_upload_bytes(),
        Some(expected_angle_bytes),
        "the angle source must remain held after stream synchronization fails"
    );
    device_state.abort(transaction).expect("abort refused step");
    // The injected CUDA error is quarantined as device loss; dropping this
    // refused lease intentionally withholds its device ranges and source.
    drop(refused);
}

#[test]
fn routed_gemma_lowers_through_the_dense_package() {
    let fixture =
        moxie_cli::gemma::build(moxie_cli::gemma::Shape::C).expect("build routed Gemma fixture");
    let capability = DeviceCapability {
        ordinal: 0,
        uuid: DeviceUuid::from_bytes([1u8; 16]),
        name: "fixture".into(),
        compute_major: 8,
        compute_minor: 6,
        total_memory_bytes: 1 << 30,
        multiprocessor_count: 1,
        pci_bus_id: "0000:00:00.0".into(),
        peer_access: Vec::new(),
        max_grid: (2_147_483_647, 65_535, 65_535),
    };
    let rows = 5;
    let workload = ResourceWorkload {
        phase: Phase::Prefill,
        rows,
        visible_tokens: rows,
        branch_rows: rows,
        output: fixture.graph.output(),
        device: capability.uuid,
        paged_state_capacity: None,
    };
    let catalogue = moxie_kernels::dense_graph_catalogue();
    let candidate = lower_selected(&fixture.graph, workload, &capability, &catalogue)
        .expect("routed graph lowers through the dense package");
    let route_nodes: Vec<_> = fixture
        .graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.params, moxie_graph::OpParams::Route { .. }))
        .collect();
    let routed_layers = route_nodes.len();
    assert!(routed_layers > 0, "Shape C has routed layers");
    for operation in [
        moxie_types::SemanticKernelOp::Route,
        moxie_types::SemanticKernelOp::ExpertMlp(moxie_types::GateTransform::GeluTanh),
        moxie_types::SemanticKernelOp::Combine,
    ] {
        assert_eq!(
            candidate
                .nodes()
                .iter()
                .filter(|node| node.descriptor.operation == operation)
                .count(),
            routed_layers,
            "one {operation:?} descriptor per routed layer"
        );
    }
    for node in route_nodes {
        let moxie_graph::OpParams::Route { top_k, .. } = node.params else {
            unreachable!("filtered route node")
        };
        assert_eq!(
            candidate
                .value(node.output)
                .expect("planned route value")
                .logical_bytes,
            rows * top_k * 8
        );
    }
}

#[test]
fn routed_gemma_host_experts_match_the_grouped_reference_on_every_gpu() {
    let _guard = one_at_a_time();
    let count = device_count().expect("enumerate CUDA devices");
    assert!(
        count >= 3,
        "host expert gate requires all three GPUs; saw {count}"
    );

    let mut config = moxie_cli::gemma::Shape::C.config();
    config.moe.as_mut().expect("Shape C MoE").experts = 8;
    let mut fixture = moxie_cli::gemma::build_with_config(config.clone())
        .expect("build eight-expert Shape C fixture");
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

    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("register graph oracles");
    let scaled_graph = {
        let source = &fixture.graph;
        let mut builder =
            GraphBuilder::new(source.nodes()[0].contract.oracle, source.rows_symbol());
        let mut values = BTreeMap::new();
        for value in source.inputs() {
            values.insert(
                *value,
                builder.input(
                    source.name(*value).expect("input name"),
                    source.spec(*value).expect("input spec").clone(),
                ),
            );
        }
        for value in source.weights() {
            values.insert(
                *value,
                builder
                    .weight(
                        source.name(*value).expect("weight name"),
                        source.spec(*value).expect("weight spec").clone(),
                    )
                    .expect("copy weight"),
            );
        }
        for node in source.nodes() {
            let mut params = node.params.clone();
            if let OpParams::Combine { output_scale, .. } = &mut params {
                *output_scale = 2.0;
            }
            let inputs: Vec<_> = node.inputs.iter().map(|value| values[value]).collect();
            values.insert(
                node.output,
                builder.node(params, &inputs).expect("copy graph node"),
            );
        }
        builder
            .finish(values[&source.output()], &oracles)
            .expect("finish scaled graph")
    };
    assert!(matches!(
        lower_host_experts(&scaled_graph, moxie_oracles::HOST_REFERENCE, &oracles),
        Err(moxie_plan::TensorParallelRefused::ScaledCombine { .. })
    ));

    let lowering = lower_host_experts(&fixture.graph, moxie_oracles::HOST_REFERENCE, &oracles)
        .expect("host expert lowering");
    let stage = lowering.device_stage();
    let expected_joins = lowering.joins().len();
    assert!(expected_joins > 0, "Shape C has routed expert layers");
    let mut host_storage = Vec::new();
    for (combine, join) in lowering.joins() {
        let source_combine = &fixture.graph.nodes()[combine.0 as usize];
        let source_expert = fixture
            .graph
            .nodes()
            .iter()
            .find(|node| node.output == source_combine.inputs[1])
            .expect("source ExpertMlp");
        let local_combine = stage.graph.nodes()[combine.0 as usize].id;
        let host_slice = |value: ValueId| {
            let tensor = fixture
                .weights
                .get(value)
                .expect("host expert weight")
                .as_float()
                .expect("BF16 expert weight");
            let row_elements = tensor.shape()[1..].iter().product::<usize>();
            let start =
                usize::try_from(join.first_host_expert()).expect("host row start") * row_elements;
            let end = usize::try_from(join.first_host_expert() + join.host_experts())
                .expect("host row end")
                * row_elements;
            tensor.data()[start..end]
                .iter()
                .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
                .collect::<Vec<_>>()
        };
        host_storage.push((
            local_combine,
            host_slice(source_expert.inputs[2]),
            host_slice(source_expert.inputs[3]),
        ));
    }
    let host_weights: Vec<_> = host_storage
        .iter()
        .map(|(combine, gate_up, down)| HostExpertWeights {
            combine: *combine,
            gate_up,
            down,
        })
        .collect();

    let prompt: Vec<u64> = (0..5).map(|row| row % config.vocab).collect();
    let decode = vec![5 % config.vocab];
    let prefill_positions: Vec<u64> = (0..prompt.len() as u64).collect();
    let decode_positions = vec![prompt.len() as u64];
    let catalogue = moxie_kernels::dense_graph_catalogue();

    for ordinal in 0..count {
        let context = RankContext::acquire(RankId(68_000 + ordinal), ordinal)
            .expect("acquire GPU rank context");
        let capability = query_device(ordinal).expect("query GPU capability");
        assert_eq!(context.uuid(), capability.uuid);
        let stream = Stream::new(&context).expect("create stream");
        let prefill_workload = ResourceWorkload {
            phase: Phase::Prefill,
            rows: prompt.len() as u64,
            visible_tokens: prompt.len() as u64,
            branch_rows: prompt.len() as u64,
            output: fixture.graph.output(),
            device: capability.uuid,
            paged_state_capacity: None,
        };
        let decode_workload = ResourceWorkload {
            phase: Phase::Decode,
            rows: 1,
            visible_tokens: prompt.len() as u64 + 1,
            branch_rows: 1,
            ..prefill_workload
        };
        let reference_prefill = lower_selected_ordered(
            &fixture.graph,
            prefill_workload,
            &capability,
            &catalogue,
            &BTreeMap::new(),
            lowering.reference_orders(),
            &BTreeMap::new(),
        )
        .expect("full-device grouped reference lowering");
        let host_prefill =
            lower_selected_host_experts(prefill_workload, &capability, &catalogue, &lowering)
                .expect("host-expert selected lowering");
        for combine in lowering.joins().keys() {
            let slots = fixture.graph.nodes()[combine.0 as usize].inputs[1];
            let expert = fixture
                .graph
                .nodes()
                .iter()
                .find(|node| node.output == slots)
                .expect("source ExpertMlp");
            for original_weight in [expert.inputs[2], expert.inputs[3]] {
                let local_weight = stage
                    .weights
                    .iter()
                    .find(|weight| weight.original == original_weight)
                    .expect("device-owned expert weight")
                    .local;
                let device_bytes = host_prefill
                    .value(local_weight)
                    .expect("planned device-owned expert bytes")
                    .logical_bytes;
                let reference_bytes = reference_prefill
                    .value(original_weight)
                    .expect("planned reference expert bytes")
                    .logical_bytes;
                assert_eq!(device_bytes.checked_mul(2), Some(reference_bytes));
            }
        }
        let reference_decode = lower_selected_ordered(
            &fixture.graph,
            decode_workload,
            &capability,
            &catalogue,
            &BTreeMap::new(),
            lowering.reference_orders(),
            &BTreeMap::new(),
        )
        .expect("reference decode lowering");
        let host_decode =
            lower_selected_host_experts(decode_workload, &capability, &catalogue, &lowering)
                .expect("host decode lowering");
        let reference_outputs = run_prefill_decode(
            reference_prefill,
            reference_decode,
            &fixture.graph,
            &capability,
            &catalogue,
            &config,
            &context,
            &stream,
            prompt.len(),
            stage_bindings(&fixture, None, &prompt, &prefill_positions, &capability),
            stage_bindings(&fixture, None, &decode, &decode_positions, &capability),
            None,
            &[],
            false,
            expected_joins,
        );
        let host_outputs = run_prefill_decode(
            host_prefill,
            host_decode,
            &stage.graph,
            &capability,
            &catalogue,
            &config,
            &context,
            &stream,
            prompt.len(),
            stage_bindings(
                &fixture,
                Some(stage),
                &prompt,
                &prefill_positions,
                &capability,
            ),
            stage_bindings(
                &fixture,
                Some(stage),
                &decode,
                &decode_positions,
                &capability,
            ),
            None,
            &host_weights,
            true,
            expected_joins,
        );
        for (phase, got, want) in [
            (
                "prefill",
                host_outputs.0.as_slice(),
                reference_outputs.0.as_slice(),
            ),
            (
                "decode",
                host_outputs.1.as_slice(),
                reference_outputs.1.as_slice(),
            ),
        ] {
            assert_eq!(
                got.len(),
                want.len(),
                "{phase} bytes on {}",
                capability.uuid
            );
            for (index, (actual, expected)) in
                got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate()
            {
                if actual != expected {
                    panic!(
                        "host expert {phase} differs from grouped reference on GPU {} at row {}, element {}: got {:02x?}, expected {:02x?}",
                        capability.uuid,
                        index / config.vocab as usize,
                        index % config.vocab as usize,
                        actual,
                        expected
                    );
                }
            }
        }
        eprintln!(
            "PASS host-owned routed Gemma on UUID {} (SM{}{}) prefill+decode",
            capability.uuid, capability.compute_major, capability.compute_minor
        );
    }
}
