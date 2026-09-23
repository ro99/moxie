//! Task 0059/0065's dense and routed-device gate.
//!
//! The reduced Gemma composition root stays unchanged.  This test lowers its
//! dense and routed graph through the selected package, runs a multi-row
//! prefill and one decode row through the existing device KV authority, and
//! compares both outputs with the host interpreter on every visible GPU.
#![cfg(feature = "paged-attention-binding")]

use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::{Mutex, MutexGuard};

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::paged_attention::device::commit_paged_state;
use moxie_executor::{
    DenseGraphStep, PageGeometry, PagedAttentionRun, SelectedReservedPlan, Staging,
};
use moxie_format::bf16::f32_to_bf16_bits;
use moxie_graph::{Graph, ValueId};
use moxie_interp::{Cancel, Interpreter, KvCache};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{Phase, ResourceWorkload, lower_selected};
use moxie_state::{DeviceKvSequence, KvGeometry, LayerKv, Retention, SequenceState, StateKind};
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
    KvGeometry {
        layers: (0..config.layers)
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

fn owned_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    rows: &[u64],
    positions: &[u64],
    capability: &DeviceCapability,
) -> Vec<moxie_executor::OwnedBinding> {
    let mut bindings = Vec::new();
    for value in fixture.graph.inputs() {
        let source = if *value == fixture.tokens {
            Value::Index(rows.to_vec())
        } else if *value == fixture.positions {
            Value::Index(positions.to_vec())
        } else {
            panic!("unexpected dense input {}", value.0)
        };
        bindings.push(moxie_executor::OwnedBinding {
            value: *value,
            role: fixture.graph.spec(*value).expect("input spec").role,
            shape: concrete_shape(&fixture.graph, *value, rows.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&source),
        });
    }
    for value in fixture.graph.weights() {
        let source = fixture.weights.get(*value).expect("weight binding");
        bindings.push(moxie_executor::OwnedBinding {
            value: *value,
            role: fixture.graph.spec(*value).expect("weight spec").role,
            shape: concrete_shape(&fixture.graph, *value, rows.len()),
            layout: TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(source),
        });
    }
    bindings
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
        "task 0059 requires both 3090s and the 5060 Ti; saw {count}"
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
                    bindings: owned_bindings(fixture, &prompt, &prefill_positions, &capability),
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
                    bindings: owned_bindings(fixture, &decode, &decode_positions, &capability),
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
            bindings: owned_bindings(&fixture, &prompt, &positions, &capability),
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
