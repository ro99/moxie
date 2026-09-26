//! Task 0102 test (e), isolated from `dense_tp2_device.rs`.
//!
//! A stalled rank's worker thread parks forever (`park_lost`), so its
//! `RankContext` -- and the physical GPU it holds -- never releases for the
//! rest of the process. `dense_tp2_device.rs`'s `tp2_worker_gate(true)`
//! already ends its process with exactly that permanent loss on this same
//! rank pair, as its own comment says ("the stall case pins the pair's
//! claims for this process, so run it last"). Two permanent losses of one
//! physical pair cannot both happen in one process, so this one test gets
//! its own binary -- the same GPU-pair-isolation task 0097's own R2 already
//! used for `dense_tp2_cublas_device.rs`.
#![cfg(all(feature = "paged-attention-binding", feature = "nccl"))]

use std::collections::BTreeSet;
use std::time::Duration;

use moxie_cuda::{device_count, query_device};
use moxie_engine::{HostTensor, Value};
use moxie_executor::{ChainBucket, DenseRankWorkerConfig, DenseRankWorkers};
use moxie_format::bf16::f32_to_bf16_bits;
use moxie_graph::{Graph, OracleRegistry, ValueId, ValueRole};
use moxie_memory::CapacitySnapshot;
use moxie_plan::{StageGraph, StageWeight, TensorParallelLowering, lower_tensor_parallel};
use moxie_state::{KvGeometry, LayerKv, Retention};
use moxie_types::{DeviceUuid, Dim, Error, Precision, RankId};

const DEADLINE: Duration = Duration::from_secs(20);
const RANK_UUIDS: [&str; 2] = [
    "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9",
    "GPU-81fe4578-59b2-37c4-421e-287cdac78704",
];

fn fixture() -> (
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
        found[0].expect("first 3090 present"),
        found[1].expect("second 3090 present"),
    ]
}

fn local_config(config: &moxie_models::gemma4::TextConfig) -> moxie_models::gemma4::TextConfig {
    let mut local = config.clone();
    local.heads /= 2;
    local.local_kv_heads /= 2;
    local
}

fn geometry(config: &moxie_models::gemma4::TextConfig) -> KvGeometry {
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
        page_tokens: 4,
        max_tokens: 64,
        tentative_rows: 6,
    }
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
    match value {
        Value::Index(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Value::Float(tensor) => tensor
            .data()
            .iter()
            .flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes())
            .collect(),
        Value::Route(_) => panic!("dense TP fixture has no route input"),
    }
}

fn stage_weight_value(fixture: &moxie_cli::fixture::Fixture, weight: &StageWeight) -> Value {
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

/// Every binding one compute stage's chain step needs, weight and non-weight
/// alike -- `load_chain`/`step_chain`'s callbacks each keep only the role
/// they want.
fn stage_bindings(
    fixture: &moxie_cli::fixture::Fixture,
    stage: &StageGraph,
    tokens: &[u64],
    positions: &[u64],
    capability: &moxie_types::DeviceCapability,
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
            layout: moxie_types::TensorLayout::ContiguousRowMajorV1,
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
            layout: moxie_types::TensorLayout::ContiguousRowMajorV1,
            device: capability.uuid,
            bytes: encode_value(&stage_weight_value(fixture, weight)),
        });
    }
    bindings
}

#[test]
fn chain_stall_loses_the_group_within_the_deadline() {
    let (fixture, config) = fixture();
    let lowering: TensorParallelLowering =
        lower_tensor_parallel(&fixture.graph, 2).expect("TP2 lowering");
    let ordinals = pair_ordinals();
    let local = local_config(&config);
    let capabilities = ordinals.map(|ordinal| query_device(ordinal).expect("capability"));
    let host_capacity =
        CapacitySnapshot::measured_host(&moxie_host::read().expect("measure host"), 1 << 20)
            .expect("host capacity");
    let mut workers = DenseRankWorkers::spawn(DenseRankWorkerConfig {
        ranks: [RankId(64_001), RankId(64_002)],
        ordinals,
        geometry: geometry(&local),
        heads: local.heads,
        max_rows: 5,
        host_capacity,
        deadline: DEADLINE,
    })
    .expect("spawn stall chain workers");

    let catalogue = moxie_kernels::dense_graph_catalogue();
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).expect("oracles");
    let tokens = [1u64, 4, 7, 2, 9];
    let positions = [0u64, 1, 2, 3, 4];
    let mut weights_cb = |rank: usize, stage: &StageGraph| {
        Ok(
            stage_bindings(&fixture, stage, &tokens, &positions, &capabilities[rank])
                .into_iter()
                .filter(|binding| matches!(binding.role, ValueRole::Weight(_)))
                .collect(),
        )
    };
    workers
        .load_chain(
            &fixture.graph,
            &lowering,
            moxie_oracles::HOST_REFERENCE,
            &oracles,
            &catalogue,
            vec![ChainBucket {
                rows: 5,
                visible_tokens: 5,
            }],
            &mut weights_cb,
        )
        .expect("load chain");

    workers.set_deadline_for_test(Duration::from_millis(500));
    workers.stall_before_next_rendezvous(1);
    let start = std::time::Instant::now();
    let mut inputs_cb = |rank: usize, stage: &StageGraph| {
        Ok(
            stage_bindings(&fixture, stage, &tokens, &positions, &capabilities[rank])
                .into_iter()
                .filter(|binding| !matches!(binding.role, ValueRole::Weight(_)))
                .collect(),
        )
    };
    let lost = workers.step_chain(5, 5, &mut inputs_cb);
    let elapsed = start.elapsed();
    assert!(
        matches!(lost, Err(Error::DeviceLost { .. })),
        "a stalled rank returns no logits: {lost:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "a lost group is reported within the deadline, took {elapsed:?}"
    );
    // Dropping a lost group is a safe no-op: `Drop` checks `self.lost` and
    // `rendezvous.lost()` before attempting `shutdown_inner`.
}
