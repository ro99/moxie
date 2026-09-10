//! Real-device proof that one pure graph/resource plan binds through the shared
//! ledger and arena on every visible UUID. No semantic kernel is launched.
#![cfg(feature = "driver")]

use moxie_cuda::RankContext;
use moxie_executor::{PlanAdmitRefused, ReservedPlan};
use moxie_graph::{
    Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
    ValueId, ValueRole,
};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{Phase, ResourceWorkload, lower};
use moxie_types::{
    ActivationPrecision, DeviceUuid, Dim, Precision, RankId, SymbolId, TensorLayout,
    WeightPrecision,
};

const ROWS: SymbolId = SymbolId(71);
const ORACLE: OracleId = OracleId("resource-plan-device-test");

fn registry() -> OracleRegistry {
    let mut registry = OracleRegistry::new();
    registry
        .register(
            Op::Linear,
            ORACLE,
            OracleEvidence {
                implementation: "resource_plan",
                test_module: "resource_plan",
            },
        )
        .unwrap();
    registry
}

fn build_graph(width: u64) -> (Graph, [ValueId; 3]) {
    let mut builder = GraphBuilder::new(ORACLE, ROWS);
    let input = builder.input(
        "x",
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(ROWS), Dim::constant(width)],
        ),
    );
    let mut previous = input;
    let mut outputs = [input; 3];
    for (index, output) in outputs.iter_mut().enumerate() {
        let weight = builder
            .weight(
                &format!("w{index}"),
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                    vec![Dim::constant(width), Dim::constant(width)],
                ),
            )
            .unwrap();
        previous = builder
            .node(
                OpParams::Linear {
                    in_features: width,
                    out_features: width,
                    bias: false,
                },
                &[previous, weight],
            )
            .unwrap();
        *output = previous;
    }
    (builder.finish(previous, &registry()).unwrap(), outputs)
}

fn workload(graph: &Graph, device: DeviceUuid) -> ResourceWorkload {
    ResourceWorkload {
        phase: Phase::Prefill,
        rows: 3,
        visible_tokens: 32_768,
        branch_rows: 3,
        output: graph.output(),
        device,
    }
}

#[test]
fn graph_resource_plan_binds_and_closes_on_every_device() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "device lane requires real hardware");

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(100 + ordinal), ordinal).unwrap();
        let measurement = ctx.measure().unwrap();
        let before = ctx.memory_info().unwrap().0;
        let snapshot = CapacitySnapshot::measured(&measurement, 64 * 1024 * 1024).unwrap();
        let mut ledger = Ledger::new([snapshot]).unwrap();
        let (graph, values) = build_graph(32);
        let admitted_workload = workload(&graph, ctx.uuid());
        let candidate = lower(&graph, admitted_workload).unwrap();

        let plan = ReservedPlan::admit(candidate, &graph, &mut ledger, &ctx).unwrap();
        assert_eq!(ledger.outstanding().len(), 1);
        let first = plan.tensor(values[0]).unwrap();
        let second = plan.tensor(values[1]).unwrap();
        let third = plan.tensor(values[2]).unwrap();
        assert_eq!(first.value(), values[0]);
        assert_eq!(first.shape(), &[3, 32]);
        assert_eq!(
            first.role(),
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16))
        );
        assert_eq!(first.layout(), TensorLayout::ContiguousRowMajorV1);
        assert_eq!(first.bytes(), 3 * 32 * 2);
        assert_eq!(first.device_uuid(), ctx.uuid());
        assert_eq!(first.allocation_key(), third.allocation_key());
        assert_ne!(first.allocation_key(), second.allocation_key());
        assert_eq!(first.offset(), third.offset());
        assert!(
            plan.validate_execution_request(&graph, admitted_workload)
                .is_err_and(|error| error.kind() == "unsupported_kernel")
        );
        plan.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
        let after = ctx.memory_info().unwrap().0;
        assert_eq!(after, before, "physical allocation reconciled on {ordinal}");

        let (other, _) = build_graph(16);
        let candidate = lower(&other, workload(&other, ctx.uuid())).unwrap();
        assert!(matches!(
            ReservedPlan::admit(candidate, &graph, &mut ledger, &ctx),
            Err(PlanAdmitRefused::Invalid { .. })
        ));
        assert!(ledger.outstanding().is_empty());

        let candidate =
            lower(&graph, workload(&graph, DeviceUuid::from_bytes([0x55; 16]))).unwrap();
        assert!(matches!(
            ReservedPlan::admit(candidate, &graph, &mut ledger, &ctx),
            Err(PlanAdmitRefused::Invalid { .. })
        ));
        assert!(ledger.outstanding().is_empty());
        eprintln!("PASS graph resource plan on {}", ctx.uuid());
    }
}
