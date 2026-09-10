//! Admission and physical binding for a pure graph resource plan.

use moxie_memory::{BufferRequest, PlanRequest, StageSpan};
#[cfg(feature = "driver")]
use moxie_plan::ResourceWorkload;
use moxie_plan::{Graph, PlanCandidate, ValueBinding};
use moxie_types::{DeviceTier, Error, Result, Scope, Tier};

/// Convert a pure candidate into the exact envelope admitted by the ledger.
///
/// Logical activation intervals are diagnostics used to assign physical slots.
/// The one physical arena is charged once for the complete plan lifetime.
pub fn resource_request(candidate: &PlanCandidate) -> Result<PlanRequest> {
    let stages = candidate.stages().iter().map(String::as_str);
    let mut request = PlanRequest::new(
        format!(
            "graph-resource-plan-{}-{}",
            candidate.id().get(),
            candidate.workload().phase.name()
        ),
        stages,
    )?;
    let last = u32::try_from(candidate.stages().len() - 1).map_err(|_| {
        invalid(
            "stages",
            "resource plan stage count does not fit the ledger index",
        )
    })?;
    let scope = Scope::Device(candidate.workload().device);
    request.buffer(BufferRequest::new(
        "activation-arena",
        scope,
        Tier::Device(DeviceTier::Activations),
        candidate.activation_arena_bytes(),
        StageSpan::inclusive(0, last),
    ))?;
    for binding in candidate.bindings() {
        if let ValueBinding::ExternalWeight(weight) = binding {
            request.buffer(BufferRequest::new(
                format!("weight-value-{}", weight.value.0),
                scope,
                Tier::Device(DeviceTier::PackedResidentWeights),
                weight.required_bytes,
                StageSpan::inclusive(0, last),
            ))?;
        }
    }
    Ok(request)
}

pub fn validate_plan_binding(
    candidate: &PlanCandidate,
    graph: &Graph,
    device: moxie_types::DeviceUuid,
) -> Result<()> {
    if !candidate.matches_graph(graph) {
        return Err(invalid(
            "graph",
            format!(
                "candidate {} is bound to {}, not the supplied graph {} or its exact structure",
                candidate.id(),
                candidate.graph_id(),
                graph.id()
            ),
        ));
    }
    if candidate.workload().device != device {
        return Err(invalid(
            "device",
            format!(
                "candidate names {}, rank context owns {device}",
                candidate.workload().device
            ),
        ));
    }
    Ok(())
}

#[cfg(feature = "driver")]
fn validate_request(
    candidate: &PlanCandidate,
    graph: &Graph,
    workload: ResourceWorkload,
    device: moxie_types::DeviceUuid,
) -> Result<()> {
    validate_plan_binding(candidate, graph, device)?;
    if workload != candidate.workload() {
        return Err(invalid(
            "workload",
            "execution request differs from the admitted row/context/device bucket",
        ));
    }
    Ok(())
}

#[cfg(feature = "driver")]
mod driver_binding {
    use moxie_cuda::RankContext;
    use moxie_memory::{AdmitError, AllocateRefused, Ledger, LedgerId, Rejection, Reservation};
    use moxie_plan::{
        ArenaTensor, Graph, PlanCandidate, ResourceWorkload, ValueBinding, ValueId, ValueRole,
    };
    use moxie_types::{Error, TensorLayout};

    use super::{invalid, resource_request, validate_plan_binding, validate_request};
    use crate::{ArenaCloseRefused, DeviceArena, DeviceRange, RangeReleaseRefused};

    /// A read-only metadata view over one plan-owned range. It cannot outlive
    /// the plan, and callers cannot rewrite its checked descriptor.
    ///
    /// ```compile_fail,E0616
    /// fn corrupt(handle: &mut moxie_executor::TensorHandle<'_, '_>) {
    ///     handle.bytes = u64::MAX;
    /// }
    /// ```
    #[derive(Debug)]
    pub struct TensorHandle<'plan, 'ctx> {
        value: ValueId,
        shape: &'plan [u64],
        role: ValueRole,
        layout: TensorLayout,
        offset: u64,
        bytes: u64,
        range: &'plan DeviceRange<'ctx>,
    }

    impl TensorHandle<'_, '_> {
        pub const fn value(&self) -> ValueId {
            self.value
        }

        pub fn shape(&self) -> &[u64] {
            self.shape
        }

        pub const fn role(&self) -> ValueRole {
            self.role
        }

        pub const fn layout(&self) -> TensorLayout {
            self.layout
        }

        pub const fn offset(&self) -> u64 {
            self.offset
        }

        pub const fn bytes(&self) -> u64 {
            self.bytes
        }

        pub fn allocation_key(&self) -> moxie_memory::AllocationKey {
            self.range.key()
        }

        pub fn device_uuid(&self) -> moxie_types::DeviceUuid {
            self.range.device_uuid()
        }

        pub fn owner(&self) -> &str {
            self.range.owner()
        }
    }

    /// Resource retained when admission rollback itself cannot complete.
    #[derive(Debug)]
    pub enum HeldPlanResource<'ctx> {
        Reservation(Reservation),
        Arena {
            arena: Box<DeviceArena<'ctx>>,
            ranges: Vec<(u32, DeviceRange<'ctx>)>,
        },
    }

    /// Admission refusal. The candidate always returns to the caller.
    #[derive(Debug)]
    pub enum PlanAdmitRefused<'ctx> {
        Invalid {
            candidate: Box<PlanCandidate>,
            error: Error,
        },
        Rejected {
            candidate: Box<PlanCandidate>,
            rejection: Box<Rejection>,
        },
        Held {
            candidate: Box<PlanCandidate>,
            resource: HeldPlanResource<'ctx>,
            error: Error,
        },
    }

    /// A graph-bound, admitted activation arena. It owns no weights or kernels.
    #[derive(Debug)]
    #[must_use = "a reserved plan stays charged until close succeeds"]
    pub struct ReservedPlan<'ctx> {
        candidate: PlanCandidate,
        arena: Option<DeviceArena<'ctx>>,
        ranges: Vec<(u32, DeviceRange<'ctx>)>,
        ledger: LedgerId,
    }

    #[derive(Debug)]
    pub struct PlanCloseRefused<'ctx> {
        pub plan: ReservedPlan<'ctx>,
        pub error: Error,
    }

    impl<'ctx> ReservedPlan<'ctx> {
        pub fn admit(
            candidate: PlanCandidate,
            graph: &Graph,
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
        ) -> std::result::Result<Self, PlanAdmitRefused<'ctx>> {
            Self::admit_with(
                candidate,
                graph,
                ledger,
                ctx,
                |arena, bytes, alignment, owner| arena.allocate(bytes, alignment, owner),
            )
        }

        fn admit_with<F>(
            candidate: PlanCandidate,
            graph: &Graph,
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            mut allocate_slot: F,
        ) -> std::result::Result<Self, PlanAdmitRefused<'ctx>>
        where
            F: FnMut(
                &mut DeviceArena<'ctx>,
                u64,
                u64,
                String,
            ) -> std::result::Result<DeviceRange<'ctx>, AllocateRefused>,
        {
            let invalid_candidate = |candidate, error| PlanAdmitRefused::Invalid {
                candidate: Box::new(candidate),
                error,
            };
            if let Err(error) = validate_plan_binding(&candidate, graph, ctx.uuid()) {
                return Err(invalid_candidate(candidate, error));
            }
            let request = match resource_request(&candidate) {
                Ok(request) => request,
                Err(error) => return Err(invalid_candidate(candidate, error)),
            };
            let reservation = match ledger.admit(&request) {
                Ok(reservation) => reservation,
                Err(AdmitError::Invalid(error)) => {
                    return Err(invalid_candidate(candidate, error));
                }
                Err(AdmitError::Rejected(rejection)) => {
                    return Err(PlanAdmitRefused::Rejected {
                        candidate: Box::new(candidate),
                        rejection,
                    });
                }
            };
            let mut arena = match DeviceArena::create(
                ledger,
                reservation,
                ctx,
                moxie_types::DeviceTier::Activations,
                candidate.activation_arena_bytes(),
                format!("plan-{}-activations", candidate.id().get()),
            ) {
                Ok(arena) => arena,
                Err(refused) => {
                    let original = refused.error;
                    return match ledger.release(refused.reservation) {
                        Ok(()) => Err(invalid_candidate(candidate, original)),
                        Err(release) => Err(PlanAdmitRefused::Held {
                            candidate: Box::new(candidate),
                            resource: HeldPlanResource::Reservation(release.reservation),
                            error: release.error,
                        }),
                    };
                }
            };

            let mut ranges = Vec::with_capacity(candidate.slots().len());
            for slot in candidate.slots() {
                match allocate_slot(
                    &mut arena,
                    slot.bytes,
                    slot.alignment,
                    format!("plan-{}-slot-{}", candidate.id().get(), slot.id),
                ) {
                    Ok(range) => {
                        if range.offset() != slot.offset {
                            let error = invalid(
                                "slot",
                                format!(
                                    "planned slot {} at {}, arena returned {}",
                                    slot.id,
                                    slot.offset,
                                    range.offset()
                                ),
                            );
                            ranges.push((slot.id, range));
                            return Err(PlanAdmitRefused::Held {
                                candidate: Box::new(candidate),
                                resource: HeldPlanResource::Arena {
                                    arena: Box::new(arena),
                                    ranges,
                                },
                                error,
                            });
                        }
                        ranges.push((slot.id, range));
                    }
                    Err(AllocateRefused { error, .. }) => {
                        return match unwind(arena, ranges, ledger) {
                            Ok(()) => Err(invalid_candidate(candidate, error)),
                            Err(refused) => Err(PlanAdmitRefused::Held {
                                candidate: Box::new(candidate),
                                resource: HeldPlanResource::Arena {
                                    arena: Box::new(refused.arena),
                                    ranges: refused.ranges,
                                },
                                error: refused.error,
                            }),
                        };
                    }
                }
            }
            Ok(Self {
                candidate,
                arena: Some(arena),
                ranges,
                ledger: ledger.id(),
            })
        }

        pub fn candidate(&self) -> &PlanCandidate {
            &self.candidate
        }

        pub fn tensor(&self, value: ValueId) -> Option<TensorHandle<'_, 'ctx>> {
            let ValueBinding::ArenaTensor(tensor) = self.candidate.binding(value)? else {
                return None;
            };
            let range = self
                .ranges
                .iter()
                .find(|(slot, _)| *slot == tensor.slot)
                .map(|(_, range)| range)?;
            Some(handle(tensor, range))
        }

        /// Validate the exact admitted bucket, then fail closed until kernels
        /// are bound by the next task.
        pub fn validate_execution_request(
            &self,
            graph: &Graph,
            workload: ResourceWorkload,
        ) -> moxie_types::Result<()> {
            let device = self.candidate.workload().device;
            validate_request(&self.candidate, graph, workload, device)?;
            let operation = graph
                .nodes()
                .first()
                .map(|node| node.params.op().name())
                .unwrap_or("graph");
            Err(Error::UnsupportedKernel {
                operation,
                detail: "resource plan has no registry-backed semantic kernel selections".into(),
            })
        }

        #[allow(clippy::result_large_err)]
        pub fn close(
            mut self,
            ledger: &mut Ledger,
        ) -> std::result::Result<(), PlanCloseRefused<'ctx>> {
            if ledger.id() != self.ledger {
                return Err(PlanCloseRefused {
                    plan: self,
                    error: invalid("reservation", "reserved plan belongs to another ledger"),
                });
            }
            while let Some((slot, range)) = self.ranges.pop() {
                let arena = self.arena.as_mut().expect("open plan has an arena");
                if let Err(RangeReleaseRefused { range, error }) = arena.release(range) {
                    self.ranges.push((slot, range));
                    return Err(PlanCloseRefused { plan: self, error });
                }
            }
            let arena = self.arena.take().expect("open plan has an arena");
            match arena.close(ledger) {
                Ok(()) => Ok(()),
                Err(ArenaCloseRefused { arena, error }) => {
                    self.arena = Some(arena);
                    Err(PlanCloseRefused { plan: self, error })
                }
            }
        }
    }

    fn handle<'plan, 'ctx>(
        tensor: &'plan ArenaTensor,
        range: &'plan DeviceRange<'ctx>,
    ) -> TensorHandle<'plan, 'ctx> {
        TensorHandle {
            value: tensor.value,
            shape: &tensor.shape,
            role: tensor.role,
            layout: tensor.layout,
            offset: tensor.offset,
            bytes: tensor.bytes,
            range,
        }
    }

    struct UnwindRefused<'ctx> {
        arena: DeviceArena<'ctx>,
        ranges: Vec<(u32, DeviceRange<'ctx>)>,
        error: Error,
    }

    fn unwind<'ctx>(
        mut arena: DeviceArena<'ctx>,
        mut ranges: Vec<(u32, DeviceRange<'ctx>)>,
        ledger: &mut Ledger,
    ) -> std::result::Result<(), Box<UnwindRefused<'ctx>>> {
        while let Some((slot, range)) = ranges.pop() {
            if let Err(RangeReleaseRefused { range, error }) = arena.release(range) {
                ranges.push((slot, range));
                return Err(Box::new(UnwindRefused {
                    arena,
                    ranges,
                    error,
                }));
            }
        }
        match arena.close(ledger) {
            Ok(()) => Ok(()),
            Err(ArenaCloseRefused { arena, error }) => Err(Box::new(UnwindRefused {
                arena,
                ranges,
                error,
            })),
        }
    }

    #[cfg(test)]
    mod tests {
        use moxie_cuda::RankContext;
        use moxie_graph::{
            Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry,
            TensorSpec, ValueRole,
        };
        use moxie_memory::{CapacitySnapshot, Ledger};
        use moxie_plan::{Phase, ResourceWorkload, lower};
        use moxie_types::{ActivationPrecision, Dim, Precision, RankId, SymbolId, WeightPrecision};

        use super::{PlanAdmitRefused, ReservedPlan};

        const ROWS: SymbolId = SymbolId(211);
        const ORACLE: OracleId = OracleId("integrated-slot-fault");

        fn graph() -> Graph {
            let mut registry = OracleRegistry::new();
            registry
                .register(
                    Op::Linear,
                    ORACLE,
                    OracleEvidence {
                        implementation: "executor::plan::driver_binding::tests",
                        test_module: "executor::plan::driver_binding::tests",
                    },
                )
                .unwrap();
            let mut builder = GraphBuilder::new(ORACLE, ROWS);
            let input = builder.input(
                "x",
                TensorSpec::new(
                    ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    vec![Dim::symbol(ROWS), Dim::constant(8)],
                ),
            );
            let mut previous = input;
            for index in 0..3 {
                let weight = builder
                    .weight(
                        &format!("w{index}"),
                        TensorSpec::new(
                            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                            vec![Dim::constant(8), Dim::constant(8)],
                        ),
                    )
                    .unwrap();
                previous = builder
                    .node(
                        OpParams::Linear {
                            in_features: 8,
                            out_features: 8,
                            bias: false,
                        },
                        &[previous, weight],
                    )
                    .unwrap();
            }
            builder.finish(previous, &registry).unwrap()
        }

        #[test]
        fn integrated_slot_failure_returns_candidate_and_unwinds_the_arena() {
            let _serial = crate::DRIVER_TEST_LOCK.lock().unwrap();
            let count = moxie_cuda::device_count().expect("device lane requires the CUDA driver");
            assert!(count > 0, "device lane requires real hardware");
            for ordinal in 0..count {
                let ctx = RankContext::acquire(RankId(200 + ordinal), ordinal).unwrap();
                let measurement = ctx.measure().unwrap();
                let before = ctx.memory_info().unwrap().0;
                let snapshot = CapacitySnapshot::measured(&measurement, 64 * 1024 * 1024).unwrap();
                let mut ledger = Ledger::new([snapshot]).unwrap();
                let graph = graph();
                let workload = ResourceWorkload {
                    phase: Phase::Prefill,
                    rows: 2,
                    visible_tokens: 32_768,
                    branch_rows: 2,
                    output: graph.output(),
                    device: ctx.uuid(),
                };
                let candidate = lower(&graph, workload).unwrap();
                let candidate_id = candidate.id();
                let unavailable_bytes = candidate.activation_arena_bytes() + 256;
                let mut slot = 0;
                let refusal = ReservedPlan::admit_with(
                    candidate,
                    &graph,
                    &mut ledger,
                    &ctx,
                    |arena, bytes, alignment, owner| {
                        slot += 1;
                        if slot == 2 {
                            arena.allocate(unavailable_bytes, alignment, owner)
                        } else {
                            arena.allocate(bytes, alignment, owner)
                        }
                    },
                )
                .unwrap_err();
                let PlanAdmitRefused::Invalid { candidate, error } = refusal else {
                    panic!("successful cleanup must return the unchanged candidate");
                };
                assert_eq!(candidate.id(), candidate_id);
                assert!(candidate.matches_graph(&graph));
                assert_eq!(error.kind(), "capacity_exceeded");
                assert_eq!(slot, 2);
                assert!(ledger.outstanding().is_empty());
                assert_eq!(ctx.memory_info().unwrap().0, before);
            }
        }
    }
}

#[cfg(feature = "driver")]
pub use driver_binding::{
    HeldPlanResource, PlanAdmitRefused, PlanCloseRefused, ReservedPlan, TensorHandle,
};

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use moxie_graph::{
        Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
        ValueRole,
    };
    use moxie_memory::{CapacitySnapshot, Ledger, Scaling};
    use moxie_plan::{Phase, ResourceWorkload, lower};
    use moxie_types::{
        ActivationPrecision, DeviceTier, DeviceUuid, Dim, Precision, Scope, SymbolId, Tier,
        WeightPrecision,
    };

    use super::*;

    const ROWS: SymbolId = SymbolId(18);
    const ORACLE: OracleId = OracleId("executor-plan-test");

    fn uuid(n: u8) -> DeviceUuid {
        DeviceUuid::from_bytes([n; 16])
    }

    fn graph(width: u64) -> Graph {
        let mut registry = OracleRegistry::new();
        registry
            .register(
                Op::Linear,
                ORACLE,
                OracleEvidence {
                    implementation: "executor::plan::tests",
                    test_module: "executor::plan::tests",
                },
            )
            .unwrap();
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let input = builder.input(
            "x",
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                vec![Dim::symbol(ROWS), Dim::constant(width)],
            ),
        );
        let weight = builder
            .weight(
                "w",
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                    vec![Dim::constant(width), Dim::constant(width)],
                ),
            )
            .unwrap();
        let output = builder
            .node(
                OpParams::Linear {
                    in_features: width,
                    out_features: width,
                    bias: false,
                },
                &[input, weight],
            )
            .unwrap();
        builder.finish(output, &registry).unwrap()
    }

    fn workload(graph: &Graph, device: DeviceUuid) -> ResourceWorkload {
        ResourceWorkload {
            phase: Phase::Prefill,
            rows: 3,
            visible_tokens: 17,
            branch_rows: 3,
            output: graph.output(),
            device,
        }
    }

    #[test]
    fn request_charges_one_physical_arena_and_exact_weights() {
        let graph = graph(8);
        let device = uuid(1);
        let candidate = lower(&graph, workload(&graph, device)).unwrap();
        let request = resource_request(&candidate).unwrap();
        assert_eq!(request.stages(), &["node-0-linear", "terminal-output"]);
        assert_eq!(request.buffers().len(), 2);
        let activation = &request.buffers()[0];
        assert_eq!(activation.bytes, 256);
        assert_eq!(activation.tier, Tier::Device(DeviceTier::Activations));
        assert_eq!(activation.live, StageSpan::inclusive(0, 1));
        let weight = &request.buffers()[1];
        assert_eq!(weight.bytes, 8 * 8 * 2);
        assert_eq!(weight.tier, Tier::Device(DeviceTier::PackedResidentWeights));

        let snapshot = CapacitySnapshot::new(Scope::Device(device), 4096, 0).unwrap();
        let mut ledger = Ledger::new([snapshot]).unwrap();
        let reservation = ledger.admit(&request).unwrap();
        assert_eq!(
            ledger.committed(
                Scope::Device(device),
                Tier::Device(DeviceTier::PackedResidentWeights)
            ),
            8 * 8 * 2
        );
        ledger.release(reservation).unwrap();
        assert!(ledger.outstanding().is_empty());
    }

    #[test]
    fn ledger_rejection_is_atomic_for_the_derived_request() {
        let device = uuid(2);
        let graph = graph(32);
        let candidate = lower(&graph, workload(&graph, device)).unwrap();
        let snapshot = CapacitySnapshot::new(Scope::Device(device), 1024, 1).unwrap();
        let mut ledger = Ledger::new([snapshot]).unwrap();
        let request = resource_request(&candidate).unwrap();
        assert!(ledger.admit(&request).is_err());
        assert!(ledger.outstanding().is_empty());
        assert_eq!(ledger.scope_committed(Scope::Device(device)), 0);
    }

    #[test]
    fn scaling_is_never_invented_for_fixed_graph_bytes() {
        let graph = graph(4);
        let candidate = lower(&graph, workload(&graph, uuid(3))).unwrap();
        let request = resource_request(&candidate).unwrap();
        assert!(
            request
                .buffers()
                .iter()
                .all(|buffer| buffer.scales_with != Some(Scaling::Context))
        );
    }

    #[test]
    fn graph_and_uuid_mismatch_are_refused_before_admission() {
        let expected = graph(4);
        let candidate = lower(&expected, workload(&expected, uuid(4))).unwrap();
        assert!(validate_plan_binding(&candidate, &expected, uuid(4)).is_ok());
        assert!(validate_plan_binding(&candidate, &expected, uuid(5)).is_err());
        assert!(validate_plan_binding(&candidate, &graph(8), uuid(4)).is_err());
    }
}
