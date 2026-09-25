//! Binding and event-retained execution of the selected BF16 layer chain.

#![cfg(feature = "driver")]

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "cublas")]
use moxie_cuda::Blas;
use moxie_cuda::{
    CapturedGraph, Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage,
};
#[cfg(feature = "paged-attention-binding")]
use moxie_graph::NodeId;
use moxie_memory::{
    AdmitError, BufferRequest, Ledger, LedgerId, PlanRequest, Reservation, StageSpan,
};
use moxie_plan::{
    Graph, OpParams, SelectedNode, SelectedPlanCandidate, StorageRegion, ValueId, ValueRole,
    WeightFormat,
};
use moxie_types::{
    DeviceCapability, DeviceTier, Error, HostTier, KernelCatalogue, Result, Scope,
    SemanticKernelOp, TensorLayout, Tier,
};

use crate::{
    ArenaCloseRefused, Completion, DeviceArena, DeviceRange, OperationLease,
    OperationRetireRefused, RangeReleaseRefused,
};

#[derive(Debug)]
pub struct OwnedBinding {
    pub value: ValueId,
    pub role: ValueRole,
    pub shape: Vec<u64>,
    pub layout: TensorLayout,
    pub device: moxie_types::DeviceUuid,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub enum SelectedAdmitRefused<'ctx> {
    Invalid {
        candidate: Box<SelectedPlanCandidate>,
        error: Error,
    },
    Rejected {
        candidate: Box<SelectedPlanCandidate>,
        rejection: moxie_memory::Rejection,
    },
    Held {
        candidate: Box<SelectedPlanCandidate>,
        resource: SelectedHeldResource<'ctx>,
        error: Error,
    },
}

impl SelectedAdmitRefused<'_> {
    pub(crate) fn close(self, ledger: &mut Ledger) -> std::result::Result<(), Self> {
        match self {
            Self::Invalid { .. } | Self::Rejected { .. } => Ok(()),
            Self::Held {
                candidate,
                resource: SelectedHeldResource::Reservation(reservation),
                ..
            } => match ledger.release(reservation) {
                Ok(()) => Ok(()),
                Err(refused) => Err(Self::Held {
                    candidate,
                    resource: SelectedHeldResource::Reservation(refused.reservation),
                    error: refused.error,
                }),
            },
            Self::Held {
                candidate,
                resource: SelectedHeldResource::Arena { arena, ranges },
                ..
            } => match unwind_selected(*arena, ranges, ledger) {
                Ok(()) => Ok(()),
                Err(refused) => Err(Self::Held {
                    candidate,
                    resource: SelectedHeldResource::Arena {
                        arena: Box::new(refused.arena),
                        ranges: refused.ranges,
                    },
                    error: refused.error,
                }),
            },
        }
    }
}

#[derive(Debug)]
pub enum SelectedHeldResource<'ctx> {
    Reservation(Reservation),
    Arena {
        arena: Box<DeviceArena<'ctx>>,
        ranges: BTreeMap<(StorageRegion, u32), DeviceRange<'ctx>>,
    },
}

#[derive(Debug)]
pub struct SelectedCloseRefused<'ctx> {
    pub plan: SelectedReservedPlan<'ctx>,
    pub error: Error,
}

#[derive(Debug)]
pub struct SelectedLaunchRefused<'ctx> {
    pub plan: Option<SelectedReservedPlan<'ctx>>,
    pub bindings: Vec<OwnedBinding>,
    pub held: Option<OperationLease<SelectedCompletion<'ctx>, ChainOperation<'ctx>>>,
    pub error: Error,
}

/// The selected chain's event plus the identity needed to classify failures
/// before the shared lifecycle persists them.
#[derive(Debug)]
pub struct SelectedCompletion<'ctx> {
    pub(crate) event: Event<'ctx>,
    pub(crate) device_ordinal: u32,
    pub(crate) attribution: String,
}

impl Completion for SelectedCompletion<'_> {
    fn query_complete(&self) -> Result<bool> {
        self.event
            .is_complete()
            .map_err(|error| attribute_error(error, self.device_ordinal, self.attribution.clone()))
    }

    fn synchronize(&self) -> Result<()> {
        self.event
            .synchronize()
            .map_err(|error| attribute_error(error, self.device_ordinal, self.attribution.clone()))
    }

    fn describe(&self) -> String {
        format!("{} ({})", self.event.device_uuid(), self.attribution)
    }
}

impl<'ctx> SelectedCompletion<'ctx> {
    #[cfg(feature = "paged-attention-binding")]
    pub(crate) fn new(event: Event<'ctx>, device_ordinal: u32, attribution: String) -> Self {
        Self {
            event,
            device_ordinal,
            attribution,
        }
    }
}

#[derive(Debug)]
pub struct ChainResult<'ctx> {
    pub plan: SelectedReservedPlan<'ctx>,
    pub output: Vec<u8>,
    pub returned_inputs: Vec<OwnedBinding>,
    pub launch_order: Vec<String>,
}

#[derive(Debug)]
pub struct SelectedReservedPlan<'ctx> {
    candidate: SelectedPlanCandidate,
    /// Ordinary Drop quarantines the arena; `close` tears it down explicitly.
    arena: Option<DeviceArena<'ctx>>,
    /// Ordinary Drop forgets these ranges with their arena allocation.
    ranges: BTreeMap<(StorageRegion, u32), DeviceRange<'ctx>>,
    bound_weights: BTreeMap<ValueId, OwnedBinding>,
    resident_weights: BTreeMap<ValueId, u64>,
    /// Captured segments must be dropped before the module whose functions
    /// they reference.
    /// Ordinary Drop forgets graphs; `close` clears them before module teardown.
    pub(crate) captured: Vec<CapturedGraph<'ctx>>,
    pub(crate) capture_enabled: bool,
    graph_reservation: Option<Reservation>,
    graph_pool_bytes: u64,
    /// The dense kernel module, loaded by the first dense step and dropped
    /// with the plan, so it is never unloaded while a step's kernels may still
    /// run. Ordinary Drop forgets it; `close` explicitly drops it after graphs.
    pub(crate) package: Option<ResolvedModule<'ctx>>,
    /// CUDA symbols only; backend pseudo-symbols are omitted.
    #[cfg(feature = "paged-attention-binding")]
    pub(crate) module_symbols: Vec<String>,
    /// First module-symbol index for each selected node.
    #[cfg(feature = "paged-attention-binding")]
    pub(crate) symbol_indices: BTreeMap<NodeId, usize>,
    /// Ordinary Drop forgets the handle; close destroys it between graphs and module.
    #[cfg(feature = "cublas")]
    pub(crate) blas: Option<Blas<'ctx>>,
    ledger: LedgerId,
}

#[derive(Debug)]
pub struct ChainOperation<'ctx> {
    plan: Option<SelectedReservedPlan<'ctx>>,
    sources: Vec<OwnedBinding>,
    package: ResolvedModule<'ctx>,
    launch_order: Vec<String>,
    device_ordinal: u32,
}

impl<'ctx> ChainOperation<'ctx> {
    pub fn retained_source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn has_plan(&self) -> bool {
        self.plan.is_some()
    }

    fn settle_sources(&mut self) -> (SelectedReservedPlan<'ctx>, Vec<OwnedBinding>) {
        let mut plan = self.plan.take().expect("chain operation retains its plan");
        let mut returned_inputs = Vec::new();
        for binding in self.sources.drain(..) {
            if matches!(binding.role, ValueRole::Weight(_)) {
                plan.bound_weights.insert(binding.value, binding);
            } else {
                returned_inputs.push(binding);
            }
        }
        (plan, returned_inputs)
    }

    /// Recover a completed operation retired by the shared turn sweep. The
    /// sweep has already observed its event, so uploaded weights become the
    /// plan's immutable bindings exactly as on normal completion; only the
    /// caller-owned per-step inputs are handed back.
    pub fn into_parts(mut self) -> (SelectedReservedPlan<'ctx>, Vec<OwnedBinding>) {
        self.settle_sources()
    }
}

impl<'ctx> SelectedReservedPlan<'ctx> {
    /// Per-node graph memory bound derived from `docs/evidence/graph-memory.md`.
    pub const CAPTURED_KERNEL_BOUND_BYTES: u64 = 8_192;
    /// Per-graph graph memory bound derived from `docs/evidence/graph-memory.md`.
    pub const CAPTURED_GRAPH_BOUND_BYTES: u64 = 131_072;

    pub fn admit(
        candidate: SelectedPlanCandidate,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
    ) -> std::result::Result<Self, SelectedAdmitRefused<'ctx>> {
        Self::admit_inner(
            candidate,
            graph,
            capability,
            catalogue,
            ledger,
            ctx,
            BTreeMap::new(),
            false,
        )
    }

    pub(crate) fn admit_with_resident_weights(
        candidate: SelectedPlanCandidate,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
        resident_weights: BTreeMap<ValueId, u64>,
    ) -> std::result::Result<Self, SelectedAdmitRefused<'ctx>> {
        Self::admit_inner(
            candidate,
            graph,
            capability,
            catalogue,
            ledger,
            ctx,
            resident_weights,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_inner(
        candidate: SelectedPlanCandidate,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
        resident_weights: BTreeMap<ValueId, u64>,
        use_resident_weights: bool,
    ) -> std::result::Result<Self, SelectedAdmitRefused<'ctx>> {
        let fail = |candidate, error| SelectedAdmitRefused::Invalid {
            candidate: Box::new(candidate),
            error,
        };
        if use_resident_weights
            && (resident_weights.len() != graph.weights().len()
                || graph
                    .weights()
                    .iter()
                    .any(|value| !resident_weights.contains_key(value)))
        {
            return Err(fail(
                candidate,
                invalid("weights", "resident addresses must name every graph weight"),
            ));
        }
        if !candidate.matches(graph, capability, catalogue) || ctx.uuid() != capability.uuid {
            return Err(fail(
                candidate,
                invalid("plan", "graph/catalogue/capability/UUID binding changed"),
            ));
        }
        let arena_capacity = if use_resident_weights {
            match candidate
                .combined_arena_bytes()
                .checked_sub(candidate.weight_region_bytes())
            {
                Some(bytes) => bytes,
                None => {
                    return Err(fail(
                        candidate,
                        invalid("range", "resident weight region exceeds arena size"),
                    ));
                }
            }
        } else {
            candidate.combined_arena_bytes()
        };
        let mut regions = match moxie_memory::fallible::with_capacity(3) {
            Ok(regions) => regions,
            Err(error) => return Err(fail(candidate, error)),
        };
        if !use_resident_weights && candidate.weight_region_bytes() != 0 {
            regions.push((
                DeviceTier::PackedResidentWeights,
                candidate.weight_region_bytes(),
            ));
        }
        for region in [
            (DeviceTier::Activations, candidate.activation_region_bytes()),
            (
                DeviceTier::KernelWorkspace,
                candidate.workspace_region_bytes(),
            ),
        ] {
            if region.1 != 0 {
                regions.push(region);
            }
        }
        let request =
            match selected_resource_request_with_weights(&candidate, !use_resident_weights) {
                Ok(request) => request,
                Err(error) => return Err(fail(candidate, error)),
            };
        let reservation = match ledger.admit(&request) {
            Ok(value) => value,
            Err(AdmitError::Invalid(error)) => return Err(fail(candidate, error)),
            Err(AdmitError::Rejected(rejection)) => {
                return Err(SelectedAdmitRefused::Rejected {
                    candidate: Box::new(candidate),
                    rejection,
                });
            }
        };
        let mut arena = match DeviceArena::create_partitioned(
            ledger,
            reservation,
            ctx,
            &regions,
            arena_capacity,
            format!("selected-plan-{}", candidate.base().id().get()),
        ) {
            Ok(value) => value,
            Err(refused) => {
                let original = refused.error;
                return match ledger.release(refused.reservation) {
                    Ok(()) => Err(fail(candidate, original)),
                    Err(release) => Err(SelectedAdmitRefused::Held {
                        candidate: Box::new(candidate),
                        resource: SelectedHeldResource::Reservation(release.reservation),
                        error: release.error,
                    }),
                };
            }
        };
        let mut specs: BTreeMap<(StorageRegion, u32), (u64, u64)> = BTreeMap::new();
        for value in candidate.values() {
            if use_resident_weights && value.region == StorageRegion::Weights {
                continue;
            }
            let offset = if use_resident_weights {
                match value.offset.checked_sub(candidate.weight_region_bytes()) {
                    Some(offset) => offset,
                    None => {
                        let error = invalid("range", "resident weight region exceeds value offset");
                        return match unwind_selected(arena, BTreeMap::new(), ledger) {
                            Ok(()) => Err(fail(candidate, error)),
                            Err(refused) => Err(SelectedAdmitRefused::Held {
                                candidate: Box::new(candidate),
                                resource: SelectedHeldResource::Arena {
                                    arena: Box::new(refused.arena),
                                    ranges: refused.ranges,
                                },
                                error: refused.error,
                            }),
                        };
                    }
                }
            } else {
                value.offset
            };
            specs
                .entry((value.region, value.slot))
                .or_insert((offset, value.physical_bytes));
        }
        if candidate.workspace().physical_bytes != 0 {
            let offset = if use_resident_weights {
                match candidate
                    .workspace()
                    .offset
                    .checked_sub(candidate.weight_region_bytes())
                {
                    Some(offset) => offset,
                    None => {
                        let error =
                            invalid("range", "resident weight region exceeds workspace offset");
                        return match unwind_selected(arena, BTreeMap::new(), ledger) {
                            Ok(()) => Err(fail(candidate, error)),
                            Err(refused) => Err(SelectedAdmitRefused::Held {
                                candidate: Box::new(candidate),
                                resource: SelectedHeldResource::Arena {
                                    arena: Box::new(refused.arena),
                                    ranges: refused.ranges,
                                },
                                error: refused.error,
                            }),
                        };
                    }
                }
            } else {
                candidate.workspace().offset
            };
            specs.insert(
                (StorageRegion::Workspace, 0),
                (offset, candidate.workspace().physical_bytes),
            );
        }
        let mut ranges = BTreeMap::new();
        for (key, (offset, bytes)) in specs {
            match arena.allocate(bytes, 256, format!("{:?}-{}", key.0, key.1)) {
                Ok(range) if range.offset() == offset => {
                    ranges.insert(key, range);
                }
                Ok(range) => {
                    ranges.insert(key, range);
                    let error = invalid("range", "physical arena offset differs from pure plan");
                    return match unwind_selected(arena, ranges, ledger) {
                        Ok(()) => Err(fail(candidate, error)),
                        Err(refused) => Err(SelectedAdmitRefused::Held {
                            candidate: Box::new(candidate),
                            resource: SelectedHeldResource::Arena {
                                arena: Box::new(refused.arena),
                                ranges: refused.ranges,
                            },
                            error: refused.error,
                        }),
                    };
                }
                Err(error) => {
                    let original = error.error;
                    return match unwind_selected(arena, ranges, ledger) {
                        Ok(()) => Err(fail(candidate, original)),
                        Err(refused) => Err(SelectedAdmitRefused::Held {
                            candidate: Box::new(candidate),
                            resource: SelectedHeldResource::Arena {
                                arena: Box::new(refused.arena),
                                ranges: refused.ranges,
                            },
                            error: refused.error,
                        }),
                    };
                }
            }
        }
        #[cfg(feature = "paged-attention-binding")]
        let (module_symbols, symbol_indices) = selected_module_symbols(&candidate);
        Ok(Self {
            candidate,
            arena: Some(arena),
            ranges,
            bound_weights: BTreeMap::new(),
            resident_weights,
            captured: Vec::new(),
            capture_enabled: false,
            graph_reservation: None,
            graph_pool_bytes: 0,
            package: None,
            #[cfg(feature = "paged-attention-binding")]
            module_symbols,
            #[cfg(feature = "paged-attention-binding")]
            symbol_indices,
            #[cfg(feature = "cublas")]
            blas: None,
            ledger: ledger.id(),
        })
    }

    pub fn candidate(&self) -> &SelectedPlanCandidate {
        &self.candidate
    }
    pub fn bound_weight_count(&self) -> usize {
        self.bound_weights.len()
    }

    pub(crate) fn value_address(&self, value: ValueId) -> Result<u64> {
        if let Some(address) = self.resident_weights.get(&value) {
            return Ok(*address);
        }
        self.range_for_value(value)?.device_address()
    }

    pub fn set_segment_capture(&mut self, enabled: bool, ledger: &mut Ledger) -> Result<()> {
        if enabled {
            if !self.candidate.is_dense()
                || !self.candidate.host_expert_joins().is_empty()
                || !self.candidate.linear_orders().is_empty()
                || !self.candidate.combine_orders().is_empty()
                || !self.candidate.expert_ownership().is_empty()
            {
                return Err(invalid(
                    "capture",
                    "segment capture requires a dense plan without host joins, reduction orders, or expert ownership",
                ));
            }
            if ledger.id() != self.ledger {
                return Err(invalid("ledger", "selected plan belongs to another ledger"));
            }
            if self.capture_enabled {
                return Ok(());
            }
            let (kernels, segments) = captured_graph_counts(&self.candidate)?;
            let bytes = kernels
                .checked_mul(Self::CAPTURED_KERNEL_BOUND_BYTES)
                .and_then(|nodes| {
                    segments
                        .checked_mul(Self::CAPTURED_GRAPH_BOUND_BYTES)
                        .and_then(|graphs| nodes.checked_add(graphs))
                })
                .ok_or_else(|| invalid("capture", "graph memory bound overflowed"))?;
            let mut request = PlanRequest::new("selected-plan-graph-pools", ["capture"])?;
            request.buffer(BufferRequest::new(
                "captured-graph-pools",
                Scope::Device(self.candidate.workload().device),
                Tier::Device(DeviceTier::GraphPools),
                bytes,
                StageSpan::at(0),
            ))?;
            let reservation = ledger.admit(&request)?;
            self.graph_reservation = Some(reservation);
            self.graph_pool_bytes = bytes;
            self.capture_enabled = true;
        } else {
            if ledger.id() != self.ledger {
                return Err(invalid("ledger", "selected plan belongs to another ledger"));
            }
            self.captured.clear();
            if let Some(reservation) = self.graph_reservation.take() {
                match ledger.release(reservation) {
                    Ok(()) => self.graph_pool_bytes = 0,
                    Err(refused) => {
                        self.graph_reservation = Some(refused.reservation);
                        return Err(refused.error);
                    }
                }
            } else {
                self.graph_pool_bytes = 0;
            }
            self.capture_enabled = false;
        }
        Ok(())
    }

    pub fn captured_segments(&self) -> usize {
        self.captured.len()
    }

    pub fn graph_pool_bytes(&self) -> u64 {
        self.graph_pool_bytes
    }

    #[cfg(feature = "paged-attention-binding")]
    pub(crate) fn settle_sources(&mut self, sources: Vec<OwnedBinding>) -> Vec<OwnedBinding> {
        let mut returned_inputs = Vec::new();
        for binding in sources {
            if matches!(binding.role, ValueRole::Weight(_)) {
                self.bound_weights.insert(binding.value, binding);
            } else {
                returned_inputs.push(binding);
            }
        }
        returned_inputs
    }

    #[allow(clippy::result_large_err)]
    pub fn launch(
        self,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        ctx: &'ctx RankContext,
        stream: &Stream<'ctx>,
        bindings: Vec<OwnedBinding>,
    ) -> std::result::Result<
        OperationLease<SelectedCompletion<'ctx>, ChainOperation<'ctx>>,
        SelectedLaunchRefused<'ctx>,
    > {
        let reject = |plan, bindings, error| SelectedLaunchRefused {
            plan: Some(plan),
            bindings,
            held: None,
            error,
        };
        if !self.candidate.matches(graph, capability, catalogue)
            || ctx.uuid() != capability.uuid
            || stream.device_uuid() != capability.uuid
        {
            return Err(reject(
                self,
                bindings,
                invalid(
                    "execution",
                    "admitted graph/catalogue/capability/stream changed",
                ),
            ));
        }
        if self.candidate.is_paged_attention() {
            return Err(reject(
                self,
                bindings,
                invalid("execution", "paged attention uses execute_paged_attention"),
            ));
        }
        if catalogue.digest() != moxie_kernels::bf16_chain_catalogue().digest() {
            return Err(reject(
                self,
                bindings,
                Error::UnsupportedKernel {
                    operation: "graph",
                    detail: "executor package does not match the selected catalogue".into(),
                },
            ));
        }
        if let Err(error) = validate_bindings(&self, graph, &bindings) {
            return Err(reject(self, bindings, error));
        }
        let symbols: Vec<String> = self
            .candidate
            .nodes()
            .iter()
            .flat_map(|node| {
                node.descriptor
                    .symbols
                    .iter()
                    .map(|symbol| symbol.0.clone())
            })
            .collect();
        // SAFETY: the bytes are embedded nvcc output from this build, and the
        // candidate's image SHA is checked through the built-in catalogue.
        let trusted =
            match unsafe { TrustedImage::from_build_output(moxie_kernels::BF16_CHAIN_FATBIN) } {
                Ok(value) => value,
                Err(error) => return Err(reject(self, bindings, error)),
            };
        let package = match Module::load(ctx, ModuleImage::Binary(trusted))
            .and_then(|module| module.resolve_all(&symbols))
        {
            Ok(value) => value,
            Err(error) => return Err(reject(self, bindings, error)),
        };
        let operation = ChainOperation {
            plan: Some(self),
            sources: bindings,
            package,
            launch_order: Vec::new(),
            device_ordinal: ctx.ordinal(),
        };
        let mut lease = OperationLease::new("selected BF16 device chain", operation)
            .expect("static label is nonempty");
        if let Err(error) = enqueue(&mut lease, graph, stream, ctx.ordinal()) {
            lease.mark_lost(
                ctx.ordinal(),
                format!("selected chain submission failed: {error}"),
            );
            return Err(SelectedLaunchRefused {
                plan: None,
                bindings: Vec::new(),
                held: Some(lease),
                error,
            });
        }
        let event = match Event::new(ctx) {
            Ok(event) => event,
            Err(error) => {
                let error = attribute_chain_error(
                    error,
                    ctx.ordinal(),
                    lease
                        .resource()
                        .plan
                        .as_ref()
                        .expect("operation retains plan")
                        .candidate(),
                    "completion event creation",
                );
                lease.mark_lost(
                    ctx.ordinal(),
                    format!("completion event creation failed: {error}"),
                );
                return Err(SelectedLaunchRefused {
                    plan: None,
                    bindings: Vec::new(),
                    held: Some(lease),
                    error,
                });
            }
        };
        if let Err(error) = event.record(stream) {
            let error = attribute_chain_error(
                error,
                ctx.ordinal(),
                lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("operation retains plan")
                    .candidate(),
                "completion event record",
            );
            lease.mark_lost(
                ctx.ordinal(),
                format!("completion event record failed: {error}"),
            );
            return Err(SelectedLaunchRefused {
                plan: None,
                bindings: Vec::new(),
                held: Some(lease),
                error,
            });
        }
        let attribution = chain_attribution(
            lease
                .resource()
                .plan
                .as_ref()
                .expect("operation retains plan")
                .candidate(),
            "completion event",
        );
        lease
            .submit_tracked(SelectedCompletion {
                event,
                device_ordinal: ctx.ordinal(),
                attribution,
            })
            .expect("new operation is live");
        Ok(lease)
    }

    #[allow(clippy::result_large_err)]
    pub fn close(
        mut self,
        ledger: &mut Ledger,
    ) -> std::result::Result<(), SelectedCloseRefused<'ctx>> {
        if ledger.id() != self.ledger {
            return Err(SelectedCloseRefused {
                plan: self,
                error: invalid("ledger", "selected plan belongs to another ledger"),
            });
        }
        self.captured.clear();
        if let Some(reservation) = self.graph_reservation.take() {
            if let Err(refused) = ledger.release(reservation) {
                self.graph_reservation = Some(refused.reservation);
                return Err(SelectedCloseRefused {
                    plan: self,
                    error: refused.error,
                });
            }
            self.graph_pool_bytes = 0;
            self.capture_enabled = false;
        }
        #[cfg(feature = "cublas")]
        if let Some(blas) = self.blas.take() {
            // SAFETY: submitted dense work returns the plan only after its
            // completion event is observed; lost leases withhold the plan.
            if let Err(error) = unsafe { blas.destroy() } {
                return Err(SelectedCloseRefused { plan: self, error });
            }
        }
        drop(self.package.take());
        while let Some((key, range)) = self.ranges.pop_last() {
            let arena = self.arena.as_mut().expect("open plan has arena");
            if let Err(RangeReleaseRefused { range, error }) = arena.release(range) {
                self.ranges.insert(key, range);
                return Err(SelectedCloseRefused { plan: self, error });
            }
        }
        let arena = self.arena.take().expect("open plan has arena");
        match arena.close(ledger) {
            Ok(()) => Ok(()),
            Err(ArenaCloseRefused { arena, error }) => {
                self.arena = Some(arena);
                Err(SelectedCloseRefused { plan: self, error })
            }
        }
    }

    pub(crate) fn range_for_value(&self, value: ValueId) -> Result<&DeviceRange<'ctx>> {
        let planned = self
            .candidate
            .value(value)
            .ok_or_else(|| invalid("value", "value is absent from selected plan"))?;
        self.ranges
            .get(&(planned.region, planned.slot))
            .ok_or_else(|| invalid("range", "selected range is absent"))
    }

    #[cfg(feature = "paged-attention-binding")]
    pub(crate) fn workspace_range(&self) -> Result<&DeviceRange<'ctx>> {
        self.ranges
            .get(&(StorageRegion::Workspace, 0))
            .ok_or_else(|| invalid("workspace", "selected workspace range is absent"))
    }

    #[cfg(feature = "cublas")]
    pub(crate) fn blas_workspace(&self) -> Result<(u64, usize)> {
        let offset = self
            .candidate
            .blas_workspace_offset()
            .ok_or_else(|| invalid("workspace", "selected plan has no cuBLAS workspace"))?;
        let workspace_bytes = self
            .candidate
            .blas_workspace_bytes()
            .ok_or_else(|| invalid("workspace", "selected plan has no cuBLAS workspace"))?;
        let end = offset
            .checked_add(workspace_bytes)
            .ok_or_else(|| invalid("workspace", "cuBLAS workspace extent overflowed"))?;
        let range = self.workspace_range()?;
        if end > range.bytes() {
            return Err(invalid(
                "workspace",
                "cuBLAS workspace exceeds the selected workspace range",
            ));
        }
        let address = range
            .device_address()?
            .checked_add(offset)
            .ok_or_else(|| invalid("workspace", "cuBLAS workspace address overflowed"))?;
        let bytes = usize::try_from(workspace_bytes)
            .map_err(|_| invalid("workspace", "cuBLAS workspace size is not addressable"))?;
        Ok((address, bytes))
    }

    #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
    pub(crate) fn range_for_selected_value(&self, value: ValueId) -> Result<&DeviceRange<'ctx>> {
        self.range_for_value(value)
    }

    #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
    pub(crate) fn take_range_for_value(
        &mut self,
        value: ValueId,
    ) -> Result<((StorageRegion, u32), DeviceRange<'ctx>)> {
        let planned = self
            .candidate
            .value(value)
            .ok_or_else(|| invalid("value", "value is absent from selected plan"))?;
        let key = (planned.region, planned.slot);
        self.ranges
            .remove(&key)
            .map(|range| (key, range))
            .ok_or_else(|| invalid("range", "selected range is absent"))
    }

    /// Return to this plan's arena a range it allocated that a nested owner
    /// kept across an unobserved launch and has handed back after the
    /// caller observed the launch's stream drained. A range from another
    /// arena comes back refused.
    #[cfg(feature = "paged-attention-binding")]
    #[allow(clippy::result_large_err)]
    pub(crate) fn release_reclaimed(
        &mut self,
        range: DeviceRange<'ctx>,
    ) -> std::result::Result<(), RangeReleaseRefused<'ctx>> {
        match self.arena.as_mut() {
            Some(arena) => arena.release(range),
            None => Err(RangeReleaseRefused {
                range,
                error: invalid("arena", "selected plan arena is already closed"),
            }),
        }
    }

    #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
    pub(crate) fn restore_range(
        &mut self,
        key: (StorageRegion, u32),
        range: DeviceRange<'ctx>,
    ) -> Result<()> {
        if self.ranges.contains_key(&key) {
            core::mem::forget(range);
            return Err(invalid("range", "selected range was already present"));
        }
        self.ranges.insert(key, range);
        Ok(())
    }
}

impl Drop for SelectedReservedPlan<'_> {
    fn drop(&mut self) {
        // A plan dropped without `close` has no proof that submitted work is
        // settled. Each field below is withheld or has an inert Drop so it
        // cannot unload or free a resource still reachable by device work.
        // CapturedGraph::drop destroys graph executables, so quarantine graphs.
        std::mem::forget(std::mem::take(&mut self.captured));
        #[cfg(feature = "cublas")]
        // Blas deliberately has no Drop; dropping this wrapper leaks its handle.
        let _ = self.blas.take();
        if let Some(package) = self.package.take() {
            // ResolvedModule::drop unloads the CUDA module.
            std::mem::forget(package);
        }
        // DeviceRange and DeviceArena drop free device allocations; keep the
        // ledger reservation below charged while their allocations are held.
        std::mem::forget(std::mem::take(&mut self.ranges));
        // Reservation has no releasing Drop; the ledger stays charged until
        // explicit close releases this token.
        self.graph_reservation.take();
        if let Some(arena) = self.arena.take() {
            std::mem::forget(arena);
        }
    }
}

#[cfg(feature = "paged-attention-binding")]
fn selected_module_symbols(
    candidate: &SelectedPlanCandidate,
) -> (Vec<String>, BTreeMap<NodeId, usize>) {
    let mut symbols = Vec::new();
    let mut indices = BTreeMap::new();
    for node in candidate.nodes() {
        for symbol in &node.descriptor.symbols {
            if symbol.0.starts_with("cublas:") {
                continue;
            }
            indices.entry(node.node).or_insert(symbols.len());
            symbols.push(symbol.0.clone());
        }
    }
    (symbols, indices)
}

// CUDA 13.1 raw captures measured cublasGemmEx at (rows,in,out) 1x5376x21504,
// 33x1024x3072 and 512x4096x4096: nodes were [1,1,1] on GPU-97fe4889-4874-a378-198e-955d2e72c4a3,
// and [2,1,1] on both GPU-3032cfa3-19df-028f-5ebd-43314911e0b9 and
// GPU-81fe4578-59b2-37c4-421e-287cdac78704; every cuMemGetInfo pool delta was 0 bytes.
// Four nodes charges 32 KiB/call at the existing 8 KiB per-node graph bound.
const CUBLAS_CAPTURE_NODE_BOUND: u64 = 4;

impl<'ctx> OperationLease<SelectedCompletion<'ctx>, ChainOperation<'ctx>> {
    #[allow(clippy::result_large_err)]
    pub fn finish(
        mut self,
    ) -> std::result::Result<
        ChainResult<'ctx>,
        OperationRetireRefused<SelectedCompletion<'ctx>, ChainOperation<'ctx>>,
    > {
        if let Err(error) = self.synchronize() {
            let error = attribute_chain_error(
                error,
                self.resource().device_ordinal,
                self.resource()
                    .plan
                    .as_ref()
                    .expect("operation retains plan")
                    .candidate(),
                "completion event",
            );
            return Err(OperationRetireRefused { lease: self, error });
        }
        let output_value = self
            .resource()
            .plan
            .as_ref()
            .expect("operation has plan")
            .candidate
            .workload()
            .output;
        let bytes = self
            .resource()
            .plan
            .as_ref()
            .expect("operation has plan")
            .candidate
            .value(output_value)
            .expect("planned output")
            .logical_bytes;
        let mut output = vec![0u8; bytes as usize];
        if let Err(error) = self
            .resource()
            .plan
            .as_ref()
            .expect("operation has plan")
            .range_for_value(output_value)
            .and_then(|range| range.copy_to_host(&mut output))
        {
            let error = attribute_chain_error(
                error,
                self.resource().device_ordinal,
                self.resource()
                    .plan
                    .as_ref()
                    .expect("operation retains plan")
                    .candidate(),
                "final output readback",
            );
            // A readback can surface an asynchronous fatal device status even
            // after the event synchronized successfully. Persist that
            // classification through the shared lifecycle before handing the
            // sole ownership token back. Nonfatal refusals remain retryable.
            self.persist_loss(error.clone());
            return Err(OperationRetireRefused { lease: self, error });
        }
        if output
            .chunks_exact(2)
            .any(|word| !bf16_is_finite(u16::from_le_bytes([word[0], word[1]])))
        {
            return Err(OperationRetireRefused {
                lease: self,
                error: Error::Numerical {
                    detail: "selected chain produced a nonfinite BF16 output".into(),
                },
            });
        }
        let (_, mut operation) = self.retire()?;
        let launch_order = std::mem::take(&mut operation.launch_order);
        let (plan, returned_inputs) = operation.settle_sources();
        // The reservation covered the output allocation while every upload
        // source was still retained. Returning this owned Vec is the explicit
        // handoff from engine-accounted transient storage to the caller.
        Ok(ChainResult {
            plan,
            output,
            returned_inputs,
            launch_order,
        })
    }
}

fn captured_graph_counts(candidate: &SelectedPlanCandidate) -> Result<(u64, u64)> {
    let mut kernels = 0u64;
    let mut segments = 0u64;
    let mut in_segment = false;
    for node in candidate.nodes() {
        if matches!(
            node.descriptor.operation,
            SemanticKernelOp::PagedAttention | SemanticKernelOp::CombineHostJoin
        ) {
            in_segment = false;
            continue;
        }
        if !in_segment {
            segments = segments
                .checked_add(1)
                .ok_or_else(|| invalid("capture", "segment count overflowed"))?;
            in_segment = true;
        }
        let node_kernels = if node
            .descriptor
            .symbols
            .iter()
            .any(|symbol| symbol.0.starts_with("cublas:"))
        {
            CUBLAS_CAPTURE_NODE_BOUND
        } else {
            u64::try_from(node.descriptor.symbols.len())
                .map_err(|_| invalid("capture", "kernel count exceeds u64"))?
        };
        kernels = kernels
            .checked_add(node_kernels)
            .ok_or_else(|| invalid("capture", "kernel count overflowed"))?;
    }
    Ok((kernels, segments))
}

/// Exact admission envelope derived from an immutable selected candidate.
///
/// This is public so validation and diagnostics can inspect the tier charges,
/// real stage spans, and host-source retention without allocating a device.
pub fn selected_resource_request(candidate: &SelectedPlanCandidate) -> Result<PlanRequest> {
    selected_resource_request_with_weights(candidate, true)
}

fn selected_resource_request_with_weights(
    candidate: &SelectedPlanCandidate,
    include_weights: bool,
) -> Result<PlanRequest> {
    // Both the label and the stage names are copied **fallibly**: the stages
    // borrow the candidate, which does not outlive the request, and `format!`
    // aborts where this has a refusal to return.
    let mut stages = moxie_memory::fallible::with_capacity(candidate.stages().len())?;
    for stage in candidate.stages() {
        stages.push(moxie_memory::request::Label::from(
            moxie_memory::fallible::string(stage)?,
        ));
    }
    let mut request = PlanRequest::new(
        moxie_memory::fallible::text(format_args!(
            "selected-plan-{}",
            candidate.base().id().get()
        ))?,
        stages,
    )?;
    let scope = Scope::Device(candidate.workload().device);
    let last = u32::try_from(candidate.stages().len() - 1)
        .map_err(|_| invalid("stages", "selected stage count exceeds u32"))?;
    for (label, tier, bytes, span) in [
        (
            "weights",
            DeviceTier::PackedResidentWeights,
            candidate.weight_region_bytes(),
            StageSpan::inclusive(0, last),
        ),
        (
            "activations",
            DeviceTier::Activations,
            candidate.activation_region_bytes(),
            StageSpan::inclusive(0, last),
        ),
        (
            "workspace",
            DeviceTier::KernelWorkspace,
            candidate.workspace_region_bytes(),
            StageSpan::inclusive(
                candidate.workspace().first_stage,
                candidate.workspace().last_stage,
            ),
        ),
    ] {
        if bytes == 0 || (!include_weights && tier == DeviceTier::PackedResidentWeights) {
            continue;
        }
        request.buffer(BufferRequest::new(
            label,
            scope,
            Tier::Device(tier),
            bytes,
            span,
        ))?;
    }
    let source_bytes = candidate
        .base()
        .bindings()
        .iter()
        .try_fold(0u64, |sum, binding| {
            let bytes = match binding {
                moxie_plan::ValueBinding::ExternalInput(value) => value.required_bytes,
                moxie_plan::ValueBinding::ExternalWeight(value) if include_weights => {
                    value.required_bytes
                }
                moxie_plan::ValueBinding::ExternalWeight(_) => 0,
                moxie_plan::ValueBinding::ArenaTensor(_) => 0,
            };
            sum.checked_add(bytes)
                .ok_or_else(|| invalid("host_sources", "source extent overflowed"))
        })?;
    let host_bytes = source_bytes
        .checked_add(candidate.host_workspace_bytes())
        .ok_or_else(|| invalid("host_sources", "dense host workspace overflowed"))?;
    request.buffer(BufferRequest::new(
        "retained-upload-sources",
        Scope::Host,
        Tier::Host(HostTier::Pageable),
        host_bytes,
        StageSpan::inclusive(0, last),
    ))?;
    if !candidate.is_paged_attention() {
        let output_bytes = candidate
            .value(candidate.workload().output)
            .ok_or_else(|| invalid("output", "selected output is absent from the physical plan"))?
            .logical_bytes;
        request.buffer(BufferRequest::new(
            "final-output-readback",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            output_bytes,
            StageSpan::at(last),
        ))?;
    }
    Ok(request)
}

pub(crate) fn validate_bindings(
    plan: &SelectedReservedPlan<'_>,
    graph: &Graph,
    bindings: &[OwnedBinding],
) -> Result<()> {
    validate_bindings_except(plan, graph, bindings, &BTreeSet::new())
}

/// [`validate_bindings`], except that the `resident` inputs are already in
/// the plan's own ranges and so have no host binding.
pub(crate) fn validate_bindings_except(
    plan: &SelectedReservedPlan<'_>,
    graph: &Graph,
    bindings: &[OwnedBinding],
    resident: &BTreeSet<ValueId>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for binding in bindings {
        if plan.resident_weights.contains_key(&binding.value) {
            return Err(invalid("bindings", "this plan's weights are resident"));
        }
        if !seen.insert(binding.value) {
            return Err(invalid("bindings", "duplicate value binding"));
        }
        if resident.contains(&binding.value) {
            // Its upload would overwrite the resident value.
            return Err(invalid("bindings", "a host binding names a resident input"));
        }
        let planned = plan
            .candidate
            .value(binding.value)
            .ok_or_else(|| invalid("bindings", "binding names no external plan value"))?;
        let external =
            graph.inputs().contains(&binding.value) || graph.weights().contains(&binding.value);
        if !external
            || binding.role != planned.role
            || binding.shape != planned.shape
            || binding.layout != TensorLayout::ContiguousRowMajorV1
            || binding.device != plan.candidate.workload().device
            || binding.bytes.len() as u64 != planned.logical_bytes
            || binding.bytes.capacity() != binding.bytes.len()
        {
            return Err(invalid(
                "bindings",
                format!(
                    "binding {} differs from its checked descriptor",
                    binding.value.0
                ),
            ));
        }
        if let Some(format) = plan.candidate.weight_formats().get(&binding.value) {
            let bad_scale = || {
                invalid(
                    "bindings",
                    format!(
                        "binding {} contains an invalid affine scale",
                        binding.value.0
                    ),
                )
            };
            let WeightFormat::Affine { scale, .. } = format else {
                return Err(bad_scale());
            };
            let [outputs, inputs] = planned.shape.as_slice() else {
                return Err(bad_scale());
            };
            let sections = format.sections(*outputs, *inputs).ok_or_else(bad_scale)?;
            let start = usize::try_from(sections.scales.0).map_err(|_| bad_scale())?;
            let length = usize::try_from(sections.scales.1).map_err(|_| bad_scale())?;
            let end = start.checked_add(length).ok_or_else(bad_scale)?;
            let scales = binding.bytes.get(start..end).ok_or_else(bad_scale)?;
            let valid = match scale {
                moxie_types::Precision::F16 => {
                    scales.len().is_multiple_of(2)
                        && scales.chunks_exact(2).all(|word| {
                            let bits = u16::from_le_bytes([word[0], word[1]]);
                            bits & 0x7c00 != 0x7c00 && bits & 0x7fff != 0
                        })
                }
                moxie_types::Precision::Bf16 => {
                    scales.len().is_multiple_of(2)
                        && scales.chunks_exact(2).all(|word| {
                            let bits = u16::from_le_bytes([word[0], word[1]]);
                            bf16_is_finite(bits) && bits & 0x7fff != 0
                        })
                }
                moxie_types::Precision::F32 => {
                    scales.len().is_multiple_of(4)
                        && scales.chunks_exact(4).all(|word| {
                            let value =
                                f32::from_le_bytes(word.try_into().expect("four-byte word"));
                            value.is_finite() && value != 0.0
                        })
                }
                _ => false,
            };
            if !valid {
                return Err(bad_scale());
            }
        } else if matches!(
            binding.role,
            ValueRole::Activation(_) | ValueRole::Weight(_)
        ) && binding
            .bytes
            .chunks_exact(2)
            .any(|word| !bf16_is_finite(u16::from_le_bytes([word[0], word[1]])))
        {
            return Err(invalid(
                "bindings",
                format!("binding {} contains nonfinite BF16", binding.value.0),
            ));
        }
        if matches!(binding.role, ValueRole::Weight(_))
            && plan.bound_weights.contains_key(&binding.value)
        {
            return Err(invalid("bindings", "an immutable weight cannot be rebound"));
        }
    }
    for value in graph.inputs() {
        if !seen.contains(value) && !resident.contains(value) {
            return Err(invalid(
                "bindings",
                format!("missing per-step input {}", value.0),
            ));
        }
    }
    for value in graph.weights() {
        if !seen.contains(value)
            && !plan.bound_weights.contains_key(value)
            && !plan.resident_weights.contains_key(value)
        {
            return Err(invalid(
                "bindings",
                format!("missing immutable weight {}", value.0),
            ));
        }
    }
    Ok(())
}

struct SelectedUnwindRefused<'ctx> {
    arena: DeviceArena<'ctx>,
    ranges: BTreeMap<(StorageRegion, u32), DeviceRange<'ctx>>,
    error: Error,
}

fn unwind_selected<'ctx>(
    mut arena: DeviceArena<'ctx>,
    mut ranges: BTreeMap<(StorageRegion, u32), DeviceRange<'ctx>>,
    ledger: &mut Ledger,
) -> std::result::Result<(), Box<SelectedUnwindRefused<'ctx>>> {
    while let Some((key, range)) = ranges.pop_last() {
        if let Err(RangeReleaseRefused { range, error }) = arena.release(range) {
            ranges.insert(key, range);
            return Err(Box::new(SelectedUnwindRefused {
                arena,
                ranges,
                error,
            }));
        }
    }
    match arena.close(ledger) {
        Ok(()) => Ok(()),
        Err(ArenaCloseRefused { arena, error }) => Err(Box::new(SelectedUnwindRefused {
            arena,
            ranges,
            error,
        })),
    }
}

fn enqueue<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, ChainOperation<'ctx>>,
    graph: &Graph,
    stream: &Stream<'ctx>,
    device_ordinal: u32,
) -> Result<()> {
    let operation = lease.resource();
    let plan = operation.plan.as_ref().expect("operation retains plan");
    for source in &operation.sources {
        // SAFETY: lease owns source and complete plan through the event that is
        // recorded only after every launch below.
        unsafe {
            plan.range_for_value(source.value)?
                .copy_from_host_async(&source.bytes, stream)?;
        }
    }
    let package = &operation.package;
    let rows = plan.candidate.workload().rows;
    let linear = &graph.nodes()[0];
    let rms = &graph.nodes()[1];
    let residual = &graph.nodes()[2];
    let address = |value| {
        plan.range_for_value(value)
            .and_then(DeviceRange::device_address)
    };
    let mut x = address(linear.inputs[0])?;
    let mut weight = address(linear.inputs[1])?;
    let mut h = address(linear.output)?;
    let (mut k, mut output_width) = match linear.params {
        OpParams::Linear {
            in_features,
            out_features,
            ..
        } => (in_features, out_features),
        _ => unreachable!(),
    };
    let mut launch_rows = rows;
    let mut params: [*mut c_void; 6] = [
        (&raw mut x).cast(),
        (&raw mut weight).cast(),
        (&raw mut h).cast(),
        (&raw mut launch_rows).cast(),
        (&raw mut k).cast(),
        (&raw mut output_width).cast(),
    ];
    let elements = rows
        .checked_mul(output_width)
        .ok_or_else(|| invalid("launch", "linear grid overflowed"))?;
    // SAFETY: selected descriptor fixes this symbol ABI; every pointer names a
    // checked admitted range and dimensions were bounded during pure lowering.
    unsafe {
        package
            .launch_async(
                0,
                stream,
                (elements.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut params,
            )
            .map_err(|error| {
                attribute_node_error(
                    error,
                    device_ordinal,
                    &plan.candidate.nodes()[0],
                    moxie_kernels::BF16_LINEAR,
                )
            })?;
    }

    let mut workspace = plan
        .ranges
        .get(&(StorageRegion::Workspace, 0))
        .expect("workspace range")
        .device_address()?;
    let mut hidden = match rms.params {
        OpParams::RmsNorm { hidden, .. } => hidden,
        _ => unreachable!(),
    };
    let mut rms_rows = rows;
    let mut reduce_params: [*mut c_void; 4] = [
        (&raw mut h).cast(),
        (&raw mut workspace).cast(),
        (&raw mut rms_rows).cast(),
        (&raw mut hidden).cast(),
    ];
    // SAFETY: exact RMS-reduce ABI and selected admitted ranges.
    unsafe {
        package
            .launch_async(
                1,
                stream,
                (rows.div_ceil(64) as u32, 1, 1),
                (64, 1, 1),
                0,
                &mut reduce_params,
            )
            .map_err(|error| {
                attribute_node_error(
                    error,
                    device_ordinal,
                    &plan.candidate.nodes()[1],
                    moxie_kernels::BF16_RMS_SUM,
                )
            })?;
    }
    let mut gain = address(rms.inputs[1])?;
    let mut n = address(rms.output)?;
    let mut epsilon = match rms.params {
        OpParams::RmsNorm { eps, .. } => eps,
        _ => unreachable!(),
    };
    let mut apply_params: [*mut c_void; 7] = [
        (&raw mut h).cast(),
        (&raw mut gain).cast(),
        (&raw mut workspace).cast(),
        (&raw mut n).cast(),
        (&raw mut rms_rows).cast(),
        (&raw mut hidden).cast(),
        (&raw mut epsilon).cast(),
    ];
    let rms_elements = rows
        .checked_mul(hidden)
        .ok_or_else(|| invalid("launch", "RMS grid overflowed"))?;
    // SAFETY: exact RMS-apply ABI and selected admitted ranges.
    unsafe {
        package
            .launch_async(
                2,
                stream,
                (rms_elements.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut apply_params,
            )
            .map_err(|error| {
                attribute_node_error(
                    error,
                    device_ordinal,
                    &plan.candidate.nodes()[1],
                    moxie_kernels::BF16_RMS_APPLY,
                )
            })?;
    }
    let mut left = address(residual.inputs[0])?;
    let mut right = address(residual.inputs[1])?;
    let mut y = address(residual.output)?;
    let mut residual_elements = rms_elements;
    let mut residual_params: [*mut c_void; 4] = [
        (&raw mut left).cast(),
        (&raw mut right).cast(),
        (&raw mut y).cast(),
        (&raw mut residual_elements).cast(),
    ];
    // SAFETY: exact residual ABI and selected admitted ranges.
    unsafe {
        package
            .launch_async(
                3,
                stream,
                (residual_elements.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut residual_params,
            )
            .map_err(|error| {
                attribute_node_error(
                    error,
                    device_ordinal,
                    &plan.candidate.nodes()[2],
                    moxie_kernels::BF16_RESIDUAL,
                )
            })?;
    }
    // The mutable diagnostic is behind the operation resource; use a narrow
    // reborrow after all immutable launch borrows have ended.
    let operation = lease.resource_mut();
    operation.launch_order = vec![
        "linear".into(),
        "rms-reduce".into(),
        "rms-apply".into(),
        "residual".into(),
    ];
    Ok(())
}

fn bf16_is_finite(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

pub(crate) fn attribute_node_error(
    error: Error,
    device: u32,
    node: &SelectedNode,
    symbol: &str,
) -> Error {
    attribute_error(
        error,
        device,
        format!(
            "node {} kernel {} symbol {symbol}",
            node.node.0, node.descriptor.id.0
        ),
    )
}

pub(crate) fn attribute_chain_error(
    error: Error,
    device: u32,
    candidate: &SelectedPlanCandidate,
    boundary: &str,
) -> Error {
    attribute_error(error, device, chain_attribution(candidate, boundary))
}

fn chain_attribution(candidate: &SelectedPlanCandidate, boundary: &str) -> String {
    let kernels = candidate
        .nodes()
        .iter()
        .map(|node| format!("node {} kernel {}", node.node.0, node.descriptor.id.0))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{boundary} for [{kernels}]")
}

pub(crate) fn attribute_error(error: Error, device: u32, attribution: String) -> Error {
    match error {
        Error::DeviceLost { detail, .. } => Error::DeviceLost {
            device,
            detail: format!("{attribution}: {detail}"),
        },
        Error::UnsupportedKernel { operation, detail } => Error::UnsupportedKernel {
            operation,
            detail: format!("{attribution}: {detail}"),
        },
        Error::InvalidArtifact { detail } => Error::InvalidArtifact {
            detail: format!("{attribution}: {detail}").into(),
        },
        Error::Numerical { detail } => Error::Numerical {
            detail: format!("{attribution}: {detail}"),
        },
        Error::InvalidRequest { field, detail } => Error::InvalidRequest {
            field,
            detail: format!("{attribution}: {detail}"),
        },
        Error::Unsupported { capability, reason } => Error::Unsupported {
            capability,
            reason: format!("{attribution}: {reason}"),
        },
        Error::Cancelled { at } => Error::InvalidRequest {
            field: "chain",
            detail: format!("{attribution}: cancelled at {at}"),
        },
        Error::CapacityExceeded { .. } | Error::Dim(_) => Error::DeviceLost {
            device,
            detail: format!("{attribution}: {error}"),
        },
        // Passed through, uniquely among these. Every other arm carries the
        // chain attribution in a free-text field; `Reclaimed` has none, and
        // wrapping it in one that does would destroy the distinction it exists
        // to make -- a caller has to be able to tell "that history is gone,
        // re-prefill" from a malformed request. Its own fields already name the
        // layer and the position, which is the more actionable attribution.
        //
        // No device path produces it today: paged state is host-resident and
        // reclamation happens in `moxie-state`. This arm is here because the
        // match is exhaustive on purpose, so a new variant forces the decision
        // rather than falling into a wildcard.
        Error::Reclaimed { .. } => error,
    }
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_memory::CapacitySnapshot;
    use moxie_types::RankId;

    #[test]
    fn device_loss_attribution_uses_the_selected_real_ordinal() {
        let attributed = attribute_error(
            Error::DeviceLost {
                device: u32::MAX,
                detail: "driver classifier had no ordinal".into(),
            },
            7,
            "node 2 kernel selected".into(),
        );
        let Error::DeviceLost { device, detail } = attributed else {
            panic!("device loss changed classifier")
        };
        assert_eq!(device, 7);
        assert!(detail.contains("node 2 kernel selected"));
    }

    #[test]
    fn forced_range_failure_unwinds_selected_arena_and_charge() {
        let _serial = crate::DRIVER_TEST_LOCK.lock().unwrap();
        let ctx = RankContext::acquire(RankId(12_012), 0)
            .expect("device-feature tests require the first visible GPU");
        let snapshot = CapacitySnapshot::new(Scope::Device(ctx.uuid()), 1 << 20, 1024).unwrap();
        let mut ledger = Ledger::new([snapshot]).unwrap();
        let mut request = PlanRequest::new("selected unwind fixture", ["bind"]).unwrap();
        request
            .buffer(BufferRequest::new(
                "activation arena",
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::Activations),
                512,
                StageSpan::at(0),
            ))
            .unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut arena = DeviceArena::create_partitioned(
            &ledger,
            reservation,
            &ctx,
            &[(DeviceTier::Activations, 512)],
            512,
            "selected unwind fixture",
        )
        .unwrap();
        let first = arena.allocate(256, 256, "first selected range").unwrap();
        let ranges = BTreeMap::from([((StorageRegion::Activations, 0), first)]);
        let refusal = arena
            .allocate(512, 256, "forced unavailable selected range")
            .unwrap_err();
        assert_eq!(refusal.error.kind(), "capacity_exceeded");
        assert!(unwind_selected(arena, ranges, &mut ledger).is_ok());
        assert!(ledger.outstanding().is_empty());
    }
}
