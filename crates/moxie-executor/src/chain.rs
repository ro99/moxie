//! Binding and event-retained execution of the selected BF16 layer chain.

#![cfg(feature = "driver")]

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

use moxie_cuda::{Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage};
use moxie_memory::{
    AdmitError, BufferRequest, Ledger, LedgerId, PlanRequest, Reservation, StageSpan,
};
use moxie_plan::{
    Graph, OpParams, SelectedNode, SelectedPlanCandidate, StorageRegion, ValueId, ValueRole,
};
use moxie_types::{
    DeviceCapability, DeviceTier, Error, HostTier, KernelCatalogue, Result, Scope, TensorLayout,
    Tier,
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
        rejection: Box<moxie_memory::Rejection>,
    },
    Held {
        candidate: Box<SelectedPlanCandidate>,
        resource: SelectedHeldResource<'ctx>,
        error: Error,
    },
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
    event: Event<'ctx>,
    device_ordinal: u32,
    attribution: String,
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
    arena: Option<DeviceArena<'ctx>>,
    ranges: BTreeMap<(StorageRegion, u32), DeviceRange<'ctx>>,
    bound_weights: BTreeMap<ValueId, OwnedBinding>,
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

    /// Recover a completed operation retired by the shared turn sweep. The
    /// sweep has already observed its event, so these resources may be closed
    /// or rebound explicitly by their next owner.
    pub fn into_parts(mut self) -> (SelectedReservedPlan<'ctx>, Vec<OwnedBinding>) {
        (
            self.plan.take().expect("chain operation retains its plan"),
            std::mem::take(&mut self.sources),
        )
    }
}

impl<'ctx> SelectedReservedPlan<'ctx> {
    pub fn admit(
        candidate: SelectedPlanCandidate,
        graph: &Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
    ) -> std::result::Result<Self, SelectedAdmitRefused<'ctx>> {
        let fail = |candidate, error| SelectedAdmitRefused::Invalid {
            candidate: Box::new(candidate),
            error,
        };
        if !candidate.matches(graph, capability, catalogue) || ctx.uuid() != capability.uuid {
            return Err(fail(
                candidate,
                invalid("plan", "graph/catalogue/capability/UUID binding changed"),
            ));
        }
        let request = match selected_resource_request(&candidate) {
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
        let regions = [
            (
                DeviceTier::PackedResidentWeights,
                candidate.weight_region_bytes(),
            ),
            (DeviceTier::Activations, candidate.activation_region_bytes()),
            (
                DeviceTier::KernelWorkspace,
                candidate.workspace_region_bytes(),
            ),
        ];
        let mut arena = match DeviceArena::create_partitioned(
            ledger,
            reservation,
            ctx,
            &regions,
            candidate.combined_arena_bytes(),
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
            specs
                .entry((value.region, value.slot))
                .or_insert((value.offset, value.physical_bytes));
        }
        specs.insert(
            (StorageRegion::Workspace, 0),
            (
                candidate.workspace().offset,
                candidate.workspace().physical_bytes,
            ),
        );
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
        Ok(Self {
            candidate,
            arena: Some(arena),
            ranges,
            bound_weights: BTreeMap::new(),
            ledger: ledger.id(),
        })
    }

    pub fn candidate(&self) -> &SelectedPlanCandidate {
        &self.candidate
    }
    pub fn bound_weight_count(&self) -> usize {
        self.bound_weights.len()
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

    fn range_for_value(&self, value: ValueId) -> Result<&DeviceRange<'ctx>> {
        let planned = self
            .candidate
            .value(value)
            .ok_or_else(|| invalid("value", "value is absent from selected plan"))?;
        self.ranges
            .get(&(planned.region, planned.slot))
            .ok_or_else(|| invalid("range", "selected range is absent"))
    }
}

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
        let mut plan = operation.plan.take().expect("operation retains plan");
        let mut returned_inputs = Vec::new();
        for binding in operation.sources.drain(..) {
            if matches!(binding.role, ValueRole::Weight(_)) {
                plan.bound_weights.insert(binding.value, binding);
            } else {
                returned_inputs.push(binding);
            }
        }
        Ok(ChainResult {
            plan,
            output,
            returned_inputs,
            launch_order: operation.launch_order,
        })
    }
}

/// Exact admission envelope derived from an immutable selected candidate.
///
/// This is public so validation and diagnostics can inspect the tier charges,
/// real stage spans, and host-source retention without allocating a device.
pub fn selected_resource_request(candidate: &SelectedPlanCandidate) -> Result<PlanRequest> {
    let mut request = PlanRequest::new(
        format!("selected-bf16-chain-{}", candidate.base().id().get()),
        candidate.stages().iter().map(String::as_str),
    )?;
    let scope = Scope::Device(candidate.workload().device);
    for (label, tier, bytes, span) in [
        (
            "weights",
            DeviceTier::PackedResidentWeights,
            candidate.weight_region_bytes(),
            StageSpan::inclusive(0, 4),
        ),
        (
            "activations",
            DeviceTier::Activations,
            candidate.activation_region_bytes(),
            StageSpan::inclusive(0, 4),
        ),
        (
            "workspace",
            DeviceTier::KernelWorkspace,
            candidate.workspace_region_bytes(),
            StageSpan::inclusive(1, 2),
        ),
    ] {
        request.buffer(BufferRequest::new(
            label,
            scope,
            Tier::Device(tier),
            bytes,
            span,
        ))?;
    }
    let host_bytes = candidate
        .base()
        .bindings()
        .iter()
        .try_fold(0u64, |sum, binding| {
            let bytes = match binding {
                moxie_plan::ValueBinding::ExternalInput(value) => value.required_bytes,
                moxie_plan::ValueBinding::ExternalWeight(value) => value.required_bytes,
                moxie_plan::ValueBinding::ArenaTensor(_) => 0,
            };
            sum.checked_add(bytes)
                .ok_or_else(|| invalid("host_sources", "source extent overflowed"))
        })?;
    request.buffer(BufferRequest::new(
        "retained-upload-sources",
        Scope::Host,
        Tier::Host(HostTier::Pageable),
        host_bytes,
        StageSpan::inclusive(0, 3),
    ))?;
    Ok(request)
}

fn validate_bindings(
    plan: &SelectedReservedPlan<'_>,
    graph: &Graph,
    bindings: &[OwnedBinding],
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for binding in bindings {
        if !seen.insert(binding.value) {
            return Err(invalid("bindings", "duplicate value binding"));
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
        if binding
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
        if !seen.contains(value) {
            return Err(invalid(
                "bindings",
                format!("missing per-step input {}", value.0),
            ));
        }
    }
    for value in graph.weights() {
        if !seen.contains(value) && !plan.bound_weights.contains_key(value) {
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

fn attribute_node_error(error: Error, device: u32, node: &SelectedNode, symbol: &str) -> Error {
    attribute_error(
        error,
        device,
        format!(
            "node {} kernel {} symbol {symbol}",
            node.node.0, node.descriptor.id.0
        ),
    )
}

fn attribute_chain_error(
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

fn attribute_error(error: Error, device: u32, attribution: String) -> Error {
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
            detail: format!("{attribution}: {detail}"),
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
