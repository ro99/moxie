//! Device execution of the selected reduced dense graph.
//!
//! This is a graph executor, not a model loop.  The planner owns the checked
//! node order and arena; this module only binds those values to the shared
//! semantic kernels and hands conventional KV state to the existing paged
//! attention authority.

#![cfg(all(feature = "driver", feature = "paged-attention-binding"))]

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "cublas")]
use moxie_cuda::copy_2d_async;
use moxie_cuda::{Event, Module, ModuleImage, RankContext, Stream, TrustedImage};
use moxie_graph::{Graph, NodeId, OpParams, RopeLayout, ValueId, ValueRole};
use moxie_kernels::cpu_expert::{ExpertAssignment, ExpertShape, ExpertTiling};
use moxie_plan::{HostExpertJoin, SelectedNode, Visibility, WeightFormat};
use moxie_state::DeviceKvSequence;
#[cfg(feature = "cublas")]
use moxie_types::{AccumulationPolicy, SmVersion};
use moxie_types::{
    DeviceCapability, Error, GateTransform, KernelCatalogue, Precision, Result, SemanticKernelOp,
    StateTransactionId,
};

use crate::arena::{OperationLease, OperationRetireRefused};
use crate::chain::{
    OwnedBinding, SelectedCompletion, SelectedReservedPlan, attribute_chain_error,
    attribute_node_error, validate_bindings_except,
};
use crate::paged_attention::device::{PagedAttentionRun, append_paged_layer_from_device};
use crate::{AttentionLayer, PageGeometry, PagedAttentionLaunch};

/// The inputs and state authorities needed for one selected dense graph step.
#[derive(Debug)]
pub struct DenseGraphStep<'step, 'ctx> {
    pub graph: &'step Graph,
    pub capability: &'step DeviceCapability,
    pub catalogue: &'step KernelCatalogue,
    pub ctx: &'ctx RankContext,
    pub stream: &'step Stream<'ctx>,
    pub state: &'step mut DeviceKvSequence,
    pub transaction: StateTransactionId,
    pub runs: &'step mut [PagedAttentionRun<'ctx>],
    pub bindings: Vec<OwnedBinding>,
    pub host_experts: &'step [HostExpertWeights<'step>],
}

/// Host-owned BF16 expert weights for one routed Combine.
#[derive(Debug)]
pub struct HostExpertWeights<'a> {
    pub combine: NodeId,
    pub gate_up: &'a [u8],
    pub down: &'a [u8],
}

/// A completed dense step.  The returned non-weight bindings are the caller's
/// ownership again; weights stay attached to the admitted plan for reuse.
#[derive(Debug)]
pub struct DenseGraphResult<'ctx> {
    pub plan: SelectedReservedPlan<'ctx>,
    pub output: Vec<u8>,
    pub returned_inputs: Vec<OwnedBinding>,
    pub launch_order: Vec<String>,
}

/// A dense launch refusal.  Once device work has been submitted, the lease is
/// retained here rather than exposing a plan whose ranges may still be live.
#[derive(Debug)]
pub struct DensePlanRunRefused<'ctx> {
    pub plan: Option<SelectedReservedPlan<'ctx>>,
    pub bindings: Vec<OwnedBinding>,
    pub error: Error,
    pub held: Option<OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>>,
}

#[derive(Debug)]
pub struct DenseOperation<'ctx> {
    pub(crate) plan: Option<SelectedReservedPlan<'ctx>>,
    #[cfg(feature = "cublas")]
    ctx: &'ctx RankContext,
    /// Every source copied by `upload_sources` stays here until the graph's
    /// completion event is observed. This includes token and position indices.
    pub(crate) sources: Vec<OwnedBinding>,
    /// Every angle table uploaded by the step stays here until the lease retires
    /// after the completion event, like `sources`.
    rope_tables: Vec<((u64, u64, u32), Vec<u8>)>,
    pub(crate) launch_order: Vec<String>,
    pub(crate) device_ordinal: u32,
    mode: DenseStepMode,
    segment: usize,
    open: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseStepMode {
    Eager,
    Capture,
    Replay,
}

impl<'ctx> SelectedReservedPlan<'ctx> {
    /// Execute one admitted dense graph step through the shared arena and
    /// conventional device KV state.  The transaction remains the caller's to
    /// commit or abort, exactly as it is for the standalone attention path.
    #[allow(clippy::result_large_err)]
    pub fn execute_dense(
        self,
        step: DenseGraphStep<'_, 'ctx>,
    ) -> std::result::Result<
        OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
        DensePlanRunRefused<'ctx>,
    > {
        self.execute_dense_stage(step, &BTreeSet::new(), &BTreeMap::new())
    }

    /// [`Self::execute_dense`] for one tensor-parallel stage graph. `resident`
    /// inputs were already copied into this plan's input ranges on `stream`,
    /// so no host binding names them. `layers` restores each stage-local
    /// attention node's layer in the rank's full KV authority.
    #[allow(clippy::result_large_err)]
    pub(crate) fn execute_dense_stage(
        mut self,
        step: DenseGraphStep<'_, 'ctx>,
        resident: &BTreeSet<ValueId>,
        layers: &BTreeMap<NodeId, u32>,
    ) -> std::result::Result<
        OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
        DensePlanRunRefused<'ctx>,
    > {
        let DenseGraphStep {
            graph,
            capability,
            catalogue,
            ctx,
            stream,
            state,
            transaction,
            runs,
            bindings,
            host_experts,
        } = step;
        let reject = |plan, bindings, error| DensePlanRunRefused {
            plan: Some(plan),
            bindings,
            error,
            held: None,
        };
        #[cfg(feature = "cublas")]
        let sm = SmVersion {
            major: capability.compute_major,
            minor: capability.compute_minor,
        };
        let known_catalogue = catalogue.digest() == moxie_kernels::dense_graph_catalogue().digest();
        #[cfg(feature = "cublas")]
        let known_catalogue = known_catalogue
            || catalogue.digest() == moxie_kernels::dense_graph_catalogue_unordered(sm).digest();
        if !self.candidate().is_dense()
            || !self.candidate().matches(graph, capability, catalogue)
            || !known_catalogue
            || ctx.uuid() != capability.uuid
            || stream.device_uuid() != capability.uuid
        {
            return Err(reject(
                self,
                bindings,
                invalid("execution", "admitted dense graph identity changed"),
            ));
        }
        #[cfg(feature = "cublas")]
        if let Err(error) = validate_cublas_descriptors(self.candidate()) {
            return Err(reject(self, bindings, error));
        }
        if let Err(error) =
            validate_host_expert_weights(graph, self.candidate().host_expert_joins(), host_experts)
        {
            return Err(reject(self, bindings, error));
        }
        if let Err(error) = validate_bindings_except(&self, graph, &bindings, resident) {
            return Err(reject(self, bindings, error));
        }
        if self.package.is_none() && !self.module_symbols.is_empty() {
            let symbols = self.module_symbols.clone();
            // SAFETY: this is the nvcc output embedded by this build, and selection
            // above binds the plan to the dense catalogue's image identity.
            let trusted =
                match unsafe { TrustedImage::from_build_output(moxie_kernels::DENSE_GRAPH_FATBIN) }
                {
                    Ok(image) => image,
                    Err(error) => return Err(reject(self, bindings, error)),
                };
            let package = match Module::load(ctx, ModuleImage::Binary(trusted))
                .and_then(|module| module.resolve_all(&symbols))
            {
                Ok(package) => package,
                Err(error) => return Err(reject(self, bindings, error)),
            };
            self.package = Some(package);
        }
        #[cfg(feature = "cublas")]
        if self.candidate().nodes().iter().any(is_cublas_node) {
            let (workspace, bytes) = match self.blas_workspace() {
                Ok(value) => value,
                Err(error) => return Err(reject(self, bindings, error)),
            };
            if self.blas.is_none() {
                self.blas = match moxie_cuda::Blas::new(ctx) {
                    Ok(blas) => Some(blas),
                    Err(error) => return Err(reject(self, bindings, error)),
                };
            }
            // SAFETY: the selected plan owns this workspace and the step's event
            // retains it with the handle through observed completion.
            if let Err(error) = unsafe {
                self.blas
                    .as_mut()
                    .expect("cuBLAS handle was created")
                    .bind(stream, workspace, bytes)
            } {
                return Err(reject(self, bindings, error));
            }
        }
        let operation = DenseOperation {
            plan: Some(self),
            #[cfg(feature = "cublas")]
            ctx,
            sources: bindings,
            rope_tables: Vec::new(),
            launch_order: Vec::new(),
            device_ordinal: ctx.ordinal(),
            mode: DenseStepMode::Eager,
            segment: 0,
            open: false,
        };
        let mut lease = OperationLease::new("selected reduced dense graph", operation)
            .expect("static label is nonempty");
        if let Err(error) = enqueue_dense(
            &mut lease,
            graph,
            state,
            transaction,
            runs,
            layers,
            host_experts,
            stream,
        ) {
            lease.mark_lost(
                ctx.ordinal(),
                format!("selected dense graph submission failed: {error}"),
            );
            return Err(DensePlanRunRefused {
                plan: None,
                bindings: Vec::new(),
                held: Some(lease),
                error,
            });
        }
        let event = match Event::new(ctx) {
            Ok(event) => event,
            Err(error) => {
                lease.mark_lost(
                    ctx.ordinal(),
                    format!("dense completion event creation failed: {error}"),
                );
                return Err(DensePlanRunRefused {
                    plan: None,
                    bindings: Vec::new(),
                    held: Some(lease),
                    error,
                });
            }
        };
        if let Err(error) = event.record(stream) {
            lease.mark_lost(
                ctx.ordinal(),
                format!("dense completion event record failed: {error}"),
            );
            return Err(DensePlanRunRefused {
                plan: None,
                bindings: Vec::new(),
                held: Some(lease),
                error,
            });
        }
        let attribution = {
            let plan = lease
                .resource()
                .plan
                .as_ref()
                .expect("dense operation retains plan");
            dense_attribution(plan.candidate(), "completion event")
        };
        lease
            .submit_tracked(SelectedCompletion::new(event, ctx.ordinal(), attribution))
            .expect("new dense operation is live");
        Ok(lease)
    }
}

impl<'ctx> OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>> {
    /// Synchronize, read the FP32 vocabulary output, and retire the one event
    /// lease.  The output is intentionally not rounded: that is the graph
    /// contract of `VocabProjection`.
    #[allow(clippy::result_large_err)]
    pub fn finish(
        mut self,
    ) -> std::result::Result<
        DenseGraphResult<'ctx>,
        OperationRetireRefused<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    > {
        if let Err(error) = self.synchronize() {
            let operation = self.resource();
            let candidate = operation
                .plan
                .as_ref()
                .expect("dense operation retains plan")
                .candidate();
            let error = attribute_chain_error(
                error,
                operation.device_ordinal,
                candidate,
                "completion event",
            );
            return Err(OperationRetireRefused { lease: self, error });
        }
        let (output_value, bytes, role, device_ordinal) = {
            let operation = self.resource();
            let plan = operation
                .plan
                .as_ref()
                .expect("dense operation retains plan");
            let output = plan.candidate().workload().output;
            let planned = plan
                .candidate()
                .value(output)
                .expect("planned dense output");
            (
                output,
                planned.logical_bytes,
                planned.role,
                operation.device_ordinal,
            )
        };
        let mut output = vec![0u8; bytes as usize];
        let read = self
            .resource()
            .plan
            .as_ref()
            .expect("dense operation retains plan")
            .range_for_value(output_value)
            .and_then(|range| range.copy_to_host(&mut output));
        if let Err(error) = read {
            let error = attribute_chain_error(
                error,
                device_ordinal,
                self.resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .candidate(),
                "dense logits readback",
            );
            self.persist_loss(error.clone());
            return Err(OperationRetireRefused { lease: self, error });
        }
        // Only an FP32 output (logits, or a TP partial) is checked here; a
        // stage's BF16 boundary is not four-byte words.
        let f32_output = matches!(role, ValueRole::Activation(p) if p.get() == Precision::F32);
        if f32_output
            && output.chunks_exact(4).any(|word| {
                f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_nan()
                    || !f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_finite()
            })
        {
            return Err(OperationRetireRefused {
                lease: self,
                error: Error::Numerical {
                    detail: "selected dense graph produced nonfinite FP32 logits".into(),
                },
            });
        }
        let (_, mut operation) = self.retire()?;
        let launch_order = core::mem::take(&mut operation.launch_order);
        let mut plan = operation.plan.take().expect("dense operation retains plan");
        let returned_inputs = plan.settle_sources(operation.sources);
        Ok(DenseGraphResult {
            plan,
            output,
            returned_inputs,
            launch_order,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn enqueue_dense<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    graph: &Graph,
    state: &mut DeviceKvSequence,
    transaction: StateTransactionId,
    runs: &mut [PagedAttentionRun<'ctx>],
    layers: &BTreeMap<NodeId, u32>,
    host_experts: &[HostExpertWeights<'_>],
    stream: &Stream<'ctx>,
) -> Result<()> {
    let mode = {
        let operation = lease.resource();
        let plan = operation
            .plan
            .as_ref()
            .expect("dense operation retains plan");
        if !plan.capture_enabled {
            DenseStepMode::Eager
        } else if plan.captured.is_empty() {
            DenseStepMode::Capture
        } else {
            DenseStepMode::Replay
        }
    };
    {
        let operation = lease.resource_mut();
        operation.mode = mode;
        operation.segment = 0;
        operation.open = false;
    }

    match enqueue_dense_segments(
        lease,
        graph,
        state,
        transaction,
        runs,
        layers,
        host_experts,
        stream,
    ) {
        Ok(()) => Ok(()),
        Err(error) => {
            if mode == DenseStepMode::Capture {
                if lease.resource().open {
                    if let Ok(graph) = stream.end_capture() {
                        drop(graph);
                    }
                    lease.resource_mut().open = false;
                }
                if !lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .captured
                    .is_empty()
                {
                    // Keep the plan and graphs together unless synchronization proves
                    // launches completed.
                    if stream.synchronize().is_ok() {
                        lease
                            .resource_mut()
                            .plan
                            .as_mut()
                            .expect("dense operation retains plan")
                            .captured
                            .clear();
                    }
                }
            }
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enqueue_dense_segments<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    graph: &Graph,
    state: &mut DeviceKvSequence,
    transaction: StateTransactionId,
    runs: &mut [PagedAttentionRun<'ctx>],
    layers: &BTreeMap<NodeId, u32>,
    host_experts: &[HostExpertWeights<'_>],
    stream: &Stream<'ctx>,
) -> Result<()> {
    let rows = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .workload()
        .rows;
    let positions_value = graph.nodes().iter().find_map(|node| match node.params {
        OpParams::Rope { .. } => node.inputs.get(1).copied(),
        _ => None,
    });
    // A tensor-parallel stage without RoPE has no position input.
    let positions = || positions_value.ok_or_else(|| invalid("positions", "no position input"));
    let mut attention_index = 0usize;
    upload_sources(lease, stream)?;
    upload_rope_tables(lease, graph, rows, stream)?;
    let selected_nodes = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .nodes()
        .to_vec();

    for (node, selected) in graph.nodes().iter().zip(selected_nodes.iter()) {
        let plan = lease
            .resource()
            .plan
            .as_ref()
            .expect("dense operation retains plan");
        #[cfg(feature = "cublas")]
        let cublas_node = is_cublas_node(selected);
        #[cfg(not(feature = "cublas"))]
        let cublas_node = false;
        let base = match plan.symbol_indices.get(&node.id).copied() {
            Some(index) => index,
            None if cublas_node => 0,
            None => {
                return Err(invalid(
                    "symbols",
                    "selected node has no module symbol index",
                ));
            }
        };
        match node.params {
            OpParams::Embedding {
                vocab,
                hidden,
                scale,
            } => {
                let token_value = node.inputs[0];
                let token_ids = index_values(lease.resource(), token_value, rows)?;
                if (0..token_ids.len()).any(|index| token_ids.get(index) >= vocab) {
                    return Err(invalid(
                        "token",
                        "token id is outside the embedding vocabulary",
                    ));
                }
                let mut token_address = address(lease.resource(), token_value)?;
                let mut table_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let mut launch_rows = rows;
                let mut launch_vocab = vocab;
                let mut launch_hidden = hidden;
                let mut launch_scale = scale;
                let mut params: [*mut c_void; 7] = [
                    (&raw mut token_address).cast(),
                    (&raw mut table_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_vocab).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_scale).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (rows.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_EMBEDDING,
                )?;
                push_launch(lease, "embedding");
            }
            OpParams::Linear {
                in_features,
                out_features,
                bias: false,
            } => {
                if cublas_node {
                    #[cfg(feature = "cublas")]
                    {
                        match selected.descriptor.operation {
                            SemanticKernelOp::Linear => launch_cublas_linear(
                                lease,
                                stream,
                                rows,
                                in_features,
                                out_features,
                                false,
                                node,
                                selected,
                            )?,
                            SemanticKernelOp::LinearPartial => launch_cublas_linear(
                                lease,
                                stream,
                                rows,
                                in_features,
                                out_features,
                                true,
                                node,
                                selected,
                            )?,
                            SemanticKernelOp::LinearSplit => launch_cublas_split(
                                lease,
                                base,
                                stream,
                                rows,
                                in_features,
                                out_features,
                                node,
                                selected,
                            )?,
                            _ => {
                                return Err(invalid(
                                    "catalogue",
                                    "cuBLAS symbol is attached to an unsupported linear operation",
                                ));
                            }
                        }
                        push_launch(lease, "linear");
                    }
                    #[cfg(not(feature = "cublas"))]
                    return Err(invalid("linear", "the cuBLAS backend is not enabled"));
                } else if let Some(format) = lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .candidate()
                    .weight_formats()
                    .get(&node.inputs[1])
                {
                    let WeightFormat::Affine {
                        width,
                        group,
                        scale,
                        ..
                    } = *format
                    else {
                        return Err(invalid("linear", "a formatted dense weight is not affine"));
                    };
                    let sections = format
                        .sections(out_features, in_features)
                        .ok_or_else(|| invalid("linear", "affine weight sections are invalid"))?;
                    let weight_address = address(lease.resource(), node.inputs[1])?;
                    let section_address = |offset: u64| {
                        weight_address.checked_add(offset).ok_or_else(|| {
                            invalid("linear", "affine weight section address overflowed")
                        })
                    };
                    let mut input_address = address(lease.resource(), node.inputs[0])?;
                    let mut codes_address = section_address(sections.codes.0)?;
                    let mut scales_address = section_address(sections.scales.0)?;
                    let mut zero_points_address = match sections.zero_points {
                        Some((offset, _)) => section_address(offset)?,
                        None => 0,
                    };
                    let mut group_index_address = match sections.group_index {
                        Some((offset, _)) => section_address(offset)?,
                        None => 0,
                    };
                    let mut output_address = address(lease.resource(), node.output)?;
                    let mut launch_rows = rows;
                    let mut launch_input = in_features;
                    let mut launch_output = out_features;
                    let mut row_stride = in_features
                        .checked_mul(u64::from(width.bits()))
                        .ok_or_else(|| invalid("linear", "affine row stride overflowed"))?
                        .div_ceil(8);
                    let mut groups_per_row = in_features
                        .checked_add(u64::from(group) - 1)
                        .ok_or_else(|| invalid("linear", "affine group count overflowed"))?
                        / u64::from(group);
                    let mut code_bits = width.bits();
                    let mut group_size = group;
                    let mut scale_kind = match scale {
                        Precision::F16 => 0,
                        Precision::Bf16 => 1,
                        Precision::F32 => 2,
                        _ => {
                            return Err(invalid("linear", "affine scale precision is unsupported"));
                        }
                    };
                    let mut params: [*mut c_void; 14] = [
                        (&raw mut input_address).cast(),
                        (&raw mut codes_address).cast(),
                        (&raw mut scales_address).cast(),
                        (&raw mut zero_points_address).cast(),
                        (&raw mut group_index_address).cast(),
                        (&raw mut output_address).cast(),
                        (&raw mut launch_rows).cast(),
                        (&raw mut launch_input).cast(),
                        (&raw mut launch_output).cast(),
                        (&raw mut row_stride).cast(),
                        (&raw mut groups_per_row).cast(),
                        (&raw mut code_bits).cast(),
                        (&raw mut group_size).cast(),
                        (&raw mut scale_kind).cast(),
                    ];
                    let tile = moxie_kernels::AFFINE_LINEAR_TILE;
                    let grid = (
                        u32::try_from(out_features.div_ceil(tile))
                            .map_err(|_| invalid("linear", "affine output grid exceeds a u32"))?,
                        u32::try_from(rows.div_ceil(tile))
                            .map_err(|_| invalid("linear", "affine row grid exceeds a u32"))?,
                        1,
                    );
                    launch(
                        lease,
                        base,
                        stream,
                        grid,
                        (32, 1, 1),
                        &mut params,
                        selected,
                        moxie_kernels::AFFINE_LINEAR,
                    )?;
                    push_launch(lease, "affine-linear");
                } else {
                    let mut input_address = address(lease.resource(), node.inputs[0])?;
                    let mut weight_address = address(lease.resource(), node.inputs[1])?;
                    let mut output_address = address(lease.resource(), node.output)?;
                    let mut launch_rows = rows;
                    let mut launch_input = in_features;
                    let mut launch_output = out_features;
                    // Only the split kernel reads the seventh argument, the
                    // declared block count.
                    let (symbol, arguments, mut blocks) = match selected.descriptor.operation {
                        SemanticKernelOp::Linear => (moxie_kernels::BF16_LINEAR, 6, 1),
                        SemanticKernelOp::LinearPartial => {
                            (moxie_kernels::DENSE_LINEAR_PARTIAL, 6, 1)
                        }
                        SemanticKernelOp::LinearSplit => (
                            moxie_kernels::DENSE_LINEAR_SPLIT,
                            7,
                            lease
                                .resource()
                                .plan
                                .as_ref()
                                .expect("dense operation retains plan")
                                .candidate()
                                .linear_orders()
                                .get(&node.id)
                                .map(|order| u64::from(order.blocks))
                                .ok_or_else(|| invalid("linear", "split linear has no order"))?,
                        ),
                        _ => return Err(invalid("linear", "descriptor is not a linear kernel")),
                    };
                    let mut params: [*mut c_void; 7] = [
                        (&raw mut input_address).cast(),
                        (&raw mut weight_address).cast(),
                        (&raw mut output_address).cast(),
                        (&raw mut launch_rows).cast(),
                        (&raw mut launch_input).cast(),
                        (&raw mut launch_output).cast(),
                        (&raw mut blocks).cast(),
                    ];
                    let elements = rows
                        .checked_mul(out_features)
                        .ok_or_else(|| invalid("launch", "linear grid overflowed"))?;
                    launch(
                        lease,
                        base,
                        stream,
                        (elements.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        &mut params[..arguments],
                        selected,
                        symbol,
                    )?;
                    push_launch(lease, "linear");
                }
            }
            OpParams::RmsNorm {
                hidden,
                group: 1,
                eps,
            } => {
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut gain_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let mut workspace_address = workspace_address(lease.resource())?;
                let mut launch_rows = rows;
                let mut launch_hidden = hidden;
                let mut reduce_params: [*mut c_void; 4] = [
                    (&raw mut input_address).cast(),
                    (&raw mut workspace_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_hidden).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (rows.div_ceil(64) as u32, 1, 1),
                    (64, 1, 1),
                    &mut reduce_params,
                    selected,
                    moxie_kernels::BF16_RMS_SUM,
                )?;
                let mut epsilon = eps;
                let mut apply_params: [*mut c_void; 7] = [
                    (&raw mut input_address).cast(),
                    (&raw mut gain_address).cast(),
                    (&raw mut workspace_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut epsilon).cast(),
                ];
                let elements = rows
                    .checked_mul(hidden)
                    .ok_or_else(|| invalid("launch", "RMS grid overflowed"))?;
                launch(
                    lease,
                    base + 1,
                    stream,
                    (elements.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut apply_params,
                    selected,
                    moxie_kernels::BF16_RMS_APPLY,
                )?;
                push_launch(lease, "rms-norm");
            }
            OpParams::RmsNorm { hidden, group, eps } => {
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut gain_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let mut launch_rows = rows;
                let mut launch_hidden = hidden;
                let mut launch_groups = group;
                let mut epsilon = eps;
                let mut params: [*mut c_void; 7] = [
                    (&raw mut input_address).cast(),
                    (&raw mut gain_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_groups).cast(),
                    (&raw mut epsilon).cast(),
                ];
                let groups = rows
                    .checked_mul(group)
                    .ok_or_else(|| invalid("launch", "grouped RMS grid overflowed"))?;
                launch(
                    lease,
                    base,
                    stream,
                    (groups.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_GROUPED_RMS,
                )?;
                push_launch(lease, "grouped-rms-norm");
            }
            OpParams::Rope {
                heads,
                head_dim,
                rotary_dim,
                frequency_dim,
                base: rope_base,
                layout: RopeLayout::HalfSplit,
            } => {
                let key = (rotary_dim, frequency_dim, rope_base.to_bits());
                let offset = lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .candidate()
                    .rope_table_offsets()
                    .get(&key)
                    .copied()
                    .ok_or_else(|| {
                        invalid(
                            "rope_table_offsets",
                            "the selected plan has no table for this RoPE node",
                        )
                    })?;
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut angle_address = workspace_address(lease.resource())?
                    .checked_add(offset)
                    .ok_or_else(|| invalid("workspace", "RoPE table address overflowed"))?;
                let mut output_address = address(lease.resource(), node.output)?;
                let mut launch_rows = rows;
                let mut launch_heads = heads;
                let mut launch_head_dim = head_dim;
                let mut launch_rotary_dim = rotary_dim;
                let mut params: [*mut c_void; 7] = [
                    (&raw mut input_address).cast(),
                    (&raw mut angle_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_heads).cast(),
                    (&raw mut launch_head_dim).cast(),
                    (&raw mut launch_rotary_dim).cast(),
                ];
                let heads_total = rows
                    .checked_mul(heads)
                    .ok_or_else(|| invalid("launch", "RoPE grid overflowed"))?;
                launch(
                    lease,
                    base,
                    stream,
                    (heads_total.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_ROPE,
                )?;
                push_launch(lease, "rope");
            }
            OpParams::Rope { .. } => {
                return Err(invalid(
                    "rope",
                    "the reduced dense device package admits HalfSplit RoPE only",
                ));
            }
            OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
                visibility,
                layer,
            } => {
                close_segment(lease, stream)?;
                execute_attention(
                    lease,
                    node,
                    rows,
                    heads,
                    kv_heads,
                    head_dim,
                    scale,
                    visibility,
                    layers.get(&node.id).copied().unwrap_or(layer),
                    positions()?,
                    state,
                    transaction,
                    runs,
                    &mut attention_index,
                    stream,
                )?;
                push_launch(lease, "attention");
            }
            OpParams::GeGlu { width } => {
                let mut gate_address = address(lease.resource(), node.inputs[0])?;
                let mut up_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let elements = rows
                    .checked_mul(width)
                    .ok_or_else(|| invalid("launch", "GeGLU grid overflowed"))?;
                let mut launch_elements = elements;
                let mut params: [*mut c_void; 4] = [
                    (&raw mut gate_address).cast(),
                    (&raw mut up_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_elements).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (elements.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_GEGLU,
                )?;
                push_launch(lease, "geglu");
            }
            OpParams::Residual { scale: 1.0 } => {
                let mut left_address = address(lease.resource(), node.inputs[0])?;
                let mut right_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let elements = bf16_elements(lease.resource(), node.output)?;
                let mut launch_elements = elements;
                let mut params: [*mut c_void; 4] = [
                    (&raw mut left_address).cast(),
                    (&raw mut right_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_elements).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (elements.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::BF16_RESIDUAL,
                )?;
                push_launch(lease, "residual");
            }
            OpParams::Residual { scale } => {
                let mut left_address = address(lease.resource(), node.inputs[0])?;
                let mut right_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let elements = bf16_elements(lease.resource(), node.output)?;
                let mut launch_elements = elements;
                let mut launch_scale = scale;
                let mut params: [*mut c_void; 5] = [
                    (&raw mut left_address).cast(),
                    (&raw mut right_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_elements).cast(),
                    (&raw mut launch_scale).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (elements.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_RESIDUAL_SCALED,
                )?;
                push_launch(lease, "scaled-residual");
            }
            OpParams::VocabProjection {
                vocab,
                hidden,
                softcap,
            } => {
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut weight_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let elements = rows
                    .checked_mul(vocab)
                    .ok_or_else(|| invalid("launch", "vocabulary grid overflowed"))?;
                let mut launch_rows = rows;
                let mut launch_hidden = hidden;
                let mut launch_vocab = vocab;
                let mut launch_softcap = softcap.unwrap_or(0.0);
                let mut params: [*mut c_void; 7] = [
                    (&raw mut input_address).cast(),
                    (&raw mut weight_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_vocab).cast(),
                    (&raw mut launch_softcap).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (elements.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_VOCAB_PROJECTION,
                )?;
                push_launch(lease, "vocab-projection");
            }
            OpParams::Route {
                hidden,
                experts,
                top_k,
                input: moxie_graph::RouterInput::Normalized { eps, input_scale },
                score: moxie_graph::RouteScore::Softmax,
                per_expert_scale: true,
                selection_bias: false,
                coefficient: moxie_graph::RouteCoefficient::Fp32,
            } => {
                let mut x_address = address(lease.resource(), node.inputs[0])?;
                let mut projection_address = address(lease.resource(), node.inputs[1])?;
                let mut gain_address = address(lease.resource(), node.inputs[2])?;
                let mut per_expert_address = address(lease.resource(), node.inputs[3])?;
                // The route value's arena range holds rows * top_k u32 ids at
                // offset 0, then rows * top_k f32 coefficients at byte offset
                // rows * top_k * 4. The planner charges rows * top_k * 8 bytes;
                // ExpertMlp and Combine read both halves from this one range.
                let mut ids_address = address(lease.resource(), node.output)?;
                let coefficient_offset = rows
                    .checked_mul(top_k)
                    .and_then(|entries| entries.checked_mul(4))
                    .ok_or_else(|| invalid("route", "coefficient offset overflowed"))?;
                let mut coefficients_address = ids_address
                    .checked_add(coefficient_offset)
                    .ok_or_else(|| invalid("route", "coefficient address overflowed"))?;
                let mut launch_rows = rows;
                let mut launch_hidden = hidden;
                let mut launch_experts = experts;
                let mut launch_top_k = top_k;
                let mut launch_eps = eps;
                let mut launch_input_scale = input_scale;
                let mut params: [*mut c_void; 12] = [
                    (&raw mut x_address).cast(),
                    (&raw mut projection_address).cast(),
                    (&raw mut gain_address).cast(),
                    (&raw mut per_expert_address).cast(),
                    (&raw mut ids_address).cast(),
                    (&raw mut coefficients_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_experts).cast(),
                    (&raw mut launch_top_k).cast(),
                    (&raw mut launch_eps).cast(),
                    (&raw mut launch_input_scale).cast(),
                ];
                let blocks = u32::try_from(rows.div_ceil(64))
                    .map_err(|_| invalid("route", "launch grid overflowed"))?;
                launch(
                    lease,
                    base,
                    stream,
                    (blocks, 1, 1),
                    (64, 1, 1),
                    &mut params,
                    selected,
                    moxie_kernels::DENSE_ROUTE,
                )?;
                push_launch(lease, "route");
            }
            OpParams::Route { .. } => {
                return Err(invalid("route", "unsupported route reached the executor"));
            }
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts,
                top_k,
                activation: moxie_graph::ExpertActivation::GeGlu,
                ..
            } => {
                let candidate = lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .candidate();
                let first_expert =
                    candidate
                        .expert_ownership()
                        .get(&node.id)
                        .map_or(Ok(0), |ownership| {
                            u64::from(ownership.owned)
                                .checked_mul(experts)
                                .ok_or_else(|| invalid("expert-mlp", "expert offset overflowed"))
                        })?;
                let assignments = rows
                    .checked_mul(top_k)
                    .ok_or_else(|| invalid("expert-mlp", "assignment count overflowed"))?;
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut ids_address = address(lease.resource(), node.inputs[1])?;
                let mut gate_up_address = address(lease.resource(), node.inputs[2])?;
                let mut activated_address = workspace_address(lease.resource())?;
                let mut launch_assignments = assignments;
                let mut launch_top_k = top_k;
                let mut launch_hidden = hidden;
                let mut launch_intermediate = intermediate;
                let mut launch_first_expert = first_expert;
                let mut launch_local_experts = experts;
                let project_elements = assignments
                    .checked_mul(intermediate)
                    .ok_or_else(|| invalid("expert-mlp", "project grid overflowed"))?;
                let project_blocks = u32::try_from(project_elements.div_ceil(256))
                    .map_err(|_| invalid("expert-mlp", "project launch grid overflowed"))?;
                let mut project_params: [*mut c_void; 10] = [
                    (&raw mut input_address).cast(),
                    (&raw mut ids_address).cast(),
                    (&raw mut gate_up_address).cast(),
                    (&raw mut activated_address).cast(),
                    (&raw mut launch_assignments).cast(),
                    (&raw mut launch_top_k).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_intermediate).cast(),
                    (&raw mut launch_first_expert).cast(),
                    (&raw mut launch_local_experts).cast(),
                ];
                launch(
                    lease,
                    base,
                    stream,
                    (project_blocks, 1, 1),
                    (256, 1, 1),
                    &mut project_params,
                    selected,
                    moxie_kernels::DENSE_EXPERT_PROJECT_GELU,
                )?;

                let mut down_address = address(lease.resource(), node.inputs[3])?;
                let mut slots_address = address(lease.resource(), node.output)?;
                let down_elements = assignments
                    .checked_mul(hidden)
                    .ok_or_else(|| invalid("expert-mlp", "down grid overflowed"))?;
                let down_blocks = u32::try_from(down_elements.div_ceil(256))
                    .map_err(|_| invalid("expert-mlp", "down launch grid overflowed"))?;
                let mut down_params: [*mut c_void; 9] = [
                    (&raw mut activated_address).cast(),
                    (&raw mut ids_address).cast(),
                    (&raw mut down_address).cast(),
                    (&raw mut slots_address).cast(),
                    (&raw mut launch_assignments).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_intermediate).cast(),
                    (&raw mut launch_first_expert).cast(),
                    (&raw mut launch_local_experts).cast(),
                ];
                launch(
                    lease,
                    base + 1,
                    stream,
                    (down_blocks, 1, 1),
                    (256, 1, 1),
                    &mut down_params,
                    selected,
                    moxie_kernels::DENSE_EXPERT_DOWN,
                )?;
                push_launch(lease, "expert-mlp");
            }
            OpParams::ExpertMlp { .. } => {
                return Err(invalid(
                    "expert-mlp",
                    "unsupported expert activation reached the executor",
                ));
            }
            OpParams::Combine {
                hidden,
                top_k,
                order: moxie_graph::CombineOrder::AscendingExpertId,
                output_scale,
            } => {
                if selected.descriptor.operation == SemanticKernelOp::CombineHostJoin {
                    close_segment(lease, stream)?;
                }
                let candidate = lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .candidate();
                let slots_value = node.inputs[1];
                let expert_node = graph
                    .producer(slots_value)
                    .ok_or_else(|| invalid("combine", "expert slots have no producer"))?;
                let OpParams::ExpertMlp {
                    experts,
                    intermediate,
                    ..
                } = expert_node.params
                else {
                    return Err(invalid(
                        "combine",
                        "expert slots are not produced by ExpertMlp",
                    ));
                };
                let ownership_groups = candidate
                    .expert_ownership()
                    .get(&expert_node.id)
                    .map_or(1, |ownership| u64::from(ownership.groups));
                let experts_total = experts
                    .checked_mul(ownership_groups)
                    .ok_or_else(|| invalid("combine", "expert count overflowed"))?;
                let order = candidate.combine_orders().get(&node.id);
                let groups = order.map_or(1, |order| u64::from(order.groups));
                if groups == 0 || !experts_total.is_multiple_of(groups) {
                    return Err(invalid(
                        "combine",
                        "expert count is not divisible by combine groups",
                    ));
                }
                let experts_per_group = experts_total / groups;
                let mut ids_address = address(lease.resource(), node.inputs[0])?;
                let coefficient_offset = rows
                    .checked_mul(top_k)
                    .and_then(|entries| entries.checked_mul(4))
                    .ok_or_else(|| invalid("combine", "coefficient offset overflowed"))?;
                let mut coefficients_address = ids_address
                    .checked_add(coefficient_offset)
                    .ok_or_else(|| invalid("combine", "coefficient address overflowed"))?;
                let mut slots_address = address(lease.resource(), node.inputs[1])?;
                let mut output_address = address(lease.resource(), node.output)?;
                let elements = rows
                    .checked_mul(hidden)
                    .ok_or_else(|| invalid("combine", "launch grid overflowed"))?;
                let blocks = u32::try_from(elements.div_ceil(256))
                    .map_err(|_| invalid("combine", "launch grid overflowed"))?;
                if selected.descriptor.operation == SemanticKernelOp::CombineHostJoin {
                    let join = *candidate
                        .host_expert_joins()
                        .get(&node.id)
                        .expect("planner selects host join only with join metadata");
                    let weights = host_experts
                        .iter()
                        .find(|weights| weights.combine == node.id)
                        .expect("host join weights were validated at step entry");
                    enqueue_host_join(
                        lease,
                        base,
                        stream,
                        selected,
                        rows,
                        hidden,
                        top_k,
                        experts_per_group,
                        intermediate,
                        &join,
                        node.inputs[0],
                        expert_node.inputs[0],
                        node.inputs[1],
                        node.output,
                        ids_address,
                        coefficients_address,
                        slots_address,
                        output_address,
                        blocks,
                        weights,
                    )?;
                    push_launch(lease, "combine-host-join");
                    continue;
                }
                let mut launch_rows = rows;
                let mut launch_top_k = top_k;
                let mut launch_hidden = hidden;
                let mut launch_scale = output_scale;
                let mut launch_experts_per_group = experts_per_group;
                if selected.descriptor.operation == SemanticKernelOp::CombinePartial {
                    let owned = order
                        .and_then(|order| order.owned)
                        .expect("planner selects CombinePartial only for an owned group");
                    launch_combine_partial(
                        lease,
                        base,
                        stream,
                        selected,
                        blocks,
                        [
                            ids_address,
                            coefficients_address,
                            slots_address,
                            output_address,
                        ],
                        [rows, top_k, hidden, experts_per_group, u64::from(owned)],
                    )?;
                    push_launch(lease, "combine-partial");
                } else if selected.descriptor.operation == SemanticKernelOp::Combine {
                    let mut launch_groups = groups;
                    let mut params: [*mut c_void; 10] = [
                        (&raw mut ids_address).cast(),
                        (&raw mut coefficients_address).cast(),
                        (&raw mut slots_address).cast(),
                        (&raw mut output_address).cast(),
                        (&raw mut launch_rows).cast(),
                        (&raw mut launch_top_k).cast(),
                        (&raw mut launch_hidden).cast(),
                        (&raw mut launch_scale).cast(),
                        (&raw mut launch_experts_per_group).cast(),
                        (&raw mut launch_groups).cast(),
                    ];
                    launch(
                        lease,
                        base,
                        stream,
                        (blocks, 1, 1),
                        (256, 1, 1),
                        &mut params,
                        selected,
                        moxie_kernels::DENSE_COMBINE,
                    )?;
                    push_launch(lease, "combine");
                } else {
                    return Err(invalid("combine", "descriptor is not a combine kernel"));
                }
            }
            OpParams::Combine { .. } => {
                return Err(invalid(
                    "combine",
                    "unsupported combine reached the executor",
                ));
            }
            _ => {
                return Err(Error::UnsupportedKernel {
                    operation: node.params.op().name(),
                    detail: "operation reached the dense executor outside its selected package"
                        .into(),
                });
            }
        }
    }
    close_segment(lease, stream)?;
    if lease.resource().mode == DenseStepMode::Replay
        && lease.resource().segment
            != lease
                .resource()
                .plan
                .as_ref()
                .expect("dense operation retains plan")
                .captured
                .len()
    {
        return Err(invalid(
            "capture",
            "replay did not consume every captured segment",
        ));
    }
    // A stage graph's attention nodes each name their layer's run; only a
    // whole graph must use every run.
    if layers.is_empty() && attention_index != runs.len() {
        return Err(invalid(
            "runs",
            "dense graph attention nodes and admitted device runs differ",
        ));
    }
    Ok(())
}

fn close_segment<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let (mode, open, segment) = {
        let operation = lease.resource();
        (operation.mode, operation.open, operation.segment)
    };
    if !open {
        return Ok(());
    }
    match mode {
        DenseStepMode::Eager => {
            return Err(invalid("capture", "an eager step has an open segment"));
        }
        DenseStepMode::Capture => {
            let graph = stream.end_capture();
            lease.resource_mut().open = false;
            let graph = graph?;
            let plan = lease
                .resource_mut()
                .plan
                .as_mut()
                .expect("dense operation retains plan");
            if plan.captured.len() != segment {
                return Err(invalid(
                    "capture",
                    "captured segment index differs from the plan sequence",
                ));
            }
            plan.captured.push(graph);
            // SAFETY: the plan retains the graph, its module, and every
            // admitted buffer until the completion event is observed.
            unsafe {
                plan.captured[segment].launch(stream)?;
            }
        }
        DenseStepMode::Replay => {}
    }
    let operation = lease.resource_mut();
    operation.segment = operation
        .segment
        .checked_add(1)
        .ok_or_else(|| invalid("capture", "segment count overflowed"))?;
    operation.open = false;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_attention<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    node: &moxie_graph::Node,
    rows: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    scale: f32,
    visibility: Visibility,
    layer: u32,
    positions_value: ValueId,
    state: &mut DeviceKvSequence,
    transaction: StateTransactionId,
    runs: &mut [PagedAttentionRun<'ctx>],
    attention_index: &mut usize,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let positions = index_values(lease.resource(), positions_value, rows)?;
    if positions.is_empty() || positions.len() as u64 != rows {
        return Err(invalid(
            "positions",
            "attention position count differs from rows",
        ));
    }
    for index in 1..positions.len() {
        let previous = positions.get(index - 1);
        let current = positions.get(index);
        if current
            != previous.checked_add(1).ok_or_else(|| {
                invalid(
                    "positions",
                    "attention positions overflowed while checking contiguity",
                )
            })?
        {
            return Err(invalid(
                "positions",
                "the dense paged device path requires contiguous absolute positions",
            ));
        }
    }
    let first_position = positions.get(0);
    let run_count = runs.len();
    let run = runs.get_mut(layer as usize).ok_or_else(|| {
        invalid(
            "runs",
            "dense graph has more attention nodes than admitted runs",
        )
    })?;
    *attention_index += 1;
    let key_range = address_range(lease.resource(), node.inputs[1])?;
    let value_range = address_range(lease.resource(), node.inputs[2])?;
    if state.layer_count()? <= layer as usize || run_count != state.layer_count()? {
        return Err(invalid(
            "state",
            "dense attention state and admitted runs do not have one entry per layer",
        ));
    }
    append_paged_layer_from_device(
        state,
        transaction,
        layer as usize,
        rows,
        run,
        stream,
        key_range,
        value_range,
    )?;
    let retained = state.layer_retained(layer as usize)?;
    let geometry = PageGeometry {
        kv_heads,
        head_dim,
        page_tokens: state.geometry()?.page_tokens as u64,
        pages: state.layout(layer as usize)?.pages,
    };
    let launch = PagedAttentionLaunch::new(
        AttentionLayer {
            geometry,
            heads,
            scale,
            visibility,
        },
        rows,
        first_position,
        retained.start,
        retained.end - retained.start,
    )?;
    let last_query = launch
        .first_position()
        .checked_add(launch.rows() - 1)
        .ok_or_else(|| invalid("positions", "attention query position overflowed"))?;
    let visible = match launch.visibility() {
        Visibility::Causal => last_query
            .checked_sub(launch.history_base())
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("visibility", "attention visible extent underflowed"))?,
        Visibility::SlidingWindow { window } => window.min(
            last_query
                .checked_sub(launch.history_base())
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("visibility", "attention visible extent underflowed"))?,
        ),
    };
    let admitted_visible = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .workload()
        .visible_tokens;
    if visible > admitted_visible {
        return Err(invalid(
            "visible_tokens",
            "state history exceeds the admitted attention bucket",
        ));
    }
    let plan = lease
        .resource_mut()
        .plan
        .as_mut()
        .expect("dense operation retains plan");
    let query_value = node.inputs[0];
    let output_value = node.output;
    let (query_key, query_range) = plan.take_range_for_value(query_value)?;
    let (output_key, output_range) = match plan.take_range_for_value(output_value) {
        Ok(range) => range,
        Err(error) => {
            plan.restore_range(query_key, query_range)?;
            return Err(error);
        }
    };
    match run.attend_into_deferred(stream, &launch, query_range, output_range) {
        Ok((query_range, output_range)) => {
            plan.restore_range(query_key, query_range)?;
            plan.restore_range(output_key, output_range)
        }
        Err(refused) => {
            if let Some((query_range, output_range)) = refused.ranges {
                plan.restore_range(query_key, query_range)?;
                plan.restore_range(output_key, output_range)?;
            }
            Err(refused.error)
        }
    }
}

fn upload_sources<'ctx>(
    lease: &OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let operation = lease.resource();
    let plan = operation
        .plan
        .as_ref()
        .expect("dense operation retains plan");
    for source in &operation.sources {
        // SAFETY: the operation owns the source and admitted destination until
        // the completion event retires.
        unsafe {
            plan.range_for_value(source.value)?
                .copy_from_host_async(&source.bytes, stream)?;
        }
    }
    Ok(())
}

fn upload_rope_tables<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    graph: &Graph,
    rows: u64,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let DenseOperation {
        plan,
        sources,
        rope_tables,
        ..
    } = lease.resource_mut();
    let plan = plan.as_ref().expect("dense operation retains plan");
    let offsets = plan.candidate().rope_table_offsets();
    if offsets.is_empty() {
        return Ok(());
    }
    let workspace = plan.workspace_range()?;
    for (key, &offset) in offsets {
        let key = *key;
        let (rotary_dim, frequency_dim, _) = key;
        let Some(node) = graph.nodes().iter().find(|node| {
            matches!(node.params, OpParams::Rope { rotary_dim: dim, frequency_dim: freq, base, .. }
                if (dim, freq, base.to_bits()) == key)
        }) else {
            return Err(invalid(
                "rope_table_offsets",
                "the selected plan contains a RoPE table absent from the graph",
            ));
        };
        let Some(positions_value) = node.inputs.get(1).copied() else {
            return Err(invalid("positions", "a RoPE node has no position input"));
        };
        let OpParams::Rope { base, .. } = node.params else {
            unreachable!("matched RoPE node")
        };
        let positions = index_values_from_sources(sources, positions_value, rows)?;
        let angles = angle_table(positions, rotary_dim, frequency_dim, base)?;
        rope_tables
            .try_reserve(1)
            .map_err(|_| capacity(std::mem::size_of::<((u64, u64, u32), Vec<u8>)>()))?;
        rope_tables.push((key, angles));
        // SAFETY: the operation retains the table until the lease retires
        // after its completion event; the offset and destination are admitted
        // by the selected plan.
        unsafe {
            workspace.copy_from_host_async_at(
                offset,
                &rope_tables.last().expect("just retained").1,
                stream,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    symbol_index: usize,
    stream: &Stream<'ctx>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &mut [*mut c_void],
    node: &SelectedNode,
    symbol: &str,
) -> Result<()> {
    let (mode, open, segment, symbol_matches) = {
        let operation = lease.resource();
        let package = operation
            .plan
            .as_ref()
            .expect("dense operation retains plan")
            .package
            .as_ref()
            .ok_or_else(|| invalid("module", "dense kernel module is not loaded"))?;
        (
            operation.mode,
            operation.open,
            operation.segment,
            package.symbols().get(symbol_index).map(String::as_str) == Some(symbol),
        )
    };
    if !symbol_matches {
        return Err(attribute_node_error(
            invalid(
                "launch",
                "the selected symbol is not the kernel this operation prepared arguments for",
            ),
            lease.resource().device_ordinal,
            node,
            symbol,
        ));
    }

    if mode == DenseStepMode::Replay {
        if open {
            return Ok(());
        }
        let captured = lease
            .resource()
            .plan
            .as_ref()
            .expect("dense operation retains plan")
            .captured
            .get(segment)
            .ok_or_else(|| invalid("capture", "replay has no captured segment"))?;
        // SAFETY: the plan retains the graph, its module, and every admitted
        // buffer until the completion event is observed.
        unsafe {
            captured.launch(stream)?;
        }
        lease.resource_mut().open = true;
        return Ok(());
    }

    if mode == DenseStepMode::Capture && !open {
        stream.begin_capture()?;
        lease.resource_mut().open = true;
    }
    let operation = lease.resource();
    let package = operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .package
        .as_ref()
        .ok_or_else(|| invalid("module", "dense kernel module is not loaded"))?;
    // SAFETY: descriptor selection fixes this ABI, the symbol identity is
    // checked here, and the caller passed only addresses within ranges
    // admitted for this plan.
    unsafe {
        package
            .launch_async(symbol_index, stream, grid, block, 0, params)
            .map_err(|error| attribute_node_error(error, operation.device_ordinal, node, symbol))
    }
}

#[cfg(feature = "cublas")]
#[allow(clippy::too_many_arguments)]
fn launch_cublas_linear<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    stream: &Stream<'ctx>,
    rows: u64,
    in_features: u64,
    out_features: u64,
    c_f32: bool,
    node: &moxie_graph::Node,
    selected: &SelectedNode,
) -> Result<()> {
    if !begin_cublas_work(lease, stream)? {
        return Ok(());
    }
    let plan = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan");
    let input = address(lease.resource(), node.inputs[0])?;
    let weight = address(lease.resource(), node.inputs[1])?;
    let output = address(lease.resource(), node.output)?;
    // Row-major Y = X * W^T is column-major Y^T = W * X^T.
    // SAFETY: the plan owns all three admitted BF16 ranges and the bound
    // workspace; its completion event retains the plan through GEMM completion.
    unsafe {
        plan.blas
            .as_ref()
            .expect("cuBLAS handle was bound before dense submission")
            .gemm_bf16(
                out_features,
                rows,
                in_features,
                weight,
                in_features,
                input,
                in_features,
                output,
                out_features,
                c_f32,
            )
    }
    .map_err(|error| {
        attribute_node_error(
            error,
            lease.resource().device_ordinal,
            selected,
            if c_f32 {
                "cublas:gemm_ex_partial"
            } else {
                "cublas:gemm_ex"
            },
        )
    })
}

#[cfg(feature = "cublas")]
#[allow(clippy::too_many_arguments)]
fn launch_cublas_split<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    symbol_index: usize,
    stream: &Stream<'ctx>,
    rows: u64,
    in_features: u64,
    out_features: u64,
    node: &moxie_graph::Node,
    selected: &SelectedNode,
) -> Result<()> {
    if !begin_cublas_work(lease, stream)? {
        return Ok(());
    }
    let order = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .linear_orders()
        .get(&node.id)
        .copied()
        .ok_or_else(|| invalid("linear", "split linear has no declared reduction order"))?;
    if order.blocks != 2 || in_features & 1 != 0 {
        return Err(invalid(
            "linear",
            "cuBLAS split linear requires exactly two equal blocks",
        ));
    }
    let width = in_features / 2;
    let plan = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan");
    let slots = *plan
        .candidate()
        .linear_split_workspace()
        .ok_or_else(|| invalid("workspace", "split linear has no admitted scratch slots"))?;
    let packed_weight_bytes = out_features
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| invalid("workspace", "packed weight extent overflowed"))?;
    let packed_input_bytes = rows
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| invalid("workspace", "packed input extent overflowed"))?;
    let partial_bytes = rows
        .checked_mul(out_features)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid("workspace", "FP32 partial extent overflowed"))?;
    if packed_weight_bytes > slots.packed_weight_bytes()
        || packed_input_bytes > slots.packed_input_bytes()
        || partial_bytes > slots.partial_bytes()
    {
        return Err(invalid(
            "workspace",
            "split linear exceeds its admitted scratch slots",
        ));
    }
    let workspace = plan.workspace_range()?.device_address()?;
    let slot_address = |offset: u64| {
        workspace
            .checked_add(offset)
            .ok_or_else(|| invalid("workspace", "split scratch address overflowed"))
    };
    let packed_weight = slot_address(slots.packed_weight_offset())?;
    let packed_input = slot_address(slots.packed_input_offset())?;
    let partial_0 = slot_address(slots.partial_0_offset())?;
    let partial_1 = slot_address(slots.partial_1_offset())?;
    let input = address(lease.resource(), node.inputs[0])?;
    let weight = address(lease.resource(), node.inputs[1])?;
    let output = address(lease.resource(), node.output)?;
    let row_bytes = in_features
        .checked_mul(2)
        .ok_or_else(|| invalid("linear", "split source pitch overflowed"))?;
    let packed_row_bytes = width
        .checked_mul(2)
        .ok_or_else(|| invalid("linear", "packed split pitch overflowed"))?;
    let partials = [partial_0, partial_1];
    let ctx = lease.resource().ctx;
    let ordinal = lease.resource().device_ordinal;
    let plan = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan");
    for (block, partial) in partials.into_iter().enumerate() {
        let source_offset = u64::try_from(block)
            .ok()
            .and_then(|block| block.checked_mul(width))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid("linear", "split source offset overflowed"))?;
        let weight_source = weight
            .checked_add(source_offset)
            .ok_or_else(|| invalid("linear", "split weight address overflowed"))?;
        let input_source = input
            .checked_add(source_offset)
            .ok_or_else(|| invalid("linear", "split input address overflowed"))?;
        // SAFETY: source values and admitted packed slots remain live with the
        // plan until the step's completion event is observed.
        unsafe {
            copy_2d_async(
                ctx,
                packed_weight,
                packed_row_bytes,
                weight_source,
                row_bytes,
                packed_row_bytes,
                out_features,
                stream,
            )
        }
        .map_err(|error| attribute_node_error(error, ordinal, selected, "cublas:gemm_ex_split"))?;
        // SAFETY: as above; this input slot is reused only after this block's
        // GEMM has been queued on the same stream.
        unsafe {
            copy_2d_async(
                ctx,
                packed_input,
                packed_row_bytes,
                input_source,
                row_bytes,
                packed_row_bytes,
                rows,
                stream,
            )
        }
        .map_err(|error| attribute_node_error(error, ordinal, selected, "cublas:gemm_ex_split"))?;
        // SAFETY: the two packed BF16 matrices and partial range are admitted
        // workspace/source ranges, and their stream ordering retains them.
        unsafe {
            plan.blas
                .as_ref()
                .expect("cuBLAS handle was bound before dense submission")
                .gemm_bf16(
                    out_features,
                    rows,
                    width,
                    packed_weight,
                    width,
                    packed_input,
                    width,
                    partial,
                    out_features,
                    true,
                )
        }
        .map_err(|error| attribute_node_error(error, ordinal, selected, "cublas:gemm_ex_split"))?;
    }
    let mut rank_zero = partial_0;
    let mut rank_one = partial_1;
    let mut output = output;
    let mut elements = rows
        .checked_mul(out_features)
        .ok_or_else(|| invalid("launch", "split reduction extent overflowed"))?;
    let grid = u32::try_from(elements.div_ceil(256))
        .map_err(|_| invalid("launch", "split reduction grid exceeds u32"))?;
    let mut params: [*mut c_void; 4] = [
        (&raw mut rank_zero).cast(),
        (&raw mut rank_one).cast(),
        (&raw mut output).cast(),
        (&raw mut elements).cast(),
    ];
    launch(
        lease,
        symbol_index,
        stream,
        (grid, 1, 1),
        (256, 1, 1),
        &mut params,
        selected,
        moxie_kernels::TP_REDUCE_F32,
    )
}

#[cfg(feature = "cublas")]
fn begin_cublas_work<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    stream: &Stream<'ctx>,
) -> Result<bool> {
    let (mode, open, segment) = {
        let operation = lease.resource();
        (operation.mode, operation.open, operation.segment)
    };
    if mode == DenseStepMode::Replay {
        if open {
            return Ok(false);
        }
        let captured = lease
            .resource()
            .plan
            .as_ref()
            .expect("dense operation retains plan")
            .captured
            .get(segment)
            .ok_or_else(|| invalid("capture", "replay has no captured segment"))?;
        // SAFETY: the plan retains graph, module, and all captured buffers
        // through the completion event recorded by this dense step.
        unsafe { captured.launch(stream)? };
        lease.resource_mut().open = true;
        return Ok(false);
    }
    if mode == DenseStepMode::Capture && !open {
        stream.begin_capture()?;
        lease.resource_mut().open = true;
    }
    Ok(true)
}

fn push_launch<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    name: &str,
) {
    lease.resource_mut().launch_order.push(name.into());
}

#[allow(clippy::too_many_arguments)]
fn launch_combine_partial<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    base: usize,
    stream: &Stream<'ctx>,
    selected: &SelectedNode,
    blocks: u32,
    addresses: [u64; 4],
    values: [u64; 5],
) -> Result<()> {
    let mut args = [
        addresses[0],
        addresses[1],
        addresses[2],
        addresses[3],
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
    ];
    let mut params: [*mut c_void; 9] = std::array::from_fn(|index| (&raw mut args[index]).cast());
    launch(
        lease,
        base,
        stream,
        (blocks, 1, 1),
        (256, 1, 1),
        &mut params,
        selected,
        moxie_kernels::DENSE_COMBINE_PARTIAL,
    )
}

fn validate_host_expert_weights(
    graph: &Graph,
    joins: &BTreeMap<NodeId, HostExpertJoin>,
    host_experts: &[HostExpertWeights<'_>],
) -> Result<()> {
    if host_experts.len() != joins.len() {
        return Err(invalid(
            "host_experts",
            "each selected host join needs exactly one weight entry",
        ));
    }
    for (index, weights) in host_experts.iter().enumerate() {
        let Some(join) = joins.get(&weights.combine) else {
            return Err(invalid(
                "host_experts",
                "host weights name no selected join",
            ));
        };
        if host_experts[..index]
            .iter()
            .any(|previous| previous.combine == weights.combine)
        {
            return Err(invalid(
                "host_experts",
                "a selected host join has duplicate weight entries",
            ));
        }
        let combine = graph
            .nodes()
            .get(weights.combine.0 as usize)
            .filter(|node| node.id == weights.combine)
            .ok_or_else(|| invalid("host_experts", "host join node is absent"))?;
        let expert = combine
            .inputs
            .get(1)
            .and_then(|slots| graph.producer(*slots))
            .ok_or_else(|| invalid("host_experts", "host join has no ExpertMlp producer"))?;
        let moxie_graph::OpParams::ExpertMlp {
            hidden,
            intermediate,
            ..
        } = expert.params
        else {
            return Err(invalid(
                "host_experts",
                "host join slots are not produced by ExpertMlp",
            ));
        };
        let expected = |widths: &[u64]| -> Result<usize> {
            let bytes = widths
                .iter()
                .try_fold(1u64, |size, width| size.checked_mul(*width))
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| invalid("host_experts", "host weight extent overflowed"))?;
            Ok(bytes)
        };
        let host_count = u64::from(join.host_experts());
        let gate_up_bytes = expected(&[host_count, 2, intermediate, hidden])?;
        let down_bytes = expected(&[host_count, hidden, intermediate])?;
        if weights.gate_up.len() != gate_up_bytes || weights.down.len() != down_bytes {
            return Err(invalid(
                "host_experts",
                "host BF16 expert weights do not match the selected join extent",
            ));
        }
    }
    Ok(())
}

struct HostJoinExtents {
    rows: usize,
    hidden: usize,
    top_k: usize,
    entries: usize,
    route_words: usize,
    route_bytes: usize,
    input_bytes: usize,
    slot_bytes: usize,
    partial_bytes: usize,
    workspace_bytes: usize,
    output_bytes: usize,
    elements: u64,
    hidden_u32: u32,
    intermediate_u32: u32,
    workspace_floats: usize,
    gate_up_expert_bytes: usize,
    down_expert_bytes: usize,
}

fn host_join_extents(
    rows: u64,
    hidden: u64,
    top_k: u64,
    intermediate: u64,
) -> Result<HostJoinExtents> {
    let invalid_extent = || {
        invalid(
            "host_experts",
            "host join extent overflowed or is not addressable",
        )
    };
    let fit = |value| usize::try_from(value).map_err(|_| invalid_extent());
    let mul = |left: usize, right: usize| left.checked_mul(right).ok_or_else(invalid_extent);
    let (rows, hidden, top_k, intermediate) =
        (fit(rows)?, fit(hidden)?, fit(top_k)?, fit(intermediate)?);
    let (entries, elements) = (mul(rows, top_k)?, mul(rows, hidden)?);
    let (route_words, route_bytes) = (mul(entries, 2)?, mul(entries, 8)?);
    let (input_bytes, slot_bytes) = (mul(elements, 2)?, mul(mul(entries, hidden)?, 2)?);
    let partial_bytes = mul(elements, 4)?;
    let workspace_bytes = mul(partial_bytes, 2)?;
    let output_bytes = input_bytes;
    let (hidden_u32, intermediate_u32) = (
        u32::try_from(hidden).map_err(|_| invalid_extent())?,
        u32::try_from(intermediate).map_err(|_| invalid_extent())?,
    );
    let shape = ExpertShape {
        hidden: hidden_u32,
        intermediate: intermediate_u32,
    };
    let workspace_floats = shape
        .workspace_f32(ExpertTiling::lanes(intermediate_u32))
        .ok_or_else(invalid_extent)?;
    let gate_up_expert_bytes = mul(mul(intermediate, hidden)?, 4)?;
    let down_expert_bytes = mul(mul(hidden, intermediate)?, 2)?;
    Ok(HostJoinExtents {
        rows,
        hidden,
        top_k,
        entries,
        route_words,
        route_bytes,
        input_bytes,
        slot_bytes,
        partial_bytes,
        workspace_bytes,
        output_bytes,
        elements: u64::try_from(elements).map_err(|_| invalid_extent())?,
        hidden_u32,
        intermediate_u32,
        workspace_floats,
        gate_up_expert_bytes,
        down_expert_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn enqueue_host_join<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    base: usize,
    stream: &Stream<'ctx>,
    selected: &SelectedNode,
    rows: u64,
    hidden: u64,
    top_k: u64,
    experts_per_group: u64,
    intermediate: u64,
    join: &HostExpertJoin,
    route_value: ValueId,
    input_value: ValueId,
    slots_value: ValueId,
    output_value: ValueId,
    ids_address: u64,
    coefficients_address: u64,
    slots_address: u64,
    output_address: u64,
    blocks: u32,
    weights: &HostExpertWeights<'_>,
) -> Result<()> {
    let extents = host_join_extents(rows, hidden, top_k, intermediate)?;
    let workspace_bytes = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .workspace_range()?
        .bytes();
    if workspace_bytes
        < u64::try_from(extents.workspace_bytes)
            .map_err(|_| invalid("host_experts", "workspace extent exceeds u64"))?
    {
        return Err(invalid(
            "host_experts",
            "admitted device workspace is smaller than the host join extent",
        ));
    }
    for (value, required) in [
        (route_value, extents.route_bytes),
        (input_value, extents.input_bytes),
        (slots_value, extents.slot_bytes),
        (output_value, extents.output_bytes),
    ] {
        if address_range(lease.resource(), value)?.bytes()
            < u64::try_from(required)
                .map_err(|_| invalid("host_experts", "host extent exceeds u64"))?
        {
            return Err(invalid(
                "host_experts",
                "graph range is smaller than its admitted host join extent",
            ));
        }
    }

    // Group zero is the device partial A. The synchronous stream drain below
    // makes both its bytes and the route/input readbacks safe to consume.
    let partial_a = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .workspace_range()?
        .device_address()?;
    launch_combine_partial(
        lease,
        base,
        stream,
        selected,
        blocks,
        [ids_address, coefficients_address, slots_address, partial_a],
        [rows, top_k, hidden, experts_per_group, 0],
    )?;
    stream.synchronize()?;

    let mut route = host_zeroed::<u32>(extents.route_words)?;
    let mut input = host_zeroed::<u8>(extents.input_bytes)?;
    let mut slots = host_zeroed::<u8>(extents.slot_bytes)?;
    let mut host_partial = host_zeroed::<u8>(extents.partial_bytes)?;
    {
        let operation = lease.resource();
        address_range(operation, route_value)?.copy_to_host(u32_bytes_mut(&mut route))?;
        address_range(operation, input_value)?.copy_to_host(&mut input)?;
    }

    let shape = ExpertShape {
        hidden: extents.hidden_u32,
        intermediate: extents.intermediate_u32,
    };
    let tiling = ExpertTiling::lanes(extents.intermediate_u32);
    let mut cpu_workspace = host_zeroed::<f32>(extents.workspace_floats)?;
    let host_count = usize::try_from(join.host_experts())
        .map_err(|_| invalid("host_experts", "host expert count is not addressable"))?;
    let first_host = join.first_host_expert();
    let mut route_dirty = false;
    for local_expert in 0..host_count {
        if route_dirty {
            address_range(lease.resource(), route_value)?
                .copy_to_host(u32_bytes_mut(&mut route))?;
            route_dirty = false;
        }
        let expert_id = first_host
            .checked_add(local_expert as u32)
            .ok_or_else(|| invalid("host_experts", "host expert id overflowed"))?;
        let assigned = route[..extents.entries]
            .iter()
            .filter(|id| u32::from_le(**id) == expert_id)
            .count();
        if assigned == 0 {
            continue;
        }
        let mut seen = 0usize;
        for index in (0..extents.entries).rev() {
            if u32::from_le(route[index]) != expert_id {
                continue;
            }
            let destination = extents.entries - 1 - seen;
            let row = u32::try_from(index as u64 / top_k)
                .map_err(|_| invalid("host_experts", "row index exceeds CPU expert ABI"))?;
            let slot = u32::try_from(index)
                .map_err(|_| invalid("host_experts", "slot index exceeds CPU expert ABI"))?;
            route[destination] = row;
            route[extents.entries + destination] = slot;
            seen += 1;
        }
        let first = extents.entries - assigned;
        let gate_start = local_expert
            .checked_mul(extents.gate_up_expert_bytes)
            .ok_or_else(|| invalid("host_experts", "gate/up weight offset overflowed"))?;
        let down_start = local_expert
            .checked_mul(extents.down_expert_bytes)
            .ok_or_else(|| invalid("host_experts", "down weight offset overflowed"))?;
        let gate_up = weights
            .gate_up
            .get(gate_start..gate_start + extents.gate_up_expert_bytes)
            .ok_or_else(|| invalid("host_experts", "gate/up weight slice is absent"))?;
        let down = weights
            .down
            .get(down_start..down_start + extents.down_expert_bytes)
            .ok_or_else(|| invalid("host_experts", "down weight slice is absent"))?;
        moxie_kernels::cpu_expert::expert_group_bf16(
            &input,
            ExpertAssignment {
                rows: &route[first..extents.entries],
                slots: &route[extents.entries + first..],
            },
            gate_up,
            down,
            GateTransform::GeluTanh,
            shape,
            tiling,
            &mut cpu_workspace,
            &mut slots,
        )?;
        route_dirty = true;
    }
    if route_dirty {
        address_range(lease.resource(), route_value)?.copy_to_host(u32_bytes_mut(&mut route))?;
    }

    for row in 0..extents.rows {
        for column in 0..extents.hidden {
            let mut accumulated = 0.0f32;
            let mut previous: Option<(u32, usize)> = None;
            for _ in 0..extents.top_k {
                let mut next: Option<(u32, usize)> = None;
                for slot in 0..extents.top_k {
                    let index = row * extents.top_k + slot;
                    let expert = u32::from_le(route[index]);
                    let pair = (expert, slot);
                    if expert >= first_host
                        && previous.is_none_or(|prior| pair > prior)
                        && next.is_none_or(|current| pair < current)
                    {
                        next = Some(pair);
                    }
                }
                let Some((expert, slot)) = next else {
                    break;
                };
                let route_index = row * extents.top_k + slot;
                let coefficient =
                    f32::from_bits(u32::from_le(route[extents.entries + route_index]));
                let slot_index = route_index
                    .checked_mul(extents.hidden)
                    .and_then(|offset| offset.checked_add(column))
                    .ok_or_else(|| invalid("host_experts", "host slot offset overflowed"))?;
                let byte_offset = slot_index
                    .checked_mul(2)
                    .ok_or_else(|| invalid("host_experts", "host slot byte offset overflowed"))?;
                let value = f32::from_bits(
                    (u16::from_le_bytes([slots[byte_offset], slots[byte_offset + 1]]) as u32) << 16,
                );
                accumulated += coefficient * value;
                previous = Some((expert, slot));
            }
            let output_index = row
                .checked_mul(extents.hidden)
                .and_then(|offset| offset.checked_add(column))
                .ok_or_else(|| invalid("host_experts", "host partial offset overflowed"))?;
            host_partial[output_index * 4..output_index * 4 + 4]
                .copy_from_slice(&accumulated.to_le_bytes());
        }
    }

    let partial_offset = u64::try_from(extents.partial_bytes)
        .map_err(|_| invalid("workspace", "host partial extent exceeds u64"))?;
    lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .workspace_range()?
        .copy_from_host_at(partial_offset, &host_partial)?;
    let mut rank_zero = partial_a;
    let mut rank_one = partial_a
        .checked_add(partial_offset)
        .ok_or_else(|| invalid("workspace", "second partial address overflowed"))?;
    let mut output = output_address;
    let mut elements = extents.elements;
    let mut reduce_params: [*mut c_void; 4] = [
        (&raw mut rank_zero).cast(),
        (&raw mut rank_one).cast(),
        (&raw mut output).cast(),
        (&raw mut elements).cast(),
    ];
    launch(
        lease,
        base + 1,
        stream,
        (blocks, 1, 1),
        (256, 1, 1),
        &mut reduce_params,
        selected,
        moxie_kernels::TP_REDUCE_F32,
    )?;
    Ok(())
}

fn host_zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| invalid("host_workspace", "host allocation extent overflowed"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| capacity(bytes))?;
    if values.capacity() != len {
        return Err(capacity(bytes));
    }
    values.resize(len, T::default());
    Ok(values)
}

fn u32_bytes_mut(words: &mut [u32]) -> &mut [u8] {
    let bytes = std::mem::size_of_val(words);
    // SAFETY: `u32` has no padding, every element is initialized, and the
    // returned byte slice has exactly the same allocation and lifetime.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), bytes) }
}

fn address<'ctx>(operation: &DenseOperation<'ctx>, value: ValueId) -> Result<u64> {
    operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .value_address(value)
}

fn address_range<'op, 'ctx>(
    operation: &'op DenseOperation<'ctx>,
    value: ValueId,
) -> Result<&'op crate::DeviceRange<'ctx>> {
    operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .range_for_value(value)
}

fn workspace_address<'ctx>(operation: &DenseOperation<'ctx>) -> Result<u64> {
    operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .workspace_range()
        .and_then(|range| range.device_address())
}

fn bf16_elements<'ctx>(operation: &DenseOperation<'ctx>, value: ValueId) -> Result<u64> {
    let bytes = operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .value(value)
        .ok_or_else(|| invalid("value", "dense output is absent from the selected plan"))?
        .logical_bytes;
    if !bytes.is_multiple_of(2) {
        return Err(invalid("value", "BF16 output has an odd byte extent"));
    }
    Ok(bytes / 2)
}

#[derive(Clone, Copy, Debug)]
struct IndexView<'a> {
    bytes: &'a [u8],
}

impl IndexView<'_> {
    fn len(self) -> usize {
        self.bytes.len() / 8
    }

    fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    fn get(self, index: usize) -> u64 {
        let start = index * 8;
        u64::from_le_bytes(
            self.bytes[start..start + 8]
                .try_into()
                .expect("validated index word"),
        )
    }
}

fn index_values<'a, 'ctx>(
    operation: &'a DenseOperation<'ctx>,
    value: ValueId,
    rows: u64,
) -> Result<IndexView<'a>> {
    index_values_from_sources(&operation.sources, value, rows)
}

fn index_values_from_sources<'a>(
    sources: &'a [OwnedBinding],
    value: ValueId,
    rows: u64,
) -> Result<IndexView<'a>> {
    let source = sources
        .iter()
        .find(|binding| binding.value == value)
        .ok_or_else(|| {
            invalid(
                "bindings",
                "dense index input was not retained for the step",
            )
        })?;
    if !matches!(source.role, ValueRole::Index(_)) {
        return Err(invalid(
            "bindings",
            "dense index operand has a non-index role",
        ));
    }
    let expected = rows
        .checked_mul(8)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| invalid("index", "dense index extent is not addressable"))?;
    if source.bytes.len() != expected {
        return Err(invalid(
            "index",
            "dense index byte extent differs from rows",
        ));
    }
    Ok(IndexView {
        bytes: &source.bytes,
    })
}

fn angle_table(
    positions: IndexView<'_>,
    rotary_dim: u64,
    frequency_dim: u64,
    base: f32,
) -> Result<Vec<u8>> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || frequency_dim == 0 {
        return Err(invalid(
            "rope",
            "dense RoPE dimensions are not valid pair geometry",
        ));
    }
    if !(base.is_finite() && base > 1.0) {
        return Err(invalid(
            "rope_base",
            "dense RoPE base must be finite and > 1",
        ));
    }
    let pairs = rotary_dim / 2;
    let bytes = positions
        .len()
        .checked_mul(pairs as usize)
        .and_then(|values| values.checked_mul(8))
        .ok_or_else(|| invalid("rope", "dense angle table extent overflowed"))?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(bytes)
        .map_err(|_| capacity(bytes))?;
    table.resize(bytes, 0);
    for row in 0..positions.len() {
        let position = positions.get(row);
        for pair in 0..pairs as usize {
            let inverse = (base as f64).powf(-2.0 * pair as f64 / frequency_dim as f64);
            let theta = position as f64 * inverse;
            let offset = (row * pairs as usize + pair) * 8;
            table[offset..offset + 4].copy_from_slice(&(theta.cos() as f32).to_le_bytes());
            table[offset + 4..offset + 8].copy_from_slice(&(theta.sin() as f32).to_le_bytes());
        }
    }
    Ok(table)
}

fn dense_attribution(candidate: &moxie_plan::SelectedPlanCandidate, boundary: &str) -> String {
    let kernels = candidate
        .nodes()
        .iter()
        .map(|node| format!("node {} kernel {}", node.node.0, node.descriptor.id.0))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{boundary} for [{kernels}]")
}

fn capacity(bytes: usize) -> Error {
    Error::CapacityExceeded {
        tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
        requested_bytes: bytes as u64,
        available_bytes: 0,
    }
}

#[cfg(feature = "cublas")]
fn is_cublas_node(node: &SelectedNode) -> bool {
    node.descriptor
        .symbols
        .iter()
        .any(|symbol| symbol.0.starts_with("cublas:"))
}

#[cfg(feature = "cublas")]
fn validate_cublas_descriptors(candidate: &moxie_plan::SelectedPlanCandidate) -> Result<()> {
    for node in candidate.nodes().iter().filter(|node| is_cublas_node(node)) {
        let expected_symbols: &[&str] = match node.descriptor.operation {
            SemanticKernelOp::Linear => &["cublas:gemm_ex"],
            SemanticKernelOp::LinearPartial => &["cublas:gemm_ex_partial"],
            SemanticKernelOp::LinearSplit => {
                &["cublas:gemm_ex_split", moxie_kernels::TP_REDUCE_F32]
            }
            _ => &[],
        };
        if node.descriptor.abi_version != moxie_kernels::DENSE_GRAPH_ABI
            || !node
                .descriptor
                .symbols
                .iter()
                .map(|symbol| symbol.0.as_str())
                .eq(expected_symbols.iter().copied())
            || node.descriptor.image_sha256 != moxie_kernels::cublas_sha256()
            || node.descriptor.accumulation != AccumulationPolicy::Bf16InF32AccUnordered
        {
            return Err(invalid(
                "catalogue",
                "cuBLAS operation, symbols, accumulation, ABI, or image identity changed",
            ));
        }
    }
    Ok(())
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;

    use moxie_graph::RopeLayout;
    use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
    use moxie_oracles::rope::{Rotation, rope_head};
    use moxie_types::{DeviceTier, RankId, Scope, Tier};

    use crate::arena::DeviceArena;

    use super::{IndexView, Module, ModuleImage, RankContext, Stream, TrustedImage, angle_table};

    #[test]
    fn uploaded_angles_match_the_host_rope_oracle() {
        let positions: [u64; 3] = [0, 7, 123];
        let rotary_dim = 8;
        let frequency_dim = 16;
        let base = 10_000.0;
        let position_bytes: Vec<u8> = positions
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect();
        let table = angle_table(
            IndexView {
                bytes: &position_bytes,
            },
            rotary_dim,
            frequency_dim,
            base,
        )
        .unwrap();
        let pairs = rotary_dim / 2;
        let rotation = Rotation {
            base,
            rotary_dim: rotary_dim as usize,
            frequency_dim: frequency_dim as usize,
            layout: RopeLayout::HalfSplit,
        };

        for (row, position) in positions.iter().copied().enumerate() {
            for pair in 0..pairs as usize {
                let mut basis = vec![0.0; rotary_dim as usize];
                basis[pair] = 1.0;
                let expected = rope_head(&basis, position, rotation).unwrap();
                let offset = (row * pairs as usize + pair) * 8;
                let cos = f32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                let sin = f32::from_le_bytes(table[offset + 4..offset + 8].try_into().unwrap());
                assert_eq!(cos.to_bits(), expected[pair].to_bits());
                assert_eq!(
                    sin.to_bits(),
                    expected[pair + rotary_dim as usize / 2].to_bits()
                );
            }
        }
    }

    #[test]
    fn combine_kernel_sums_in_ascending_expert_order() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let context = RankContext::acquire(RankId(65_012), 0).unwrap();
        let stream = Stream::new(&context).unwrap();
        let scope = Scope::Device(context.uuid());
        let capacity = 1 << 20;
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, capacity, 4096).unwrap(),
            CapacitySnapshot::new(scope, capacity, 0).unwrap(),
        ])
        .unwrap();
        let mut request = PlanRequest::new("dense combine order test", ["combine"]).unwrap();
        request
            .buffer(
                BufferRequest::try_new(
                    "combine inputs and output",
                    scope,
                    Tier::Device(DeviceTier::Activations),
                    256,
                    StageSpan { first: 0, last: 0 },
                )
                .unwrap(),
            )
            .unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut arena = DeviceArena::create(
            &ledger,
            reservation,
            &context,
            DeviceTier::Activations,
            256,
            "dense combine order test",
        )
        .unwrap();
        let range = arena.allocate(256, 256, "combine fixture").unwrap();

        const COEFFICIENTS: u64 = 16;
        const SLOTS: u64 = 32;
        const OUTPUT: u64 = 40;
        let mut upload = [0; 48];
        for (slot, id) in [2u32, 0, 1].into_iter().enumerate() {
            upload[slot * 4..slot * 4 + 4].copy_from_slice(&id.to_le_bytes());
            upload[COEFFICIENTS as usize + slot * 4..COEFFICIENTS as usize + slot * 4 + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());
        }
        for (slot, value) in [-1.0f32, 1.0, 2.0f32.powi(-30)].into_iter().enumerate() {
            upload[SLOTS as usize + slot * 2..SLOTS as usize + slot * 2 + 2]
                .copy_from_slice(&moxie_kernels::cpu_expert::to_bf16_bits(value).to_le_bytes());
        }
        // SAFETY: `upload` and the admitted destination range stay alive until the stream sync.
        unsafe { range.copy_from_host_async_at(0, &upload, &stream).unwrap() };
        stream.synchronize().unwrap();

        // SAFETY: the image is this build's pinned nvcc output.
        let image =
            unsafe { TrustedImage::from_build_output(moxie_kernels::DENSE_GRAPH_FATBIN).unwrap() };
        let module = Module::load(&context, ModuleImage::Binary(image))
            .unwrap()
            .resolve_all(&[
                moxie_kernels::DENSE_COMBINE.to_string(),
                moxie_kernels::DENSE_COMBINE_PARTIAL.to_string(),
            ])
            .unwrap();
        let mut ids = range.device_address().unwrap();
        let mut coefficients = ids + COEFFICIENTS;
        let mut slots = ids + SLOTS;
        let mut output = ids + OUTPUT;
        let mut rows = 1u64;
        let mut top_k = 3u64;
        let mut hidden = 1u64;
        let mut output_scale = 1.0f32;
        let mut experts_per_group = 4u64;
        let mut groups = 1u64;
        let mut params: [*mut c_void; 10] = [
            (&raw mut ids).cast(),
            (&raw mut coefficients).cast(),
            (&raw mut slots).cast(),
            (&raw mut output).cast(),
            (&raw mut rows).cast(),
            (&raw mut top_k).cast(),
            (&raw mut hidden).cast(),
            (&raw mut output_scale).cast(),
            (&raw mut experts_per_group).cast(),
            (&raw mut groups).cast(),
        ];
        // SAFETY: these addresses and dimensions match the Combine ABI and fit the admitted range.
        unsafe {
            module
                .launch_async(0, &stream, (1, 1, 1), (256, 1, 1), 0, &mut params)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut actual = [0; 2];
        range.copy_to_host_at(OUTPUT, &mut actual).unwrap();
        // Ascending ids yield 1 + 2^-30 -> 1, then -1 -> 0.
        // Selection order yields -1 + 1 -> 0, then BF16 2^-30.
        assert_eq!(
            u16::from_le_bytes(actual),
            moxie_kernels::cpu_expert::to_bf16_bits(0.0)
        );

        upload = [0; 48];
        for (slot, id) in [3u32, 0, 2].into_iter().enumerate() {
            upload[slot * 4..slot * 4 + 4].copy_from_slice(&id.to_le_bytes());
            upload[COEFFICIENTS as usize + slot * 4..COEFFICIENTS as usize + slot * 4 + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());
        }
        for (slot, value) in [2.0f32.powi(-30), -1.0, 1.0].into_iter().enumerate() {
            upload[SLOTS as usize + slot * 2..SLOTS as usize + slot * 2 + 2]
                .copy_from_slice(&moxie_kernels::cpu_expert::to_bf16_bits(value).to_le_bytes());
        }
        // SAFETY: the same admitted input range stays alive through the next stream sync.
        unsafe { range.copy_from_host_async_at(0, &upload, &stream).unwrap() };
        stream.synchronize().unwrap();
        let mut grouped_experts_per_group = 2u64;
        let mut grouped_groups = 2u64;
        let mut grouped_params: [*mut c_void; 10] = [
            (&raw mut ids).cast(),
            (&raw mut coefficients).cast(),
            (&raw mut slots).cast(),
            (&raw mut output).cast(),
            (&raw mut rows).cast(),
            (&raw mut top_k).cast(),
            (&raw mut hidden).cast(),
            (&raw mut output_scale).cast(),
            (&raw mut grouped_experts_per_group).cast(),
            (&raw mut grouped_groups).cast(),
        ];
        // SAFETY: these addresses and dimensions match the grouped Combine ABI and admitted range.
        unsafe {
            module
                .launch_async(0, &stream, (1, 1, 1), (256, 1, 1), 0, &mut grouped_params)
                .unwrap();
        }
        stream.synchronize().unwrap();
        range.copy_to_host_at(OUTPUT, &mut actual).unwrap();
        // Ungrouped: -1 + 1 = 0, then +2^-30 gives 2^-30.
        // Grouped: group 0 is -1; group 1 is 1 + 2^-30 = 1 (absorbed), total 0.
        assert_eq!(
            u16::from_le_bytes(actual),
            moxie_kernels::cpu_expert::to_bf16_bits(0.0)
        );

        let mut partial_experts_per_group = 2u64;
        let mut owned = 0u64;
        let mut partial_params: [*mut c_void; 9] = [
            (&raw mut ids).cast(),
            (&raw mut coefficients).cast(),
            (&raw mut slots).cast(),
            (&raw mut output).cast(),
            (&raw mut rows).cast(),
            (&raw mut top_k).cast(),
            (&raw mut hidden).cast(),
            (&raw mut partial_experts_per_group).cast(),
            (&raw mut owned).cast(),
        ];
        // SAFETY: these addresses and dimensions match the partial Combine ABI and admitted range.
        unsafe {
            module
                .launch_async(1, &stream, (1, 1, 1), (256, 1, 1), 0, &mut partial_params)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut actual_partial = [0; 4];
        range.copy_to_host_at(OUTPUT, &mut actual_partial).unwrap();
        // Owned group 0 visits only expert 0, yielding exactly -1.0.
        // Without the owner filter, ascending ids sum -1 + 1 + 2^-30 = 2^-30.
        assert_eq!(
            u32::from_le_bytes(actual_partial),
            (-1.0f32).to_bits(),
            "partial combine includes only the owned expert group"
        );

        arena.release(range).unwrap();
        drop(module);
        arena.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}
