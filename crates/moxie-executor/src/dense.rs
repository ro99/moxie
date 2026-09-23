//! Device execution of the selected reduced dense graph.
//!
//! This is a graph executor, not a model loop.  The planner owns the checked
//! node order and arena; this module only binds those values to the shared
//! semantic kernels and hands conventional KV state to the existing paged
//! attention authority.

#![cfg(all(feature = "driver", feature = "paged-attention-binding"))]

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

use moxie_cuda::{Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage};
use moxie_graph::{Graph, NodeId, OpParams, RopeLayout, ValueId, ValueRole};
use moxie_plan::{SelectedNode, Visibility};
use moxie_state::DeviceKvSequence;
use moxie_types::{
    DeviceCapability, Error, KernelCatalogue, Precision, Result, SemanticKernelOp,
    StateTransactionId,
};

use crate::arena::{OperationLease, OperationRetireRefused};
use crate::chain::{
    OwnedBinding, SelectedCompletion, SelectedReservedPlan, attribute_chain_error,
    attribute_node_error, validate_bindings_except,
};
use crate::paged_attention::device::{PagedAttentionRun, PagedKvRows, append_paged_layer};
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

impl DensePlanRunRefused<'_> {
    /// Bytes of a host upload still retained after an asynchronous submission
    /// error.  This is intentionally observable so the driver-boundary test can
    /// prove the source was not dropped while the copy's completion is unknown.
    pub fn retained_host_upload_bytes(&self) -> Option<usize> {
        self.held
            .as_ref()
            .and_then(|lease| lease.resource().pending_host_upload.as_ref().map(Vec::len))
    }
}

#[derive(Debug)]
pub struct DenseOperation<'ctx> {
    pub(crate) plan: Option<SelectedReservedPlan<'ctx>>,
    /// Every source copied by `upload_sources` stays here until the graph's
    /// completion event is observed. This includes token and position indices.
    pub(crate) sources: Vec<OwnedBinding>,
    /// The current angle-table source remains here until its copy's stream
    /// synchronization succeeds. On either copy or synchronization failure the
    /// operation lease retains it for quarantine.
    pending_host_upload: Option<Vec<u8>>,
    pub(crate) package: ResolvedModule<'ctx>,
    pub(crate) launch_order: Vec<String>,
    pub(crate) device_ordinal: u32,
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
        self,
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
        } = step;
        let reject = |plan, bindings, error| DensePlanRunRefused {
            plan: Some(plan),
            bindings,
            error,
            held: None,
        };
        if !self.candidate().is_dense()
            || !self.candidate().matches(graph, capability, catalogue)
            || catalogue.digest() != moxie_kernels::dense_graph_catalogue().digest()
            || ctx.uuid() != capability.uuid
            || stream.device_uuid() != capability.uuid
        {
            return Err(reject(
                self,
                bindings,
                invalid("execution", "admitted dense graph identity changed"),
            ));
        }
        if let Err(error) = validate_bindings_except(&self, graph, &bindings, resident) {
            return Err(reject(self, bindings, error));
        }
        let symbols: Vec<String> = self
            .candidate()
            .nodes()
            .iter()
            .flat_map(|node| {
                node.descriptor
                    .symbols
                    .iter()
                    .map(|symbol| symbol.0.clone())
            })
            .collect();
        // SAFETY: this is the nvcc output embedded by this build, and selection
        // above binds the plan to the dense catalogue's image identity.
        let trusted =
            match unsafe { TrustedImage::from_build_output(moxie_kernels::DENSE_GRAPH_FATBIN) } {
                Ok(image) => image,
                Err(error) => return Err(reject(self, bindings, error)),
            };
        let package = match Module::load(ctx, ModuleImage::Binary(trusted))
            .and_then(|module| module.resolve_all(&symbols))
        {
            Ok(package) => package,
            Err(error) => return Err(reject(self, bindings, error)),
        };
        let operation = DenseOperation {
            plan: Some(self),
            sources: bindings,
            pending_host_upload: None,
            package,
            launch_order: Vec::new(),
            device_ordinal: ctx.ordinal(),
        };
        let mut lease = OperationLease::new("selected reduced dense graph", operation)
            .expect("static label is nonempty");
        if let Err(error) =
            enqueue_dense(&mut lease, graph, state, transaction, runs, layers, stream)
        {
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

fn enqueue_dense<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    graph: &Graph,
    state: &mut DeviceKvSequence,
    transaction: StateTransactionId,
    runs: &mut [PagedAttentionRun<'ctx>],
    layers: &BTreeMap<NodeId, u32>,
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
    let mut symbol_index = 0usize;
    upload_sources(lease, stream)?;
    let selected_nodes = lease
        .resource()
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .candidate()
        .nodes()
        .to_vec();

    for (node, selected) in graph.nodes().iter().zip(selected_nodes.iter()) {
        let base = symbol_index;
        symbol_index = symbol_index
            .checked_add(selected.descriptor.symbols.len())
            .ok_or_else(|| invalid("symbols", "dense symbol index overflowed"))?;
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
                    SemanticKernelOp::LinearPartial => (moxie_kernels::DENSE_LINEAR_PARTIAL, 6, 1),
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
                let positions = index_values(lease.resource(), positions()?, rows)?;
                let angles = angle_table(positions, rotary_dim, frequency_dim, rope_base)?;
                let mut input_address = address(lease.resource(), node.inputs[0])?;
                let mut angle_address = workspace_address(lease.resource())?;
                let mut output_address = address(lease.resource(), node.output)?;
                lease.resource_mut().pending_host_upload = Some(angles);
                // The table source is pageable host memory. Observe the copy
                // before reusing the one bounded host workspace Vec for the
                // next Rope node. The source lives in the operation so either
                // copy or synchronization failure can quarantine it safely.
                let workspace = lease
                    .resource()
                    .plan
                    .as_ref()
                    .expect("dense operation retains plan")
                    .workspace_range()?;
                // SAFETY: the operation retains the source until the
                // synchronous copy has completed; the destination is the
                // admitted workspace range.
                unsafe {
                    workspace.copy_from_host_async(
                        lease
                            .resource()
                            .pending_host_upload
                            .as_ref()
                            .expect("angle source retained before copy"),
                        stream,
                    )?
                };
                stream.synchronize()?;
                lease.resource_mut().pending_host_upload = None;
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
                top_k,
                activation: moxie_graph::ExpertActivation::GeGlu,
                ..
            } => {
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
                let project_elements = assignments
                    .checked_mul(intermediate)
                    .ok_or_else(|| invalid("expert-mlp", "project grid overflowed"))?;
                let project_blocks = u32::try_from(project_elements.div_ceil(256))
                    .map_err(|_| invalid("expert-mlp", "project launch grid overflowed"))?;
                let mut project_params: [*mut c_void; 8] = [
                    (&raw mut input_address).cast(),
                    (&raw mut ids_address).cast(),
                    (&raw mut gate_up_address).cast(),
                    (&raw mut activated_address).cast(),
                    (&raw mut launch_assignments).cast(),
                    (&raw mut launch_top_k).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_intermediate).cast(),
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
                let mut down_params: [*mut c_void; 7] = [
                    (&raw mut activated_address).cast(),
                    (&raw mut ids_address).cast(),
                    (&raw mut down_address).cast(),
                    (&raw mut slots_address).cast(),
                    (&raw mut launch_assignments).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_intermediate).cast(),
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
                let mut launch_rows = rows;
                let mut launch_top_k = top_k;
                let mut launch_hidden = hidden;
                let mut launch_scale = output_scale;
                let mut params: [*mut c_void; 8] = [
                    (&raw mut ids_address).cast(),
                    (&raw mut coefficients_address).cast(),
                    (&raw mut slots_address).cast(),
                    (&raw mut output_address).cast(),
                    (&raw mut launch_rows).cast(),
                    (&raw mut launch_top_k).cast(),
                    (&raw mut launch_hidden).cast(),
                    (&raw mut launch_scale).cast(),
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
    let key_value_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("attention", "key/value width overflowed"))?;
    let payload_bytes = rows
        .checked_mul(key_value_width)
        .and_then(|elements| elements.checked_mul(2))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| invalid("attention", "key/value staging extent is not addressable"))?;
    let key_range = address_range(lease.resource(), node.inputs[1])?;
    let value_range = address_range(lease.resource(), node.inputs[2])?;
    let mut keys = Vec::new();
    keys.try_reserve_exact(payload_bytes)
        .map_err(|_| capacity(payload_bytes))?;
    keys.resize(payload_bytes, 0);
    let mut values = Vec::new();
    values
        .try_reserve_exact(payload_bytes)
        .map_err(|_| capacity(payload_bytes))?;
    values.resize(payload_bytes, 0);
    key_range.copy_to_host(&mut keys)?;
    value_range.copy_to_host(&mut values)?;
    if state.layer_count()? <= layer as usize || run_count != state.layer_count()? {
        return Err(invalid(
            "state",
            "dense attention state and admitted runs do not have one entry per layer",
        ));
    }
    if let Err(refused) = append_paged_layer(
        state,
        transaction,
        layer as usize,
        rows,
        run,
        stream,
        PagedKvRows { keys, values },
    ) {
        return Err(refused.error);
    }
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
    match run.attend_into(stream, &launch, query_range, output_range) {
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

#[allow(clippy::too_many_arguments)]
fn launch<'ctx>(
    lease: &OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    symbol_index: usize,
    stream: &Stream<'ctx>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &mut [*mut c_void],
    node: &SelectedNode,
    symbol: &str,
) -> Result<()> {
    let operation = lease.resource();
    // SAFETY: descriptor selection fixes this ABI and the caller passed only
    // addresses within ranges admitted for this plan.
    unsafe {
        operation
            .package
            .launch_async(symbol_index, stream, grid, block, 0, params)
            .map_err(|error| attribute_node_error(error, operation.device_ordinal, node, symbol))
    }
}

fn push_launch<'ctx>(
    lease: &mut OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>,
    name: &str,
) {
    lease.resource_mut().launch_order.push(name.into());
}

fn address<'ctx>(operation: &DenseOperation<'ctx>, value: ValueId) -> Result<u64> {
    operation
        .plan
        .as_ref()
        .expect("dense operation retains plan")
        .range_for_value(value)
        .and_then(|range| range.device_address())
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
    let source = operation
        .sources
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
            .resolve_all(&[moxie_kernels::DENSE_COMBINE.to_string()])
            .unwrap();
        let mut ids = range.device_address().unwrap();
        let mut coefficients = ids + COEFFICIENTS;
        let mut slots = ids + SLOTS;
        let mut output = ids + OUTPUT;
        let mut rows = 1u64;
        let mut top_k = 3u64;
        let mut hidden = 1u64;
        let mut output_scale = 1.0f32;
        let mut params: [*mut c_void; 8] = [
            (&raw mut ids).cast(),
            (&raw mut coefficients).cast(),
            (&raw mut slots).cast(),
            (&raw mut output).cast(),
            (&raw mut rows).cast(),
            (&raw mut top_k).cast(),
            (&raw mut hidden).cast(),
            (&raw mut output_scale).cast(),
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

        arena.release(range).unwrap();
        drop(module);
        arena.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}
