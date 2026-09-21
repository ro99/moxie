use moxie_cuda::{Event, RankContext, Stream};
use moxie_plan::{Graph, OpParams, Visibility};
use moxie_state::{DeviceKvSequence, Retention};
use moxie_types::{DeviceCapability, Error, KernelCatalogue, Result, StateTransactionId};

use crate::paged_attention::device::{PagedKvRows, append_paged_layer};
use crate::{
    AttentionLayer, PageGeometry, PagedAttentionLaunch, PagedAttentionRun, SelectedReservedPlan,
};

#[derive(Debug)]
pub struct PagedAttentionInputs {
    pub query: Vec<u8>,
    pub keys: Vec<u8>,
    pub values: Vec<u8>,
    pub positions: Vec<u64>,
}

#[derive(Debug)]
pub struct PagedAttentionStep<'step, 'ctx> {
    pub graph: &'step Graph,
    pub capability: &'step DeviceCapability,
    pub catalogue: &'step KernelCatalogue,
    pub ctx: &'ctx RankContext,
    pub stream: &'step Stream<'ctx>,
    pub state: &'step mut DeviceKvSequence,
    pub transaction: StateTransactionId,
    pub run: &'step mut PagedAttentionRun<'ctx>,
    pub inputs: PagedAttentionInputs,
}

#[derive(Debug)]
pub struct PagedPlanRunRefused {
    pub inputs: Option<PagedAttentionInputs>,
    pub error: Error,
}

impl<'ctx> SelectedReservedPlan<'ctx> {
    #[allow(clippy::result_large_err)]
    pub fn execute_paged_attention(
        &mut self,
        step: PagedAttentionStep<'_, 'ctx>,
    ) -> std::result::Result<(), PagedPlanRunRefused> {
        if let Err(error) = validate(self, &step) {
            return Err(PagedPlanRunRefused {
                inputs: Some(step.inputs),
                error,
            });
        }
        let PagedAttentionStep {
            graph,
            capability: _,
            catalogue: _,
            ctx,
            stream,
            state,
            transaction,
            run,
            inputs,
        } = step;

        let node = &graph.nodes()[0];
        let OpParams::Attention {
            heads,
            kv_heads,
            head_dim,
            scale,
            visibility,
            layer,
        } = node.params
        else {
            unreachable!("validated attention graph")
        };
        let query_value = node.inputs[0];
        let output_value = node.output;
        let event = match Event::new(ctx) {
            Ok(event) => event,
            Err(error) => {
                return Err(PagedPlanRunRefused {
                    inputs: Some(inputs),
                    error,
                });
            }
        };
        let PagedAttentionInputs {
            query,
            keys,
            values,
            positions,
        } = inputs;
        let (query_key, query_range) = match self.take_range_for_value(query_value) {
            Ok(range) => range,
            Err(error) => {
                return Err(PagedPlanRunRefused {
                    inputs: Some(PagedAttentionInputs {
                        query,
                        keys,
                        values,
                        positions,
                    }),
                    error,
                });
            }
        };
        let mut upload = match query_range.prepare_upload(query, "paged attention query") {
            Ok(upload) => upload,
            Err(refused) => {
                if let Err(error) = self.restore_range(query_key, refused.range) {
                    return Err(PagedPlanRunRefused {
                        inputs: Some(PagedAttentionInputs {
                            query: refused.source,
                            keys,
                            values,
                            positions,
                        }),
                        error,
                    });
                }
                return Err(PagedPlanRunRefused {
                    inputs: Some(PagedAttentionInputs {
                        query: refused.source,
                        keys,
                        values,
                        positions,
                    }),
                    error: refused.error,
                });
            }
        };
        if let Err(error) = upload.submit(stream, event) {
            core::mem::forget(upload);
            return Err(PagedPlanRunRefused {
                inputs: None,
                error,
            });
        }
        if let Err(error) = upload.synchronize() {
            drop(upload);
            return Err(PagedPlanRunRefused {
                inputs: None,
                error,
            });
        }
        let (_, upload) = match upload.retire() {
            Ok(retired) => retired,
            Err(refused) => {
                let error = refused.error;
                drop(refused.lease);
                return Err(PagedPlanRunRefused {
                    inputs: None,
                    error,
                });
            }
        };
        let (query_range, query) = upload.finish();
        if let Err(error) = self.restore_range(query_key, query_range) {
            return Err(PagedPlanRunRefused {
                inputs: Some(PagedAttentionInputs {
                    query,
                    keys,
                    values,
                    positions,
                }),
                error,
            });
        }

        if let Err(refused) = append_paged_layer(
            state,
            transaction,
            layer as usize,
            positions.len() as u64,
            run,
            stream,
            PagedKvRows { keys, values },
        ) {
            return Err(PagedPlanRunRefused {
                inputs: refused.source.map(|source| PagedAttentionInputs {
                    query,
                    keys: source.keys,
                    values: source.values,
                    positions,
                }),
                error: refused.error,
            });
        }

        let retained = match state.layer_retained(layer as usize) {
            Ok(retained) => retained,
            Err(error) => {
                return Err(PagedPlanRunRefused {
                    inputs: None,
                    error,
                });
            }
        };
        let launch = match PagedAttentionLaunch::new(
            AttentionLayer {
                geometry: PageGeometry {
                    kv_heads,
                    head_dim,
                    page_tokens: state.geometry().expect("validated state").page_tokens as u64,
                    pages: state.layout(layer as usize).expect("validated layer").pages,
                },
                heads,
                scale,
                visibility,
            },
            positions.len() as u64,
            positions[0],
            retained.start,
            retained.end - retained.start,
        ) {
            Ok(launch) => launch,
            Err(error) => {
                return Err(PagedPlanRunRefused {
                    inputs: None,
                    error,
                });
            }
        };
        let last_query = launch.first_position() + launch.rows() - 1;
        let visible = match launch.visibility() {
            Visibility::Causal => last_query - launch.history_base() + 1,
            Visibility::SlidingWindow { window } => {
                window.min(last_query - launch.history_base() + 1)
            }
        };
        if visible > self.candidate().workload().visible_tokens {
            return Err(PagedPlanRunRefused {
                inputs: None,
                error: invalid(
                    "visible_tokens",
                    "state history exceeds the admitted attention bucket",
                ),
            });
        }
        let (query_key, query_range) = match self.take_range_for_value(query_value) {
            Ok(range) => range,
            Err(error) => {
                return Err(PagedPlanRunRefused {
                    inputs: None,
                    error,
                });
            }
        };
        let (output_key, output_range) = match self.take_range_for_value(output_value) {
            Ok(range) => range,
            Err(error) => {
                if let Err(restore) = self.restore_range(query_key, query_range) {
                    return Err(PagedPlanRunRefused {
                        inputs: None,
                        error: restore,
                    });
                }
                return Err(PagedPlanRunRefused {
                    inputs: None,
                    error,
                });
            }
        };
        match run.attend_into(stream, &launch, query_range, output_range) {
            Ok((query_range, output_range)) => {
                let query_result = self.restore_range(query_key, query_range);
                let output_result = self.restore_range(output_key, output_range);
                query_result
                    .and(output_result)
                    .map_err(|error| PagedPlanRunRefused {
                        inputs: None,
                        error,
                    })
            }
            Err(refused) => {
                if let Some((query_range, output_range)) = refused.ranges {
                    let query_result = self.restore_range(query_key, query_range);
                    let output_result = self.restore_range(output_key, output_range);
                    if let Err(error) = query_result.and(output_result) {
                        return Err(PagedPlanRunRefused {
                            inputs: None,
                            error,
                        });
                    }
                }
                Err(PagedPlanRunRefused {
                    inputs: None,
                    error: refused.error,
                })
            }
        }
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn read_paged_attention_output(&self, destination: &mut [u8]) -> Result<()> {
        let value = self.candidate().workload().output;
        let planned = self
            .candidate()
            .value(value)
            .ok_or_else(|| invalid("output", "selected output is absent"))?;
        if destination.len() as u64 != planned.logical_bytes {
            return Err(invalid(
                "output",
                "readback size differs from planned output",
            ));
        }
        self.range_for_selected_value(value)?
            .copy_to_host(destination)
    }
}

fn validate(plan: &SelectedReservedPlan<'_>, step: &PagedAttentionStep<'_, '_>) -> Result<()> {
    let PagedAttentionStep {
        graph,
        capability,
        catalogue,
        ctx,
        stream,
        state,
        transaction: _,
        run,
        inputs,
    } = step;
    if !plan.candidate().matches(graph, capability, catalogue)
        || !plan.candidate().is_paged_attention()
        || ctx.uuid() != capability.uuid
        || stream.device_uuid() != capability.uuid
    {
        return Err(invalid(
            "execution",
            "admitted graph/catalogue/capability/stream changed",
        ));
    }
    let [selected] = plan.candidate().nodes() else {
        return Err(invalid("plan", "attention plan has one selected node"));
    };
    if &selected.descriptor != run.descriptor() {
        return Err(invalid(
            "kernel",
            "state run differs from the selected descriptor",
        ));
    }
    let [node] = graph.nodes() else {
        return Err(invalid("graph", "attention execution requires one node"));
    };
    let OpParams::Attention {
        heads,
        kv_heads,
        head_dim,
        visibility,
        layer,
        ..
    } = node.params
    else {
        return Err(invalid("graph", "selected graph is not attention"));
    };
    let rows = plan.candidate().workload().rows;
    let query_bytes = rows
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| invalid("query", "query extent overflowed"))?;
    let kv_bytes = rows
        .checked_mul(kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| invalid("keys", "key/value extent overflowed"))?;
    if inputs.query.len() as u64 != query_bytes
        || inputs.keys.len() as u64 != kv_bytes
        || inputs.values.len() as u64 != kv_bytes
        || inputs.positions.len() as u64 != rows
        || inputs.query.capacity() != inputs.query.len()
        || inputs.keys.capacity() != inputs.keys.len()
        || inputs.values.capacity() != inputs.values.len()
        || inputs.positions.capacity() != inputs.positions.len()
    {
        return Err(invalid(
            "inputs",
            "attention inputs differ from the planned extents",
        ));
    }
    if [&inputs.query, &inputs.keys, &inputs.values]
        .into_iter()
        .any(|bytes| {
            bytes
                .chunks_exact(2)
                .any(|word| !bf16_is_finite(u16::from_le_bytes([word[0], word[1]])))
        })
    {
        return Err(invalid("inputs", "attention inputs contain nonfinite BF16"));
    }
    let first = state.layer_published_rows(layer as usize)?;
    if inputs
        .positions
        .iter()
        .copied()
        .enumerate()
        .any(|(offset, position)| first.checked_add(offset as u64) != Some(position))
    {
        return Err(invalid(
            "positions",
            "positions must continue the layer frontier",
        ));
    }
    let geometry = state.geometry()?;
    let state_layer = geometry
        .layers
        .get(layer as usize)
        .ok_or_else(|| invalid("layer", "state has no selected layer"))?;
    let expected_retention = match visibility {
        Visibility::Causal => Retention::All,
        Visibility::SlidingWindow { window } => Retention::Window {
            window: usize::try_from(window)
                .map_err(|_| invalid("visibility", "window exceeds usize"))?,
        },
    };
    let layout = state.layout(layer as usize)?;
    let expected_run = PageGeometry {
        kv_heads,
        head_dim,
        page_tokens: geometry.page_tokens as u64,
        pages: layout.pages,
    };
    if state_layer.kv_heads as u64 != kv_heads
        || state_layer.key_dim as u64 != head_dim
        || state_layer.value_dim as u64 != head_dim
        || state_layer.retention != expected_retention
        || run.geometry() != &expected_run
    {
        return Err(invalid(
            "state",
            "state geometry differs from the attention node",
        ));
    }
    plan.range_for_selected_value(node.inputs[0])?;
    plan.range_for_selected_value(node.output)?;
    Ok(())
}

fn bf16_is_finite(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

fn invalid(field: &'static str, detail: &'static str) -> Error {
    Error::InvalidRequest {
        field,
        detail: moxie_memory::fallible::text(format_args!("{detail}")).unwrap_or_default(),
    }
}
