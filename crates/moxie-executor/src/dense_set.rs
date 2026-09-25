//! Dense bucket plans sharing one residency-authority weight copy.

#![cfg(all(feature = "driver", feature = "paged-attention-binding"))]

use std::collections::{BTreeMap, BTreeSet};

use moxie_cuda::{RankContext, Stream};
use moxie_graph::{Graph, ValueId};
use moxie_memory::{Ledger, LedgerId, ResidencyAuthority, ResidencyLease};
use moxie_plan::{SelectedPlanCandidate, StorageRegion};
use moxie_state::DeviceKvSequence;
use moxie_types::{DeviceCapability, Error, KernelCatalogue, Result, Scope, StateTransactionId};

use crate::arena::OperationLease;
use crate::chain::{OwnedBinding, SelectedAdmitRefused, SelectedCompletion, SelectedReservedPlan};
use crate::dense::{DenseGraphStep, DenseOperation, DensePlanRunRefused};
use crate::paged_attention::device::PagedAttentionRun;
use crate::residency::DeviceResidency;

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

/// One dense plan-set step. The graph is fixed when the set is admitted.
#[derive(Debug)]
pub struct DenseSetStep<'step, 'ctx> {
    pub capability: &'step DeviceCapability,
    pub catalogue: &'step KernelCatalogue,
    pub ctx: &'ctx RankContext,
    pub stream: &'step Stream<'ctx>,
    pub state: &'step mut DeviceKvSequence,
    pub transaction: StateTransactionId,
    pub runs: &'step mut [PagedAttentionRun<'ctx>],
    pub bindings: Vec<OwnedBinding>,
    pub authority: &'step ResidencyAuthority,
}

/// A completed plan-set step.
#[derive(Debug)]
pub struct DenseSetOutput {
    pub output: Vec<u8>,
    pub returned_inputs: Vec<OwnedBinding>,
    pub launch_order: Vec<String>,
}

/// Admission refused while retaining every residency lease and any plan that
/// could not be closed during unwind.
#[derive(Debug)]
pub struct DensePlanSetRefused<'r, 'ctx> {
    pub residency: &'r DeviceResidency<'ctx>,
    pub leases: BTreeMap<ValueId, ResidencyLease>,
    pub plans: BTreeMap<u64, SelectedReservedPlan<'ctx>>,
    pub admission: Option<SelectedAdmitRefused<'ctx>>,
    pub error: Error,
}

/// A close refusal, with the plan set and its leases intact for retry.
#[derive(Debug)]
pub struct DenseSetCloseRefused<'r, 'ctx> {
    pub set: DensePlanSet<'r, 'ctx>,
    pub error: Error,
}

/// Several row buckets over one leased, device-resident copy of graph weights.
#[derive(Debug)]
pub struct DensePlanSet<'r, 'ctx> {
    graph: &'r Graph,
    residency: &'r DeviceResidency<'ctx>,
    leases: BTreeMap<ValueId, ResidencyLease>,
    plans: BTreeMap<u64, SelectedReservedPlan<'ctx>>,
    lost: Option<OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>>,
    ledger: LedgerId,
}

impl<'r, 'ctx> DensePlanSet<'r, 'ctx> {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn admit(
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
        graph: &'r Graph,
        capability: &DeviceCapability,
        catalogue: &KernelCatalogue,
        residency: &'r DeviceResidency<'ctx>,
        authority: &ResidencyAuthority,
        leases: BTreeMap<ValueId, ResidencyLease>,
        candidates: Vec<SelectedPlanCandidate>,
    ) -> std::result::Result<Self, DensePlanSetRefused<'r, 'ctx>> {
        let refuse = |leases, error| DensePlanSetRefused {
            residency,
            leases,
            plans: BTreeMap::new(),
            admission: None,
            error,
        };
        if leases.len() != graph.weights().len()
            || graph
                .weights()
                .iter()
                .any(|value| !leases.contains_key(value))
        {
            return Err(refuse(
                leases,
                invalid("weights", "resident leases must name every graph weight"),
            ));
        }
        if residency.scope() != Scope::Device(ctx.uuid())
            || residency.authority_id() != Some(authority.id())
            || ctx.uuid() != capability.uuid
        {
            return Err(refuse(
                leases,
                invalid(
                    "residency",
                    "the backing, authority, and context must name one device",
                ),
            ));
        }

        let mut ranges = BTreeMap::new();
        for (value, lease) in &leases {
            if lease.scope() != Scope::Device(ctx.uuid()) {
                return Err(refuse(
                    leases,
                    invalid("weights", "every lease must belong to the context device"),
                ));
            }
            let (offset, len) = match authority.device_range(lease) {
                Ok(range) => range,
                Err(error) => return Err(refuse(leases, error)),
            };
            ranges.insert(*value, (offset, len));
        }

        if candidates.is_empty() {
            return Err(refuse(
                leases,
                invalid(
                    "candidates",
                    "a dense plan set needs at least one row bucket",
                ),
            ));
        }
        let mut rows = BTreeSet::new();
        for candidate in &candidates {
            if !candidate.is_dense()
                || !candidate.matches(graph, capability, catalogue)
                || !candidate.weight_formats().is_empty()
                || !candidate.host_expert_joins().is_empty()
            {
                return Err(refuse(
                    leases,
                    invalid(
                        "weights",
                        "resident plan sets require matching dense candidates with unformatted weights",
                    ),
                ));
            }
            if !rows.insert(candidate.workload().rows) {
                return Err(refuse(
                    leases,
                    invalid("rows", "candidate row buckets must be distinct"),
                ));
            }
            for value in graph.weights() {
                let Some(planned) = candidate.value(*value) else {
                    return Err(refuse(
                        leases,
                        invalid("weights", "a candidate omits a graph weight"),
                    ));
                };
                let Some((_, len)) = ranges.get(value) else {
                    return Err(refuse(
                        leases,
                        invalid("weights", "a graph weight has no resident lease"),
                    ));
                };
                if *len != planned.logical_bytes || planned.region != StorageRegion::Weights {
                    return Err(refuse(
                        leases,
                        invalid(
                            "weights",
                            "resident range length differs from a candidate weight",
                        ),
                    ));
                }
            }
        }

        let mut addresses = BTreeMap::new();
        for (value, (offset, len)) in ranges {
            let address = match residency.device_address(offset, len) {
                Ok(address) => address,
                Err(error) => return Err(refuse(leases, error)),
            };
            addresses.insert(value, address);
        }

        let mut plans = BTreeMap::new();
        for candidate in candidates {
            let bucket = candidate.workload().rows;
            match SelectedReservedPlan::admit_with_resident_weights(
                candidate,
                graph,
                capability,
                catalogue,
                ledger,
                ctx,
                addresses.clone(),
            ) {
                Ok(plan) => {
                    plans.insert(bucket, plan);
                }
                Err(admission) => {
                    let error = match &admission {
                        SelectedAdmitRefused::Invalid { error, .. }
                        | SelectedAdmitRefused::Held { error, .. } => error.clone(),
                        SelectedAdmitRefused::Rejected { rejection, .. } => {
                            Error::from(rejection.clone())
                        }
                    };
                    let close_error = close_plans(&mut plans, ledger);
                    return Err(DensePlanSetRefused {
                        residency,
                        leases,
                        plans,
                        admission: Some(admission),
                        error: close_error.unwrap_or(error),
                    });
                }
            }
        }

        Ok(Self {
            graph,
            residency,
            leases,
            plans,
            lost: None,
            ledger: ledger.id(),
        })
    }

    pub fn set_segment_capture(
        &mut self,
        rows: u64,
        enabled: bool,
        ledger: &mut Ledger,
    ) -> Result<()> {
        if self.lost.is_some() {
            return Err(invalid("plan_set", "a lost operation poisons this set"));
        }
        let plan = self
            .plans
            .get_mut(&rows)
            .ok_or_else(|| invalid("rows", "the requested row bucket is not admitted"))?;
        plan.set_segment_capture(enabled, ledger)
    }

    #[allow(clippy::result_large_err)]
    pub fn step(&mut self, rows: u64, step: DenseSetStep<'_, 'ctx>) -> Result<DenseSetOutput> {
        if self.lost.is_some() {
            return Err(invalid("plan_set", "a lost operation poisons this set"));
        }
        let plan = self
            .plans
            .get(&rows)
            .ok_or_else(|| invalid("rows", "the requested row bucket is not admitted"))?;
        if self.residency.authority_id() != Some(step.authority.id()) {
            return Err(invalid(
                "authority",
                "the set's backing belongs to another authority",
            ));
        }
        for (value, lease) in &self.leases {
            if lease.scope() != Scope::Device(step.ctx.uuid()) {
                return Err(invalid("weights", "a resident lease changed device scope"));
            }
            let (offset, len) = step.authority.device_range(lease)?;
            let planned = plan
                .candidate()
                .value(*value)
                .ok_or_else(|| invalid("weights", "the selected bucket omits a graph weight"))?;
            if len != planned.logical_bytes
                || self.residency.device_address(offset, len)? != plan.value_address(*value)?
            {
                return Err(invalid(
                    "weights",
                    "a resident lease range changed after admission",
                ));
            }
        }

        let plan = self
            .plans
            .remove(&rows)
            .expect("checked admitted row bucket");
        let DenseSetStep {
            capability,
            catalogue,
            ctx,
            stream,
            state,
            transaction,
            runs,
            bindings,
            authority: _,
        } = step;
        let operation = match plan.execute_dense(DenseGraphStep {
            graph: self.graph,
            capability,
            catalogue,
            ctx,
            stream,
            state,
            transaction,
            runs,
            bindings,
            host_experts: &[],
        }) {
            Ok(operation) => operation,
            Err(DensePlanRunRefused {
                plan, held, error, ..
            }) => {
                if let Some(held) = held {
                    self.lost = Some(held);
                }
                if let Some(plan) = plan {
                    self.plans.insert(rows, plan);
                }
                return Err(error);
            }
        };
        match operation.finish() {
            Ok(result) => {
                self.plans.insert(rows, result.plan);
                Ok(DenseSetOutput {
                    output: result.output,
                    returned_inputs: result.returned_inputs,
                    launch_order: result.launch_order,
                })
            }
            Err(refused) => {
                self.lost = Some(refused.lease);
                Err(refused.error)
            }
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn close(
        mut self,
        ledger: &mut Ledger,
        authority: &mut ResidencyAuthority,
    ) -> std::result::Result<(), DenseSetCloseRefused<'r, 'ctx>> {
        let refuse = |set, error| DenseSetCloseRefused { set, error };
        if self.ledger != ledger.id() {
            return Err(refuse(
                self,
                invalid("ledger", "the plan set belongs to another ledger"),
            ));
        }
        if self.lost.is_some() {
            return Err(refuse(
                self,
                invalid(
                    "plan_set",
                    "a lost operation keeps its plans and leases held",
                ),
            ));
        }
        while let Some((rows, plan)) = self.plans.pop_first() {
            if let Err(refused) = plan.close(ledger) {
                self.plans.insert(rows, refused.plan);
                return Err(refuse(self, refused.error));
            }
        }
        while let Some((value, lease)) = self.leases.pop_first() {
            if let Err(refused) = authority.release(lease) {
                self.leases.insert(value, refused.lease);
                return Err(refuse(self, refused.error));
            }
        }
        Ok(())
    }
}

fn close_plans<'ctx>(
    plans: &mut BTreeMap<u64, SelectedReservedPlan<'ctx>>,
    ledger: &mut Ledger,
) -> Option<Error> {
    while let Some((rows, plan)) = plans.pop_first() {
        if let Err(refused) = plan.close(ledger) {
            plans.insert(rows, refused.plan);
            return Some(refused.error);
        }
    }
    None
}
