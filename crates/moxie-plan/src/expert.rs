//! The routed layer's expert work, as one plan over two candidates.
//!
//! Roadmap M2 item 3 asks for "CPU expert fallback and GPU grouped candidate
//! plans under one interface, with bounded queues and NUMA-aware host placement.
//! Begin with conservative deterministic scheduling." This module is the
//! interface and the choice; `moxie-executor` performs what it decides and
//! `moxie-kernels` supplies the two kernels.
//!
//! It obeys this crate's rule without exception: **pure with respect to live
//! resources.** It receives a snapshot, allocates nothing, opens nothing, and
//! queries no device. Document 02 puts the atomic reservation in `admit` and
//! forbids `execute` from evading it, and a planner that peeked at a live cache
//! would make both of those promises about how the code is written rather than
//! properties of what it can reach.
//!
//! Three decisions here are worth finding quickly.
//!
//! **The choice is arithmetic, not a cost model.** Document 03 says grouped GPU
//! execution "is favored where row reuse amortizes transfer", and
//! [`GroupDecision::bytes_per_row`] is that sentence as a quantity. The
//! threshold it is compared against is a **declared policy parameter and not a
//! measured crossover**: there is no measured host expert throughput and no
//! measured device grouped throughput on this machine, so a number presented as
//! a crossover would be a fabrication. What "conservative deterministic
//! scheduling" can honestly mean at M2 is that the same input yields the same
//! decision, the parameter is visible, and the rejected alternative is reported
//! with the numbers that decided it. Measuring the crossover is M6's.
//!
//! **The reduction order is data.** [`ExpertPlan::reduction_order`] is an
//! explicit per-row permutation of slot positions, computed here from the
//! graph's [`CombineOrder`]. Task 0019 made that order a parameter because
//! floating-point addition is not associative; carrying it as a permutation is
//! what lets a plan that mixed both candidates reduce without either one's
//! completion order reaching the sum.
//!
//! **A candidate is admissible plan-wide before it is chosen per group.** The
//! device arena and the host workspace are whole-plan quantities, so they are
//! decided once, up front, from bounds that do not depend on the assignment. A
//! chooser that assigned first and discovered the envelope afterwards would need
//! to demote groups in a loop, and a loop is where a deterministic scheduler
//! stops being one.

use std::collections::BTreeMap;

use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_types::{
    DeviceCapability, DeviceTier, DeviceUuid, Error, GateTransform, HostPlacement, HostTier,
    KernelCatalogue, NumaTopology, Result, SemanticKernelDescriptor, SemanticKernelOp,
    StrategyControl, Tier, WorkspaceExpression,
};

/// Bytes of one BF16 element.
const BF16: u64 = 2;
/// Bytes of one FP32 workspace element.
const F32: u64 = 4;
/// Device range alignment, as every admitted range in this workspace uses.
const DEVICE_ALIGNMENT: u64 = 256;

/// Round up to a whole device range.
///
/// The envelope is charged in aligned units because that is what the executor
/// will actually allocate: charging the logical extent and allocating the
/// aligned one is how a plan comes to need more memory than it reserved.
fn align_device(bytes: u64) -> Option<u64> {
    bytes
        .checked_add(DEVICE_ALIGNMENT - 1)
        .map(|v| v / DEVICE_ALIGNMENT * DEVICE_ALIGNMENT)
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

fn overflow(what: &str) -> Error {
    Error::InvalidRequest {
        field: "expert_plan",
        detail: format!("{what} overflows u64"),
    }
}

/// Which side of the interface a group runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Candidate {
    /// Grouped execution on the plan's device.
    Device,
    /// Tiled execution on the host, over canonical packed weights in place.
    Host,
}

impl Candidate {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Device => "gpu-grouped",
            Self::Host => "cpu-tiled",
        }
    }

    const fn other(self) -> Self {
        match self {
            Self::Device => Self::Host,
            Self::Host => Self::Device,
        }
    }
}

/// Where a group's work actually happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Device(DeviceUuid),
    Host(HostPlacement),
}

impl Placement {
    pub const fn candidate(self) -> Candidate {
        match self {
            Self::Device(_) => Candidate::Device,
            Self::Host(_) => Candidate::Host,
        }
    }
}

/// Why a candidate was not taken. Every arm carries the numbers that decided it,
/// because document 02 requires a plan's diagnostics to explain "rejected
/// alternatives" and a reason without its quantities cannot be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// The control says this candidate may not be considered at all.
    CandidateOff,
    /// Reuse does not amortise the transfer at the declared policy.
    TransferNotAmortised { bytes_per_row: u64, limit: u64 },
    /// The device expert cache cannot hold the incoming chunks even after
    /// displacing everything that is displaceable.
    DeviceCacheTooSmall { incoming: u64, displaceable: u64 },
    /// The device arena cannot hold this plan's activations and workspace.
    DeviceArenaTooSmall { needed: u64, free: u64 },
    /// The host workspace budget cannot hold the queue's FP32 tiles.
    HostWorkspaceTooSmall { needed: u64, free: u64 },
    /// The host budget cannot hold the activation block, slot buffer and output.
    HostBuffersTooSmall { needed: u64, free: u64 },
    /// The other candidate is `required`, so this one may not be considered.
    OtherCandidateRequired,
    /// No kernel in the injected catalogue matches this operation on this
    /// device. Distinct from every capacity reason: the bytes are there and the
    /// code is not.
    NoQualifiedKernel,
    /// Chosen on merit; the other candidate was admissible and not preferred.
    NotPreferred,
}

impl RejectionReason {
    pub const fn is_inadmissible(self) -> bool {
        !matches!(self, Self::NotPreferred)
    }
}

impl core::fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CandidateOff => f.write_str("the control is off"),
            Self::TransferNotAmortised {
                bytes_per_row,
                limit,
            } => write!(
                f,
                "{bytes_per_row} B/row transferred exceeds the declared {limit} B/row"
            ),
            Self::DeviceCacheTooSmall {
                incoming,
                displaceable,
            } => write!(
                f,
                "the expert cache offers {displaceable} B against {incoming} B incoming"
            ),
            Self::DeviceArenaTooSmall { needed, free } => {
                write!(
                    f,
                    "the device arena offers {free} B against {needed} B needed"
                )
            }
            Self::HostWorkspaceTooSmall { needed, free } => {
                write!(
                    f,
                    "the host workspace offers {free} B against {needed} B needed"
                )
            }
            Self::HostBuffersTooSmall { needed, free } => {
                write!(
                    f,
                    "the host buffers offer {free} B against {needed} B needed"
                )
            }
            Self::OtherCandidateRequired => {
                f.write_str("the other candidate is required, so this one is not considered")
            }
            Self::NoQualifiedKernel => {
                f.write_str("no catalogue kernel matches this operation on this device")
            }
            Self::NotPreferred => f.write_str("admissible but not preferred"),
        }
    }
}

/// What was chosen for one expert, what was not, and the arithmetic behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupDecision {
    pub chosen: Candidate,
    pub rejected: Candidate,
    pub reason: RejectionReason,
    /// Distinct (row, slot) pairs routed to this expert.
    pub reuse_rows: u64,
    /// Bytes that would move to the device for this expert: zero when the
    /// snapshot already had it resident.
    pub transfer_bytes: u64,
    /// `ceil(transfer_bytes / reuse_rows)`.
    pub bytes_per_row: u64,
    pub already_resident: bool,
}

/// One expert's work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertGroup {
    pub expert: u32,
    /// Which activation rows this group reads, ascending.
    pub rows: Vec<u32>,
    /// Where each result goes in the slot-major buffer: `row * top_k + j`.
    /// Parallel to `rows`.
    pub slots: Vec<u32>,
    pub placement: Placement,
    /// One expert's two chunks together: `6 * intermediate * hidden` bytes.
    pub chunk_bytes: u64,
    pub decision: GroupDecision,
}

/// The exact bytes `admit` must reserve, per tier and scope.
///
/// `residency_demand_bytes` is deliberately not in either list: the residency
/// authority admits weight bytes against its own cap and is the only thing that
/// may. Reserving them here as well would charge them twice and would make this
/// crate a second residency owner by accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertEnvelope {
    pub device: Vec<(DeviceTier, u64)>,
    pub host: Vec<(HostTier, u64)>,
    pub residency_demand_bytes: u64,
}

impl ExpertEnvelope {
    pub fn device_bytes(&self, tier: DeviceTier) -> u64 {
        self.device
            .iter()
            .find(|(t, _)| *t == tier)
            .map(|(_, b)| *b)
            .unwrap_or(0)
    }

    pub fn host_bytes(&self, tier: HostTier) -> u64 {
        self.host
            .iter()
            .find(|(t, _)| *t == tier)
            .map(|(_, b)| *b)
            .unwrap_or(0)
    }

    pub fn total_device_bytes(&self) -> u64 {
        self.device.iter().map(|(_, b)| *b).sum()
    }

    pub fn total_host_bytes(&self) -> u64 {
        self.host.iter().map(|(_, b)| *b).sum()
    }
}

/// The snapshot a plan is compiled against.
///
/// Every field is a **reading**, taken before compilation and not consulted
/// again. That is what makes `compile` pure: a number that changed underneath it
/// changes the plan's validity, which is what document 02's replan conditions
/// exist for -- not this function's return value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBudget {
    pub device: DeviceUuid,
    /// The device's PCI bus id, which is how the operating system's NUMA view
    /// and the CUDA device are reconciled. A UUID would not do: the kernel has
    /// never heard of it.
    pub device_pci_bus_id: String,
    /// The residency authority's expert-cache cap on this device.
    pub device_cache_cap_bytes: u64,
    /// Of that cap, what a live lease pins and eviction therefore cannot reach.
    pub device_cache_leased_bytes: u64,
    /// Free bytes in the device activation arena available to this plan.
    pub device_arena_free_bytes: u64,
    /// Host FP32 workspace bytes this plan may use.
    pub host_workspace_bytes: u64,
    /// Host bytes for the activation block, slot buffer and output.
    pub host_buffer_bytes: u64,
    /// Experts whose chunks the snapshot found already resident on the device.
    /// Order is irrelevant and duplicates are harmless; membership is the only
    /// question asked of it.
    pub resident_experts: Vec<u32>,
}

impl ExpertBudget {
    fn displaceable(&self) -> u64 {
        self.device_cache_cap_bytes
            .saturating_sub(self.device_cache_leased_bytes)
    }
}

/// The declared parameters of a conservative deterministic schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPolicy {
    pub device: StrategyControl,
    pub host: StrategyControl,
    /// The amortisation threshold. **Declared, not measured**; see the module
    /// documentation.
    pub max_transfer_bytes_per_row: u64,
    /// Upper bound on queue depth. The queue's real capacity is the smaller of
    /// this and the group count, and every slot's workspace is in the envelope.
    pub max_inflight_orders: u32,
    /// Gate/up lanes the host kernel projects before activating them.
    pub cpu_tile_lanes: u32,
    /// Whether host buffers are bound to a NUMA node. `Required` errors when the
    /// topology cannot answer; `Auto` reports `Unspecified` and proceeds.
    pub host_placement: StrategyControl,
}

impl Default for ExpertPolicy {
    /// Both candidates considered, a queue of four, a 64-lane host tile, and an
    /// amortisation threshold of 1 MiB per row.
    ///
    /// The threshold's value is a declaration and nothing more. It is stated
    /// here so that every caller that does not choose one is visibly using the
    /// same declared number rather than an implicit one.
    fn default() -> Self {
        Self {
            device: StrategyControl::Auto,
            host: StrategyControl::Auto,
            max_transfer_bytes_per_row: 1 << 20,
            max_inflight_orders: 4,
            cpu_tile_lanes: 64,
            host_placement: StrategyControl::Auto,
        }
    }
}

/// The shape the plan works in, taken from the graph's own parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    pub hidden: u64,
    pub intermediate: u64,
    pub experts: u64,
    pub top_k: u64,
    pub activation: ExpertActivation,
}

impl ExpertShape {
    /// One expert's two chunks: `[2I, H]` plus `[H, I]`, both BF16.
    pub fn chunk_bytes(self) -> Option<u64> {
        self.intermediate
            .checked_mul(self.hidden)?
            .checked_mul(3)?
            .checked_mul(BF16)
    }

    /// FP32 values one queue slot's host tile needs: the whole activated
    /// intermediate plus one tile of gate and up lanes.
    pub fn host_workspace_values(self, lanes: u32) -> Option<u64> {
        let lanes = u64::from(lanes).min(self.intermediate);
        self.intermediate.checked_add(lanes.checked_mul(2)?)
    }

    pub fn gate_transform(self) -> GateTransform {
        match self.activation {
            ExpertActivation::GeGlu => GateTransform::GeluTanh,
            ExpertActivation::SwiGlu => GateTransform::Silu,
        }
    }
}

/// The device half of a compilation: what the hardware is and what kernels
/// exist for it.
///
/// Optional, because a host-only plan needs neither and must still compile on a
/// machine with no catalogue at all.
#[derive(Debug, Clone, Copy)]
pub struct ExpertKernels<'a> {
    pub capability: &'a DeviceCapability,
    pub catalogue: &'a KernelCatalogue,
}

/// A routed layer's expert work, decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlan {
    kernel: Option<SemanticKernelDescriptor>,
    shape: ExpertShape,
    rows: u64,
    device: DeviceUuid,
    host_placement: HostPlacement,
    groups: Vec<ExpertGroup>,
    queue_capacity: u32,
    cpu_tile_lanes: u32,
    reduction_order: Vec<u32>,
    envelope: ExpertEnvelope,
    policy: ExpertPolicy,
}

impl ExpertPlan {
    pub const fn shape(&self) -> ExpertShape {
        self.shape
    }
    pub const fn rows(&self) -> u64 {
        self.rows
    }
    pub const fn device(&self) -> DeviceUuid {
        self.device
    }
    /// Where every host buffer of this plan belongs.
    pub const fn host_placement(&self) -> HostPlacement {
        self.host_placement
    }
    pub fn groups(&self) -> &[ExpertGroup] {
        &self.groups
    }
    pub const fn queue_capacity(&self) -> u32 {
        self.queue_capacity
    }
    pub const fn cpu_tile_lanes(&self) -> u32 {
        self.cpu_tile_lanes
    }
    /// `[rows * top_k]`: the slot position to add at each step, per row.
    pub fn reduction_order(&self) -> &[u32] {
        &self.reduction_order
    }
    pub const fn envelope(&self) -> &ExpertEnvelope {
        &self.envelope
    }
    pub const fn policy(&self) -> &ExpertPolicy {
        &self.policy
    }
    /// The descriptor selected for the device candidate, when there is one.
    ///
    /// Document 02 puts "chosen kernels" in the plan, so selection happens here
    /// -- pure, from an injected immutable catalogue -- rather than at launch
    /// time where a fallback would be easy to write.
    pub const fn kernel(&self) -> Option<&SemanticKernelDescriptor> {
        self.kernel.as_ref()
    }

    pub fn groups_on(&self, candidate: Candidate) -> impl Iterator<Item = &ExpertGroup> {
        self.groups
            .iter()
            .filter(move |g| g.placement.candidate() == candidate)
    }

    pub fn uses(&self, candidate: Candidate) -> bool {
        self.groups_on(candidate).next().is_some()
    }

    /// Total slots this plan produces: `rows * top_k`.
    pub fn slot_count(&self) -> u64 {
        self.rows * self.shape.top_k
    }

    /// BF16 bytes of the activation block, `[rows, hidden]`.
    pub fn activation_bytes(&self) -> u64 {
        self.rows * self.shape.hidden * BF16
    }

    /// BF16 bytes of the slot-major buffer, `[rows * top_k, hidden]`.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_count() * self.shape.hidden * BF16
    }

    /// The structural properties every plan must have, stated once.
    ///
    /// Task 0020's twenty-three findings shared one shape -- an individually
    /// reasonable step leaving the structure inconsistent in a combination
    /// nobody had written a test for -- and its answer was one function stating
    /// the invariants plus a sweep that calls it. The same method, applied here
    /// rather than rediscovered: the sweep in `tests/expert_plan_matrix.rs`
    /// calls this for every combination it generates.
    pub fn check_invariants(&self) -> Result<()> {
        let slots = self.slot_count();
        let mut covered = vec![false; slots as usize];
        let mut previous: Option<u32> = None;
        for group in &self.groups {
            if let Some(p) = previous
                && group.expert <= p
            {
                return Err(invalid(
                    "groups",
                    format!("expert {} follows {p}; groups must ascend", group.expert),
                ));
            }
            previous = Some(group.expert);
            if group.rows.len() != group.slots.len() || group.rows.is_empty() {
                return Err(invalid(
                    "group",
                    format!(
                        "expert {} has {} row(s) and {} slot(s)",
                        group.expert,
                        group.rows.len(),
                        group.slots.len()
                    ),
                ));
            }
            if group.decision.reuse_rows != group.rows.len() as u64 {
                return Err(invalid(
                    "group",
                    format!(
                        "expert {} reports {} reuse row(s) against {} assigned",
                        group.expert,
                        group.decision.reuse_rows,
                        group.rows.len()
                    ),
                ));
            }
            if group.decision.chosen != group.placement.candidate()
                || group.decision.rejected != group.decision.chosen.other()
            {
                return Err(invalid(
                    "group",
                    format!("expert {}'s decision and placement disagree", group.expert),
                ));
            }
            for (row, slot) in group.rows.iter().zip(&group.slots) {
                let slot = *slot as u64;
                if slot >= slots || slot / self.shape.top_k != u64::from(*row) {
                    return Err(invalid(
                        "group",
                        format!("slot {slot} does not belong to row {row}"),
                    ));
                }
                if covered[slot as usize] {
                    return Err(invalid("group", format!("slot {slot} is claimed twice")));
                }
                covered[slot as usize] = true;
            }
            match group.placement {
                Placement::Device(uuid) if uuid != self.device => {
                    return Err(invalid("group", "a group names another device".into()));
                }
                Placement::Host(p) if p != self.host_placement => {
                    return Err(invalid("group", "a group names another host node".into()));
                }
                _ => {}
            }
        }
        // Every slot computed exactly once. A row reduced over `top_k - 1` slots
        // is a wrong answer, not a degraded one.
        if let Some(slot) = covered.iter().position(|c| !c) {
            return Err(invalid(
                "groups",
                format!("slot {slot} is computed by nothing"),
            ));
        }
        if self.reduction_order.len() as u64 != slots {
            return Err(invalid(
                "reduction_order",
                format!("{} entries for {slots} slot(s)", self.reduction_order.len()),
            ));
        }
        let top_k = self.shape.top_k as usize;
        for (r, row) in self.reduction_order.chunks_exact(top_k).enumerate() {
            let mut seen = vec![false; top_k];
            for position in row {
                let j = *position as usize;
                if j >= top_k || seen[j] {
                    return Err(invalid(
                        "reduction_order",
                        format!("row {r} is not a permutation of {top_k} slot positions"),
                    ));
                }
                seen[j] = true;
            }
        }
        if self.queue_capacity == 0 || self.queue_capacity as usize > self.groups.len() {
            return Err(invalid(
                "queue_capacity",
                format!(
                    "{} against {} group(s)",
                    self.queue_capacity,
                    self.groups.len()
                ),
            ));
        }
        Ok(())
    }
}

/// A refused plan, with the report that explains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlanRefused {
    pub error: Error,
    /// Why each candidate was unavailable, when it was.
    pub device: Option<RejectionReason>,
    pub host: Option<RejectionReason>,
}

impl core::fmt::Display for ExpertPlanRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)?;
        if let Some(reason) = self.device {
            write!(f, "; gpu-grouped: {reason}")?;
        }
        if let Some(reason) = self.host {
            write!(f, "; cpu-tiled: {reason}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ExpertPlanRefused {}

impl From<ExpertPlanRefused> for Error {
    fn from(value: ExpertPlanRefused) -> Self {
        value.error
    }
}

/// Read the shape out of the graph's own parameters.
///
/// Taking `OpParams` rather than five integers is deliberate: the plan's shape
/// must be the node's shape, and a constructor that accepted loose numbers would
/// let a caller plan for a layer the graph does not contain.
pub fn shape_of(mlp: &OpParams, combine: &OpParams) -> Result<(ExpertShape, CombineOrder)> {
    let OpParams::ExpertMlp {
        hidden,
        intermediate,
        experts,
        top_k,
        activation,
    } = *mlp
    else {
        return Err(invalid("mlp", "this is not an ExpertMlp node".into()));
    };
    let OpParams::Combine {
        hidden: combine_hidden,
        top_k: combine_top_k,
        order,
    } = *combine
    else {
        return Err(invalid("combine", "this is not a Combine node".into()));
    };
    if combine_hidden != hidden || combine_top_k != top_k {
        return Err(invalid(
            "combine",
            format!(
                "combine is [{combine_hidden}, {combine_top_k}] against the expert node's \
                 [{hidden}, {top_k}]"
            ),
        ));
    }
    Ok((
        ExpertShape {
            hidden,
            intermediate,
            experts,
            top_k,
            activation,
        },
        order,
    ))
}

/// Compile one routed layer's expert work into a plan.
///
/// `route_experts` is the route table's expert ids, slot-major: entry
/// `r * top_k + j` is the expert row `r` selected at position `j`. It is the
/// output of task 0019's `Route`, which is why this function does not compute
/// it -- the union of experts a batch demands is a *result* of the routing
/// equation, and duplicating that equation here would be a second router.
#[allow(clippy::result_large_err)]
pub fn compile_experts(
    mlp: &OpParams,
    combine: &OpParams,
    route_experts: &[u32],
    budget: &ExpertBudget,
    policy: &ExpertPolicy,
    topology: Option<&NumaTopology>,
    kernels: Option<ExpertKernels<'_>>,
) -> std::result::Result<ExpertPlan, ExpertPlanRefused> {
    let refuse = |error: Error| ExpertPlanRefused {
        error,
        device: None,
        host: None,
    };
    let (shape, order) = shape_of(mlp, combine).map_err(refuse)?;
    if shape.hidden == 0 || shape.intermediate == 0 || shape.experts == 0 || shape.top_k == 0 {
        return Err(refuse(invalid(
            "shape",
            format!(
                "degenerate: hidden {}, intermediate {}, experts {}, top_k {}",
                shape.hidden, shape.intermediate, shape.experts, shape.top_k
            ),
        )));
    }
    if shape.top_k > u64::from(MAX_TOP_K) {
        return Err(refuse(invalid(
            "top_k",
            format!("{} exceeds the checked maximum {MAX_TOP_K}", shape.top_k),
        )));
    }
    if route_experts.is_empty() || !(route_experts.len() as u64).is_multiple_of(shape.top_k) {
        return Err(refuse(invalid(
            "route",
            format!(
                "{} route entries is not a positive multiple of top_k {}",
                route_experts.len(),
                shape.top_k
            ),
        )));
    }
    let rows = route_experts.len() as u64 / shape.top_k;
    if let Some(bad) = route_experts
        .iter()
        .find(|e| u64::from(**e) >= shape.experts)
    {
        return Err(refuse(invalid(
            "route",
            format!("expert {bad} of {}", shape.experts),
        )));
    }
    // A row selecting the same expert twice would give that expert two slots of
    // the same row and make the reduction's permutation ambiguous. Task 0019's
    // selection cannot produce it; refusing rather than assuming is what keeps
    // that a checked fact instead of a remembered one.
    for (r, row) in route_experts.chunks_exact(shape.top_k as usize).enumerate() {
        let mut sorted = row.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(refuse(invalid(
                "route",
                format!("row {r} selects one expert twice"),
            )));
        }
    }

    // Expert-major grouping, ascending expert id, each group's rows ascending.
    let mut grouped: BTreeMap<u32, (Vec<u32>, Vec<u32>)> = BTreeMap::new();
    for (index, expert) in route_experts.iter().enumerate() {
        let entry = grouped.entry(*expert).or_default();
        entry.0.push((index as u64 / shape.top_k) as u32);
        entry.1.push(index as u32);
    }
    let group_count = grouped.len();

    let chunk_bytes = shape
        .chunk_bytes()
        .ok_or_else(|| refuse(overflow("one expert's chunk extent")))?;
    let queue_capacity = policy
        .max_inflight_orders
        .min(u32::try_from(group_count).unwrap_or(u32::MAX))
        .max(1);

    // --- plan-wide admissibility, decided once and from bounds ---------------

    let activation_bytes = rows
        .checked_mul(shape.hidden)
        .and_then(|v| v.checked_mul(BF16))
        .ok_or_else(|| refuse(overflow("the activation block")))?;
    let slot_bytes = rows
        .checked_mul(shape.top_k)
        .and_then(|v| v.checked_mul(shape.hidden))
        .and_then(|v| v.checked_mul(BF16))
        .ok_or_else(|| refuse(overflow("the slot buffer")))?;
    // The workspace bound uses `rows` rather than the largest group, because the
    // largest group is not known until the assignment exists and the assignment
    // must not depend on the envelope. It over-reserves by construction, which
    // is the conservative direction.
    let device_workspace_bytes = u64::from(queue_capacity)
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(shape.intermediate))
        .and_then(|v| v.checked_mul(F32))
        .ok_or_else(|| refuse(overflow("the device workspace")))?;
    let device_needed = activation_bytes
        .checked_add(slot_bytes)
        .and_then(|v| v.checked_add(device_workspace_bytes))
        .ok_or_else(|| refuse(overflow("the device envelope")))?;
    let host_workspace_bytes = shape
        .host_workspace_values(policy.cpu_tile_lanes)
        .and_then(|v| v.checked_mul(F32))
        .and_then(|v| v.checked_mul(u64::from(queue_capacity)))
        .ok_or_else(|| refuse(overflow("the host workspace")))?;
    let index_staging_bytes = rows
        .checked_mul(shape.top_k)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| refuse(overflow("the index staging")))?;
    let host_buffer_bytes = activation_bytes
        .checked_add(slot_bytes)
        .and_then(|v| v.checked_add(rows * shape.hidden * BF16))
        .ok_or_else(|| refuse(overflow("the host buffers")))?;

    if policy.device == StrategyControl::Required && policy.host == StrategyControl::Required {
        // Each is blocked by the other, and the refusal says so on both sides:
        // a refusal that named no candidate would be the one shape of report
        // this plan is not allowed to produce.
        return Err(ExpertPlanRefused {
            error: invalid(
                "strategy",
                "both candidates are `required`; a group runs on one of them".into(),
            ),
            device: Some(RejectionReason::OtherCandidateRequired),
            host: Some(RejectionReason::OtherCandidateRequired),
        });
    }
    // Kernel selection is part of compilation, not of launching. A device
    // candidate with no qualified kernel is inadmissible for that reason and
    // says so, rather than being discovered at the launch site where writing a
    // fallback would be the easy thing to do.
    let selected = match kernels {
        Some(kernels) => select_expert_kernel(shape, rows, kernels).ok(),
        None => None,
    };
    let device_unavailable = if policy.host == StrategyControl::Required {
        // `required` on one candidate removes the other from consideration.
        // Erroring instead -- on the ground that a device group would silently
        // disable what was asked for -- would make `host = required` fail on
        // every plan a device could have served, which is not an actionable
        // error, it is a refusal to plan.
        Some(RejectionReason::OtherCandidateRequired)
    } else if !policy.device.may_select() {
        Some(RejectionReason::CandidateOff)
    } else if selected.is_none() {
        Some(RejectionReason::NoQualifiedKernel)
    } else if device_needed > budget.device_arena_free_bytes {
        Some(RejectionReason::DeviceArenaTooSmall {
            needed: device_needed,
            free: budget.device_arena_free_bytes,
        })
    } else {
        None
    };
    let host_unavailable = if policy.device == StrategyControl::Required {
        Some(RejectionReason::OtherCandidateRequired)
    } else if !policy.host.may_select() {
        Some(RejectionReason::CandidateOff)
    } else if host_workspace_bytes > budget.host_workspace_bytes {
        Some(RejectionReason::HostWorkspaceTooSmall {
            needed: host_workspace_bytes,
            free: budget.host_workspace_bytes,
        })
    } else if host_buffer_bytes > budget.host_buffer_bytes {
        Some(RejectionReason::HostBuffersTooSmall {
            needed: host_buffer_bytes,
            free: budget.host_buffer_bytes,
        })
    } else {
        None
    };

    // `required` must produce an actionable error rather than silently
    // disabling what was asked for (document 04, encoded in `StrategyControl`).
    if let Some(reason) = device_unavailable
        && policy.device == StrategyControl::Required
        && reason != RejectionReason::OtherCandidateRequired
    {
        return Err(ExpertPlanRefused {
            error: required_error("gpu-grouped", reason),
            device: Some(reason),
            host: host_unavailable,
        });
    }
    if let Some(reason) = host_unavailable
        && policy.host == StrategyControl::Required
        && reason != RejectionReason::OtherCandidateRequired
    {
        return Err(ExpertPlanRefused {
            error: required_error("cpu-tiled", reason),
            device: device_unavailable,
            host: Some(reason),
        });
    }
    if device_unavailable.is_some() && host_unavailable.is_some() {
        return Err(ExpertPlanRefused {
            error: Error::CapacityExceeded {
                tier: None,
                requested_bytes: device_needed.max(host_workspace_bytes),
                available_bytes: budget
                    .device_arena_free_bytes
                    .max(budget.host_workspace_bytes),
            },
            device: device_unavailable,
            host: host_unavailable,
        });
    }

    // --- host placement -------------------------------------------------------

    let host_placement = match (policy.host_placement, topology) {
        (StrategyControl::Off, _) => HostPlacement::Unspecified,
        (_, Some(t)) => t.placement_for_pci(&budget.device_pci_bus_id),
        (_, None) => HostPlacement::Unspecified,
    };
    if policy.host_placement == StrategyControl::Required
        && host_placement == HostPlacement::Unspecified
    {
        return Err(ExpertPlanRefused {
            error: invalid(
                "host_placement",
                format!(
                    "placement is required and the topology gives no node for {}",
                    budget.device_pci_bus_id
                ),
            ),
            device: device_unavailable,
            host: host_unavailable,
        });
    }

    // --- per-group choice -----------------------------------------------------

    let displaceable = budget.displaceable();
    let mut groups = Vec::with_capacity(group_count);
    let mut demand_bytes = 0u64;
    for (expert, (rows_of, slots_of)) in grouped {
        let reuse_rows = rows_of.len() as u64;
        let already_resident = budget.resident_experts.contains(&expert);
        let transfer_bytes = if already_resident { 0 } else { chunk_bytes };
        let bytes_per_row = transfer_bytes.div_ceil(reuse_rows);

        let device_reason = match device_unavailable {
            Some(reason) => Some(reason),
            None if transfer_bytes > displaceable => Some(RejectionReason::DeviceCacheTooSmall {
                incoming: transfer_bytes,
                displaceable,
            }),
            None if bytes_per_row > policy.max_transfer_bytes_per_row => {
                Some(RejectionReason::TransferNotAmortised {
                    bytes_per_row,
                    limit: policy.max_transfer_bytes_per_row,
                })
            }
            None => None,
        };
        let (chosen, reason) = match (device_reason, host_unavailable) {
            (None, _) => (Candidate::Device, RejectionReason::NotPreferred),
            (Some(reason), None) => (Candidate::Host, reason),
            (Some(device), Some(host)) => {
                // Both refused this expert. `required` was already handled
                // plan-wide; what is left is a genuine dead end.
                return Err(ExpertPlanRefused {
                    error: Error::CapacityExceeded {
                        tier: Some(Tier::Device(DeviceTier::ExpertCache)),
                        requested_bytes: transfer_bytes,
                        available_bytes: displaceable,
                    },
                    device: Some(device),
                    host: Some(host),
                });
            }
        };
        // A `required` device candidate passed the plan-wide gate but can
        // still be inadmissible for one expert -- the cache is per chunk. That
        // is exactly the case document 04 wants an actionable error for, and
        // falling back here would be the silent disabling it forbids.
        if chosen == Candidate::Host && policy.device == StrategyControl::Required {
            return Err(ExpertPlanRefused {
                error: required_error("gpu-grouped", reason),
                device: Some(reason),
                host: host_unavailable,
            });
        }
        if chosen == Candidate::Device && !already_resident {
            demand_bytes = demand_bytes
                .checked_add(chunk_bytes)
                .ok_or_else(|| refuse(overflow("the residency demand")))?;
        }
        groups.push(ExpertGroup {
            expert,
            rows: rows_of,
            slots: slots_of,
            placement: match chosen {
                Candidate::Device => Placement::Device(budget.device),
                Candidate::Host => Placement::Host(host_placement),
            },
            chunk_bytes,
            decision: GroupDecision {
                chosen,
                rejected: chosen.other(),
                reason,
                reuse_rows,
                transfer_bytes,
                bytes_per_row,
                already_resident,
            },
        });
    }

    // --- reduction order, and the envelope the assignment actually needs -----

    let reduction_order = reduction_permutation(route_experts, shape.top_k as usize, order);

    let any_device = groups
        .iter()
        .any(|g| g.placement.candidate() == Candidate::Device);
    let any_host = groups
        .iter()
        .any(|g| g.placement.candidate() == Candidate::Host);
    let mut device = Vec::new();
    if any_device {
        let aligned = |bytes: u64| {
            align_device(bytes).ok_or_else(|| refuse(overflow("an aligned device region")))
        };
        device.push((
            DeviceTier::Activations,
            aligned(activation_bytes)? + aligned(slot_bytes)?,
        ));
        device.push((
            DeviceTier::KernelWorkspace,
            aligned(device_workspace_bytes)?,
        ));
        // The two `RouteIndex` operands: which rows a launch serves and where
        // each result goes. Sized for the largest group a plan of this shape can
        // produce, which is every slot, so a group's staging is admitted before
        // the group is known.
        device.push((
            DeviceTier::TransferStaging,
            aligned(index_staging_bytes)? * 2,
        ));
    }
    let mut host = vec![(HostTier::Pageable, host_buffer_bytes)];
    if any_host {
        host.push((HostTier::CpuWorkspace, host_workspace_bytes));
    }

    let plan = ExpertPlan {
        kernel: if any_device { selected } else { None },
        shape,
        rows,
        device: budget.device,
        host_placement,
        groups,
        queue_capacity,
        cpu_tile_lanes: policy.cpu_tile_lanes,
        reduction_order,
        envelope: ExpertEnvelope {
            device,
            host,
            residency_demand_bytes: demand_bytes,
        },
        policy: *policy,
    };
    plan.check_invariants().map_err(refuse)?;
    Ok(plan)
}

fn required_error(candidate: &'static str, reason: RejectionReason) -> Error {
    Error::InvalidRequest {
        field: "strategy",
        detail: format!("{candidate} is `required` and is not admissible: {reason}"),
    }
}

/// The widest `top_k` the permutation checks can represent in a bitmask.
pub const MAX_TOP_K: u32 = 64;

/// The order each row's slots are reduced in, as an explicit permutation.
///
/// `AscendingExpertId` sorts by the expert a slot selected, with the lower
/// expert id first -- the same tie rule task 0019 fixed for selection, applied
/// to reduction. `SelectionOrder` is the identity: the row's own order, highest
/// score first, is how the route table already arrives.
fn reduction_permutation(route_experts: &[u32], top_k: usize, order: CombineOrder) -> Vec<u32> {
    let mut out = Vec::with_capacity(route_experts.len());
    for row in route_experts.chunks_exact(top_k) {
        let mut positions: Vec<u32> = (0..top_k as u32).collect();
        match order {
            CombineOrder::SelectionOrder => {}
            CombineOrder::AscendingExpertId => {
                positions.sort_by_key(|j| (row[*j as usize], *j));
            }
        }
        out.extend(positions);
    }
    out
}

/// Select the one catalogue descriptor that serves this operation on this
/// device, or fail.
///
/// Exactly one, or none: an ambiguous catalogue is a defect, not a choice to be
/// made by ordering. The match is on semantic operation (gate transform
/// included), operand roles, precisions, accumulation, rounding, layout, SM and
/// shape bounds -- never on a model name, which is document 02's rule and
/// task 0012's mechanism.
pub fn select_expert_kernel(
    shape: ExpertShape,
    rows: u64,
    kernels: ExpertKernels<'_>,
) -> Result<SemanticKernelDescriptor> {
    let operation = SemanticKernelOp::ExpertMlp(shape.gate_transform());
    let assignments = rows
        .checked_mul(shape.top_k)
        .ok_or_else(|| invalid("rows", "assignment count overflows".into()))?;
    let matches: Vec<_> = kernels
        .catalogue
        .descriptors()
        .iter()
        .filter(|d| {
            d.operation == operation
                && d.sm.major == kernels.capability.compute_major
                && d.sm.minor == kernels.capability.compute_minor
                && d.layout == moxie_types::TensorLayout::ContiguousRowMajorV1
                && d.workspace == WorkspaceExpression::RowsTimesIntermediateF32
                && assignments <= d.shape.max_rows
                && shape.hidden <= d.shape.max_input
                && shape.hidden <= d.shape.max_output
        })
        .collect();
    if matches.len() != 1 {
        return Err(Error::UnsupportedKernel {
            operation: "expert_mlp",
            detail: format!(
                "expected exactly one descriptor for {} on sm_{}{} at {assignments} assignment(s) \
                 of width {}; found {}",
                operation.name(),
                kernels.capability.compute_major,
                kernels.capability.compute_minor,
                shape.hidden,
                matches.len()
            ),
        });
    }
    let descriptor = matches[0];
    if descriptor.symbols.len() != 2 {
        return Err(Error::UnsupportedKernel {
            operation: "expert_mlp",
            detail: "a grouped expert descriptor names a projection and a down symbol".into(),
        });
    }
    Ok(descriptor.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending_expert_id_sorts_and_selection_order_does_not() {
        let route = [5u32, 1, 9, 3];
        assert_eq!(
            reduction_permutation(&route, 4, CombineOrder::AscendingExpertId),
            vec![1, 3, 0, 2]
        );
        assert_eq!(
            reduction_permutation(&route, 4, CombineOrder::SelectionOrder),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn one_expert_is_three_times_its_intermediate_by_hidden_in_bf16() {
        // The designated artifact's layer 0, from the bring-up record.
        let shape = ExpertShape {
            hidden: 2816,
            intermediate: 704,
            experts: 128,
            top_k: 8,
            activation: ExpertActivation::GeGlu,
        };
        assert_eq!(shape.chunk_bytes(), Some(11_894_784));
        assert_eq!(shape.chunk_bytes(), Some(7_929_856 + 3_964_928));
    }
}
