//! Hardware and kernel capabilities, and the shared `off` / `auto` / `required`
//! control semantics.
//!
//! Document 04: "`off`, `auto`, and `required` are common control semantics.
//! Report the selected plan and why a requested strategy was rejected." The
//! asymmetry encoded here is the whole point: `auto` may decline a strategy and
//! must say so; `required` must produce an actionable error rather than silently
//! disabling what was asked for.

use core::fmt;

use crate::{
    AccumulationPolicy, ActivationPrecision, TensorLayout, WeightPrecision, ids::DeviceUuid,
};

/// Stable identity of one audited semantic kernel implementation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub String);

/// Which gate transform a gated feed-forward kernel implements.
///
/// A dispatch key, and a **closed** one: document 02 keeps SwiGLU and GeGLU as
/// distinct operations because they are distinct functions with distinct
/// declared rounding boundaries, and task 0019 carried that into
/// `ExpertActivation`. This is that distinction at the kernel boundary, so a
/// GeGLU kernel can never be selected for a SiLU-gated node. It is spelled here
/// rather than reached for from the graph crate because kernel dispatch keys
/// live at the bottom of the dependency graph (document 02).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateTransform {
    /// `bf16(gelu_tanh(gate)) * up`.
    GeluTanh,
    /// `silu(gate) * up`, evaluated in FP64 and rounded once.
    Silu,
}

impl GateTransform {
    pub const fn name(self) -> &'static str {
        match self {
            Self::GeluTanh => "gelu_tanh",
            Self::Silu => "silu",
        }
    }
}

/// Closed semantic operations that may cross the planning/execution boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SemanticKernelOp {
    /// Token-id gather with the operation's optional output scale.
    Embedding,
    Linear,
    /// One full-device linear that evaluates declared input-axis blocks from
    /// zero in FP32, combines those blocks in order, and rounds once.
    ///
    /// This is deliberately distinct from [`SemanticKernelOp::LinearPartial`]
    /// so the single-device S>1 reference cannot share the TP partial path.
    LinearSplit,
    /// A row-parallel linear partial: BF16 inputs, FP32 output, no rounding.
    LinearPartial,
    RmsNorm,
    /// RMSNorm over independent contiguous groups (one group per head).
    GroupedRmsNorm,
    /// Host-angle-table rotary position encoding.
    Rope,
    /// `bf16(bf16(gelu_tanh(g)) * u)`.
    GeGlu,
    Residual,
    /// Residual with the declared intermediate BF16 boundary and scale.
    ScaledResidual,
    /// Unrounded FP32 vocabulary projection, optionally soft-capped.
    VocabProjection,
    /// Selects routed experts and emits ids with FP32 coefficients.
    Route,
    /// The gated expert feed-forward, evaluated per selected slot. The gate
    /// transform is part of the operation's identity, not a parameter of it.
    ExpertMlp(GateTransform),
    /// Combines routed expert slots in the declared expert order.
    Combine,
    /// Attention over paged key/value state: prefill, append and decode.
    ///
    /// One operation rather than three, because whole prefill, a prefill chunk
    /// and a single decode row differ only in how many query rows a launch
    /// serves. Visibility, head grouping and the score scale are semantic
    /// *parameters* of the graph node, not separate operations: a descriptor
    /// per mask would make the catalogue grow with every model that windows
    /// differently, which is the shape document 02 forbids.
    PagedAttention,
}

impl SemanticKernelOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Linear => "linear",
            Self::LinearSplit => "linear_split",
            Self::LinearPartial => "linear_partial",
            Self::RmsNorm => "rms_norm",
            Self::GroupedRmsNorm => "grouped_rms_norm",
            Self::Rope => "rope",
            Self::GeGlu => "geglu",
            Self::Residual => "residual",
            Self::ScaledResidual => "scaled_residual",
            Self::VocabProjection => "vocab_projection",
            Self::Route => "route",
            Self::ExpertMlp(GateTransform::GeluTanh) => "expert_mlp_gelu_tanh",
            Self::ExpertMlp(GateTransform::Silu) => "expert_mlp_silu",
            Self::Combine => "combine",
            Self::PagedAttention => "paged_attention",
        }
    }
}

/// Operand role and stored precision accepted by a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelOperand {
    Activation(ActivationPrecision),
    Weight(WeightPrecision),
    /// A `u32` selection index per routed slot.
    ///
    /// Document 02 requires integer roles to be described as integer roles:
    /// "Token IDs, positions, page/group/sparse indices and masks also need
    /// integer/boolean descriptor roles; they are not quantized weights or
    /// floating activations." A grouped expert kernel takes exactly one such
    /// operand -- which rows this launch serves -- and a descriptor that spelled
    /// it as an activation precision would invite a precision predicate to be
    /// applied to a row number.
    RouteIndex,
    /// A token id consumed by an embedding gather.
    TokenIndex,
    /// An absolute sequence position consumed by RoPE or attention.
    PositionIndex,
    /// A `u32` physical page identity per logical page of one layer's history.
    ///
    /// The other half of document 02's integer roles, and distinct from
    /// [`KernelOperand::RouteIndex`] because it indexes *storage* rather than
    /// selection: a route index says which expert a row goes to, a page index
    /// says where a row already is. Reading one as the other is a wrong answer
    /// with the right shape, so they are separate roles.
    PageIndex,
}

/// The one output-rounding boundary qualified by task 0012.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoundingProfile {
    /// The operation stores its FP32 result without a BF16 output boundary.
    Unrounded,
    FinalBf16Rne,
}

/// Hardware identity used for dispatch. Device names and ordinals are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion {
    pub major: u32,
    pub minor: u32,
}

impl SmVersion {
    pub const SM86: Self = Self { major: 8, minor: 6 };
    pub const SM120: Self = Self {
        major: 12,
        minor: 0,
    };

    pub fn name(self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }
}

/// Checked shape domain of the initial BF16 semantic package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelShapeBounds {
    pub max_rows: u64,
    pub max_input: u64,
    pub max_output: u64,
}

/// Exact logical workspace expression, evaluated by the pure planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkspaceExpression {
    Zero,
    RowsTimesF32,
    /// `rows * rotary_pairs * 2 * 4`: one FP32 cosine/sine pair per row.
    RowsTimesRopeAnglesF32,
    /// `rows * intermediate * 4`: one FP32 gated-intermediate vector per row.
    ///
    /// The expert feed-forward's intermediate is `intermediate` wide and does
    /// not fit in registers at the designated artifact's 704, so it is a
    /// declared workspace rather than a hidden allocation inside a launch. The
    /// width is not baked into the descriptor: it is the *operation's*
    /// parameter, and a catalogue entry per intermediate width would make
    /// kernel selection depend on a shape the kernel does not care about.
    RowsTimesIntermediateF32,
}

impl WorkspaceExpression {
    /// Bytes, for an expression that needs only the row count.
    ///
    /// `None` for [`WorkspaceExpression::RowsTimesIntermediateF32`], which
    /// cannot be evaluated without the operation's width. Returning `None`
    /// rather than a plausible number is the point: a caller that ignores it
    /// gets no workspace figure instead of a wrong one.
    pub fn evaluate(self, rows: u64) -> Option<u64> {
        match self {
            Self::Zero => Some(0),
            Self::RowsTimesF32 => rows.checked_mul(4),
            Self::RowsTimesRopeAnglesF32 | Self::RowsTimesIntermediateF32 => None,
        }
    }

    /// Bytes, given the operation's intermediate width where one is needed.
    pub fn evaluate_with(self, rows: u64, intermediate: u64) -> Option<u64> {
        match self {
            Self::RowsTimesRopeAnglesF32 => rows.checked_mul(intermediate)?.checked_mul(8),
            Self::RowsTimesIntermediateF32 => rows
                .checked_mul(intermediate)
                .and_then(|v| v.checked_mul(4)),
            other => other.evaluate(rows),
        }
    }
}

/// Ordered symbols needed to execute one semantic node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelSymbol(pub String);

/// Immutable data selected by the pure planner. It carries no callback, image
/// pointer, model identity or device ordinal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticKernelDescriptor {
    pub id: KernelId,
    pub abi_version: u32,
    pub operation: SemanticKernelOp,
    pub inputs: Vec<KernelOperand>,
    pub output: ActivationPrecision,
    pub accumulation: AccumulationPolicy,
    pub rounding: RoundingProfile,
    pub layout: TensorLayout,
    pub shape: KernelShapeBounds,
    pub sm: SmVersion,
    pub workspace: WorkspaceExpression,
    pub image_sha256: [u8; 32],
    pub symbols: Vec<KernelSymbol>,
}

impl SemanticKernelDescriptor {
    /// Clone without an allocation that can abort the process.
    ///
    /// `Clone` grows three heap values -- the identifier, the operand list and
    /// the symbol list -- and every one of them aborts when the allocator
    /// refuses. Selection returns an owned descriptor on its **success** path,
    /// which is the path a caller has no refusal to fall back to. Each buffer
    /// is reserved before it is filled, so a refusal is a typed capacity
    /// error rather than an abort.
    ///
    /// `moxie-types` cannot depend on `moxie-memory` (the dependency runs the
    /// other way), so this repeats its fallible-string vocabulary rather than
    /// sharing it.
    pub fn try_clone(&self) -> crate::Result<Self> {
        fn no_room() -> crate::Error {
            crate::Error::CapacityExceeded {
                tier: None,
                requested_bytes: 0,
                available_bytes: 0,
            }
        }
        fn try_string(source: &str) -> crate::Result<String> {
            let mut out = String::new();
            out.try_reserve_exact(source.len()).map_err(|_| no_room())?;
            out.push_str(source);
            Ok(out)
        }
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(self.inputs.len())
            .map_err(|_| no_room())?;
        inputs.extend_from_slice(&self.inputs);
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(self.symbols.len())
            .map_err(|_| no_room())?;
        for symbol in &self.symbols {
            symbols.push(KernelSymbol(try_string(&symbol.0)?));
        }
        Ok(Self {
            id: KernelId(try_string(&self.id.0)?),
            abi_version: self.abi_version,
            operation: self.operation,
            inputs,
            output: self.output,
            accumulation: self.accumulation,
            rounding: self.rounding,
            layout: self.layout,
            shape: self.shape,
            sm: self.sm,
            workspace: self.workspace,
            image_sha256: self.image_sha256,
            symbols,
        })
    }
}

/// Read-only, closed catalogue injected into planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCatalogue {
    descriptors: Vec<SemanticKernelDescriptor>,
    digest: [u8; 32],
}

impl KernelCatalogue {
    pub fn new(mut descriptors: Vec<SemanticKernelDescriptor>) -> crate::Result<Self> {
        descriptors.sort_by(|a, b| a.id.cmp(&b.id));
        for (index, first) in descriptors.iter().enumerate() {
            for second in &descriptors[index + 1..] {
                if first.id == second.id || indistinguishable(first, second) {
                    return Err(crate::Error::InvalidArtifact {
                        detail: format!(
                            "duplicate or indistinguishable kernel descriptors {} and {}",
                            first.id.0, second.id.0
                        )
                        .into(),
                    });
                }
            }
        }
        let digest = catalogue_digest(&descriptors);
        Ok(Self {
            descriptors,
            digest,
        })
    }

    pub fn descriptors(&self) -> &[SemanticKernelDescriptor] {
        &self.descriptors
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

fn indistinguishable(a: &SemanticKernelDescriptor, b: &SemanticKernelDescriptor) -> bool {
    a.operation == b.operation
        && a.inputs == b.inputs
        && a.output == b.output
        && a.accumulation == b.accumulation
        && a.rounding == b.rounding
        && a.layout == b.layout
        && a.shape == b.shape
        && a.sm == b.sm
        && a.workspace == b.workspace
}

fn catalogue_digest(descriptors: &[SemanticKernelDescriptor]) -> [u8; 32] {
    // Four domain-separated FNV-1a lanes over an explicit Debug rendering.
    // This is an identity digest, not a security boundary; image integrity is
    // separately the build-produced SHA-256 in every descriptor.
    let bytes = format!("{descriptors:?}");
    let mut out = [0u8; 32];
    for lane in 0..4u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ lane.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in bytes.as_bytes() {
            h ^= u64::from(*byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        out[(lane as usize) * 8..(lane as usize + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrategyControl {
    /// Never select this strategy.
    Off,
    /// Select it when it is applicable, admissible and measured to help.
    #[default]
    Auto,
    /// Select it, or fail with a typed error naming why it is unavailable.
    Required,
}

impl StrategyControl {
    /// Whether an unavailable strategy is an error rather than a fallback.
    pub const fn must_error_when_unavailable(self) -> bool {
        matches!(self, StrategyControl::Required)
    }

    /// Whether the planner is permitted to consider this strategy at all.
    pub const fn may_select(self) -> bool {
        !matches!(self, StrategyControl::Off)
    }
}

impl fmt::Display for StrategyControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StrategyControl::Off => "off",
            StrategyControl::Auto => "auto",
            StrategyControl::Required => "required",
        })
    }
}

/// What a device can actually do, from device queries and launch probes -- never
/// inferred from a marketing name (document 03).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapability {
    /// Diagnostic only. Valid inside one process's visible device set and
    /// nowhere else; a plan, a manifest or a result names the UUID (AGENTS.md).
    pub ordinal: u32,
    /// Stable identity. Evidence records this, not the ordinal (document 07).
    pub uuid: DeviceUuid,
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub total_memory_bytes: u64,
    pub multiprocessor_count: u32,
    /// PCI bus id, so a record can be reconciled with `nvidia-smi` and with the
    /// NUMA topology.
    pub pci_bus_id: String,
    /// Peer access, per peer ordinal. Discovered, never assumed: on this machine
    /// it depends on a patched kernel module that a driver upgrade can remove.
    /// See docs/evidence/topology-p2p.md.
    pub peer_access: Vec<(u32, bool)>,
    /// The largest grid this device will launch, per dimension.
    ///
    /// Discovered for the same reason peer access is. The `y` and `z` limits are
    /// **65,535** on every NVIDIA architecture to date while `x` is `2^31 - 1`,
    /// and a binding that assumed a `u32` fits all three would submit its
    /// operands and only then learn otherwise from `cuLaunchKernel` — turning a
    /// refusal that was knowable before any device work into an unknown
    /// submission. A launch geometry is checked against these before it is
    /// admitted.
    pub max_grid: (u32, u32, u32),
}

impl DeviceCapability {
    /// `sm_86`, `sm_120`, ... Used as a kernel dispatch key, together with shape
    /// and layout. Never the device name (document 02).
    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.compute_major, self.compute_minor)
    }

    pub fn can_access_peer(&self, peer: u32) -> bool {
        self.peer_access
            .iter()
            .find(|(p, _)| *p == peer)
            .map(|(_, ok)| *ok)
            .unwrap_or(false)
    }
}

/// One device's capacity as the driver reported it at one instant.
///
/// This is a **reading**, not a reservation and not a property of the card.
/// `free_bytes` includes whatever other processes hold, and re-measuring may
/// return something different; document 03 is explicit that an admission report
/// shows "physical capacity, already committed resources, reserved peak, and
/// remaining headroom", and only the first of those is a constant.
///
/// It lives here, at the bottom of the dependency graph, because the crate that
/// takes the reading sits *below* the crate that turns it into a budget:
/// document 02 permits `memory` -> `cuda` and forbids the reverse, so `cuda`
/// cannot name a `CapacitySnapshot`. Putting the descriptor in `types` lets both
/// sides use it without `cuda` reaching upward, and leaves the composition root
/// free to do the wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredDevice {
    /// The device's identity. Every decision and every record keys on this.
    pub uuid: DeviceUuid,
    /// The ordinal this reading was taken through, for reconciling a log with
    /// `nvidia-smi`. Never an identity, never a map key.
    pub ordinal_label: u32,
    pub name: String,
    /// `sm_86`, `sm_120`, ... A kernel dispatch key, not a marketing name.
    pub sm: String,
    pub pci_bus_id: String,
    pub multiprocessor_count: u32,
    /// Total device memory.
    pub total_bytes: u64,
    /// Free device memory at the moment of the reading, other processes
    /// included.
    pub free_bytes: u64,
}

/// Which view of host memory a reading was taken through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostLimit {
    /// No cgroup limit applies; the machine's own figures govern.
    Machine,
    /// A cgroup v2 limit binds, from this cgroup or one of its ancestors.
    ///
    /// Total and available headroom minimise independently over the chain and
    /// can bind at different levels: the smallest `memory.max` caps the total,
    /// while the smallest saturating `memory.max - memory.current` caps what
    /// a new allocation can obtain. A sibling scope holding most of an
    /// ancestor slice is the ordinary shape where they diverge, so both
    /// binding points are recorded rather than reading headroom off the
    /// tightest limit.
    Cgroup {
        /// Path of the smallest `memory.max`; binds the total.
        path: String,
        /// The smallest `memory.max` on the chain.
        limit_bytes: u64,
        /// Usage at `path`, diagnostic only; not a budget input.
        current_bytes: u64,
        /// Path of the smallest saturating `memory.max - memory.current`;
        /// binds the available figure. Equals `path` when one level binds both.
        avail_path: String,
        /// The `memory.max` at `avail_path`.
        avail_limit_bytes: u64,
        /// The `memory.current` at `avail_path`.
        avail_current_bytes: u64,
    },
}

/// The host's memory capacity as the kernel reported it at one instant.
///
/// A reading, on the same terms as [`MeasuredDevice`]: another process can take
/// memory a moment later and this will not know. Document 03's answer to that is
/// a pressure-driven replan, which is M2's.
///
/// Only `total_bytes` and `available_bytes` are budget. Everything else is a
/// diagnostic, and two of them are deliberately excluded: **swap is never
/// budget** (document 03 forbids relying on it as an invisible fourth execution
/// tier), and the machine-view pair is kept only so a report can show what a
/// cgroup limit cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredHost {
    /// Effective total, after any cgroup limit.
    pub total_bytes: u64,
    /// Effective available, after any cgroup limit. This is `MemAvailable`, not
    /// `MemFree`: the kernel's own estimate of what a new allocation can obtain
    /// without swapping, which already accounts for reclaimable page cache.
    pub available_bytes: u64,
    /// What the machine reports before any cgroup limit is applied.
    pub machine_total_bytes: u64,
    pub machine_available_bytes: u64,
    /// Completely unused memory. Small on a healthy system, and **not** the
    /// budget input -- most of a busy machine's spendable memory is reclaimable
    /// cache rather than free.
    pub free_bytes: u64,
    pub buffers_bytes: u64,
    /// Page cache. Reported so page-cache pressure is visible; managing it is
    /// M2's (document 03).
    pub cached_bytes: u64,
    /// Reported, and never in any budget.
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    pub limit: HostLimit,
}

/// A kernel's declared applicability. The planner matches against this rather
/// than against a model name (document 02).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCapability {
    pub operation: &'static str,
    /// Compute capabilities this kernel has been *qualified* on. Document 03:
    /// SM100/Hopper recipes are not automatically compatible with SM120, and
    /// SM120 must be qualified separately from SM86.
    pub qualified_sm: Vec<String>,
    pub workspace_upper_bound_bytes: u64,
}

impl KernelCapability {
    pub fn supports(&self, device: &DeviceCapability) -> bool {
        self.qualified_sm.contains(&device.sm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(major: u32, minor: u32) -> DeviceCapability {
        DeviceCapability {
            ordinal: 0,
            uuid: DeviceUuid::from_bytes([0; 16]),
            name: "test".into(),
            compute_major: major,
            compute_minor: minor,
            total_memory_bytes: 0,
            multiprocessor_count: 0,
            pci_bus_id: "0000:00:00.0".into(),
            peer_access: vec![(1, true), (2, false)],
            // The real limits on every NVIDIA architecture to date.
            max_grid: (2_147_483_647, 65_535, 65_535),
        }
    }

    #[test]
    fn required_errors_where_auto_falls_back() {
        assert!(StrategyControl::Required.must_error_when_unavailable());
        assert!(!StrategyControl::Auto.must_error_when_unavailable());
        assert!(!StrategyControl::Off.must_error_when_unavailable());

        assert!(StrategyControl::Auto.may_select());
        assert!(StrategyControl::Required.may_select());
        assert!(!StrategyControl::Off.may_select());
    }

    #[test]
    fn default_control_is_auto() {
        assert_eq!(StrategyControl::default(), StrategyControl::Auto);
    }

    #[test]
    fn sm120_is_not_covered_by_an_sm86_qualification() {
        // R17 / document 03: qualification is per compute capability. A kernel
        // proven on the 3090s says nothing about the 5060 Ti.
        let k = KernelCapability {
            operation: "linear_nvfp4",
            qualified_sm: vec!["sm_86".to_string()],
            workspace_upper_bound_bytes: 0,
        };
        assert!(k.supports(&dev(8, 6)));
        assert!(!k.supports(&dev(12, 0)));
    }

    #[test]
    fn peer_access_defaults_to_false_for_unknown_peers() {
        // An unprobed pair is not a usable pair.
        let d = dev(8, 6);
        assert!(d.can_access_peer(1));
        assert!(!d.can_access_peer(2));
        assert!(!d.can_access_peer(99));
    }
}
