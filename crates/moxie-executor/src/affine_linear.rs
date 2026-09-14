//! Binding and launch for the shared W4A16 / W8A16 dense linear.
//!
//! Task 0028. The kernel and its catalogue identities belong to
//! `moxie-kernels`; what is here is the three things only this crate may do —
//! resolve a selected descriptor's symbol, turn admitted ranges into addresses,
//! and wait on the event that says the answer exists.
//!
//! **Why this is not in `chain.rs`.** That file binds task 0012's selected
//! three-node BF16 chain: a fixed graph, a fixed edge shape, and a launch order
//! written out node by node. A quantized dense linear is a different operation
//! with a different operand set — three device buffers for one logical weight
//! rather than one — and folding it in would make `chain.rs` two things whose
//! only shared property is the word "linear". Task 0028's contract listed
//! `chain.rs` among its allowed files before either shape was known; this is
//! the same crate and the same ownership, in a module that says what it is.
//!
//! **The weight is not copied here and no cache lives in this file.** Its three
//! components are resident in the one production weight-residency owner
//! (task 0020) and reach the launch as `device_address` of a lease's offset.
//! `moxie-memory` decided what is resident; this file only reads addresses.

use moxie_format::affine::{AffineDescriptor, AffineTensor, Grouping, IntWidth};
use moxie_format::payload::ZeroPointSection;
use moxie_format::scale::ScaleDtype;
use moxie_types::{
    ActivationPrecision, DeviceCapability, Error, KernelCatalogue, KernelOperand, Precision,
    Result, SemanticKernelDescriptor, SemanticKernelOp, WeightPrecision,
};

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

fn unsupported(capability: &'static str, reason: impl Into<String>) -> Error {
    Error::Unsupported {
        capability,
        reason: reason.into(),
    }
}

/// Everything the kernel needs that is not an address, derived once from the
/// canonical descriptor and checked before any device work starts.
///
/// This type is the group map. The kernel reads a group index per `(row, k
/// tile)` and never per element, which is only correct while a group boundary
/// cannot fall inside a `k` tile — so that is checked here, against
/// [`moxie_kernels::AFFINE_LINEAR_TILE`], rather than assumed in CUDA where it
/// could only fail as a wrong number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineLaunch {
    pub rows: u64,
    pub in_features: u64,
    pub out_features: u64,
    /// Bytes one packed code row occupies. INT4 rows are byte-aligned, so an
    /// odd row carries one nibble that is never a logical value.
    pub row_stride: u64,
    pub groups_per_row: u64,
    /// Logical input columns per group. Equal to `in_features` for a
    /// per-output-channel tensor, which has exactly one group.
    pub group_size: u64,
    pub width: IntWidth,
    pub scale_dtype: ScaleDtype,
    pub symmetric: bool,
}

impl AffineLaunch {
    /// Derive the launch from a canonical descriptor, or refuse.
    ///
    /// Refusals here are the kernel's declared domain, not a convenience:
    /// an activation-order map and a group size the tile cannot honour are both
    /// things this kernel does not implement, and ADR 0027 names `actorder:
    /// static` a continuation rather than a prerequisite. Silently treating a
    /// permuted tensor as contiguous would produce a plausible wrong answer.
    pub fn derive(
        descriptor: &AffineDescriptor,
        zero_points: ZeroPointSection,
        rows: u64,
    ) -> Result<Self> {
        descriptor.validate()?;
        if rows == 0 {
            return Err(invalid("rows", "a launch needs at least one row"));
        }
        if descriptor.group_index.is_some() {
            return Err(unsupported(
                "w4a16_group_index",
                "this tensor carries an activation-order group map; the shared linear \
                 implements the contiguous and per-channel rules only, and a permuted \
                 tensor is a named continuation rather than a case to approximate",
            ));
        }
        let groups_per_row = descriptor.groups_per_row()? as u64;
        let in_features = descriptor.in_features as u64;
        let out_features = descriptor.out_features as u64;
        let group_size = match descriptor.grouping {
            Grouping::PerOutputChannel => in_features,
            Grouping::Contiguous { size } => u64::from(size),
        };
        // Every size in the closed set is a multiple of the tile today, and
        // `every_allowed_group_size_fits_the_tile` holds the host lane to that.
        // This refusal is what fires the day the set widens: without it, a
        // group boundary inside a tile would silently read one group's scale
        // for another group's codes, which is a plausible wrong answer rather
        // than a failure.
        let tile = moxie_kernels::AFFINE_LINEAR_TILE;
        if groups_per_row > 1 && !group_size.is_multiple_of(tile) {
            return Err(unsupported(
                "w4a16_group_size",
                format!(
                    "group size {group_size} is not a multiple of the {tile}-wide tensor-core \
                     tile, so a group boundary would fall inside one tile and the kernel's \
                     one-scale-per-tile conversion would read the wrong group"
                ),
            ));
        }
        out_features
            .checked_mul(groups_per_row)
            .ok_or_else(|| invalid("groups", "the group table size overflows"))?;
        let symmetric = matches!(zero_points, ZeroPointSection::Absent);
        let launch = Self {
            rows,
            in_features,
            out_features,
            row_stride: descriptor.width.row_stride(descriptor.in_features) as u64,
            groups_per_row,
            group_size,
            width: descriptor.width,
            scale_dtype: descriptor.scale_dtype,
            symmetric,
        };
        // Every extent the kernel indexes, checked for overflow before it is a
        // pointer. A `u64` product that wraps here is a read past the end of a
        // real allocation there.
        launch.code_bytes()?;
        launch.scale_bytes()?;
        launch.activation_bytes()?;
        launch.output_bytes()?;
        Ok(launch)
    }

    /// Derive the launch from a tensor already decoded on the host.
    ///
    /// The section comes from the tensor rather than from a caller's claim
    /// about it, so an asymmetric tensor cannot be launched as symmetric.
    pub fn for_tensor(tensor: &AffineTensor, rows: u64) -> Result<Self> {
        let section = if tensor.zero_points().is_symmetric() {
            ZeroPointSection::Absent
        } else {
            ZeroPointSection::PerGroup
        };
        Self::derive(tensor.descriptor(), section, rows)
    }

    /// The group a logical input column belongs to.
    ///
    /// The same arithmetic the kernel performs, expressed once on the host so a
    /// test can check it against [`AffineDescriptor::group_of`] — which is the
    /// decoder's own map — on shapes whose last group is short.
    pub fn group_of(&self, k: u64) -> Result<u64> {
        if k >= self.in_features {
            return Err(invalid(
                "column",
                format!("column {k} is outside {} input features", self.in_features),
            ));
        }
        Ok(if self.groups_per_row == 1 {
            0
        } else {
            k / self.group_size
        })
    }

    pub fn code_bytes(&self) -> Result<u64> {
        self.row_stride
            .checked_mul(self.out_features)
            .ok_or_else(|| invalid("codes", "the packed code extent overflows"))
    }

    pub fn scale_bytes(&self) -> Result<u64> {
        self.out_features
            .checked_mul(self.groups_per_row)
            .and_then(|entries| entries.checked_mul(self.scale_dtype.bytes() as u64))
            .ok_or_else(|| invalid("scales", "the scale extent overflows"))
    }

    /// `None` for a symmetric tensor: the section is **absent**, not zero-filled.
    pub fn zero_point_bytes(&self) -> Result<Option<u64>> {
        if self.symmetric {
            return Ok(None);
        }
        self.out_features
            .checked_mul(self.groups_per_row)
            .and_then(|entries| entries.checked_mul(2))
            .map(Some)
            .ok_or_else(|| invalid("zero_points", "the zero-point extent overflows"))
    }

    pub fn activation_bytes(&self) -> Result<u64> {
        self.rows
            .checked_mul(self.in_features)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("activations", "the activation extent overflows"))
    }

    pub fn output_bytes(&self) -> Result<u64> {
        self.rows
            .checked_mul(self.out_features)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("output", "the output extent overflows"))
    }

    /// What a BF16 copy of this weight would occupy.
    ///
    /// Not a size this path ever allocates — it is the number the memory bound
    /// is measured against. ADR 0003: the codes are unpacked into 16-bit
    /// tensor-core tiles, and a whole-tensor copy is the shortcut M3 item 3
    /// forbids by name.
    pub fn dequantized_weight_bytes(&self) -> Result<u64> {
        self.out_features
            .checked_mul(self.in_features)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("weight", "the dequantized extent overflows"))
    }

    /// The kernel's `scale_kind` tag: the source's own scalar encoding.
    pub const fn scale_kind(&self) -> u32 {
        match self.scale_dtype {
            ScaleDtype::F16 => 0,
            ScaleDtype::Bf16 => 1,
            ScaleDtype::F32 => 2,
        }
    }

    pub const fn code_bits(&self) -> u32 {
        self.width.bits()
    }

    /// The execution profile this weight width names.
    pub const fn profile(&self) -> &'static str {
        moxie_kernels::profile_name(self.width.precision())
    }
}

/// Select the one catalogue descriptor that serves this weight width on this
/// device, or fail.
///
/// Exactly one, or none. The match is on semantic operation, operand roles and
/// precisions, accumulation, rounding, layout, SM and shape bounds — the same
/// keys task 0012 and task 0021 match on, and never a model name or a
/// quantization method. An INT4 weight selects the `w4a16` entry; a BF16 weight
/// finds nothing here, because the BF16 linear lives in a different package.
pub fn select_affine_linear_kernel(
    catalogue: &KernelCatalogue,
    capability: &DeviceCapability,
    weight: WeightPrecision,
    launch: &AffineLaunch,
) -> Result<SemanticKernelDescriptor> {
    let inputs = vec![
        KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
        KernelOperand::Weight(weight),
    ];
    let matches: Vec<_> = catalogue
        .descriptors()
        .iter()
        .filter(|d| {
            d.operation == SemanticKernelOp::Linear
                && d.abi_version == moxie_kernels::AFFINE_LINEAR_ABI
                && d.inputs == inputs
                && d.output == ActivationPrecision::expect(Precision::Bf16)
                && d.accumulation == moxie_types::AccumulationPolicy::Bf16InF32Acc
                && d.rounding == moxie_types::RoundingProfile::FinalBf16Rne
                && d.layout == moxie_types::TensorLayout::ContiguousRowMajorV1
                && d.workspace == moxie_types::WorkspaceExpression::Zero
                && d.sm.major == capability.compute_major
                && d.sm.minor == capability.compute_minor
                && launch.rows <= d.shape.max_rows
                && launch.in_features <= d.shape.max_input
                && launch.out_features <= d.shape.max_output
        })
        .collect();
    if matches.len() != 1 {
        return Err(Error::UnsupportedKernel {
            operation: "linear",
            detail: format!(
                "expected exactly one {} descriptor for a {} weight at {}x{}x{} on sm_{}{}; \
                 found {}",
                moxie_kernels::profile_name(weight.get()),
                weight,
                launch.rows,
                launch.in_features,
                launch.out_features,
                capability.compute_major,
                capability.compute_minor,
                matches.len()
            ),
        });
    }
    let descriptor = matches[0];
    if descriptor.symbols.len() != 1 || descriptor.symbols[0].0 != moxie_kernels::AFFINE_LINEAR {
        return Err(Error::UnsupportedKernel {
            operation: "linear",
            detail: format!(
                "descriptor {} names {:?} rather than the shared quantized linear symbol",
                descriptor.id.0, descriptor.symbols
            ),
        });
    }
    Ok(descriptor.clone())
}

#[cfg(feature = "driver")]
mod device {
    use core::ffi::c_void;

    use moxie_cuda::{
        Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage,
    };
    use moxie_memory::{
        BufferRequest, Ledger, LedgerId, PlanRequest, Rejection, Reservation, ResidencyAuthority,
        ResidencyLease, StageSpan,
    };
    use moxie_types::{DeviceTier, Error, HostTier, Result, Scope, SemanticKernelDescriptor, Tier};

    use super::{AffineLaunch, invalid};
    use crate::arena::{DeviceArena, DeviceRange};
    use crate::residency::DeviceResidency;

    /// 256-byte alignment, as every other device range in this crate uses.
    const ALIGNMENT: u64 = 256;

    /// One logical weight, resident as three components.
    ///
    /// Three leases rather than one because they are three separately sized
    /// sections whose bytes the source stores apart, and because a symmetric
    /// tensor genuinely has no third one. `zero_points: None` is the absent
    /// section; the kernel receives a null pointer and adds nothing.
    #[derive(Debug)]
    pub struct ResidentAffineWeight<'a> {
        pub codes: &'a ResidencyLease,
        pub scales: &'a ResidencyLease,
        pub zero_points: Option<&'a ResidencyLease>,
    }

    /// An admitted quantized linear: one arena for the per-step tensors, one
    /// resolved symbol, and the launch geometry both were checked against.
    ///
    /// The weight is **not** in this arena. Its components live in the
    /// residency authority's one allocation, and this type never holds their
    /// bytes — only, briefly, their addresses.
    #[derive(Debug)]
    #[must_use = "an unclosed run keeps its arena and its reservation"]
    pub struct AffineLinearRun<'ctx> {
        module: ResolvedModule<'ctx>,
        descriptor: SemanticKernelDescriptor,
        launch: AffineLaunch,
        arena: Option<DeviceArena<'ctx>>,
        activations: Option<DeviceRange<'ctx>>,
        output: Option<DeviceRange<'ctx>>,
        arena_bytes: u64,
        ledger: LedgerId,
        ctx: &'ctx RankContext,
        /// Set when work was enqueued and its completion could not be
        /// established. The ranges it may still be reading are then never
        /// released: `close` refuses and dropping keeps the charge.
        quarantined: bool,
    }

    /// A refusal that happened before anything was allocated, with the
    /// reservation handed back so no charge is stranded.
    #[derive(Debug)]
    pub struct AffineAdmitRefused {
        pub error: Error,
        /// Returned held only when giving it back *also* failed. `None` is the
        /// ordinary refusal: nothing is charged and nothing is stranded.
        pub reservation: Option<Reservation>,
        /// The ledger's own breakdown, when the refusal was a capacity
        /// rejection. Summarising it into a byte count would throw away the
        /// explanation a caller needs to act.
        pub rejection: Option<Box<Rejection>>,
    }

    #[derive(Debug)]
    pub struct AffineCloseRefused<'ctx> {
        pub run: AffineLinearRun<'ctx>,
        pub error: Error,
    }

    impl<'ctx> AffineLinearRun<'ctx> {
        /// Charge the per-step tensors, materialize their arena, and resolve
        /// the selected symbol.
        ///
        /// The admission covers activations and output only. Whether the
        /// weight fits is the residency authority's question and it answered it
        /// before this is called; charging the same bytes twice would make the
        /// ledger's total a number about nothing.
        pub fn admit(
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            descriptor: SemanticKernelDescriptor,
            launch: AffineLaunch,
        ) -> std::result::Result<Self, AffineAdmitRefused> {
            let fail = |error| AffineAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            };
            if descriptor.sm.major != ctx.capability().compute_major
                || descriptor.sm.minor != ctx.capability().compute_minor
            {
                return Err(fail(Error::UnsupportedKernel {
                    operation: "linear",
                    detail: format!(
                        "descriptor {} is qualified for {} and this device is sm_{}{}",
                        descriptor.id.0,
                        descriptor.sm.name(),
                        ctx.capability().compute_major,
                        ctx.capability().compute_minor
                    ),
                }));
            }
            let request = match resource_request(&launch, ctx) {
                Ok(request) => request,
                Err(error) => return Err(fail(error)),
            };
            let reservation = match ledger.admit(&request) {
                Ok(reservation) => reservation,
                Err(moxie_memory::AdmitError::Invalid(error)) => return Err(fail(error)),
                Err(moxie_memory::AdmitError::Rejected(rejection)) => {
                    return Err(AffineAdmitRefused {
                        error: Error::CapacityExceeded {
                            tier: None,
                            requested_bytes: rejection.shortfall_bytes,
                            available_bytes: 0,
                        },
                        reservation: None,
                        rejection: Some(rejection),
                    });
                }
            };
            let activation_bytes = match launch.activation_bytes().and_then(align_up) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(ledger, reservation, error)),
            };
            let output_bytes = match launch.output_bytes().and_then(align_up) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(ledger, reservation, error)),
            };
            let total = match activation_bytes
                .checked_add(output_bytes)
                .ok_or_else(|| invalid("arena", "the per-step arena extent overflows"))
            {
                Ok(total) => total,
                Err(error) => return Err(give_back(ledger, reservation, error)),
            };
            let mut arena = match DeviceArena::create_partitioned(
                ledger,
                reservation,
                ctx,
                &[(DeviceTier::Activations, total)],
                total,
                format!("affine-linear-{}", descriptor.id.0),
            ) {
                Ok(arena) => arena,
                Err(refused) => {
                    return Err(give_back(ledger, refused.reservation, refused.error));
                }
            };
            let mut hold = Vec::new();
            let allocate = |arena: &mut DeviceArena<'ctx>, bytes, label: &str| {
                arena
                    .allocate(bytes, ALIGNMENT, label.to_string())
                    .map_err(|refused| refused.error)
            };
            let activations = match allocate(&mut arena, activation_bytes, "affine-activations") {
                Ok(range) => range,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            hold.push(activations);
            let output = match allocate(&mut arena, output_bytes, "affine-output") {
                Ok(range) => range,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            hold.push(output);
            // SAFETY: the bytes are this build's own nvcc output, embedded by
            // `include_bytes!`, and the descriptor's image digest is the one
            // the built-in catalogue published for them.
            let image = match unsafe {
                TrustedImage::from_build_output(moxie_kernels::AFFINE_LINEAR_FATBIN)
            } {
                Ok(image) => image,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            let symbols: Vec<String> = descriptor.symbols.iter().map(|s| s.0.clone()).collect();
            let module = match Module::load(ctx, ModuleImage::Binary(image))
                .and_then(|module| module.resolve_all(&symbols))
            {
                Ok(module) => module,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            let output = hold.pop().expect("output range");
            let activations = hold.pop().expect("activation range");
            Ok(Self {
                module,
                descriptor,
                launch,
                arena: Some(arena),
                activations: Some(activations),
                output: Some(output),
                arena_bytes: total,
                ledger: ledger.id(),
                ctx,
                quarantined: false,
            })
        }

        /// The real device bytes this run holds for its per-step tensors.
        ///
        /// Acceptance 4 compares this with
        /// [`AffineLaunch::dequantized_weight_bytes`]. It is a memory bound and
        /// **not** a speed measurement: O6 and O7 are open and nothing here is
        /// timed.
        pub const fn arena_bytes(&self) -> u64 {
            self.arena_bytes
        }

        pub const fn launch(&self) -> &AffineLaunch {
            &self.launch
        }

        pub fn descriptor(&self) -> &SemanticKernelDescriptor {
            &self.descriptor
        }

        /// Upload one step's activations, launch against the resident weight,
        /// and return the output once the completion event says it exists.
        ///
        /// `x` stays owned by the caller for the whole call and the call does
        /// not return until the event is observed, so no source is reusable
        /// before its copy completed.
        pub fn run(
            &mut self,
            stream: &Stream<'ctx>,
            authority: &ResidencyAuthority,
            residency: &DeviceResidency<'ctx>,
            weight: ResidentAffineWeight<'_>,
            x: &[u8],
        ) -> Result<Vec<u8>> {
            if self.quarantined {
                return Err(invalid(
                    "run",
                    "this run is quarantined: work was enqueued whose completion is unknown",
                ));
            }
            if Some(stream.device_uuid()) != residency.scope().device() {
                return Err(invalid(
                    "stream",
                    "the stream and the residency backing name different devices",
                ));
            }
            if x.len() as u64 != self.launch.activation_bytes()? {
                return Err(invalid(
                    "activations",
                    format!(
                        "{} activation byte(s) for {} row(s) of {}",
                        x.len(),
                        self.launch.rows,
                        self.launch.in_features
                    ),
                ));
            }
            let codes = self.component(
                authority,
                residency,
                weight.codes,
                "codes",
                self.launch.code_bytes()?,
            )?;
            let scales = self.component(
                authority,
                residency,
                weight.scales,
                "scales",
                self.launch.scale_bytes()?,
            )?;
            let zero_points = match (weight.zero_points, self.launch.zero_point_bytes()?) {
                (Some(lease), Some(bytes)) => {
                    Some(self.component(authority, residency, lease, "zero_points", bytes)?)
                }
                (None, None) => None,
                // A symmetric launch handed a zero-point section, or an
                // asymmetric one handed none, is a wiring mistake that would
                // otherwise produce a confident wrong answer.
                (Some(_), None) => {
                    return Err(invalid(
                        "zero_points",
                        "a symmetric tensor was given a zero-point component",
                    ));
                }
                (None, Some(_)) => {
                    return Err(invalid(
                        "zero_points",
                        "an asymmetric tensor was given no zero-point component",
                    ));
                }
            };

            let activations = self.activations.as_ref().expect("live activation range");
            let output = self.output.as_ref().expect("live output range");
            // SAFETY: `x` outlives the synchronize below, and the destination
            // is an admitted range of this run's own arena.
            unsafe {
                activations.copy_from_host_async(x, stream)?;
            }
            let mut x_address = activations.device_address()?;
            let mut codes_address = codes;
            let mut scales_address = scales;
            let mut zero_address = zero_points.unwrap_or(0);
            let mut output_address = output.device_address()?;
            let mut rows = self.launch.rows;
            let mut in_features = self.launch.in_features;
            let mut out_features = self.launch.out_features;
            let mut row_stride = self.launch.row_stride;
            let mut groups_per_row = self.launch.groups_per_row;
            let mut code_bits = self.launch.code_bits();
            let mut group_size = u32::try_from(self.launch.group_size)
                .map_err(|_| invalid("group_size", "the group size exceeds a u32"))?;
            let mut scale_kind = self.launch.scale_kind();
            let mut params: [*mut c_void; 13] = [
                (&raw mut x_address).cast(),
                (&raw mut codes_address).cast(),
                (&raw mut scales_address).cast(),
                (&raw mut zero_address).cast(),
                (&raw mut output_address).cast(),
                (&raw mut rows).cast(),
                (&raw mut in_features).cast(),
                (&raw mut out_features).cast(),
                (&raw mut row_stride).cast(),
                (&raw mut groups_per_row).cast(),
                (&raw mut code_bits).cast(),
                (&raw mut group_size).cast(),
                (&raw mut scale_kind).cast(),
            ];
            let tile = moxie_kernels::AFFINE_LINEAR_TILE;
            let grid_x = u32::try_from(out_features.div_ceil(tile))
                .map_err(|_| invalid("grid", "the output grid exceeds a u32"))?;
            let grid_y = u32::try_from(rows.div_ceil(tile))
                .map_err(|_| invalid("grid", "the row grid exceeds a u32"))?;
            // SAFETY: the symbol's ABI is the one declared in
            // `affine_linear.cu`; every pointer names a live admitted or
            // resident range whose length was checked above, and the grid
            // covers exactly the output tiles.
            let launched = unsafe {
                self.module
                    .launch_async(0, stream, (grid_x, grid_y, 1), (32, 1, 1), 0, &mut params)
            };
            self.settle(launched, stream)?;
            let mut host = vec![0u8; self.launch.output_bytes()? as usize];
            self.output
                .as_ref()
                .expect("live output range")
                .copy_to_host(&mut host)?;
            if host
                .chunks_exact(2)
                .any(|word| !bf16_is_finite(u16::from_le_bytes([word[0], word[1]])))
            {
                return Err(Error::Numerical {
                    detail: format!("{} produced a nonfinite BF16 output", self.descriptor.id.0),
                });
            }
            Ok(host)
        }

        /// Resolve one resident component to an address, after checking that
        /// the lease really covers the bytes the launch will read.
        ///
        /// The check is the point. A component whose residency range is shorter
        /// than its descriptor implies is the difference between a refusal and
        /// a read past the end of the cache's allocation into another tensor.
        fn component(
            &self,
            authority: &ResidencyAuthority,
            residency: &DeviceResidency<'ctx>,
            lease: &ResidencyLease,
            what: &'static str,
            need: u64,
        ) -> Result<u64> {
            if authority.id()
                != residency.authority_id().ok_or_else(|| {
                    invalid(
                        "residency",
                        "this residency backing has already been closed",
                    )
                })?
            {
                return Err(invalid(
                    "residency",
                    "the lease's authority does not own this backing, so its offset \
                     would name another allocation's bytes",
                ));
            }
            let (offset, len) = authority.device_range(lease)?;
            if len < need {
                return Err(invalid(
                    what,
                    format!(
                        "the resident {what} component is {len} byte(s); the launch reads {need}"
                    ),
                ));
            }
            residency.device_address(offset, need)
        }

        /// Record and wait on the completion event, quarantining the run if the
        /// answer to "did it finish" is unknown.
        fn settle(&mut self, launched: Result<()>, stream: &Stream<'ctx>) -> Result<()> {
            if let Err(error) = launched {
                // Nothing was submitted for this launch, but the activation
                // copy above was. Its completion is still unknown.
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            let event = match Event::new(self.ctx) {
                Ok(event) => event,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            if let Err(error) = event.record(stream) {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            if let Err(error) = event.synchronize() {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            Ok(())
        }

        fn attribute(&self, error: Error) -> Error {
            match error {
                Error::DeviceLost { detail, .. } => Error::DeviceLost {
                    device: self.ctx.ordinal(),
                    detail: format!("kernel {}: {detail}", self.descriptor.id.0),
                },
                other => other,
            }
        }

        /// Release the arena and its charge.
        ///
        /// Refuses while quarantined: work whose completion is unknown may
        /// still be reading these ranges, and freeing them would be the one
        /// failure the whole lease discipline exists to prevent.
        #[allow(clippy::result_large_err)]
        pub fn close(
            mut self,
            ledger: &mut Ledger,
        ) -> std::result::Result<(), AffineCloseRefused<'ctx>> {
            if self.quarantined {
                let error = invalid(
                    "close",
                    "this run is quarantined; its ranges may still be in flight",
                );
                return Err(AffineCloseRefused { run: self, error });
            }
            if ledger.id() != self.ledger {
                let error = invalid("ledger", "this run belongs to another ledger");
                return Err(AffineCloseRefused { run: self, error });
            }
            for output_first in [true, false] {
                let taken = if output_first {
                    self.output.take()
                } else {
                    self.activations.take()
                };
                let Some(range) = taken else { continue };
                let arena = self.arena.as_mut().expect("an open run has its arena");
                if let Err(refused) = arena.release(range) {
                    if output_first {
                        self.output = Some(refused.range);
                    } else {
                        self.activations = Some(refused.range);
                    }
                    return Err(AffineCloseRefused {
                        run: self,
                        error: refused.error,
                    });
                }
            }
            let arena = self.arena.take().expect("an open run has its arena");
            match arena.close(ledger) {
                Ok(()) => Ok(()),
                Err(refused) => {
                    self.arena = Some(refused.arena);
                    Err(AffineCloseRefused {
                        run: self,
                        error: refused.error,
                    })
                }
            }
        }
    }

    /// The exact admission envelope for one quantized linear step.
    ///
    /// Public so a caller can see the charge before allocating a device, the
    /// same way `selected_resource_request` does for the BF16 chain.
    pub fn resource_request(launch: &AffineLaunch, ctx: &RankContext) -> Result<PlanRequest> {
        let mut request = PlanRequest::new(
            format!("affine-linear-{}", launch.profile()),
            ["bind", "launch", "read"],
        )?;
        let scope = Scope::Device(ctx.uuid());
        request.buffer(BufferRequest::new(
            "activations",
            scope,
            Tier::Device(DeviceTier::Activations),
            align_up(launch.activation_bytes()?)?,
            StageSpan::inclusive(0, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "output",
            scope,
            Tier::Device(DeviceTier::Activations),
            align_up(launch.output_bytes()?)?,
            StageSpan::inclusive(1, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "retained-activation-source",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            launch.activation_bytes()?,
            StageSpan::inclusive(0, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "output-readback",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            launch.output_bytes()?,
            StageSpan::at(2),
        ))?;
        Ok(request)
    }

    fn align_up(bytes: u64) -> Result<u64> {
        bytes
            .checked_add(ALIGNMENT - 1)
            .map(|v| v / ALIGNMENT * ALIGNMENT)
            .ok_or_else(|| invalid("align", "the aligned extent overflows"))
    }

    fn give_back(
        ledger: &mut Ledger,
        reservation: Reservation,
        error: Error,
    ) -> AffineAdmitRefused {
        match ledger.release(reservation) {
            Ok(()) => AffineAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            },
            Err(refused) => AffineAdmitRefused {
                error,
                reservation: Some(refused.reservation),
                rejection: None,
            },
        }
    }

    fn unwind<'ctx>(
        mut arena: DeviceArena<'ctx>,
        mut ranges: Vec<DeviceRange<'ctx>>,
        ledger: &mut Ledger,
        error: Error,
    ) -> AffineAdmitRefused {
        while let Some(range) = ranges.pop() {
            if let Err(refused) = arena.release(range) {
                // The arena keeps its allocation and its charge, visibly, which
                // is what `DeviceArena` already does with a failed release.
                return AffineAdmitRefused {
                    error: refused.error,
                    reservation: None,
                    rejection: None,
                };
            }
        }
        match arena.close(ledger) {
            Ok(()) => AffineAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            },
            Err(refused) => AffineAdmitRefused {
                error: refused.error,
                reservation: None,
                rejection: None,
            },
        }
    }

    fn bf16_is_finite(bits: u16) -> bool {
        bits & 0x7f80 != 0x7f80
    }
}

#[cfg(feature = "driver")]
pub use device::{
    AffineAdmitRefused, AffineCloseRefused, AffineLinearRun, ResidentAffineWeight, resource_request,
};

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_format::affine::{ALLOWED_GROUP_SIZES, ZeroPoints};
    use moxie_format::scale::ScaleValues;
    use moxie_types::{
        AccumulationPolicy, DeviceUuid, KernelId, KernelShapeBounds, KernelSymbol, RoundingProfile,
        SmVersion, TensorLayout, WorkspaceExpression,
    };

    fn descriptor(
        width: IntWidth,
        in_features: usize,
        out_features: usize,
        grouping: Grouping,
    ) -> AffineDescriptor {
        AffineDescriptor {
            width,
            out_features,
            in_features,
            grouping,
            group_index: None,
            scale_dtype: ScaleDtype::Bf16,
        }
    }

    fn zeros(_descriptor: &AffineDescriptor) -> ZeroPointSection {
        ZeroPointSection::PerGroup
    }

    #[test]
    fn every_allowed_group_size_fits_the_tile() {
        // The kernel converts one scale per (row, k tile). That is only correct
        // while no group boundary can fall inside a tile, which is a statement
        // about the closed set in `moxie-format` and the tile width in
        // `moxie-kernels` -- two crates that otherwise never meet. Widening the
        // set without revisiting the kernel has to break here, in the host
        // lane, rather than as a wrong number on a GPU.
        for size in ALLOWED_GROUP_SIZES {
            assert!(
                u64::from(*size).is_multiple_of(moxie_kernels::AFFINE_LINEAR_TILE),
                "group size {size} is not a multiple of the {}-wide tile",
                moxie_kernels::AFFINE_LINEAR_TILE
            );
        }
    }

    #[test]
    fn the_group_map_matches_the_decoders_own_on_short_final_groups() {
        // Checked against `AffineDescriptor::group_of`, which is task 0024's
        // exhaustively tested decoder -- not against a second copy of this
        // function. The shapes have a tail on purpose: 100 columns is three
        // full groups of 32 and a final group of four, and an off-by-one in
        // either direction lands inside the tail.
        for (width, in_features, grouping) in [
            (IntWidth::Int4, 100usize, Grouping::Contiguous { size: 32 }),
            (IntWidth::Int8, 100, Grouping::Contiguous { size: 32 }),
            (IntWidth::Int4, 300, Grouping::Contiguous { size: 128 }),
            (IntWidth::Int8, 129, Grouping::Contiguous { size: 128 }),
            (IntWidth::Int4, 100, Grouping::PerOutputChannel),
            (IntWidth::Int8, 1, Grouping::PerOutputChannel),
        ] {
            let d = descriptor(width, in_features, 7, grouping);
            let launch = AffineLaunch::derive(&d, zeros(&d), 3).unwrap();
            assert_eq!(launch.groups_per_row, d.groups_per_row().unwrap() as u64);
            for k in 0..in_features {
                assert_eq!(
                    launch.group_of(k as u64).unwrap(),
                    d.group_of(k).unwrap() as u64,
                    "column {k} of {in_features} under {grouping:?}"
                );
            }
            assert!(launch.group_of(in_features as u64).is_err());
        }
    }

    #[test]
    fn the_component_extents_follow_the_descriptor_and_not_the_other_way_round() {
        // INT4 packs two codes per byte and its rows are byte-aligned, so an
        // odd row carries one nibble that is never a logical value. INT8 does
        // not pad at all. Sizing either from the other is a read past the end.
        let d = descriptor(IntWidth::Int4, 101, 5, Grouping::Contiguous { size: 32 });
        let launch = AffineLaunch::derive(&d, zeros(&d), 2).unwrap();
        assert_eq!(launch.row_stride, 51);
        assert_eq!(launch.code_bytes().unwrap(), 51 * 5);
        assert_eq!(launch.groups_per_row, 4);
        assert_eq!(launch.scale_bytes().unwrap(), 4 * 5 * 2);
        assert_eq!(launch.zero_point_bytes().unwrap(), Some(4 * 5 * 2));
        assert_eq!(launch.dequantized_weight_bytes().unwrap(), 101 * 5 * 2);

        let d = descriptor(IntWidth::Int8, 101, 5, Grouping::Contiguous { size: 32 });
        let launch = AffineLaunch::derive(&d, ZeroPointSection::Absent, 2).unwrap();
        assert_eq!(launch.row_stride, 101);
        assert_eq!(launch.code_bytes().unwrap(), 101 * 5);
        // Absent, not zero-filled: a symmetric tensor has no third component.
        assert_eq!(launch.zero_point_bytes().unwrap(), None);
        assert!(launch.symmetric);
    }

    #[test]
    fn a_permuted_tensor_is_refused_rather_than_read_as_contiguous() {
        // ADR 0027 names `actorder: static` a continuation. Treating a permuted
        // tensor as contiguous produces a plausible wrong answer no accuracy
        // threshold would catch, so the kernel's domain is stated instead.
        let mut d = descriptor(IntWidth::Int4, 64, 4, Grouping::Contiguous { size: 32 });
        d.group_index = Some((0..64).map(|k| (k as u32) % 2).collect());
        let error = AffineLaunch::derive(&d, zeros(&d), 1).unwrap_err();
        assert_eq!(error.kind(), "unsupported");
        assert!(
            error.to_string().contains("activation-order"),
            "{error} does not say what it refused"
        );
    }

    #[test]
    fn a_decoded_tensor_cannot_be_launched_as_the_other_zero_point_mode() {
        // The section comes from the tensor, so an asymmetric weight launched
        // through `for_tensor` can never be told the kernel to add nothing --
        // which would be a plausible wrong answer rather than a failure.
        let d = descriptor(IntWidth::Int4, 64, 4, Grouping::Contiguous { size: 32 });
        let entries = d.group_entries().unwrap();
        let codes = vec![0u8; d.code_bytes().unwrap()];
        let scales = ScaleValues::Bf16(vec![0x3F80u16; entries]);
        let asymmetric = AffineTensor::new(
            d.clone(),
            codes.clone(),
            scales.clone(),
            ZeroPoints::PerGroup(vec![1i16; entries]),
        )
        .unwrap();
        assert!(!AffineLaunch::for_tensor(&asymmetric, 1).unwrap().symmetric);
        let symmetric = AffineTensor::new(d, codes, scales, ZeroPoints::Symmetric).unwrap();
        assert!(AffineLaunch::for_tensor(&symmetric, 1).unwrap().symmetric);
        // Zero rows is not a launch.
        assert!(AffineLaunch::for_tensor(&symmetric, 0).is_err());
    }

    fn catalogue_entry(weight: Precision, sm: SmVersion) -> SemanticKernelDescriptor {
        SemanticKernelDescriptor {
            id: KernelId(format!(
                "{}-linear-v1-{}",
                moxie_kernels::profile_name(weight),
                sm.name()
            )),
            abi_version: moxie_kernels::AFFINE_LINEAR_ABI,
            operation: SemanticKernelOp::Linear,
            inputs: vec![
                KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                KernelOperand::Weight(WeightPrecision::expect(weight)),
            ],
            output: ActivationPrecision::expect(Precision::Bf16),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            rounding: RoundingProfile::FinalBf16Rne,
            layout: TensorLayout::ContiguousRowMajorV1,
            shape: KernelShapeBounds {
                max_rows: 64,
                max_input: 1024,
                max_output: 1024,
            },
            sm,
            workspace: WorkspaceExpression::Zero,
            image_sha256: [7u8; 32],
            symbols: vec![KernelSymbol(moxie_kernels::AFFINE_LINEAR.to_string())],
        }
    }

    fn capability(major: u32, minor: u32) -> DeviceCapability {
        DeviceCapability {
            ordinal: 0,
            uuid: DeviceUuid::from_bytes([0u8; 16]),
            name: "fixture".into(),
            compute_major: major,
            compute_minor: minor,
            total_memory_bytes: 1 << 30,
            multiprocessor_count: 1,
            pci_bus_id: "0000:00:00.0".into(),
            peer_access: Vec::new(),
        }
    }

    #[test]
    fn the_weight_width_picks_the_profile_and_a_float_weight_finds_nothing() {
        let catalogue = KernelCatalogue::new(vec![
            catalogue_entry(Precision::Int4, SmVersion::SM86),
            catalogue_entry(Precision::Int8, SmVersion::SM86),
        ])
        .unwrap();
        let cap = capability(8, 6);
        let d = descriptor(IntWidth::Int4, 64, 32, Grouping::Contiguous { size: 32 });
        let launch = AffineLaunch::derive(&d, zeros(&d), 4).unwrap();

        let chosen = select_affine_linear_kernel(
            &catalogue,
            &cap,
            WeightPrecision::expect(Precision::Int4),
            &launch,
        )
        .unwrap();
        assert_eq!(chosen.id.0, "w4a16-linear-v1-sm_86");
        let chosen = select_affine_linear_kernel(
            &catalogue,
            &cap,
            WeightPrecision::expect(Precision::Int8),
            &launch,
        )
        .unwrap();
        assert_eq!(chosen.id.0, "w8a16-linear-v1-sm_86");

        // A BF16 weight is not this kernel's operand. The refusal names the
        // profile it looked for, so the mismatch is readable rather than a bare
        // "unsupported".
        let error = select_affine_linear_kernel(
            &catalogue,
            &cap,
            WeightPrecision::expect(Precision::Bf16),
            &launch,
        )
        .unwrap_err();
        assert_eq!(error.kind(), "unsupported_kernel");
        let text = error.to_string();
        assert!(text.contains("bf16"), "{text}");
        assert!(text.contains("found 0"), "{text}");
    }

    #[test]
    fn qualification_does_not_leak_across_architectures_or_past_the_shape_bounds() {
        let catalogue =
            KernelCatalogue::new(vec![catalogue_entry(Precision::Int4, SmVersion::SM86)]).unwrap();
        let d = descriptor(IntWidth::Int4, 64, 32, Grouping::Contiguous { size: 32 });
        let launch = AffineLaunch::derive(&d, zeros(&d), 4).unwrap();
        let int4 = WeightPrecision::expect(Precision::Int4);
        // An SM86 entry says nothing about SM120, and the catalogue is the only
        // thing that may say so.
        assert!(
            select_affine_linear_kernel(&catalogue, &capability(12, 0), int4, &launch).is_err()
        );
        // Outside the descriptor's declared shape domain.
        let wide = descriptor(IntWidth::Int4, 2048, 32, Grouping::Contiguous { size: 32 });
        let wide = AffineLaunch::derive(&wide, zeros(&wide), 4).unwrap();
        assert!(select_affine_linear_kernel(&catalogue, &capability(8, 6), int4, &wide).is_err());
    }

    #[test]
    fn a_descriptor_naming_another_symbol_is_refused() {
        // The four catalogue identities exist so selection can be by capability.
        // They are one implementation, and an entry that named a second symbol
        // would be the moment "shared W4A16/W8A16 path" stopped being true.
        let mut entry = catalogue_entry(Precision::Int4, SmVersion::SM86);
        entry.symbols = vec![KernelSymbol("moxie_bf16_linear_v1".into())];
        let catalogue = KernelCatalogue::new(vec![entry]).unwrap();
        let d = descriptor(IntWidth::Int4, 64, 32, Grouping::Contiguous { size: 32 });
        let launch = AffineLaunch::derive(&d, zeros(&d), 4).unwrap();
        let error = select_affine_linear_kernel(
            &catalogue,
            &capability(8, 6),
            WeightPrecision::expect(Precision::Int4),
            &launch,
        )
        .unwrap_err();
        assert!(error.to_string().contains("shared quantized linear symbol"));
    }
}
