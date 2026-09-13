//! The device half of a grouped expert run.
//!
//! It owns no policy. The plan chose which experts run here, selected the
//! descriptor, and fixed the envelope; the residency authority decided which
//! bytes are resident and where. What is left is the three things only this
//! crate may do: allocate the admitted arena, enqueue the copies and launches,
//! and wait on their events.
//!
//! Two properties are worth finding quickly.
//!
//! **The weight operand is a lease's address and nothing else.** A group's
//! `gate_up` and `down` pointers are `device_range(&lease)` inside the
//! residency cache's one real allocation. Nothing here copies a weight, keeps
//! one, or knows where it came from -- a second cache in this file would make
//! M2's "exactly one production weight-residency owner" false, and an
//! `arch-check` rule says so.
//!
//! **Device slots are read back into the same slot-major buffer the host groups
//! write.** Slot `j` of row `r` is at index `r * top_k + j` on both sides, so a
//! mixed plan reduces one buffer in one declared order. The readback is
//! per-slot rather than whole-buffer because the host groups' slots must not be
//! overwritten by a device buffer that never held them.

#![cfg(feature = "driver")]

use core::ffi::c_void;

use moxie_cuda::{Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage};
use moxie_memory::{Ledger, Reservation, ResidencyAuthority, ResidencyLease};
use moxie_plan::expert::{ExpertGroup, ExpertPlan};
use moxie_types::{DeviceTier, Error, Result, Scope, SemanticKernelDescriptor};

use crate::arena::{DeviceArena, DeviceRange};
use crate::grouped::{ExpertDeviceLane, ExpertStaging, GroupedRun, LaunchRefused};
use crate::residency::DeviceResidency;

/// 256-byte alignment, as every other device range in this crate uses.
const ALIGNMENT: u64 = 256;
/// One launch's thread block. Correctness-first, like task 0012's: no occupancy
/// claim is made and none is needed.
const BLOCK: u32 = 128;

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

/// The device side of one run: one admitted arena, five ranges in it, and one
/// resolved module.
#[derive(Debug)]
#[must_use = "an unclosed device attachment keeps its arena and its reservation"]
pub struct DeviceExperts<'ctx> {
    ctx: &'ctx RankContext,
    stream: Stream<'ctx>,
    module: ResolvedModule<'ctx>,
    descriptor: SemanticKernelDescriptor,
    /// `None` only between taking it for `close` and putting it back on a
    /// refusal, so a failed cleanup never destroys the handle to a charged
    /// allocation.
    arena: Option<DeviceArena<'ctx>>,
    activations: Option<DeviceRange<'ctx>>,
    slots: Option<DeviceRange<'ctx>>,
    workspace: Option<DeviceRange<'ctx>>,
    row_index: Option<DeviceRange<'ctx>>,
    slot_index: Option<DeviceRange<'ctx>>,
    hidden: u64,
    intermediate: u64,
    /// The batch this attachment was built for. Launch indices are checked
    /// against these, not against whatever a caller's `ExpertGroup` claims.
    rows: u64,
    slot_count: u64,
    activations_loaded: bool,
    /// Set when a copy or launch was enqueued and its completion could not be
    /// established. The ranges it may still be reading are then never released:
    /// `close` refuses, and dropping keeps the arena and its charge, exactly as
    /// `DeviceArena` already does with its own allocation.
    quarantined: bool,
}

/// The byte extents this attachment needs, derived from the plan alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceExtents {
    pub activations: u64,
    pub slots: u64,
    pub workspace: u64,
    /// Both index arrays, sized for the largest group a plan can produce.
    pub indices: u64,
}

impl DeviceExtents {
    pub fn of(plan: &ExpertPlan) -> Result<Self> {
        let assignments = plan.slot_count();
        let workspace = u64::from(plan.queue_capacity())
            .checked_mul(plan.rows())
            .and_then(|v| v.checked_mul(plan.shape().intermediate))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| invalid("workspace", "the device workspace overflows".into()))?;
        // One index array's extent. Two are allocated, and the plan charges two.
        let indices = assignments
            .checked_mul(4)
            .ok_or_else(|| invalid("indices", "the index staging overflows".into()))?;
        Ok(DeviceExtents {
            activations: plan.activation_bytes(),
            slots: plan.slot_bytes(),
            workspace,
            indices,
        })
    }
}

fn align_up(bytes: u64) -> Result<u64> {
    bytes
        .checked_add(ALIGNMENT - 1)
        .map(|v| v / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| invalid("align", "aligned extent overflows".into()))
}

impl<'ctx> DeviceExperts<'ctx> {
    /// Take the run's reservation, build the arena it admitted, and resolve the
    /// plan's selected symbols.
    ///
    /// The reservation moves here because a `DeviceArena` owns the charge it
    /// materialises: that is how task 0010 made "simulated placement presented
    /// as a reservation" impossible to write, and splitting the ownership would
    /// undo it.
    /// The image is **not** a parameter.
    ///
    /// `TrustedImage::from_build_output` is unsafe and its contract is that the
    /// bytes are "an entire, unmodified cubin or fatbin as produced by the
    /// pinned CUDA toolchain at build time"; it checks four magic bytes and
    /// warns that the driver will read past the slice if the image claims to be
    /// larger. A public function taking `&'static [u8]` cannot establish that,
    /// so it asserted a safety contract on its caller's behalf. It now names
    /// this build's own fatbin at the call site, exactly as
    /// `crate::chain` does, and checks that the descriptor the **plan** selected
    /// identifies that same package -- a review attached a plan whose image hash
    /// was all zero and loaded the real fatbin anyway.
    pub fn attach(
        ledger: &mut Ledger,
        reservation: Reservation,
        ctx: &'ctx RankContext,
        plan: &ExpertPlan,
    ) -> std::result::Result<Self, AttachRefused> {
        let fail = |reservation: Reservation, error| AttachRefused {
            reservation: Some(reservation),
            error,
        };
        if plan.device() != ctx.uuid() {
            return Err(fail(
                reservation,
                invalid("device", "this plan names another device".into()),
            ));
        }
        let Some(descriptor) = plan.kernel().cloned() else {
            return Err(fail(
                reservation,
                Error::UnsupportedKernel {
                    operation: "expert_mlp",
                    detail: "this plan selected no device kernel".into(),
                },
            ));
        };
        let extents = match DeviceExtents::of(plan) {
            Ok(extents) => extents,
            Err(error) => return Err(fail(reservation, error)),
        };
        let envelope = plan.envelope();
        let activation_region = envelope.device_bytes(DeviceTier::Activations);
        let workspace_region = envelope.device_bytes(DeviceTier::KernelWorkspace);
        let staging_region = envelope.device_bytes(DeviceTier::TransferStaging);
        // The arena partitions exactly what the plan reserved. A region larger
        // than its charge would be the unadmitted allocation this whole split
        // exists to prevent.
        let regions = [
            (DeviceTier::Activations, activation_region),
            (DeviceTier::KernelWorkspace, workspace_region),
            (DeviceTier::TransferStaging, staging_region),
        ];
        let capacity = activation_region
            .checked_add(workspace_region)
            .and_then(|v| v.checked_add(staging_region));
        let Some(capacity) = capacity else {
            return Err(fail(
                reservation,
                invalid("envelope", "the device envelope overflows".into()),
            ));
        };
        let mut arena = match DeviceArena::create_partitioned(
            ledger,
            reservation,
            ctx,
            &regions,
            capacity,
            format!("expert plan on {}", ctx.uuid()),
        ) {
            Ok(arena) => arena,
            Err(refused) => return Err(fail(refused.reservation, refused.error)),
        };

        // Everything below can fail, and every failure has to give the arena --
        // and with it the reservation -- back. `unwind` collects whatever was
        // allocated so far and releases it in order before closing, because a
        // dropped `DeviceRange` leaves its allocation live and `close` then
        // refuses. That refusal would strand the charge *and* the device memory
        // behind a tidy-looking error return.
        let mut taken: Vec<DeviceRange<'ctx>> = Vec::with_capacity(5);
        macro_rules! unwind {
            ($error:expr) => {{
                let error = $error;
                for range in taken.drain(..) {
                    let _ = arena.release(range);
                }
                return Err(close_and_fail(arena, ledger, error));
            }};
        }
        for (bytes, owner) in [
            (extents.activations, "expert activations"),
            (extents.slots, "expert slots"),
            (extents.workspace, "expert workspace"),
            (extents.indices, "expert row index"),
            (extents.indices, "expert slot index"),
        ] {
            let aligned = match align_up(bytes) {
                Ok(aligned) => aligned,
                Err(error) => unwind!(error),
            };
            match arena.allocate(aligned, ALIGNMENT, owner.to_string()) {
                Ok(range) => taken.push(range),
                Err(refused) => unwind!(refused.error),
            }
        }

        let module = (|| -> Result<ResolvedModule<'ctx>> {
            // Identity before loading, and **whole** identity. The hash says
            // the plan named this package; it says nothing about whether the
            // rest of the descriptor is one this package actually declares. A
            // review kept the built-in GeGLU descriptor's operation, ABI and
            // hash and swapped its projection symbol for the SiLU one: planning,
            // attachment and execution all succeeded, and the answer was SwiGLU
            // on all three GPUs -- 3,959 of 4,096 components wrong, and wrong in
            // a way no gate would have noticed, because it was exactly the other
            // activation.
            //
            // So the descriptor must **be** one of the built-in package's own.
            // That binds operation, ABI, operand roles, precisions, rounding,
            // layout, shape bounds, SM, workspace and symbols together, which is
            // the only form of this check that cannot be half-satisfied. The
            // planner still selects from an injected catalogue -- that is task
            // 0012's design -- and this is the boundary where selection becomes
            // a launch.
            if descriptor.image_sha256 != expert_package_sha256()? {
                return Err(Error::UnsupportedKernel {
                    operation: "expert_mlp",
                    detail: "the selected descriptor does not identify this build's grouped \
                             expert package"
                        .into(),
                });
            }
            if !moxie_kernels::expert_mlp_catalogue()
                .descriptors()
                .contains(&descriptor)
            {
                return Err(Error::UnsupportedKernel {
                    operation: "expert_mlp",
                    detail: format!(
                        "descriptor {} names this build's package but is not one of the {} it \
                         declares; its operation and its symbols are not bound",
                        descriptor.id.0,
                        moxie_kernels::expert_mlp_catalogue().descriptors().len()
                    ),
                });
            }
            // SAFETY: the image is this build's own fatbin, embedded by
            // `moxie-kernels`'s build script through `include_bytes!` of its
            // `build.rs` output. That provenance is exactly what `TrustedImage`
            // asks the caller to assert, and it is why the bytes are named here
            // rather than accepted from a caller who cannot assert it.
            let trusted =
                unsafe { TrustedImage::from_build_output(moxie_kernels::EXPERT_MLP_FATBIN)? };
            let names: Vec<String> = descriptor
                .symbols
                .iter()
                .map(|symbol| symbol.0.clone())
                .collect();
            Module::load(ctx, ModuleImage::Binary(trusted))?.resolve_all(&names)
        })();
        let module = match module {
            Ok(module) => module,
            Err(error) => unwind!(error),
        };
        let stream = match Stream::new(ctx) {
            Ok(stream) => stream,
            Err(error) => unwind!(error),
        };

        let mut taken = taken.into_iter();
        Ok(DeviceExperts {
            ctx,
            stream,
            module,
            descriptor,
            arena: Some(arena),
            activations: taken.next(),
            slots: taken.next(),
            workspace: taken.next(),
            row_index: taken.next(),
            slot_index: taken.next(),
            hidden: plan.shape().hidden,
            intermediate: plan.shape().intermediate,
            rows: plan.rows(),
            slot_count: plan.slot_count(),
            activations_loaded: false,
            quarantined: false,
        })
    }

    pub const fn stream(&self) -> &Stream<'ctx> {
        &self.stream
    }

    pub const fn descriptor(&self) -> &SemanticKernelDescriptor {
        &self.descriptor
    }

    /// Upload the activation block once per run.
    ///
    /// Every failure after the copy is enqueued is reported with the submission
    /// state **unknown**: the source is the run's host buffer and something may
    /// still be reading it. This path used to return an ordinary error, which is
    /// the launch path's old defect on its neighbour.
    /// Crate-internal: after an unknown submission the **caller's** buffer is
    /// still being read, and only `GroupedRun` retains one. A public entry point
    /// taking a borrowed slice ends that borrow on return, so a direct caller
    /// could free the source while the copy is in flight.
    pub(crate) fn load_activations(&mut self, x: &[u8]) -> std::result::Result<(), LaunchRefused> {
        let plain = |error: Error| LaunchRefused {
            error,
            submission_unknown: false,
        };
        if self.quarantined {
            return Err(plain(invalid(
                "attachment",
                "this attachment is quarantined; its ranges may still be in use".into(),
            )));
        }
        // The **exact** logical extent. The range is padded to 256 bytes, so
        // "no larger than the allocation" accepted an empty block and left the
        // device holding whatever the previous load put there -- which a review
        // reproduced against a 4,096-byte activation block.
        let expected = self.rows * self.hidden * 2;
        if x.len() as u64 != expected {
            return Err(plain(Error::InvalidArtifact {
                detail: format!(
                    "activations are {} B, expected exactly {expected} for [{}, {}] BF16",
                    x.len(),
                    self.rows,
                    self.hidden
                ),
            }));
        }
        let range = self.activations.as_ref().expect("live range");
        if expected > range.bytes() {
            return Err(plain(invalid(
                "activations",
                format!("{expected} B exceeds the admitted {} B", range.bytes()),
            )));
        }
        // SAFETY: the copy is enqueued on this attachment's own stream and the
        // event below is waited on before anything reads the destination, so the
        // source cannot be reused while the copy is in flight (document 02).
        if let Err(error) = unsafe { range.copy_from_host_async(x, &self.stream) } {
            // The enqueue itself failed, so nothing was submitted.
            return Err(plain(error));
        }
        // Past the enqueue, every failure leaves the source in use and this
        // attachment unable to give its ranges back.
        let mut unknown = |error: Error| {
            self.quarantined = true;
            LaunchRefused {
                error,
                submission_unknown: true,
            }
        };
        let event = match Event::new(self.ctx) {
            Ok(event) => event,
            Err(error) => return Err(unknown(error)),
        };
        if let Err(error) = event.record(&self.stream) {
            return Err(unknown(error));
        }
        if let Err(error) = event.synchronize() {
            return Err(unknown(error));
        }
        self.activations_loaded = true;
        Ok(())
    }

    pub const fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    /// Run one group: stage its indices, project, reduce, and read its slots
    /// back into the host slot buffer.
    ///
    /// Every operand is checked **before** anything is enqueued. The leases are
    /// checked against the authority that issued them *and* against the backing
    /// they are about to be resolved through, because a lease is an offset and an
    /// offset means nothing without the allocation it belongs to: a review drove
    /// authority A's leases through authority B's backing on a real GPU and got
    /// a confident, different answer. The indices are checked against this
    /// attachment's own extents, because `ExpertGroup` is data and a row index
    /// one larger than the batch reads outside the activation block.
    /// Crate-internal, for the reason [`DeviceExperts::load_activations`] gives:
    /// the staging buffers and the weight leases are the **run's**, and the run
    /// is what withholds them when a launch's submission state is unknown.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_group(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        residency: &DeviceResidency<'ctx>,
        staging: ExpertStaging<'_>,
        host_slots: &mut [u8],
    ) -> std::result::Result<(), LaunchRefused> {
        let result = self.run_group_inner(
            authority, group, gate_up, down, residency, staging, host_slots,
        );
        // One place decides that this attachment can never give its ranges back,
        // rather than each of the dozen early returns remembering to.
        if let Err(refused) = &result
            && refused.submission_unknown
        {
            self.quarantine();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_group_inner(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        residency: &DeviceResidency<'ctx>,
        staging: ExpertStaging<'_>,
        host_slots: &mut [u8],
    ) -> std::result::Result<(), LaunchRefused> {
        let plain = |error: Error| LaunchRefused {
            error,
            submission_unknown: false,
        };
        if self.quarantined {
            return Err(plain(invalid(
                "attachment",
                "this attachment is quarantined; its ranges may still be in use".into(),
            )));
        }
        if !self.activations_loaded {
            return Err(plain(invalid(
                "activations",
                "no activation block has been uploaded to this device".into(),
            )));
        }
        // The backing must belong to the authority that issued these leases and
        // to this attachment's device. Without both checks a lease's offset is
        // resolved inside whatever allocation the caller happened to pass.
        if residency.authority_id() != Some(authority.id()) {
            return Err(plain(invalid(
                "backing",
                format!(
                    "this residency backs authority {:?}; the leases came from {}",
                    residency.authority_id().map(moxie_memory::AuthorityId::get),
                    authority.id().get()
                ),
            )));
        }
        let here = Scope::Device(self.ctx.uuid());
        if residency.scope() != here {
            return Err(plain(invalid(
                "backing",
                format!(
                    "this residency is {}; this attachment is on {}",
                    residency.scope(),
                    self.ctx.uuid()
                ),
            )));
        }
        // And the leases' own scope. One authority may hold a cache on **every**
        // device, so matching the backing to the attachment says nothing about
        // where these particular leases live: a review resolved a 3090 lease's
        // offset inside a 5060 Ti backing, under one authority, and computed
        // another expert's weights. Checking the backing and not the lease is
        // the same half-check, one level down.
        for (what, lease) in [("gate/up", gate_up), ("down", down)] {
            if lease.scope() != here {
                return Err(plain(invalid(
                    "lease",
                    format!(
                        "the {what} lease is resident on {}; this attachment is on {}",
                        lease.scope(),
                        here
                    ),
                )));
            }
        }

        let assignments = group.rows().len() as u64;
        if assignments == 0 || group.slots().len() as u64 != assignments {
            return Err(plain(invalid(
                "group",
                format!(
                    "{} row(s) against {} slot(s)",
                    group.rows().len(),
                    group.slots().len()
                ),
            )));
        }
        if assignments > self.slot_count {
            return Err(plain(invalid(
                "group",
                format!(
                    "{assignments} assignment(s) exceed the {} this attachment admitted",
                    self.slot_count
                ),
            )));
        }
        for row in group.rows() {
            if u64::from(*row) >= self.rows {
                return Err(plain(invalid(
                    "group",
                    format!("row {row} is outside the {} this plan batches", self.rows),
                )));
            }
        }
        let width = self.hidden * 2;
        for slot in group.slots() {
            if u64::from(*slot) >= self.slot_count {
                return Err(plain(invalid(
                    "group",
                    format!(
                        "slot {slot} is outside the {} this plan produces",
                        self.slot_count
                    ),
                )));
            }
        }
        if host_slots.len() as u64 != self.slot_count * width {
            return Err(plain(Error::InvalidArtifact {
                detail: format!(
                    "the host slot buffer is {} B, expected {}",
                    host_slots.len(),
                    self.slot_count * width
                ),
            }));
        }
        let index_bytes = (assignments * 4) as usize;
        if staging.rows.len() < index_bytes || staging.slots.len() < index_bytes {
            return Err(plain(Error::CapacityExceeded {
                tier: Some(moxie_types::Tier::Device(DeviceTier::TransferStaging)),
                requested_bytes: assignments * 4,
                available_bytes: staging.rows.len().min(staging.slots.len()) as u64,
            }));
        }

        let (gate_up_offset, gate_up_len) = authority.device_range(gate_up).map_err(plain)?;
        let (down_offset, down_len) = authority.device_range(down).map_err(plain)?;
        let expected_gate_up = 2 * self.intermediate * self.hidden * 2;
        let expected_down = self.hidden * self.intermediate * 2;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(plain(Error::InvalidArtifact {
                detail: format!(
                    "expert {} is {gate_up_len} + {down_len} B resident, expected \
                     {expected_gate_up} + {expected_down}",
                    group.expert()
                ),
            }));
        }
        let mut gate_up_ptr = residency
            .device_address(gate_up_offset, gate_up_len)
            .map_err(plain)?;
        let mut down_ptr = residency
            .device_address(down_offset, down_len)
            .map_err(plain)?;

        // The index arrays are the operation's `RouteIndex` operand, written
        // into admitted host storage that outlives the copy.
        for (index, row) in group.rows().iter().enumerate() {
            staging.rows[index * 4..index * 4 + 4].copy_from_slice(&row.to_le_bytes());
        }
        for (index, slot) in group.slots().iter().enumerate() {
            staging.slots[index * 4..index * 4 + 4].copy_from_slice(&slot.to_le_bytes());
        }
        let row_range = self.row_index.as_ref().expect("live range");
        let slot_range = self.slot_index.as_ref().expect("live range");

        // From here on an enqueue may have happened, so every failure is
        // reported with the submission state unknown, the caller withholds, and
        // this attachment quarantines its own ranges.
        let unknown = |error: Error| LaunchRefused {
            error,
            submission_unknown: true,
        };
        // SAFETY: both copies are enqueued on this stream and the launches that
        // read them are enqueued after, on the same stream, so ordering is the
        // stream's; the staging slices are the run's admitted buffers, which
        // outlive the copy or are quarantined with it.
        unsafe {
            row_range
                .copy_from_host_async(&staging.rows[..index_bytes], &self.stream)
                .map_err(unknown)?;
            slot_range
                .copy_from_host_async(&staging.slots[..index_bytes], &self.stream)
                .map_err(unknown)?;
        }

        let mut x_ptr = self
            .activations
            .as_ref()
            .expect("live range")
            .device_address()
            .map_err(unknown)?;
        let mut slots_ptr = self
            .slots
            .as_ref()
            .expect("live range")
            .device_address()
            .map_err(unknown)?;
        let mut workspace_ptr = self
            .workspace
            .as_ref()
            .expect("live range")
            .device_address()
            .map_err(unknown)?;
        let mut row_ptr = row_range.device_address().map_err(unknown)?;
        let mut slot_ptr = slot_range.device_address().map_err(unknown)?;
        let mut count = assignments;
        let mut hidden = self.hidden;
        let mut intermediate = self.intermediate;

        let lanes = assignments
            .checked_mul(self.intermediate)
            .ok_or_else(|| unknown(invalid("launch", "projection grid overflowed".into())))?;
        let mut project: [*mut c_void; 7] = [
            (&raw mut x_ptr).cast(),
            (&raw mut row_ptr).cast(),
            (&raw mut gate_up_ptr).cast(),
            (&raw mut workspace_ptr).cast(),
            (&raw mut count).cast(),
            (&raw mut hidden).cast(),
            (&raw mut intermediate).cast(),
        ];
        // SAFETY: the selected descriptor fixes this symbol's ABI; every
        // pointer names a checked admitted range or a live residency lease
        // resolved through this attachment's own backing, and every index was
        // bounds-checked above. The ranges outlive the launch because the event
        // below is waited on, or the operands are withheld.
        unsafe {
            self.module
                .launch_async(
                    0,
                    &self.stream,
                    (grid(lanes).map_err(unknown)?, 1, 1),
                    (BLOCK, 1, 1),
                    0,
                    &mut project,
                )
                .map_err(unknown)?;
        }

        let components = assignments
            .checked_mul(self.hidden)
            .ok_or_else(|| unknown(invalid("launch", "down grid overflowed".into())))?;
        let mut down_params: [*mut c_void; 7] = [
            (&raw mut workspace_ptr).cast(),
            (&raw mut down_ptr).cast(),
            (&raw mut slot_ptr).cast(),
            (&raw mut slots_ptr).cast(),
            (&raw mut count).cast(),
            (&raw mut hidden).cast(),
            (&raw mut intermediate).cast(),
        ];
        // SAFETY: as above, for the second symbol of the same descriptor.
        unsafe {
            self.module
                .launch_async(
                    1,
                    &self.stream,
                    (grid(components).map_err(unknown)?, 1, 1),
                    (BLOCK, 1, 1),
                    0,
                    &mut down_params,
                )
                .map_err(unknown)?;
        }

        let event = Event::new(self.ctx).map_err(unknown)?;
        event.record(&self.stream).map_err(unknown)?;
        event.synchronize().map_err(unknown)?;

        // Read back only this group's slots. The device slot buffer never held
        // the host groups' results, so copying it whole would overwrite them.
        // Past the event, nothing is in flight: a readback failure is ordinary.
        let width = usize::try_from(width).map_err(|_| {
            plain(invalid(
                "hidden",
                "a row wider than this address space".into(),
            ))
        })?;
        let base = self.slots.as_ref().expect("live range");
        for slot in group.slots() {
            let offset = u64::from(*slot) * self.hidden * 2;
            let start = (*slot as usize) * width;
            base.copy_to_host_at(offset, &mut host_slots[start..start + width])
                .map_err(plain)?;
        }
        Ok(())
    }

    /// Mark this attachment's ranges as unreleasable. Irreversible.
    fn quarantine(&mut self) {
        self.quarantined = true;
    }

    /// Release every range and give the arena -- and with it the reservation --
    /// back to the ledger.
    ///
    /// Refuses while this attachment is quarantined: a copy or launch whose
    /// completion could not be established may still be reading these ranges,
    /// and releasing them would report the memory free. Dropping a refused
    /// attachment keeps the arena, its allocation and its charge, which is
    /// `DeviceArena`'s own rule.
    // The refusal carries the whole attachment back, because destroying it on
    // the error path is what left a charged arena unreachable. Boxing it to
    // satisfy a size lint would put an allocation on the failure path of the
    // function whose job is not to lose anything.
    #[allow(clippy::result_large_err)]
    pub fn close(
        mut self,
        ledger: &mut Ledger,
    ) -> std::result::Result<(), DeviceCloseRefused<'ctx>> {
        if self.quarantined {
            return Err(DeviceCloseRefused {
                error: invalid(
                    "close",
                    "this attachment is quarantined; something may still be reading its ranges"
                        .into(),
                ),
                experts: self,
            });
        }
        let Some(mut arena) = self.arena.take() else {
            return Err(DeviceCloseRefused {
                error: invalid("close", "this attachment has already been closed".into()),
                experts: self,
            });
        };
        for range in [
            self.activations.take(),
            self.slots.take(),
            self.workspace.take(),
            self.row_index.take(),
            self.slot_index.take(),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(refused) = arena.release(range) {
                // Put both back, so the attachment stays whole and closable
                // again rather than becoming an unreachable charge.
                let error = refused.error;
                self.activations = Some(refused.range);
                self.arena = Some(arena);
                return Err(DeviceCloseRefused {
                    error,
                    experts: self,
                });
            }
        }
        // `ArenaCloseRefused` hands the arena back; discarding it would destroy
        // the only handle to an allocation that is still charged.
        match arena.close(ledger) {
            Ok(()) => Ok(()),
            Err(refused) => {
                self.arena = Some(refused.arena);
                Err(DeviceCloseRefused {
                    error: refused.error,
                    experts: self,
                })
            }
        }
    }
}

/// A refused attachment close, carrying the attachment back.
#[derive(Debug)]
#[must_use = "the device envelope is still charged; correct the cause and close again"]
pub struct DeviceCloseRefused<'ctx> {
    pub error: Error,
    pub experts: DeviceExperts<'ctx>,
}

impl core::fmt::Display for DeviceCloseRefused<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

/// The grouped expert package's build identity, as bytes.
///
/// `moxie-kernels` records it as lowercase hex at build time; this parses it
/// once so a descriptor can be compared against the package it claims.
fn expert_package_sha256() -> Result<[u8; 32]> {
    let text = moxie_kernels::EXPERT_MLP_FATBIN_SHA256;
    if text.len() != 64 {
        return Err(Error::InvalidArtifact {
            detail: format!("the kernel package digest is {} characters", text.len()),
        });
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).map_err(|_| {
            Error::InvalidArtifact {
                detail: "the kernel package digest is not hexadecimal".into(),
            }
        })?;
    }
    Ok(out)
}

fn grid(elements: u64) -> Result<u32> {
    let blocks = elements.div_ceil(u64::from(BLOCK));
    u32::try_from(blocks).map_err(|_| invalid("launch", "grid exceeds one dimension".into()))
}

/// Close the arena and report the original failure, not the cleanup's.
///
/// If `close` itself refuses -- a quarantined arena, a range still live -- the
/// arena is dropped, which keeps its physical allocation alive and leaves the
/// charge outstanding and visible in `Ledger::outstanding`. That is the same
/// rule a dropped `Reservation` follows everywhere else in this crate:
/// withholding wins, and a leak that is visible beats memory the ledger says is
/// free.
fn close_and_fail(arena: DeviceArena<'_>, ledger: &mut Ledger, error: Error) -> AttachRefused {
    match arena.close(ledger) {
        Ok(()) => AttachRefused {
            reservation: None,
            error,
        },
        Err(refused) => {
            drop(refused);
            AttachRefused {
                reservation: None,
                error,
            }
        }
    }
}

/// A refused attachment.
///
/// `reservation` comes back whenever the failure happened before an arena took
/// it, so the caller can retry or release it. Once an arena owns it, the arena
/// is closed on the way out and the reservation is released with it; if even
/// that refuses, the charge stays outstanding and visible rather than being
/// silently dropped.
#[derive(Debug)]
#[must_use = "the reservation or the charge is still live"]
pub struct AttachRefused {
    pub reservation: Option<Reservation>,
    pub error: Error,
}

impl core::fmt::Display for AttachRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for AttachRefused {}

/// The device lane a [`GroupedRun`] drives, owned by that run.
///
/// It pairs the two halves a device group needs, which are owned differently on
/// purpose: the attachment belongs to this plan alone and dies with it, while
/// the residency cache is the authority's one allocation and outlives any plan,
/// so it is borrowed.
#[derive(Debug)]
pub struct ExpertLane<'a, 'ctx> {
    /// `None` only between taking it for `close` and putting it back on a
    /// refusal.
    experts: Option<DeviceExperts<'ctx>>,
    residency: &'a mut DeviceResidency<'ctx>,
}

impl ExpertDeviceLane for ExpertLane<'_, '_> {
    fn load_activations(&mut self, x: &[u8]) -> std::result::Result<(), LaunchRefused> {
        match self.experts.as_mut() {
            Some(experts) => experts.load_activations(x),
            None => Err(LaunchRefused {
                error: invalid("attachment", "this lane has already been closed".into()),
                submission_unknown: false,
            }),
        }
    }

    fn perform_upload(
        &mut self,
        authority: &mut ResidencyAuthority,
        order: &moxie_memory::WorkOrder,
    ) -> Result<()> {
        // The stream is the attachment's, so the upload and the launches that
        // read it are ordered by the same stream rather than by hope.
        let ExpertLane { experts, residency } = self;
        let Some(experts) = experts.as_ref() else {
            return Err(invalid(
                "attachment",
                "this lane has already been closed".into(),
            ));
        };
        match residency.perform_upload(authority, experts.stream(), order) {
            Ok(()) => Ok(()),
            Err(refused) => {
                // A refusal that never reaches the authority leaves the
                // placement `Uploading` forever, and the cache then refuses to
                // close because a range it never filled is still live. The
                // ticket belongs to *this* authority -- the check that refused
                // was about the backing -- so settling it here is correct and is
                // the only place that can.
                let outcome = if refused.submission_unknown {
                    moxie_memory::Outcome::SubmissionUnknown(refused.error.clone())
                } else {
                    moxie_memory::Outcome::Failed(refused.error.clone())
                };
                let _ = authority.complete_upload(order.ticket(), outcome);
                Err(refused.error)
            }
        }
    }

    fn run_group(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        staging: ExpertStaging<'_>,
        host_slots: &mut [u8],
    ) -> std::result::Result<(), LaunchRefused> {
        let ExpertLane { experts, residency } = self;
        let Some(experts) = experts.as_mut() else {
            return Err(LaunchRefused {
                error: invalid("attachment", "this lane has already been closed".into()),
                submission_unknown: false,
            });
        };
        experts.run_group(
            authority, group, gate_up, down, residency, staging, host_slots,
        )
    }

    fn close(&mut self, ledger: &mut Ledger) -> Result<()> {
        let Some(experts) = self.experts.take() else {
            return Err(invalid(
                "attachment",
                "this lane has already been closed".into(),
            ));
        };
        match experts.close(ledger) {
            Ok(()) => Ok(()),
            Err(refused) => {
                // Put it back, so the run can close again once the cause is
                // corrected rather than losing the handle to a charged arena.
                self.experts = Some(refused.experts);
                Err(refused.error)
            }
        }
    }

    fn is_quarantined(&self) -> bool {
        self.experts
            .as_ref()
            .is_some_and(DeviceExperts::is_quarantined)
    }
}

impl<'lane> GroupedRun<'lane> {
    /// Attach a device to this run: reserve its arena, resolve its kernel, and
    /// keep both inside the run.
    ///
    /// The reservation never leaves the run. An earlier design handed it out and
    /// let the caller release it while these host buffers stayed live and
    /// writable -- a charge of zero against memory the run could still use. One
    /// owner, one `close`.
    pub fn attach_device<'a, 'ctx>(
        &mut self,
        ledger: &mut Ledger,
        ctx: &'ctx RankContext,
        residency: &'a mut DeviceResidency<'ctx>,
    ) -> Result<()>
    where
        'a: 'lane,
        'ctx: 'lane,
    {
        if residency.scope() != Scope::Device(ctx.uuid()) {
            return Err(invalid(
                "residency",
                format!(
                    "this residency is {}; the attachment is on {}",
                    residency.scope(),
                    ctx.uuid()
                ),
            ));
        }
        let reservation = self.take_reservation_for_device()?;
        let experts = match DeviceExperts::attach(ledger, reservation, ctx, self.plan()) {
            Ok(experts) => experts,
            Err(refused) => {
                match refused.reservation {
                    // No arena took it: it goes straight back into the run, so
                    // the host buffers and their charge stay together.
                    Some(reservation) => self.restore_reservation(reservation),
                    // An arena took it and released it on the way out. The
                    // charge is gone, so the buffers go now rather than
                    // outliving it -- which is the gap this whole ownership
                    // change exists to close.
                    None => self.envelope_was_released(&refused.error),
                }
                return Err(refused.error);
            }
        };
        self.install_lane(Box::new(ExpertLane {
            experts: Some(experts),
            residency,
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The operand checks a `GroupedRun` cannot reach.
    //!
    //! `run_group` is crate-internal precisely because the run is what owns and
    //! retains its operands, so the three checks below -- a foreign authority's
    //! backing, a lease resident on another device, and a group from another
    //! plan -- have no external caller to test them through. They live here,
    //! beside the code, and they need real hardware.

    use std::io::Write;
    use std::path::{Path, PathBuf};

    use moxie_cuda::RankContext;
    use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
    use moxie_memory::{
        AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, Content, Ledger,
        ResidencyAuthority, ResidencyRequest, TurnId, UseClass,
    };
    use moxie_plan::expert::{
        ExpertBudget, ExpertKernels, ExpertPolicy, ExpertShape, compile_experts,
    };
    use moxie_storage::Shard;
    use moxie_types::{RankId, StrategyControl};

    use crate::grouped::{ExpertRoles, ExpertStaging, GroupedRun};
    use crate::residency::{ShardSource, drain_reads};

    use super::*;

    const HIDDEN: u64 = 64;
    const INTERMEDIATE: u64 = 16;
    const EXPERTS: u64 = 4;
    const TOP_K: u64 = 2;
    const ROWS: u64 = 4;
    const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
    const MIB: u64 = 1024 * 1024;
    /// Two rows share both experts, two do not: three distinct experts.
    const ROUTE: [u32; 8] = [0, 1, 0, 1, 0, 2, 1, 3];

    fn values(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len * 2);
        for _ in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
            out.extend_from_slice(&moxie_kernels::cpu_expert::to_bf16_bits(unit).to_le_bytes());
        }
        out
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("moxie-grouped-unit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_shard(dir: &Path) -> PathBuf {
        let gate_up = values(0x11, (EXPERTS * 2 * INTERMEDIATE * HIDDEN) as usize);
        let down = values(0x22, (EXPERTS * HIDDEN * INTERMEDIATE) as usize);
        let split = gate_up.len();
        let end = split + down.len();
        let header = format!(
            "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{EXPERTS},{},{HIDDEN}],\
             \"data_offsets\":[0,{split}]}},\
             \"experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{EXPERTS},{HIDDEN},{INTERMEDIATE}],\
             \"data_offsets\":[{split},{end}]}}}}",
            2 * INTERMEDIATE
        );
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&gate_up).unwrap();
        f.write_all(&down).unwrap();
        path
    }

    fn roles() -> ExpertRoles {
        ExpertRoles {
            artifact: ArtifactId::new("grouped-unit-fixture-v1").unwrap(),
            gate_up_role: "experts_gate_up".into(),
            down_role: "experts_down".into(),
            format_version: 1,
        }
    }

    fn source(path: &Path) -> ShardSource {
        ShardSource::new(roles().artifact, vec![Shard::open(path).unwrap()])
            .role("experts_gate_up", 0, "experts.gate_up_proj")
            .unwrap()
            .role("experts_down", 0, "experts.down_proj")
            .unwrap()
    }

    fn mlp(experts: u64, top_k: u64) -> OpParams {
        OpParams::ExpertMlp {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            experts,
            top_k,
            activation: ExpertActivation::GeGlu,
        }
    }

    fn combine(top_k: u64) -> OpParams {
        OpParams::Combine {
            hidden: HIDDEN,
            top_k,
            order: CombineOrder::AscendingExpertId,
        }
    }

    fn budget(ctx: &RankContext) -> ExpertBudget {
        ExpertBudget {
            device: ctx.uuid(),
            device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
            device_cache_cap_bytes: 8 * CHUNK,
            device_cache_leased_bytes: 0,
            device_arena_free_bytes: 16 * MIB,
            host_workspace_bytes: MIB,
            host_buffer_bytes: MIB,
            resident_experts: Vec::new(),
        }
    }

    fn policy() -> ExpertPolicy {
        ExpertPolicy {
            device: StrategyControl::Required,
            host: StrategyControl::Auto,
            host_placement: StrategyControl::Off,
            ..ExpertPolicy::default()
        }
    }

    fn plan_for(ctx: &RankContext, route: &[u32], experts: u64, top_k: u64) -> ExpertPlan {
        let catalogue = moxie_kernels::expert_mlp_catalogue();
        compile_experts(
            &mlp(experts, top_k),
            &combine(top_k),
            route,
            &budget(ctx),
            &policy(),
            None,
            Some(ExpertKernels {
                capability: ctx.capability(),
                catalogue: &catalogue,
            }),
        )
        .expect("a device plan")
    }

    /// Acquire one expert's two chunks into `scope`, uploading through
    /// `residency` on `stream`.
    fn resident<'ctx>(
        authority: &mut ResidencyAuthority,
        residency: &mut DeviceResidency<'ctx>,
        stream: &moxie_cuda::Stream<'ctx>,
        source: &mut ShardSource,
        scope: Scope,
        expert: u32,
        shape: ExpertShape,
    ) -> Vec<moxie_memory::ResidencyLease> {
        let (gate_up, down) = roles().chunks(expert, shape).unwrap();
        let mut leases = Vec::new();
        for chunk in [&gate_up, &down] {
            let acquired = authority
                .acquire(AcquireRequest {
                    chunk,
                    destination: scope,
                    now: 0,
                    deadline: u64::MAX,
                    class: UseClass::demand(Content::Expert),
                    turn: TurnId::new(1),
                })
                .unwrap();
            match acquired {
                Acquired::Ready(lease) => leases.push(lease),
                Acquired::Pending { lease, work, .. } => {
                    for order in drain_reads(authority, source, work).unwrap() {
                        residency.perform_upload(authority, stream, &order).unwrap();
                    }
                    leases.push(lease);
                }
            }
        }
        leases
    }

    struct Staging {
        rows: Vec<u8>,
        slots: Vec<u8>,
        host_slots: Vec<u8>,
    }

    fn staging(slot_count: u64) -> Staging {
        Staging {
            rows: vec![0u8; 256],
            slots: vec![0u8; 256],
            host_slots: vec![0u8; (slot_count * HIDDEN * 2) as usize],
        }
    }

    /// A lease resolved through another authority's backing is refused, and the
    /// same call through the right backing succeeds.
    #[test]
    fn a_launch_refuses_leases_that_belong_to_another_authority() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().unwrap() == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(0), 0).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let dir = scratch("foreign-authority");
        let path = write_shard(&dir);

        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 128 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(scope, 128 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let mut a = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("a", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
        )
        .unwrap();
        let mut b = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("b", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
        )
        .unwrap();
        let mut ra = DeviceResidency::create(&ctx, &mut a).unwrap();
        let rb = DeviceResidency::create(&ctx, &mut b).unwrap();

        let plan = plan_for(&ctx, &ROUTE, EXPERTS, TOP_K);
        let group = plan.groups()[0].clone();
        let request = GroupedRun::request_for(&plan).unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut experts = DeviceExperts::attach(&mut ledger, reservation, &ctx, &plan).unwrap();
        experts
            .load_activations(&values(0x33, (ROWS * HIDDEN) as usize))
            .unwrap();

        let mut src = source(&path);
        let leases = resident(
            &mut a,
            &mut ra,
            experts.stream(),
            &mut src,
            scope,
            group.expert(),
            plan.shape(),
        );
        let mut s = staging(plan.slot_count());
        let refused = experts
            .run_group(
                &a,
                &group,
                &leases[0],
                &leases[1],
                &rb,
                ExpertStaging {
                    rows: &mut s.rows,
                    slots: &mut s.slots,
                },
                &mut s.host_slots,
            )
            .expect_err("authority A's leases must not resolve inside B's allocation");
        assert!(!refused.submission_unknown);
        assert!(
            format!("{}", refused.error).contains("backing"),
            "{}",
            refused.error
        );
        assert!(s.host_slots.iter().all(|byte| *byte == 0));

        // The same call through A's own backing succeeds, so the check refused
        // and not the fixture.
        experts
            .run_group(
                &a,
                &group,
                &leases[0],
                &leases[1],
                &ra,
                ExpertStaging {
                    rows: &mut s.rows,
                    slots: &mut s.slots,
                },
                &mut s.host_slots,
            )
            .expect("the right backing resolves the same leases");
        assert!(s.host_slots.iter().any(|byte| *byte != 0));

        for lease in leases {
            a.release(lease).unwrap();
        }
        experts.close(&mut ledger).unwrap();
        a.end_turn(TurnId::new(1));
        a.retire_all(scope);
        ra.close(&mut a).unwrap();
        rb.close(&mut b).unwrap();
        a.close(&mut ledger).unwrap();
        b.close(&mut ledger).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One authority, two device caches: a lease resident elsewhere is refused.
    #[test]
    fn a_launch_refuses_leases_resident_on_another_device() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let count = moxie_cuda::device_count().unwrap();
        if count < 2 {
            eprintln!("SKIPPED: this case needs two GPUs; {count} visible");
            return;
        }
        let here = RankContext::acquire(RankId(0), 0).unwrap();
        let there = RankContext::acquire(RankId(1), 1).unwrap();
        let dir = scratch("foreign-device");
        let path = write_shard(&dir);

        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 128 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(Scope::Device(here.uuid()), 128 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(Scope::Device(there.uuid()), 128 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        // **One** authority, two device caches.
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("both", 64 * CHUNK)
                .device(here.uuid(), 8 * CHUNK)
                .device(there.uuid(), 8 * CHUNK),
        )
        .unwrap();
        let residency_here = DeviceResidency::create(&here, &mut authority).unwrap();
        let mut residency_there = DeviceResidency::create(&there, &mut authority).unwrap();

        let plan = plan_for(&here, &ROUTE, EXPERTS, TOP_K);
        let group = plan.groups()[0].clone();
        let request = GroupedRun::request_for(&plan).unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut experts = DeviceExperts::attach(&mut ledger, reservation, &here, &plan).unwrap();
        experts
            .load_activations(&values(0x44, (ROWS * HIDDEN) as usize))
            .unwrap();

        let mut src = source(&path);
        let their_stream = moxie_cuda::Stream::new(&there).unwrap();
        let elsewhere = resident(
            &mut authority,
            &mut residency_there,
            &their_stream,
            &mut src,
            Scope::Device(there.uuid()),
            group.expert(),
            plan.shape(),
        );
        let mut s = staging(plan.slot_count());
        let refused = experts
            .run_group(
                &authority,
                &group,
                &elsewhere[0],
                &elsewhere[1],
                &residency_here,
                ExpertStaging {
                    rows: &mut s.rows,
                    slots: &mut s.slots,
                },
                &mut s.host_slots,
            )
            .expect_err("a lease resident elsewhere must not resolve here");
        assert!(!refused.submission_unknown);
        assert!(
            format!("{}", refused.error).contains("resident on"),
            "{}",
            refused.error
        );
        assert!(s.host_slots.iter().all(|byte| *byte == 0));

        for lease in elsewhere {
            authority.release(lease).unwrap();
        }
        experts.close(&mut ledger).unwrap();
        authority.end_turn(TurnId::new(1));
        authority.retire_all(Scope::Device(here.uuid()));
        authority.retire_all(Scope::Device(there.uuid()));
        residency_here.close(&mut authority).unwrap();
        residency_there.close(&mut authority).unwrap();
        authority.close(&mut ledger).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A group from another plan names indices this attachment does not have.
    #[test]
    fn a_group_from_another_plan_is_refused_before_anything_is_enqueued() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().unwrap() == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(0), 0).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let dir = scratch("stranger-group");
        let path = write_shard(&dir);

        // The wide plan's first group names rows the narrow attachment lacks.
        let wide = plan_for(&ctx, &ROUTE, EXPERTS, TOP_K);
        let stranger = wide.groups()[0].clone();
        assert!(stranger.rows().iter().any(|row| *row >= 2));

        // Two rows, eight slots each: the assignment-count bound is satisfied so
        // the **row** bound is the one that has to bite. A fixture that trips
        // three checks at once pins none of them.
        let narrow_route: Vec<u32> = (0..16).collect();
        let narrow = plan_for(&ctx, &narrow_route, 16, 8);
        assert_eq!(narrow.rows(), 2);
        assert!(stranger.rows().len() as u64 <= narrow.slot_count());

        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("narrow", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
        )
        .unwrap();
        let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
        let request = GroupedRun::request_for(&narrow).unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut experts = DeviceExperts::attach(&mut ledger, reservation, &ctx, &narrow).unwrap();
        experts
            .load_activations(&values(0x55, (2 * HIDDEN) as usize))
            .unwrap();

        let mut src = source(&path);
        let leases = resident(
            &mut authority,
            &mut residency,
            experts.stream(),
            &mut src,
            scope,
            stranger.expert(),
            wide.shape(),
        );
        let mut s = staging(narrow.slot_count());
        let refused = experts
            .run_group(
                &authority,
                &stranger,
                &leases[0],
                &leases[1],
                &residency,
                ExpertStaging {
                    rows: &mut s.rows,
                    slots: &mut s.slots,
                },
                &mut s.host_slots,
            )
            .expect_err("a group from another plan names indices this attachment lacks");
        assert!(!refused.submission_unknown);
        let message = format!("{}", refused.error);
        assert!(
            message.contains("row ") && message.contains("batches"),
            "the refusal must name the row index it rejected: {message}"
        );
        assert!(s.host_slots.iter().all(|byte| *byte == 0));

        for lease in leases {
            authority.release(lease).unwrap();
        }
        experts.close(&mut ledger).unwrap();
        authority.end_turn(TurnId::new(1));
        authority.retire_all(scope);
        residency.close(&mut authority).unwrap();
        authority.close(&mut ledger).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A descriptor that names this build's package but is not one of its own
    /// is refused, on every axis separately.
    ///
    /// The review's counterexample is the third case: the built-in GeGLU
    /// descriptor with its projection symbol swapped for the SiLU one. Operation,
    /// ABI and hash all matched, execution succeeded, and the answer was
    /// **exactly SwiGLU** -- 3,959 of 4,096 components wrong on device 0. A
    /// numerical gate cannot catch that, because the result is a correct
    /// evaluation of the wrong function.
    #[test]
    fn a_descriptor_the_package_does_not_declare_is_refused() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().unwrap() == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(0), 0).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let builtin = moxie_kernels::expert_mlp_catalogue();
        let gelu = builtin
            .descriptors()
            .iter()
            .find(|d| {
                d.operation
                    == moxie_types::SemanticKernelOp::ExpertMlp(
                        moxie_types::GateTransform::GeluTanh,
                    )
                    && d.sm.major == ctx.capability().compute_major
                    && d.sm.minor == ctx.capability().compute_minor
            })
            .expect("this build declares a GeGLU descriptor for this device")
            .clone();

        // The unmodified descriptor attaches, so the refusals below are the
        // check biting and not the fixture failing.
        /// One named change to a descriptor, applied alone.
        type Change = Box<dyn Fn(&mut moxie_types::SemanticKernelDescriptor)>;
        let mutate: Vec<(&str, Change)> = vec![
            (
                "none",
                Box::new(|_: &mut moxie_types::SemanticKernelDescriptor| {}),
            ),
            (
                "the other activation's projection symbol",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.symbols[0] = moxie_types::KernelSymbol(
                        moxie_kernels::BF16_EXPERT_PROJECT_SILU.to_string(),
                    );
                }),
            ),
            (
                "a symbol this package does not export",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.symbols[1] = moxie_types::KernelSymbol("moxie_smoke_axpy_f32".into());
                }),
            ),
            (
                "widened shape bounds",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.shape.max_rows = u64::MAX;
                }),
            ),
            (
                "a zeroed image hash",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.image_sha256 = [0; 32];
                }),
            ),
        ];
        for (what, change) in mutate {
            let mut descriptor = gelu.clone();
            change(&mut descriptor);
            let catalogue = moxie_types::KernelCatalogue::new(vec![descriptor]).unwrap();
            let mut ledger = Ledger::new([
                CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
                CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
            ])
            .unwrap();
            let plan = compile_experts(
                &mlp(EXPERTS, TOP_K),
                &combine(TOP_K),
                &ROUTE,
                &budget(&ctx),
                &policy(),
                None,
                Some(ExpertKernels {
                    capability: ctx.capability(),
                    catalogue: &catalogue,
                }),
            )
            .expect("the planner selects from whatever catalogue it is given");
            let request = GroupedRun::request_for(&plan).unwrap();
            let reservation = ledger.admit(&request).unwrap();
            let attached = DeviceExperts::attach(&mut ledger, reservation, &ctx, &plan);
            if what == "none" {
                let experts = attached.expect("the package's own descriptor attaches");
                experts.close(&mut ledger).unwrap();
                assert!(ledger.outstanding().is_empty());
                continue;
            }
            let refused = attached
                .err()
                .unwrap_or_else(|| panic!("a descriptor with {what} was accepted for a launch"));
            assert!(
                format!("{}", refused.error).contains("expert_mlp"),
                "{}",
                refused.error
            );
        }
    }

    /// An activation block of the wrong logical size is refused, whatever the
    /// allocation's padding allows.
    #[test]
    fn an_activation_block_must_be_exactly_its_logical_extent() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().unwrap() == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(0), 0).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let plan = plan_for(&ctx, &ROUTE, EXPERTS, TOP_K);
        let exact = (ROWS * HIDDEN * 2) as usize;
        let request = GroupedRun::request_for(&plan).unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let mut experts = DeviceExperts::attach(&mut ledger, reservation, &ctx, &plan).unwrap();

        experts.load_activations(&values(0x66, exact / 2)).unwrap();
        // Empty, short and long are all refused. Before the fix, the range's
        // 256-byte padding let an empty reload through and the device kept the
        // previous block.
        for wrong in [0usize, exact - 2, exact + 2] {
            let refused = experts
                .load_activations(&vec![0u8; wrong])
                .expect_err("only the exact logical extent is an activation block");
            assert!(!refused.submission_unknown);
            assert!(
                format!("{}", refused.error).contains("expected exactly"),
                "{}",
                refused.error
            );
        }
        experts.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}
