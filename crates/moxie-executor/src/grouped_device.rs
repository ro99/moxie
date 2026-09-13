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
    pub fn attach(
        ledger: &mut Ledger,
        reservation: Reservation,
        ctx: &'ctx RankContext,
        plan: &ExpertPlan,
        image: &'static [u8],
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
            // SAFETY: the image is this build's own fatbin, embedded by
            // `moxie-kernels`'s build script. That provenance is exactly what
            // `TrustedImage` asks the caller to assert.
            let trusted = unsafe { TrustedImage::from_build_output(image)? };
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
    pub fn load_activations(&mut self, x: &[u8]) -> std::result::Result<(), LaunchRefused> {
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
        let range = self.activations.as_ref().expect("live range");
        if x.len() as u64 > range.bytes() {
            return Err(plain(invalid(
                "activations",
                format!("{} B exceeds the admitted {} B", x.len(), range.bytes()),
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
    #[allow(clippy::too_many_arguments)]
    pub fn run_group(
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
        image: &'static [u8],
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
        let experts = match DeviceExperts::attach(ledger, reservation, ctx, self.plan(), image) {
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
