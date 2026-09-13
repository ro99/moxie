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
use moxie_types::{DeviceTier, Error, Result, SemanticKernelDescriptor};

use crate::arena::{DeviceArena, DeviceRange};
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
    arena: DeviceArena<'ctx>,
    activations: Option<DeviceRange<'ctx>>,
    slots: Option<DeviceRange<'ctx>>,
    workspace: Option<DeviceRange<'ctx>>,
    row_index: Option<DeviceRange<'ctx>>,
    slot_index: Option<DeviceRange<'ctx>>,
    hidden: u64,
    intermediate: u64,
    activations_loaded: bool,
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
            arena,
            activations: taken.next(),
            slots: taken.next(),
            workspace: taken.next(),
            row_index: taken.next(),
            slot_index: taken.next(),
            hidden: plan.shape().hidden,
            intermediate: plan.shape().intermediate,
            activations_loaded: false,
        })
    }

    pub const fn stream(&self) -> &Stream<'ctx> {
        &self.stream
    }

    pub const fn descriptor(&self) -> &SemanticKernelDescriptor {
        &self.descriptor
    }

    /// Upload the activation block once per run.
    pub fn load_activations(&mut self, x: &[u8]) -> Result<()> {
        let range = self.activations.as_ref().expect("live range");
        if x.len() as u64 > range.bytes() {
            return Err(invalid(
                "activations",
                format!("{} B exceeds the admitted {} B", x.len(), range.bytes()),
            ));
        }
        // SAFETY: the copy is enqueued on this attachment's own stream and the
        // event below is waited on before anything reads the destination, so the
        // source cannot be reused while the copy is in flight (document 02).
        unsafe { range.copy_from_host_async(x, &self.stream)? };
        let event = Event::new(self.ctx)?;
        event.record(&self.stream)?;
        event.synchronize()?;
        self.activations_loaded = true;
        Ok(())
    }

    /// Run one group: stage its indices, project, reduce, and read its slots
    /// back into the host slot buffer.
    pub fn run_group(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        residency: &DeviceResidency<'ctx>,
        host_slots: &mut [u8],
    ) -> Result<()> {
        if !self.activations_loaded {
            return Err(invalid(
                "activations",
                "no activation block has been uploaded to this device".into(),
            ));
        }
        let assignments = group.rows.len() as u64;
        if assignments == 0 {
            return Err(invalid("group", "a group with no assignment".into()));
        }

        let (gate_up_offset, gate_up_len) = authority.device_range(gate_up)?;
        let (down_offset, down_len) = authority.device_range(down)?;
        let expected_gate_up = 2 * self.intermediate * self.hidden * 2;
        let expected_down = self.hidden * self.intermediate * 2;
        if gate_up_len != expected_gate_up || down_len != expected_down {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "expert {} is {gate_up_len} + {down_len} B resident, expected \
                     {expected_gate_up} + {expected_down}",
                    group.expert
                ),
            });
        }
        let mut gate_up_ptr = residency.device_address(gate_up_offset, gate_up_len)?;
        let mut down_ptr = residency.device_address(down_offset, down_len)?;

        // The index arrays are the operation's `RouteIndex` operand: which rows
        // this launch serves and where each result belongs.
        let rows: Vec<u8> = group.rows.iter().flat_map(|r| r.to_le_bytes()).collect();
        let slots: Vec<u8> = group.slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let row_range = self.row_index.as_ref().expect("live range");
        let slot_range = self.slot_index.as_ref().expect("live range");
        // SAFETY: both copies are enqueued on this stream and the launches that
        // read them are enqueued after, on the same stream, so ordering is the
        // stream's; the event at the end of this function is what releases the
        // host vectors.
        unsafe {
            row_range.copy_from_host_async(&rows, &self.stream)?;
            slot_range.copy_from_host_async(&slots, &self.stream)?;
        }

        let mut x_ptr = self
            .activations
            .as_ref()
            .expect("live range")
            .device_address()?;
        let mut slots_ptr = self.slots.as_ref().expect("live range").device_address()?;
        let mut workspace_ptr = self
            .workspace
            .as_ref()
            .expect("live range")
            .device_address()?;
        let mut row_ptr = row_range.device_address()?;
        let mut slot_ptr = slot_range.device_address()?;
        let mut count = assignments;
        let mut hidden = self.hidden;
        let mut intermediate = self.intermediate;

        let lanes = assignments
            .checked_mul(self.intermediate)
            .ok_or_else(|| invalid("launch", "projection grid overflowed".into()))?;
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
        // pointer names a checked admitted range or a live residency lease, and
        // every dimension was bounded during pure lowering. The ranges outlive
        // the launch because the event below is waited on before this function
        // returns.
        unsafe {
            self.module.launch_async(
                0,
                &self.stream,
                (grid(lanes)?, 1, 1),
                (BLOCK, 1, 1),
                0,
                &mut project,
            )?;
        }

        let components = assignments
            .checked_mul(self.hidden)
            .ok_or_else(|| invalid("launch", "down grid overflowed".into()))?;
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
            self.module.launch_async(
                1,
                &self.stream,
                (grid(components)?, 1, 1),
                (BLOCK, 1, 1),
                0,
                &mut down_params,
            )?;
        }

        let event = Event::new(self.ctx)?;
        event.record(&self.stream)?;
        event.synchronize()?;

        // Read back only this group's slots. The device slot buffer never held
        // the host groups' results, so copying it whole would overwrite them.
        let width = usize::try_from(self.hidden * 2)
            .map_err(|_| invalid("hidden", "a row wider than this address space".into()))?;
        let base = self.slots.as_ref().expect("live range");
        for slot in &group.slots {
            let offset = u64::from(*slot)
                .checked_mul(self.hidden * 2)
                .ok_or_else(|| invalid("slot", "slot offset overflowed".into()))?;
            let start = (*slot as usize)
                .checked_mul(width)
                .ok_or_else(|| invalid("slot", "host slot offset overflowed".into()))?;
            let end = start
                .checked_add(width)
                .filter(|end| *end <= host_slots.len())
                .ok_or_else(|| invalid("slot", "slot lies outside the host buffer".into()))?;
            base.copy_to_host_at(offset, &mut host_slots[start..end])?;
        }
        Ok(())
    }

    /// Release every range and give the arena -- and with it the reservation --
    /// back to the ledger.
    pub fn close(mut self, ledger: &mut Ledger) -> Result<()> {
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
            self.arena.release(range).map_err(|refused| refused.error)?;
        }
        self.arena.close(ledger).map_err(|refused| refused.error)
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

/// The device lane a [`crate::grouped::GroupedRun`] drives.
///
/// It pairs the two halves a device group needs and which are owned separately
/// for good reason: the residency cache is the authority's one allocation and
/// outlives any plan, while the arena and module belong to this plan alone.
#[derive(Debug)]
pub struct ExpertLane<'a, 'ctx> {
    pub experts: &'a mut DeviceExperts<'ctx>,
    pub residency: &'a mut DeviceResidency<'ctx>,
}

impl crate::grouped::ExpertDeviceLane for ExpertLane<'_, '_> {
    fn load_activations(&mut self, x: &[u8]) -> Result<()> {
        self.experts.load_activations(x)
    }

    fn perform_upload(
        &mut self,
        authority: &mut ResidencyAuthority,
        order: &moxie_memory::WorkOrder,
    ) -> Result<()> {
        // The stream is the attachment's, so the upload and the launches that
        // read it are ordered by the same stream rather than by hope.
        self.residency
            .perform_upload(authority, self.experts.stream(), order)
            .map_err(|refused| refused.error)
    }

    fn run_group(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        host_slots: &mut [u8],
    ) -> Result<()> {
        self.experts
            .run_group(authority, group, gate_up, down, self.residency, host_slots)
    }
}
