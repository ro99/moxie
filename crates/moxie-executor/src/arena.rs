//! Suballocated device ranges bound to the shared completion state machine.
//!
//! `moxie-memory::Arena` owns exact range metadata. The driver binding below
//! owns one physical allocation and one parent reservation. Operation leases
//! temporarily own a range while asynchronous work is pending and return it
//! only after the same lifecycle used by whole-reservation leases permits reuse.

use crate::lease::{Completion, LeaseId, LeaseState, Lifecycle, TrackableManual};
use moxie_types::{Error, Result};

/// Refused operation-lease construction, returning the resource unchanged.
#[derive(Debug)]
pub struct OperationAcquireRefused<R> {
    pub resource: R,
    pub error: Error,
}

/// One event-retained use of an already allocated resource.
#[derive(Debug)]
#[must_use = "an operation lease must retire before its resource is reusable"]
pub struct OperationLease<C, R> {
    id: LeaseId,
    label: String,
    lifecycle: Lifecycle<C>,
    resource: Option<R>,
}

impl<C, R> Drop for OperationLease<C, R> {
    fn drop(&mut self) {
        if self.lifecycle.is_tracked_or_lost() {
            // The operation may still touch this resource. Withhold it rather
            // than letting ordinary Drop advertise or free its storage.
            std::mem::forget(self.resource.take());
        }
    }
}

impl<C: Completion, R> OperationLease<C, R> {
    pub fn new(
        label: impl Into<String>,
        resource: R,
    ) -> std::result::Result<Self, OperationAcquireRefused<R>> {
        let label = label.into();
        if label.is_empty() {
            return Err(OperationAcquireRefused {
                resource,
                error: invalid("label", "an operation lease must be named"),
            });
        }
        Ok(Self {
            id: LeaseId::next(),
            label,
            lifecycle: Lifecycle::new(),
            resource: Some(resource),
        })
    }

    pub const fn id(&self) -> LeaseId {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn state(&self) -> LeaseState {
        self.lifecycle.state()
    }

    pub fn resource(&self) -> &R {
        self.resource
            .as_ref()
            .expect("operation lease retains its resource until retirement")
    }

    #[cfg(feature = "driver")]
    pub(crate) fn resource_mut(&mut self) -> &mut R {
        self.resource
            .as_mut()
            .expect("operation lease retains its resource until retirement")
    }

    pub fn track_manual(&mut self, completion: C) -> Result<()>
    where
        C: TrackableManual,
    {
        self.lifecycle
            .track_manual(completion, &format!("{} ({})", self.id, self.label))
    }

    #[cfg(feature = "driver")]
    pub(crate) fn submit_tracked(&mut self, completion: C) -> Result<()> {
        if self.lifecycle.state() != LeaseState::Live {
            return Err(invalid("lease", "completion can only be recorded once"));
        }
        self.lifecycle.submit(completion);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.lifecycle.cancel();
    }

    pub fn mark_lost(&mut self, device: u32, detail: impl Into<String>) {
        self.lifecycle.mark_lost(device, detail);
    }

    pub fn synchronize(&mut self) -> Result<()> {
        self.lifecycle.synchronize()
    }

    #[allow(clippy::result_large_err)]
    pub fn retire(mut self) -> std::result::Result<(LeaseId, R), OperationRetireRefused<C, R>> {
        let refuse = |lease: Self, error| OperationRetireRefused { lease, error };
        if self.lifecycle.state() == LeaseState::Lost {
            let description = format!("{} ({})", self.id, self.label);
            let error = self.lifecycle.lost_error(&description);
            return Err(refuse(self, error));
        }
        let complete = match self.lifecycle.observe_complete() {
            Ok(done) => done,
            Err(error) => return Err(refuse(self, error)),
        };
        if !complete {
            let error = invalid(
                "lease",
                format!(
                    "{} ({}) is still {:?}: completion not observed",
                    self.id,
                    self.label,
                    self.lifecycle.state()
                ),
            );
            return Err(refuse(self, error));
        }
        let id = self.id;
        Ok((
            id,
            self.resource
                .take()
                .expect("operation lease retains its resource"),
        ))
    }
}

/// Refused nonblocking retirement, returning the entire operation lease.
#[derive(Debug)]
pub struct OperationRetireRefused<C, R> {
    pub lease: OperationLease<C, R>,
    pub error: Error,
}

/// One operation returned by a turn sweep.
#[derive(Debug)]
pub struct OperationRetired<R> {
    pub id: LeaseId,
    pub resource: R,
}

/// One still-held operation returned by a turn sweep.
#[derive(Debug)]
pub struct OperationHeld<C, R> {
    pub id: LeaseId,
    pub label: String,
    pub reason: String,
    pub lease: OperationLease<C, R>,
}

#[derive(Debug)]
pub struct OperationTurnReport<C, R> {
    pub retired: Vec<OperationRetired<R>>,
    pub held: Vec<OperationHeld<C, R>>,
}

impl<C, R> OperationTurnReport<C, R> {
    pub fn is_clean(&self) -> bool {
        self.held.is_empty()
    }
}

/// Turn-boundary owner for suballocated operation leases.
#[derive(Debug, Default)]
pub struct OperationTurn<C, R> {
    label: String,
    leases: Vec<OperationLease<C, R>>,
}

impl<C: Completion, R> OperationTurn<C, R> {
    pub fn new(label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(invalid("label", "an operation turn must be named"));
        }
        Ok(Self {
            label,
            leases: Vec::new(),
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn hold(&mut self, lease: OperationLease<C, R>) {
        self.leases.push(lease);
    }

    pub fn synchronize(&mut self) -> Result<()> {
        for lease in &mut self.leases {
            lease.synchronize()?;
        }
        Ok(())
    }

    pub fn release_turn(self) -> OperationTurnReport<C, R> {
        let mut report = OperationTurnReport {
            retired: Vec::new(),
            held: Vec::new(),
        };
        for lease in self.leases {
            let id = lease.id();
            let label = lease.label().to_string();
            match lease.retire() {
                Ok((id, resource)) => report.retired.push(OperationRetired { id, resource }),
                Err(OperationRetireRefused { lease, error }) => {
                    report.held.push(OperationHeld {
                        id,
                        label,
                        reason: error.to_string(),
                        lease,
                    });
                }
            }
        }
        report
    }
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(feature = "driver")]
mod driver_binding {
    use std::cell::Cell;
    use std::rc::Rc;

    use moxie_cuda::{DeviceBuffer, Event, RankContext, Stream};
    use moxie_memory::{
        AllocateRefused, Allocation, Arena, ArenaOccupancy, Ledger, LedgerId,
        OutstandingAllocation, Reservation,
    };
    use moxie_types::{DeviceTier, Error, HostTier, Scope, Tier};

    use super::{OperationLease, invalid};
    use crate::lease::LeaseState;

    const DEVICE_ARENA_ALIGNMENT: u64 = 256;

    #[derive(Debug)]
    struct ArenaCore<'ctx> {
        buffer: DeviceBuffer<'ctx>,
        device_ordinal: u32,
        active_upload: Cell<bool>,
        host_pageable_budget: u64,
    }

    /// A real bounded device arena and its parent ledger reservation.
    #[derive(Debug)]
    pub struct DeviceArena<'ctx> {
        metadata: Arena,
        core: Option<Rc<ArenaCore<'ctx>>>,
        reservation: Option<Reservation>,
        ledger: LedgerId,
        scope: Scope,
        tier: DeviceTier,
        quarantined: Option<Error>,
    }

    /// Arena creation refusal, returning the still-charged reservation.
    #[derive(Debug)]
    pub struct ArenaCreateRefused {
        pub reservation: Reservation,
        pub error: Error,
    }

    /// Arena close refusal, returning the complete arena for reporting.
    #[derive(Debug)]
    pub struct ArenaCloseRefused<'ctx> {
        pub arena: DeviceArena<'ctx>,
        pub error: Error,
    }

    /// One live range in a device arena.
    #[derive(Debug)]
    #[must_use = "a dropped device range stays allocated; release it through its arena"]
    pub struct DeviceRange<'ctx> {
        allocation: Option<Allocation>,
        core: Rc<ArenaCore<'ctx>>,
    }

    impl<'ctx> DeviceRange<'ctx> {
        fn allocation(&self) -> &Allocation {
            self.allocation
                .as_ref()
                .expect("a live range has its handle")
        }

        pub fn key(&self) -> moxie_memory::AllocationKey {
            self.allocation().key()
        }

        pub fn offset(&self) -> u64 {
            self.allocation().offset()
        }

        pub fn bytes(&self) -> u64 {
            self.allocation().bytes()
        }

        pub fn owner(&self) -> &str {
            self.allocation().owner()
        }

        pub fn device_uuid(&self) -> moxie_types::DeviceUuid {
            self.core.buffer.device_uuid()
        }

        pub(crate) fn device_address(&self) -> moxie_types::Result<u64> {
            self.core
                .buffer
                .device_ptr()
                .checked_add(self.offset())
                .ok_or_else(|| invalid("range", "device address overflowed"))
        }

        /// Enqueue a checked copy while a higher-level operation lease owns
        /// this range and `source` through its completion event.
        pub(crate) unsafe fn copy_from_host_async(
            &self,
            source: &[u8],
            stream: &Stream<'ctx>,
        ) -> moxie_types::Result<()> {
            if source.len() as u64 > self.bytes() {
                return Err(invalid("source", "binding exceeds its admitted range"));
            }
            let offset = usize::try_from(self.offset())
                .map_err(|_| invalid("range", "range offset is not addressable"))?;
            // SAFETY: forwarded to the caller; the chain operation lease owns
            // source and this range until its recorded event completes.
            unsafe {
                self.core
                    .buffer
                    .copy_from_host_async_at(offset, source, stream)
            }
        }

        pub(crate) fn copy_to_host(&self, destination: &mut [u8]) -> moxie_types::Result<()> {
            if destination.len() as u64 > self.bytes() {
                return Err(invalid(
                    "destination",
                    "readback exceeds its admitted range",
                ));
            }
            let offset = usize::try_from(self.offset())
                .map_err(|_| invalid("range", "range offset is not addressable"))?;
            self.core.buffer.copy_to_host_at(offset, destination)
        }

        #[allow(clippy::result_large_err)]
        pub fn prepare_upload(
            self,
            source: Vec<u8>,
            label: impl Into<String>,
        ) -> std::result::Result<
            OperationLease<Event<'ctx>, ArenaUpload<'ctx>>,
            PrepareArenaUploadRefused<'ctx>,
        > {
            let label = label.into();
            let error = if label.is_empty() {
                Some(invalid("label", "an arena upload must be named"))
            } else if source.is_empty() || source.len() as u64 > self.bytes() {
                Some(invalid(
                    "source",
                    "an arena upload must be nonempty and fit its range",
                ))
            } else if source.capacity() as u64 > self.core.host_pageable_budget {
                Some(Error::CapacityExceeded {
                    tier: Some(Tier::Host(HostTier::Pageable)),
                    requested_bytes: source.capacity() as u64,
                    available_bytes: self.core.host_pageable_budget,
                })
            } else if self.core.active_upload.replace(true) {
                Some(invalid(
                    "upload",
                    "this basic arena permits one retained host source at a time",
                ))
            } else {
                None
            };
            if let Some(error) = error {
                return Err(PrepareArenaUploadRefused {
                    range: self,
                    source,
                    error,
                });
            }
            let upload = ArenaUpload {
                range: Some(self),
                source: Some(source),
            };
            // Label was validated above, so this cannot refuse.
            Ok(OperationLease::new(label, upload).expect("validated nonempty label"))
        }
    }

    /// Retained source and destination range for one async upload.
    #[derive(Debug)]
    pub struct ArenaUpload<'ctx> {
        range: Option<DeviceRange<'ctx>>,
        source: Option<Vec<u8>>,
    }

    impl<'ctx> ArenaUpload<'ctx> {
        pub fn source(&self) -> &[u8] {
            self.source.as_deref().expect("upload retains source")
        }

        pub fn range(&self) -> &DeviceRange<'ctx> {
            self.range.as_ref().expect("upload retains range")
        }

        pub fn finish(mut self) -> (DeviceRange<'ctx>, Vec<u8>) {
            let range = self.range.take().expect("upload retains range");
            let source = self.source.take().expect("upload retains source");
            range.core.active_upload.set(false);
            (range, source)
        }
    }

    impl Drop for ArenaUpload<'_> {
        fn drop(&mut self) {
            if let Some(range) = &self.range {
                // Ordinary Drop is reachable only before submission or after a
                // successful retirement. In-flight OperationLease drop forgets
                // this value, preserving the hold instead.
                range.core.active_upload.set(false);
            }
        }
    }

    #[derive(Debug)]
    pub struct PrepareArenaUploadRefused<'ctx> {
        pub range: DeviceRange<'ctx>,
        pub source: Vec<u8>,
        pub error: Error,
    }

    #[derive(Debug)]
    pub struct RangeReleaseRefused<'ctx> {
        pub range: DeviceRange<'ctx>,
        pub error: Error,
    }

    #[derive(Debug)]
    pub struct RangeTransferRefused<'ctx> {
        pub range: DeviceRange<'ctx>,
        pub error: Error,
    }

    impl<'ctx> DeviceArena<'ctx> {
        /// Materialize one physical allocation whose charged regions span
        /// multiple device tiers. The region byte sum is exact and every tier
        /// is validated against the same parent reservation before allocation.
        pub(crate) fn create_partitioned(
            ledger: &Ledger,
            reservation: Reservation,
            ctx: &'ctx RankContext,
            regions: &[(DeviceTier, u64)],
            capacity: u64,
            label: impl Into<String>,
        ) -> std::result::Result<Self, ArenaCreateRefused> {
            let fail = |reservation, error| ArenaCreateRefused { reservation, error };
            let scope = Scope::Device(ctx.uuid());
            if reservation.ledger() != ledger.id() {
                return Err(fail(
                    reservation,
                    invalid("reservation", "reservation belongs to another ledger"),
                ));
            }
            if regions.is_empty() {
                return Err(fail(
                    reservation,
                    invalid("regions", "a partitioned arena needs charged regions"),
                ));
            }
            let mut total = 0u64;
            let mut seen = std::collections::BTreeSet::new();
            for (tier, bytes) in regions {
                if *bytes == 0
                    || !seen.insert(*tier)
                    || matches!(
                        tier,
                        DeviceTier::SafetyHeadroom | DeviceTier::AllocatorFragmentation
                    )
                {
                    return Err(fail(
                        reservation,
                        invalid("regions", "regions must be unique, nonzero payload tiers"),
                    ));
                }
                total = match total.checked_add(*bytes) {
                    Some(total) => total,
                    None => {
                        return Err(fail(
                            reservation,
                            invalid("regions", "region byte sum overflowed"),
                        ));
                    }
                };
            }
            if total != capacity
                || capacity == 0
                || !capacity.is_multiple_of(DEVICE_ARENA_ALIGNMENT)
                || capacity > usize::MAX as u64
            {
                return Err(fail(
                    reservation,
                    invalid(
                        "capacity",
                        "partitioned region sum must equal one aligned addressable arena",
                    ),
                ));
            }
            let Some(record) = ledger
                .outstanding()
                .into_iter()
                .find(|record| record.id == reservation.id())
            else {
                return Err(fail(
                    reservation,
                    invalid("reservation", "reservation is not outstanding"),
                ));
            };
            if record.scope_charges.iter().any(|(candidate, bytes)| {
                matches!(candidate, Scope::Device(_)) && *candidate != scope && *bytes != 0
            }) {
                return Err(fail(
                    reservation,
                    invalid(
                        "reservation",
                        "one device arena cannot own another UUID's charge",
                    ),
                ));
            }
            let scope_budget = record
                .scope_charges
                .iter()
                .find(|(candidate, _)| *candidate == scope)
                .map(|(_, bytes)| *bytes)
                .unwrap_or(0);
            if capacity > scope_budget {
                return Err(fail(
                    reservation,
                    Error::CapacityExceeded {
                        tier: None,
                        requested_bytes: capacity,
                        available_bytes: scope_budget,
                    },
                ));
            }
            for (tier, bytes) in regions {
                let tier_budget = record
                    .charges
                    .iter()
                    .find(|(candidate, candidate_tier, _)| {
                        *candidate == scope && *candidate_tier == Tier::Device(*tier)
                    })
                    .map(|(_, _, bytes)| *bytes)
                    .unwrap_or(0);
                if *bytes > tier_budget {
                    return Err(fail(
                        reservation,
                        Error::CapacityExceeded {
                            tier: Some(Tier::Device(*tier)),
                            requested_bytes: *bytes,
                            available_bytes: tier_budget,
                        },
                    ));
                }
            }
            let metadata = match Arena::new(label, capacity, DEVICE_ARENA_ALIGNMENT) {
                Ok(arena) => arena,
                Err(error) => return Err(fail(reservation, error)),
            };
            let host_pageable_budget = record
                .charges
                .iter()
                .find(|(candidate, candidate_tier, _)| {
                    *candidate == Scope::Host && *candidate_tier == Tier::Host(HostTier::Pageable)
                })
                .map(|(_, _, bytes)| *bytes)
                .unwrap_or(0);
            let buffer = match DeviceBuffer::alloc(ctx, capacity as usize) {
                Ok(buffer) => buffer,
                Err(error) => return Err(fail(reservation, error)),
            };
            Ok(Self {
                metadata,
                core: Some(Rc::new(ArenaCore {
                    buffer,
                    device_ordinal: ctx.ordinal(),
                    active_upload: Cell::new(false),
                    host_pageable_budget,
                })),
                reservation: Some(reservation),
                ledger: ledger.id(),
                scope,
                tier: regions[0].0,
                quarantined: None,
            })
        }

        pub fn create(
            ledger: &Ledger,
            reservation: Reservation,
            ctx: &'ctx RankContext,
            tier: DeviceTier,
            capacity: u64,
            label: impl Into<String>,
        ) -> std::result::Result<Self, ArenaCreateRefused> {
            let fail = |reservation, error| ArenaCreateRefused { reservation, error };
            let scope = Scope::Device(ctx.uuid());
            if reservation.ledger() != ledger.id() {
                return Err(fail(
                    reservation,
                    invalid("reservation", "reservation belongs to another ledger"),
                ));
            }
            if matches!(
                tier,
                DeviceTier::SafetyHeadroom | DeviceTier::AllocatorFragmentation
            ) {
                return Err(fail(
                    reservation,
                    invalid("tier", "headroom and fragmentation are not payload arenas"),
                ));
            }
            let Some(record) = ledger
                .outstanding()
                .into_iter()
                .find(|record| record.id == reservation.id())
            else {
                return Err(fail(
                    reservation,
                    invalid("reservation", "reservation is not outstanding"),
                ));
            };
            if record.scope_charges.iter().any(|(candidate, bytes)| {
                matches!(candidate, Scope::Device(_)) && *candidate != scope && *bytes != 0
            }) {
                return Err(fail(
                    reservation,
                    invalid(
                        "reservation",
                        "one device arena cannot own charges for another device UUID",
                    ),
                ));
            }
            let scope_budget = record
                .scope_charges
                .iter()
                .find(|(candidate, _)| *candidate == scope)
                .map(|(_, bytes)| *bytes)
                .unwrap_or(0);
            let tier_budget = record
                .charges
                .iter()
                .find(|(candidate, candidate_tier, _)| {
                    *candidate == scope && *candidate_tier == Tier::Device(tier)
                })
                .map(|(_, _, bytes)| *bytes)
                .unwrap_or(0);
            if capacity == 0
                || !capacity.is_multiple_of(DEVICE_ARENA_ALIGNMENT)
                || capacity > usize::MAX as u64
            {
                return Err(fail(
                    reservation,
                    invalid(
                        "capacity",
                        "device arena capacity must be nonzero, addressable and 256-byte aligned",
                    ),
                ));
            }
            if capacity > scope_budget || capacity > tier_budget {
                return Err(fail(
                    reservation,
                    Error::CapacityExceeded {
                        tier: Some(Tier::Device(tier)),
                        requested_bytes: capacity,
                        available_bytes: scope_budget.min(tier_budget),
                    },
                ));
            }
            let metadata = match Arena::new(label, capacity, DEVICE_ARENA_ALIGNMENT) {
                Ok(arena) => arena,
                Err(error) => return Err(fail(reservation, error)),
            };
            let host_pageable_budget = record
                .charges
                .iter()
                .find(|(candidate, candidate_tier, _)| {
                    *candidate == Scope::Host && *candidate_tier == Tier::Host(HostTier::Pageable)
                })
                .map(|(_, _, bytes)| *bytes)
                .unwrap_or(0);
            let buffer = match DeviceBuffer::alloc(ctx, capacity as usize) {
                Ok(buffer) => buffer,
                Err(error) => return Err(fail(reservation, error)),
            };
            Ok(Self {
                metadata,
                core: Some(Rc::new(ArenaCore {
                    buffer,
                    device_ordinal: ctx.ordinal(),
                    active_upload: Cell::new(false),
                    host_pageable_budget,
                })),
                reservation: Some(reservation),
                ledger: ledger.id(),
                scope,
                tier,
                quarantined: None,
            })
        }

        pub fn scope(&self) -> Scope {
            self.scope
        }

        pub const fn tier(&self) -> DeviceTier {
            self.tier
        }

        pub fn occupancy(&self) -> ArenaOccupancy {
            self.metadata.occupancy()
        }

        pub fn outstanding(&self) -> Vec<OutstandingAllocation> {
            self.metadata.outstanding()
        }

        pub fn allocate(
            &mut self,
            bytes: u64,
            alignment: u64,
            owner: impl Into<String>,
        ) -> std::result::Result<DeviceRange<'ctx>, AllocateRefused> {
            if let Some(error) = &self.quarantined {
                return Err(AllocateRefused {
                    error: error.clone(),
                    occupancy: self.metadata.occupancy(),
                });
            }
            let allocation = self.metadata.allocate(bytes, alignment, owner)?;
            Ok(DeviceRange {
                allocation: Some(allocation),
                core: Rc::clone(self.core.as_ref().expect("open arena has a core")),
            })
        }

        #[allow(clippy::result_large_err)]
        pub fn release(
            &mut self,
            mut range: DeviceRange<'ctx>,
        ) -> std::result::Result<(), RangeReleaseRefused<'ctx>> {
            if !Rc::ptr_eq(
                self.core.as_ref().expect("open arena has a core"),
                &range.core,
            ) {
                return Err(RangeReleaseRefused {
                    range,
                    error: invalid("range", "range belongs to another device arena"),
                });
            }
            let allocation = range.allocation.take().expect("live range has allocation");
            match self.metadata.release(allocation) {
                Ok(()) => Ok(()),
                Err(refused) => {
                    range.allocation = Some(refused.allocation);
                    Err(RangeReleaseRefused {
                        range,
                        error: refused.error,
                    })
                }
            }
        }

        #[allow(clippy::result_large_err)]
        pub fn transfer(
            &mut self,
            mut range: DeviceRange<'ctx>,
            owner: impl Into<String>,
        ) -> std::result::Result<DeviceRange<'ctx>, RangeTransferRefused<'ctx>> {
            if !Rc::ptr_eq(
                self.core.as_ref().expect("open arena has a core"),
                &range.core,
            ) {
                return Err(RangeTransferRefused {
                    range,
                    error: invalid("range", "range belongs to another device arena"),
                });
            }
            let allocation = range.allocation.take().expect("live range has allocation");
            match self.metadata.transfer(allocation, owner) {
                Ok(allocation) => {
                    range.allocation = Some(allocation);
                    Ok(range)
                }
                Err(refused) => {
                    range.allocation = Some(refused.allocation);
                    Err(RangeTransferRefused {
                        range,
                        error: refused.error,
                    })
                }
            }
        }

        #[allow(clippy::result_large_err)]
        pub fn close(
            mut self,
            ledger: &mut Ledger,
        ) -> std::result::Result<(), ArenaCloseRefused<'ctx>> {
            let refuse = |arena, error| ArenaCloseRefused { arena, error };
            if ledger.id() != self.ledger {
                return Err(refuse(
                    self,
                    invalid("reservation", "arena belongs to another ledger"),
                ));
            }
            if let Some(error) = &self.quarantined {
                let error = Error::DeviceLost {
                    device: u32::MAX,
                    detail: format!("arena cleanup was quarantined: {error}"),
                };
                return Err(refuse(self, error));
            }
            if self.metadata.occupancy().live_allocations != 0 {
                let live = self.metadata.occupancy().live_allocations;
                return Err(refuse(
                    self,
                    invalid("arena", format!("{live} allocation(s) remain live")),
                ));
            }
            let core = self.core.take().expect("open arena has a core");
            let mut core = match Rc::try_unwrap(core) {
                Ok(core) => core,
                Err(core) => {
                    self.core = Some(core);
                    return Err(refuse(
                        self,
                        invalid("arena", "a device range still retains the physical arena"),
                    ));
                }
            };
            // SAFETY: no range remains live, so no operation can still name or
            // use a subrange. Close is the sole owner of the physical buffer.
            if let Err(error) = unsafe { core.buffer.try_free() } {
                self.quarantined = Some(error.clone());
                self.core = Some(Rc::new(core));
                return Err(refuse(self, error));
            }
            ledger
                .release(
                    self.reservation
                        .take()
                        .expect("open arena retains its reservation"),
                )
                .expect("ledger identity and outstanding reservation were validated");
            Ok(())
        }
    }

    impl Drop for DeviceArena<'_> {
        fn drop(&mut self) {
            if self.reservation.is_some() {
                // Open or quarantined: keep the physical allocation alive just
                // as the dropped Reservation remains charged in the ledger.
                if let Some(core) = self.core.take() {
                    std::mem::forget(core);
                }
            }
        }
    }

    impl<'ctx> OperationLease<Event<'ctx>, ArenaUpload<'ctx>> {
        pub fn submit(&mut self, stream: &Stream<'ctx>, event: Event<'ctx>) -> Result<(), Error> {
            if self.state() != LeaseState::Live {
                return Err(invalid(
                    "lease",
                    format!("cannot submit while {:?}", self.state()),
                ));
            }
            let upload = self
                .resource
                .as_ref()
                .expect("operation lease retains upload");
            let uuid = upload.range().device_uuid();
            if stream.device_uuid() != uuid || event.device_uuid() != uuid {
                return Err(invalid(
                    "upload",
                    "range, stream and event must belong to one device",
                ));
            }
            let offset = usize::try_from(upload.range().offset()).map_err(|_| {
                invalid(
                    "range",
                    "range offset does not fit the driver address space",
                )
            })?;
            // SAFETY: the operation lease owns both source and range until its
            // recorded event completes or they are quarantined.
            let copied = unsafe {
                upload
                    .range()
                    .core
                    .buffer
                    .copy_from_host_async_at(offset, upload.source(), stream)
            };
            let result = copied.and_then(|()| event.record(stream));
            if let Err(error) = result {
                self.mark_lost(
                    upload.range().core.device_ordinal,
                    format!("arena upload submission failed: {error}"),
                );
                return Err(error);
            }
            self.lifecycle.submit(event);
            Ok(())
        }

        pub fn readback(&mut self, destination: &mut [u8]) -> Result<(), Error> {
            if self.state() != LeaseState::InFlight && self.state() != LeaseState::Cancelled {
                return Err(invalid(
                    "lease",
                    "readback requires a submitted, non-lost arena upload",
                ));
            }
            self.synchronize()?;
            let upload = self
                .resource
                .as_ref()
                .expect("operation lease retains upload");
            if destination.len() > upload.source().len() {
                return Err(invalid("destination", "readback exceeds uploaded bytes"));
            }
            let offset = usize::try_from(upload.range().offset()).map_err(|_| {
                invalid(
                    "range",
                    "range offset does not fit the driver address space",
                )
            })?;
            let result = upload
                .range()
                .core
                .buffer
                .copy_to_host_at(offset, destination);
            if let Err(error) = &result {
                self.lifecycle.persist_loss(error.clone());
            }
            result
        }
    }
}

#[cfg(feature = "driver")]
pub use driver_binding::{
    ArenaCloseRefused, ArenaCreateRefused, ArenaUpload, DeviceArena, DeviceRange,
    PrepareArenaUploadRefused, RangeReleaseRefused, RangeTransferRefused,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lease::{ManualCompletion, Script, ScriptedCompletion};

    #[test]
    fn operation_reuse_waits_for_completion_and_cancel_does_not_release() {
        let completion = ManualCompletion::new();
        let mut lease = OperationLease::new("range", vec![1u8; 4]).unwrap();
        lease.track_manual(completion.clone()).unwrap();
        lease.cancel();
        let refused = lease.retire().unwrap_err();
        assert_eq!(refused.lease.state(), LeaseState::Cancelled);
        completion.complete();
        let (_, resource) = refused.lease.retire().unwrap();
        assert_eq!(resource, vec![1u8; 4]);

        let mut arena = moxie_memory::Arena::new("drop proof", 256, 256).unwrap();
        let allocation = arena.allocate(256, 256, "held operation").unwrap();
        let key = allocation.key();
        let mut lease = OperationLease::new("dropped in flight", allocation).unwrap();
        lease.track_manual(ManualCompletion::new()).unwrap();
        drop(lease);
        assert!(arena.contains(key));
        assert_eq!(
            arena
                .allocate(1, 1, "cannot reuse")
                .unwrap_err()
                .occupancy
                .free_bytes,
            0
        );
    }

    #[test]
    fn operation_loss_persists_past_a_racing_ready_observation() {
        let completion =
            ScriptedCompletion::new([Script::Lost(4, "gone".into()), Script::Ready(true)]);
        let mut lease = OperationLease::new("range", ()).unwrap();
        lease.track_manual(completion).unwrap();
        let refused = lease.retire().unwrap_err();
        assert_eq!(refused.lease.state(), LeaseState::Lost);
        let refused = refused.lease.retire().unwrap_err();
        assert_eq!(refused.error.kind(), "device_lost");
    }

    #[test]
    fn operation_turn_returns_held_then_retires_without_a_next_token() {
        let completion = ManualCompletion::new();
        let mut lease = OperationLease::new("range", 9u8).unwrap();
        lease.track_manual(completion.clone()).unwrap();
        let mut turn = OperationTurn::new("turn one").unwrap();
        turn.hold(lease);
        let first = turn.release_turn();
        assert!(!first.is_clean());
        completion.complete();
        let mut turn = OperationTurn::new("turn two").unwrap();
        turn.hold(first.held.into_iter().next().unwrap().lease);
        let second = turn.release_turn();
        assert!(second.is_clean());
        assert_eq!(second.retired[0].resource, 9);
    }
}
