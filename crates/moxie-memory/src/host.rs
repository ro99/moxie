//! Synchronous host storage owned by the resource authority.
//!
//! No raw pointer or asynchronous lease escapes this buffer. It must be released
//! explicitly; dropping it frees the host bytes but leaves the charge visible.

use moxie_types::{Error, HostTier, Result, Scope, Tier};

use crate::{BufferRequest, Ledger, PlanRequest, Reservation, Scaling, StageSpan};

/// One admitted host allocation plus a reserve for its consumer's bounded
/// control metadata. Neither the allocation nor its reservation is cloneable.
#[derive(Debug)]
pub struct HostBuffer {
    data: Vec<u8>,
    reservation: Option<Reservation>,
}

impl HostBuffer {
    /// Admit the complete overlapping envelope before allocating or touching
    /// payload bytes. `control_bytes` covers metadata allocated by the consumer;
    /// the caller must bound that metadata independently.
    pub fn allocate(
        ledger: &mut Ledger,
        label: &str,
        data_bytes: usize,
        control_bytes: usize,
    ) -> Result<Self> {
        Self::allocate_with_workspace(ledger, label, data_bytes, 0, control_bytes)
    }

    /// The same buffer charged to a caller-named host tier.
    ///
    /// Document 03 tracks host bytes by what they *are*, not by who allocated
    /// them, and a resident weight cache is not spilled sequence state. Task
    /// 0020 needs `HostTier::Pageable`; the two constructors above keep
    /// `StateSpill`, which is what their callers hold.
    pub fn allocate_in(
        ledger: &mut Ledger,
        label: &str,
        data_tier: HostTier,
        data_bytes: usize,
        control_bytes: usize,
    ) -> Result<Self> {
        Self::allocate_tiered(ledger, label, data_tier, data_bytes, 0, control_bytes)
    }

    /// One physical allocation with separately charged state and CPU workspace.
    /// The workspace is the suffix after `state_bytes`; no raw pointer escapes.
    pub fn allocate_with_workspace(
        ledger: &mut Ledger,
        label: &str,
        state_bytes: usize,
        workspace_bytes: usize,
        control_bytes: usize,
    ) -> Result<Self> {
        Self::allocate_tiered(
            ledger,
            label,
            HostTier::StateSpill,
            state_bytes,
            workspace_bytes,
            control_bytes,
        )
    }

    fn allocate_tiered(
        ledger: &mut Ledger,
        label: &str,
        data_tier: HostTier,
        state_bytes: usize,
        workspace_bytes: usize,
        control_bytes: usize,
    ) -> Result<Self> {
        let data_bytes = state_bytes
            .checked_add(workspace_bytes)
            .ok_or(moxie_types::DimError::Overflow)?;
        let mut plan = PlanRequest::new(label, ["live"])?;
        // `Scaling::Context` is what licenses a refusal to suggest
        // `LowerContext`, so it is declared per row rather than per tier: a
        // vocabulary workspace does not shrink with requested context, and
        // neither does a resident weight cache. Suggesting a shorter context
        // for bytes a shorter context would not free is a suggestion that
        // cannot help, which is the one thing `Scaling` exists to prevent.
        let backing_scaling = (data_tier == HostTier::StateSpill).then_some(Scaling::Context);
        for (name, tier, bytes, scaling) in [
            ("host backing", data_tier, state_bytes, backing_scaling),
            (
                "cpu workspace",
                HostTier::CpuWorkspace,
                workspace_bytes,
                None,
            ),
            (
                "bounded control",
                HostTier::Pageable,
                control_bytes,
                Some(Scaling::Context),
            ),
        ] {
            if bytes == 0 {
                continue;
            }
            let request = BufferRequest::new(
                name,
                Scope::Host,
                Tier::Host(tier),
                u64::try_from(bytes).map_err(|_| moxie_types::DimError::Overflow)?,
                StageSpan::at(0),
            );
            plan.buffer(match scaling {
                Some(s) => request.scaling(s),
                None => request,
            })?;
        }
        let reservation = ledger.admit(&plan).map_err(Error::from)?;
        let mut data = Vec::new();
        if data.try_reserve_exact(data_bytes).is_err() || data.capacity() != data_bytes {
            // An allocator may overallocate. Do not retain unadmitted capacity.
            drop(data);
            ledger.release(reservation).expect("the admitting ledger");
            return Err(Error::CapacityExceeded {
                tier: match (state_bytes != 0, workspace_bytes != 0) {
                    (true, false) => Some(Tier::Host(data_tier)),
                    (false, true) => Some(Tier::Host(HostTier::CpuWorkspace)),
                    _ => None,
                },
                requested_bytes: data_bytes as u64,
                available_bytes: 0,
            });
        }
        data.resize(data_bytes, 0);
        Ok(Self {
            data,
            reservation: Some(reservation),
        })
    }

    /// Abandon the physical bytes **without freeing them**.
    ///
    /// For the one case document 02 describes: "Retirement is event-driven;
    /// Rust `Drop` alone must not free in-flight CUDA memory", and "buffer reuse
    /// waits for all dependent streams/ranks, including cancelled work". A
    /// device-to-host copy reads these bytes by address; returning them to the
    /// allocator while that copy may still be running is a use-after-free the
    /// allocator will happily hand to somebody else.
    ///
    /// So the pages stay mapped and the charge stays visible in the ledger,
    /// which is the failure mode that can be *found*. `moxie-executor` makes the
    /// same trade for a lost context: withhold forever rather than advertise
    /// memory nothing can recover.
    pub fn withhold(&mut self) {
        core::mem::forget(core::mem::take(&mut self.data));
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Refusal preserves both the bytes and the only release authority.
    /// Success leaves this buffer empty and releases its charge immediately.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        let reservation = self.reservation.as_ref().ok_or(Error::InvalidRequest {
            field: "host_buffer",
            detail: "already released".into(),
        })?;
        if reservation.ledger() != ledger.id() {
            return Err(Error::InvalidRequest {
                field: "ledger",
                detail: "host buffer belongs to another ledger".into(),
            });
        }
        // Free physical storage before making the charge available for reuse.
        self.data = Vec::new();
        ledger
            .release(self.reservation.take().expect("checked above"))
            .expect("the sole reservation against its admitting ledger");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapacitySnapshot;

    fn ledger(bytes: u64) -> Ledger {
        Ledger::new([CapacitySnapshot::new(Scope::Host, bytes + 1, 1).unwrap()]).unwrap()
    }

    #[test]
    fn workspace_is_charged_separately_and_its_allocation_failure_names_its_tier() {
        let mut owner = ledger(256);
        let mut buffer =
            HostBuffer::allocate_with_workspace(&mut owner, "state and probabilities", 80, 40, 16)
                .unwrap();
        assert_eq!(buffer.bytes().len(), 120);
        assert_eq!(
            owner.committed(Scope::Host, Tier::Host(HostTier::StateSpill)),
            80
        );
        assert_eq!(
            owner.committed(Scope::Host, Tier::Host(HostTier::CpuWorkspace)),
            40
        );
        assert_eq!(
            owner.committed(Scope::Host, Tier::Host(HostTier::Pageable)),
            16
        );
        buffer.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
        let mut owner = ledger(u64::MAX - 1);
        let bytes = isize::MAX as usize + 1;
        assert_eq!(
            HostBuffer::allocate_with_workspace(&mut owner, "workspace only", 0, bytes, 0)
                .unwrap_err(),
            Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::CpuWorkspace)),
                requested_bytes: bytes as u64,
                available_bytes: 0
            }
        );
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn admission_release_and_foreign_ledger_preserve_authority() {
        let mut owner = ledger(100);
        let mut other = ledger(100);
        let mut buffer = HostBuffer::allocate(&mut owner, "test", 80, 20).unwrap();
        buffer.bytes_mut()[0] = 7;
        assert_eq!(owner.scope_committed(Scope::Host), 100);
        assert!(buffer.release(&mut other).is_err());
        assert_eq!(buffer.bytes()[0], 7);
        assert!(HostBuffer::allocate(&mut owner, "too much", 1, 0).is_err());
        buffer.release(&mut owner).unwrap();
        assert_eq!(owner.scope_committed(Scope::Host), 0);
        assert!(buffer.release(&mut owner).is_err());
    }

    #[test]
    fn allocator_failure_after_admission_unwinds_and_drop_stays_visible() {
        let mut owner = ledger(u64::MAX - 1);
        // Vec cannot address this extent; no operating-system OOM is induced.
        assert!(
            HostBuffer::allocate(&mut owner, "impossible", isize::MAX as usize + 1, 0).is_err()
        );
        assert!(owner.outstanding().is_empty());
        drop(HostBuffer::allocate(&mut owner, "forgotten", 8, 4).unwrap());
        assert_eq!(owner.scope_committed(Scope::Host), 12);
        assert_eq!(owner.outstanding().len(), 1);
    }
}
