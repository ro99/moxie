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

    /// One physical allocation with separately charged state and CPU workspace.
    /// The workspace is the suffix after `state_bytes`; no raw pointer escapes.
    pub fn allocate_with_workspace(
        ledger: &mut Ledger,
        label: &str,
        state_bytes: usize,
        workspace_bytes: usize,
        control_bytes: usize,
    ) -> Result<Self> {
        let data_bytes = state_bytes
            .checked_add(workspace_bytes)
            .ok_or(moxie_types::DimError::Overflow)?;
        let mut plan = PlanRequest::new(label, ["live"])?;
        for (name, tier, bytes) in [
            ("host backing", HostTier::StateSpill, state_bytes),
            ("cpu workspace", HostTier::CpuWorkspace, workspace_bytes),
            ("bounded control", HostTier::Pageable, control_bytes),
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
            // Vocabulary workspace does not shrink with requested context.
            plan.buffer(if tier == HostTier::CpuWorkspace {
                request
            } else {
                request.scaling(Scaling::Context)
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
                    (true, false) => Some(Tier::Host(HostTier::StateSpill)),
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
