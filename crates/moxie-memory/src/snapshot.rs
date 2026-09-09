//! What a scope physically has, as measured somewhere else.
//!
//! The ledger never probes. Document 03 requires host admission to reserve
//! operating-system and application headroom "from measured available memory";
//! measuring is the job of whoever owns the device context or reads
//! `/proc/meminfo`, and this crate's job is to refuse to spend what the
//! measurement did not offer.

use std::collections::BTreeMap;

use moxie_types::{Error, MeasuredDevice, Result, Scope, Tier};

/// One scope's physical capacity and the headroom that must stay unspent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacitySnapshot {
    scope: Scope,
    physical_bytes: u64,
    system_headroom_bytes: u64,
    per_tier_cap: BTreeMap<Tier, u64>,
}

impl CapacitySnapshot {
    /// A snapshot for `scope`, reserving `system_headroom_bytes` that admission
    /// may never touch.
    ///
    /// A **host** snapshot must reserve something. Document 03: host admission
    /// "must not treat all 251 GB as an expert cache", and the only way to
    /// enforce that here rather than hope for it is to refuse a zero. A device
    /// snapshot may reserve zero, because a device's own safety headroom is a
    /// declared tier (`DeviceTier::SafetyHeadroom`) rather than a subtraction.
    pub fn new(scope: Scope, physical_bytes: u64, system_headroom_bytes: u64) -> Result<Self> {
        if system_headroom_bytes > physical_bytes {
            return Err(Error::InvalidRequest {
                field: "system_headroom_bytes",
                detail: format!(
                    "{scope}: headroom {system_headroom_bytes} B exceeds physical {physical_bytes} B"
                ),
            });
        }
        if scope == Scope::Host && system_headroom_bytes == 0 {
            return Err(Error::InvalidRequest {
                field: "system_headroom_bytes",
                detail: "host admission must reserve operating-system and application headroom \
                         from measured available memory (document 03); zero is not a measurement"
                    .into(),
            });
        }
        Ok(CapacitySnapshot {
            scope,
            physical_bytes,
            system_headroom_bytes,
            per_tier_cap: BTreeMap::new(),
        })
    }

    /// A snapshot for one device, from a reading the driver gave.
    ///
    /// ```text
    /// physical_bytes  = total
    /// system_headroom = (total - free) + extra_reserve
    /// admissible      = free - extra_reserve
    /// ```
    ///
    /// The headroom carries two different things on purpose. `total - free` is
    /// what is *already gone* on this card -- this process's own context, and
    /// anything another process holds. `extra_reserve_bytes` is what this engine
    /// chooses to leave alone on top of that. Both must stay unspent, and only
    /// the second is a decision.
    ///
    /// The result inherits the reading's nature: it is capacity at an instant,
    /// not a promise. Another process can take memory a moment later, and this
    /// snapshot will not know. Document 03's answer to that is a replan on a
    /// memory-pressure event, which is M2's; here it is a documented limit.
    pub fn measured(measurement: &MeasuredDevice, extra_reserve_bytes: u64) -> Result<Self> {
        let bad = |detail: String| Error::InvalidRequest {
            field: "measurement",
            detail,
        };
        let taken = measurement
            .total_bytes
            .checked_sub(measurement.free_bytes)
            .ok_or_else(|| {
                bad(format!(
                    "{}: {} B free of {} B total",
                    measurement.uuid, measurement.free_bytes, measurement.total_bytes
                ))
            })?;
        if extra_reserve_bytes > measurement.free_bytes {
            return Err(bad(format!(
                "{}: a {extra_reserve_bytes} B reserve does not fit {} B of free memory",
                measurement.uuid, measurement.free_bytes
            )));
        }
        let headroom = taken
            .checked_add(extra_reserve_bytes)
            .ok_or_else(|| bad(format!("{}: headroom overflows", measurement.uuid)))?;
        CapacitySnapshot::new(
            Scope::Device(measurement.uuid),
            measurement.total_bytes,
            headroom,
        )
    }

    /// Cap one tier below what the scope alone would allow. Absent means the
    /// tier is bounded only by the scope.
    pub fn with_tier_cap(mut self, tier: Tier, cap_bytes: u64) -> Result<Self> {
        if tier.scope_kind() != self.scope.kind() {
            return Err(Error::InvalidRequest {
                field: "tier",
                detail: format!("{} is not a tier of {}", tier.name(), self.scope),
            });
        }
        self.per_tier_cap.insert(tier, cap_bytes);
        Ok(self)
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    pub fn system_headroom_bytes(&self) -> u64 {
        self.system_headroom_bytes
    }

    /// What admission may spend: physical less the reserved headroom.
    pub fn admissible_bytes(&self) -> u64 {
        self.physical_bytes - self.system_headroom_bytes
    }

    pub fn tier_cap(&self, tier: Tier) -> Option<u64> {
        self.per_tier_cap.get(&tier).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_types::{DeviceTier, DeviceUuid, HostTier};

    fn device() -> Scope {
        Scope::Device(DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap())
    }

    #[test]
    fn a_host_snapshot_with_no_headroom_is_refused() {
        let e = CapacitySnapshot::new(Scope::Host, 251 << 30, 0).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert!(e.to_string().contains("headroom"), "{e}");
        // A device may declare zero: its headroom is a tier, not a subtraction.
        assert!(CapacitySnapshot::new(device(), 24 << 30, 0).is_ok());
    }

    #[test]
    fn headroom_cannot_exceed_the_measurement() {
        let e = CapacitySnapshot::new(device(), 1000, 1001).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert_eq!(
            CapacitySnapshot::new(device(), 1000, 400)
                .unwrap()
                .admissible_bytes(),
            600
        );
    }

    fn measurement(total: u64, free: u64) -> MeasuredDevice {
        MeasuredDevice {
            uuid: DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap(),
            ordinal_label: 2,
            name: "test device".into(),
            sm: "sm_86".into(),
            pci_bus_id: "0000:83:00.0".into(),
            multiprocessor_count: 82,
            total_bytes: total,
            free_bytes: free,
        }
    }

    #[test]
    fn a_measured_snapshot_spends_free_memory_and_reserves_the_rest() {
        let s = CapacitySnapshot::measured(&measurement(24_000, 20_000), 1_000).unwrap();
        assert_eq!(s.physical_bytes(), 24_000);
        // 4,000 B already gone on the card, plus the 1,000 B this engine leaves.
        assert_eq!(s.system_headroom_bytes(), 5_000);
        assert_eq!(s.admissible_bytes(), 19_000);
        // Identity is the UUID; the ordinal the reading came through is not in
        // the snapshot at all.
        assert_eq!(s.scope(), Scope::Device(measurement(1, 1).uuid));
    }

    #[test]
    fn a_reserve_larger_than_free_memory_is_refused() {
        let e = CapacitySnapshot::measured(&measurement(24_000, 1_000), 2_000).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        // Exactly all of it is legal, and leaves nothing to admit.
        let s = CapacitySnapshot::measured(&measurement(24_000, 1_000), 1_000).unwrap();
        assert_eq!(s.admissible_bytes(), 0);
    }

    #[test]
    fn a_driver_reporting_more_free_than_total_is_an_error_not_a_wrap() {
        let e = CapacitySnapshot::measured(&measurement(1_000, 2_000), 0).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
    }

    #[test]
    fn a_fully_occupied_device_measures_as_zero_rather_than_failing() {
        // A card with nothing free is a legal snapshot that admits nothing. It
        // is not an error: refusing to describe it would leave the ledger
        // unable to say *why* a plan does not fit.
        let s = CapacitySnapshot::measured(&measurement(24_000, 0), 0).unwrap();
        assert_eq!(s.admissible_bytes(), 0);
        assert_eq!(s.system_headroom_bytes(), 24_000);
    }

    #[test]
    fn a_tier_cap_must_belong_to_the_scope() {
        let e = CapacitySnapshot::new(device(), 1000, 0)
            .unwrap()
            .with_tier_cap(Tier::Host(HostTier::Pinned), 10)
            .unwrap_err();
        assert_eq!(e.kind(), "invalid_request");

        let ok = CapacitySnapshot::new(device(), 1000, 0)
            .unwrap()
            .with_tier_cap(Tier::Device(DeviceTier::ExpertCache), 10)
            .unwrap();
        assert_eq!(ok.tier_cap(Tier::Device(DeviceTier::ExpertCache)), Some(10));
        assert_eq!(ok.tier_cap(Tier::Device(DeviceTier::Logits)), None);
    }
}
