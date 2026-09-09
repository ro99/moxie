//! Pure metadata for one bounded, aligned allocation arena.
//!
//! The arena owns no pointer and calls no allocator. It is the exact byte and
//! generation authority paired with a physical allocation by `moxie-executor`.
//! Allocation is deterministic address-ordered first fit; release is explicit.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{Error, Result};

/// Process-unique identity of one arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaId(u64);

impl ArenaId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identity assigned once to an allocation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Copyable diagnostic identity. It can inspect, never release, an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationKey {
    pub arena: ArenaId,
    pub allocation: AllocationId,
    pub generation: u64,
}

/// The sole release/transfer authority for one live range.
///
/// Deliberately not `Clone`: release consumes it, so a double release cannot
/// be expressed through the safe API. Dropping it does not release its record.
///
/// ```compile_fail
/// use moxie_memory::Arena;
/// let mut arena = Arena::new("a", 256, 256).unwrap();
/// let allocation = arena.allocate(256, 256, "owner").unwrap();
/// arena.release(allocation).unwrap();
/// arena.release(allocation).unwrap(); // moved into the first release
/// ```
#[derive(Debug)]
#[must_use = "a dropped allocation stays live and visible; release it explicitly"]
pub struct Allocation {
    key: AllocationKey,
    offset: u64,
    bytes: u64,
    reserved_offset: u64,
    reserved_bytes: u64,
    alignment: u64,
    owner: String,
}

impl Allocation {
    pub const fn key(&self) -> AllocationKey {
        self.key
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    bytes: u64,
}

#[derive(Debug, Clone)]
struct LiveRecord {
    generation: u64,
    offset: u64,
    bytes: u64,
    reserved_offset: u64,
    reserved_bytes: u64,
    alignment: u64,
    owner: String,
}

/// Exact current occupancy, including fragmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaOccupancy {
    pub capacity_bytes: u64,
    pub live_bytes: u64,
    pub free_bytes: u64,
    pub largest_free_bytes: u64,
    pub free_ranges: usize,
    pub live_allocations: usize,
}

/// One visible live allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutstandingAllocation {
    pub key: AllocationKey,
    pub owner: String,
    pub offset: u64,
    pub bytes: u64,
    pub reserved_bytes: u64,
    pub alignment: u64,
}

/// Allocation refusal with the fragmentation facts that caused it.
#[derive(Debug, Clone)]
pub struct AllocateRefused {
    pub error: Error,
    pub occupancy: ArenaOccupancy,
}

impl core::fmt::Display for AllocateRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for AllocateRefused {}

/// Failed release, carrying the still-live authority back.
#[derive(Debug)]
#[must_use = "the allocation is still live; correct the arena and retry"]
pub struct ReleaseAllocationRefused {
    pub allocation: Allocation,
    pub error: Error,
}

/// Failed owner transfer, carrying the still-live authority back.
#[derive(Debug)]
#[must_use = "the allocation keeps its prior owner; correct the request and retry"]
pub struct TransferRefused {
    pub allocation: Allocation,
    pub error: Error,
}

/// One bounded metadata arena.
#[derive(Debug)]
pub struct Arena {
    id: ArenaId,
    label: String,
    capacity: u64,
    base_alignment: u64,
    free: Vec<FreeRange>,
    live: BTreeMap<AllocationId, LiveRecord>,
    generations: BTreeMap<u64, u64>,
    next_allocation: u64,
}

impl Arena {
    pub fn new(label: impl Into<String>, capacity: u64, base_alignment: u64) -> Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(invalid("label", "an arena must be named"));
        }
        if capacity == 0
            || !base_alignment.is_power_of_two()
            || !capacity.is_multiple_of(base_alignment)
        {
            return Err(invalid(
                "arena",
                "capacity must be nonzero and a multiple of a power-of-two base alignment",
            ));
        }
        Ok(Self {
            id: ArenaId::next(),
            label,
            capacity,
            base_alignment,
            free: vec![FreeRange {
                offset: 0,
                bytes: capacity,
            }],
            live: BTreeMap::new(),
            generations: BTreeMap::new(),
            next_allocation: 1,
        })
    }

    pub const fn id(&self) -> ArenaId {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    pub const fn base_alignment(&self) -> u64 {
        self.base_alignment
    }

    pub fn occupancy(&self) -> ArenaOccupancy {
        let free_bytes = self.free.iter().map(|r| r.bytes).sum();
        ArenaOccupancy {
            capacity_bytes: self.capacity,
            live_bytes: self.capacity - free_bytes,
            free_bytes,
            largest_free_bytes: self.free.iter().map(|r| r.bytes).max().unwrap_or(0),
            free_ranges: self.free.len(),
            live_allocations: self.live.len(),
        }
    }

    pub fn outstanding(&self) -> Vec<OutstandingAllocation> {
        self.live
            .iter()
            .map(|(id, r)| OutstandingAllocation {
                key: AllocationKey {
                    arena: self.id,
                    allocation: *id,
                    generation: r.generation,
                },
                owner: r.owner.clone(),
                offset: r.offset,
                bytes: r.bytes,
                reserved_bytes: r.reserved_bytes,
                alignment: r.alignment,
            })
            .collect()
    }

    pub fn contains(&self, key: AllocationKey) -> bool {
        key.arena == self.id
            && self
                .live
                .get(&key.allocation)
                .is_some_and(|r| r.generation == key.generation)
    }

    pub fn allocate(
        &mut self,
        bytes: u64,
        alignment: u64,
        owner: impl Into<String>,
    ) -> std::result::Result<Allocation, AllocateRefused> {
        let owner = owner.into();
        if bytes == 0 {
            return Err(self.refusal(invalid("bytes", "an allocation must contain bytes")));
        }
        if owner.is_empty() {
            return Err(self.refusal(invalid("owner", "an allocation must name its owner")));
        }
        if !alignment.is_power_of_two() || alignment > self.base_alignment {
            return Err(self.refusal(invalid(
                "alignment",
                "alignment must be a nonzero power of two no greater than the arena base alignment",
            )));
        }

        let mut selected = None;
        for (index, range) in self.free.iter().copied().enumerate() {
            let Some(aligned) = align_up(range.offset, alignment) else {
                return Err(self.refusal(invalid("bytes", "alignment arithmetic overflowed")));
            };
            let padding = aligned - range.offset;
            let Some(reserved) = padding.checked_add(bytes) else {
                return Err(self.refusal(invalid("bytes", "allocation size overflowed")));
            };
            if reserved <= range.bytes {
                selected = Some((index, range, aligned, reserved));
                break;
            }
        }
        let Some((index, range, offset, reserved_bytes)) = selected else {
            let occupancy = self.occupancy();
            return Err(AllocateRefused {
                error: Error::CapacityExceeded {
                    tier: None,
                    requested_bytes: bytes,
                    available_bytes: occupancy.largest_free_bytes,
                },
                occupancy,
            });
        };

        // Compute every fallible identity change before mutating the free list:
        // a refused allocation is byte-for-byte atomic.
        let generation = self
            .generations
            .get(&offset)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                self.refusal(invalid("generation", "allocation generation overflowed"))
            })?;
        let next_allocation = self
            .next_allocation
            .checked_add(1)
            .ok_or_else(|| self.refusal(invalid("allocation", "allocation identity overflowed")))?;

        let remaining = range.bytes - reserved_bytes;
        if remaining == 0 {
            self.free.remove(index);
        } else {
            self.free[index] = FreeRange {
                offset: range.offset + reserved_bytes,
                bytes: remaining,
            };
        }
        self.generations.insert(offset, generation);
        let id = AllocationId(self.next_allocation);
        self.next_allocation = next_allocation;
        let record = LiveRecord {
            generation,
            offset,
            bytes,
            reserved_offset: range.offset,
            reserved_bytes,
            alignment,
            owner: owner.clone(),
        };
        self.live.insert(id, record);
        Ok(Allocation {
            key: AllocationKey {
                arena: self.id,
                allocation: id,
                generation,
            },
            offset,
            bytes,
            reserved_offset: range.offset,
            reserved_bytes,
            alignment,
            owner,
        })
    }

    fn refusal(&self, error: Error) -> AllocateRefused {
        AllocateRefused {
            error,
            occupancy: self.occupancy(),
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn transfer(
        &mut self,
        mut allocation: Allocation,
        owner: impl Into<String>,
    ) -> std::result::Result<Allocation, TransferRefused> {
        let owner = owner.into();
        let error = if owner.is_empty() {
            Some(invalid("owner", "an allocation must name its owner"))
        } else {
            self.validate(&allocation).err()
        };
        if let Some(error) = error {
            return Err(TransferRefused { allocation, error });
        }
        self.live
            .get_mut(&allocation.key.allocation)
            .expect("validated allocation is live")
            .owner
            .clone_from(&owner);
        allocation.owner = owner;
        Ok(allocation)
    }

    #[allow(clippy::result_large_err)]
    pub fn release(
        &mut self,
        allocation: Allocation,
    ) -> std::result::Result<(), ReleaseAllocationRefused> {
        if let Err(error) = self.validate(&allocation) {
            return Err(ReleaseAllocationRefused { allocation, error });
        }
        self.live.remove(&allocation.key.allocation);
        self.insert_free(FreeRange {
            offset: allocation.reserved_offset,
            bytes: allocation.reserved_bytes,
        });
        Ok(())
    }

    fn validate(&self, allocation: &Allocation) -> Result<()> {
        if allocation.key.arena != self.id {
            return Err(invalid("allocation", "allocation belongs to another arena"));
        }
        let Some(record) = self.live.get(&allocation.key.allocation) else {
            return Err(invalid(
                "allocation",
                "allocation is not live in this arena",
            ));
        };
        if record.generation != allocation.key.generation
            || record.offset != allocation.offset
            || record.bytes != allocation.bytes
            || record.reserved_offset != allocation.reserved_offset
            || record.reserved_bytes != allocation.reserved_bytes
            || record.alignment != allocation.alignment
            || record.owner != allocation.owner
        {
            return Err(invalid(
                "allocation",
                "allocation identity, generation, bounds or owner is stale",
            ));
        }
        Ok(())
    }

    fn insert_free(&mut self, range: FreeRange) {
        let mut index = self.free.partition_point(|r| r.offset < range.offset);
        self.free.insert(index, range);
        if index > 0 {
            let previous = self.free[index - 1];
            let current = self.free[index];
            if previous.offset + previous.bytes == current.offset {
                self.free[index - 1].bytes += current.bytes;
                self.free.remove(index);
                index -= 1;
            }
        }
        if index + 1 < self.free.len() {
            let current = self.free[index];
            let next = self.free[index + 1];
            if current.offset + current.bytes == next.offset {
                self.free[index].bytes += next.bytes;
                self.free.remove(index + 1);
            }
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fit_and_generation_on_reuse() {
        let mut arena = Arena::new("weights", 1024, 256).unwrap();
        let first = arena.allocate(1024, 256, "loader").unwrap();
        let old = first.key();
        assert_eq!(arena.occupancy().free_bytes, 0);
        arena.release(first).unwrap();
        let second = arena.allocate(1024, 256, "executor").unwrap();
        assert_eq!(second.offset(), 0);
        assert!(second.key().generation > old.generation);
        assert!(!arena.contains(old));
    }

    #[test]
    fn padding_is_charged_and_release_coalesces_both_sides() {
        let mut arena = Arena::new("scratch", 1024, 256).unwrap();
        let a = arena.allocate(3, 1, "a").unwrap();
        let b = arena.allocate(8, 8, "b").unwrap();
        let c = arena.allocate(5, 1, "c").unwrap();
        assert_eq!(b.offset(), 8);
        assert_eq!(b.reserved_bytes(), 13);
        assert_eq!(arena.occupancy().live_bytes, 21);
        arena.release(b).unwrap();
        arena.release(a).unwrap();
        arena.release(c).unwrap();
        assert_eq!(
            arena.occupancy(),
            ArenaOccupancy {
                capacity_bytes: 1024,
                live_bytes: 0,
                free_bytes: 1024,
                largest_free_bytes: 1024,
                free_ranges: 1,
                live_allocations: 0,
            }
        );
    }

    #[test]
    fn fragmentation_is_distinct_from_total_free() {
        let mut arena = Arena::new("fragmented", 1024, 256).unwrap();
        let a = arena.allocate(256, 256, "a").unwrap();
        let b = arena.allocate(256, 256, "b").unwrap();
        let c = arena.allocate(256, 256, "c").unwrap();
        let d = arena.allocate(256, 256, "d").unwrap();
        arena.release(a).unwrap();
        arena.release(c).unwrap();
        let refused = arena.allocate(512, 256, "large").unwrap_err();
        assert_eq!(refused.occupancy.free_bytes, 512);
        assert_eq!(refused.occupancy.largest_free_bytes, 256);
        arena.release(b).unwrap();
        arena.release(d).unwrap();
        assert_eq!(arena.occupancy().largest_free_bytes, 1024);

        // A live separator prevents merging left, but must not prevent merging
        // the newly returned range with its free right neighbor.
        let a = arena.allocate(256, 256, "a").unwrap();
        let b = arena.allocate(256, 256, "separator").unwrap();
        let c = arena.allocate(256, 256, "c").unwrap();
        let d = arena.allocate(256, 256, "d").unwrap();
        arena.release(a).unwrap();
        arena.release(d).unwrap();
        arena.release(c).unwrap();
        assert_eq!(arena.occupancy().largest_free_bytes, 512);
        assert_eq!(arena.occupancy().free_ranges, 2);
        arena.release(b).unwrap();
        assert_eq!(arena.occupancy().largest_free_bytes, 1024);
    }

    #[test]
    fn drop_stays_visible_and_transfer_changes_only_owner() {
        let mut arena = Arena::new("persistent", 1024, 256).unwrap();
        let allocation = arena.allocate(256, 256, "importer").unwrap();
        let key = allocation.key();
        let before = arena.occupancy();
        let allocation = arena.transfer(allocation, "executor").unwrap();
        assert_eq!(allocation.key(), key);
        assert_eq!(arena.occupancy(), before);
        assert_eq!(arena.outstanding()[0].owner, "executor");
        drop(allocation);
        assert!(arena.contains(key));
        assert_eq!(arena.outstanding()[0].owner, "executor");
    }

    #[test]
    fn invalid_requests_are_atomic() {
        assert!(Arena::new("", 256, 256).is_err());
        assert!(Arena::new("x", 255, 256).is_err());
        let mut arena = Arena::new("x", 256, 256).unwrap();
        let before = arena.occupancy();
        for result in [
            arena.allocate(0, 1, "x"),
            arena.allocate(1, 3, "x"),
            arena.allocate(1, 512, "x"),
            arena.allocate(1, 1, ""),
        ] {
            assert!(result.is_err());
            assert_eq!(arena.occupancy(), before);
        }
        assert!(arena.allocate(u64::MAX, 256, "overflow").is_err());
        assert_eq!(arena.occupancy(), before);

        let mut huge = Arena::new("checked", u64::MAX - 1, 2).unwrap();
        let _prefix = huge.allocate(1, 1, "prefix").unwrap();
        let before = huge.occupancy();
        assert_eq!(
            huge.allocate(u64::MAX, 2, "overflow")
                .unwrap_err()
                .error
                .kind(),
            "invalid_request"
        );
        assert_eq!(huge.occupancy(), before);
    }

    #[test]
    fn foreign_release_and_empty_transfer_return_the_handle() {
        let mut left = Arena::new("left", 256, 256).unwrap();
        let mut right = Arena::new("right", 256, 256).unwrap();
        let allocation = left.allocate(256, 256, "owner").unwrap();
        let refused = right.release(allocation).unwrap_err();
        assert_eq!(right.occupancy().live_bytes, 0);
        let refused = left.transfer(refused.allocation, "").unwrap_err();
        assert_eq!(left.outstanding()[0].owner, "owner");
        left.release(refused.allocation).unwrap();
        assert_eq!(left.occupancy().free_bytes, 256);
    }
}
