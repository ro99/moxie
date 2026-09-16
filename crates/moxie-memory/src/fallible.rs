//! Growth that refuses instead of aborting.
//!
//! `format!`, `vec!`, `to_string`, `collect` and `BTreeMap::insert` all call
//! `handle_alloc_error` when the allocator returns null, and that **aborts the
//! process**. Admission is exactly the wrong place for it: a ledger exists to
//! say "this does not fit", and it may not answer that question by dying.
//!
//! Independent review reached the abort through `AffineLinearRun::admit` and
//! then walked the call graph: `PlanRequest::new`, `Ledger::admit`,
//! `Arena::new`, `Arena::allocate` and `Module::function` all grew infallibly,
//! two of them **after** a device allocation had succeeded or a free list had
//! been mutated ([task 0029](../../../docs/tasks/0029-allocation-fallible-admission-vocabulary.md)).
//!
//! What this module provides is the small vocabulary that path needs. It is
//! deliberately not a general-purpose allocator wrapper: every function here
//! reserves before it fills, and every one of them returns
//! [`moxie_types::Error::CapacityExceeded`], which is the answer a ledger was
//! already able to give.

use moxie_types::{Error, Result};

/// The refusal every function here returns.
///
/// This vocabulary allocates host metadata, attributed to pageable host memory.
/// Zero sizes mean the host allocator did not report its available capacity;
/// `tier: None` remains reserved for unattributed driver allocation failures.
pub fn no_room() -> Error {
    Error::CapacityExceeded {
        tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
        requested_bytes: 0,
        available_bytes: 0,
    }
}

/// A `String` that grows only through `try_reserve`.
struct FallibleString(String);

impl core::fmt::Write for FallibleString {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        // Reserve first, then push: `push_str` cannot reallocate once the
        // capacity is there, so no infallible growth happens on this path.
        self.0.try_reserve(s.len()).map_err(|_| core::fmt::Error)?;
        self.0.push_str(s);
        Ok(())
    }
}

/// `format!` that refuses.
///
/// Used for refusal prose, which is the paradoxical case: the message that
/// says an allocation failed must not need one.
pub fn text(args: core::fmt::Arguments<'_>) -> Result<String> {
    use core::fmt::Write;
    let mut sink = FallibleString(String::new());
    sink.write_fmt(args).map(|()| sink.0).map_err(|_| no_room())
}

/// `Vec::push` that refuses.
///
/// Reserves one element first. A `Vec` with spare capacity cannot reallocate
/// on `push`, so this is a push that has already asked.
pub fn push<T>(vec: &mut Vec<T>, value: T) -> Result<()> {
    vec.try_reserve(1).map_err(|_| no_room())?;
    vec.push(value);
    Ok(())
}

/// `Vec::with_capacity` that refuses.
pub fn with_capacity<T>(capacity: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity).map_err(|_| no_room())?;
    Ok(out)
}

/// A copy of a label that refuses.
///
/// **`Cow::clone` is not free.** A borrowed label clones a pointer, but an
/// owned one copies its bytes through `handle_alloc_error` -- and the labels
/// that reach an arena from `AffineLinearRun::admit` are owned, built by its
/// own fallible formatter. Independent review found `Arena::allocate` cloning
/// one *after* the free list had moved, which is the abort this task exists to
/// remove wearing the type that was supposed to remove it.
pub fn clone_label(source: &crate::request::Label) -> Result<crate::request::Label> {
    Ok(match source {
        std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(s),
        std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(string(s)?),
    })
}

/// A copy of `source` that refuses.
pub fn string(source: &str) -> Result<String> {
    let mut out = String::new();
    out.try_reserve_exact(source.len()).map_err(|_| no_room())?;
    out.push_str(source);
    Ok(out)
}

/// A sorted map whose inserts can **refuse**.
///
/// `BTreeMap::insert` allocates a node through `handle_alloc_error`, so every
/// map on the admission path was an abort waiting for memory pressure — and one
/// of them sat between charging a ledger counter and recording what had been
/// charged. There is no fallible `BTreeMap`, so this is the smallest thing that
/// is one: a `Vec` of pairs kept in key order, searched by binary search.
///
/// The sizes are what make that the right trade. A scope commits a handful of
/// tiers, a ledger holds a handful of outstanding reservations, and an arena
/// holds one record per live range. Linear insertion cost over a few dozen
/// elements is nothing next to an allocation, and the memory is one contiguous
/// block rather than a node each.
///
/// [`Map::try_reserve_one`] and [`Map::insert`] are separate on purpose: a
/// caller that must not fail partway can take the room it needs **first**, do
/// the work that cannot fail, and insert last.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Map<K, V> {
    entries: Vec<(K, V)>,
}

impl<K: Ord, V> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> Map<K, V> {
    pub const fn new() -> Self {
        Map {
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn position(&self, key: &K) -> std::result::Result<usize, usize> {
        self.entries.binary_search_by(|(k, _)| k.cmp(key))
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.position(key).ok().map(|at| &self.entries[at].1)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.position(key) {
            Ok(at) => Some(&mut self.entries[at].1),
            Err(_) => None,
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.position(key).is_ok()
    }

    /// Room for one more entry, taken before anything irreversible happens.
    pub fn try_reserve_one(&mut self) -> Result<()> {
        self.try_reserve(1)
    }

    /// Room for `additional` more entries.
    ///
    /// Taking the room for a whole batch is what makes a batch atomic:
    /// inserting entries one reserved-at-a-time leaves the earlier ones
    /// installed when a later reservation fails, which is the same
    /// half-changed state one level down.
    pub fn try_reserve(&mut self, additional: usize) -> Result<()> {
        self.entries.try_reserve(additional).map_err(|_| no_room())
    }

    /// Insert into capacity that is already there.
    ///
    /// # Panics
    /// If the map is full. Call [`Map::try_reserve_one`] or
    /// [`Map::try_insert`]; this exists for the caller that reserved first
    /// precisely so that its own insert cannot fail.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.position(&key) {
            Ok(at) => Some(core::mem::replace(&mut self.entries[at].1, value)),
            Err(at) => {
                assert!(
                    self.entries.len() < self.entries.capacity(),
                    "insert without reserved capacity"
                );
                self.entries.insert(at, (key, value));
                None
            }
        }
    }

    /// Reserve and insert.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        self.try_reserve_one()?;
        Ok(self.insert(key, value))
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.position(key) {
            Ok(at) => Some(self.entries.remove(at).1),
            Err(_) => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&(K, V)) -> (&K, &V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}
