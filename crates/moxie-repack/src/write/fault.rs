//! Named failure points, so every durable boundary can be failed on purpose.
//!
//! Task 0025's acceptance asks for failures injected "at every allocation and
//! durable I/O boundary, including after publication". A test cannot fill a
//! disk or crash a syscall from outside, so the boundaries name themselves and
//! a run can be told to fail at the nth visit to one.
//!
//! This is a **product** facility rather than a `cfg(test)` one, deliberately:
//! the CLI exposes it behind explicitly test-named flags, which is what lets
//! the acceptance gates drive the real binary rather than a harness that
//! reimplements its workflow. A plan with no sites configured costs one
//! comparison against an empty slice per boundary.

use moxie_types::{Error, Result};

/// Every boundary a run can be failed at.
///
/// The list is the state machine's durable surface: if a new syscall is added
/// to the publication path and does not appear here, the enumeration test that
/// walks `Site::ALL` will not cover it, which is the point of naming them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// Creating the destination directory.
    DestinationCreate,
    /// Taking the exclusive run lock.
    LockAcquire,
    /// Creating the journal file.
    JournalCreate,
    /// Appending a journal record.
    JournalAppend,
    /// Syncing a journal record.
    JournalSync,
    /// Creating a shard file, in the pass that writes its header.
    ShardCreate,
    /// Writing a shard's safetensors header.
    ShardHeaderWrite,
    /// Syncing a shard's safetensors header.
    ShardHeaderSync,
    /// Opening a shard file to write a payload unit into it.
    ChunkCreate,
    /// Writing payload bytes.
    ChunkWrite,
    /// Syncing a shard file after a payload unit.
    ChunkSync,
    /// Reading staged bytes back, on resume or for a rehash.
    ChunkReadBack,
    /// Writing the staged manifest.
    ManifestWrite,
    /// Syncing the staged manifest.
    ManifestSync,
    /// Validating the staged artifact through the production reader.
    Validate,
    /// The publication rename itself.
    Publish,
    /// The durability confirmation **after** the publication rename. Failing
    /// here is the "published, durability unconfirmed" outcome, which is a
    /// different thing from a failed publish.
    PublishDurability,
    /// Syncing the destination directory.
    DirectorySync,
    /// Removing the journal after a successful publish.
    JournalRemove,
    /// An admitted allocation.
    Allocate,
}

impl Site {
    pub const ALL: &'static [Site] = &[
        Site::DestinationCreate,
        Site::LockAcquire,
        Site::JournalCreate,
        Site::JournalAppend,
        Site::JournalSync,
        Site::ShardCreate,
        Site::ShardHeaderWrite,
        Site::ShardHeaderSync,
        Site::ChunkCreate,
        Site::ChunkWrite,
        Site::ChunkSync,
        Site::ChunkReadBack,
        Site::ManifestWrite,
        Site::ManifestSync,
        Site::Validate,
        Site::Publish,
        Site::PublishDurability,
        Site::DirectorySync,
        Site::JournalRemove,
        Site::Allocate,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Site::DestinationCreate => "destination-create",
            Site::LockAcquire => "lock-acquire",
            Site::JournalCreate => "journal-create",
            Site::JournalAppend => "journal-append",
            Site::JournalSync => "journal-sync",
            Site::ShardCreate => "shard-create",
            Site::ShardHeaderWrite => "shard-header-write",
            Site::ShardHeaderSync => "shard-header-sync",
            Site::ChunkCreate => "chunk-create",
            Site::ChunkWrite => "chunk-write",
            Site::ChunkSync => "chunk-sync",
            Site::ChunkReadBack => "chunk-read-back",
            Site::ManifestWrite => "manifest-write",
            Site::ManifestSync => "manifest-sync",
            Site::Validate => "validate",
            Site::Publish => "publish",
            Site::PublishDurability => "publish-durability",
            Site::DirectorySync => "directory-sync",
            Site::JournalRemove => "journal-remove",
            Site::Allocate => "allocate",
        }
    }

    /// Parse a site by the name a CLI flag spells.
    pub fn from_name(name: &str) -> Option<Site> {
        Site::ALL.iter().copied().find(|s| s.name() == name)
    }
}

/// Which visit to a site fails, and how many have happened.
#[derive(Debug)]
struct Armed {
    site: Site,
    /// The 1-based visit that fails. Zero never fires.
    at: u64,
    seen: std::cell::Cell<u64>,
}

/// A run's fault plan. Empty by default.
#[derive(Debug)]
pub struct Faults {
    armed: Vec<Armed>,
    /// How many times each site was visited, **counted rather than listed**.
    ///
    /// The first version kept every visit in a `Vec`, which grew with the
    /// conversion -- an unbounded allocation on the ordinary path, in a program
    /// whose whole point is bounded memory, and one an independent review
    /// found. A counter per named boundary is fixed-size and answers the same
    /// two questions: how often, and which boundaries were reached at all.
    visits: std::cell::RefCell<[u64; Site::ALL.len()]>,
}

impl Default for Faults {
    fn default() -> Self {
        Self {
            armed: Vec::new(),
            visits: std::cell::RefCell::new([0; Site::ALL.len()]),
        }
    }
}

impl Faults {
    /// A plan that never fires.
    pub fn none() -> Self {
        Self::default()
    }

    /// Fail the `at`-th visit to `site`, counting from one.
    pub fn fail_at(mut self, site: Site, at: u64) -> Self {
        self.armed.push(Armed {
            site,
            at,
            seen: std::cell::Cell::new(0),
        });
        self
    }

    /// Called at each boundary. Returns the injected failure, or `Ok`.
    pub fn check(&self, site: Site) -> Result<()> {
        if let Some(at) = Site::ALL.iter().position(|s| *s == site) {
            self.visits.borrow_mut()[at] += 1;
        }
        for a in &self.armed {
            if a.site != site {
                continue;
            }
            let seen = a.seen.get() + 1;
            a.seen.set(seen);
            if seen == a.at {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "injected failure at '{}' visit {seen}: this is a test facility, and the \
                         run must handle it exactly as it would the real failure",
                        site.name()
                    )
                    .into(),
                });
            }
        }
        Ok(())
    }

    /// How many times a site was visited.
    pub fn visits(&self, site: Site) -> usize {
        Site::ALL
            .iter()
            .position(|s| *s == site)
            .map(|at| self.visits.borrow()[at] as usize)
            .unwrap_or(0)
    }

    /// Every site visited at least once, for measured coverage.
    pub fn visited_sites(&self) -> Vec<Site> {
        let counts = self.visits.borrow();
        Site::ALL
            .iter()
            .enumerate()
            .filter(|(at, _)| counts[*at] > 0)
            .map(|(_, s)| *s)
            .collect()
    }

    /// Whether any armed fault is still waiting to fire. A gate that expected
    /// an injected failure and did not get one is a gate that measured
    /// nothing.
    pub fn all_fired(&self) -> bool {
        self.armed.iter().all(|a| a.seen.get() >= a.at)
    }
}
