//! Canonical write authority: creating, staging and publishing an artifact.
//!
//! This lives inside the program that publishes, and that is the whole
//! confinement: no other crate can reach it, because it is not a crate. An
//! earlier arrangement made it one (`moxie-storage-write`) so that a
//! dependency edge could be forbidden. It bought a rule and cost a second copy
//! of the reader's `pread`, which is the duplication this repository exists to
//! refuse. The module boundary is stronger and free: `moxie-engine` cannot call
//! what it cannot name, and it cannot name what is not in its dependency tree.
//!
//! What it owns: confined output creation, bounded chunk writing, the restart
//! journal's file half, and atomic publication. What it does not own: the
//! canonical encoding (`moxie-format`, I/O-free), reading (`moxie-storage`,
//! whose bounded-read primitives this module calls rather than reimplements),
//! what to select or how to convert it (the rest of this program), residency,
//! CUDA, models, or a quantizer.
//!
//! It never overwrites a published artifact, never writes outside the
//! destination it was given, and never deletes a file it did not create.

pub mod fault;
pub mod plan;
pub mod run;

pub use fault::{Faults, Site};
pub use plan::{OutputPlan, PlacedComponent, TensorRequest};
pub use run::{
    JOURNAL_FILE, LOCK_FILE, MANIFEST_FILE, Options, Outcome, ResumeReport, Run,
    STAGED_MANIFEST_FILE, SealedTensor, Start,
};

use moxie_types::{Error, Result};

/// This crate's refusals.
pub(crate) fn invalid(detail: String) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// What a run may spend: memory, per-file disk, and total disk.
///
/// Every one of them is required rather than defaulted. A repack is an
/// operation on a user's disk, and "how much may it use" is not a question a
/// library should answer on the user's behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteBudget {
    scratch_bytes: usize,
    chunk_file_bytes: u64,
    disk_bytes: u64,
}

impl WriteBudget {
    /// The largest scratch a run may admit. Not a default: a ceiling, so a
    /// caller cannot ask for a gigabyte of "bounded" staging.
    pub const MAX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

    pub fn new(scratch_bytes: usize, chunk_file_bytes: u64, disk_bytes: u64) -> Result<Self> {
        if scratch_bytes == 0 || scratch_bytes > Self::MAX_SCRATCH_BYTES {
            return Err(invalid(format!(
                "payload scratch must be between 1 and {} byte(s), got {scratch_bytes}",
                Self::MAX_SCRATCH_BYTES
            )));
        }
        if chunk_file_bytes == 0 {
            return Err(invalid("a chunk file of zero bytes holds no tensor".into()));
        }
        if disk_bytes < chunk_file_bytes {
            return Err(invalid(format!(
                "the disk budget ({disk_bytes}) is below one chunk file ({chunk_file_bytes}): the \
                 run could not write its first chunk"
            )));
        }
        Ok(Self {
            scratch_bytes,
            chunk_file_bytes,
            disk_bytes,
        })
    }

    pub fn scratch_bytes(self) -> usize {
        self.scratch_bytes
    }

    pub fn chunk_file_bytes(self) -> u64 {
        self.chunk_file_bytes
    }

    pub fn disk_bytes(self) -> u64 {
        self.disk_bytes
    }
}
