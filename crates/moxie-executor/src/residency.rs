//! The driver that performs what the residency authority decides.
//!
//! It holds **no cache**, chooses **no victim** and keeps **no policy**. Every
//! decision in this file was already made in `moxie_memory::residency`; what is
//! here is the two effects that authority is forbidden to perform itself --
//! opening a file and touching a device.
//!
//! That split is the same one document 02 already draws twice. `moxie-memory`
//! owns "reservations, leases, arenas, tier movement, admission"; `moxie-storage`
//! owns "bounded reads/mappings"; `moxie-executor` "schedules actual effects".
//! A `crate::arena::DeviceArena` is a `moxie_memory::Arena`'s pure ranges paired
//! with one real allocation, for exactly this reason, and a residency driver is
//! the same pairing one layer up.
//!
//! The consequence worth stating plainly: **this file must stay boring.** The
//! moment it grows a map from chunk to bytes, M2's "exactly one production
//! weight-residency owner" is false, and an `arch-check` rule now says so.

use std::collections::BTreeMap;

use moxie_memory::{ArtifactId, ChunkId, Outcome, PendingWork, ResidencyAuthority, WorkOrder};
use moxie_storage::Shard;
use moxie_types::{Error, Result};

/// Where a chunk's bytes come from.
///
/// A trait, not a concrete reader, because the authority's correctness has to
/// be testable against a source that fails halfway through -- which no real
/// filesystem will do on request. The production implementation is
/// [`ShardSource`]; the test doubles live beside the tests that need them.
pub trait ChunkSource {
    /// Fill `into`, exactly, with the chunk's logical range.
    ///
    /// `into.len()` is always the chunk's length: the authority sized it from
    /// the identity. An implementation that fills less must return an error --
    /// a partial read reported as success is the defect this whole layer exists
    /// to make impossible.
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> Result<()>;
}

/// One artifact's chunks, served from opened safetensors shards.
///
/// The role-to-tensor mapping arrives as data. This crate may not learn what a
/// model family is any more than the authority may, and the names in the table
/// below are the artifact's own -- read from its index, supplied by whoever
/// opened it.
#[derive(Debug)]
pub struct ShardSource {
    artifact: ArtifactId,
    shards: Vec<Shard>,
    /// canonical role -> (shard index, tensor name in that shard)
    roles: BTreeMap<String, (usize, String)>,
}

impl ShardSource {
    pub fn new(artifact: ArtifactId, shards: Vec<Shard>) -> Self {
        ShardSource {
            artifact,
            shards,
            roles: BTreeMap::new(),
        }
    }

    /// Bind one canonical role to one tensor in one opened shard.
    pub fn role(
        mut self,
        role: impl Into<String>,
        shard: usize,
        tensor: impl Into<String>,
    ) -> Result<Self> {
        let role = role.into();
        let tensor = tensor.into();
        if shard >= self.shards.len() {
            return Err(Error::InvalidRequest {
                field: "shard",
                detail: format!("shard {shard} of {}", self.shards.len()),
            });
        }
        if self.shards[shard].header().get(&tensor).is_err() {
            return Err(Error::InvalidArtifact {
                detail: format!("shard {shard} has no tensor {tensor:?}"),
            });
        }
        self.roles.insert(role, (shard, tensor));
        Ok(self)
    }

    pub fn artifact(&self) -> &ArtifactId {
        &self.artifact
    }
}

impl ChunkSource for ShardSource {
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> Result<()> {
        if chunk.artifact() != &self.artifact {
            return Err(Error::InvalidRequest {
                field: "artifact",
                detail: format!(
                    "chunk names artifact {:?}; this source serves {:?}",
                    chunk.artifact().as_str(),
                    self.artifact.as_str()
                ),
            });
        }
        let role = chunk.slot().role();
        let (shard, tensor) = self.roles.get(role).ok_or_else(|| Error::InvalidRequest {
            field: "role",
            detail: format!("no tensor is bound to role {role:?}"),
        })?;
        self.shards[*shard].read_tensor_range(tensor, chunk.range().offset_bytes(), into)
    }
}

/// Perform one read order and report its outcome.
///
/// Returns the follow-on orders the completion released -- this ticket's own
/// upload, and any ticket that was waiting for the same host bytes. A failure
/// is reported to the authority *and* propagated: the caller learns its chunk
/// did not arrive, and the authority has already given back every byte the
/// attempt charged.
pub fn perform_read<S: ChunkSource>(
    authority: &mut ResidencyAuthority,
    source: &mut S,
    order: &WorkOrder,
) -> Result<Vec<WorkOrder>> {
    let WorkOrder::Read {
        ticket,
        chunk,
        len_bytes,
        ..
    } = order
    else {
        return Err(Error::InvalidRequest {
            field: "order",
            detail: "this is not a read order".into(),
        });
    };
    let outcome = {
        let destination = authority.read_destination(*ticket)?;
        if destination.len() as u64 != *len_bytes {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "the authority offered {} byte(s) for a {len_bytes}-byte order",
                    destination.len()
                ),
            });
        }
        match source.read_chunk(chunk, destination) {
            Ok(()) => Outcome::Completed,
            Err(e) => Outcome::Failed(e),
        }
    };
    if let Outcome::Failed(error) = &outcome {
        let error = error.clone();
        authority.complete_read(*ticket, outcome)?;
        return Err(error);
    }
    authority.complete_read(*ticket, outcome)
}

/// Perform every read reachable from one pending acquire, plus any prefetch the
/// authority releases once demand goes idle.
///
/// Upload orders are **returned, not performed**: a copy needs a device, and a
/// host lane has none. The caller hands them to the device driver.
pub fn drain_reads<S: ChunkSource>(
    authority: &mut ResidencyAuthority,
    source: &mut S,
    work: PendingWork,
) -> Result<Vec<WorkOrder>> {
    let mut pending = match work {
        PendingWork::Issued(order) => vec![order],
        // Coalesced: another acquire owns the transfer, and performing a second
        // read here would be the duplicate the authority exists to prevent.
        PendingWork::Coalesced => Vec::new(),
        PendingWork::Queued => Vec::new(),
    };
    let mut uploads = Vec::new();
    while let Some(order) = pending.pop() {
        match order {
            WorkOrder::Read { .. } => {
                pending.extend(perform_read(authority, source, &order)?);
            }
            WorkOrder::Upload { .. } => uploads.push(order),
        }
    }
    while let Some(order) = authority.next_prefetch() {
        match order {
            WorkOrder::Read { .. } => {
                pending.push(order);
                while let Some(next) = pending.pop() {
                    match next {
                        WorkOrder::Read { .. } => {
                            pending.extend(perform_read(authority, source, &next)?);
                        }
                        WorkOrder::Upload { .. } => uploads.push(next),
                    }
                }
            }
            WorkOrder::Upload { .. } => uploads.push(order),
        }
    }
    Ok(uploads)
}

#[cfg(feature = "driver")]
pub use device::{DeviceResidency, UploadRefused};

#[cfg(feature = "driver")]
mod device {
    use moxie_cuda::{DeviceBuffer, Event, RankContext, Stream};
    use moxie_memory::{Outcome, ResidencyAuthority, WorkOrder};
    use moxie_types::{Error, Result, Scope};

    /// One device's residency cache: the single real allocation whose ranges
    /// the authority hands out.
    ///
    /// It is the physical half of `moxie_memory`'s device arena for this scope,
    /// and it owns nothing else. Every offset it copies into was chosen by the
    /// authority; this type never decides where a chunk goes.
    #[derive(Debug)]
    pub struct DeviceResidency<'ctx> {
        buffer: DeviceBuffer<'ctx>,
        ctx: &'ctx RankContext,
        capacity: u64,
    }

    /// An upload the device refused, with the copy's identity intact.
    #[derive(Debug)]
    pub struct UploadRefused {
        pub error: Error,
        /// True when the submission state is unknown, so the bytes at both ends
        /// must be withheld rather than reused. R07.
        pub submission_unknown: bool,
    }

    impl core::fmt::Display for UploadRefused {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.error)
        }
    }

    impl std::error::Error for UploadRefused {}

    impl<'ctx> DeviceResidency<'ctx> {
        /// Back one scope's cache with one real allocation of exactly the
        /// capacity the authority admitted.
        pub fn create(ctx: &'ctx RankContext, authority: &ResidencyAuthority) -> Result<Self> {
            let scope = Scope::Device(ctx.uuid());
            let capacity = authority
                .cap_bytes(scope)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "scope",
                    detail: format!("the authority opened no cache for {scope}"),
                })?;
            let bytes = usize::try_from(capacity).map_err(|_| Error::CapacityExceeded {
                tier: None,
                requested_bytes: capacity,
                available_bytes: 0,
            })?;
            let buffer = DeviceBuffer::alloc(ctx, bytes)?;
            Ok(DeviceResidency {
                buffer,
                ctx,
                capacity,
            })
        }

        pub const fn capacity(&self) -> u64 {
            self.capacity
        }

        pub fn scope(&self) -> Scope {
            Scope::Device(self.ctx.uuid())
        }

        /// Read a device range back, for an oracle that has to compare what
        /// arrived against what was asked for.
        pub fn read_back(&self, offset: u64, into: &mut [u8]) -> Result<()> {
            self.buffer.copy_to_host_at(offset as usize, into)
        }

        /// Copy one upload order's host bytes into the device range the
        /// authority chose, and report the outcome.
        ///
        /// Readiness is the event, not the return of the enqueue call.
        /// Document 02: "An upload owns or leases its source bytes through a
        /// completion event. A scratch source cannot be overwritten merely
        /// because the enqueue function returned." The authority pins the host
        /// source for the copy's whole lifetime, and this function does not
        /// report completion until the event says so.
        pub fn perform_upload(
            &mut self,
            authority: &mut ResidencyAuthority,
            stream: &Stream<'ctx>,
            order: &WorkOrder,
        ) -> std::result::Result<(), UploadRefused> {
            let WorkOrder::Upload {
                ticket,
                scope,
                device_offset,
                len_bytes,
                ..
            } = order
            else {
                return Err(UploadRefused {
                    error: Error::InvalidRequest {
                        field: "order",
                        detail: "this is not an upload order".into(),
                    },
                    submission_unknown: false,
                });
            };
            let refuse = |error: Error, unknown: bool| UploadRefused {
                error,
                submission_unknown: unknown,
            };
            if *scope != self.scope() {
                return Err(refuse(
                    Error::InvalidRequest {
                        field: "scope",
                        detail: format!("order names {scope}; this cache is {}", self.scope()),
                    },
                    false,
                ));
            }
            if device_offset.saturating_add(*len_bytes) > self.capacity {
                return Err(refuse(
                    Error::CapacityExceeded {
                        tier: None,
                        requested_bytes: device_offset.saturating_add(*len_bytes),
                        available_bytes: self.capacity,
                    },
                    false,
                ));
            }

            // The source stays borrowed from the authority for the enqueue
            // only; the pin that keeps it alive across the copy is the
            // authority's, not this borrow.
            let enqueued = {
                let source = match authority.upload_source(*ticket) {
                    Ok(s) => s,
                    Err(e) => return Err(refuse(e, false)),
                };
                if source.len() as u64 != *len_bytes {
                    return Err(refuse(
                        Error::InvalidArtifact {
                            detail: format!(
                                "the authority offered {} source byte(s) for a {len_bytes}-byte \
                                 copy",
                                source.len()
                            ),
                        },
                        false,
                    ));
                }
                // SAFETY: the destination range lies inside this cache's one
                // allocation (checked above), the source slice is pinned by the
                // authority for the copy's lifetime, and the copy is recorded
                // on `stream` whose event is waited on below before completion
                // is reported.
                unsafe {
                    self.buffer
                        .copy_from_host_async_at(*device_offset as usize, source, stream)
                }
            };

            match enqueued {
                Ok(()) => {}
                Err(error) => {
                    // Enqueue failed before any byte moved.
                    let _ = authority.complete_upload(*ticket, Outcome::Failed(error.clone()));
                    return Err(refuse(error, false));
                }
            }

            let event = match Event::new(self.ctx).and_then(|e| e.record(stream).map(|()| e)) {
                Ok(e) => e,
                Err(error) => {
                    // The copy was submitted and there is now no event to wait
                    // on: the submission state is unknown, so both ends are
                    // withheld rather than reused.
                    let _ = authority
                        .complete_upload(*ticket, Outcome::SubmissionUnknown(error.clone()));
                    return Err(refuse(error, true));
                }
            };
            match event.synchronize() {
                Ok(()) => {
                    authority
                        .complete_upload(*ticket, Outcome::Completed)
                        .map_err(|e| refuse(e, false))?;
                    Ok(())
                }
                Err(error) => {
                    let _ = authority
                        .complete_upload(*ticket, Outcome::SubmissionUnknown(error.clone()));
                    Err(refuse(error, true))
                }
            }
        }
    }
}
