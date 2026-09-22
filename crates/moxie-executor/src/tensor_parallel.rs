//! Two-rank tensor parallelism over a peer-connected GPU pair (task 0057, M5.2).
//!
//! A [`RankGroup`] is formed from two rank contexts whose devices grant each
//! other peer access; forming it enables that access, and a pair the driver
//! does not grant is refused. There is no host-staged fallback: `moxie-cuda`
//! refuses a peer copy the destination context has no grant for.
//!
//! The one collective is an ordered all-gather of column shards. Every call
//! consumes the group's next sequence number, and both ranks must declare that
//! number and the same shape and precision before any byte moves. Each rank
//! then pulls every rank's shard into its own full-width output, on its own
//! stream, after waiting on the event that recorded the shard. Rank order is
//! global column order, so the gather concatenates and never sums.
//!
//! A collective refused before its first copy settles each shard's producer
//! and returns every range to its arena. Once copies are enqueued, a source is
//! read by both destination streams, so no range is released until both
//! streams are drained within the group's deadline; if either cannot be, every
//! range of the collective is withheld and the error is device loss naming the
//! collective and its deadline. No rank ever holds a partial result.
//!
//! One host thread drives both ranks. Document 01's one execution thread per
//! GPU is not met yet: a device range is `!Send`, and a rank thread would need
//! a cross-thread handle to its peer's range.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use core::ffi::c_void;

use moxie_cuda::{Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage};
use moxie_graph::PartitionRule;
use moxie_types::{DeviceUuid, Error, Precision, Result};

use crate::affine_linear::{WeightShardRanges, WeightShardSpec, shard_weight_ranges};
use crate::arena::{DeviceArena, DeviceRange, OperationLease};
use crate::lease::LeaseState;

const ALIGNMENT: u64 = 256;
const BF16_BYTES: u64 = 2;

/// What every rank must agree on before a collective moves bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherDeclaration {
    pub sequence: u64,
    pub rows: u64,
    /// Columns each rank contributes.
    pub columns: u64,
    pub precision: Precision,
}

/// The logical linear a column shard belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearShape {
    pub rows: u64,
    pub in_features: u64,
    pub out_features: u64,
}

/// Two ranks on a peer-connected pair, and their collective schedule.
#[derive(Debug)]
pub struct RankGroup<'ctx> {
    ranks: [&'ctx RankContext; 2],
    sequence: u64,
    deadline: Duration,
}

/// A refused collective: the typed error each rank receives.
#[derive(Debug)]
pub struct GatherRefused {
    pub errors: [Error; 2],
}

impl core::fmt::Display for GatherRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "rank 0: {}; rank 1: {}", self.errors[0], self.errors[1])
    }
}

impl std::error::Error for GatherRefused {}

/// One rank's buffers for a submitted column-sharded linear.
#[derive(Debug)]
pub struct ShardBuffers<'ctx> {
    input: DeviceRange<'ctx>,
    weight: DeviceRange<'ctx>,
    local: DeviceRange<'ctx>,
    /// Host sources of the two uploads, retained until completion.
    _sources: [Vec<u8>; 2],
}

/// One rank's submitted column shard, leased until its event completes.
#[derive(Debug)]
pub struct RankShard<'ctx> {
    declaration: GatherDeclaration,
    lease: OperationLease<Event<'ctx>, ShardBuffers<'ctx>>,
}

/// One rank's complete gathered output.
#[derive(Debug)]
pub struct Gathered<'ctx> {
    range: DeviceRange<'ctx>,
}

/// The BF16 linear kernel loaded on one rank.
#[derive(Debug)]
pub struct ColumnLinear<'ctx> {
    ctx: &'ctx RankContext,
    package: ResolvedModule<'ctx>,
}

impl<'ctx> RankGroup<'ctx> {
    /// Form a group, enabling peer access in both directions. `deadline`
    /// bounds every wait the group makes.
    pub fn form(ranks: [&'ctx RankContext; 2], deadline: Duration) -> Result<Self> {
        ranks[0].enable_peer_access(ranks[1])?;
        ranks[1].enable_peer_access(ranks[0])?;
        Ok(Self {
            ranks,
            sequence: 0,
            deadline,
        })
    }

    pub fn uuids(&self) -> [DeviceUuid; 2] {
        [self.ranks[0].uuid(), self.ranks[1].uuid()]
    }

    /// The sequence number the next collective must declare.
    pub fn next_sequence(&self) -> u64 {
        self.sequence
    }

    /// Gather both ranks' column shards into a full-width output on each rank.
    ///
    /// `shards[r]` is rank `r`'s own result, including its failure. Every
    /// range, gathered or not, is returned to `arenas[r]` unless the call
    /// succeeds, when each rank keeps only its gathered output.
    #[allow(clippy::result_large_err)]
    pub fn all_gather(
        &mut self,
        shards: [Result<RankShard<'ctx>>; 2],
        arenas: [&mut DeviceArena<'ctx>; 2],
        streams: [&Stream<'ctx>; 2],
        cancel: &AtomicBool,
    ) -> std::result::Result<[Gathered<'ctx>; 2], GatherRefused> {
        let sequence = self.sequence;
        self.sequence += 1;
        let deadline = Instant::now() + self.deadline;
        let [first, second] = shards;
        let mut failure = None;
        let mut submitted = Vec::new();
        for (rank, shard) in [first, second].into_iter().enumerate() {
            match shard {
                Ok(shard) => submitted.push((rank, shard)),
                Err(error) => failure = failure.or(Some(error)),
            }
        }
        if failure.is_none() {
            failure = self.agree(sequence, &submitted, streams, cancel).err();
        }
        let mut arenas = arenas;
        if let Some(error) = failure {
            return Err(self.refuse_before_copies(error, submitted, &mut arenas, deadline));
        }
        let declaration = submitted[0].1.declaration;

        // Everything that can fail before a copy is obtained first, so a
        // failure here has no peer read in flight.
        let bytes = declaration.rows * declaration.columns * BF16_BYTES * 2;
        let mut prepared = Vec::new();
        for (rank, arena) in arenas.iter_mut().enumerate() {
            let made = Event::new(self.ranks[rank]).and_then(|done| {
                let range = arena
                    .allocate(bytes, ALIGNMENT, "all-gather output")
                    .map_err(|refused| refused.error)?;
                Ok((done, range))
            });
            match made {
                Ok(parts) => prepared.push(parts),
                Err(error) => {
                    for (rank, (_, range)) in prepared.into_iter().enumerate() {
                        // A range that cannot be released stays allocated and visible.
                        let _ = arenas[rank].release(range);
                    }
                    return Err(self.refuse_before_copies(error, submitted, &mut arenas, deadline));
                }
            }
        }

        // From the first copy on, a source is read by both destination
        // streams. No range is released until both are drained; if either
        // cannot be, every range of this collective is withheld.
        let shards = [&submitted[0].1, &submitted[1].1];
        let mut outputs = Vec::new();
        let mut error = None;
        for (rank, (done, range)) in prepared.into_iter().enumerate() {
            let mut lease = OperationLease::new(format!("all-gather {sequence} output"), range)
                .expect("a named lease");
            match self.enqueue_gather(declaration, shards, lease.resource(), streams[rank], &done) {
                Ok(()) => {
                    lease
                        .submit_tracked(done)
                        .expect("a fresh lease records once");
                    outputs.push(lease);
                }
                Err(e) => {
                    lease.mark_lost(self.ranks[rank].ordinal(), format!("all-gather: {e}"));
                    error = Some(e);
                    break;
                }
            }
        }
        let mut drained = Vec::new();
        if error.is_none() {
            for (rank, lease) in outputs.into_iter().enumerate() {
                match self.settle(lease, rank, deadline) {
                    Ok(range) => drained.push(range),
                    Err(e) => {
                        error = Some(e);
                        break;
                    }
                }
            }
        }
        if let Some(error) = error {
            // Dropping in-flight leases and live ranges withholds them.
            drop(drained);
            drop(submitted);
            return Err(GatherRefused {
                errors: [error.clone(), error],
            });
        }

        let mut error = None;
        for (rank, shard) in submitted {
            if let Err(e) = self.release_shard(rank, shard, arenas[rank], deadline) {
                error = error.or(Some(e));
            }
        }
        if let Some(error) = error {
            for (rank, range) in drained.into_iter().enumerate() {
                // A range that cannot be released stays allocated and visible.
                let _ = arenas[rank].release(range);
            }
            return Err(GatherRefused {
                errors: [error.clone(), error],
            });
        }
        let mut gathered = drained.into_iter().map(|range| Gathered { range });
        Ok([
            gathered.next().expect("two ranks"),
            gathered.next().expect("two ranks"),
        ])
    }

    /// Refuse a collective that moved no byte. A source's only reader is its
    /// own producer, so settling that is enough to release it.
    fn refuse_before_copies(
        &self,
        error: Error,
        submitted: Vec<(usize, RankShard<'ctx>)>,
        arenas: &mut [&mut DeviceArena<'ctx>; 2],
        deadline: Instant,
    ) -> GatherRefused {
        let mut errors = [error.clone(), error];
        for (rank, shard) in submitted {
            if let Err(error) = self.release_shard(rank, shard, arenas[rank], deadline) {
                errors = [error.clone(), error];
            }
        }
        GatherRefused { errors }
    }

    /// Both ranks declare this collective's sequence, one shape and one
    /// precision, on their own device, before anything is enqueued.
    fn agree(
        &self,
        sequence: u64,
        submitted: &[(usize, RankShard<'ctx>)],
        streams: [&Stream<'ctx>; 2],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled { at: "all-gather" });
        }
        for (rank, shard) in submitted {
            let uuid = self.ranks[*rank].uuid();
            if shard.lease.resource().local.device_uuid() != uuid
                || streams[*rank].device_uuid() != uuid
            {
                return Err(collective(format!(
                    "rank {rank}'s shard or stream is not on its device {uuid}"
                )));
            }
            if shard.declaration.sequence != sequence {
                return Err(collective(format!(
                    "rank {rank} declared sequence {} but the group is at {sequence}",
                    shard.declaration.sequence
                )));
            }
        }
        // The sequence is each rank's agreement with the schedule, checked
        // above; the shape and precision are the ranks' agreement with each
        // other.
        let [d0, d1] = [submitted[0].1.declaration, submitted[1].1.declaration];
        if (d0.rows, d0.columns, d0.precision) != (d1.rows, d1.columns, d1.precision) {
            return Err(collective(format!("ranks declared {d0:?} and {d1:?}")));
        }
        Ok(())
    }

    /// Enqueue one destination's pulls of every shard, then its `done` event.
    fn enqueue_gather(
        &self,
        declaration: GatherDeclaration,
        shards: [&RankShard<'ctx>; 2],
        gathered: &DeviceRange<'ctx>,
        stream: &Stream<'ctx>,
        done: &Event<'ctx>,
    ) -> Result<()> {
        let part = declaration.columns * BF16_BYTES;
        let row = part * 2;
        for shard in shards {
            let ready = shard
                .lease
                .completion()
                .expect("a submitted shard has an event");
            stream.wait_event(ready)?;
        }
        // ponytail: one peer copy per row and rank; a pitched peer copy
        // (cuMemcpy3DPeerAsync) replaces them when row counts grow.
        for (source, shard) in shards.iter().enumerate() {
            let local = &shard.lease.resource().local;
            for r in 0..declaration.rows {
                // SAFETY: the shard leases retain `local`, and the caller's
                // lease retains `gathered`, until `done` is observed or both
                // are withheld.
                unsafe {
                    gathered.copy_from_peer_async_at(
                        r * row + source as u64 * part,
                        local,
                        r * part,
                        part,
                        stream,
                    )?;
                }
            }
        }
        done.record(stream)
    }

    fn release_shard(
        &self,
        rank: usize,
        shard: RankShard<'ctx>,
        arena: &mut DeviceArena<'ctx>,
        deadline: Instant,
    ) -> Result<()> {
        let buffers = self.settle(shard.lease, rank, deadline)?;
        release(arena, [buffers.input, buffers.weight, buffers.local])
    }

    fn settle<R>(
        &self,
        lease: OperationLease<Event<'ctx>, R>,
        rank: usize,
        deadline: Instant,
    ) -> Result<R> {
        settle(lease, self.ranks[rank].ordinal(), deadline, self.deadline)
    }
}

impl<'ctx> RankShard<'ctx> {
    pub fn declaration(&self) -> GatherDeclaration {
        self.declaration
    }

    /// Wait for this shard alone, read its local output and release it. The
    /// single-rank path, with no collective.
    pub fn finish(self, arena: &mut DeviceArena<'ctx>, deadline: Duration) -> Result<Vec<u8>> {
        let buffers = settle(self.lease, u32::MAX, Instant::now() + deadline, deadline)?;
        let mut out = vec![0; buffers.local.bytes() as usize];
        let read = buffers.local.copy_to_host(&mut out);
        release(arena, [buffers.input, buffers.weight, buffers.local])?;
        read.map(|()| out)
    }
}

impl<'ctx> Gathered<'ctx> {
    pub fn read(&self, destination: &mut [u8]) -> Result<()> {
        self.range.copy_to_host(destination)
    }

    pub fn release(self, arena: &mut DeviceArena<'ctx>) -> Result<()> {
        release(arena, [self.range])
    }
}

impl<'ctx> ColumnLinear<'ctx> {
    pub fn load(ctx: &'ctx RankContext) -> Result<Self> {
        // SAFETY: the build's own `include_bytes!` of the pinned nvcc output.
        let image = unsafe { TrustedImage::from_build_output(moxie_kernels::BF16_CHAIN_FATBIN) }?;
        let package = Module::load(ctx, ModuleImage::Binary(image))?
            .resolve_all(&[moxie_kernels::BF16_LINEAR.to_string()])?;
        Ok(Self { ctx, package })
    }

    /// Upload rank `rank`'s rows of `weight` and the whole `input`, and run
    /// `moxie_bf16_linear_v1` over that rank's columns.
    ///
    /// The shard is addressed by `shard_weight_ranges`, before anything is
    /// allocated; a shape it refuses touches no device. `cancel` is checked
    /// first, so a cancellation between two ranks' launches stops the second.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        arena: &mut DeviceArena<'ctx>,
        stream: &Stream<'ctx>,
        shape: LinearShape,
        input: Vec<u8>,
        weight: &[u8],
        rank: u32,
        ranks: u32,
        sequence: u64,
        cancel: &AtomicBool,
    ) -> Result<RankShard<'ctx>> {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled {
                at: "column linear launch",
            });
        }
        let LinearShape {
            rows,
            in_features,
            out_features,
        } = shape;
        let WeightShardRanges::Bf16 { weights } = shard_weight_ranges(
            PartitionRule::ColumnShardable,
            WeightShardSpec::Bf16 {
                out_features,
                in_features,
            },
            ranks,
            rank,
        )?
        else {
            unreachable!("a BF16 weight has BF16 ranges");
        };
        let bytes = |n: u64| {
            n.checked_mul(BF16_BYTES)
                .and_then(|n| usize::try_from(n).ok())
        };
        if Some(input.len()) != rows.checked_mul(in_features).and_then(bytes)
            || Some(weight.len()) != out_features.checked_mul(in_features).and_then(bytes)
        {
            return Err(invalid(
                "operands",
                "input or weight bytes disagree with the shape",
            ));
        }
        let start = weights.offset_bytes() as usize;
        let shard = weight[start..start + weights.len_bytes() as usize].to_vec();
        let columns = out_features / u64::from(ranks);
        let elements = rows * columns;
        let grid = u32::try_from(elements.div_ceil(256))
            .map_err(|_| invalid("launch", "linear grid overflowed"))?;

        let mut ranges = Vec::new();
        for (size, owner) in [
            (input.len() as u64, "column linear input"),
            (shard.len() as u64, "column linear weight shard"),
            (elements * BF16_BYTES, "column linear local output"),
        ] {
            match arena.allocate(size, ALIGNMENT, owner) {
                Ok(range) => ranges.push(range),
                Err(refused) => {
                    release(arena, ranges)?;
                    return Err(refused.error);
                }
            }
        }
        let local = ranges.pop().expect("three ranges");
        let weight_range = ranges.pop().expect("three ranges");
        let input_range = ranges.pop().expect("three ranges");
        let mut lease = OperationLease::new(
            "column linear",
            ShardBuffers {
                input: input_range,
                weight: weight_range,
                local,
                _sources: [input, shard],
            },
        )
        .map_err(|refused| refused.error)?;
        let enqueued = (|| {
            let buffers = lease.resource();
            // SAFETY: the lease retains both sources and ranges until the
            // event recorded below completes.
            unsafe {
                buffers
                    .input
                    .copy_from_host_async(&buffers._sources[0], stream)?;
                buffers
                    .weight
                    .copy_from_host_async(&buffers._sources[1], stream)?;
            }
            let (mut x, mut w, mut h) = (
                buffers.input.device_address()?,
                buffers.weight.device_address()?,
                buffers.local.device_address()?,
            );
            let (mut rows, mut k, mut n) = (rows, in_features, columns);
            let mut params: [*mut c_void; 6] = [
                (&raw mut x).cast(),
                (&raw mut w).cast(),
                (&raw mut h).cast(),
                (&raw mut rows).cast(),
                (&raw mut k).cast(),
                (&raw mut n).cast(),
            ];
            // SAFETY: the parameters match `moxie_bf16_linear_v1`'s ABI and
            // every address names a range sized for this shape above.
            unsafe {
                self.package
                    .launch_async(0, stream, (grid, 1, 1), (256, 1, 1), 0, &mut params)?;
            }
            let ready = Event::new(self.ctx)?;
            ready.record(stream)?;
            Ok(ready)
        })();
        match enqueued {
            Ok(ready) => lease.submit_tracked(ready)?,
            Err(error) => {
                lease.mark_lost(self.ctx.ordinal(), format!("column linear: {error}"));
                return Err(error);
            }
        }
        Ok(RankShard {
            declaration: GatherDeclaration {
                sequence,
                rows,
                columns,
                precision: Precision::Bf16,
            },
            lease,
        })
    }
}

/// Retire `lease` once its event completes, polling until `deadline`.
fn settle<R>(
    mut lease: OperationLease<Event<'_>, R>,
    device: u32,
    deadline: Instant,
    bound: Duration,
) -> Result<R> {
    loop {
        match lease.retire() {
            Ok((_, resource)) => return Ok(resource),
            Err(refused) => {
                if refused.lease.state() == LeaseState::Lost {
                    return Err(refused.error);
                }
                if Instant::now() >= deadline {
                    // Dropping the in-flight lease withholds its resource.
                    return Err(Error::DeviceLost {
                        device,
                        detail: format!(
                            "{}: completion not observed within the collective's {} ms \
                             deadline; its ranges are withheld",
                            refused.lease.label(),
                            bound.as_millis()
                        ),
                    });
                }
                lease = refused.lease;
                std::thread::yield_now();
            }
        }
    }
}

fn release<'ctx>(
    arena: &mut DeviceArena<'ctx>,
    ranges: impl IntoIterator<Item = DeviceRange<'ctx>>,
) -> Result<()> {
    let mut result = Ok(());
    for range in ranges {
        if let Err(refused) = arena.release(range) {
            result = result.and(Err(refused.error));
        }
    }
    result
}

fn collective(detail: String) -> Error {
    Error::InvalidRequest {
        field: "collective",
        detail,
    }
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
