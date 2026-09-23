//! Task 0057 low-level all-gather declarations and the shared worker rendezvous.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "paged-attention-binding")]
use std::sync::{Condvar, Mutex};
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

/// Fixed-size collective declaration shared by the two rank workers.
#[cfg(feature = "paged-attention-binding")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RankDeclaration {
    pub sequence: u64,
    pub operation: u16,
    pub rank: u8,
    pub shape: [u64; 4],
    pub precision: Precision,
}

#[cfg(feature = "paged-attention-binding")]
#[derive(Debug, Clone)]
struct RankSubmission {
    declaration: RankDeclaration,
    status: u32,
    error: Option<Error>,
}

#[cfg(feature = "paged-attention-binding")]
#[derive(Debug, Clone)]
struct RankRoundResult {
    status: u32,
    error: Option<Error>,
}

#[cfg(feature = "paged-attention-binding")]
#[derive(Debug)]
struct RankRendezvousState {
    sequence: u64,
    submissions: [Option<RankSubmission>; 2],
    arrived: u8,
    departed: u8,
    result: Option<RankRoundResult>,
    lost: Option<Error>,
}

/// Two-party host rendezvous. A rank cannot leave a failed operation without
/// submitting its status, and a missing worker makes the group loss sticky.
#[cfg(feature = "paged-attention-binding")]
#[derive(Debug)]
pub(crate) struct RankRendezvous {
    device: u32,
    state: Mutex<RankRendezvousState>,
    changed: Condvar,
}

#[cfg(feature = "paged-attention-binding")]
impl RankRendezvous {
    pub(crate) fn new(device: u32) -> Self {
        Self {
            device,
            state: Mutex::new(RankRendezvousState {
                sequence: 0,
                submissions: [None, None],
                arrived: 0,
                departed: 0,
                result: None,
                lost: None,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn lose(&self, detail: impl Into<String>) -> Error {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.lose_locked(&mut state, detail.into())
    }

    fn lose_locked(&self, state: &mut RankRendezvousState, detail: String) -> Error {
        let lost = state.lost.get_or_insert(Error::DeviceLost {
            device: self.device,
            detail,
        });
        self.changed.notify_all();
        lost.clone()
    }

    pub(crate) fn lost(&self) -> Option<Error> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lost
            .clone()
    }

    pub(crate) fn enter(
        &self,
        rank: usize,
        declaration: RankDeclaration,
        local: Result<()>,
        deadline: Instant,
    ) -> Result<()> {
        if rank >= 2 {
            return Err(Error::InvalidRequest {
                field: "rank",
                detail: "a two-rank rendezvous received an invalid rank".into(),
            });
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(lost) = &state.lost {
                return Err(lost.clone());
            }
            if state.result.is_none() {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(self.lose_locked(
                    &mut state,
                    "a rank did not leave the prior rendezvous before the deadline".into(),
                ));
            }
            let (next, wait) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && state.result.is_some() {
                return Err(self.lose_locked(
                    &mut state,
                    "a rank did not leave the prior rendezvous before the deadline".into(),
                ));
            }
        }
        if state.submissions[rank].is_some() {
            let sequence = state.sequence;
            return Err(self.lose_locked(
                &mut state,
                format!("rank {rank} entered rendezvous {sequence} twice"),
            ));
        }
        state.submissions[rank] = Some(RankSubmission {
            declaration,
            status: u32::from(local.is_err()),
            error: local.err(),
        });
        state.arrived += 1;
        if state.arrived == 2 {
            let [Some(first), Some(second)] = &state.submissions else {
                return Err(self.lose_locked(
                    &mut state,
                    "rendezvous arrival count disagrees with rank slots".into(),
                ));
            };
            let declarations_match = first.declaration.sequence == state.sequence
                && second.declaration.sequence == state.sequence
                && first.declaration.rank == 0
                && second.declaration.rank == 1
                && first.declaration.operation == second.declaration.operation
                && first.declaration.shape == second.declaration.shape
                && first.declaration.precision == second.declaration.precision;
            let status = first
                .status
                .max(second.status)
                .max(u32::from(!declarations_match));
            let lost_error = [&first.error, &second.error]
                .into_iter()
                .flatten()
                .find(|error| matches!(error, Error::DeviceLost { .. }))
                .cloned();
            let error = lost_error
                .or_else(|| first.error.clone())
                .or_else(|| second.error.clone())
                .or_else(|| {
                    (!declarations_match).then(|| Error::InvalidRequest {
                        field: "collective",
                        detail: format!(
                            "rendezvous declarations differ at sequence {}",
                            state.sequence
                        ),
                    })
                });
            if matches!(error, Some(Error::DeviceLost { .. })) {
                state.lost = error.clone();
            }
            state.result = Some(RankRoundResult { status, error });
            self.changed.notify_all();
        }

        loop {
            if let Some(lost) = &state.lost {
                return Err(lost.clone());
            }
            if let Some(result) = &state.result {
                let result = result.clone();
                state.departed += 1;
                if state.departed == 2 {
                    let Some(next_sequence) = state.sequence.checked_add(1) else {
                        return Err(
                            self.lose_locked(&mut state, "rendezvous sequence overflowed".into())
                        );
                    };
                    state.sequence = next_sequence;
                    state.submissions = [None, None];
                    state.arrived = 0;
                    state.departed = 0;
                    state.result = None;
                    self.changed.notify_all();
                }
                return if result.status == 0 {
                    Ok(())
                } else {
                    Err(result.error.unwrap_or_else(|| Error::InvalidRequest {
                        field: "collective",
                        detail: "a rank reported failure".into(),
                    }))
                };
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let sequence = state.sequence;
                return Err(self.lose_locked(
                    &mut state,
                    format!("rank did not arrive at rendezvous {sequence} before the deadline"),
                ));
            }
            let (next, wait) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && state.result.is_none() {
                let sequence = state.sequence;
                return Err(self.lose_locked(
                    &mut state,
                    format!("rank did not arrive at rendezvous {sequence} before the deadline"),
                ));
            }
        }
    }
}

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
            let [(_, first), (_, second)] = [&submitted[0], &submitted[1]];
            failure = self
                .agree(
                    sequence,
                    [first.declaration, second.declaration],
                    [
                        &first.lease.resource().local,
                        &second.lease.resource().local,
                    ],
                    streams,
                    cancel,
                )
                .err();
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
        declarations: [GatherDeclaration; 2],
        sources: [&DeviceRange<'ctx>; 2],
        streams: [&Stream<'ctx>; 2],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled { at: "all-gather" });
        }
        for rank in 0..2 {
            let uuid = self.ranks[rank].uuid();
            if sources[rank].device_uuid() != uuid || streams[rank].device_uuid() != uuid {
                return Err(collective(format!(
                    "rank {rank}'s shard or stream is not on its device {uuid}"
                )));
            }
            if declarations[rank].sequence != sequence {
                return Err(collective(format!(
                    "rank {rank} declared sequence {} but the group is at {sequence}",
                    declarations[rank].sequence
                )));
            }
        }
        // The sequence is each rank's agreement with the schedule, checked
        // above; the shape and precision are the ranks' agreement with each
        // other.
        let [d0, d1] = declarations;
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
        for shard in shards {
            let ready = shard
                .lease
                .completion()
                .expect("a submitted shard has an event");
            stream.wait_event(ready)?;
        }
        let locals = shards.map(|shard| &shard.lease.resource().local);
        copy_columns(declaration, locals, gathered, stream)?;
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

/// Concatenate two row-major `[rows, columns]` sources by column into
/// `gathered`, rank 0 first.
// ponytail: one peer copy per row and rank; a pitched peer copy
// (cuMemcpy3DPeerAsync) replaces them when row counts grow.
fn copy_columns<'ctx>(
    declaration: GatherDeclaration,
    sources: [&DeviceRange<'ctx>; 2],
    gathered: &DeviceRange<'ctx>,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let width = match declaration.precision {
        Precision::F32 => 4,
        _ => BF16_BYTES,
    };
    let part = declaration.columns * width;
    for (rank, source) in sources.into_iter().enumerate() {
        for r in 0..declaration.rows {
            // SAFETY: the caller retains every source, and its lease retains
            // `gathered`, until the copy's event is observed or withheld.
            unsafe {
                gathered.copy_from_peer_async_at(
                    r * part * 2 + rank as u64 * part,
                    source,
                    r * part,
                    part,
                    stream,
                )?;
            }
        }
    }
    Ok(())
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

#[cfg(all(test, feature = "paged-attention-binding"))]
mod tests {
    use super::{RankDeclaration, RankRendezvous};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use moxie_types::{Error, Precision};

    #[test]
    fn rank_one_device_loss_dominates_rank_zero_ordinary_error() {
        let rendezvous = Arc::new(RankRendezvous::new(1));
        let deadline = Instant::now() + Duration::from_secs(1);
        let rank_zero = {
            let rendezvous = Arc::clone(&rendezvous);
            thread::spawn(move || {
                rendezvous.enter(
                    0,
                    RankDeclaration {
                        sequence: 0,
                        operation: 1,
                        rank: 0,
                        shape: [0; 4],
                        precision: Precision::F32,
                    },
                    Err(Error::InvalidRequest {
                        field: "rank",
                        detail: "ordinary rank error".into(),
                    }),
                    deadline,
                )
            })
        };
        let rank_one = {
            let rendezvous = Arc::clone(&rendezvous);
            thread::spawn(move || {
                rendezvous.enter(
                    1,
                    RankDeclaration {
                        sequence: 0,
                        operation: 1,
                        rank: 1,
                        shape: [0; 4],
                        precision: Precision::F32,
                    },
                    Err(Error::DeviceLost {
                        device: 1,
                        detail: "rank one lost".into(),
                    }),
                    deadline,
                )
            })
        };
        let zero = rank_zero.join().unwrap().unwrap_err();
        let one = rank_one.join().unwrap().unwrap_err();
        assert!(matches!(zero, Error::DeviceLost { .. }));
        assert_eq!(one, zero);
        assert_eq!(rendezvous.lost(), Some(zero));
    }
}
