//! Performing an expert plan: bounded queue, placed host buffers, one kernel
//! per group, one reduction at the end.
//!
//! Every decision this file acts on was made in `moxie_plan::expert`. What is
//! here is the three effects a pure planner may not perform -- reserving the
//! envelope, holding residency leases while a kernel reads them, and binding a
//! thread so its allocations land on the right NUMA node -- plus the loop that
//! sequences them. Document 02 gives this crate "rank-local execution, events,
//! transfers, collectives, **work queues**", and the queue below is that noun.
//!
//! Three properties are mechanisms rather than intentions.
//!
//! **The queue is bounded and does not wait.** [`OrderQueue::push`] refuses with
//! `CapacityExceeded` when full; it never grows and never blocks. What it bounds
//! is how many experts are **pinned at once**, which is what makes a budget
//! smaller than the working set executable at all: a run that acquired every
//! group's leases first would need the whole layer resident before it computed
//! anything.
//!
//! **A refused acquire with work in the queue is backpressure, not a failure.**
//! When the residency authority cannot admit the next expert and something is
//! already queued, this drains instead of failing -- the queued group finishes,
//! its leases go back, and the acquire is retried once. When the queue is empty
//! the refusal is real and is propagated with the authority's own report.
//!
//! **Partial outputs are placed, not accumulated.** Every group writes distinct
//! slots of one slot-major buffer, so no sum can depend on which group finished
//! first, and the reduction then runs once over the plan's explicit per-row
//! permutation. This is the whole of task 0021's answer to "deterministic
//! reduction of partial outputs", and it is why nothing here adds into a shared
//! accumulator.

use std::collections::VecDeque;

use moxie_memory::{
    AcquireRequest, Acquired, AdmitError, ArtifactId, BufferRequest, ChunkId, Content, Ledger,
    LedgerId, LogicalRange, PlanRequest, Reservation, ResidencyAuthority, ResidencyLease,
    StageSpan, TensorSlot, TurnId, UseClass, WorkOrder,
};
use moxie_plan::expert::{Candidate, ExpertGroup, ExpertPlan, ExpertShape, Placement};
use moxie_types::{
    DeviceTier, Error, HostPlacement, HostTier, NumaTopology, Result, Scope, StrategyControl, Tier,
};

use crate::residency::{ChunkSource, drain_reads};

/// What a run needs from a device, stated as a trait so this file names no CUDA
/// type and compiles identically with and without the driver.
///
/// The implementation lives in `crate::grouped_device`, which is the only place
/// in this crate that may hold a device address. A host-lane double can also
/// implement it, which is how the device *sequencing* is exercised without
/// hardware while the device *arithmetic* is exercised only on hardware.
pub trait ExpertDeviceLane {
    /// Put the activation block on the device, once per run.
    fn load_activations(&mut self, x: &[u8]) -> Result<()>;
    /// Perform one upload order the residency authority issued, and report its
    /// outcome to the authority.
    fn perform_upload(
        &mut self,
        authority: &mut ResidencyAuthority,
        order: &WorkOrder,
    ) -> Result<()>;
    /// Run one group and read its slots back into `host_slots`.
    fn run_group(
        &mut self,
        authority: &ResidencyAuthority,
        group: &ExpertGroup,
        gate_up: &ResidencyLease,
        down: &ResidencyLease,
        host_slots: &mut [u8],
    ) -> Result<()>;
}

/// BF16 bytes per element.
const BF16: u64 = 2;
/// The smallest page this machine can place independently. Touching one byte in
/// every 4 KiB faults every page of a 4 KiB-page mapping and every page of a
/// larger one too, so it is a lower bound rather than an assumption.
const PAGE: usize = 4096;

/// Fault one page in, so its NUMA placement is decided by this thread.
fn first_touch(page: &mut [u8]) {
    let byte = &mut page[0];
    *byte = 1;
    std::hint::black_box(&mut *byte);
    *byte = 0;
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

// ---------------------------------------------------------------------------
// The bounded queue
// ---------------------------------------------------------------------------

/// One group, acquired and waiting to be performed.
#[derive(Debug)]
pub struct QueuedGroup {
    /// Index into [`ExpertPlan::groups`].
    pub index: usize,
    pub expert: u32,
    pub candidate: Candidate,
    gate_up: ResidencyLease,
    down: ResidencyLease,
}

impl QueuedGroup {
    pub const fn leases(&self) -> (&ResidencyLease, &ResidencyLease) {
        (&self.gate_up, &self.down)
    }
}

/// A fixed-capacity queue of acquired groups.
///
/// It refuses rather than grows and refuses rather than waits. A queue that
/// waited would reintroduce exactly the blocking path task 0020 removed from
/// `acquire`, and the review that found a cycle between that authority's demand
/// counter and its prefetch gate is why this is stated as a mechanism.
#[derive(Debug)]
pub struct OrderQueue {
    capacity: u32,
    entries: VecDeque<QueuedGroup>,
}

impl OrderQueue {
    pub fn with_capacity(capacity: u32) -> Result<Self> {
        if capacity == 0 {
            return Err(invalid(
                "capacity",
                "a queue of zero can never hold the group it is about to run".into(),
            ));
        }
        Ok(OrderQueue {
            capacity,
            entries: VecDeque::with_capacity(capacity as usize),
        })
    }

    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() as u32 >= self.capacity
    }

    /// Enqueue, or refuse. The refused group is handed back so its leases are
    /// never stranded by the error path -- the same rule `Lease::retire` and
    /// `ResidencyAuthority::release` already carry.
    pub fn push(&mut self, group: QueuedGroup) -> std::result::Result<(), QueueFull> {
        if self.is_full() {
            return Err(QueueFull {
                group,
                capacity: self.capacity,
            });
        }
        self.entries.push_back(group);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<QueuedGroup> {
        self.entries.pop_front()
    }
}

/// A refused enqueue, carrying the group back.
#[derive(Debug)]
#[must_use = "the group still holds its residency leases; drain and push it again"]
pub struct QueueFull {
    pub group: QueuedGroup,
    pub capacity: u32,
}

impl QueueFull {
    pub fn error(&self) -> Error {
        Error::CapacityExceeded {
            tier: Some(Tier::Device(DeviceTier::KernelWorkspace)),
            requested_bytes: u64::from(self.capacity) + 1,
            available_bytes: u64::from(self.capacity),
        }
    }
}

impl core::fmt::Display for QueueFull {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "the expert order queue holds its capacity of {}",
            self.capacity
        )
    }
}

// ---------------------------------------------------------------------------
// Chunk identity
// ---------------------------------------------------------------------------

/// Which artifact and which two canonical roles one routed layer's experts live
/// in.
///
/// The role names arrive as data. This crate may not learn what a model family
/// is, so the strings are the artifact's own, supplied by whoever opened it --
/// the same rule `ShardSource` already follows.
#[derive(Debug, Clone)]
pub struct ExpertRoles {
    pub artifact: ArtifactId,
    pub gate_up_role: String,
    pub down_role: String,
    pub format_version: u32,
}

impl ExpertRoles {
    /// The two chunks of one expert: its slice of the fused gate/up tensor and
    /// its slice of the fused down tensor.
    ///
    /// The arithmetic is task 0019's declared layout. It is here rather than in
    /// the plan because a `ChunkId` is `moxie-memory`'s vocabulary and
    /// `moxie-plan` may not name it.
    pub fn chunks(&self, expert: u32, shape: ExpertShape) -> Result<(ChunkId, ChunkId)> {
        if u64::from(expert) >= shape.experts {
            return Err(invalid(
                "expert",
                format!("expert {expert} of {}", shape.experts),
            ));
        }
        let gate_up_stride = shape
            .intermediate
            .checked_mul(2)
            .and_then(|v| v.checked_mul(shape.hidden))
            .and_then(|v| v.checked_mul(BF16))
            .ok_or_else(|| invalid("shape", "the gate/up stride overflows".into()))?;
        let down_stride = shape
            .hidden
            .checked_mul(shape.intermediate)
            .and_then(|v| v.checked_mul(BF16))
            .ok_or_else(|| invalid("shape", "the down stride overflows".into()))?;
        let e = u64::from(expert);
        let gate_up = ChunkId::new(
            self.artifact.clone(),
            TensorSlot::expert(self.gate_up_role.clone(), expert)?,
            LogicalRange::new(
                e.checked_mul(gate_up_stride)
                    .ok_or_else(|| invalid("shape", "the gate/up offset overflows".into()))?,
                gate_up_stride,
            )?,
            self.format_version,
        );
        let down = ChunkId::new(
            self.artifact.clone(),
            TensorSlot::expert(self.down_role.clone(), expert)?,
            LogicalRange::new(
                e.checked_mul(down_stride)
                    .ok_or_else(|| invalid("shape", "the down offset overflows".into()))?,
                down_stride,
            )?,
            self.format_version,
        );
        Ok((gate_up, down))
    }
}

// ---------------------------------------------------------------------------
// NUMA placement
// ---------------------------------------------------------------------------

/// What was actually done about placement, as distinct from what was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementReport {
    /// What the plan asked for.
    pub requested: HostPlacement,
    /// The CPUs the thread was bound to. Empty when nothing was bound.
    pub bound_cpus: Vec<u32>,
    /// Why nothing was bound, when nothing was. `None` when the binding
    /// happened.
    pub not_bound: Option<&'static str>,
    /// How the pages themselves were placed: `"mbind"` when the node was made
    /// binding, `"preferred"` when it was a hint the kernel may decline, and
    /// `"heap"` when this run does not own its pages and claims no placement.
    pub page_policy: &'static str,
}

impl PlacementReport {
    pub fn is_bound(&self) -> bool {
        self.not_bound.is_none() && !self.bound_cpus.is_empty()
    }
}

/// Bind this thread to a CPU set, and put it back on drop.
///
/// First touch is Linux's placement policy: a page lands on the node of the CPU
/// that first writes it. So placement here is binding the thread, allocating and
/// **writing** every page, and leaving the thread bound for the run so the
/// kernel that reads those pages runs beside them too.
///
/// It is `mbind`-free on purpose: `mbind` needs libnuma, and this needs one
/// glibc call the C library already exports.
#[cfg(feature = "numa")]
mod affinity {
    use moxie_types::{Error, Result};

    /// `cpu_set_t` is a fixed 1024-bit mask in glibc. Sixteen `u64` is that,
    /// exactly, and the kernel is given the size in bytes either way.
    const WORDS: usize = 16;

    // SAFETY (declaration): both are plain glibc entry points with C linkage.
    // `sched_getaffinity` writes at most `cpusetsize` bytes through `mask` and
    // `sched_setaffinity` reads at most that many; every call below passes
    // `size_of::<[u64; WORDS]>()` for a `[u64; WORDS]` it owns, so neither can
    // reach past the buffer. Both return `0` or `-1` and never unwind.
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }

    const SIZE: usize = core::mem::size_of::<[u64; WORDS]>();

    fn get() -> Result<[u64; WORDS]> {
        let mut mask = [0u64; WORDS];
        // SAFETY: `mask` is a live, owned `[u64; WORDS]` and `SIZE` is its exact
        // byte length, so the kernel writes only inside it. `0` means "this
        // thread".
        let rc = unsafe { sched_getaffinity(0, SIZE, mask.as_mut_ptr()) };
        if rc != 0 {
            return Err(Error::InvalidRequest {
                field: "affinity",
                detail: format!(
                    "sched_getaffinity failed: {}",
                    std::io::Error::last_os_error()
                ),
            });
        }
        Ok(mask)
    }

    fn set(mask: &[u64; WORDS]) -> Result<()> {
        // SAFETY: `mask` is a live, owned `[u64; WORDS]` and `SIZE` is its exact
        // byte length, so the kernel reads only inside it. `0` means "this
        // thread".
        let rc = unsafe { sched_setaffinity(0, SIZE, mask.as_ptr()) };
        if rc != 0 {
            return Err(Error::InvalidRequest {
                field: "affinity",
                detail: format!(
                    "sched_setaffinity failed: {}",
                    std::io::Error::last_os_error()
                ),
            });
        }
        Ok(())
    }

    /// Restores the previous affinity when dropped.
    ///
    /// Restoring in `Drop` is safe in a way freeing device memory in `Drop` is
    /// not: nothing asynchronous depends on a thread's CPU set, and a scheduler
    /// mask that outlived the run would quietly narrow every later thread of
    /// this process.
    #[derive(Debug)]
    pub struct AffinityGuard {
        previous: [u64; WORDS],
    }

    impl AffinityGuard {
        /// Bind to `cpus`, returning the guard and the CPUs actually set.
        pub fn bind(cpus: &[u32]) -> Result<(Self, Vec<u32>)> {
            let previous = get()?;
            let mut mask = [0u64; WORDS];
            let mut applied = Vec::with_capacity(cpus.len());
            for cpu in cpus {
                let word = (*cpu as usize) / 64;
                if word >= WORDS {
                    return Err(Error::InvalidRequest {
                        field: "affinity",
                        detail: format!("CPU {cpu} is outside the {} CPU mask", WORDS * 64),
                    });
                }
                mask[word] |= 1u64 << ((*cpu as usize) % 64);
                applied.push(*cpu);
            }
            if applied.is_empty() {
                return Err(Error::InvalidRequest {
                    field: "affinity",
                    detail: "an empty CPU set would leave the thread unrunnable".into(),
                });
            }
            set(&mask)?;
            Ok((AffinityGuard { previous }, applied))
        }
    }

    impl Drop for AffinityGuard {
        fn drop(&mut self) {
            // Best effort by necessity: `Drop` cannot report. A failure here
            // leaves the thread on the node it was working on, which is a
            // performance fact and not a correctness one.
            let _ = set(&self.previous);
        }
    }
}

#[cfg(not(feature = "numa"))]
mod affinity {
    use moxie_types::{Error, Result};

    #[derive(Debug)]
    pub struct AffinityGuard;

    impl AffinityGuard {
        pub fn bind(_cpus: &[u32]) -> Result<(Self, Vec<u32>)> {
            Err(Error::InvalidRequest {
                field: "affinity",
                detail: "this build has no `numa` feature, so nothing can be bound".into(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Host buffers
// ---------------------------------------------------------------------------

/// Anonymous private pages this run owns outright.
///
/// A placed buffer cannot come from the general allocator, and that is a
/// measured fact rather than a preference. First touch places a page the first
/// time **anything** writes it, so a chunk the allocator has already faulted is
/// already placed; glibc raises its dynamic `mmap` threshold after a large
/// mapped block is freed, and then serves the next large request from the brk
/// heap. Measured here, on this machine: a 32 MiB buffer allocated that way,
/// with the thread bound to node 1's CPUs and every page written, came back
/// **3,317 pages on node 0** against 4,877 on node 1 -- the node-0 pages having
/// been faulted minutes earlier by an unrelated allocation.
///
/// Owning the mapping fixes both halves of that: the pages are certainly fresh,
/// so this thread's write is certainly the first, and the mapping is certainly
/// this run's, so reading `numa_maps` back measures these buffers and nothing
/// else.
#[cfg(feature = "numa")]
mod mapped {
    use moxie_types::{Error, Result};

    // SAFETY (declaration): the two plain POSIX entry points, with C linkage.
    // `mmap` with a null address, `MAP_PRIVATE | MAP_ANONYMOUS` and `fd = -1`
    // reads nothing through a pointer; `munmap` is called exactly once per
    // successful mapping, from `Drop`, with that mapping's own base and length.
    // Neither unwinds.
    unsafe extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
        fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
    }

    // SAFETY (declaration): glibc's variadic syscall gate. It is used below for
    // exactly one call, `mbind`, with the six arguments that call takes; every
    // pointer passed is a live object of the stated length. It returns a
    // `c_long` and never unwinds.
    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" {
        fn syscall(number: core::ffi::c_long, ...) -> core::ffi::c_long;
    }

    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    /// `__NR_mbind` on x86-64. The number is architecture-specific, so this
    /// module supports exactly the architecture it was checked on; anywhere else
    /// [`NodePolicy`] is refused rather than guessed at.
    #[cfg(target_arch = "x86_64")]
    const SYS_MBIND: core::ffi::c_long = 237;
    const MPOL_PREFERRED: core::ffi::c_ulong = 1;
    const MPOL_BIND: core::ffi::c_ulong = 2;

    /// What to ask the kernel for, and how hard.
    ///
    /// The distinction is document 01's `auto` / `required` rule at the page
    /// level, and on this machine it is not academic. Node 1 has **334 MB**
    /// free against node 0's **5.1 GB**, both nodes holding about 125 GB of
    /// page cache, so the default policy -- prefer local, fall back rather than
    /// reclaim -- puts a large share of a "locally" touched buffer on the other
    /// socket. Measured here: a 32 MiB buffer, thread bound to node 1, came back
    /// with 4,471 of 6,144 pages on node 0.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NodePolicy {
        /// `MPOL_PREFERRED`: name the node explicitly; still a hint the kernel
        /// may decline when that node is short.
        Preferred(u32),
        /// `MPOL_BIND`: allocate on this node or reclaim on it. No silent
        /// fallback to another socket, which is what `required` means.
        Bind(u32),
    }

    /// One owned anonymous mapping. Zero-length mappings are not representable:
    /// `mmap` refuses them, and a buffer of nothing is a caller error.
    pub struct Mapping {
        ptr: *mut u8,
        len: usize,
    }

    impl core::fmt::Debug for Mapping {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "Mapping({} B)", self.len)
        }
    }

    impl Mapping {
        pub fn new(len: usize, policy: NodePolicy) -> Result<Self> {
            if len == 0 {
                return Err(Error::InvalidRequest {
                    field: "mapping",
                    detail: "a zero-length mapping".into(),
                });
            }
            // SAFETY: a null hint lets the kernel choose the address; the
            // mapping is private and anonymous, so `fd` and `offset` are
            // unused and passed as the POSIX-required `-1` and `0`.
            let ptr = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    len,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr.is_null() || ptr as isize == -1 {
                return Err(Error::InvalidRequest {
                    field: "mapping",
                    detail: format!(
                        "mmap of {len} B failed: {}",
                        std::io::Error::last_os_error()
                    ),
                });
            }
            let mapping = Mapping {
                ptr: ptr.cast::<u8>(),
                len,
            };
            // Before a single page is faulted: a policy applied afterwards
            // would leave every already-placed page where it is.
            mapping.apply(policy)?;
            Ok(mapping)
        }

        #[cfg(target_arch = "x86_64")]
        fn apply(&self, policy: NodePolicy) -> Result<()> {
            let (mode, node) = match policy {
                NodePolicy::Preferred(node) => (MPOL_PREFERRED, node),
                NodePolicy::Bind(node) => (MPOL_BIND, node),
            };
            if node >= 64 {
                return Err(Error::InvalidRequest {
                    field: "node",
                    detail: format!("node {node} is outside this 64-node mask"),
                });
            }
            let mask: u64 = 1u64 << node;
            // `maxnode` counts the bits the mask covers, and the kernel requires
            // room for one more than the highest node it may name.
            let maxnode: core::ffi::c_ulong = 64;
            // SAFETY: `SYS_MBIND`'s six arguments, in order: the mapping's own
            // base and length, the mode, a pointer to one live `u64` nodemask
            // and the bit count it covers, and no flags. The kernel reads
            // `maxnode / 8` bytes through the mask pointer, which is exactly the
            // eight bytes `mask` owns.
            let rc = unsafe {
                syscall(
                    SYS_MBIND,
                    self.ptr as *mut core::ffi::c_void,
                    self.len,
                    mode,
                    &mask as *const u64,
                    maxnode,
                    0 as core::ffi::c_ulong,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidRequest {
                    field: "mbind",
                    detail: format!(
                        "mbind of {} B to node {node} failed: {}",
                        self.len,
                        std::io::Error::last_os_error()
                    ),
                });
            }
            Ok(())
        }

        /// Anywhere but x86-64, a named node policy is refused rather than
        /// guessed: `mbind`'s syscall number is architecture-specific and this
        /// module has been checked on one architecture.
        #[cfg(not(target_arch = "x86_64"))]
        fn apply(&self, policy: NodePolicy) -> Result<()> {
            Err(Error::InvalidRequest {
                field: "mbind",
                detail: format!("{policy:?} needs mbind, which is x86-64 only here"),
            })
        }

        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: `ptr` is a live mapping of `len` readable bytes, owned by
            // `self`, and the borrow cannot outlive it.
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }

        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: as above, and `&mut self` makes this the only borrow.
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }

        /// The mapping as `f32`. `mmap` returns page-aligned memory, so the
        /// alignment requirement is satisfied by construction; the caller
        /// guarantees `len` is a multiple of four by asking for one.
        pub fn as_mut_f32(&mut self) -> &mut [f32] {
            // SAFETY: page-aligned base, so aligned for `f32`; `len / 4`
            // elements lie entirely inside the mapping; `&mut self` makes this
            // the only borrow. Every byte pattern is a valid `f32`.
            unsafe { core::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), self.len / 4) }
        }

        pub fn as_ptr(&self) -> *const u8 {
            self.ptr
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            // SAFETY: exactly the base and length this mapping was created
            // with, unmapped once. Unlike device memory, nothing asynchronous
            // can still be reading host pages here: this crate's host path is
            // synchronous, and the device path copies back and waits on the
            // copy's event before the run can be dropped.
            unsafe {
                munmap(self.ptr.cast::<core::ffi::c_void>(), self.len);
            }
        }
    }
}

#[cfg(feature = "numa")]
use mapped::NodePolicy;

/// The policy vocabulary still exists without the `numa` feature, so the rest of
/// this file has one shape. Nothing can be placed, so nothing constructs one.
#[cfg(not(feature = "numa"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodePolicy {
    Preferred(u32),
    Bind(u32),
}

/// A buffer whose pages this run either owns or borrows from the allocator.
#[derive(Debug)]
enum Placed {
    /// Pages this run mapped itself. Placement is this run's to claim.
    #[cfg(feature = "numa")]
    Mapped(mapped::Mapping),
    /// Ordinary heap. Whatever node these pages are on, they were placed by
    /// whoever faulted them first, which may have been another allocation
    /// entirely. Nothing here claims otherwise.
    Heap(Vec<u8>),
}

impl Placed {
    fn allocate(len: usize, policy: Option<NodePolicy>) -> Result<Self> {
        #[cfg(feature = "numa")]
        if let Some(policy) = policy
            && len > 0
        {
            return Ok(Placed::Mapped(mapped::Mapping::new(len, policy)?));
        }
        let _ = policy;
        Ok(Placed::Heap(vec![0u8; len]))
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "numa")]
            Placed::Mapped(m) => m.as_slice(),
            Placed::Heap(v) => v,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            #[cfg(feature = "numa")]
            Placed::Mapped(m) => m.as_mut_slice(),
            Placed::Heap(v) => v,
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            #[cfg(feature = "numa")]
            Placed::Mapped(m) => m.as_ptr(),
            Placed::Heap(v) => v.as_ptr(),
        }
    }

    /// `true` when this run owns the pages, and therefore when a placement
    /// measurement over them means anything.
    fn owns_pages(&self) -> bool {
        match self {
            #[cfg(feature = "numa")]
            Placed::Mapped(_) => true,
            Placed::Heap(_) => false,
        }
    }
}

/// The FP32 tile workspace, in the same two flavours.
#[derive(Debug)]
enum PlacedF32 {
    #[cfg(feature = "numa")]
    Mapped(mapped::Mapping),
    Heap(Vec<f32>),
}

impl PlacedF32 {
    fn allocate(values: usize, policy: Option<NodePolicy>) -> Result<Self> {
        #[cfg(feature = "numa")]
        if let Some(policy) = policy
            && values > 0
        {
            return Ok(PlacedF32::Mapped(mapped::Mapping::new(values * 4, policy)?));
        }
        let _ = policy;
        Ok(PlacedF32::Heap(vec![0f32; values]))
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        match self {
            #[cfg(feature = "numa")]
            PlacedF32::Mapped(m) => m.as_mut_f32(),
            PlacedF32::Heap(v) => v,
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            #[cfg(feature = "numa")]
            PlacedF32::Mapped(m) => m.as_ptr(),
            PlacedF32::Heap(v) => v.as_ptr().cast::<u8>(),
        }
    }
}

/// The host side of one run: the activation block, the slot buffer, the reduced
/// output and the FP32 tile workspace.
#[derive(Debug)]
pub struct HostBuffers {
    x: Placed,
    slots: Placed,
    out: Placed,
    workspace: PlacedF32,
}

impl HostBuffers {
    /// Allocate and **first-touch** every page, so placement is decided now
    /// rather than by whichever thread happens to write first.
    fn allocate(
        x: usize,
        slots: usize,
        out: usize,
        workspace: usize,
        policy: Option<NodePolicy>,
    ) -> Result<Self> {
        let mut buffers = HostBuffers {
            x: Placed::allocate(x, policy)?,
            slots: Placed::allocate(slots, policy)?,
            out: Placed::allocate(out, policy)?,
            workspace: PlacedF32::allocate(workspace, policy)?,
        };
        buffers.touch();
        Ok(buffers)
    }

    fn touch(&mut self) {
        // A fresh anonymous mapping has no pages until something writes one, so
        // placement is decided by this write, on the bound thread.
        //
        // Writing zero is not enough, and this is not a theoretical point: the
        // buffer is known-zero, so the compiler may delete the store. Writing a
        // non-zero byte through a `black_box` and clearing it makes the fault
        // unavoidable.
        for page in self.x.as_mut_slice().chunks_mut(PAGE) {
            first_touch(page);
        }
        for page in self.slots.as_mut_slice().chunks_mut(PAGE) {
            first_touch(page);
        }
        for page in self.out.as_mut_slice().chunks_mut(PAGE) {
            first_touch(page);
        }
        for page in self.workspace.as_mut_slice().chunks_mut(PAGE / 4) {
            let value = &mut page[0];
            *value = 1.0;
            std::hint::black_box(&mut *value);
            *value = 0.0;
        }
    }

    pub fn slots(&self) -> &[u8] {
        self.slots.as_slice()
    }

    pub fn output(&self) -> &[u8] {
        self.out.as_slice()
    }

    /// Whether this run owns its pages, and therefore whether a placement
    /// measurement over [`HostBuffers::addresses`] means anything.
    pub fn owns_pages(&self) -> bool {
        self.x.owns_pages()
    }

    /// The addresses a placement read-back is taken at. Reading `numa_maps` is
    /// `moxie-host`'s alone (ADR 0006), so this hands out the addresses and the
    /// measurement happens where it is permitted.
    pub fn addresses(&self) -> BufferAddresses {
        BufferAddresses {
            activations: self.x.as_ptr() as u64,
            slots: self.slots.as_ptr() as u64,
            output: self.out.as_ptr() as u64,
            workspace: self.workspace.as_ptr() as u64,
        }
    }
}

/// Where this run's host buffers are, for a placement measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferAddresses {
    pub activations: u64,
    pub slots: u64,
    pub output: u64,
    pub workspace: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// What a run did. Counts only: document 03 requires these to be recorded, and
/// **none of them is a performance measurement** -- there is no baseline on this
/// machine to compare any of them against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupedStats {
    pub groups_run: u64,
    pub host_groups: u64,
    pub device_groups: u64,
    pub slots_written: u64,
    /// Acquires refused while the queue held work, which drained and retried.
    pub backpressure_drains: u64,
    pub leases_released: u64,
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// How far a run has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// One group ran.
    Ran { expert: u32, candidate: Candidate },
    /// Every group has run. The slot buffer is complete.
    Done,
}

/// One admitted, placed, executable expert plan.
#[derive(Debug)]
#[must_use = "an unclosed run keeps its reservation charged"]
pub struct GroupedRun {
    plan: ExpertPlan,
    roles: ExpertRoles,
    reservation: Option<Reservation>,
    ledger: LedgerId,
    buffers: HostBuffers,
    queue: OrderQueue,
    placement: PlacementReport,
    guard: Option<affinity::AffinityGuard>,
    next_group: usize,
    stats: GroupedStats,
    cancelled: bool,
    /// True once a device attachment took the reservation. The charge is then
    /// that attachment's to release, and this run's `close` must not release it
    /// a second time.
    detached: bool,
}

impl GroupedRun {
    /// Reserve the plan's envelope atomically, then place and allocate the host
    /// buffers.
    ///
    /// In that order, and the order is the assertion: the ledger is charged
    /// before a byte exists. Task 0020's device cache was a placement simulator
    /// until a test asserted exactly this, and the same test exists here.
    pub fn admit(
        ledger: &mut Ledger,
        plan: ExpertPlan,
        roles: ExpertRoles,
        topology: Option<&NumaTopology>,
    ) -> std::result::Result<Self, GroupedAdmitRefused> {
        Self::admit_inner(ledger, plan, roles, topology)
    }

    fn admit_inner(
        ledger: &mut Ledger,
        plan: ExpertPlan,
        roles: ExpertRoles,
        topology: Option<&NumaTopology>,
    ) -> std::result::Result<Self, GroupedAdmitRefused> {
        macro_rules! invalid_out {
            ($plan:expr, $error:expr) => {
                return Err(GroupedAdmitRefused::Invalid {
                    plan: Box::new($plan),
                    error: $error,
                })
            };
        }
        let request = match Self::request_for(&plan) {
            Ok(request) => request,
            Err(error) => invalid_out!(plan, error),
        };
        let reservation = match ledger.admit(&request) {
            Ok(reservation) => reservation,
            Err(AdmitError::Invalid(error)) => invalid_out!(plan, error),
            Err(AdmitError::Rejected(rejection)) => {
                return Err(GroupedAdmitRefused::Rejected {
                    plan: Box::new(plan),
                    rejection,
                });
            }
        };
        let ledger_id = reservation.ledger();

        // Placement, then allocation. Binding after allocating would place the
        // pages by whichever CPU happened to run the allocation.
        let (guard, placement, page_policy) = Self::bind(&plan, topology);
        let extents = Self::host_extents(&plan);
        let queue = OrderQueue::with_capacity(plan.queue_capacity());
        let (extents, queue) = match (extents, queue) {
            (Ok(extents), Ok(queue)) => (extents, queue),
            (Err(error), _) | (_, Err(error)) => {
                // The reservation is handed back before anything else, so a
                // failure between `admit` and the first byte cannot leave the
                // ledger charged for a run that never existed.
                let _ = ledger.release(reservation);
                invalid_out!(plan, error)
            }
        };
        let buffers =
            match HostBuffers::allocate(extents.0, extents.1, extents.2, extents.3, page_policy) {
                Ok(buffers) => buffers,
                Err(error) => {
                    let _ = ledger.release(reservation);
                    invalid_out!(plan, error)
                }
            };
        Ok(GroupedRun {
            plan,
            roles,
            reservation: Some(reservation),
            ledger: ledger_id,
            buffers,
            queue,
            placement,
            guard,
            next_group: 0,
            stats: GroupedStats::default(),
            cancelled: false,
            detached: false,
        })
    }

    fn bind(
        plan: &ExpertPlan,
        topology: Option<&NumaTopology>,
    ) -> (
        Option<affinity::AffinityGuard>,
        PlacementReport,
        Option<NodePolicy>,
    ) {
        let requested = plan.host_placement();
        let strict = plan.policy().host_placement == StrategyControl::Required;
        let unbound = |reason: &'static str| {
            (
                None,
                PlacementReport {
                    requested,
                    bound_cpus: Vec::new(),
                    not_bound: Some(reason),
                    page_policy: "heap",
                },
                None,
            )
        };
        if plan.policy().host_placement == StrategyControl::Off {
            return unbound("the placement control is off");
        }
        let Some(node) = requested.node() else {
            return unbound("the plan placed host buffers on no node");
        };
        let Some(topology) = topology else {
            return unbound("no topology was supplied to the run");
        };
        let Some(cpus) = topology.cpus_of(node) else {
            return unbound("the topology has no CPU list for the planned node");
        };
        // Two mechanisms, and they answer different questions. The affinity
        // binding decides which CPUs run the kernel that reads these pages; the
        // page policy decides which node the pages are on. On this machine the
        // second cannot be left to the first: node 1's free memory is an order
        // of magnitude below node 0's, and the default policy falls back rather
        // than reclaiming.
        let policy = if strict {
            NodePolicy::Bind(node.get())
        } else {
            NodePolicy::Preferred(node.get())
        };
        match affinity::AffinityGuard::bind(cpus) {
            Ok((guard, bound_cpus)) => (
                Some(guard),
                PlacementReport {
                    requested,
                    bound_cpus,
                    not_bound: None,
                    page_policy: if strict { "mbind" } else { "preferred" },
                },
                Some(policy),
            ),
            Err(_) => unbound("the thread could not be bound"),
        }
    }

    pub const fn plan(&self) -> &ExpertPlan {
        &self.plan
    }
    pub const fn placement(&self) -> &PlacementReport {
        &self.placement
    }
    pub const fn stats(&self) -> GroupedStats {
        self.stats
    }
    pub const fn buffers(&self) -> &HostBuffers {
        &self.buffers
    }
    pub const fn queue(&self) -> &OrderQueue {
        &self.queue
    }
    pub const fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Load the activation block. `[rows, hidden]` BF16, exactly.
    pub fn load_activations(&mut self, x: &[u8]) -> Result<()> {
        if x.len() as u64 != self.plan.activation_bytes() {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "activations are {} B, expected {} for [{}, {}] BF16",
                    x.len(),
                    self.plan.activation_bytes(),
                    self.plan.rows(),
                    self.plan.shape().hidden
                ),
            });
        }
        self.buffers.x.as_mut_slice().copy_from_slice(x);
        Ok(())
    }

    /// Load the activation block and put it on the device too.
    pub fn load_activations_with(
        &mut self,
        x: &[u8],
        lane: &mut dyn ExpertDeviceLane,
    ) -> Result<()> {
        self.load_activations(x)?;
        lane.load_activations(self.buffers.x.as_slice())
    }
}

fn too_large(what: &'static str) -> Error {
    Error::InvalidRequest {
        field: "buffer",
        detail: format!("the {what} buffer does not fit this machine's address space"),
    }
}

/// One group's acquire, as a single argument.
#[derive(Debug, Clone, Copy)]
struct GroupAcquire {
    shape: ExpertShape,
    expert: u32,
    scope: Scope,
    turn: TurnId,
    now: u64,
    deadline: u64,
}

/// A failed acquire, carrying back every lease it had already taken.
///
/// The error path may not strand a pin. Task 0020's `ReleaseLeaseRefused` and
/// this crate's `retire` refusals carry their handle back for the same reason:
/// a lease destroyed by a failure is a chunk pinned forever.
#[derive(Debug)]
#[must_use = "the held leases are still pinned; release them"]
struct AcquireFailed {
    error: Error,
    held: Vec<ResidencyLease>,
}

/// Acquire one group's two chunks into `scope`.
fn acquire_pair<S: ChunkSource>(
    authority: &mut ResidencyAuthority,
    source: &mut S,
    roles: &ExpertRoles,
    what: GroupAcquire,
    mut lane: Option<&mut (dyn ExpertDeviceLane + '_)>,
) -> std::result::Result<(ResidencyLease, ResidencyLease), AcquireFailed> {
    let GroupAcquire {
        shape,
        expert,
        scope,
        turn,
        now,
        deadline,
    } = what;
    let (gate_up_chunk, down_chunk) = match roles.chunks(expert, shape) {
        Ok(pair) => pair,
        Err(error) => {
            return Err(AcquireFailed {
                error,
                held: Vec::new(),
            });
        }
    };
    let class = UseClass::demand(Content::Expert);
    let mut held: Vec<ResidencyLease> = Vec::with_capacity(2);
    for chunk in [&gate_up_chunk, &down_chunk] {
        let request = AcquireRequest {
            chunk,
            destination: scope,
            now,
            deadline,
            class,
            turn,
        };
        let acquired = match authority.acquire(request) {
            Ok(acquired) => acquired,
            Err(refused) => {
                return Err(AcquireFailed {
                    error: refused.error,
                    held,
                });
            }
        };
        match acquired {
            Acquired::Ready(lease) => held.push(lease),
            Acquired::Pending { lease, work, .. } => {
                held.push(lease);
                let uploads = match drain_reads(authority, source, work) {
                    Ok(uploads) => uploads,
                    Err(error) => return Err(AcquireFailed { error, held }),
                };
                for order in &uploads {
                    let Some(lane) = lane.as_deref_mut() else {
                        // A host destination produces no upload order. Reaching
                        // one with no device lane is a wiring mistake rather
                        // than a capacity one, and saying so beats a copy that
                        // silently never happens.
                        return Err(AcquireFailed {
                            error: invalid(
                                "scope",
                                "an upload order was issued with no device lane to perform it"
                                    .into(),
                            ),
                            held,
                        });
                    };
                    if let Err(error) = lane.perform_upload(authority, order) {
                        return Err(AcquireFailed { error, held });
                    }
                }
            }
        }
    }
    let down = held.pop().expect("two leases");
    let gate_up = held.pop().expect("two leases");
    Ok((gate_up, down))
}

impl GroupedRun {
    /// Perform at most one group, filling the bounded queue first.
    ///
    /// The loop, in full: enqueue acquired groups until the queue is full or the
    /// plan is exhausted, then perform exactly one. An acquire the residency
    /// authority refuses while the queue holds work is **backpressure** -- the
    /// queued group runs, gives its leases back, and the acquire is retried on
    /// the next call. An acquire refused with an empty queue is a real refusal
    /// and is propagated with the authority's own error.
    pub fn step<S: ChunkSource>(
        &mut self,
        authority: &mut ResidencyAuthority,
        source: &mut S,
        turn: TurnId,
        now: u64,
        deadline: u64,
    ) -> Result<Progress> {
        self.step_with(authority, source, None, turn, now, deadline)
    }

    /// The same step, with a device lane for the groups the plan put there.
    pub fn step_with<S: ChunkSource>(
        &mut self,
        authority: &mut ResidencyAuthority,
        source: &mut S,
        mut lane: Option<&mut (dyn ExpertDeviceLane + '_)>,
        turn: TurnId,
        now: u64,
        deadline: u64,
    ) -> Result<Progress> {
        if self.cancelled {
            return Err(Error::Cancelled { at: "expert-run" });
        }
        while self.next_group < self.plan.groups().len() && !self.queue.is_full() {
            let index = self.next_group;
            let group = &self.plan.groups()[index];
            let scope = match group.placement {
                Placement::Device(uuid) => Scope::Device(uuid),
                Placement::Host(_) => Scope::Host,
            };
            let expert = group.expert;
            let candidate = group.placement.candidate();
            match acquire_pair(
                authority,
                source,
                &self.roles,
                GroupAcquire {
                    shape: self.plan.shape(),
                    expert,
                    scope,
                    turn,
                    now,
                    deadline,
                },
                lane.as_deref_mut(),
            ) {
                Ok((gate_up, down)) => {
                    let queued = QueuedGroup {
                        index,
                        expert,
                        candidate,
                        gate_up,
                        down,
                    };
                    match self.queue.push(queued) {
                        Ok(()) => self.next_group += 1,
                        Err(full) => {
                            // Unreachable by construction: the loop condition
                            // already checked `is_full`. It is written out
                            // rather than asserted because the failure mode of
                            // getting it wrong is a stranded residency lease,
                            // and an error return that gives the leases back
                            // costs nothing while a panic here would strand
                            // them. The queue's refusal itself is a real,
                            // separately tested behaviour of `OrderQueue`; what
                            // is unreachable is this call site reaching it.
                            let error = full.error();
                            let (gate_up, down) = (full.group.gate_up, full.group.down);
                            self.release_pair(authority, gate_up, down);
                            return Err(error);
                        }
                    }
                }
                Err(failed) => {
                    for lease in failed.held {
                        self.release_one(authority, lease);
                    }
                    if self.queue.is_empty() {
                        return Err(failed.error);
                    }
                    // Backpressure: the cache cannot hold another expert while
                    // this one is pinned. Drain and try again next call.
                    self.stats.backpressure_drains += 1;
                    break;
                }
            }
        }
        let Some(queued) = self.queue.pop() else {
            return Ok(Progress::Done);
        };
        let expert = queued.expert;
        let candidate = queued.candidate;
        let result = self.perform(authority, &queued, lane);
        self.release_pair(authority, queued.gate_up, queued.down);
        result?;
        self.stats.groups_run += 1;
        match candidate {
            Candidate::Host => self.stats.host_groups += 1,
            Candidate::Device => self.stats.device_groups += 1,
        }
        Ok(Progress::Ran { expert, candidate })
    }

    /// Run every group, then reduce. The bounded queue still governs how many
    /// experts are pinned at once.
    pub fn run_to_completion<S: ChunkSource>(
        &mut self,
        authority: &mut ResidencyAuthority,
        source: &mut S,
        turn: TurnId,
        now: u64,
        deadline: u64,
    ) -> Result<()> {
        self.run_to_completion_with(authority, source, None, turn, now, deadline)
    }

    /// Run every group, with a device lane for the groups the plan put there.
    pub fn run_to_completion_with<S: ChunkSource>(
        &mut self,
        authority: &mut ResidencyAuthority,
        source: &mut S,
        mut lane: Option<&mut (dyn ExpertDeviceLane + '_)>,
        turn: TurnId,
        now: u64,
        deadline: u64,
    ) -> Result<()> {
        loop {
            match self.step_with(authority, source, lane.as_deref_mut(), turn, now, deadline)? {
                Progress::Done => return Ok(()),
                Progress::Ran { .. } => {}
            }
        }
    }

    fn perform(
        &mut self,
        authority: &ResidencyAuthority,
        queued: &QueuedGroup,
        lane: Option<&mut (dyn ExpertDeviceLane + '_)>,
    ) -> Result<()> {
        let group = &self.plan.groups()[queued.index];
        match queued.candidate {
            Candidate::Host => {
                let gate_up = authority.chunk_bytes(&queued.gate_up)?;
                let down = authority.chunk_bytes(&queued.down)?;
                let shape = self.plan.shape();
                moxie_kernels::cpu_expert::expert_group_bf16(
                    self.buffers.x.as_slice(),
                    moxie_kernels::cpu_expert::ExpertAssignment {
                        rows: &group.rows,
                        slots: &group.slots,
                    },
                    gate_up,
                    down,
                    shape.gate_transform(),
                    moxie_kernels::cpu_expert::ExpertShape {
                        hidden: u32::try_from(shape.hidden)
                            .map_err(|_| invalid("hidden", "hidden exceeds u32".into()))?,
                        intermediate: u32::try_from(shape.intermediate).map_err(|_| {
                            invalid("intermediate", "intermediate exceeds u32".into())
                        })?,
                    },
                    moxie_kernels::cpu_expert::ExpertTiling::lanes(self.plan.cpu_tile_lanes()),
                    self.buffers.workspace.as_mut_slice(),
                    self.buffers.slots.as_mut_slice(),
                )?;
                self.stats.slots_written += group.slots.len() as u64;
                Ok(())
            }
            Candidate::Device => {
                let Some(lane) = lane else {
                    return Err(Error::UnsupportedKernel {
                        operation: "expert_mlp",
                        detail: "this plan has device groups and no device lane to run them".into(),
                    });
                };
                lane.run_group(
                    authority,
                    group,
                    &queued.gate_up,
                    &queued.down,
                    self.buffers.slots.as_mut_slice(),
                )?;
                self.stats.slots_written += group.slots.len() as u64;
                Ok(())
            }
        }
    }

    fn release_one(&mut self, authority: &mut ResidencyAuthority, lease: ResidencyLease) {
        match authority.release(lease) {
            Ok(()) => self.stats.leases_released += 1,
            Err(refused) => {
                // The lease comes back rather than being destroyed. Dropping it
                // here is the one thing that would strand the pin, so it is
                // forgotten deliberately and stays visible in
                // `ResidencyAuthority::outstanding`.
                drop(refused);
            }
        }
    }

    fn release_pair(
        &mut self,
        authority: &mut ResidencyAuthority,
        gate_up: ResidencyLease,
        down: ResidencyLease,
    ) {
        self.release_one(authority, gate_up);
        self.release_one(authority, down);
    }

    /// Hand the envelope's reservation to a device attachment.
    ///
    /// A `DeviceArena` owns the charge it materialises -- that is how task 0010
    /// made "simulated placement presented as a reservation" impossible to
    /// write -- so the reservation has to move rather than be borrowed. After
    /// this, [`GroupedRun::close`] releases nothing and the attachment's own
    /// `close` is what gives the envelope back.
    pub fn detach_reservation(&mut self) -> Result<Reservation> {
        let reservation = self.reservation.take().ok_or_else(|| {
            invalid(
                "reservation",
                "this run has no reservation to hand over".into(),
            )
        })?;
        self.detached = true;
        Ok(reservation)
    }

    /// Reduce every row's slots in the plan's declared order.
    ///
    /// `coefficients` is `[rows * top_k]`, slot-major: the route's own weights.
    /// Every slot must already have been written, which
    /// [`GroupedRun::run_to_completion`] guarantees and which this checks by
    /// refusing to reduce a run that has groups left.
    pub fn reduce(&mut self, coefficients: &[f32]) -> Result<&[u8]> {
        if self.cancelled {
            return Err(Error::Cancelled { at: "expert-run" });
        }
        if self.next_group < self.plan.groups().len() || !self.queue.is_empty() {
            return Err(invalid(
                "reduce",
                format!(
                    "{} group(s) have not run and {} are queued; a row reduced over fewer slots \
                     than it selected is a wrong answer, not a partial one",
                    self.plan.groups().len() - self.next_group,
                    self.queue.len()
                ),
            ));
        }
        let shape = self.plan.shape();
        let (slots, out) = (&self.buffers.slots, &mut self.buffers.out);
        moxie_kernels::cpu_expert::combine_rows_bf16(
            slots.as_slice(),
            coefficients,
            self.plan.reduction_order(),
            u32::try_from(self.plan.rows())
                .map_err(|_| invalid("rows", "rows exceed u32".into()))?,
            u32::try_from(shape.top_k).map_err(|_| invalid("top_k", "top_k exceeds u32".into()))?,
            u32::try_from(shape.hidden)
                .map_err(|_| invalid("hidden", "hidden exceeds u32".into()))?,
            out.as_mut_slice(),
        )?;
        Ok(self.buffers.out.as_slice())
    }

    /// Abandon the run.
    ///
    /// Queued groups give their leases back, the slot buffer is not read again,
    /// and the reservation stays charged until [`GroupedRun::close`] releases
    /// it. Cancellation retires the *intent*; it does not make bytes free.
    pub fn cancel(&mut self, authority: &mut ResidencyAuthority) {
        self.cancelled = true;
        while let Some(queued) = self.queue.pop() {
            self.release_pair(authority, queued.gate_up, queued.down);
        }
        // Nothing further will be enqueued: the plan is treated as exhausted so
        // a second cancel, or a stray `step`, cannot start new work.
        self.next_group = self.plan.groups().len();
    }

    /// Give the envelope back.
    ///
    /// Refuses while the queue still holds leases, because releasing the
    /// reservation under a live lease is the shape of R02 this workspace has
    /// already met twice: the ledger saying zero while something is still using
    /// the bytes.
    // The refusal carries the whole run back, because destroying it on the
    // error path is the one thing that would strand the envelope. Boxing it to
    // satisfy a size lint would put an allocation on the failure path of the
    // function whose job is not to lose anything.
    #[allow(clippy::result_large_err)]
    pub fn close(mut self, ledger: &mut Ledger) -> std::result::Result<(), GroupedCloseRefused> {
        if !self.queue.is_empty() {
            return Err(GroupedCloseRefused {
                error: invalid(
                    "close",
                    format!(
                        "{} group(s) still hold residency leases; cancel or finish first",
                        self.queue.len()
                    ),
                ),
                run: self,
            });
        }
        if self.detached {
            // The device attachment owns the charge. Releasing here as well
            // would double-release, which the ledger refuses -- but relying on
            // that refusal instead of knowing whose charge it is is exactly the
            // kind of accounting that made R02 possible.
            self.guard = None;
            return Ok(());
        }
        let Some(reservation) = self.reservation.take() else {
            return Err(GroupedCloseRefused {
                error: invalid("close", "this run has already been closed".into()),
                run: self,
            });
        };
        match ledger.release(reservation) {
            Ok(()) => {
                // The affinity guard goes back here rather than at some
                // arbitrary drop point, so the thread's CPU set is restored at a
                // named place in the run's life.
                self.guard = None;
                Ok(())
            }
            Err(refused) => {
                let error = invalid("close", format!("{}", refused.error));
                self.reservation = Some(refused.reservation);
                Err(GroupedCloseRefused { error, run: self })
            }
        }
    }

    pub const fn ledger(&self) -> LedgerId {
        self.ledger
    }
}

/// A refused close, carrying the run back intact.
#[derive(Debug)]
#[must_use = "the envelope is still charged; correct the cause and close again"]
pub struct GroupedCloseRefused {
    pub error: Error,
    pub run: GroupedRun,
}

impl core::fmt::Display for GroupedCloseRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl Drop for GroupedRun {
    fn drop(&mut self) {
        // A dropped run's reservation stays charged and visible in
        // `Ledger::outstanding`, exactly like a dropped `Reservation`. Releasing
        // it here would make a forgotten run look tidy.
        if self.reservation.is_some() {
            self.reservation = None;
        }
    }
}

impl GroupedRun {
    /// The exact ledger request for a plan's envelope.
    ///
    /// Separated so a caller can `preview` it without admitting: document 02
    /// makes `compile` pure and `admit` atomic, and previewing is how a caller
    /// finds out whether the atomic step will succeed without taking it.
    pub fn request_for(plan: &ExpertPlan) -> Result<PlanRequest> {
        let envelope = plan.envelope();
        let mut request = PlanRequest::new("expert-plan", ["experts"])?;
        let span = StageSpan::inclusive(0, 0);
        for (tier, bytes) in envelope.host.iter() {
            if *bytes == 0 {
                continue;
            }
            request.buffer(BufferRequest::new(
                format!("host-{}", tier_name_host(*tier)),
                Scope::Host,
                Tier::Host(*tier),
                *bytes,
                span,
            ))?;
        }
        for (tier, bytes) in envelope.device.iter() {
            if *bytes == 0 {
                continue;
            }
            request.buffer(BufferRequest::new(
                format!("device-{}", tier_name_device(*tier)),
                Scope::Device(plan.device()),
                Tier::Device(*tier),
                *bytes,
                span,
            ))?;
        }
        Ok(request)
    }

    fn host_extents(plan: &ExpertPlan) -> Result<(usize, usize, usize, usize)> {
        let workspace_values = plan
            .shape()
            .host_workspace_values(plan.cpu_tile_lanes())
            .ok_or_else(|| invalid("workspace", "the host workspace extent overflows".into()))?;
        Ok((
            usize::try_from(plan.activation_bytes()).map_err(|_| too_large("activations"))?,
            usize::try_from(plan.slot_bytes()).map_err(|_| too_large("slots"))?,
            usize::try_from(plan.rows() * plan.shape().hidden * BF16)
                .map_err(|_| too_large("output"))?,
            usize::try_from(workspace_values).map_err(|_| too_large("workspace"))?,
        ))
    }
}

fn tier_name_host(tier: HostTier) -> &'static str {
    match tier {
        HostTier::Pageable => "pageable",
        HostTier::Pinned => "pinned",
        HostTier::MappedResident => "mapped-resident",
        HostTier::CpuWorkspace => "cpu-workspace",
        HostTier::StateSpill => "state-spill",
        HostTier::ConversionReadBuffers => "conversion-read-buffers",
    }
}

fn tier_name_device(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Activations => "activations",
        DeviceTier::KernelWorkspace => "kernel-workspace",
        DeviceTier::TransferStaging => "transfer-staging",
        DeviceTier::ExpertCache => "expert-cache",
        DeviceTier::PackedResidentWeights => "packed-resident-weights",
        DeviceTier::KvStatePages => "kv-state-pages",
        DeviceTier::RecurrentState => "recurrent-state",
        DeviceTier::Logits => "logits",
        DeviceTier::CollectiveBuffers => "collective-buffers",
        DeviceTier::GraphPools => "graph-pools",
        DeviceTier::SpeculativeTargetState => "speculative-target-state",
        DeviceTier::SpeculativeDraftState => "speculative-draft-state",
        DeviceTier::EntropyBranches => "entropy-branches",
        DeviceTier::AllocatorFragmentation => "allocator-fragmentation",
        DeviceTier::SafetyHeadroom => "safety-headroom",
    }
}

/// A refused admission, carrying the plan back and the ledger's own report.
#[derive(Debug)]
pub enum GroupedAdmitRefused {
    /// The request could not be evaluated.
    Invalid { plan: Box<ExpertPlan>, error: Error },
    /// It was evaluated and does not fit. The report says which constraint bound
    /// and what the legal alternatives are, which is what document 03 requires a
    /// rejection to carry.
    Rejected {
        plan: Box<ExpertPlan>,
        rejection: Box<moxie_memory::Rejection>,
    },
}

impl GroupedAdmitRefused {
    pub fn plan(&self) -> &ExpertPlan {
        match self {
            Self::Invalid { plan, .. } | Self::Rejected { plan, .. } => plan,
        }
    }
}

impl core::fmt::Display for GroupedAdmitRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid { error, .. } => write!(f, "{error}"),
            Self::Rejected { rejection, .. } => write!(f, "{rejection}"),
        }
    }
}

impl std::error::Error for GroupedAdmitRefused {}
