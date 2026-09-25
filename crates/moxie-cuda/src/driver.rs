//! Typed CUDA driver wrapper. Compiled only with the `driver` feature.
//!
//! Responsibility (document 02): a safe-ish typed CUDA wrapper. It owns device
//! contexts, allocations, streams, events and module loading. It must not know
//! about models, placement policy or checkpoint discovery.
//!
//! Two contracts from document 02 are load-bearing here and are tested:
//!
//! * Errors are **typed**, by numeric code. The classification itself lives in
//!   [`crate::status`] so that it can be tested with no driver present.
//! * A buffer's lifetime is tied to completion, not to the return of an enqueue
//!   call (R07). `DeviceBuffer` therefore synchronises on drop rather than
//!   freeing memory that an in-flight copy may still be reading.

use core::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ffi::CString;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::Instant;

use moxie_types::{DeviceCapability, DeviceUuid, Error, MeasuredDevice, RankId, Result};

use crate::claims;
use crate::ffi;
use crate::status::classify;

/// Attach the driver's own message to a failing code, then classify by code.
fn check(code: ffi::CUresult, context: &str) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    classify(code, format!("{context}: {} ({code})", error_string(code)))
}

fn error_string(code: ffi::CUresult) -> String {
    let mut ptr: *const c_char = core::ptr::null();
    // SAFETY: `cuGetErrorString` writes a pointer to a static, NUL-terminated
    // string owned by the driver. The driver never frees these, so reading it
    // immediately and copying it out is sound.
    unsafe {
        if ffi::cuGetErrorString(code, &mut ptr) != ffi::CUDA_SUCCESS || ptr.is_null() {
            return format!("unknown CUresult {code}");
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Initialise the driver. Idempotent; safe to call more than once.
pub fn init() -> Result<()> {
    check(
        // SAFETY: `cuInit` takes no pointers and is documented as safe to call
        // repeatedly. Flags must be 0 for all current driver versions.
        unsafe { ffi::cuInit(0) },
        "cuInit",
    )
}

pub fn device_count() -> Result<u32> {
    // `cuInit` is idempotent, and calling it here removes a whole class of
    // "initialization error (3)" failures that look like missing hardware.
    init()?;
    let mut n: c_int = 0;
    check(
        // SAFETY: `n` is a valid, aligned, initialised out-parameter.
        unsafe { ffi::cuDeviceGetCount(&mut n) },
        "cuDeviceGetCount",
    )?;
    Ok(n.max(0) as u32)
}

fn attribute(dev: ffi::CUdevice, attr: c_int) -> Result<i32> {
    let mut v: c_int = 0;
    check(
        // SAFETY: valid out-parameter; `attr` is one of the ordinals declared in
        // `ffi`, and `dev` came from `cuDeviceGet`.
        unsafe { ffi::cuDeviceGetAttribute(&mut v, attr, dev) },
        "cuDeviceGetAttribute",
    )?;
    Ok(v)
}

/// Query one device's capabilities, including peer access against every other
/// device. Nothing here is inferred from the device name (document 03).
pub fn query_device(ordinal: u32) -> Result<DeviceCapability> {
    init()?;
    let count = device_count()?;
    let mut dev: ffi::CUdevice = 0;
    check(
        // SAFETY: valid out-parameter; the driver validates the ordinal.
        unsafe { ffi::cuDeviceGet(&mut dev, ordinal as c_int) },
        "cuDeviceGet",
    )?;

    let mut name_buf = [0 as c_char; 256];
    check(
        // SAFETY: the driver writes at most `len` bytes including the NUL into a
        // buffer we own for the duration of the call.
        unsafe { ffi::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as c_int, dev) },
        "cuDeviceGetName",
    )?;
    // SAFETY: the driver NUL-terminates within the buffer we supplied.
    let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut uuid = ffi::CUuuid::default();
    check(
        // SAFETY: valid out-parameter of exactly the expected 16-byte layout.
        unsafe { ffi::cuDeviceGetUuid(&mut uuid, dev) },
        "cuDeviceGetUuid",
    )?;
    let mut uuid_bytes = [0u8; 16];
    for (dst, src) in uuid_bytes.iter_mut().zip(uuid.bytes.iter()) {
        *dst = *src as u8;
    }
    let uuid = DeviceUuid::from_bytes(uuid_bytes);

    let mut bus_buf = [0 as c_char; 32];
    check(
        // SAFETY: as for `cuDeviceGetName`.
        unsafe { ffi::cuDeviceGetPCIBusId(bus_buf.as_mut_ptr(), bus_buf.len() as c_int, dev) },
        "cuDeviceGetPCIBusId",
    )?;
    // SAFETY: NUL-terminated by the driver within our buffer.
    let pci_bus_id = unsafe { CStr::from_ptr(bus_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut total: usize = 0;
    check(
        // SAFETY: valid out-parameter.
        unsafe { ffi::cuDeviceTotalMem_v2(&mut total, dev) },
        "cuDeviceTotalMem",
    )?;

    let mut peer_access = Vec::new();
    for peer in 0..count {
        if peer == ordinal {
            continue;
        }
        let mut peer_dev: ffi::CUdevice = 0;
        check(
            // SAFETY: valid out-parameter, in-range ordinal.
            unsafe { ffi::cuDeviceGet(&mut peer_dev, peer as c_int) },
            "cuDeviceGet(peer)",
        )?;
        let mut can: c_int = 0;
        check(
            // SAFETY: valid out-parameter; both devices came from `cuDeviceGet`.
            unsafe { ffi::cuDeviceCanAccessPeer(&mut can, dev, peer_dev) },
            "cuDeviceCanAccessPeer",
        )?;
        peer_access.push((peer, can != 0));
    }

    Ok(DeviceCapability {
        ordinal,
        uuid,
        name,
        compute_major: attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u32,
        compute_minor: attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u32,
        total_memory_bytes: total as u64,
        multiprocessor_count: attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
        pci_bus_id,
        peer_access,
        max_grid: (
            attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as u32,
            attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as u32,
            attribute(dev, ffi::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)? as u32,
        ),
    })
}

/// A rank-owned device context: one rank, one GPU, one context.
///
/// Document 01: "Local single-process control with one rank execution thread per
/// GPU is the initial topology. Isolate unsafe CUDA state behind rank-owned
/// contexts." Isolation is the point, so the pairing is enforced rather than
/// assumed. A second context for a device another rank holds is refused and
/// names the holder; so is a second device for a rank that already has one. The
/// claim is released when the context drops, so a later rank can take the card.
///
/// The context stays on the thread that acquired it -- it is `!Send` and
/// `!Sync`, and that is deliberate rather than incidental:
///
/// ```compile_fail
/// use moxie_cuda::RankContext;
/// use moxie_types::RankId;
/// let ctx = RankContext::acquire(RankId(0), 0).unwrap();
/// std::thread::spawn(move || {
///     // A context made current on one thread, used from another.
///     let _ = ctx.make_current();
/// });
/// ```
#[derive(Debug)]
pub struct RankContext {
    device: ffi::CUdevice,
    ctx: ffi::CUcontext,
    rank: RankId,
    capability: DeviceCapability,
    /// This acquisition's process-unique identity. A context handle can be
    /// reused after release, so a grant names the acquisition, not the handle
    /// or the device.
    generation: u64,
    /// Acquisitions whose memory this context has been granted direct access
    /// to. A peer copy is refused unless its source is listed here: without
    /// the grant, `cuMemcpyPeerAsync` silently stages through host memory.
    peers: core::cell::RefCell<Vec<PeerGrant>>,
}

/// Opaque identity for a peer context, exchanged only while its owner thread
/// remains alive. The integer handle is passed to driver calls as a context
/// parameter; it is never made current on the receiving thread.
#[derive(Debug, Clone)]
pub struct PeerContextToken {
    context: usize,
    device: ffi::CUdevice,
    uuid: DeviceUuid,
    generation: u64,
}

impl PeerContextToken {
    /// Acquisition generation used to reject stale peer grants.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug)]
struct PeerGrant {
    context: usize,
    generation: u64,
}

/// Sendable, non-owning description of a source byte span. Its private
/// one-shot channels establish producer readiness and consumer completion.
#[derive(Debug)]
pub struct PeerReadHandle {
    address: u64,
    bytes: usize,
    generation: u64,
    ready: Receiver<()>,
    completed: SyncSender<()>,
}

/// Source-thread half of a peer read. It owns the exported range until the
/// destination acknowledges completion; an unfinished owner leaks that range.
#[derive(Debug)]
pub struct PeerReadOwner<T> {
    source: Option<T>,
    ready: Option<SyncSender<()>>,
    completed: Receiver<()>,
    ordinal: u32,
}

impl<T> PeerReadOwner<T> {
    /// Publish the source only after its producer event has completed.
    pub fn signal_ready(&mut self, producer: &Event<'_>) -> Result<()> {
        if !producer.is_complete()? {
            return Err(Error::InvalidRequest {
                field: "peer_read",
                detail: "producer event is not complete".into(),
            });
        }
        self.ready
            .take()
            .ok_or_else(|| Error::InvalidRequest {
                field: "peer_read",
                detail: "producer readiness was already signalled".into(),
            })?
            .send(())
            .map_err(|_| Error::InvalidRequest {
                field: "peer_read",
                detail: "consumer dropped the peer read before it became ready".into(),
            })
    }

    /// Return the source only after the destination reports observed completion.
    pub fn finish(mut self, deadline: Instant) -> Result<T> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match self.completed.recv_timeout(remaining) {
            Ok(()) => self.source.take().ok_or_else(|| Error::InvalidRequest {
                field: "peer_read",
                detail: "source range was already returned".into(),
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(Error::DeviceLost {
                device: self.ordinal,
                detail: "peer copy completion was not acknowledged before the group deadline"
                    .into(),
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Error::InvalidRequest {
                field: "peer_read",
                detail: "consumer did not acknowledge peer copy completion".into(),
            }),
        }
    }
}

impl<T> Drop for PeerReadOwner<T> {
    fn drop(&mut self) {
        if let Some(source) = self.source.take() {
            // An unacknowledged source may still be read by the peer GPU.
            std::mem::forget(source);
        }
    }
}

#[cfg(test)]
mod peer_read_owner_tests {
    use super::PeerReadOwner;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc::sync_channel};

    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn dropping_an_unfinished_owner_leaks_its_source_range() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (ready, _) = sync_channel(1);
        let (_, completed) = sync_channel(1);
        drop(PeerReadOwner {
            source: Some(DropProbe(Arc::clone(&dropped))),
            ready: Some(ready),
            completed,
            ordinal: 0,
        });
        assert!(!dropped.load(Ordering::SeqCst));
    }
}

/// How a failed `attach` left the rank claim.
///
/// The distinction is the whole of the error path's correctness: whether a
/// context reference was ever retained decides whether the claim may simply be
/// dropped, and if one was, whether its cleanup succeeded decides whether the
/// device may be handed to anyone else.
enum AttachError {
    /// Nothing was retained. The claim is untouched and the caller must drop it.
    ClaimUntouched(Error),
    /// A reference was retained and the claim has already been resolved through
    /// the fail-closed release path. The caller must not touch it.
    ClaimResolved(Error),
}

type AttachResult = std::result::Result<RankContext, AttachError>;

impl RankContext {
    /// Acquire `ordinal`'s context for `rank`.
    ///
    /// The ordinal selects the card; everything afterwards identifies it by
    /// UUID. Under a different `CUDA_VISIBLE_DEVICES` the same ordinal is a
    /// different device, and this is the last place in the engine where that
    /// matters (AGENTS.md).
    pub fn acquire(rank: RankId, ordinal: u32) -> Result<Self> {
        init()?;
        let capability = query_device(ordinal)?;
        let uuid = capability.uuid;

        claims::claim(uuid, rank)?;
        match Self::attach(rank, capability) {
            Ok(ctx) => Ok(ctx),
            // Nothing was retained, so the claim is this function's to drop --
            // otherwise a transient driver error strands the card for the life
            // of the process.
            Err(AttachError::ClaimUntouched(e)) => {
                claims::abandon(uuid);
                Err(e)
            }
            // A reference was retained and `attach` already resolved the claim
            // through the same fail-closed path `Drop` uses: removed if the
            // cleanup release succeeded, withheld if it did not. Touching it
            // here would undo exactly that decision.
            Err(AttachError::ClaimResolved(e)) => Err(e),
        }
    }

    fn attach(rank: RankId, capability: DeviceCapability) -> AttachResult {
        let uuid = capability.uuid;
        let mut device: ffi::CUdevice = 0;
        check(
            // SAFETY: valid out-parameter; the driver validates the ordinal.
            unsafe { ffi::cuDeviceGet(&mut device, capability.ordinal as c_int) },
            "cuDeviceGet",
        )
        .map_err(AttachError::ClaimUntouched)?;
        let mut ctx: ffi::CUcontext = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter. The primary context is reference-counted
            // by the driver; `Drop` releases exactly the reference taken here.
            unsafe { ffi::cuDevicePrimaryCtxRetain(&mut ctx, device) },
            "cuDevicePrimaryCtxRetain",
        )
        .map_err(AttachError::ClaimUntouched)?;

        // A reference now exists, and from here every failure path must balance
        // it *and* resolve the claim the same way `Drop` does. An earlier
        // version returned the `cuCtxSetCurrent` error directly and leaked the
        // reference; the version after that released it but ignored the result
        // and let the caller abandon the claim regardless, which handed the next
        // rank a device whose reference was still outstanding.
        if let Err(e) = check(
            // SAFETY: `ctx` is a live context we just retained.
            unsafe { ffi::cuCtxSetCurrent(ctx) },
            "cuCtxSetCurrent",
        ) {
            claims::release_with(uuid, rank, || {
                check(
                    // SAFETY: releases exactly the reference retained above.
                    unsafe { ffi::cuDevicePrimaryCtxRelease_v2(device) },
                    "cuDevicePrimaryCtxRelease",
                )
            });
            return Err(AttachError::ClaimResolved(e));
        }
        Ok(Self {
            device,
            ctx,
            rank,
            capability,
            generation: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            peers: core::cell::RefCell::new(Vec::new()),
        })
    }

    pub fn rank(&self) -> RankId {
        self.rank
    }

    /// The device's identity. What every plan, record and map key uses.
    pub fn uuid(&self) -> DeviceUuid {
        self.capability.uuid
    }

    pub fn capability(&self) -> &DeviceCapability {
        &self.capability
    }

    /// Diagnostic only, for reconciling a log with `nvidia-smi`.
    pub fn ordinal(&self) -> u32 {
        self.capability.ordinal
    }

    /// The raw context handle, for sibling types in this crate that must make
    /// it current before touching a resource it owns. Deliberately not public:
    /// nothing outside `moxie-cuda` may hold a bare context.
    pub(crate) fn raw(&self) -> ffi::CUcontext {
        self.ctx
    }

    /// Make this context current on the calling thread.
    pub fn make_current(&self) -> Result<()> {
        check(
            // SAFETY: `self.ctx` is live for as long as `self` is.
            unsafe { ffi::cuCtxSetCurrent(self.ctx) },
            "cuCtxSetCurrent",
        )
    }

    /// Free and total device memory, as the driver sees it.
    ///
    /// Note for document 03's ledger: this is *not* an admission budget. It does
    /// not account for fragmentation, another process, or a reservation this
    /// engine has already promised.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.make_current()?;
        let (mut free, mut total) = (0usize, 0usize);
        check(
            // SAFETY: two valid out-parameters; context is current.
            unsafe { ffi::cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free as u64, total as u64))
    }

    /// One reading of this device's capacity, for the resource ledger.
    ///
    /// Taken **through the live context**, so what the context itself costs is
    /// already gone from `free_bytes`. A reading taken before the context
    /// existed would over-promise by exactly that much.
    ///
    /// It is a reading and not a reservation: another process can take memory a
    /// moment later, and re-measuring may return something different. Nothing
    /// downstream may treat one reading as a property of the card.
    pub fn measure(&self) -> Result<MeasuredDevice> {
        let (free_bytes, total_bytes) = self.memory_info()?;
        Ok(MeasuredDevice {
            uuid: self.capability.uuid,
            ordinal_label: self.capability.ordinal,
            name: self.capability.name.clone(),
            sm: self.capability.sm(),
            pci_bus_id: self.capability.pci_bus_id.clone(),
            multiprocessor_count: self.capability.multiprocessor_count,
            total_bytes,
            free_bytes,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        check(
            // SAFETY: context is current.
            unsafe { ffi::cuCtxSynchronize() },
            "cuCtxSynchronize",
        )
    }

    /// Let this context read and write `peer`'s device memory directly.
    ///
    /// Refused with `Unsupported { capability: "peer_access" }` when the driver
    /// does not grant the pair. Idempotent: an already enabled grant succeeds.
    pub fn enable_peer_access(&self, peer: &RankContext) -> Result<()> {
        if peer.uuid() == self.uuid() {
            return Err(Error::InvalidRequest {
                field: "peer",
                detail: "a device is not its own peer".into(),
            });
        }
        let mut can: c_int = 0;
        check(
            // SAFETY: valid out-parameter; both devices came from `cuDeviceGet`.
            unsafe { ffi::cuDeviceCanAccessPeer(&mut can, self.device, peer.device) },
            "cuDeviceCanAccessPeer",
        )?;
        if can == 0 {
            return Err(Error::Unsupported {
                capability: "peer_access",
                reason: format!("{} cannot access {}", self.uuid(), peer.uuid()),
            });
        }
        self.make_current()?;
        // SAFETY: this context is current and `peer.ctx` is live for as long
        // as `peer` is; flags must be zero.
        let code = unsafe { ffi::cuCtxEnablePeerAccess(peer.ctx, 0) };
        // PEER_ACCESS_ALREADY_ENABLED: the grant this call exists to make.
        if code != 704 {
            check(code, "cuCtxEnablePeerAccess")?;
        }
        self.remember_peer(PeerGrant {
            context: peer.ctx as usize,
            generation: peer.generation,
        });
        Ok(())
    }

    /// A Send identity for installing peer access from another rank worker.
    /// The rank group must keep this context owner alive while the token can
    /// be used; it is an identity, not an owning context reference.
    pub fn peer_context_token(&self) -> PeerContextToken {
        PeerContextToken {
            context: self.ctx as usize,
            device: self.device,
            uuid: self.uuid(),
            generation: self.generation,
        }
    }

    /// Install a generation-keyed peer grant on this context's owner thread.
    /// The rank group pins the token's owner thread until the grant is no
    /// longer used.
    pub fn enable_peer_access_to(&self, peer: &PeerContextToken) -> Result<()> {
        if peer.uuid == self.uuid() {
            return Err(Error::InvalidRequest {
                field: "peer",
                detail: "a device is not its own peer".into(),
            });
        }
        let mut can: c_int = 0;
        check(
            // SAFETY: both device handles were returned by `cuDeviceGet`.
            unsafe { ffi::cuDeviceCanAccessPeer(&mut can, self.device, peer.device) },
            "cuDeviceCanAccessPeer",
        )?;
        if can == 0 {
            return Err(Error::Unsupported {
                capability: "peer_access",
                reason: format!("{} cannot access {}", self.uuid(), peer.uuid),
            });
        }
        self.make_current()?;
        // SAFETY: this rank context is current. `peer.context` is supplied by
        // its live owner, and the group retains that owner through every peer
        // copy acknowledgement. CUDA's driver API takes the source context as
        // an explicit handle here; it is not made current on this thread.
        let code = unsafe { ffi::cuCtxEnablePeerAccess(peer.context as ffi::CUcontext, 0) };
        if code != 704 {
            check(code, "cuCtxEnablePeerAccess")?;
        }
        self.remember_peer(PeerGrant {
            context: peer.context,
            generation: peer.generation,
        });
        Ok(())
    }

    /// Disable a generation-keyed peer grant on this context's owner thread.
    pub fn disable_peer_access_to(&self, peer: &PeerContextToken) -> Result<()> {
        let grant = self
            .peer_grant(peer.generation)
            .ok_or_else(|| Error::InvalidRequest {
                field: "peer",
                detail: "this context has no grant for the peer acquisition".into(),
            })?;
        if grant.context != peer.context {
            return Err(Error::InvalidRequest {
                field: "peer",
                detail: "peer generation resolves to a different context".into(),
            });
        }
        self.make_current()?;
        // SAFETY: this rank's context is current and the peer handle is pinned
        // by the startup group until the grant is disabled.
        let code = unsafe { ffi::cuCtxDisablePeerAccess(peer.context as ffi::CUcontext) };
        // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED means the desired state already
        // holds; either way, forget the generation-keyed grant.
        if code != 705 {
            check(code, "cuCtxDisablePeerAccess")?;
        }
        self.forget_peer(peer.generation);
        Ok(())
    }

    fn remember_peer(&self, grant: PeerGrant) {
        let mut peers = self.peers.borrow_mut();
        if let Some(existing) = peers
            .iter_mut()
            .find(|existing| existing.generation == grant.generation)
        {
            *existing = grant;
        } else {
            peers.push(grant);
        }
    }

    fn forget_peer(&self, generation: u64) {
        self.peers
            .borrow_mut()
            .retain(|grant| grant.generation != generation);
    }

    fn peer_grant(&self, generation: u64) -> Option<PeerGrant> {
        self.peers
            .borrow()
            .iter()
            .find(|peer| peer.generation == generation)
            .map(|peer| PeerGrant {
                context: peer.context,
                generation: peer.generation,
            })
    }

    fn can_address(&self, other: &RankContext) -> bool {
        other.generation == self.generation
            || self
                .peers
                .borrow()
                .iter()
                .any(|peer| peer.generation == other.generation)
    }
}

impl Drop for RankContext {
    fn drop(&mut self) {
        let device = self.device;
        // The claim is released *after* the driver reference, not before. A
        // release-then-teardown order lets another rank acquire the card while
        // this context reference is still outstanding, and two live primary
        // contexts on one device is the state rank ownership exists to prevent.
        // A failed teardown withholds the device rather than advertising it.
        claims::release_with(self.capability.uuid, self.rank, || {
            check(
                // SAFETY: releases exactly the primary-context reference taken
                // in `attach`.
                unsafe { ffi::cuDevicePrimaryCtxRelease_v2(device) },
                "cuDevicePrimaryCtxRelease",
            )
        });
    }
}

/// An instantiated CUDA graph bound to its owning context.
///
/// The owner must not drop a graph while a launch of it may still run; whoever
/// holds the kernels' buffers holds the graph with them.
#[derive(Debug)]
pub struct CapturedGraph<'ctx> {
    graph_exec: ffi::CUgraphExec,
    ctx: &'ctx RankContext,
}

impl<'ctx> CapturedGraph<'ctx> {
    /// Enqueue this graph on a stream from the same device.
    ///
    /// # Safety
    /// Every buffer a captured kernel or copy names must be live and not
    /// concurrently written until a completion recorded after this launch is
    /// observed.
    pub unsafe fn launch(&self, stream: &Stream<'ctx>) -> Result<()> {
        if stream.device_uuid() != self.ctx.uuid() {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "captured graph and stream belong to different devices".into(),
            });
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: the graph and stream are live, belong to this device,
            // and the buffer lifetime obligation is forwarded to the caller.
            unsafe { ffi::cuGraphLaunch(self.graph_exec, stream.raw()) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for CapturedGraph<'_> {
    fn drop(&mut self) {
        // SAFETY: the owning context is made current before destroying the
        // executable graph. Teardown errors are not actionable.
        unsafe {
            let _ = ffi::cuCtxSetCurrent(self.ctx.raw());
            let _ = ffi::cuGraphExecDestroy(self.graph_exec);
        }
    }
}

/// A non-blocking stream owned by one context.
///
/// M0 scope: enough to record and wait on an event, which is the bounded
/// stream/event completion smoke document 06 M0.4 asks for. It is **not** the
/// event-retained lease mechanism from R07; that arrives with the memory
/// authority in M1/M2, and this type must not be mistaken for it.
#[derive(Debug)]
pub struct Stream<'ctx> {
    stream: ffi::CUstream,
    ctx: &'ctx RankContext,
}

/// Enqueue a device-to-device rectangular copy on one stream.
///
/// # Safety
/// Both device regions must remain allocated in `ctx` through completion, and
/// each must cover `pitch * (height - 1) + width_bytes` bytes from its address.
/// This copy is capturable during stream capture.
#[allow(clippy::too_many_arguments)]
pub unsafe fn copy_2d_async(
    ctx: &RankContext,
    dst: u64,
    dst_pitch: u64,
    src: u64,
    src_pitch: u64,
    width_bytes: u64,
    height: u64,
    stream: &Stream<'_>,
) -> Result<()> {
    if !core::ptr::eq(ctx, stream.ctx)
        || dst == 0
        || src == 0
        || width_bytes == 0
        || height == 0
        || dst_pitch < width_bytes
        || src_pitch < width_bytes
    {
        return Err(Error::InvalidRequest {
            field: "copy_2d",
            detail: "copy dimensions, pitches, addresses, and stream context must agree".into(),
        });
    }
    let extent = |pitch: u64| {
        pitch
            .checked_mul(height - 1)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .ok_or_else(|| Error::InvalidRequest {
                field: "copy_2d",
                detail: "copy extent overflowed".into(),
            })
    };
    let (dst_extent, src_extent) = (extent(dst_pitch)?, extent(src_pitch)?);
    if dst.checked_add(dst_extent).is_none() || src.checked_add(src_extent).is_none() {
        return Err(Error::InvalidRequest {
            field: "copy_2d",
            detail: "copy address extent overflowed".into(),
        });
    }
    let (dst_pitch, src_pitch, width_bytes, height) = (
        usize::try_from(dst_pitch).map_err(|_| Error::InvalidRequest {
            field: "copy_2d",
            detail: "destination pitch exceeds the host ABI".into(),
        })?,
        usize::try_from(src_pitch).map_err(|_| Error::InvalidRequest {
            field: "copy_2d",
            detail: "source pitch exceeds the host ABI".into(),
        })?,
        usize::try_from(width_bytes).map_err(|_| Error::InvalidRequest {
            field: "copy_2d",
            detail: "copy width exceeds the host ABI".into(),
        })?,
        usize::try_from(height).map_err(|_| Error::InvalidRequest {
            field: "copy_2d",
            detail: "copy height exceeds the host ABI".into(),
        })?,
    );
    ctx.make_current()?;
    let copy = ffi::CUDA_MEMCPY2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        src_host: core::ptr::null(),
        src_device: src,
        src_array: core::ptr::null_mut(),
        src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        dst_host: core::ptr::null_mut(),
        dst_device: dst,
        dst_array: core::ptr::null_mut(),
        dst_pitch,
        width_in_bytes: width_bytes,
        height,
    };
    check(
        // SAFETY: callers guarantee the source and destination extents remain
        // live through stream completion; dimensions and stream context were checked.
        unsafe { ffi::cuMemcpy2DAsync_v2(&copy, stream.raw()) },
        "cuMemcpy2DAsync",
    )
}

impl<'ctx> Stream<'ctx> {
    /// Stable identity of the device owning this handle.
    pub fn device_uuid(&self) -> DeviceUuid {
        self.ctx.uuid()
    }

    pub fn new(ctx: &'ctx RankContext) -> Result<Self> {
        ctx.make_current()?;
        let mut stream: ffi::CUstream = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter; context is current.
            unsafe { ffi::cuStreamCreate(&mut stream, ffi::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(Self { stream, ctx })
    }

    pub(crate) fn raw(&self) -> ffi::CUstream {
        self.stream
    }

    /// Begin capturing kernel and copy work enqueued on this stream.
    pub fn begin_capture(&self) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: the stream is live on the current context, and the
            // thread-local mode keeps unrelated rank threads out of this
            // capture's validity rules.
            unsafe {
                ffi::cuStreamBeginCapture_v2(self.stream, ffi::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            },
            "cuStreamBeginCapture_v2",
        )
    }

    /// End capture and instantiate the captured graph for replay.
    pub fn end_capture(&self) -> Result<CapturedGraph<'ctx>> {
        self.ctx.make_current()?;
        let mut graph = core::ptr::null_mut();
        // SAFETY: the stream is live on the current context and `graph` is a
        // valid out-parameter; failure may leave it null.
        let ended = unsafe { ffi::cuStreamEndCapture(self.stream, &mut graph) };
        if let Err(error) = check(ended, "cuStreamEndCapture") {
            if !graph.is_null() {
                // SAFETY: a non-null graph was returned by this stream's
                // capture; it belongs to the current context.
                let _ = unsafe { ffi::cuGraphDestroy(graph) };
            }
            return Err(error);
        }
        if graph.is_null() {
            return Err(Error::InvalidRequest {
                field: "cuda_graph",
                detail: "capture succeeded without returning a graph".into(),
            });
        }

        let mut graph_exec = core::ptr::null_mut();
        // SAFETY: `graph` is the live graph returned by this stream, and
        // `graph_exec` is a valid out-parameter on the current context.
        let instantiated = unsafe { ffi::cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        // The original graph is no longer needed after instantiation, even if
        // instantiation failed.
        // SAFETY: `graph` is a live graph in the current context.
        let destroyed = unsafe { ffi::cuGraphDestroy(graph) };
        if let Err(error) = check(instantiated, "cuGraphInstantiateWithFlags") {
            if !graph_exec.is_null() {
                // SAFETY: the failed call nevertheless returned a graph exec.
                let _ = unsafe { ffi::cuGraphExecDestroy(graph_exec) };
            }
            let _ = destroyed;
            return Err(error);
        }
        if let Err(error) = check(destroyed, "cuGraphDestroy") {
            // SAFETY: the just-instantiated executable graph is live in the
            // current context and cannot be returned after this error.
            let _ = unsafe { ffi::cuGraphExecDestroy(graph_exec) };
            return Err(error);
        }
        if graph_exec.is_null() {
            return Err(Error::InvalidRequest {
                field: "cuda_graph_exec",
                detail: "instantiation succeeded without returning an executable graph".into(),
            });
        }
        Ok(CapturedGraph {
            graph_exec,
            ctx: self.ctx,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: live stream created on the context made current above.
            unsafe { ffi::cuStreamSynchronize(self.stream) },
            "cuStreamSynchronize",
        )
    }

    /// Order this stream's later work after `event`, which may belong to
    /// another device. Nothing blocks on the host.
    pub fn wait_event(&self, event: &Event<'_>) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: live stream on the current context and a live event;
            // the driver permits an event from another context here.
            unsafe { ffi::cuStreamWaitEvent(self.stream, event.event, 0) },
            "cuStreamWaitEvent",
        )
    }

    /// Enqueue a host callback after work already submitted to this stream.
    ///
    /// # Safety
    /// The callback and `data` must remain valid until invocation. CUDA forbids
    /// CUDA API calls from the callback; it must not call back into this
    /// context or any stream.
    pub unsafe fn launch_host_func(
        &self,
        callback: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: the caller guarantees the callback and data outlive the
            // queued invocation and obey CUDA's callback restrictions.
            unsafe { ffi::cuLaunchHostFunc(self.stream, callback, data) },
            "cuLaunchHostFunc",
        )
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        // SAFETY: the owning context is made current first. `cuStreamDestroy`
        // is documented to return immediately and complete asynchronously once
        // pending work finishes, so no in-flight work is cut short. Teardown
        // errors are not actionable.
        unsafe {
            let _ = ffi::cuCtxSetCurrent(self.ctx.raw());
            let _ = ffi::cuStreamDestroy_v2(self.stream);
        }
    }
}

/// A completion event.
///
/// This is the object that will later retire a lease: "Retirement is
/// event-driven; Rust `Drop` alone must not free in-flight CUDA memory" (R07).
/// At M0 it only proves the mechanism works end to end.
#[derive(Debug)]
pub struct Event<'ctx> {
    event: ffi::CUevent,
    ctx: &'ctx RankContext,
}

impl<'ctx> Event<'ctx> {
    /// Stable identity of the device owning this handle.
    pub fn device_uuid(&self) -> DeviceUuid {
        self.ctx.uuid()
    }

    pub fn new(ctx: &'ctx RankContext) -> Result<Self> {
        ctx.make_current()?;
        let mut event: ffi::CUevent = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter; context is current.
            unsafe { ffi::cuEventCreate(&mut event, ffi::CU_EVENT_DEFAULT) },
            "cuEventCreate",
        )?;
        Ok(Self { event, ctx })
    }

    /// Record this event into `stream`'s ordered work.
    pub fn record(&self, stream: &Stream<'ctx>) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: live event and live stream, both created on this context.
            unsafe { ffi::cuEventRecord(self.event, stream.raw()) },
            "cuEventRecord",
        )
    }

    /// Whether every preceding item in the recorded stream has completed.
    ///
    /// `CUDA_ERROR_NOT_READY` (600) is a *state*, not a failure: it is the whole
    /// point of a query. Any other code is a real error and stays typed.
    pub fn is_complete(&self) -> Result<bool> {
        self.ctx.make_current()?;
        // SAFETY: live event on the current context.
        let code = unsafe { ffi::cuEventQuery(self.event) };
        match code {
            ffi::CUDA_SUCCESS => Ok(true),
            600 => Ok(false),
            other => check(other, "cuEventQuery").map(|()| true),
        }
    }

    /// Block until the recorded work has completed.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: live event on the current context.
            unsafe { ffi::cuEventSynchronize(self.event) },
            "cuEventSynchronize",
        )
    }

    /// Milliseconds between two completed events on the same context.
    pub fn elapsed_ms(start: &Event<'ctx>, end: &Event<'ctx>) -> Result<f32> {
        start.ctx.make_current()?;
        let mut ms = 0f32;
        check(
            // SAFETY: valid out-parameter; both events are live and were
            // recorded on this context.
            unsafe { ffi::cuEventElapsedTime_v2(&mut ms, start.event, end.event) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        // SAFETY: owning context made current first; teardown errors are not
        // actionable.
        unsafe {
            let _ = ffi::cuCtxSetCurrent(self.ctx.raw());
            let _ = ffi::cuEventDestroy_v2(self.event);
        }
    }
}

/// Page-locked host memory for asynchronous CUDA copies.
///
/// The caller must keep this buffer alive and must not mutate a source region
/// while an asynchronous copy may still read it. Dropping it while an async
/// copy may read it is the owner's error, as for any host source.
#[derive(Debug)]
pub struct PinnedHostBuffer<'ctx> {
    ptr: *mut u8,
    len: usize,
    ctx: &'ctx RankContext,
}

impl<'ctx> PinnedHostBuffer<'ctx> {
    /// Allocate page-locked host memory with the default CUDA allocation flags.
    pub fn alloc(ctx: &'ctx RankContext, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidRequest {
                field: "pinned_host_buffer",
                detail: "zero-length allocations are not supported".into(),
            });
        }
        ctx.make_current()?;
        let mut ptr = core::ptr::null_mut();
        if let Err(error) = check(
            // SAFETY: `ptr` is a valid out-parameter, the current context is
            // live, and the default allocation flags are zero.
            unsafe { ffi::cuMemHostAlloc(&mut ptr, len, 0) },
            "cuMemHostAlloc",
        ) {
            return Err(match error {
                Error::CapacityExceeded { tier, .. } => Error::CapacityExceeded {
                    tier,
                    requested_bytes: len as u64,
                    available_bytes: 0,
                },
                other => other,
            });
        }
        if ptr.is_null() {
            return Err(Error::InvalidRequest {
                field: "pinned_host_buffer",
                detail: "CUDA returned a null pointer for a nonzero allocation".into(),
            });
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
            ctx,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: allocation succeeds only with a non-null pointer and nonzero
        // length, and the borrow is bounded by the live allocation.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: allocation succeeds only with a non-null pointer and nonzero
        // length, and the mutable borrow is exclusive on the Rust side.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Free this allocation, returning it unchanged if the driver refuses.
    pub fn free(mut self) -> std::result::Result<(), (Self, Error)> {
        if let Err(error) = self.ctx.make_current() {
            return Err((self, error));
        }
        if let Err(error) = check(
            // SAFETY: this pointer is the live allocation returned by
            // `cuMemHostAlloc`, and its asynchronous users have been drained.
            unsafe { ffi::cuMemFreeHost(self.ptr.cast()) },
            "cuMemFreeHost",
        ) {
            return Err((self, error));
        }
        self.ptr = core::ptr::null_mut();
        Ok(())
    }
}

impl Drop for PinnedHostBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.ctx.make_current().is_err() {
            return;
        }
        // SAFETY: this pointer came from `cuMemHostAlloc` on the current
        // context; the owner must have completed every asynchronous use first.
        let _ = check(
            unsafe { ffi::cuMemFreeHost(self.ptr.cast()) },
            "cuMemFreeHost",
        );
    }
}

/// A device allocation.
///
/// R07 and document 02: "A kernel launch leases its inputs/outputs/workspace
/// until completion. Retirement is event-driven; Rust `Drop` alone must not free
/// in-flight CUDA memory." This type discharges that obligation the blunt way
/// -- it synchronises the context before freeing. The executor's admitted arena
/// owns the checked event-retained path; other consumers keep this conservative
/// fallback. The invariant is the same, and this type must not be "optimised"
/// by simply deleting the synchronise.
/// A buffer cannot outlive the context that owns it. The compiler enforces it:
///
/// ```compile_fail
/// use moxie_cuda::{DeviceBuffer, RankContext};
/// use moxie_types::RankId;
/// let escaped = {
///     let ctx = RankContext::acquire(RankId(0), 0).unwrap();
///     DeviceBuffer::alloc(&ctx, 16).unwrap()
/// }; // `ctx` dropped here, releasing the primary context
/// drop(escaped); // would free into a released context
/// ```
#[derive(Debug)]
pub struct DeviceBuffer<'ctx> {
    ptr: ffi::CUdeviceptr,
    len: usize,
    /// The context that owns this allocation.
    ///
    /// Borrowed, not copied, so the compiler refuses a buffer that outlives its
    /// context. Without this the allocation could be freed after
    /// `cuDevicePrimaryCtxRelease`, and `Drop` would synchronise whatever
    /// context happened to be current on the dropping thread -- protecting the
    /// wrong stream while freeing into the wrong context.
    ctx: &'ctx RankContext,
}

impl<'ctx> DeviceBuffer<'ctx> {
    pub fn alloc(ctx: &'ctx RankContext, len: usize) -> Result<Self> {
        ctx.make_current()?;
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                ctx,
            });
        }
        let mut ptr: ffi::CUdeviceptr = 0;
        let r = check(
            // SAFETY: valid out-parameter, non-zero size, context is current.
            unsafe { ffi::cuMemAlloc_v2(&mut ptr, len) },
            "cuMemAlloc",
        );
        if let Err(e) = r {
            // Re-attach the size actually requested, so an admission report can
            // show a real breakdown rather than two zeroes (document 03).
            return Err(match e {
                Error::CapacityExceeded { tier, .. } => Error::CapacityExceeded {
                    tier,
                    requested_bytes: len as u64,
                    available_bytes: ctx.memory_info().map(|(f, _)| f).unwrap_or(0),
                },
                other => other,
            });
        }
        Ok(Self { ptr, len, ctx })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stable device identity of this allocation.
    pub fn device_uuid(&self) -> DeviceUuid {
        self.ctx.uuid()
    }

    pub fn device_ptr(&self) -> ffi::CUdeviceptr {
        self.ptr
    }

    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.len {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: format!("{} bytes into a {}-byte buffer", src.len(), self.len),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: `src` is a valid readable slice of `src.len()` bytes and the
            // destination holds at least that many, checked above. The synchronous
            // form returns only once the copy completed, so `src` needs no lifetime
            // extension -- the asynchronous form would (R07).
            unsafe { ffi::cuMemcpyHtoD_v2(self.ptr, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Synchronously copy host bytes into a checked byte range.
    pub fn copy_from_host_at(&self, offset: usize, src: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(src.len())
            .ok_or_else(|| Error::InvalidRequest {
                field: "src",
                detail: "copy range overflowed".into(),
            })?;
        if end > self.len {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: format!(
                    "{} bytes at offset {offset} into a {}-byte buffer",
                    src.len(),
                    self.len
                ),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        let destination =
            self.ptr
                .checked_add(offset as u64)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "src",
                    detail: "device address overflowed".into(),
                })?;
        self.ctx.make_current()?;
        check(
            // SAFETY: `src` is a valid readable slice of `src.len()` bytes and
            // the destination holds at least that many, checked above. The
            // synchronous form returns only once the copy completed, so `src`
            // needs no lifetime extension -- the asynchronous form would (R07).
            unsafe { ffi::cuMemcpyHtoD_v2(destination, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Enqueue a host-to-device copy on `stream`.
    ///
    /// # Safety
    /// This returns before the copy has happened. The caller must keep `src`
    /// alive and unmodified, and must keep this buffer alive, until an event
    /// recorded after the copy has completed. That is exactly the obligation
    /// R07 records as the one the legacy engine got wrong; there is no lease
    /// mechanism at M0 to discharge it automatically, so it is the caller's.
    pub unsafe fn copy_from_host_async(&self, src: &[u8], stream: &Stream<'ctx>) -> Result<()> {
        // SAFETY: this method's source-lifetime obligation is forwarded
        // unchanged; offset zero is within every allocation.
        unsafe { self.copy_from_host_async_at(0, src, stream) }
    }

    /// Enqueue a host-to-device copy into a checked byte range.
    ///
    /// # Safety
    /// As [`DeviceBuffer::copy_from_host_async`]: `src` and this allocation
    /// must remain live and unmodified until a following event completes.
    pub unsafe fn copy_from_host_async_at(
        &self,
        offset: usize,
        src: &[u8],
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        let end = offset
            .checked_add(src.len())
            .ok_or_else(|| Error::InvalidRequest {
                field: "src",
                detail: "copy range overflowed".into(),
            })?;
        if end > self.len {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: format!(
                    "{} bytes at offset {offset} into a {}-byte buffer",
                    src.len(),
                    self.len
                ),
            });
        }
        if stream.device_uuid() != self.device_uuid() {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "copy stream belongs to another device".into(),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        let destination =
            self.ptr
                .checked_add(offset as u64)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "src",
                    detail: "device address overflowed".into(),
                })?;
        self.ctx.make_current()?;
        check(
            // SAFETY: bounds checked above; the source-lifetime obligation is
            // delegated to this function's own safety contract, which the caller
            // accepted.
            unsafe {
                ffi::cuMemcpyHtoDAsync_v2(
                    destination,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    stream.raw(),
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Enqueue a device-to-device copy between allocations on the same GPU.
    ///
    /// # Safety
    /// Both allocations and the stream must remain valid until the copy is
    /// observed complete by the caller. The typed lifetimes bind the context;
    /// the device identity checks below bind the actual GPU.
    pub unsafe fn copy_from_device_async_at(
        &self,
        offset: usize,
        source: &DeviceBuffer<'ctx>,
        source_offset: usize,
        byte_count: usize,
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        let destination_end =
            offset
                .checked_add(byte_count)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "dst",
                    detail: "copy range overflowed".into(),
                })?;
        if destination_end > self.len {
            return Err(Error::InvalidRequest {
                field: "dst",
                detail: format!(
                    "{byte_count} bytes at offset {offset} into a {}-byte buffer",
                    self.len
                ),
            });
        }
        let source_end =
            source_offset
                .checked_add(byte_count)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "src",
                    detail: "copy range overflowed".into(),
                })?;
        if source_end > source.len {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: format!(
                    "{byte_count} bytes at offset {source_offset} out of a {}-byte buffer",
                    source.len
                ),
            });
        }
        if self.device_uuid() != source.device_uuid() || stream.device_uuid() != self.device_uuid()
        {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "device-to-device copy requires one device and its stream".into(),
            });
        }
        if byte_count == 0 {
            return Ok(());
        }
        let destination =
            self.ptr
                .checked_add(offset as u64)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "dst",
                    detail: "device address overflowed".into(),
                })?;
        let source = source
            .ptr
            .checked_add(source_offset as u64)
            .ok_or_else(|| Error::InvalidRequest {
                field: "src",
                detail: "device address overflowed".into(),
            })?;
        self.ctx.make_current()?;
        check(
            // SAFETY: both ranges, their device identity and the stream were
            // checked above; the caller owns their lifetime through completion.
            unsafe { ffi::cuMemcpyDtoDAsync_v2(destination, source, byte_count, stream.raw()) },
            "cuMemcpyDtoDAsync",
        )
    }

    /// Enqueue a direct copy from an allocation on this device or on a peer
    /// this context has been granted access to.
    ///
    /// A source on a device without an enabled grant is refused rather than
    /// handed to the driver, which would stage it through host memory.
    ///
    /// # Safety
    /// Both allocations and the stream must remain valid until the copy is
    /// observed complete by the caller.
    pub unsafe fn copy_from_peer_async_at(
        &self,
        offset: usize,
        source: &DeviceBuffer<'_>,
        source_offset: usize,
        byte_count: usize,
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        let destination = span(self, offset, byte_count, "dst")?;
        let from = span(source, source_offset, byte_count, "src")?;
        if stream.device_uuid() != self.device_uuid() {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "a peer copy is enqueued on the destination device's stream".into(),
            });
        }
        if !self.ctx.can_address(source.ctx) {
            return Err(Error::Unsupported {
                capability: "peer_access",
                reason: format!(
                    "{} has no peer access to {}; refusing a host-staged copy",
                    self.device_uuid(),
                    source.device_uuid()
                ),
            });
        }
        if byte_count == 0 {
            return Ok(());
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: both extents, the stream's device and the peer grant were
            // checked above; the caller owns their lifetime through completion.
            unsafe {
                ffi::cuMemcpyPeerAsync(
                    destination,
                    self.ctx.raw(),
                    from,
                    source.ctx.raw(),
                    byte_count,
                    stream.raw(),
                )
            },
            "cuMemcpyPeerAsync",
        )
    }

    /// Export one checked source span to the other rank thread, transferring
    /// its owner into the returned guard. The guard leaks an unfinished source.
    ///
    /// # Safety
    /// `source` must own the device memory at `[address, address+len)` and keep it alive for as long as it is held.
    pub unsafe fn export_peer_read_at<T>(
        &self,
        offset: usize,
        byte_count: usize,
        source: T,
    ) -> Result<(PeerReadHandle, PeerReadOwner<T>)> {
        let address = span(self, offset, byte_count, "src")?;
        let (ready_tx, ready_rx) = sync_channel(1);
        let (completed_tx, completed_rx) = sync_channel(1);
        Ok((
            PeerReadHandle {
                address,
                bytes: byte_count,
                generation: self.ctx.generation,
                ready: ready_rx,
                completed: completed_tx,
            },
            PeerReadOwner {
                source: Some(source),
                ready: Some(ready_tx),
                completed: completed_rx,
                ordinal: self.ctx.ordinal(),
            },
        ))
    }

    /// Copy a source-thread handle into this allocation and acknowledge it
    /// only after an event on the destination stream proves completion.
    pub fn copy_from_peer_read_at(
        &self,
        offset: usize,
        handle: PeerReadHandle,
        source_offset: usize,
        byte_count: usize,
        stream: &Stream<'ctx>,
        deadline: Instant,
    ) -> Result<()> {
        let ready_timeout = deadline.saturating_duration_since(Instant::now());
        handle
            .ready
            .recv_timeout(ready_timeout)
            .map_err(|error| match error {
                std::sync::mpsc::RecvTimeoutError::Timeout => Error::DeviceLost {
                    device: self.ctx.ordinal(),
                    detail: "peer producer readiness missed the group deadline".into(),
                },
                std::sync::mpsc::RecvTimeoutError::Disconnected => Error::InvalidRequest {
                    field: "peer_read",
                    detail: "source owner dropped the peer read before publishing readiness".into(),
                },
            })?;
        let source_end =
            source_offset
                .checked_add(byte_count)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "src",
                    detail: "peer copy extent overflowed".into(),
                })?;
        if source_end > handle.bytes {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: "peer copy exceeds the exported source span".into(),
            });
        }
        let destination = span(self, offset, byte_count, "dst")?;
        if stream.device_uuid() != self.device_uuid() {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "a peer copy is enqueued on the destination device stream".into(),
            });
        }
        let grant = self
            .ctx
            .peer_grant(handle.generation)
            .ok_or_else(|| Error::Unsupported {
                capability: "peer_access",
                reason: "this context has no grant for the source acquisition".into(),
            })?;
        let source = handle
            .address
            .checked_add(source_offset as u64)
            .ok_or_else(|| Error::InvalidRequest {
                field: "src",
                detail: "peer source address overflowed".into(),
            })?;
        if byte_count != 0 {
            self.ctx.make_current()?;
            // SAFETY: `destination` is checked against this allocation;
            // `source` and `byte_count` are within the exported span. The
            // generation-keyed grant supplies the live source CUcontext, and
            // the source owner remains borrowed until the completion message.
            // CUDA Driver API 12.9 `cuMemcpyPeerAsync` takes `dstContext` and
            // `srcContext` explicitly; only the destination stream's context
            // is current on this thread. The source handle is not made current.
            let copied = check(
                unsafe {
                    ffi::cuMemcpyPeerAsync(
                        destination,
                        self.ctx.raw(),
                        source,
                        grant.context as ffi::CUcontext,
                        byte_count,
                        stream.raw(),
                    )
                },
                "cuMemcpyPeerAsync",
            );
            if let Err(error) = copied {
                // A refused enqueue did not establish completion. Record a
                // later event on the same stream before acknowledging; this
                // also settles any earlier local work in the operation.
                if Self::observe_stream(stream, deadline).is_ok() {
                    let _ = handle.completed.send(());
                }
                return Err(error);
            }
            Self::observe_stream(stream, deadline)?;
        }
        handle.completed.send(()).map_err(|_| Error::DeviceLost {
            device: self.ctx.ordinal(),
            detail: "source owner disappeared before peer copy acknowledgement".into(),
        })
    }

    fn observe_stream(stream: &Stream<'_>, deadline: Instant) -> Result<()> {
        let complete = Event::new(stream.ctx)?;
        complete.record(stream)?;
        loop {
            if complete.is_complete()? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(Error::DeviceLost {
                    device: stream.ctx.ordinal(),
                    detail: "peer copy completion missed the group deadline".into(),
                });
            }
            std::thread::yield_now();
        }
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        self.copy_to_host_at(0, dst)
    }

    /// Synchronously copy a checked byte range to the host.
    pub fn copy_to_host_at(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| Error::InvalidRequest {
                field: "dst",
                detail: "copy range overflowed".into(),
            })?;
        if end > self.len {
            return Err(Error::InvalidRequest {
                field: "dst",
                detail: format!(
                    "{} bytes at offset {offset} out of a {}-byte buffer",
                    dst.len(),
                    self.len
                ),
            });
        }
        if dst.is_empty() {
            return Ok(());
        }
        let source = self
            .ptr
            .checked_add(offset as u64)
            .ok_or_else(|| Error::InvalidRequest {
                field: "dst",
                detail: "device address overflowed".into(),
            })?;
        self.ctx.make_current()?;
        check(
            // SAFETY: `dst` is a valid writable slice of `dst.len()` bytes and the
            // source holds at least that many, checked above.
            unsafe { ffi::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, source, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Enqueue a device-to-host copy on `stream`.
    ///
    /// # Safety
    /// As [`DeviceBuffer::copy_from_host_async`]: `dst` and this buffer must
    /// both outlive the completion of the enqueued copy, and `dst` must not be
    /// read until then.
    pub unsafe fn copy_to_host_async(&self, dst: &mut [u8], stream: &Stream<'ctx>) -> Result<()> {
        if dst.len() > self.len {
            return Err(Error::InvalidRequest {
                field: "dst",
                detail: format!("{} bytes out of a {}-byte buffer", dst.len(), self.len),
            });
        }
        if dst.is_empty() {
            return Ok(());
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: bounds checked above; the destination-lifetime obligation
            // is delegated to this function's safety contract.
            unsafe {
                ffi::cuMemcpyDtoHAsync_v2(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    dst.len(),
                    stream.raw(),
                )
            },
            "cuMemcpyDtoHAsync",
        )
    }
    /// Free this allocation, checked, without synchronizing. The caller must
    /// have ordered all prior work — an observed event, never hope: freeing
    /// unordered work hands live pages to the next allocation (R07). Failure
    /// leaves the allocation live for quarantine. On failure the owner must
    /// withhold it rather than run the conservative destructor's retry.
    ///
    /// # Safety
    /// All uses of this allocation must have completed, or nothing was submitted.
    /// No caller may use its raw pointer again after success.
    pub unsafe fn try_free(&mut self) -> Result<()> {
        if self.ptr == 0 {
            return Ok(());
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: live allocation on the current context; ordering is the
            // caller's documented obligation.
            unsafe { ffi::cuMemFree_v2(self.ptr) },
            "cuMemFree",
        )?;
        self.ptr = 0;
        Ok(())
    }
}

/// The device address of `len` bytes at `offset` inside `buffer`, checked.
fn span(buffer: &DeviceBuffer<'_>, offset: usize, len: usize, field: &'static str) -> Result<u64> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidRequest {
            field,
            detail: "copy range overflowed".into(),
        })?;
    if end > buffer.len {
        return Err(Error::InvalidRequest {
            field,
            detail: format!(
                "{len} bytes at offset {offset} in a {}-byte buffer",
                buffer.len
            ),
        });
    }
    buffer
        .ptr
        .checked_add(offset as u64)
        .ok_or_else(|| Error::InvalidRequest {
            field,
            detail: "device address overflowed".into(),
        })
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: the owning context is made current first -- synchronising or
        // freeing under a foreign context would protect the wrong stream and
        // free into the wrong context. Synchronising before the free is then
        // what makes this sound: an asynchronous copy or kernel may still be
        // reading this allocation, and the driver would happily hand the pages
        // to the next allocation. R07.
        //
        // **A failed synchronize is not a teardown error to ignore.** It is the
        // one answer that means "whether anything is still reading this is
        // unknown". When the context cannot be made current, or cannot be
        // synchronized, the allocation is **permanently withheld**: leaking
        // device memory at teardown is a bounded, visible cost, and freeing
        // pages a live copy is reading is a silent wrong answer somewhere else.
        unsafe {
            if ffi::cuCtxSetCurrent(self.ctx.raw()) != ffi::CUDA_SUCCESS {
                self.ptr = 0;
                return;
            }
            if ffi::cuCtxSynchronize() != ffi::CUDA_SUCCESS {
                self.ptr = 0;
                return;
            }
            let _ = ffi::cuMemFree_v2(self.ptr);
        }
    }
}

/// A binary device image whose bytes come from a trusted producer.
///
/// `cuModuleLoadData` takes a bare pointer and **no length**. Its contract is a
/// complete, valid, supported binary image; the driver parses forward from the
/// pointer using the image's own internal sizes. A Rust `&[u8]` carries a length
/// the C API never receives, so slice bounds protect nothing here: an empty,
/// truncated or arbitrary buffer lets the driver's parser read past the end of
/// the allocation. That is why constructing this type is `unsafe` and loading
/// one is not.
///
/// The magic-number check below is a **sanity check, not validation**. It
/// catches an empty or obviously-wrong buffer early and with a typed error. It
/// does not establish that the image is well-formed, and this type must never be
/// described as accepting untrusted input.
#[derive(Debug, Clone, Copy)]
pub struct TrustedImage<'a> {
    bytes: &'a [u8],
}

/// `0xBA55ED50` ("BASSED50"), little-endian, at the start of a fatbin.
const FATBIN_MAGIC: [u8; 4] = [0x50, 0xED, 0x55, 0xBA];
/// ELF magic, at the start of a cubin.
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

impl<'a> TrustedImage<'a> {
    /// Declare that `bytes` is a complete device image from the pinned build.
    ///
    /// # Safety
    /// The caller must guarantee that `bytes` is an entire, unmodified cubin or
    /// fatbin as produced by the pinned CUDA toolchain at build time -- for
    /// example an `include_bytes!` of a `build.rs` output. It must not be
    /// truncated, concatenated, or read from a file at run time without a
    /// separate integrity check. Nothing in this type verifies the image, and
    /// the driver will read beyond `bytes.len()` if the image claims to be
    /// larger.
    pub unsafe fn from_build_output(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::InvalidArtifact {
                detail: format!("device image is {} bytes; too short to be one", bytes.len())
                    .into(),
            });
        }
        let head = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if head != FATBIN_MAGIC && head != ELF_MAGIC {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "device image does not begin with a fatbin or ELF magic: {head:02x?}"
                )
                .into(),
            });
        }
        Ok(Self { bytes })
    }

    fn as_ptr(&self) -> *const c_void {
        self.bytes.as_ptr() as *const c_void
    }
}

/// Validated PTX module text.
///
/// NUL-termination is what the C API requires of PTX, but it is **not** what
/// makes a buffer PTX. `cuModuleLoadData` sniffs the leading bytes and decides
/// for itself whether it is looking at a binary image or at source text; the
/// Rust enum label is never passed to the driver. So a `&CStr` that happens to
/// begin with ELF or fatbin magic would be handed to the binary-image parser
/// through what looks like the text path -- exactly the boundary the trusted-
/// image type exists to keep closed.
///
/// This type closes it by construction: the bytes must be UTF-8 text, must not
/// begin with any image magic the driver recognises, and must carry the
/// `.version` directive that every PTX module opens with.
///
/// That last check is **necessary, not sufficient**. It does not make the text
/// valid PTX -- the driver's parser decides that, and returns
/// `CUDA_ERROR_INVALID_PTX`, which classifies as `UnsupportedKernel`. What it
/// does guarantee is that the driver cannot take this buffer for a binary image.
#[derive(Debug, Clone, Copy)]
pub struct PtxSource<'a> {
    text: &'a CStr,
}

impl<'a> PtxSource<'a> {
    pub fn new(text: &'a CStr) -> Result<Self> {
        let bytes = text.to_bytes();
        if bytes.len() < 4 {
            return Err(Error::InvalidRequest {
                field: "ptx",
                detail: format!("{} bytes is too short to be a PTX module", bytes.len()),
            });
        }
        let head = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if head == FATBIN_MAGIC || head == ELF_MAGIC {
            return Err(Error::InvalidRequest {
                field: "ptx",
                detail: "text begins with a binary image magic; the driver would parse it                          as a cubin or fatbin, not as PTX"
                    .into(),
            });
        }
        let Ok(source) = core::str::from_utf8(bytes) else {
            return Err(Error::InvalidRequest {
                field: "ptx",
                detail: "PTX is text and this is not valid UTF-8".into(),
            });
        };
        // Every PTX module begins with a `.version` directive, ahead of any
        // other directive or instruction. Comments and blank lines may precede
        // it, so this looks for the token rather than the first byte.
        if !source
            .lines()
            .any(|l| l.trim_start().starts_with(".version"))
        {
            return Err(Error::InvalidRequest {
                field: "ptx",
                detail: "no `.version` directive: a PTX module must declare its ISA version".into(),
            });
        }
        Ok(Self { text })
    }

    fn as_ptr(&self) -> *const c_void {
        self.text.as_ptr() as *const c_void
    }
}

/// What `Module::load` accepts.
///
/// Both variants satisfy `cuModuleLoadData`'s contract by construction, and
/// neither can be mistaken for the other by the driver's format sniffing: a
/// `TrustedImage` carries the caller's build-time guarantee and begins with an
/// image magic, and a [`PtxSource`] is NUL-terminated text that is guaranteed
/// not to. An arbitrary `&[u8]`, or a bare `&CStr`, satisfies neither and is
/// deliberately not accepted.
#[derive(Debug, Clone, Copy)]
pub enum ModuleImage<'a> {
    /// A cubin or fatbin from the build.
    Binary(TrustedImage<'a>),
    /// Validated PTX module text.
    Ptx(PtxSource<'a>),
}

/// A loaded device image (fatbin, cubin or PTX).
#[derive(Debug)]
pub struct Module<'ctx> {
    module: ffi::CUmodule,
    /// The context that owns this module, for the same reason as
    /// `DeviceBuffer::ctx`.
    ctx: &'ctx RankContext,
}

impl<'ctx> Module<'ctx> {
    /// Load a device image.
    ///
    /// A fatbin with no binary for the current device fails here with
    /// `UnsupportedKernel`, which is the behaviour M0 asserts: an architecture
    /// we did not compile for must be an error, never a silent no-op.
    pub fn load(ctx: &'ctx RankContext, image: ModuleImage<'_>) -> Result<Self> {
        let ptr = match image {
            ModuleImage::Binary(b) => b.as_ptr(),
            ModuleImage::Ptx(p) => p.as_ptr(),
        };
        // SAFETY: `ModuleImage` cannot be constructed except from a caller's
        // build-time guarantee about a binary image, or from a `PtxSource`,
        // which is NUL-terminated text that has been checked not to begin with
        // an image magic. The driver's format sniffing therefore routes each
        // variant to the parser it was built for, and both outlive the call.
        unsafe { Self::load_raw(ctx, ptr) }
    }

    /// Load from a raw image pointer.
    ///
    /// # Safety
    /// `image` must point at a complete, valid, supported binary image, or at
    /// NUL-terminated PTX, and must remain valid for the duration of the call.
    /// The driver receives no length and will read as far as the image's own
    /// headers direct.
    pub unsafe fn load_raw(ctx: &'ctx RankContext, image: *const c_void) -> Result<Self> {
        ctx.make_current()?;
        let mut module: ffi::CUmodule = core::ptr::null_mut();
        check(
            // SAFETY: delegated to this function's own safety contract, which the
            // caller accepted. The driver copies what it needs before returning.
            unsafe { ffi::cuModuleLoadData(&mut module, image) },
            "cuModuleLoadData",
        )?;
        Ok(Self { module, ctx })
    }

    /// Look up a kernel. The returned `Function` borrows this module, so it
    /// cannot outlive the image it was resolved from.
    pub fn function(&self, name: &str) -> Result<Function<'_>> {
        // **`CString::new` allocates infallibly**, and this is reached from
        // `resolve_all` once per symbol, inside admission, after a device
        // allocation already exists. The bytes are reserved first and the NUL
        // appended, which is what `CString::new` does -- fallibly.
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(name.len() + 1)
            .map_err(|_| Error::CapacityExceeded {
                tier: None,
                requested_bytes: 0,
                available_bytes: 0,
            })?;
        bytes.extend_from_slice(name.as_bytes());
        let cname = CString::new(bytes).map_err(|_| Error::InvalidRequest {
            field: "kernel_name",
            detail: "interior NUL".into(),
        })?;
        // The module belongs to one context, and a symbol lookup is a context
        // operation. Without this the call runs under whatever context the
        // thread last touched.
        self.ctx.make_current()?;
        let mut f: ffi::CUfunction = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter, live module, NUL-terminated name,
            // owning context current.
            unsafe { ffi::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function {
            func: f,
            ctx: self.ctx,
        })
    }

    /// Resolve a complete ordered symbol set before any launch and retain the
    /// module beside the raw function handles. This avoids per-node lookup and
    /// keeps the image loaded until the caller's completion lease retires.
    /// Resolve every symbol, growing **fallibly**.
    ///
    /// `with_capacity` and `to_vec` abort when the allocator refuses, and this
    /// runs inside admission, where a caller has a typed refusal to return and
    /// a reservation to hand back: an infallible allocation here would abort
    /// underneath a caller its own callers were made fallible for.
    pub fn resolve_all(self, names: &[String]) -> Result<ResolvedModule<'ctx>> {
        if names.is_empty() {
            return Err(Error::InvalidRequest {
                field: "kernel_symbols",
                detail: "a kernel package must resolve at least one symbol".into(),
            });
        }
        let no_room = || Error::CapacityExceeded {
            tier: None,
            requested_bytes: 0,
            available_bytes: 0,
        };
        let mut functions = Vec::new();
        functions
            .try_reserve_exact(names.len())
            .map_err(|_| no_room())?;
        let mut owned: Vec<String> = Vec::new();
        owned
            .try_reserve_exact(names.len())
            .map_err(|_| no_room())?;
        for name in names {
            let function = self.function(name)?;
            functions.push(function.func);
            let mut copy = String::new();
            copy.try_reserve_exact(name.len()).map_err(|_| no_room())?;
            copy.push_str(name);
            owned.push(copy);
        }
        Ok(ResolvedModule {
            module: self,
            names: owned,
            functions,
        })
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        // SAFETY: the owning context is made current before unloading, for the
        // same reason as `DeviceBuffer`. Teardown errors are not actionable.
        unsafe {
            let _ = ffi::cuCtxSetCurrent(self.ctx.raw());
            let _ = ffi::cuModuleUnload(self.module);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Function<'m> {
    func: ffi::CUfunction,
    /// Borrowed from the owning `Module`, which borrows it from the context.
    /// A `Function` therefore cannot outlive either.
    ctx: &'m RankContext,
}

/// One loaded image with every required symbol resolved in declared order.
#[derive(Debug)]
pub struct ResolvedModule<'ctx> {
    module: Module<'ctx>,
    names: Vec<String>,
    functions: Vec<ffi::CUfunction>,
}

impl ResolvedModule<'_> {
    pub fn symbols(&self) -> &[String] {
        &self.names
    }

    /// Enqueue one already-resolved function on a caller-owned stream.
    ///
    /// # Safety
    /// `params` must exactly match the selected symbol ABI, and every device
    /// address must remain valid until a later event on `stream` completes.
    pub unsafe fn launch_async(
        &self,
        index: usize,
        stream: &Stream<'_>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        if stream.device_uuid() != self.module.ctx.uuid() {
            return Err(Error::InvalidRequest {
                field: "stream",
                detail: "kernel package and stream belong to different devices".into(),
            });
        }
        let function = *self
            .functions
            .get(index)
            .ok_or_else(|| Error::InvalidRequest {
                field: "kernel_symbol",
                detail: format!("resolved symbol index {index} is out of range"),
            })?;
        self.module.ctx.make_current()?;
        check(
            // SAFETY: delegated to this method's contract. The function came
            // from the retained live module and the stream/device was checked.
            unsafe {
                ffi::cuLaunchKernel(
                    function,
                    grid.0 as c_uint,
                    grid.1 as c_uint,
                    grid.2 as c_uint,
                    block.0 as c_uint,
                    block.1 as c_uint,
                    block.2 as c_uint,
                    shared_bytes as c_uint,
                    stream.raw(),
                    params.as_mut_ptr(),
                    core::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Function<'_> {
    /// Launch on the default stream and synchronise.
    ///
    /// # Safety
    /// The caller must guarantee that `params` matches the kernel's signature in
    /// count, order, type and size, and that every device pointer among them is
    /// valid and large enough for what the kernel will access. CUDA cannot check
    /// this, which is why this wrapper is narrow and its callers are few.
    pub unsafe fn launch_blocking(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: delegated to this function's own safety contract, which the
            // caller accepted. Handles and geometry are validated by the driver,
            // which returns an error rather than faulting for bad dimensions.
            unsafe {
                ffi::cuLaunchKernel(
                    self.func,
                    grid.0 as c_uint,
                    grid.1 as c_uint,
                    grid.2 as c_uint,
                    block.0 as c_uint,
                    block.1 as c_uint,
                    block.2 as c_uint,
                    shared_bytes as c_uint,
                    core::ptr::null_mut(),
                    params.as_mut_ptr(),
                    core::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        // A launch is asynchronous, so a fault surfaces at the next
        // synchronisation. Attributing it here keeps the error at its launch
        // site rather than blaming whatever ran next (document 02).
        self.ctx.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only image-boundary checks that need no device live here. The error
    // classification is tested in `crate::status`, which the host lane compiles
    // without this feature at all.

    #[test]
    fn an_empty_or_non_image_buffer_is_refused_before_the_driver_sees_it() {
        // The whole F4 point: a Rust slice cannot establish `cuModuleLoadData`'s
        // valid-image contract, so an obviously-wrong buffer must not reach it.
        // SAFETY: these calls only exercise the header check and never load.
        unsafe {
            assert!(TrustedImage::from_build_output(&[]).is_err());
            assert!(TrustedImage::from_build_output(&[0u8; 3]).is_err());
            assert!(TrustedImage::from_build_output(&[0xDE, 0xAD, 0xBE, 0xEF]).is_err());
        }
    }

    #[test]
    fn ptx_text_cannot_be_mistaken_for_a_binary_image() {
        // `ModuleImage::Ptx(&CStr)` must not pass its pointer to the same
        // auto-detecting entry point a binary image uses, or a short
        // NUL-terminated buffer beginning with ELF magic would reach the
        // binary-image parser through the text path. The Rust label is not
        // passed to CUDA.
        let elf = CString::new([0x7Fu8, b'E', b'L', b'F', b'x'].as_slice()).unwrap();
        assert!(PtxSource::new(&elf).is_err());
        let fatbin = CString::new([0x50u8, 0xED, 0x55, 0xBA, b'x'].as_slice()).unwrap();
        assert!(PtxSource::new(&fatbin).is_err());
    }

    #[test]
    fn ptx_must_be_text_that_declares_its_version() {
        assert!(PtxSource::new(c"").is_err());
        assert!(PtxSource::new(c"abc").is_err());
        assert!(
            PtxSource::new(c"this is not ptx").is_err(),
            "no .version directive"
        );
        // Non-UTF-8 is not source text.
        let bad = CString::new([0xC3u8, 0x28, 0x2E, 0x76, 0x65].as_slice()).unwrap();
        assert!(PtxSource::new(&bad).is_err());

        // A minimal well-formed module header is accepted. This says the driver
        // will treat it as text, not that the text compiles.
        assert!(PtxSource::new(c"//\n.version 8.0\n.target sm_86\n").is_ok());
        assert!(PtxSource::new(c".version 8.0\n").is_ok());
    }

    #[test]
    fn fatbin_and_elf_headers_are_accepted() {
        // SAFETY: header check only; nothing is loaded, so the build-output
        // guarantee is vacuous here.
        unsafe {
            assert!(TrustedImage::from_build_output(&[0x50, 0xED, 0x55, 0xBA, 0, 0]).is_ok());
            assert!(TrustedImage::from_build_output(&[0x7F, b'E', b'L', b'F', 0, 0]).is_ok());
        }
    }
}
