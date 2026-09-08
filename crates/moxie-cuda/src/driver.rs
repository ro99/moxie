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

use moxie_types::{DeviceCapability, Error, Result};

use crate::ffi;
use crate::status::{classify, format_uuid};

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
    let uuid = format_uuid(&uuid_bytes);

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
    })
}

/// A rank-owned device context.
///
/// Document 01: "Local single-process control with one rank execution thread per
/// GPU is the initial topology. Isolate unsafe CUDA state behind rank-owned
/// contexts." Holding this type is what makes a device current for the thread.
#[derive(Debug)]
pub struct DeviceContext {
    device: ffi::CUdevice,
    ctx: ffi::CUcontext,
    ordinal: u32,
}

impl DeviceContext {
    pub fn new(ordinal: u32) -> Result<Self> {
        init()?;
        let mut device: ffi::CUdevice = 0;
        check(
            // SAFETY: valid out-parameter; the driver validates the ordinal.
            unsafe { ffi::cuDeviceGet(&mut device, ordinal as c_int) },
            "cuDeviceGet",
        )?;
        let mut ctx: ffi::CUcontext = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter. The primary context is reference-counted
            // by the driver; `Drop` releases exactly the reference taken here.
            unsafe { ffi::cuDevicePrimaryCtxRetain(&mut ctx, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        // From here the retain must be balanced on **every** path. An earlier
        // version returned the `cuCtxSetCurrent` error directly and leaked the
        // reference, which keeps the primary context alive for the life of the
        // process and pins its memory.
        if let Err(e) = check(
            // SAFETY: `ctx` is a live context we just retained.
            unsafe { ffi::cuCtxSetCurrent(ctx) },
            "cuCtxSetCurrent",
        ) {
            // SAFETY: releases exactly the reference taken immediately above.
            unsafe {
                let _ = ffi::cuDevicePrimaryCtxRelease_v2(device);
            }
            return Err(e);
        }
        Ok(Self {
            device,
            ctx,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
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

    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        check(
            // SAFETY: context is current.
            unsafe { ffi::cuCtxSynchronize() },
            "cuCtxSynchronize",
        )
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        // SAFETY: releases exactly the primary-context reference taken in `new`.
        // Errors during teardown are not actionable and must not panic.
        unsafe {
            let _ = ffi::cuDevicePrimaryCtxRelease_v2(self.device);
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
    ctx: &'ctx DeviceContext,
}

impl<'ctx> Stream<'ctx> {
    pub fn new(ctx: &'ctx DeviceContext) -> Result<Self> {
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

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.make_current()?;
        check(
            // SAFETY: live stream created on the context made current above.
            unsafe { ffi::cuStreamSynchronize(self.stream) },
            "cuStreamSynchronize",
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
    ctx: &'ctx DeviceContext,
}

impl<'ctx> Event<'ctx> {
    pub fn new(ctx: &'ctx DeviceContext) -> Result<Self> {
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

/// A device allocation.
///
/// R07 and document 02: "A kernel launch leases its inputs/outputs/workspace
/// until completion. Retirement is event-driven; Rust `Drop` alone must not free
/// in-flight CUDA memory." This type discharges that obligation the blunt way
/// -- it synchronises the context before freeing. The real engine replaces this
/// with an event-retained lease; the invariant is the same, and this type must
/// not be "optimised" by simply deleting the synchronise.
/// A buffer cannot outlive the context that owns it. The compiler enforces it:
///
/// ```compile_fail
/// use moxie_cuda::{DeviceBuffer, DeviceContext};
/// let escaped = {
///     let ctx = DeviceContext::new(0).unwrap();
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
    ctx: &'ctx DeviceContext,
}

impl<'ctx> DeviceBuffer<'ctx> {
    pub fn alloc(ctx: &'ctx DeviceContext, len: usize) -> Result<Self> {
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

    /// Enqueue a host-to-device copy on `stream`.
    ///
    /// # Safety
    /// This returns before the copy has happened. The caller must keep `src`
    /// alive and unmodified, and must keep this buffer alive, until an event
    /// recorded after the copy has completed. That is exactly the obligation
    /// R07 records as the one the legacy engine got wrong; there is no lease
    /// mechanism at M0 to discharge it automatically, so it is the caller's.
    pub unsafe fn copy_from_host_async(&mut self, src: &[u8], stream: &Stream<'ctx>) -> Result<()> {
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
            // SAFETY: bounds checked above; the source-lifetime obligation is
            // delegated to this function's own safety contract, which the caller
            // accepted.
            unsafe {
                ffi::cuMemcpyHtoDAsync_v2(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    stream.raw(),
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
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
            // SAFETY: `dst` is a valid writable slice of `dst.len()` bytes and the
            // source holds at least that many, checked above.
            unsafe { ffi::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len()) },
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
        // to the next allocation. R07. Teardown errors are not actionable.
        unsafe {
            let _ = ffi::cuCtxSetCurrent(self.ctx.raw());
            let _ = ffi::cuCtxSynchronize();
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
                detail: format!("device image is {} bytes; too short to be one", bytes.len()),
            });
        }
        let head = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if head != FATBIN_MAGIC && head != ELF_MAGIC {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "device image does not begin with a fatbin or ELF magic: {head:02x?}"
                ),
            });
        }
        Ok(Self { bytes })
    }

    fn as_ptr(&self) -> *const c_void {
        self.bytes.as_ptr() as *const c_void
    }
}

/// What `Module::load` accepts.
///
/// Both variants satisfy `cuModuleLoadData`'s contract by construction: a
/// `TrustedImage` carries the caller's build-time guarantee, and a `&CStr` is
/// NUL-terminated, which is what the C API requires of PTX. An arbitrary
/// `&[u8]` satisfies neither and is deliberately not accepted.
#[derive(Debug, Clone, Copy)]
pub enum ModuleImage<'a> {
    /// A cubin or fatbin from the build.
    Binary(TrustedImage<'a>),
    /// PTX source text, NUL-terminated.
    Ptx(&'a CStr),
}

/// A loaded device image (fatbin, cubin or PTX).
#[derive(Debug)]
pub struct Module<'ctx> {
    module: ffi::CUmodule,
    /// The context that owns this module, for the same reason as
    /// `DeviceBuffer::ctx`.
    ctx: &'ctx DeviceContext,
}

impl<'ctx> Module<'ctx> {
    /// Load a device image.
    ///
    /// A fatbin with no binary for the current device fails here with
    /// `UnsupportedKernel`, which is the behaviour M0 asserts: an architecture
    /// we did not compile for must be an error, never a silent no-op.
    pub fn load(ctx: &'ctx DeviceContext, image: ModuleImage<'_>) -> Result<Self> {
        let ptr = match image {
            ModuleImage::Binary(b) => b.as_ptr(),
            ModuleImage::Ptx(p) => p.as_ptr() as *const c_void,
        };
        // SAFETY: `ModuleImage` cannot be constructed except from a caller's
        // build-time guarantee about a binary image, or from a `&CStr`, which is
        // NUL-terminated as `cuModuleLoadData` requires for PTX. Both outlive
        // the call.
        unsafe { Self::load_raw(ctx, ptr) }
    }

    /// Load from a raw image pointer.
    ///
    /// # Safety
    /// `image` must point at a complete, valid, supported binary image, or at
    /// NUL-terminated PTX, and must remain valid for the duration of the call.
    /// The driver receives no length and will read as far as the image's own
    /// headers direct.
    pub unsafe fn load_raw(ctx: &'ctx DeviceContext, image: *const c_void) -> Result<Self> {
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
        let cname = CString::new(name).map_err(|_| Error::InvalidRequest {
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
    ctx: &'m DeviceContext,
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
    fn fatbin_and_elf_headers_are_accepted() {
        // SAFETY: header check only; nothing is loaded, so the build-output
        // guarantee is vacuous here.
        unsafe {
            assert!(TrustedImage::from_build_output(&[0x50, 0xED, 0x55, 0xBA, 0, 0]).is_ok());
            assert!(TrustedImage::from_build_output(&[0x7F, b'E', b'L', b'F', 0, 0]).is_ok());
        }
    }
}
