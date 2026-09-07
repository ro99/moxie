//! Typed CUDA driver wrapper.
//!
//! Responsibility (document 02): a safe-ish typed CUDA wrapper. It owns device
//! contexts, allocations, streams, events and module loading. It must not know
//! about models, placement policy or checkpoint discovery.
//!
//! Two contracts from document 02 are load-bearing here and are tested:
//!
//! * Errors are **typed**. A `CUresult` is mapped to a `moxie_types::Error`
//!   variant by code, never by matching the driver's message text.
//! * A buffer's lifetime is tied to completion, not to the return of an enqueue
//!   call (R07). `DeviceBuffer` therefore synchronises on drop rather than
//!   freeing memory that an in-flight copy may still be reading.

use core::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ffi::CString;

use moxie_types::{DeviceCapability, Error, Result};

pub mod ffi;

/// Map a driver result to a typed error.
///
/// Document 02: "Errors are typed ... not classified by matching CUDA
/// error-message strings." The driver's string is attached as *detail* for a
/// human, but the variant is chosen from the numeric code alone.
fn check(code: ffi::CUresult, context: &str) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    let detail = format!("{context}: {} ({code})", error_string(code));
    // Codes from cuda.h, grouped by what a caller must do about them.
    match code {
        // CUDA_ERROR_OUT_OF_MEMORY
        2 => Err(Error::CapacityExceeded {
            tier: "device",
            requested_bytes: 0,
            available_bytes: 0,
        }),
        // DEINITIALIZED, LAUNCH_FAILED, ILLEGAL_ADDRESS, CONTEXT_IS_DESTROYED,
        // ECC_UNCORRECTABLE, HARDWARE_STACK_ERROR: the context is unusable.
        4 | 700 | 709 | 719 | 214 | 714 => Err(Error::DeviceLost {
            device: u32::MAX,
            detail,
        }),
        // NOT_INITIALIZED. This is our defect, not the machine's: the driver
        // API was called before `cuInit`. Kept distinct so it can never be
        // misreported as absent hardware.
        3 => Err(Error::InvalidRequest {
            field: "cuda_init",
            detail,
        }),
        // NO_DEVICE, INVALID_DEVICE
        100 | 101 => Err(Error::Unsupported {
            capability: "cuda_device",
            reason: detail,
        }),
        // NOT_FOUND, INVALID_PTX, UNSUPPORTED_PTX_VERSION, NO_BINARY_FOR_GPU
        500 | 218 | 222 | 209 => Err(Error::UnsupportedKernel {
            operation: "module",
            detail,
        }),
        _ => Err(Error::Numerical { detail }),
    }
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
    let uuid = format_uuid(&uuid);

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

fn format_uuid(u: &ffi::CUuuid) -> String {
    let b: Vec<u8> = u.bytes.iter().map(|c| *c as u8).collect();
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
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
        check(
            // SAFETY: `ctx` is a live context we just retained.
            unsafe { ffi::cuCtxSetCurrent(ctx) },
            "cuCtxSetCurrent",
        )?;
        Ok(Self {
            device,
            ctx,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
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

/// A device allocation.
///
/// R07 and document 02: "A kernel launch leases its inputs/outputs/workspace
/// until completion. Retirement is event-driven; Rust `Drop` alone must not free
/// in-flight CUDA memory." This M0 type discharges that obligation the blunt way
/// -- it synchronises the context before freeing. The real engine replaces this
/// with an event-retained lease; the invariant is the same, and this type must
/// not be "optimised" by simply deleting the synchronise.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: ffi::CUdeviceptr,
    len: usize,
}

impl DeviceBuffer {
    pub fn alloc(ctx: &DeviceContext, len: usize) -> Result<Self> {
        ctx.make_current()?;
        if len == 0 {
            return Ok(Self { ptr: 0, len: 0 });
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
        Ok(Self { ptr, len })
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

    pub fn copy_from_host(&mut self, ctx: &DeviceContext, src: &[u8]) -> Result<()> {
        if src.len() > self.len {
            return Err(Error::InvalidRequest {
                field: "src",
                detail: format!("{} bytes into a {}-byte buffer", src.len(), self.len),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        ctx.make_current()?;
        check(
            // SAFETY: `src` is a valid readable slice of `src.len()` bytes and the
            // destination holds at least that many, checked above. The synchronous
            // form returns only once the copy completed, so `src` needs no lifetime
            // extension -- the asynchronous form would (R07).
            unsafe { ffi::cuMemcpyHtoD_v2(self.ptr, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    pub fn copy_to_host(&self, ctx: &DeviceContext, dst: &mut [u8]) -> Result<()> {
        if dst.len() > self.len {
            return Err(Error::InvalidRequest {
                field: "dst",
                detail: format!("{} bytes out of a {}-byte buffer", dst.len(), self.len),
            });
        }
        if dst.is_empty() {
            return Ok(());
        }
        ctx.make_current()?;
        check(
            // SAFETY: `dst` is a valid writable slice of `dst.len()` bytes and the
            // source holds at least that many, checked above.
            unsafe { ffi::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len()) },
            "cuMemcpyDtoH",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: synchronising before the free is what makes this sound. An
        // asynchronous copy or kernel may still be reading this allocation, and
        // the driver would happily hand the pages to the next allocation. R07.
        unsafe {
            let _ = ffi::cuCtxSynchronize();
            let _ = ffi::cuMemFree_v2(self.ptr);
        }
    }
}

/// A loaded device image (fatbin, cubin or PTX).
#[derive(Debug)]
pub struct Module {
    module: ffi::CUmodule,
}

impl Module {
    /// Load a device image.
    ///
    /// A fatbin with no binary for the current device fails here with
    /// `UnsupportedKernel`, which is the behaviour M0 asserts: an architecture
    /// we did not compile for must be an error, never a silent no-op.
    pub fn load(ctx: &DeviceContext, image: &[u8]) -> Result<Self> {
        ctx.make_current()?;
        let mut module: ffi::CUmodule = core::ptr::null_mut();
        check(
            // SAFETY: `image` outlives the call; the driver copies what it needs.
            unsafe { ffi::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Self { module })
    }

    pub fn function(&self, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| Error::InvalidRequest {
            field: "kernel_name",
            detail: "interior NUL".into(),
        })?;
        let mut f: ffi::CUfunction = core::ptr::null_mut();
        check(
            // SAFETY: valid out-parameter, live module, NUL-terminated name.
            unsafe { ffi::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { func: f })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: unloading a module we loaded; errors are not actionable here.
        unsafe {
            let _ = ffi::cuModuleUnload(self.module);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Function {
    func: ffi::CUfunction,
}

impl Function {
    /// Launch on the default stream and synchronise.
    ///
    /// # Safety
    /// The caller must guarantee that `params` matches the kernel's signature in
    /// count, order, type and size, and that every device pointer among them is
    /// valid and large enough for what the kernel will access. CUDA cannot check
    /// this, which is why this wrapper is narrow and its callers are few.
    pub unsafe fn launch_blocking(
        &self,
        ctx: &DeviceContext,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        ctx.make_current()?;
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
        ctx.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure tests: they exercise the error mapping without a device, so the host
    // CI lane runs them. Real device behaviour is `cargo xtask test-gpu`.

    #[test]
    fn success_is_not_an_error() {
        assert!(check(ffi::CUDA_SUCCESS, "ctx").is_ok());
    }

    #[test]
    fn out_of_memory_maps_to_capacity_exceeded() {
        let e = check(2, "cuMemAlloc").unwrap_err();
        assert_eq!(e.kind(), "capacity_exceeded");
        assert!(!e.is_retryable());
    }

    #[test]
    fn missing_binary_for_gpu_maps_to_unsupported_kernel() {
        // NO_BINARY_FOR_GPU is what a fatbin returns when it has no image for
        // the current architecture -- directly an M0 exit-gate case.
        for code in [209, 218, 222, 500] {
            assert_eq!(
                check(code, "cuModuleLoadData").unwrap_err().kind(),
                "unsupported_kernel",
                "code {code}"
            );
        }
    }

    #[test]
    fn fatal_device_codes_map_to_device_lost() {
        for code in [4, 700, 709, 719, 214, 714] {
            assert_eq!(
                check(code, "ctx").unwrap_err().kind(),
                "device_lost",
                "code {code}"
            );
        }
    }

    #[test]
    fn uninitialised_driver_is_not_reported_as_missing_hardware() {
        // A wrapper defect and an empty machine are different problems and must
        // not share an error kind.
        let e = check(3, "cuDeviceGetCount").unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert_ne!(e.kind(), check(100, "cuInit").unwrap_err().kind());
    }

    #[test]
    fn no_device_maps_to_unsupported_not_to_a_panic() {
        assert_eq!(check(100, "cuInit").unwrap_err().kind(), "unsupported");
        assert_eq!(check(101, "cuDeviceGet").unwrap_err().kind(), "unsupported");
    }

    #[test]
    fn variant_comes_from_the_code_not_the_message() {
        match check(2, "alloc").unwrap_err() {
            Error::CapacityExceeded { tier, .. } => assert_eq!(tier, "device"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn uuid_formats_as_the_canonical_gpu_string() {
        let mut u = ffi::CUuuid::default();
        for (i, b) in [
            0x97u8, 0xfe, 0x48, 0x89, 0x48, 0x74, 0xa3, 0x78, 0x19, 0x8e, 0x95, 0x5d, 0x2e, 0x72,
            0xc3, 0xa3,
        ]
        .iter()
        .enumerate()
        {
            u.bytes[i] = *b as c_char;
        }
        assert_eq!(format_uuid(&u), "GPU-97fe4889-4874-a378-198e-955d2e72c3a3");
    }
}
