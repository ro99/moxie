//! Raw CUDA Driver API bindings.
//!
//! Hand-written rather than generated or pulled from a wrapper crate. Document
//! 03: "do not adopt a large dependency only from its README", and document 02
//! requires this boundary to be *audited*. The surface here is deliberately the
//! minimum M0 needs; extend it deliberately, with the signature checked against
//! the CUDA driver API reference for the pinned toolkit version.
//!
//! Every symbol below is the versioned (`_v2`) name where one exists. Linking
//! the unversioned alias silently binds a different ABI on some toolkits.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

// Opaque handles. Represented as pointers; never dereferenced on this side.
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CUuuid {
    pub bytes: [c_char; 16],
}

// Device attribute ordinals, from cuda.h. These are ABI-stable.
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

// Stream creation flags.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

// Event creation flags.
pub const CU_EVENT_DEFAULT: c_uint = 0;

// SAFETY (module-wide): these declarations mirror the CUDA Driver API for the
// toolkit pinned in docs/evidence/toolchain.md. Each signature was checked
// against cuda.h. Calling any of them requires the caller to uphold the driver's
// own contract (initialised driver, current context, valid pointers and sizes);
// the safe wrappers in `lib.rs` are where those obligations are discharged.
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, str_: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, str_: *mut *const c_char) -> CUresult;

    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetPCIBusId(pci: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceCanAccessPeer(
        can_access: *mut c_int,
        dev: CUdevice,
        peer_dev: CUdevice,
    ) -> CUresult;

    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const c_void,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;

    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;

    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuEventQuery(event: CUevent) -> CUresult;
    pub fn cuEventElapsedTime_v2(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;

    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}
