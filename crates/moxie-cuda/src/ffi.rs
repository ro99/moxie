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

// Declared in `crate::status` so the classification and its tests compile with
// no driver present; re-exported here so the FFI signatures read normally.
pub use crate::status::{CUDA_SUCCESS, CUresult};

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

// Opaque handles. Represented as pointers; never dereferenced on this side.
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUarray = *mut c_void;
pub type CUmemorytype = c_int;

#[cfg(feature = "nccl")]
pub type ncclComm_t = *mut c_void;
#[cfg(feature = "nccl")]
pub type ncclResult_t = c_int;
#[cfg(feature = "nccl")]
pub type ncclDataType_t = c_int;
#[cfg(feature = "nccl")]
pub type ncclRedOp_t = c_int;

#[cfg(feature = "nccl")]
pub const NCCL_SUCCESS: ncclResult_t = 0;
#[cfg(feature = "nccl")]
pub const NCCL_IN_PROGRESS: ncclResult_t = 7;
#[cfg(feature = "nccl")]
pub const NCCL_SUM: ncclRedOp_t = 0;
#[cfg(feature = "nccl")]
pub const NCCL_MAX: ncclRedOp_t = 2;
#[cfg(feature = "nccl")]
pub const NCCL_UINT32: ncclDataType_t = 3;
#[cfg(feature = "nccl")]
pub const NCCL_FLOAT32: ncclDataType_t = 7;
#[cfg(feature = "nccl")]
pub const NCCL_BFLOAT16: ncclDataType_t = 9;
#[cfg(feature = "nccl")]
pub const NCCL_CONFIG_UNDEF_INT: c_int = c_int::MIN;
#[cfg(feature = "nccl")]
pub const NCCL_API_MAGIC: c_uint = 0xcafe_beef;

#[cfg(feature = "nccl")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [c_char; 128],
}

#[cfg(feature = "nccl")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ncclConfig_t {
    pub size: usize,
    pub magic: c_uint,
    pub version: c_uint,
    pub blocking: c_int,
    pub cga_cluster_size: c_int,
    pub min_ctas: c_int,
    pub max_ctas: c_int,
    pub net_name: *const c_char,
    pub split_share: c_int,
    pub traffic_class: c_int,
    pub comm_name: *const c_char,
    pub collnet_enable: c_int,
    pub cta_policy: c_int,
    pub shrink_share: c_int,
    pub nvls_ctas: c_int,
    pub channels_per_net_peer: c_int,
    pub nvlink_centric_sched: c_int,
    pub graph_usage_mode: c_int,
    pub num_rma_ctx: c_int,
    pub max_p2p_peers: c_int,
    pub graph_stream_ordering: c_int,
    pub launch_order_implicit: c_int,
    pub num_rma_sig: c_int,
    pub rma_eager_init: c_int,
    pub host_cft_mode: c_int,
}

#[cfg(feature = "nccl")]
impl ncclConfig_t {
    pub const fn initializer(version: c_uint) -> Self {
        Self {
            size: core::mem::size_of::<Self>(),
            magic: NCCL_API_MAGIC,
            version,
            blocking: NCCL_CONFIG_UNDEF_INT,
            cga_cluster_size: NCCL_CONFIG_UNDEF_INT,
            min_ctas: NCCL_CONFIG_UNDEF_INT,
            max_ctas: NCCL_CONFIG_UNDEF_INT,
            net_name: core::ptr::null(),
            split_share: NCCL_CONFIG_UNDEF_INT,
            traffic_class: NCCL_CONFIG_UNDEF_INT,
            comm_name: core::ptr::null(),
            collnet_enable: NCCL_CONFIG_UNDEF_INT,
            cta_policy: NCCL_CONFIG_UNDEF_INT,
            shrink_share: NCCL_CONFIG_UNDEF_INT,
            nvls_ctas: NCCL_CONFIG_UNDEF_INT,
            channels_per_net_peer: NCCL_CONFIG_UNDEF_INT,
            nvlink_centric_sched: NCCL_CONFIG_UNDEF_INT,
            graph_usage_mode: NCCL_CONFIG_UNDEF_INT,
            num_rma_ctx: NCCL_CONFIG_UNDEF_INT,
            max_p2p_peers: NCCL_CONFIG_UNDEF_INT,
            graph_stream_ordering: NCCL_CONFIG_UNDEF_INT,
            launch_order_implicit: NCCL_CONFIG_UNDEF_INT,
            num_rma_sig: NCCL_CONFIG_UNDEF_INT,
            rma_eager_init: NCCL_CONFIG_UNDEF_INT,
            host_cft_mode: NCCL_CONFIG_UNDEF_INT,
        }
    }
}

#[cfg(feature = "cublas")]
pub type cublasHandle_t = *mut c_void;

#[cfg(feature = "cublas")]
pub type cublasStatus_t = c_int;

#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 11;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;
#[cfg(feature = "cublas")]
pub const CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = 16;

#[cfg(feature = "cublas")]
pub const CUBLAS_OP_N: c_int = 0;
#[cfg(feature = "cublas")]
pub const CUBLAS_OP_T: c_int = 1;
#[cfg(feature = "cublas")]
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;
#[cfg(feature = "cublas")]
pub const CUBLAS_DEFAULT_MATH: c_int = 0;
#[cfg(feature = "cublas")]
pub const CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: c_int = 16;
#[cfg(feature = "cublas")]
pub const CUBLAS_PROPERTY_MAJOR_VERSION: c_int = 0;
#[cfg(feature = "cublas")]
pub const CUBLAS_PROPERTY_MINOR_VERSION: c_int = 1;
#[cfg(feature = "cublas")]
pub const CUBLAS_PROPERTY_PATCH_LEVEL: c_int = 2;
#[cfg(feature = "cublas")]
pub const CUBLAS_COMPUTE_32F: c_int = 68;
#[cfg(feature = "cublas")]
pub const CUDA_R_16BF: c_int = 14;
#[cfg(feature = "cublas")]
pub const CUDA_R_32F: c_int = 0;

pub const CU_MEMORYTYPE_DEVICE: CUmemorytype = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUDA_MEMCPY2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: CUmemorytype,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: CUarray,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: CUmemorytype,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: CUarray,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CUuuid {
    pub bytes: [c_char; 16],
}

// Device attribute ordinals, from cuda.h. These are ABI-stable.
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

// Stream creation flags.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_uint = 1;

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
    pub fn cuCtxEnablePeerAccess(peer_context: CUcontext, flags: c_uint) -> CUresult;
    pub fn cuCtxDisablePeerAccess(peer_context: CUcontext) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemsetD32Async(
        dst_device: CUdeviceptr,
        value: c_uint,
        count: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpy2DAsync_v2(copy: *const CUDA_MEMCPY2D, stream: CUstream) -> CUresult;
    pub fn cuMemcpyPeerAsync(
        dst: CUdeviceptr,
        dst_context: CUcontext,
        src: CUdeviceptr,
        src_context: CUcontext,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;
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
    pub fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult;
    pub fn cuLaunchHostFunc(
        stream: CUstream,
        callback: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: c_uint) -> CUresult;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> CUresult;

    pub fn cuGraphInstantiateWithFlags(
        graph_exec: *mut CUgraphExec,
        graph: CUgraph,
        flags: u64,
    ) -> CUresult;
    pub fn cuGraphLaunch(graph_exec: CUgraphExec, stream: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(graph_exec: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(graph: CUgraph) -> CUresult;
    pub fn cuGraphGetNodes(
        graph: CUgraph,
        nodes: *mut CUgraphNode,
        num_nodes: *mut usize,
    ) -> CUresult;

    #[cfg(feature = "nccl")]
    pub fn ncclGetVersion(version: *mut c_int) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclCommInitRankConfig(
        comm: *mut ncclComm_t,
        nranks: c_int,
        unique_id: ncclUniqueId,
        rank: c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclCommGetAsyncError(comm: ncclComm_t, async_error: *mut ncclResult_t) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclAllReduce(
        send_buffer: *const c_void,
        receive_buffer: *mut c_void,
        count: usize,
        data_type: ncclDataType_t,
        operation: ncclRedOp_t,
        comm: ncclComm_t,
        stream: CUstream,
    ) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclAllGather(
        send_buffer: *const c_void,
        receive_buffer: *mut c_void,
        send_count: usize,
        data_type: ncclDataType_t,
        comm: ncclComm_t,
        stream: CUstream,
    ) -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclGetErrorString(result: ncclResult_t) -> *const c_char;
    #[cfg(feature = "nccl")]
    pub fn ncclGroupStart() -> ncclResult_t;
    #[cfg(feature = "nccl")]
    pub fn ncclGroupEnd() -> ncclResult_t;

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

    #[cfg(feature = "cublas")]
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasSetWorkspace_v2(
        handle: cublasHandle_t,
        workspace: *mut c_void,
        workspace_bytes: usize,
    ) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasSetMathMode(handle: cublasHandle_t, math_mode: c_int) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasGetProperty(property_type: c_int, value: *mut c_int) -> cublasStatus_t;
    #[cfg(feature = "cublas")]
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> cublasStatus_t;
}
