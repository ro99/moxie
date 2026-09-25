//! Explicitly owned cuBLAS handle for admitted dense workspaces.

use core::{ffi::c_void, marker::PhantomData};
use std::rc::Rc;

use moxie_types::{Error, Result};

use crate::{RankContext, Stream, ffi};

/// A cuBLAS handle tied to one explicitly acquired CUDA context.
///
/// It owns no stream or workspace borrow: callers bind both before each step.
/// There is deliberately no `Drop` implementation. Forgetting to call
/// [`Blas::destroy`] quarantines the handle rather than destroying it while
/// asynchronous work may still be using it.
#[derive(Debug)]
pub struct Blas<'ctx> {
    handle: ffi::cublasHandle_t,
    ctx: &'ctx RankContext,
    _not_send: PhantomData<Rc<()>>,
}

impl<'ctx> Blas<'ctx> {
    pub fn new(ctx: &'ctx RankContext) -> Result<Self> {
        ctx.make_current()?;
        let mut handle = core::ptr::null_mut();
        check(
            // SAFETY: the output pointer is valid and this context is current.
            unsafe { ffi::cublasCreate_v2(&mut handle) },
            "cublasCreate_v2",
            ctx,
        )?;
        if handle.is_null() {
            return Err(Error::Numerical {
                detail: "cublasCreate_v2 succeeded without returning a handle".into(),
            });
        }
        let loaded_version = match loaded_version(ctx) {
            Ok(version) => version,
            Err(error) => {
                // SAFETY: no work has been submitted through this new handle.
                unsafe {
                    let _ = ffi::cublasDestroy_v2(handle);
                }
                return Err(error);
            }
        };
        let built_version = env!("MOXIE_CUBLAS_FILE_VERSION");
        let built_api_version = built_version
            .split('.')
            .take(3)
            .map(str::parse::<i32>)
            .collect::<core::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::Numerical {
                detail: format!("invalid build-time cuBLAS version {built_version}"),
            })?;
        if built_api_version.as_slice() != loaded_version.as_slice() {
            // SAFETY: no work has been submitted through this handle.
            unsafe {
                let _ = ffi::cublasDestroy_v2(handle);
            }
            return Err(Error::Unsupported {
                capability: "cublas_version",
                reason: format!(
                    "build-time cuBLAS {built_version} does not match loaded cuBLAS {}.{}.{}",
                    loaded_version[0], loaded_version[1], loaded_version[2]
                ),
            });
        }
        if let Err(error) = check(
            // SAFETY: the new handle is live and its owning context is current.
            unsafe {
                ffi::cublasSetMathMode(
                    handle,
                    ffi::CUBLAS_DEFAULT_MATH
                        | ffi::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                )
            },
            "cublasSetMathMode",
            ctx,
        ) {
            // SAFETY: handle creation succeeded and its context is current.
            unsafe {
                let _ = ffi::cublasDestroy_v2(handle);
            }
            return Err(error);
        }
        Ok(Self {
            handle,
            ctx,
            _not_send: PhantomData,
        })
    }

    /// Bind this handle to a stream and its admitted device workspace.
    ///
    /// # Safety
    /// `workspace..workspace + bytes` must be live device memory allocated in
    /// this handle's context, and remain live until all work submitted through
    /// this handle has completed. This method must run outside stream capture.
    pub unsafe fn bind(
        &mut self,
        stream: &Stream<'ctx>,
        workspace: u64,
        bytes: usize,
    ) -> Result<()> {
        if stream.device_uuid() != self.ctx.uuid() || workspace == 0 || bytes == 0 {
            return Err(Error::InvalidRequest {
                field: "cublas_workspace",
                detail: "cuBLAS stream and nonempty workspace must belong to its context".into(),
            });
        }
        self.ctx.make_current()?;
        check(
            // SAFETY: the handle and same-device stream are live and current.
            unsafe { ffi::cublasSetStream_v2(self.handle, stream.raw()) },
            "cublasSetStream_v2",
            self.ctx,
        )?;
        // Setting the stream resets cuBLAS's user workspace, so this must follow it.
        check(
            // SAFETY: the caller guarantees this live workspace belongs to the context.
            unsafe {
                ffi::cublasSetWorkspace_v2(self.handle, workspace as usize as *mut c_void, bytes)
            },
            "cublasSetWorkspace_v2",
            self.ctx,
        )
    }

    /// Enqueue one BF16 GEMM with FP32 accumulation in cuBLAS's order.
    ///
    /// # Safety
    /// `a`, `b`, and `c` must name valid BF16 matrices of the dimensions and
    /// leading dimensions supplied, and must remain live until work on the
    /// bound stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_bf16(
        &self,
        m: u64,
        n: u64,
        k: u64,
        a: u64,
        lda: u64,
        b: u64,
        ldb: u64,
        c: u64,
        ldc: u64,
    ) -> Result<()> {
        let as_i32 = |value: u64, name| {
            i32::try_from(value).map_err(|_| Error::InvalidRequest {
                field: "cublas_gemm",
                detail: format!("{name} exceeds the cuBLAS integer ABI"),
            })
        };
        let (m, n, k, lda, ldb, ldc) = (
            as_i32(m, "m")?,
            as_i32(n, "n")?,
            as_i32(k, "k")?,
            as_i32(lda, "lda")?,
            as_i32(ldb, "ldb")?,
            as_i32(ldc, "ldc")?,
        );
        if [m, n, k, lda, ldb, ldc].iter().any(|&value| value <= 0) || [a, b, c].contains(&0) {
            return Err(Error::InvalidRequest {
                field: "cublas_gemm",
                detail: "GEMM dimensions, leading dimensions, and addresses must be positive"
                    .into(),
            });
        }
        self.ctx.make_current()?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        check(
            // SAFETY: the caller guarantees all matrices live through completion; scalar
            // pointers are consumed by this call and the context is current.
            unsafe {
                ffi::cublasGemmEx(
                    self.handle,
                    ffi::CUBLAS_OP_T,
                    ffi::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast(),
                    a as usize as *const c_void,
                    ffi::CUDA_R_16BF,
                    lda,
                    b as usize as *const c_void,
                    ffi::CUDA_R_16BF,
                    ldb,
                    (&beta as *const f32).cast(),
                    c as usize as *mut c_void,
                    ffi::CUDA_R_16BF,
                    ldc,
                    ffi::CUBLAS_COMPUTE_32F,
                    ffi::CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
            self.ctx,
        )
    }

    /// Destroy the handle after every submitted operation has been observed.
    ///
    /// # Safety
    /// No asynchronous operation may still use this handle.
    pub unsafe fn destroy(self) -> core::result::Result<(), (Self, Error)> {
        if let Err(error) = self.ctx.make_current() {
            return Err((self, error));
        }
        #[cfg(feature = "test-hooks")]
        DESTROY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Err(error) = check(
            // SAFETY: this consumes the live handle while its context is current.
            unsafe { ffi::cublasDestroy_v2(self.handle) },
            "cublasDestroy_v2",
            self.ctx,
        ) {
            return Err((self, error));
        }
        Ok(())
    }
}

fn loaded_version(ctx: &RankContext) -> Result<[i32; 3]> {
    ctx.make_current()?;
    let property = |property, operation| -> Result<i32> {
        let mut value = 0;
        check(
            // SAFETY: `value` points to a live integer output and the context is current.
            unsafe { ffi::cublasGetProperty(property, &mut value) },
            operation,
            ctx,
        )?;
        Ok(value)
    };
    Ok([
        property(
            ffi::CUBLAS_PROPERTY_MAJOR_VERSION,
            "cublasGetProperty(MAJOR_VERSION)",
        )?,
        property(
            ffi::CUBLAS_PROPERTY_MINOR_VERSION,
            "cublasGetProperty(MINOR_VERSION)",
        )?,
        property(
            ffi::CUBLAS_PROPERTY_PATCH_LEVEL,
            "cublasGetProperty(PATCH_LEVEL)",
        )?,
    ])
}

#[cfg(feature = "test-hooks")]
static DESTROY_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub fn destroy_calls() -> usize {
    DESTROY_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

fn check(status: ffi::cublasStatus_t, operation: &'static str, ctx: &RankContext) -> Result<()> {
    if status == ffi::CUBLAS_STATUS_SUCCESS {
        return Ok(());
    }
    let detail = format!("{operation} returned cuBLAS status {status}");
    match status {
        ffi::CUBLAS_STATUS_NOT_INITIALIZED | ffi::CUBLAS_STATUS_INVALID_VALUE => {
            Err(Error::InvalidRequest {
                field: "cublas",
                detail,
            })
        }
        ffi::CUBLAS_STATUS_ALLOC_FAILED => Err(Error::CapacityExceeded {
            tier: None,
            requested_bytes: 0,
            available_bytes: 0,
        }),
        ffi::CUBLAS_STATUS_ARCH_MISMATCH | ffi::CUBLAS_STATUS_NOT_SUPPORTED => {
            Err(Error::Unsupported {
                capability: "cublas",
                reason: detail,
            })
        }
        ffi::CUBLAS_STATUS_MAPPING_ERROR | ffi::CUBLAS_STATUS_EXECUTION_FAILED => {
            Err(Error::DeviceLost {
                device: ctx.ordinal(),
                detail,
            })
        }
        ffi::CUBLAS_STATUS_LICENSE_ERROR => Err(Error::Unsupported {
            capability: "cublas_license",
            reason: detail,
        }),
        _ => Err(Error::Numerical { detail }),
    }
}
