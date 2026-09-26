//! Per-context NCCL communicators for the opt-in TP2 path.

use std::marker::PhantomData;
use std::rc::Rc;
use std::time::Instant;

use crate::{RankContext, Stream};
use moxie_types::{Error, Result};

/// One process-wide NCCL clique identifier. Its bytes may cross rank threads.
#[derive(Debug, Clone, Copy)]
pub struct NcclId(#[cfg_attr(not(feature = "nccl"), allow(dead_code))] [u8; 128]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bf16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommState {
    /// NCCL has completed its most recent operation and accepts another call.
    Ready,
    /// The last call is still active; only `poll` is allowed until it is ready.
    InProgress,
}

#[cfg(feature = "nccl")]
mod imp {
    use super::*;
    use crate::ffi;
    use core::ffi::{c_char, c_int, c_void};
    use std::cell::Cell;
    use std::ffi::CStr;
    use std::thread;

    const NCCL_HEADER_VERSION: &str = env!("MOXIE_NCCL_HEADER_VERSION");

    fn header_version() -> Result<(String, c_int)> {
        let mut fields = NCCL_HEADER_VERSION.split('.');
        let major = fields
            .next()
            .and_then(|value| value.parse::<c_int>().ok())
            .ok_or_else(|| invalid("the NCCL header version is malformed"))?;
        let minor = fields
            .next()
            .and_then(|value| value.parse::<c_int>().ok())
            .ok_or_else(|| invalid("the NCCL header version is malformed"))?;
        let patch = fields
            .next()
            .and_then(|value| value.parse::<c_int>().ok())
            .ok_or_else(|| invalid("the NCCL header version is malformed"))?;
        if fields.next().is_some() {
            return Err(invalid("the NCCL header version is malformed"));
        }
        let code = major
            .checked_mul(10_000)
            .and_then(|version| version.checked_add(minor.checked_mul(100)?))
            .and_then(|version| version.checked_add(patch))
            .ok_or_else(|| invalid("the NCCL header version overflows"))?;
        Ok((NCCL_HEADER_VERSION.to_owned(), code))
    }

    fn invalid(detail: impl Into<String>) -> Error {
        Error::InvalidRequest {
            field: "nccl",
            detail: detail.into(),
        }
    }

    fn nccl_error(ctx: &RankContext, operation: &str, code: ffi::ncclResult_t) -> Error {
        let detail = {
            // SAFETY: NCCL returns a static, NUL-terminated message for every
            // result code, including unknown result codes.
            let pointer = unsafe { ffi::ncclGetErrorString(code) };
            if pointer.is_null() {
                format!("{operation} returned NCCL status {code}")
            } else {
                // SAFETY: the documented error string remains valid for the
                // process lifetime.
                let message = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
                format!("{operation}: {message} (NCCL status {code})")
            }
        };
        Error::DeviceLost {
            device: ctx.ordinal(),
            detail,
        }
    }

    fn check(ctx: &RankContext, operation: &str, code: ffi::ncclResult_t) -> Result<()> {
        if code == ffi::NCCL_SUCCESS {
            Ok(())
        } else {
            Err(nccl_error(ctx, operation, code))
        }
    }

    fn check_enqueued(ctx: &RankContext, operation: &str, code: ffi::ncclResult_t) -> Result<()> {
        if code == ffi::NCCL_SUCCESS || code == ffi::NCCL_IN_PROGRESS {
            Ok(())
        } else {
            Err(nccl_error(ctx, operation, code))
        }
    }

    impl NcclId {
        pub fn generate(ctx: &RankContext) -> Result<Self> {
            ctx.make_current()?;
            let mut unique_id = ffi::ncclUniqueId { internal: [0; 128] };
            // SAFETY: output points to the header-sized NCCL identifier.
            check(ctx, "ncclGetUniqueId", unsafe {
                ffi::ncclGetUniqueId(&mut unique_id)
            })?;
            Ok(Self(unique_id.internal.map(|byte| byte as u8)))
        }

        fn raw(self) -> ffi::ncclUniqueId {
            ffi::ncclUniqueId {
                internal: self.0.map(|byte| byte as c_char),
            }
        }
    }

    /// A communicator whose only owner is the rank thread for `ctx`.
    #[derive(Debug)]
    pub struct Communicator<'ctx> {
        ctx: &'ctx RankContext,
        handle: ffi::ncclComm_t,
        finalized: bool,
        in_group: Cell<bool>,
        _not_send: PhantomData<Rc<()>>,
    }

    impl<'ctx> Communicator<'ctx> {
        pub fn init(
            ctx: &'ctx RankContext,
            ranks: i32,
            id: NcclId,
            rank: i32,
            deadline: Instant,
        ) -> Result<Self> {
            if ranks <= 0 || rank < 0 || rank >= ranks {
                return Err(invalid("rank is outside the NCCL communicator"));
            }
            ctx.make_current()?;
            let (header, header_code) = header_version()?;
            let mut loaded_code = 0;
            // SAFETY: loaded_code is a valid output pointer.
            check(ctx, "ncclGetVersion", unsafe {
                ffi::ncclGetVersion(&mut loaded_code)
            })?;
            if loaded_code != header_code {
                return Err(Error::Unsupported {
                    capability: "nccl",
                    reason: format!(
                        "loaded NCCL version {} does not match header version {header}",
                        display_version(loaded_code)
                    ),
                });
            }

            let mut config = ffi::ncclConfig_t::initializer(header_code as u32);
            config.blocking = 0;
            let mut handle = core::ptr::null_mut();
            // SAFETY: NCCL_CONFIG_INITIALIZER fields are set as the header
            // requires; the ID and output handle have the declared ABI.
            let initial = unsafe {
                ffi::ncclCommInitRankConfig(&mut handle, ranks, id.raw(), rank, &mut config)
            };
            if initial != ffi::NCCL_SUCCESS && initial != ffi::NCCL_IN_PROGRESS {
                if !handle.is_null() {
                    // SAFETY: NCCL returned a partial local communicator; abort
                    // it rather than allowing an implicit destructor to race.
                    let _ = unsafe { ffi::ncclCommAbort(handle) };
                }
                return Err(nccl_error(ctx, "ncclCommInitRankConfig", initial));
            }
            if handle.is_null() {
                return Err(invalid("NCCL initialization returned a null communicator"));
            }
            let communicator = Self {
                ctx,
                handle,
                finalized: false,
                in_group: Cell::new(false),
                _not_send: PhantomData,
            };
            loop {
                match communicator.poll() {
                    Ok(CommState::Ready) => return Ok(communicator),
                    Ok(CommState::InProgress) if Instant::now() < deadline => thread::yield_now(),
                    Ok(CommState::InProgress) => {
                        let error = Error::DeviceLost {
                            device: ctx.ordinal(),
                            detail: "NCCL initialization missed the startup deadline".into(),
                        };
                        let _ = communicator.abort();
                        return Err(error);
                    }
                    Err(error) => {
                        let _ = communicator.abort();
                        return Err(error);
                    }
                }
            }
        }

        /// Poll communicator work; no later NCCL call is issued until this is
        /// `Ready` after an earlier call returned `ncclInProgress`.
        pub fn poll(&self) -> Result<CommState> {
            self.ctx.make_current()?;
            let mut asynchronous = ffi::NCCL_SUCCESS;
            // SAFETY: the handle is live and asynchronous points to a valid
            // result output.
            let returned = unsafe { ffi::ncclCommGetAsyncError(self.handle, &mut asynchronous) };
            if returned != ffi::NCCL_SUCCESS {
                return Err(nccl_error(self.ctx, "ncclCommGetAsyncError", returned));
            }
            match asynchronous {
                ffi::NCCL_SUCCESS => Ok(CommState::Ready),
                ffi::NCCL_IN_PROGRESS => Ok(CommState::InProgress),
                code => Err(nccl_error(
                    self.ctx,
                    "NCCL communicator asynchronous operation",
                    code,
                )),
            }
        }

        /// Start a rank-local group of collectives, ended by `group_end`.
        pub fn group_start(&self) -> Result<()> {
            if self.in_group.replace(true) {
                return Err(invalid("NCCL group is already open"));
            }
            if let Err(error) = self.ctx.make_current() {
                self.in_group.set(false);
                return Err(error);
            }
            // SAFETY: NCCL group state is thread-local and this communicator
            // is used only by the rank thread that owns it.
            let result = check(self.ctx, "ncclGroupStart", unsafe { ffi::ncclGroupStart() });
            if result.is_err() {
                self.in_group.set(false);
            }
            result
        }

        /// Enqueue the group; completion is observed later with `poll`.
        pub fn group_end(&self) -> Result<()> {
            if !self.in_group.replace(false) {
                return Err(invalid("NCCL group is not open"));
            }
            self.ctx.make_current()?;
            // SAFETY: this closes a group opened by this rank thread.
            check_enqueued(self.ctx, "ncclGroupEnd", unsafe { ffi::ncclGroupEnd() })
        }

        fn check_collective(&self, operation: &str, code: ffi::ncclResult_t) -> Result<()> {
            if self.in_group.get() {
                check(self.ctx, operation, code)
            } else {
                check_enqueued(self.ctx, operation, code)
            }
        }

        pub fn finalize(&mut self, deadline: Instant) -> Result<()> {
            if self.finalized {
                return Ok(());
            }
            self.ctx.make_current()?;
            // SAFETY: this rank has drained its stream before finalization.
            let status = unsafe { ffi::ncclCommFinalize(self.handle) };
            if status != ffi::NCCL_SUCCESS && status != ffi::NCCL_IN_PROGRESS {
                return Err(nccl_error(self.ctx, "ncclCommFinalize", status));
            }
            loop {
                match self.poll() {
                    Ok(CommState::Ready) => {
                        self.finalized = true;
                        return Ok(());
                    }
                    Ok(CommState::InProgress) if Instant::now() < deadline => {
                        std::thread::yield_now()
                    }
                    Ok(CommState::InProgress) => {
                        return Err(Error::DeviceLost {
                            device: self.ctx.ordinal(),
                            detail: "NCCL finalization missed the group deadline".into(),
                        });
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        /// Enqueue an FP32 sum on this rank's stream.
        ///
        /// # Safety
        /// Both addresses must span `count` live FP32 values on this
        /// communicator's device and remain allocated until stream completion.
        pub unsafe fn all_reduce_f32_sum(
            &self,
            send: u64,
            receive: u64,
            count: usize,
            stream: &Stream<'ctx>,
        ) -> Result<()> {
            self.validate_buffer(send, count, stream)?;
            self.validate_buffer(receive, count, stream)?;
            self.ctx.make_current()?;
            // SAFETY: caller guarantees both device ranges hold `count` FP32
            // values and remain live until the stream completion is observed.
            self.check_collective("ncclAllReduce(FP32 sum)", unsafe {
                ffi::ncclAllReduce(
                    send as *const c_void,
                    receive as *mut c_void,
                    count,
                    ffi::NCCL_FLOAT32,
                    ffi::NCCL_SUM,
                    self.handle,
                    stream.raw(),
                )
            })
        }

        /// Enqueue a U32 maximum on this rank's stream.
        ///
        /// # Safety
        /// Both addresses must span `count` live U32 values on this
        /// communicator's device and remain allocated until stream completion.
        pub unsafe fn all_reduce_u32_max(
            &self,
            send: u64,
            receive: u64,
            count: usize,
            stream: &Stream<'ctx>,
        ) -> Result<()> {
            self.validate_buffer(send, count, stream)?;
            self.validate_buffer(receive, count, stream)?;
            self.ctx.make_current()?;
            // SAFETY: caller guarantees both device ranges hold `count` u32
            // values and remain live until the stream completion is observed.
            self.check_collective("ncclAllReduce(U32 max)", unsafe {
                ffi::ncclAllReduce(
                    send as *const c_void,
                    receive as *mut c_void,
                    count,
                    ffi::NCCL_UINT32,
                    ffi::NCCL_MAX,
                    self.handle,
                    stream.raw(),
                )
            })
        }

        /// Enqueue an all-gather of `count_per_rank` values from each rank.
        ///
        /// # Safety
        /// `send` must span `count_per_rank` live values, and `receive` must
        /// span that count multiplied by the communicator size; both remain
        /// allocated until stream completion.
        pub unsafe fn all_gather(
            &self,
            send: u64,
            receive: u64,
            count_per_rank: usize,
            dtype: DataType,
            stream: &Stream<'ctx>,
        ) -> Result<()> {
            self.validate_buffer(send, count_per_rank, stream)?;
            self.validate_buffer(receive, count_per_rank, stream)?;
            self.ctx.make_current()?;
            let datatype = match dtype {
                DataType::Bf16 => ffi::NCCL_BFLOAT16,
                DataType::F32 => ffi::NCCL_FLOAT32,
            };
            // SAFETY: caller guarantees the send range contains `count_per_rank`
            // values and the receive range contains that count for all ranks.
            self.check_collective("ncclAllGather", unsafe {
                ffi::ncclAllGather(
                    send as *const c_void,
                    receive as *mut c_void,
                    count_per_rank,
                    datatype,
                    self.handle,
                    stream.raw(),
                )
            })
        }

        /// Stream-order a U32 write to an admitted device word.
        ///
        /// # Safety
        /// `address` must name one writable U32 on this communicator's device
        /// and remain allocated until stream completion.
        pub unsafe fn set_u32_async(
            &self,
            address: u64,
            value: u32,
            stream: &Stream<'ctx>,
        ) -> Result<()> {
            self.validate_buffer(address, 1, stream)?;
            self.ctx.make_current()?;
            // SAFETY: caller supplies an admitted device word and retains it
            // through the completion event recorded after this stream write.
            let status = unsafe { ffi::cuMemsetD32Async(address, value, 1, stream.raw()) };
            if status == crate::CUDA_SUCCESS {
                Ok(())
            } else {
                crate::status::classify(status, "cuMemsetD32Async".into())
            }
        }

        fn validate_buffer(&self, address: u64, count: usize, stream: &Stream<'ctx>) -> Result<()> {
            if address == 0 || count == 0 || stream.device_uuid() != self.ctx.uuid() {
                return Err(invalid(
                    "NCCL buffer or stream does not belong to this rank",
                ));
            }
            Ok(())
        }

        pub fn abort(mut self) -> Result<()> {
            if self.handle.is_null() {
                return Ok(());
            }
            self.ctx.make_current()?;
            // SAFETY: the live handle is aborted by its context-owning rank.
            let status = unsafe { ffi::ncclCommAbort(self.handle) };
            if status == ffi::NCCL_SUCCESS {
                self.handle = core::ptr::null_mut();
                Ok(())
            } else {
                Err(nccl_error(self.ctx, "ncclCommAbort", status))
            }
        }

        /// Destroy a finalized communicator, returning it if destruction fails.
        ///
        /// # Safety
        /// The caller must have observed all local work and must ensure the
        /// peer communicator has also completed finalization.
        pub unsafe fn destroy(mut self) -> std::result::Result<(), (Self, Error)> {
            if !self.finalized {
                let error = invalid("NCCL communicator must finalize before destroy");
                return Err((self, error));
            }
            if let Err(error) = self.ctx.make_current() {
                return Err((self, error));
            }
            // SAFETY: finalization completed and the rank stream was observed
            // drained before this local communicator destruction.
            let status = unsafe { ffi::ncclCommDestroy(self.handle) };
            if status == ffi::NCCL_SUCCESS {
                self.handle = core::ptr::null_mut();
                Ok(())
            } else {
                let error = nccl_error(self.ctx, "ncclCommDestroy", status);
                Err((self, error))
            }
        }
    }

    impl Drop for Communicator<'_> {
        fn drop(&mut self) {
            // A live communicator is quarantined. Destroying it here could race
            // work on its rank stream and release memory still owned by NCCL.
        }
    }

    fn display_version(code: c_int) -> String {
        format!("{}.{}.{}", code / 10_000, code / 100 % 100, code % 100)
    }
}

#[cfg(not(feature = "nccl"))]
mod imp {
    use super::*;

    fn unsupported() -> Error {
        Error::Unsupported {
            capability: "nccl",
            reason: "build moxie-cuda with the nccl feature".into(),
        }
    }

    impl NcclId {
        pub fn generate(_ctx: &RankContext) -> Result<Self> {
            Err(unsupported())
        }
    }

    #[derive(Debug)]
    pub struct Communicator<'ctx> {
        _ctx: &'ctx RankContext,
        _not_send: PhantomData<Rc<()>>,
    }

    impl<'ctx> Communicator<'ctx> {
        pub fn init(
            _ctx: &'ctx RankContext,
            _ranks: i32,
            _id: NcclId,
            _rank: i32,
            _deadline: Instant,
        ) -> Result<Self> {
            Err(unsupported())
        }

        pub fn poll(&self) -> Result<CommState> {
            Err(unsupported())
        }

        pub fn group_start(&self) -> Result<()> {
            Err(unsupported())
        }

        pub fn group_end(&self) -> Result<()> {
            Err(unsupported())
        }

        pub fn finalize(&mut self, _deadline: Instant) -> Result<()> {
            Err(unsupported())
        }

        /// Enqueue an FP32 sum on this rank's stream.
        ///
        /// # Safety
        /// Both addresses must span `count` live FP32 values on this
        /// communicator's device and remain allocated until stream completion.
        pub unsafe fn all_reduce_f32_sum(
            &self,
            _send: u64,
            _receive: u64,
            _count: usize,
            _stream: &Stream<'ctx>,
        ) -> Result<()> {
            Err(unsupported())
        }

        /// Enqueue a U32 maximum on this rank's stream.
        ///
        /// # Safety
        /// Both addresses must span `count` live U32 values on this
        /// communicator's device and remain allocated until stream completion.
        pub unsafe fn all_reduce_u32_max(
            &self,
            _send: u64,
            _receive: u64,
            _count: usize,
            _stream: &Stream<'ctx>,
        ) -> Result<()> {
            Err(unsupported())
        }

        /// Enqueue an all-gather of `count_per_rank` values from each rank.
        ///
        /// # Safety
        /// `send` must span `count_per_rank` live values, and `receive` must
        /// span that count multiplied by the communicator size; both remain
        /// allocated until stream completion.
        pub unsafe fn all_gather(
            &self,
            _send: u64,
            _receive: u64,
            _count_per_rank: usize,
            _dtype: DataType,
            _stream: &Stream<'ctx>,
        ) -> Result<()> {
            Err(unsupported())
        }

        /// Stream-order a U32 write to an admitted device word.
        ///
        /// # Safety
        /// `address` must name one writable U32 on this communicator's device
        /// and remain allocated until stream completion.
        pub unsafe fn set_u32_async(
            &self,
            _address: u64,
            _value: u32,
            _stream: &Stream<'ctx>,
        ) -> Result<()> {
            Err(unsupported())
        }

        pub fn abort(self) -> Result<()> {
            Ok(())
        }

        /// Destroy a finalized communicator, returning it if destruction fails.
        ///
        /// # Safety
        /// The caller must have observed all local work and must ensure the
        /// peer communicator has also completed finalization.
        pub unsafe fn destroy(self) -> std::result::Result<(), (Self, Error)> {
            Err((self, unsupported()))
        }
    }
}

pub use imp::Communicator;
