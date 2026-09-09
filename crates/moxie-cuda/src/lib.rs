//! Typed CUDA driver wrapper, split into a pure part and a driver part.
//!
//! Document 07: "Host CI must run without a checkpoint, NVIDIA driver library or
//! CUDA toolkit. Split pure tooling/host dependencies from opt-in GPU build/run
//! features; building arch-check must not compile kernels or link libcuda."
//!
//! * [`status`] is always compiled. It maps driver result codes to typed errors
//!   and formats device UUIDs. No `extern "C"` declaration, no link directive.
//! * Everything else -- contexts, buffers, streams, events, modules, launches --
//!   is behind the **`driver`** feature, which is off by default and is what
//!   adds `-lcuda`.
//!
//! The separation is not a way to make the GPU lane optional. Document 07 in the
//! same breath requires "a distinct mandatory CUDA compile/real-hardware lane
//! for affected changes, so this separation does not turn unmeasured GPU work
//! into a pass." That lane is `cargo xtask-cuda test-gpu`.

#![cfg_attr(not(feature = "driver"), forbid(unsafe_code))]

pub mod status;

pub use status::{CUDA_SUCCESS, CUresult, classify};

#[cfg(feature = "driver")]
pub mod ffi;

#[cfg(feature = "driver")]
mod driver;

#[cfg(feature = "driver")]
pub use driver::{
    DeviceBuffer, Event, Function, Module, ModuleImage, PtxSource, RankContext, Stream,
    TrustedImage, device_count, init, query_device,
};
