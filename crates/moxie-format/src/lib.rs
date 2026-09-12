//! Canonical artifact format: numerical contracts and manifest schema.
//!
//! Document 02: this crate owns the "canonical manifest, bounded reads/mappings,
//! conversion tools, chunk validation". It must not decide "which live expert
//! should evict" -- that is the memory authority's job.
//!
//! M0 scope: the pinned numerical contracts and their exhaustive oracles. The
//! manifest reader, converter and bounded reads arrive in M1 and M3.
//!
//! The canonical weight families are INT4, INT8 and BF16, under one affine
//! integer schema ([ADR 0003]). NVFP4 has no module here any more: the E2M1 /
//! E4M3FN codec that used to live in `nvfp4.rs` was removed when
//! [`affine`] replaced it, because document 03 asks for its history to be
//! preserved "without keeping NVFP4 in the active precision API". It is at
//! `crates/moxie-format/src/nvfp4.rs` in commit `84273b0`, and the scale-
//! convention finding it produced is written up in
//! `docs/evidence/experiments/0001-nvfp4-scale-conventions.md`.
//!
//! [ADR 0003]: ../../../docs/decisions/adr/0003-int4-int8-bf16-weight-family.md

#![forbid(unsafe_code)]

pub mod affine;
pub mod bf16;
pub mod compressed_tensors;
pub mod manifest;
pub mod quantize;
pub mod safetensors;
pub mod scale;
pub mod sha256;

pub use sha256::{StreamingSha256, sha256_hex};

/// A vector with exactly `capacity` reserved, or a typed capacity error.
///
/// Import sizes come from an artifact's own header, so the allocation that
/// holds a tensor is the one place a hostile or corrupt file could ask for more
/// memory than exists. Failing it must be `CapacityExceeded`, never the
/// infallible allocator's abort.
pub(crate) fn try_vec<T>(capacity: usize) -> moxie_types::Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| moxie_types::Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: capacity.saturating_mul(size_of::<T>()) as u64,
            available_bytes: 0,
        })?;
    Ok(out)
}
