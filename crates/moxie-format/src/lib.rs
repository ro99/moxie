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
pub mod canonical;
pub mod compressed_tensors;
pub mod journal;
pub mod manifest;
pub mod payload;
pub mod quantize;
pub mod safetensors;
pub mod scale;
pub mod selection;
pub mod sha256;

pub use sha256::{StreamingSha256, sha256_hex};

/// An [`Error::InvalidArtifact`] whose prose is a borrowed `&'static str`.
///
/// Allocation-free, so it is the refusal a path under memory pressure can
/// always return.
pub(crate) fn invalid_static(detail: &'static str) -> moxie_types::Error {
    moxie_types::Error::InvalidArtifact {
        detail: std::borrow::Cow::Borrowed(detail),
    }
}

/// An [`Error::InvalidArtifact`] whose prose is composed **fallibly**.
///
/// Task 0024's independent review injected one allocation failure into an
/// importer refusing a malformed artifact and got `SIGABRT`: every refusal in
/// this crate built its detail with `format!`, which aborts rather than
/// returning. The context that produces a refusal -- a corrupt or hostile file
/// being read -- is exactly the context most likely to coincide with memory
/// pressure, so this is the one path that may not depend on an allocation
/// succeeding.
///
/// The message is built into a `String` grown only through `try_reserve`. When
/// that fails, `fallback` is returned **borrowed**: a shorter true statement
/// rather than an abort, which is task 0023's conclusion -- "a diagnostic that
/// cannot be built is a process that cannot report anything". The variant is
/// what a caller branches on either way (document 02).
pub(crate) fn invalid_fmt(
    fallback: &'static str,
    args: core::fmt::Arguments<'_>,
) -> moxie_types::Error {
    use core::fmt::Write;
    let mut sink = FallibleString(String::new());
    match sink.write_fmt(args) {
        Ok(()) => moxie_types::Error::InvalidArtifact {
            detail: std::borrow::Cow::Owned(sink.0),
        },
        Err(_) => invalid_static(fallback),
    }
}

/// A `String` that grows only through `try_reserve`.
struct FallibleString(String);

impl core::fmt::Write for FallibleString {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        // Reserve first, then push: `push_str` cannot reallocate once the
        // capacity is there, so no infallible growth happens on this path.
        self.0.try_reserve(s.len()).map_err(|_| core::fmt::Error)?;
        self.0.push_str(s);
        Ok(())
    }
}

/// `<module>.<suffix>`, built with exactly one **fallible** allocation.
///
/// The second review's P1: the importer's refusal prose was made fallible and
/// the *lookup names on the way to it* were not. `format!` aborts, so a helper
/// that resolves four tensor names took the process down eight allocation
/// positions before reaching the refusal that had just been fixed. That is task
/// 0023's own sentence -- "reserving a destination says nothing about a
/// temporary the callee builds" -- one layer further out.
///
/// A `BTreeMap<String, _>` lookup needs an owned key, and searching the map
/// without one is linear in a header that holds 140,989 tensors, so the
/// allocation is real work rather than a formatting convenience. What it must
/// not be is infallible.
pub(crate) fn join_name(module: &str, suffix: &str) -> moxie_types::Result<String> {
    let mut out = String::new();
    let need = module.len() + 1 + suffix.len();
    out.try_reserve_exact(need)
        .map_err(|_| moxie_types::Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: need as u64,
            available_bytes: 0,
        })?;
    out.push_str(module);
    out.push('.');
    out.push_str(suffix);
    Ok(out)
}

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
