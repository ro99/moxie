//! Canonical artifact format: numerical contracts and manifest schema.
//!
//! Document 02: this crate owns the "canonical manifest, bounded reads/mappings,
//! conversion tools, chunk validation". It must not decide "which live expert
//! should evict" -- that is the memory authority's job.
//!
//! M0 scope: the pinned numerical contracts and their exhaustive oracles. The
//! manifest reader, converter and bounded reads arrive in M1 and M3.

#![forbid(unsafe_code)]

pub mod bf16;
pub mod int8;
pub mod nvfp4;
