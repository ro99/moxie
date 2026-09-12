//! Concrete model definitions: one module per family, and nothing else.
//!
//! # Why one crate rather than one crate per family
//!
//! The boundary that matters is the **dependency list**, not the crate count.
//! Document 02 gives a model definition three allowed workspace dependencies --
//! `moxie-types`, `moxie-graph`, `moxie-model-api` -- and document 09 §B lists
//! what its code may not contain: no device allocation or launch, no file or
//! mmap pipeline, no cache or eviction, no KV page allocation, no
//! prefill/decode/sampling loop, no collective. The check for that last list
//! is deliberately broad enough to fire on a comment, so this paragraph names
//! the boundary without naming the vendor toolkit. A module cannot import what
//! its crate does not depend on, so a
//! single crate holding every family enforces exactly the same rule as a crate
//! per family, and `arch-check` checks the same three things either way.
//!
//! What a crate per family would add is a manifest, a workspace member, an
//! allowlist entry and a build unit for every model -- ceremony that buys no
//! additional enforcement. Adding Laguna here is a new file and a `pub mod`
//! line.
//!
//! A family does earn its own crate when it needs a dependency the others must
//! not have -- a narrowly approved metadata parser, say, which document 02
//! allows case by case. `arch-check` still recognises a `moxie-models-*` crate
//! and holds it to the same list, so that stays available without being the
//! default.
//!
//! # What a model definition is not
//!
//! Nothing here executes. These modules build a [`moxie_graph::Graph`] out of
//! shared operations and name the tensor roles an importer must fill; the
//! engine owns every byte, event and state transition that follows. A module
//! that starts needing an allocation is a shared-ownership gap to fix in the
//! engine, not an exception to make here.

#![forbid(unsafe_code)]

pub mod gemma4;
