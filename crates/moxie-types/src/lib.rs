//! Identifiers, dimensions, typed errors, precision and capability types.
//!
//! This crate is the bottom of the dependency graph (document 02). It must not
//! depend on any other workspace crate, and must not know about CUDA, storage,
//! or any concrete model.

#![forbid(unsafe_code)]

pub mod capability;
pub mod dim;
pub mod error;
pub mod ids;
pub mod precision;

pub use capability::{DeviceCapability, KernelCapability, StrategyControl};
pub use dim::{Dim, DimError, SymbolId, SymbolTable};
pub use error::{Error, Result};
pub use ids::{
    ArtifactId, BranchId, ChunkId, DeviceId, GraphId, LayoutId, RankId, StateTransactionId,
    TensorId,
};
pub use precision::{
    AccumulationPolicy, ActivationPrecision, CachePrecision, ExecutionProfile, Precision,
    WeightPrecision,
};
