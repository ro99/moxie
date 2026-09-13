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
pub mod layout;
pub mod numa;
pub mod precision;
pub mod tier;

pub use capability::{
    DeviceCapability, GateTransform, HostLimit, KernelCapability, KernelCatalogue, KernelId,
    KernelOperand, KernelShapeBounds, KernelSymbol, MeasuredDevice, MeasuredHost, RoundingProfile,
    SemanticKernelDescriptor, SemanticKernelOp, SmVersion, StrategyControl, WorkspaceExpression,
};
pub use dim::{Dim, DimError, SymbolId, SymbolTable};
pub use error::{Error, Result};
pub use ids::{
    ArtifactId, BranchId, ChunkId, DeviceId, DeviceUuid, LayoutId, RankId, StateTransactionId,
    TensorId,
};
pub use layout::TensorLayout;
pub use numa::{HostPlacement, NumaNode, NumaNodeId, NumaTopology};
pub use precision::{
    AccumulationPolicy, ActivationPrecision, CachePrecision, ExecutionProfile, Precision,
    WeightPrecision,
};
pub use tier::{DeviceTier, HostTier, Scope, ScopeKind, Tier};
