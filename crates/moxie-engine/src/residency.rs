//! From a batch's routes to the chunks residency must hold.
//!
//! This is the join task 0019 deliberately left open. The routing equation
//! decides which experts a row selects; the residency authority admits against
//! the **union** of those selections. Neither half can name the other: the model
//! crate may not know what a cache is, and the cache may not know what a router
//! is. What sits between them is arithmetic over data both already have, and it
//! is small on purpose.
//!
//! Document 03 states the rule this implements in one sentence: "Do not multiply
//! active experts by batch rows when routes overlap; do not assume overlap when
//! they do not." [`expert_demand`] takes the routes as they were computed and
//! returns each distinct expert once. The bound `rows · top_k` is what the
//! bring-up record calls an upper bound that assumes no two rows share an
//! expert; this returns the real number.

use std::collections::BTreeSet;

use moxie_memory::{ArtifactId, ChunkId, LogicalRange, TensorSlot};
use moxie_types::{Error, Result};

/// One role every selected expert needs, and how many bytes of it one expert
/// occupies.
///
/// The artifact this task was written against fuses all of a layer's experts
/// into one tensor per role, so expert `e`'s bytes are the contiguous range
/// `[e · bytes_per_expert, (e+1) · bytes_per_expert)`. A checkpoint that stores
/// one tensor per expert is the same shape with `bytes_per_expert` equal to the
/// whole tensor and one role per expert; nothing here assumes either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertRole {
    pub role: String,
    pub bytes_per_expert: u64,
}

impl ExpertRole {
    pub fn new(role: impl Into<String>, bytes_per_expert: u64) -> Result<Self> {
        let role = role.into();
        if role.is_empty() {
            return Err(Error::InvalidRequest {
                field: "role",
                detail: "an expert role cannot be empty".into(),
            });
        }
        if bytes_per_expert == 0 {
            return Err(Error::InvalidRequest {
                field: "bytes_per_expert",
                detail: "an expert's slice cannot be empty".into(),
            });
        }
        Ok(ExpertRole {
            role,
            bytes_per_expert,
        })
    }
}

/// The chunks a row batch's routes demand, each exactly once.
///
/// Ordered by `(expert, role)` so the demand set is a function of the routes
/// and nothing else -- not of row order, not of which row asked first. Two
/// callers that computed the same routes produce the same demand, which is what
/// makes an eviction trace reproducible.
///
/// `routes[r]` is row `r`'s selected expert ids. Repeats within a row and across
/// rows both collapse; an id at or past `expert_count` is a refusal naming it,
/// because a route that points at an expert the artifact does not have is a
/// defect in the router, not a chunk to go looking for.
pub fn expert_demand(
    artifact: &ArtifactId,
    roles: &[ExpertRole],
    routes: &[&[u32]],
    expert_count: u32,
    format_version: u32,
) -> Result<Vec<ChunkId>> {
    if roles.is_empty() {
        return Err(Error::InvalidRequest {
            field: "roles",
            detail: "a routed expert needs at least one tensor role".into(),
        });
    }
    let mut union: BTreeSet<u32> = BTreeSet::new();
    for (row, selected) in routes.iter().enumerate() {
        for expert in selected.iter().copied() {
            if expert >= expert_count {
                return Err(Error::InvalidRequest {
                    field: "expert",
                    detail: format!("row {row} routes to expert {expert} of {expert_count}"),
                });
            }
            union.insert(expert);
        }
    }

    let mut out = Vec::new();
    out.try_reserve_exact(union.len().saturating_mul(roles.len()))
        .map_err(|_| Error::CapacityExceeded {
            tier: None,
            requested_bytes: (union.len().saturating_mul(roles.len())) as u64,
            available_bytes: 0,
        })?;
    for expert in union {
        for role in roles {
            let offset = u64::from(expert)
                .checked_mul(role.bytes_per_expert)
                .ok_or_else(|| Error::InvalidRequest {
                    field: "expert",
                    detail: format!("expert {expert}'s offset in {:?} overflows", role.role),
                })?;
            out.push(ChunkId::new(
                artifact.clone(),
                TensorSlot::expert(role.role.clone(), expert)?,
                LogicalRange::new(offset, role.bytes_per_expert)?,
                format_version,
            ));
        }
    }
    Ok(out)
}

/// What a demand set costs if none of it is resident.
///
/// The number an admission report's `incoming` is summed from, and the honest
/// counterpart to `rows · top_k · expert_bytes`: it counts each expert once.
pub fn demand_bytes(demand: &[ChunkId]) -> u64 {
    demand.iter().map(ChunkId::len_bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> ArtifactId {
        ArtifactId::new("test-artifact").unwrap()
    }

    fn roles() -> Vec<ExpertRole> {
        vec![
            ExpertRole::new("experts_gate_up", 64).unwrap(),
            ExpertRole::new("experts_down", 32).unwrap(),
        ]
    }

    #[test]
    fn overlapping_routes_demand_each_expert_once() {
        let rows: Vec<&[u32]> = vec![&[0, 3, 5], &[3, 5, 7], &[0, 7, 3]];
        let demand = expert_demand(&artifact(), &roles(), &rows, 8, 1).unwrap();
        // Four distinct experts, two roles each -- not 3 rows x 3 x 2 = 18.
        assert_eq!(demand.len(), 8);
        assert_eq!(demand_bytes(&demand), 4 * (64 + 32));
        let experts: Vec<u32> = demand
            .iter()
            .filter(|c| c.slot().role() == "experts_gate_up")
            .map(|c| c.slot().expert_index().unwrap())
            .collect();
        assert_eq!(experts, vec![0, 3, 5, 7]);
    }

    #[test]
    fn disjoint_routes_demand_their_full_union() {
        let rows: Vec<&[u32]> = vec![&[0, 1], &[2, 3]];
        let demand = expert_demand(&artifact(), &roles(), &rows, 8, 1).unwrap();
        assert_eq!(demand.len(), 8);
        assert_eq!(demand_bytes(&demand), 4 * (64 + 32));
    }

    #[test]
    fn nonuniform_row_counts_produce_the_same_union_as_their_uniform_equivalent() {
        let ragged: Vec<&[u32]> = vec![&[1], &[1, 4, 6, 6], &[], &[4]];
        let uniform: Vec<&[u32]> = vec![&[1, 4], &[6, 1], &[4, 6]];
        let a = expert_demand(&artifact(), &roles(), &ragged, 8, 1).unwrap();
        let b = expert_demand(&artifact(), &roles(), &uniform, 8, 1).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn every_row_routing_to_one_expert_demands_one_expert() {
        let rows: Vec<&[u32]> = vec![&[2], &[2], &[2], &[2]];
        let demand = expert_demand(&artifact(), &roles(), &rows, 8, 1).unwrap();
        assert_eq!(demand.len(), 2);
        assert_eq!(demand_bytes(&demand), 64 + 32);
    }

    #[test]
    fn a_route_outside_the_expert_count_is_refused_by_name() {
        let rows: Vec<&[u32]> = vec![&[0], &[9]];
        let e = expert_demand(&artifact(), &roles(), &rows, 8, 1).unwrap_err();
        assert!(format!("{e}").contains("expert 9 of 8"), "{e}");
    }

    #[test]
    fn the_fused_slice_arithmetic_is_the_artifacts_own() {
        // The designated artifact's layer tensors: `[128, 1408, 2816]` and
        // `[128, 2816, 704]` in BF16. One expert is 7,929,856 + 3,964,928 =
        // 11,894,784 bytes, which is what the bring-up record records.
        let roles = vec![
            ExpertRole::new("experts_gate_up", 1408 * 2816 * 2).unwrap(),
            ExpertRole::new("experts_down", 2816 * 704 * 2).unwrap(),
        ];
        let rows: Vec<&[u32]> = vec![&[127]];
        let demand = expert_demand(&artifact(), &roles, &rows, 128, 1).unwrap();
        assert_eq!(demand_bytes(&demand), 11_894_784);
        assert_eq!(demand[0].range().offset_bytes(), 127 * 7_929_856);
        assert_eq!(demand[0].range().len_bytes(), 7_929_856);
        assert_eq!(demand[1].range().offset_bytes(), 127 * 3_964_928);
        // The last expert's slice ends exactly at the fused tensor's length.
        assert_eq!(
            demand[0].range().offset_bytes() + demand[0].range().len_bytes(),
            1_015_021_568
        );
        assert_eq!(
            demand[1].range().offset_bytes() + demand[1].range().len_bytes(),
            507_510_784
        );
    }
}
