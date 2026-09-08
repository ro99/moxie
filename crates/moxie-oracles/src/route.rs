//! Routed experts: selection, dispatch and combination.
//!
//! Document 02 makes `Route`, `Dispatch`, `ExpertMlp` and `Combine` separate
//! shared operations. Document 07 requires "router tie fixtures with fixed
//! logits" as an exact check, because a tie broken differently changes which
//! expert weights a step needs -- a residency and a correctness difference at
//! once.
//!
//! Pinned here: the top-k selection rule including its tie rule, the
//! renormalisation of selected weights, and the dispatch/combine round trip over
//! a row batch with overlapping routes.
//!
//! Explicitly **not** pinned here: shared/always-on experts, expert capacity and
//! dropping, auxiliary load-balancing losses, grouped or hierarchical routing,
//! and any residency decision. Each needs its own fixture when the model that
//! requires it arrives.

use moxie_types::{Error, Result};

/// One row's routing decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// Selected expert ids, in selection order (descending score, then id).
    pub experts: Vec<u32>,
    /// Combination coefficients, renormalised over the selected experts only.
    pub weights: Vec<f32>,
}

/// Numerically stable softmax over router logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Select the top `k` experts for one row.
///
/// **Tie rule: the lower expert id wins.** Document 05 pins the same rule for
/// sampler score ties ("do not depend on an unstable sort or batch execution
/// order"), and routing needs it for the same reason: a comparison sort's
/// behaviour on equal keys is not a contract, and two ranks that disagree route
/// the same row to different experts.
///
/// Renormalisation is over the *selected* experts, so the coefficients sum to
/// one. Combining with unnormalised probabilities silently scales the layer's
/// output by the selected mass.
pub fn route_row(logits: &[f32], k: usize) -> Result<Route> {
    if logits.is_empty() {
        return Err(Error::InvalidRequest {
            field: "router_logits",
            detail: "empty expert set".into(),
        });
    }
    if k == 0 || k > logits.len() {
        return Err(Error::InvalidRequest {
            field: "top_k",
            detail: format!("top_k {k} outside 1..={}", logits.len()),
        });
    }
    if let Some(bad) = logits.iter().position(|l| !l.is_finite()) {
        return Err(Error::Numerical {
            detail: format!("router logit {bad} is {}", logits[bad]),
        });
    }

    let probs = softmax(logits);
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    // Descending by probability; equal probabilities keep ascending id order.
    // `sort_by` is stable, and `order` starts in ascending id order, so an equal
    // comparison preserves the lower id first.
    order.sort_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .expect("probabilities are finite")
    });

    let experts: Vec<u32> = order.into_iter().take(k).collect();
    let mass: f32 = experts.iter().map(|e| probs[*e as usize]).sum();
    if mass <= 0.0 || !mass.is_finite() {
        return Err(Error::Numerical {
            detail: format!("selected router mass is {mass}"),
        });
    }
    let weights = experts.iter().map(|e| probs[*e as usize] / mass).collect();
    Ok(Route { experts, weights })
}

/// The rows assigned to one expert, in ascending row order.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatch {
    pub expert: u32,
    pub rows: Vec<usize>,
}

/// Group rows by expert.
///
/// Document 04: "Group token rows across experts to amortize weight transfers;
/// preserve per-row route coefficients and output ordering." Rows stay in
/// ascending order inside a batch so that a grouped kernel's output can be
/// scattered back deterministically.
///
/// Only experts that actually receive a row appear. Document 03 warns against
/// multiplying active experts by batch rows when routes overlap: the union is
/// what has to be resident, and it is usually smaller than `rows * k`.
pub fn dispatch(routes: &[Route], expert_count: u32) -> Result<Vec<ExpertBatch>> {
    let mut batches: Vec<ExpertBatch> = Vec::new();
    for (row, r) in routes.iter().enumerate() {
        for e in &r.experts {
            if *e >= expert_count {
                return Err(Error::InvalidRequest {
                    field: "expert",
                    detail: format!("row {row} routes to expert {e} of {expert_count}"),
                });
            }
        }
    }
    for expert in 0..expert_count {
        let rows: Vec<usize> = routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.experts.contains(&expert))
            .map(|(i, _)| i)
            .collect();
        if !rows.is_empty() {
            batches.push(ExpertBatch { expert, rows });
        }
    }
    Ok(batches)
}

/// The distinct experts a row batch needs resident.
pub fn required_experts(routes: &[Route]) -> Vec<u32> {
    let mut all: Vec<u32> = routes
        .iter()
        .flat_map(|r| r.experts.iter().copied())
        .collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// Combine per-expert outputs back into one vector per row.
///
/// `outputs[(expert, row)]` supplies the expert's result for that row. The sum
/// is taken in the row's own expert order, so the reduction order is a property
/// of the route rather than of whatever order a scheduler happened to finish in.
pub fn combine(
    routes: &[Route],
    width: usize,
    outputs: &dyn Fn(u32, usize) -> Vec<f32>,
) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(routes.len());
    for (row, r) in routes.iter().enumerate() {
        let mut acc = vec![0f32; width];
        for (e, w) in r.experts.iter().zip(r.weights.iter()) {
            let out = outputs(*e, row);
            if out.len() != width {
                return Err(Error::InvalidRequest {
                    field: "expert_output",
                    detail: format!(
                        "expert {e} returned {} values for row {row}, expected {width}",
                        out.len()
                    ),
                });
            }
            for (a, v) in acc.iter_mut().zip(out.iter()) {
                *a += w * v;
            }
        }
        rows.push(acc);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_selects_by_score_and_renormalises() {
        let logits = [0.0f32, 2.0, 1.0, -1.0];
        let r = route_row(&logits, 2).unwrap();
        assert_eq!(r.experts, vec![1, 2]);
        let sum: f32 = r.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to {sum}");
        assert!(r.weights[0] > r.weights[1]);

        // Renormalisation is over the selected pair only, so the ratio of the
        // two coefficients is the ratio of their softmax probabilities.
        let p = softmax(&logits);
        assert!((r.weights[0] / r.weights[1] - p[1] / p[2]).abs() < 1e-5);
    }

    #[test]
    fn an_exact_tie_is_broken_by_the_lower_expert_id() {
        // Document 07 asks for "router tie fixtures with fixed logits". Four
        // experts, all identical: the selection must be 0 and 1 every time, on
        // every machine, whatever the sort implementation does with equal keys.
        let r = route_row(&[1.0, 1.0, 1.0, 1.0], 2).unwrap();
        assert_eq!(r.experts, vec![0, 1]);
        assert_eq!(r.weights, vec![0.5, 0.5]);

        // A tie between the second and third places, with a clear winner first.
        let r = route_row(&[5.0, 1.0, 1.0], 2).unwrap();
        assert_eq!(r.experts, vec![0, 1]);

        // Ties spanning the k boundary: 2 and 3 tie for the last slot.
        let r = route_row(&[9.0, 9.0, 3.0, 3.0], 3).unwrap();
        assert_eq!(r.experts, vec![0, 1, 2]);
    }

    #[test]
    fn selection_is_stable_under_reordering_of_equal_scores() {
        // Running the same fixed logits repeatedly must give one answer.
        for _ in 0..32 {
            assert_eq!(route_row(&[2.0, 2.0, 2.0], 1).unwrap().experts, vec![0]);
        }
    }

    #[test]
    fn top_k_equal_to_the_expert_count_keeps_the_full_distribution() {
        let logits = [0.5f32, -0.5, 2.0];
        let r = route_row(&logits, 3).unwrap();
        let p = softmax(&logits);
        let mut got: Vec<(u32, f32)> = r
            .experts
            .iter()
            .copied()
            .zip(r.weights.iter().copied())
            .collect();
        got.sort_by_key(|(e, _)| *e);
        for (e, w) in got {
            assert!((w - p[e as usize]).abs() < 1e-6, "expert {e}");
        }
    }

    #[test]
    fn invalid_routing_requests_are_typed_errors() {
        assert!(route_row(&[], 1).is_err());
        assert!(route_row(&[1.0, 2.0], 0).is_err());
        assert!(route_row(&[1.0, 2.0], 3).is_err());
        assert_eq!(
            route_row(&[1.0, f32::NAN], 1).unwrap_err().kind(),
            "numerical"
        );
        assert_eq!(
            route_row(&[1.0, f32::INFINITY], 1).unwrap_err().kind(),
            "numerical"
        );
    }

    #[test]
    fn dispatch_groups_rows_and_the_union_is_smaller_than_rows_times_k() {
        // Document 03: "Do not multiply active experts by batch rows when routes
        // overlap; do not assume overlap when they do not."
        let routes = vec![
            route_row(&[3.0, 2.0, 0.0, 0.0], 2).unwrap(), // 0, 1
            route_row(&[3.0, 0.0, 2.0, 0.0], 2).unwrap(), // 0, 2
            route_row(&[0.0, 0.0, 3.0, 2.0], 2).unwrap(), // 2, 3
        ];
        let batches = dispatch(&routes, 4).unwrap();
        assert_eq!(
            batches,
            vec![
                ExpertBatch {
                    expert: 0,
                    rows: vec![0, 1]
                },
                ExpertBatch {
                    expert: 1,
                    rows: vec![0]
                },
                ExpertBatch {
                    expert: 2,
                    rows: vec![1, 2]
                },
                ExpertBatch {
                    expert: 3,
                    rows: vec![2]
                },
            ]
        );
        // rows * k is 6; the union of experts is 4.
        assert_eq!(required_experts(&routes), vec![0, 1, 2, 3]);
        assert!(required_experts(&routes).len() < routes.len() * 2);

        // Disjoint routes: no overlap to exploit, and the union really is rows*k.
        let disjoint = vec![
            route_row(&[3.0, 0.0, 0.0, 0.0], 1).unwrap(),
            route_row(&[0.0, 3.0, 0.0, 0.0], 1).unwrap(),
        ];
        assert_eq!(required_experts(&disjoint).len(), 2);
    }

    #[test]
    fn an_expert_with_no_rows_is_absent_and_nonuniform_counts_survive() {
        // Document 02 M2 asks for "nonuniform row counts" to be exercised.
        let routes = vec![
            route_row(&[5.0, 0.0, 0.0], 1).unwrap(),
            route_row(&[5.0, 0.0, 0.0], 1).unwrap(),
            route_row(&[0.0, 5.0, 0.0], 1).unwrap(),
        ];
        let batches = dispatch(&routes, 3).unwrap();
        assert_eq!(batches.len(), 2, "expert 2 received nothing: {batches:?}");
        assert_eq!(batches[0].rows.len(), 2);
        assert_eq!(batches[1].rows.len(), 1);
    }

    #[test]
    fn an_out_of_range_expert_is_refused() {
        let bad = Route {
            experts: vec![7],
            weights: vec![1.0],
        };
        assert!(dispatch(&[bad], 4).is_err());
    }

    #[test]
    fn combine_reproduces_the_weighted_sum_in_route_order() {
        let routes = vec![route_row(&[1.0, 1.0, -5.0], 2).unwrap()];
        assert_eq!(routes[0].experts, vec![0, 1]);
        assert_eq!(routes[0].weights, vec![0.5, 0.5]);

        // Expert e returns a constant vector of value (e+1).
        let out = |e: u32, _row: usize| vec![(e + 1) as f32; 3];
        let combined = combine(&routes, 3, &out).unwrap();
        assert_eq!(combined, vec![vec![1.5, 1.5, 1.5]]);

        // A wrongly shaped expert output is refused rather than truncated.
        let short = |_e: u32, _row: usize| vec![0f32; 2];
        assert!(combine(&routes, 3, &short).is_err());
    }

    #[test]
    fn combine_is_the_identity_for_a_single_expert() {
        let routes = vec![route_row(&[9.0, 0.0], 1).unwrap()];
        assert_eq!(routes[0].weights, vec![1.0]);
        let out = |_e: u32, _row: usize| vec![2.0, -3.0];
        assert_eq!(combine(&routes, 2, &out).unwrap(), vec![vec![2.0, -3.0]]);
    }

    #[test]
    fn every_row_keeps_its_own_coefficients_through_dispatch_and_combine() {
        // The property a grouped kernel must not lose: rows are regrouped by
        // expert for compute, but each row's coefficients belong to that row.
        let routes = vec![
            route_row(&[2.0, 0.0, 0.0], 2).unwrap(),
            route_row(&[0.0, 0.0, 2.0], 2).unwrap(),
        ];
        let batches = dispatch(&routes, 3).unwrap();
        assert!(batches.iter().any(|b| b.rows.len() == 2));

        // Expert output depends on the row, so a mixed-up scatter shows up.
        let out = |e: u32, row: usize| vec![(e * 10 + row as u32) as f32];
        let combined = combine(&routes, 1, &out).unwrap();
        for (row, r) in routes.iter().enumerate() {
            let want: f32 = r
                .experts
                .iter()
                .zip(r.weights.iter())
                .map(|(e, w)| w * (e * 10 + row as u32) as f32)
                .sum();
            assert!((combined[row][0] - want).abs() < 1e-6, "row {row}");
        }
    }
}
