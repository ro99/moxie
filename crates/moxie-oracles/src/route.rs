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
//! Task 0019 adds the rest of the routed path: the router's own score
//! transform, the per-expert coefficient scale, the gated expert feed-forward
//! over a **fused** expert tensor, and a combination whose reduction order is
//! stated rather than left to a scheduler.
//!
//! Explicitly **not** pinned here: expert capacity and dropping, auxiliary
//! load-balancing losses, grouped or hierarchical routing, and any residency
//! decision. A shared expert is composed *around* routing rather than inside
//! it -- in the designated artifact the dense MLP takes no routing coefficient
//! and is added after both branches are normalized -- so there is no shared
//! expert parameter here, and a family whose shared expert participates in the
//! routing normalization needs its own fixture when it arrives.

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
    select_top_k(&probs, k)
}

/// Select the top `k` of an already-computed distribution and renormalise.
///
/// Split out of [`route_row`] so that every router in this crate breaks ties by
/// the same rule and renormalises over the same mass. Two selection functions
/// would be two tie rules the moment one of them was edited.
pub fn select_top_k(probs: &[f32], k: usize) -> Result<Route> {
    if probs.is_empty() {
        return Err(Error::InvalidRequest {
            field: "router_probabilities",
            detail: "empty expert set".into(),
        });
    }
    if k == 0 || k > probs.len() {
        return Err(Error::InvalidRequest {
            field: "top_k",
            detail: format!("top_k {k} outside 1..={}", probs.len()),
        });
    }
    if let Some(bad) = probs.iter().position(|p| !p.is_finite()) {
        return Err(Error::Numerical {
            detail: format!("router probability {bad} is {}", probs[bad]),
        });
    }
    let mut order: Vec<u32> = crate::try_vec(probs.len())?;
    order.extend(0..probs.len() as u32);
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
    let mut weights = crate::try_vec(experts.len())?;
    weights.extend(experts.iter().map(|e| probs[*e as usize] / mass));
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

/// The router's input transform: `n = x / rms(x)`, then `n ⊙ g`, then `· c`.
///
/// Transcribed from `Gemma4TextRouter.forward` in the pinned `transformers`
/// source. Three details are load-bearing and none of them is guessable:
///
/// * the normalization is **scale-free** (`Gemma4RMSNorm(..., with_scale=False)`)
///   and the `router.scale` tensor is applied to its output, so `gain` here is
///   that tensor rather than a norm gain the model also owns elsewhere;
/// * `c` is `hidden^(-1/2)` in this family (`scalar_root_size`) and is neither
///   a norm epsilon nor an attention scale;
/// * the epsilon is added to the *mean square* before the reciprocal square
///   root, which is where the pinned `Gemma4RMSNorm._norm` puts it.
///
/// FP32 throughout and unrounded, like every other reference in this crate; the
/// node output is the rounding boundary.
pub fn router_input_row(x: &[f32], gain: &[f32], input_scale: f32, eps: f32) -> Result<Vec<f32>> {
    if x.is_empty() {
        return Err(Error::InvalidRequest {
            field: "router_input",
            detail: "a router over zero features".into(),
        });
    }
    if gain.len() != x.len() {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "router gain has {} elements for {} features",
                gain.len(),
                x.len()
            ),
        });
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidRequest {
            field: "eps",
            detail: format!("epsilon must be finite and positive, got {eps}"),
        });
    }
    if !(input_scale.is_finite() && input_scale > 0.0) {
        return Err(Error::InvalidRequest {
            field: "router_input_scale",
            detail: format!("router input scale must be finite and positive, got {input_scale}"),
        });
    }
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    let mean = sum / x.len() as f32;
    let denom = (mean + eps).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return Err(Error::Numerical {
            detail: format!("router rms denominator is {denom}"),
        });
    }
    let mut out = crate::try_vec(x.len())?;
    out.extend(
        x.iter()
            .zip(gain)
            .map(|(v, g)| (v / denom) * g * input_scale),
    );
    Ok(out)
}

/// The router's score distribution: project, then softmax over **all** experts.
///
/// The softmax is over the whole expert set and the top-k is taken from the
/// resulting probabilities, not from the logits. For a plain top-k the two give
/// the same selection, because softmax is monotonic -- but they do not give the
/// same *coefficients*, and this router renormalises probabilities rather than
/// re-softmaxing the selected logits.
pub fn router_probabilities(t: &[f32], proj: &[f32], experts: usize) -> Result<Vec<f32>> {
    let logits = crate::linear::linear_row(t, proj, experts, None)?;
    if let Some(bad) = logits.iter().position(|l| !l.is_finite()) {
        return Err(Error::Numerical {
            detail: format!("router logit {bad} is {}", logits[bad]),
        });
    }
    Ok(softmax(&logits))
}

/// Multiply a route's coefficients by each selected expert's own scale.
///
/// Applied **after** renormalisation and never renormalised away, exactly as
/// the pinned source does (`top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]`
/// is the last statement of the router). So a scaled route's coefficients do
/// not sum to one, and a combine that "fixed" that would delete a trained
/// parameter. Negative and zero scales are legal: this is a learned tensor, not
/// a probability.
pub fn apply_per_expert_scale(route: &Route, per_expert: &[f32]) -> Result<Route> {
    let mut weights = crate::try_vec(route.weights.len())?;
    for (e, w) in route.experts.iter().zip(route.weights.iter()) {
        let s = per_expert
            .get(*e as usize)
            .copied()
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!(
                    "expert {e} has no per-expert scale among {}",
                    per_expert.len()
                ),
            })?;
        if !s.is_finite() {
            return Err(Error::Numerical {
                detail: format!("per-expert scale {e} is {s}"),
            });
        }
        weights.push(w * s);
    }
    Ok(Route {
        experts: crate::try_clone_slice(&route.experts)?,
        weights,
    })
}

/// The closed parameters of one router.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterSpec {
    pub experts: usize,
    pub top_k: usize,
    /// Epsilon of the router's own scale-free RMS normalization.
    pub eps: f32,
    /// The scalar applied to the normalized, gained row before projection.
    pub input_scale: f32,
}

/// The closed parameters of one routed expert feed-forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertSpec {
    pub experts: usize,
    pub hidden: usize,
    /// One expert's intermediate width, not the dense MLP's.
    pub intermediate: usize,
    pub activation: moxie_graph::ExpertActivation,
}

/// The whole router for one row: transform, project, softmax, select, scale.
///
/// The unfused stages above stay public so that a test can pin each one on its
/// own; this is their composition in the pinned order, and the order is the
/// point. Feeding the router a normalized stream, or applying the per-expert
/// scale before renormalisation, both produce an ordinary-looking route that
/// sends rows to the wrong experts.
pub fn router_route_row(
    x: &[f32],
    gain: &[f32],
    proj: &[f32],
    per_expert: Option<&[f32]>,
    spec: RouterSpec,
) -> Result<Route> {
    let t = router_input_row(x, gain, spec.input_scale, spec.eps)?;
    let probs = router_probabilities(&t, proj, spec.experts)?;
    let route = select_top_k(&probs, spec.top_k)?;
    match per_expert {
        Some(scale) => apply_per_expert_scale(&route, scale),
        None => Ok(route),
    }
}

/// One expert's gated feed-forward over one row, from the **fused** tensors.
///
/// `gate_up` is `[experts, 2 * intermediate, hidden]` and `down` is
/// `[experts, hidden, intermediate]`, both logically row-major, which is how
/// the designated artifact stores every expert of a layer in two tensors. The
/// gate/up split is `chunk(2, dim=-1)` in the pinned source: the **first**
/// `intermediate` outputs are the gate and the next `intermediate` are the up,
/// not interleaved pairs. Swapping them is a silent wrong answer, because both
/// halves have the same shape.
///
/// Slicing rather than materialising: this reads expert `e`'s window of the
/// fused tensor in place, which is also the shape of the residency question
/// task 0020 inherits -- a chunk of a fused tensor, not a tensor of its own.
pub fn expert_row(
    x: &[f32],
    gate_up: &[f32],
    down: &[f32],
    expert: u32,
    spec: ExpertSpec,
) -> Result<Vec<f32>> {
    let ExpertSpec {
        experts,
        hidden,
        intermediate,
        activation,
    } = spec;
    if hidden == 0 || intermediate == 0 || experts == 0 {
        return Err(Error::InvalidRequest {
            field: "expert_mlp",
            detail: format!("{experts} experts, hidden {hidden}, intermediate {intermediate}"),
        });
    }
    if x.len() != hidden {
        return Err(Error::InvalidArtifact {
            detail: format!("expert input has {} elements, expected {hidden}", x.len()),
        });
    }
    let e = expert as usize;
    if e >= experts {
        return Err(Error::InvalidRequest {
            field: "expert",
            detail: format!("expert {expert} of {experts}"),
        });
    }
    let gate_up_stride = 2 * intermediate * hidden;
    let down_stride = hidden * intermediate;
    if gate_up.len() != experts * gate_up_stride {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "fused gate/up has {} elements, expected {experts}x{}x{hidden}",
                gate_up.len(),
                2 * intermediate
            ),
        });
    }
    if down.len() != experts * down_stride {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "fused down has {} elements, expected {experts}x{hidden}x{intermediate}",
                down.len()
            ),
        });
    }
    let gu = &gate_up[e * gate_up_stride..(e + 1) * gate_up_stride];
    let projected = crate::linear::linear_row(x, gu, 2 * intermediate, None)?;
    let (gate, up) = projected.split_at(intermediate);
    let activated = match activation {
        moxie_graph::ExpertActivation::GeGlu => crate::activation::geglu_row(gate, up)?,
        moxie_graph::ExpertActivation::SwiGlu => crate::activation::swiglu_row(gate, up)?,
    };
    crate::linear::linear_row(
        &activated,
        &down[e * down_stride..(e + 1) * down_stride],
        hidden,
        None,
    )
}

/// The slot order a row's expert contributions are summed in.
///
/// Returns indices into the route's own selection order. Floating-point
/// addition is not associative, so this is the difference between two answers
/// rather than a scheduling preference -- which is why it is returned as data a
/// test can assert on rather than hidden inside the reduction.
pub fn combine_order(experts: &[u32], order: moxie_graph::CombineOrder) -> Result<Vec<usize>> {
    let mut slots: Vec<usize> = crate::try_vec(experts.len())?;
    slots.extend(0..experts.len());
    if order == moxie_graph::CombineOrder::AscendingExpertId {
        // Expert ids within one row are distinct, so this total order has no
        // ties to break.
        slots.sort_by_key(|j| experts[*j]);
    }
    Ok(slots)
}

/// `y = Σ_j w[j] · slots[j]`, summed in the stated order.
///
/// `slots` holds this row's `top_k` expert outputs contiguously, slot `j` at
/// `j * width`, in the route's **selection** order. `order` decides the
/// summation order only; it never reorders the slots themselves, because a slot
/// belongs to the expert the route selected at that position.
///
/// One declared difference from the pinned reference, stated rather than
/// discovered: `Gemma4TextExperts.forward` narrows each weighted contribution
/// to the model dtype before accumulating (`index_add_` over a BF16 buffer),
/// while this reference accumulates the `top_k` terms in FP32 and leaves the
/// single rounding to the node boundary, as every other operation in this crate
/// does. The gap is bounded by `metric::bound(top_k, Σ|w_j · y_j|)` plus one
/// BF16 rounding. Whether it matters to output quality is O2's question and
/// needs paired output against the released model.
pub fn combine_row(
    experts: &[u32],
    weights: &[f32],
    slots: &[f32],
    width: usize,
    order: moxie_graph::CombineOrder,
) -> Result<Vec<f32>> {
    if width == 0 {
        return Err(Error::InvalidRequest {
            field: "combine",
            detail: "a combination over zero features".into(),
        });
    }
    if experts.len() != weights.len() {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "route has {} experts and {} coefficients",
                experts.len(),
                weights.len()
            ),
        });
    }
    if slots.len() != experts.len() * width {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "slot tensor has {} elements, expected {}x{width}",
                slots.len(),
                experts.len()
            ),
        });
    }
    let mut acc = crate::try_vec(width)?;
    acc.extend(std::iter::repeat_n(0f32, width));
    for j in combine_order(experts, order)? {
        let w = weights[j];
        for (a, v) in acc.iter_mut().zip(&slots[j * width..(j + 1) * width]) {
            *a += w * v;
        }
    }
    Ok(acc)
}

/// The scale a [`combine_row`] error bound is stated against: `Σ_j |w_j · y_j|`
/// per output component, reduced to its maximum.
///
/// Not `|y|`: contributions with opposite signs cancel, and document 07 asks
/// for that case to be stressed rather than hidden behind a relative error.
pub fn combine_scale(weights: &[f32], slots: &[f32], width: usize) -> f64 {
    let mut worst = 0f64;
    for d in 0..width {
        let mut sum = 0f64;
        for (j, w) in weights.iter().enumerate() {
            sum += (*w as f64 * slots[j * width + d] as f64).abs();
        }
        if sum > worst {
            worst = sum;
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    use moxie_graph::{CombineOrder, ExpertActivation};

    /// An FP64 transcription of `Gemma4TextRouter.forward`, written from the
    /// pinned source rather than from [`router_route_row`].
    ///
    /// Returns the full probability vector and the scaled coefficients, so a
    /// test can check the selection exactly and the coefficients against a
    /// bound.
    fn fp64_router(
        x: &[f32],
        gain: &[f32],
        proj: &[f32],
        per_expert: Option<&[f32]>,
        spec: RouterSpec,
    ) -> (Vec<u32>, Vec<f64>) {
        let RouterSpec {
            experts,
            top_k,
            eps,
            input_scale,
        } = spec;
        let h = x.len();
        // hidden_states = self.norm(hidden_states)   -- with_scale = False
        let mean_sq: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / h as f64;
        let inv = (mean_sq + eps as f64).powf(-0.5);
        // hidden_states = hidden_states * self.scale * self.scalar_root_size
        let t: Vec<f64> = x
            .iter()
            .zip(gain)
            .map(|(v, g)| (*v as f64) * inv * (*g as f64) * (input_scale as f64))
            .collect();
        // expert_scores = self.proj(hidden_states)
        let logits: Vec<f64> = (0..experts)
            .map(|o| (0..h).map(|i| t[i] * proj[o * h + i] as f64).sum())
            .collect();
        // router_probabilities = softmax(expert_scores)
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / total).collect();
        // top_k_weights, top_k_index = torch.topk(...), with the lower id
        // winning a tie -- the shared rule, not torch's unspecified one.
        let mut order: Vec<u32> = (0..experts as u32).collect();
        order.sort_by(|a, b| {
            probs[*b as usize]
                .partial_cmp(&probs[*a as usize])
                .unwrap()
                .then(a.cmp(b))
        });
        let ids: Vec<u32> = order.into_iter().take(top_k).collect();
        // top_k_weights /= top_k_weights.sum(...)
        let mass: f64 = ids.iter().map(|e| probs[*e as usize]).sum();
        // top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        let weights: Vec<f64> = ids
            .iter()
            .map(|e| {
                let w = probs[*e as usize] / mass;
                match per_expert {
                    Some(s) => w * s[*e as usize] as f64,
                    None => w,
                }
            })
            .collect();
        (ids, weights)
    }

    /// An FP64 transcription of `Gemma4TextExperts.forward` for one expert.
    fn fp64_expert(
        x: &[f32],
        gate_up: &[f32],
        down: &[f32],
        expert: usize,
        spec: ExpertSpec,
    ) -> Vec<f64> {
        let ExpertSpec {
            hidden,
            intermediate,
            activation,
            ..
        } = spec;
        // gate, up = linear(current_state, self.gate_up_proj[e]).chunk(2, -1)
        let stride = 2 * intermediate * hidden;
        let gu = &gate_up[expert * stride..(expert + 1) * stride];
        let projected: Vec<f64> = (0..2 * intermediate)
            .map(|o| {
                (0..hidden)
                    .map(|i| x[i] as f64 * gu[o * hidden + i] as f64)
                    .sum()
            })
            .collect();
        // current_hidden_states = self.act_fn(gate) * up
        let activated: Vec<f64> = (0..intermediate)
            .map(|i| {
                let g = projected[i];
                let u = projected[intermediate + i];
                match activation {
                    // The GeGLU contract rounds the gate term to BF16 before
                    // the product; SwiGLU's does not. Both boundaries are
                    // transcribed here rather than assumed away.
                    ExpertActivation::GeGlu => {
                        let t = 0.5
                            * g
                            * (1.0
                                + (0.797_884_560_802_865_4 * (g + 0.044_715 * g * g * g)).tanh());
                        crate::bf16_round(t as f32) as f64 * u
                    }
                    ExpertActivation::SwiGlu => (g / (1.0 + (-g).exp())) * u,
                }
            })
            .collect();
        // current_hidden_states = linear(current_hidden_states, self.down_proj[e])
        let dstride = hidden * intermediate;
        let d = &down[expert * dstride..(expert + 1) * dstride];
        (0..hidden)
            .map(|o| {
                (0..intermediate)
                    .map(|i| activated[i] * d[o * intermediate + i] as f64)
                    .sum()
            })
            .collect()
    }

    fn pattern(n: usize, offset: usize) -> Vec<f32> {
        (0..n)
            .map(|i| crate::bf16_round(((i * 13 + offset) % 31) as f32 / 32.0 - 0.5))
            .collect()
    }

    #[test]
    fn the_router_matches_its_fp64_transcription_exactly_in_selection() {
        // Two shapes, as the extension rule requires: a wide router choosing
        // few, and a narrow one choosing all of them.
        for (hidden, experts, top_k) in [(24usize, 5usize, 2usize), (8, 3, 3)] {
            let x = pattern(hidden, 1);
            let gain = pattern(hidden, 5);
            let proj = pattern(experts * hidden, 7);
            let scale = pattern(experts, 3);
            let eps = 1e-6f32;
            let input_scale = (hidden as f64).sqrt().recip() as f32;

            let spec = RouterSpec {
                experts,
                top_k,
                eps,
                input_scale,
            };
            let got = router_route_row(&x, &gain, &proj, Some(&scale), spec).unwrap();
            let (ids, weights) = fp64_router(&x, &gain, &proj, Some(&scale), spec);

            // Selection is integer and must agree exactly. A coefficient that
            // is a few ulps out is a rounding difference; a different expert is
            // a different set of weights to make resident.
            assert_eq!(got.experts, ids, "selection at hidden {hidden}");

            // Coefficients against a counted bound rather than an invented
            // tolerance: the norm reduces `hidden` terms, the projection
            // another `hidden`, the softmax and the renormalisation are a
            // handful more, and the per-expert multiply is one.
            let steps = (2 * hidden + top_k + 6) as u64;
            let mut errors = Vec::new();
            for (w, want) in got.weights.iter().zip(&weights) {
                let bound = crate::metric::bound(steps, want.abs().max(1.0));
                let err = (*w as f64 - want).abs();
                assert!(
                    err <= bound,
                    "coefficient error {err:.3e} exceeded bound {bound:.3e}"
                );
                errors.push(err);
            }
            let summary = crate::metric::ErrorSummary::absolute(&got.weights, &weights);
            assert!(summary.max.is_finite() && summary.rms.is_finite());
            assert_eq!(summary.count, errors.len());
        }
    }

    #[test]
    fn the_per_expert_scale_is_applied_after_renormalisation() {
        // The property a "tidier" implementation destroys: with a per-expert
        // scale the coefficients do **not** sum to one, because the scale is
        // applied last and never renormalised away.
        let hidden = 12;
        let experts = 4;
        let x = pattern(hidden, 2);
        let gain = pattern(hidden, 9);
        let proj = pattern(experts * hidden, 4);
        let scale = vec![2.0f32, 0.5, -1.0, 4.0];
        let spec = RouterSpec {
            experts,
            top_k: 3,
            eps: 1e-6,
            input_scale: 1.0,
        };
        let unscaled = router_route_row(&x, &gain, &proj, None, spec).unwrap();
        let scaled = router_route_row(&x, &gain, &proj, Some(&scale), spec).unwrap();

        assert_eq!(unscaled.experts, scaled.experts, "the scale is not a score");
        let sum: f32 = unscaled.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "unscaled coefficients sum to {sum}"
        );
        for (j, e) in scaled.experts.iter().enumerate() {
            let want = unscaled.weights[j] * scale[*e as usize];
            assert!((scaled.weights[j] - want).abs() <= 1e-6 * want.abs().max(1.0));
        }
        // A negative scale is legal: this is a learned tensor, not a
        // probability, and clamping it would be a silent edit of the model.
        assert!(scaled.weights.iter().any(|w| *w < 0.0));
    }

    #[test]
    fn the_router_input_scale_is_load_bearing() {
        // Substituting the conventional value -- no scalar at all -- must
        // change the route. It changes the logits' spread, which changes the
        // softmax, which can change the selection and always changes the
        // coefficients.
        let hidden = 16;
        let experts = 6;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 - 8.0) / 4.0).collect();
        let gain = pattern(hidden, 6);
        let proj = pattern(experts * hidden, 10);
        let with = router_route_row(
            &x,
            &gain,
            &proj,
            None,
            RouterSpec {
                experts,
                top_k: 2,
                eps: 1e-6,
                input_scale: 0.25,
            },
        )
        .unwrap();
        let without = router_route_row(
            &x,
            &gain,
            &proj,
            None,
            RouterSpec {
                experts,
                top_k: 2,
                eps: 1e-6,
                input_scale: 1.0,
            },
        )
        .unwrap();
        assert_ne!(with.weights, without.weights);
    }

    #[test]
    fn the_router_gain_is_load_bearing() {
        let hidden = 16;
        let experts = 6;
        let x = pattern(hidden, 0);
        let gain = pattern(hidden, 6);
        let ones = vec![1.0f32; hidden];
        let proj = pattern(experts * hidden, 10);
        let spec = RouterSpec {
            experts,
            top_k: 2,
            eps: 1e-6,
            input_scale: 1.0,
        };
        let with = router_route_row(&x, &gain, &proj, None, spec).unwrap();
        let without = router_route_row(&x, &ones, &proj, None, spec).unwrap();
        assert_ne!(with.weights, without.weights);
    }

    #[test]
    fn a_router_tie_is_broken_by_the_lower_expert_id() {
        // Every expert has the same projection row, so every logit is equal and
        // the tie spans the whole set. Document 07's "router tie fixtures with
        // fixed logits", at the level of the full router rather than of bare
        // probabilities.
        let hidden = 8;
        let experts = 4;
        let x = pattern(hidden, 1);
        let gain = pattern(hidden, 2);
        let row = pattern(hidden, 3);
        let mut proj = Vec::new();
        for _ in 0..experts {
            proj.extend(row.iter().copied());
        }
        for _ in 0..16 {
            let r = router_route_row(
                &x,
                &gain,
                &proj,
                None,
                RouterSpec {
                    experts,
                    top_k: 2,
                    eps: 1e-6,
                    input_scale: 1.0,
                },
            )
            .unwrap();
            assert_eq!(r.experts, vec![0, 1]);
            assert_eq!(r.weights, vec![0.5, 0.5]);
        }
    }

    #[test]
    fn an_expert_matches_its_fp64_transcription_and_reads_only_its_own_slice() {
        for (hidden, intermediate, experts, activation) in [
            (8usize, 5usize, 3usize, ExpertActivation::GeGlu),
            (12, 4, 2, ExpertActivation::SwiGlu),
        ] {
            let spec = ExpertSpec {
                experts,
                hidden,
                intermediate,
                activation,
            };
            let x = pattern(hidden, 4);
            let gate_up = pattern(experts * 2 * intermediate * hidden, 2);
            let down = pattern(experts * hidden * intermediate, 13);
            for e in 0..experts {
                let got = expert_row(&x, &gate_up, &down, e as u32, spec).unwrap();
                let want = fp64_expert(&x, &gate_up, &down, e, spec);
                for (d, (g, w)) in got.iter().zip(&want).enumerate() {
                    // Two chained reductions of `hidden` and `intermediate`
                    // terms, plus the activation.
                    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1.0);
                    let bound = crate::metric::bound((hidden + intermediate + 4) as u64, scale);
                    assert!(
                        (*g as f64 - w).abs() <= bound,
                        "expert {e} component {d}: {:.3e} vs bound {bound:.3e}",
                        (*g as f64 - w).abs()
                    );
                }
            }
            // Overwriting a *different* expert's slice must not change this
            // one's answer. A fused tensor read with the wrong stride passes
            // every shape check and silently mixes experts.
            let mut perturbed = gate_up.clone();
            let stride = 2 * intermediate * hidden;
            for v in perturbed[stride..2 * stride].iter_mut() {
                *v = 0.25;
            }
            let before = expert_row(&x, &gate_up, &down, 0, spec).unwrap();
            let after = expert_row(&x, &perturbed, &down, 0, spec).unwrap();
            assert_eq!(before, after, "expert 0 read expert 1's slice");
        }
    }

    #[test]
    fn the_gate_block_precedes_the_up_block() {
        // Both halves have the same shape, so swapping them is a silent wrong
        // answer rather than an error. Swapping the two blocks of the fused
        // tensor must change the result.
        let (hidden, intermediate) = (6usize, 4usize);
        let gate_up = pattern(2 * intermediate * hidden, 2);
        let mut swapped = gate_up.clone();
        let half = intermediate * hidden;
        swapped[..half].copy_from_slice(&gate_up[half..]);
        swapped[half..].copy_from_slice(&gate_up[..half]);
        let down = pattern(hidden * intermediate, 5);
        let x = pattern(hidden, 1);
        let spec = ExpertSpec {
            experts: 1,
            hidden,
            intermediate,
            activation: ExpertActivation::GeGlu,
        };
        let a = expert_row(&x, &gate_up, &down, 0, spec).unwrap();
        let b = expert_row(&x, &swapped, &down, 0, spec).unwrap();
        assert_ne!(a, b, "the gate/up order is not load-bearing");
    }

    #[test]
    fn the_two_expert_activations_are_different_functions() {
        let (hidden, intermediate) = (6usize, 4usize);
        let gate_up = pattern(2 * intermediate * hidden, 3);
        let down = pattern(hidden * intermediate, 7);
        let x = pattern(hidden, 2);
        let ge = expert_row(
            &x,
            &gate_up,
            &down,
            0,
            ExpertSpec {
                experts: 1,
                hidden,
                intermediate,
                activation: ExpertActivation::GeGlu,
            },
        )
        .unwrap();
        let si = expert_row(
            &x,
            &gate_up,
            &down,
            0,
            ExpertSpec {
                experts: 1,
                hidden,
                intermediate,
                activation: ExpertActivation::SwiGlu,
            },
        )
        .unwrap();
        assert_ne!(ge, si);
    }

    #[test]
    fn the_combination_order_is_a_parameter_and_it_changes_the_answer() {
        // Ascending expert id is the pinned reference's order. Selection order
        // is the other real one. They agree only when the route already
        // selected experts in ascending id order, so the fixture picks one
        // where it did not.
        let route = Route {
            experts: vec![2, 0, 1],
            weights: vec![0.5, 0.25, 0.25],
        };
        assert_eq!(
            combine_order(&route.experts, CombineOrder::AscendingExpertId).unwrap(),
            vec![1, 2, 0]
        );
        assert_eq!(
            combine_order(&route.experts, CombineOrder::SelectionOrder).unwrap(),
            vec![0, 1, 2]
        );

        // Values chosen so that FP32 addition is not associative over them.
        // `1.0`'s ulp is `f32::EPSILON`; each small term is three eighths of
        // one. Added to `1.0` separately, each rounds away and the answer stays
        // `1.0`; added to each other first they make three quarters of an ulp,
        // which rounds `1.0` up. Same three numbers, two answers.
        let big = 1.0f32;
        let small = f32::EPSILON * 0.375;
        let slots = vec![big, small, small];
        let route = Route {
            experts: vec![2, 0, 1],
            weights: vec![1.0, 1.0, 1.0],
        };
        let ascending = combine_row(
            &route.experts,
            &route.weights,
            &slots,
            1,
            CombineOrder::AscendingExpertId,
        )
        .unwrap();
        let selection = combine_row(
            &route.experts,
            &route.weights,
            &slots,
            1,
            CombineOrder::SelectionOrder,
        )
        .unwrap();
        assert_ne!(
            ascending, selection,
            "the reduction order made no difference, so the fixture is not testing it"
        );
    }

    #[test]
    fn combination_matches_its_fp64_transcription_within_its_bound() {
        let width = 5usize;
        let route = Route {
            experts: vec![3, 1, 0],
            weights: vec![0.5, 0.3, 0.2],
        };
        let slots = pattern(route.experts.len() * width, 6);
        for order in [
            CombineOrder::AscendingExpertId,
            CombineOrder::SelectionOrder,
        ] {
            let got = combine_row(&route.experts, &route.weights, &slots, width, order).unwrap();
            let slot_order = combine_order(&route.experts, order).unwrap();
            let mut errors = Vec::new();
            for d in 0..width {
                // Transcribed independently: the same terms, summed in the same
                // order, in FP64.
                let want: f64 = slot_order
                    .iter()
                    .map(|j| route.weights[*j] as f64 * slots[j * width + d] as f64)
                    .sum();
                let bound = crate::metric::bound(
                    route.experts.len() as u64,
                    combine_scale(&route.weights, &slots, width),
                );
                let err = (got[d] as f64 - want).abs();
                assert!(err <= bound, "component {d}: {err:.3e} vs {bound:.3e}");
                errors.push(err);
            }
            // Document 07 asks for max, RMS and p99 rather than a maximum
            // alone, so the whole component vector is summarised against the
            // transcription rather than the per-component assertion above.
            let want: Vec<f64> = (0..width)
                .map(|d| {
                    slot_order
                        .iter()
                        .map(|j| route.weights[*j] as f64 * slots[j * width + d] as f64)
                        .sum()
                })
                .collect();
            let summary = crate::metric::ErrorSummary::absolute(&got, &want);
            assert_eq!(summary.count, width);
            assert!(summary.rms.is_finite() && summary.p99.is_finite());
            assert!(
                summary.max
                    <= crate::metric::bound(
                        route.experts.len() as u64,
                        combine_scale(&route.weights, &slots, width),
                    )
            );
            assert_eq!(errors.len(), width);
        }
    }

    #[test]
    fn a_real_router_overlaps_routes_and_the_union_is_smaller_than_rows_times_k() {
        // Document 03: "estimate union of required experts over a row batch ...
        // Do not multiply active experts by batch rows when routes overlap; do
        // not assume overlap when they do not." The M0 fixture proved that of
        // hand-written routes; this proves it of routes a real router produced,
        // which is the form the residency authority will actually be handed.
        let hidden = 16;
        let experts = 8;
        let top_k = 3;
        let gain = pattern(hidden, 5);
        let proj = pattern(experts * hidden, 11);
        let spec = RouterSpec {
            experts,
            top_k,
            eps: 1e-6,
            input_scale: (hidden as f64).sqrt().recip() as f32,
        };
        let rows: Vec<Route> = (0..12)
            .map(|r| {
                let x = pattern(hidden, r * 5 + 1);
                router_route_row(&x, &gain, &proj, None, spec).unwrap()
            })
            .collect();

        let union = required_experts(&rows);
        assert!(
            union.len() < rows.len() * top_k,
            "{} distinct experts for {} rows at top-{top_k}: this batch has no \
             overlap, so it is not testing the union",
            union.len(),
            rows.len()
        );
        // And the routes are not all identical either, or the union would be
        // trivially small for the wrong reason.
        assert!(
            rows.iter().any(|r| r.experts != rows[0].experts),
            "every row routed identically"
        );

        // Dispatch covers exactly the union, each expert's rows ascending, and
        // no expert that received nothing appears.
        let batches = dispatch(&rows, experts as u32).unwrap();
        assert_eq!(
            batches.iter().map(|b| b.expert).collect::<Vec<_>>(),
            union,
            "dispatch and the union disagree about what has to be resident"
        );
        for b in &batches {
            assert!(!b.rows.is_empty());
            assert!(b.rows.windows(2).all(|w| w[0] < w[1]));
        }
        // Every (row, slot) pair is dispatched exactly once.
        let dispatched: usize = batches.iter().map(|b| b.rows.len()).sum();
        assert_eq!(dispatched, rows.len() * top_k);
    }

    #[test]
    fn the_new_routing_stages_refuse_malformed_input_with_typed_errors() {
        assert!(router_input_row(&[], &[], 1.0, 1e-6).is_err());
        assert!(router_input_row(&[1.0], &[1.0, 1.0], 1.0, 1e-6).is_err());
        assert!(router_input_row(&[1.0], &[1.0], 1.0, 0.0).is_err());
        assert!(router_input_row(&[1.0], &[1.0], 0.0, 1e-6).is_err());
        assert!(router_input_row(&[1.0], &[1.0], f32::NAN, 1e-6).is_err());

        let route = Route {
            experts: vec![0],
            weights: vec![1.0],
        };
        // A per-expert scale shorter than the expert set.
        assert!(apply_per_expert_scale(&route, &[]).is_err());
        assert_eq!(
            apply_per_expert_scale(&route, &[f32::NAN])
                .unwrap_err()
                .kind(),
            "numerical"
        );
        // An expert outside the fused tensor, and a fused tensor of the wrong
        // extent.
        let x = vec![0.5f32; 4];
        let gate_up = vec![0.5f32; 2 * 3 * 4];
        let down = vec![0.5f32; 4 * 3];
        let spec = ExpertSpec {
            experts: 1,
            hidden: 4,
            intermediate: 3,
            activation: ExpertActivation::GeGlu,
        };
        assert!(expert_row(&x, &gate_up, &down, 1, spec).is_err());
        assert!(expert_row(&x, &gate_up[..4], &down, 0, spec).is_err());
        // A slot tensor that does not cover the route.
        assert!(
            combine_row(
                &route.experts,
                &route.weights,
                &[],
                4,
                CombineOrder::SelectionOrder
            )
            .is_err()
        );
    }

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
