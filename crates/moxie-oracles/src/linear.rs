//! Embedding, linear and vocabulary projection.
//!
//! The three share one reduction, so they share one implementation with one
//! declared summation order. Task 0003 fixes that order as **sequential
//! ascending `k`**: a reference whose result depends on reassociation cannot be
//! the thing a kernel is compared against. Document 07 lets a kernel reassociate
//! later, with measured error and quality evidence; this is the baseline that
//! evidence is produced against.
//!
//! Accumulation is FP32 throughout (`AccumulationPolicy::Bf16InF32Acc`). Where
//! the result is rounded to BF16 is the **caller's** decision, because task 0003
//! makes it a per-operation semantic boundary: `Linear` rounds once on the final
//! sum, `VocabProjection` does not round at all. Neither rounds inside the
//! reduction, and nothing here does either.

use moxie_types::{Error, Result};

/// One row of `y = x·Wᵀ (+ b)`.
///
/// `x` is `K` long, `w` is `[O, K]` row-major, `bias` is `O` long or absent.
/// Returns `O` values, unrounded, in FP32.
pub fn linear_row(
    x: &[f32],
    w: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let k = x.len();
    if k == 0 || out_features == 0 {
        return Err(Error::InvalidRequest {
            field: "linear",
            detail: format!("degenerate shape: {k} inputs, {out_features} outputs"),
        });
    }
    if w.len() != out_features * k {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "weight has {} elements, expected {out_features}x{k}",
                w.len()
            ),
        });
    }
    if let Some(b) = bias
        && b.len() != out_features
    {
        return Err(Error::InvalidArtifact {
            detail: format!("bias has {} elements, expected {out_features}", b.len()),
        });
    }

    let mut out = Vec::with_capacity(out_features);
    for o in 0..out_features {
        // Sequential ascending k, FP32 accumulator. Not reassociated, not
        // fused-multiply-added into a different rounding pattern.
        let mut acc = 0f32;
        for (i, xi) in x.iter().enumerate() {
            acc += xi * w[o * k + i];
        }
        if let Some(b) = bias {
            acc += b[o];
        }
        out.push(acc);
    }
    Ok(out)
}

/// The scale a `linear_row` error bound is stated against: `Σ|x_k·w_ok|`.
///
/// Not `|y|`. A dot product whose terms cancel has a small result and no
/// relative accuracy in it, and document 07 asks for that case to be stressed
/// rather than hidden behind a relative error that looks enormous.
pub fn linear_row_scale(
    x: &[f32],
    w: &[f32],
    out_features: usize,
    bias: Option<&[f32]>,
) -> Vec<f64> {
    let k = x.len();
    (0..out_features)
        .map(|o| {
            let mut s: f64 = x
                .iter()
                .enumerate()
                .map(|(i, xi)| (*xi as f64 * w[o * k + i] as f64).abs())
                .sum();
            if let Some(b) = bias {
                s += (b[o] as f64).abs();
            }
            s
        })
        .collect()
}

/// One embedding lookup: `y = table[token]`.
///
/// A copy. No arithmetic, so the result is bit-identical to the stored row, and
/// task 0003 declares its error metric to be exactness rather than a bound.
///
/// An out-of-range token is a typed error. Document 02 gives token ids their own
/// integer role precisely so they are not treated as data to be clamped.
pub fn embedding_row(
    tokens_id: u32,
    table: &[f32],
    vocab: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    if table.len() != vocab * hidden {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "embedding table has {} elements, expected {vocab}x{hidden}",
                table.len()
            ),
        });
    }
    let t = tokens_id as usize;
    if t >= vocab {
        return Err(Error::InvalidRequest {
            field: "token",
            detail: format!("token {t} is outside a vocabulary of {vocab}"),
        });
    }
    Ok(table[t * hidden..(t + 1) * hidden].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    /// The equation from task 0003, written out again in FP64. This is the
    /// independent reference: it is not the implementation with a wider type,
    /// it is the specification transcribed separately.
    fn linear_row_f64(x: &[f32], w: &[f32], out_features: usize, bias: Option<&[f32]>) -> Vec<f64> {
        let k = x.len();
        (0..out_features)
            .map(|o| {
                let mut acc = 0f64;
                for i in 0..k {
                    acc += x[i] as f64 * w[o * k + i] as f64;
                }
                if let Some(b) = bias {
                    acc += b[o] as f64;
                }
                acc
            })
            .collect()
    }

    #[test]
    fn a_linear_row_is_exact_on_representable_inputs() {
        // Powers of two and small integers: every product and partial sum is
        // exactly an FP32 number, so the contract says bit-exact, not "close".
        let x = [1.0f32, 2.0, 4.0, 0.5];
        let w = [
            1.0f32, 1.0, 1.0, 1.0, // sum = 7.5
            2.0, 0.0, -1.0, 8.0, // 2 + 0 - 4 + 4 = 2
        ];
        let b = [0.25f32, -2.0];
        let got = linear_row(&x, &w, 2, Some(&b)).unwrap();
        assert_eq!(got, vec![7.75, 0.0]);
        // ... and it agrees with the FP64 transcription exactly.
        let want = linear_row_f64(&x, &w, 2, Some(&b));
        assert_eq!(got[0] as f64, want[0]);
        assert_eq!(got[1] as f64, want[1]);
    }

    #[test]
    fn a_linear_row_stays_inside_gamma_k_plus_one() {
        // The declared bound: gamma(K+1) times the sum of absolute term
        // magnitudes, K products-and-adds plus the bias.
        let k = 256usize;
        let out = 4usize;
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let w: Vec<f32> = (0..out * k)
            .map(|i| ((i * 53 % 199) as f32 - 99.0) / 313.0)
            .collect();
        let b: Vec<f32> = (0..out).map(|o| o as f32 * 0.125).collect();

        let got = linear_row(&x, &w, out, Some(&b)).unwrap();
        let want = linear_row_f64(&x, &w, out, Some(&b));
        let scale = linear_row_scale(&x, &w, out, Some(&b));

        let s = ErrorSummary::normalized(&got, &want, &scale);
        let bound = gamma(k as u64 + 1);
        assert_eq!(s.count, out);
        assert!(
            s.within(bound),
            "{s} exceeded gamma({}) = {bound:.3e}",
            k + 1
        );
    }

    #[test]
    fn a_cancelling_dot_product_is_measured_against_term_magnitude() {
        // Document 07: "stress cancellation and near-zero outputs rather than
        // only relative error". The result here is ~0 while the terms are large;
        // a relative-to-|y| metric would report a meaningless ratio.
        let k = 128usize;
        let x: Vec<f32> = (0..k)
            .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 })
            .collect();
        let w: Vec<f32> = (0..k).map(|i| 1.0 + (i % 2) as f32 * 1e-7).collect();

        let got = linear_row(&x, &w, 1, None).unwrap();
        let want = linear_row_f64(&x, &w, 1, None);
        let scale = linear_row_scale(&x, &w, 1, None);
        assert!(scale[0] > 1e8, "the terms really are large: {}", scale[0]);
        assert!(want[0].abs() < 1e3, "and the result really does cancel");

        let s = ErrorSummary::normalized(&got, &want, &scale);
        assert!(s.within(gamma(k as u64 + 1)), "{s}");
    }

    #[test]
    fn the_summation_order_is_part_of_the_contract() {
        // Reversing the reduction changes the answer on inputs where FP32 loses
        // bits. That is not a bug; it is why the order is declared, and why a
        // kernel that reassociates owes measured evidence rather than a shrug.
        let k = 64usize;
        let x: Vec<f32> = (0..k).map(|i| if i == 0 { 1.0 } else { 1e-9 }).collect();
        let w = vec![1.0f32; k];

        let forward = linear_row(&x, &w, 1, None).unwrap()[0];
        let mut backward = 0f32;
        for i in (0..k).rev() {
            backward += x[i] * w[i];
        }
        assert_ne!(
            forward, backward,
            "if these agree the fixture is too well conditioned to prove anything"
        );
        // Both are within the bound; neither is "wrong". The reference is the
        // one the contract names.
        let want = linear_row_f64(&x, &w, 1, None);
        let scale = linear_row_scale(&x, &w, 1, None);
        assert!(ErrorSummary::normalized(&[forward], &want, &scale).within(gamma(k as u64 + 1)));
    }

    #[test]
    fn shapes_and_degenerate_cases_are_typed_errors() {
        assert!(linear_row(&[], &[], 1, None).is_err());
        assert!(linear_row(&[1.0], &[1.0], 0, None).is_err());
        assert!(
            linear_row(&[1.0, 2.0], &[1.0], 1, None).is_err(),
            "short weight"
        );
        assert!(
            linear_row(&[1.0], &[1.0], 1, Some(&[1.0, 2.0])).is_err(),
            "bias of the wrong length"
        );
    }

    #[test]
    fn an_embedding_lookup_is_an_exact_copy() {
        let table: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let row = embedding_row(2, &table, 4, 3).unwrap();
        assert_eq!(row, vec![1.5, 1.75, 2.0]);
        // Bit-identical, including the sign of zero.
        let signed = [-0.0f32, 1.0, -0.0, 2.0];
        let r = embedding_row(0, &signed, 2, 2).unwrap();
        assert!(r[0].is_sign_negative());
    }

    #[test]
    fn an_out_of_range_token_is_refused_not_wrapped() {
        let table = vec![0f32; 6];
        assert!(embedding_row(1, &table, 3, 2).is_ok());
        assert!(embedding_row(3, &table, 3, 2).is_err());
        assert!(embedding_row(u32::MAX, &table, 3, 2).is_err());
        assert!(embedding_row(0, &table, 4, 2).is_err(), "table too small");
    }
}
