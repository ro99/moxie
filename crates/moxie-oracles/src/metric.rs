//! Error metrics, and the bound every numerical gate is stated against.
//!
//! Document 07: "For each floating-point primitive, the implementation task must
//! specify its absolute/relative or normalized error metric, independently
//! generated reference, relevant scales and threshold before optimization ...
//! Include max, RMS and high-percentile errors."
//!
//! Both halves matter. A test that reports only a maximum hides a distribution
//! that has moved; a threshold chosen after seeing a candidate is not a
//! threshold. So this module supplies the summary *and* the bound, and the bound
//! is arithmetic over a counted number of rounding steps rather than a number
//! someone liked.

/// FP32 unit roundoff, `2^-24`.
pub const FP32_U: f64 = 5.960_464_477_539_063e-8;

/// FP32 underflow unit: half the smallest subnormal, `2^-150`.
///
/// The relative model `fl(x∘y) = (x∘y)(1+δ)` holds only while results stay in
/// the normal range. Under gradual underflow the correct model is
/// `fl(x∘y) = (x∘y)(1+δ) + η` with `|η| ≤ 2^-150`, and the additive term is the
/// only thing bounding an operation whose result is subnormal -- where a
/// relative bound says nothing at all.
///
/// The fifth review disproved a bound over exactly this domain: attention with
/// values at `2^-133` had an absolute error 51 times the purely relative bound.
/// Tiny in magnitude, and still a bound that did not hold over its stated inputs.
pub const FP32_ETA: f64 = 7.006_492_321_624_085e-46;

/// BF16 unit roundoff, `2^-8`.
///
/// For a binary format with `p` significand bits, `ulp(1) = 2^(1-p)` and the
/// round-to-nearest unit roundoff is half of it, `2^-p`. BF16 carries **eight**
/// significand bits (seven stored plus the implicit leading one), so `u = 2^-8`.
/// The same arithmetic gives FP32's `p = 24` the [`FP32_U`] above, which is the
/// cross-check that should have caught this constant: it was first written as
/// `2^-9`, half its true value, by taking half of `2^-8` instead of half of
/// `ulp(1) = 2^-7`.
///
/// `bf16_midpoints_round_by_exactly_this_much` is the regression, and it is
/// exhaustive rather than sampled: the counterexample is the midpoint
/// `1 + 2^-8`, which rounds to `1.0` for an error of exactly `2^-8` — twice
/// what the wrong constant allowed.
///
/// That same value was already named in this crate before this constant
/// existed: `residual.rs` pins `1 + 2^-8` as "exactly halfway between two BF16
/// values" from an earlier review. The fact needed to disprove `2^-9` was
/// therefore sitting in a sibling module, which is the argument for deriving a
/// constant from `p` and testing it exhaustively rather than reasoning about
/// it once in a doc comment.
///
/// **65,536 times FP32's**, which is why a boundary the reference declares is
/// part of an equation rather than a storage detail: it is by far the largest
/// rounding in any chain that contains one.
pub const BF16_U: f64 = 3.906_25e-3;

/// BF16 underflow unit: half the smallest BF16 subnormal, `2^-134`.
///
/// The same role `FP32_ETA` plays for FP32, at BF16's much earlier onset of
/// gradual underflow -- `2^-133` rather than `2^-149`.
pub const BF16_ETA: f64 = 4.591_774_807_899_561e-41;

/// The standard bound for `n` chained FP32 roundings: `γ(n) = n·u / (1 − n·u)`.
///
/// Higham, *Accuracy and Stability of Numerical Algorithms*, §3.1. A sequential
/// sum of `n` products accumulated in FP32 satisfies
/// `|computed − exact| <= γ(n) · Σ|terms|`, which is what the per-operation
/// bounds in task 0003 are built from. The count comes from the equation: `K`
/// products plus a bias add is `γ(K+1)`, an RMS norm over `H` features is
/// `γ(H+4)`, and so on.
///
/// Returns infinity once `n·u >= 1`, because past that point the bound says
/// nothing and pretending otherwise would be worse than failing.
pub fn gamma(n: u64) -> f64 {
    let nu = n as f64 * FP32_U;
    if nu >= 1.0 {
        return f64::INFINITY;
    }
    nu / (1.0 - nu)
}

/// The underflow-aware bound for `n` chained FP32 operations over terms whose
/// magnitudes sum to `scale`: `γ(n)·scale + n·η`.
///
/// Use this rather than `gamma(n) * scale` anywhere a result can be subnormal,
/// which in practice is anywhere real data can be small. The additive term costs
/// nothing when the result is normal -- `n·η` is around `1e-45` -- and is the
/// whole bound when it is not.
pub fn bound(n: u64, scale: f64) -> f64 {
    gamma(n) * scale + n as f64 * FP32_ETA
}

/// Max, RMS and p99 of a set of errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorSummary {
    pub count: usize,
    pub max: f64,
    pub rms: f64,
    pub p99: f64,
}

impl ErrorSummary {
    /// Summarise `|got − want|` elementwise.
    ///
    /// Panics on a length mismatch: comparing tensors of different sizes is a
    /// test defect, not a numerical result.
    pub fn absolute(got: &[f32], want: &[f64]) -> Self {
        assert_eq!(got.len(), want.len(), "comparing different-sized tensors");
        Self::of(got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()))
    }

    /// Summarise `|got − want| / scale`, one scale per element.
    ///
    /// The scale is the magnitude the bound is stated against -- for a dot
    /// product, `Σ|x_k·w_k|`, not `|y|`, because a result near zero from
    /// cancellation has no relative accuracy to speak of and document 07 asks
    /// for exactly that case to be stressed rather than hidden.
    pub fn normalized(got: &[f32], want: &[f64], scale: &[f64]) -> Self {
        assert_eq!(got.len(), want.len(), "comparing different-sized tensors");
        assert_eq!(got.len(), scale.len(), "one scale per element is required");
        Self::of(got.iter().zip(want).zip(scale).map(|((g, w), s)| {
            let e = (*g as f64 - w).abs();
            if *s == 0.0 {
                if e == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                e / s
            }
        }))
    }

    fn of(errors: impl Iterator<Item = f64>) -> Self {
        let mut v: Vec<f64> = errors.collect();
        if v.is_empty() {
            return Self {
                count: 0,
                max: 0.0,
                rms: 0.0,
                p99: 0.0,
            };
        }
        let sum_sq: f64 = v.iter().map(|e| e * e).sum();
        let rms = (sum_sq / v.len() as f64).sqrt();
        v.sort_by(|a, b| a.partial_cmp(b).expect("errors are not NaN"));
        let max = *v.last().expect("non-empty");
        // Nearest-rank p99, which for small samples is the largest element --
        // stated rather than silently interpolated.
        let idx = (((v.len() as f64) * 0.99).ceil() as usize).clamp(1, v.len()) - 1;
        Self {
            count: v.len(),
            max,
            rms,
            p99: v[idx],
        }
    }

    /// Whether every error is within `bound`.
    pub fn within(&self, bound: f64) -> bool {
        self.max <= bound
    }
}

impl core::fmt::Display for ErrorSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "n={} max={:.3e} rms={:.3e} p99={:.3e}",
            self.count, self.max, self.rms, self.p99
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every BF16 value, and the midpoint just above it.
    fn bf16_patterns() -> impl Iterator<Item = (f32, f32)> {
        (0u32..=0xFFFF).filter_map(|bits| {
            let exact = f32::from_bits(bits << 16);
            let midpoint = f32::from_bits((bits << 16) | 0x8000);
            (exact.is_finite() && midpoint.is_finite()).then_some((exact, midpoint))
        })
    }

    #[test]
    fn bf16_midpoints_round_by_exactly_this_much() {
        // The regression for `BF16_U`, exhaustive over all 65,536 patterns.
        // The constant was first written as `2^-9`, half its true value, and
        // an independent review disproved it with one line of arithmetic:
        // `1 + 2^-8` rounds to `1.0`, an error of `2^-8`, against an allowance
        // of `2^-9 · 1.0039`.
        //
        // A midpoint is the worst case by construction -- it is the furthest an
        // input can be from the value it rounds to -- so checking every one of
        // them is checking the constant itself.
        let mut worst = 0f64;
        for (exact, midpoint) in bf16_patterns() {
            for probe in [exact, midpoint] {
                let rounded = crate::bf16_round(probe);
                if !rounded.is_finite() {
                    // Rounding to infinity at the top of the range. The
                    // relative model says nothing about overflow, exactly as it
                    // says nothing about underflow without the `η` term, and
                    // pretending otherwise would be worse than excluding it.
                    // Recorded rather than skipped silently: this happens only
                    // above the largest BF16 value.
                    assert!(
                        probe.abs() > f32::from_bits(0x7F7F_0000),
                        "{probe:e} overflowed BF16 but is inside its range"
                    );
                    continue;
                }
                let error = (rounded as f64 - probe as f64).abs();
                let allowed = BF16_U * (probe as f64).abs() + BF16_ETA;
                assert!(
                    error <= allowed,
                    "{probe:e} rounded to {rounded:e}: error {error:e} exceeded \
                     {allowed:e}"
                );
                // The *relative* model holds only in the normal range -- below
                // it, gradual underflow is what `BF16_ETA` is for, and the
                // ratio there can approach 1. Tracking the worst case over
                // normals is what makes the tightness claim below meaningful.
                if probe.abs() >= 2f32.powi(-126) {
                    worst = worst.max(error / (probe as f64).abs());
                }
            }
        }
        // And the bound is not loose: some input attains it, so a smaller
        // constant would be wrong rather than merely conservative.
        // The bound holds, and is essentially attained: at every midpoint
        // `2^e · (1 + 2^-8)` the error is exactly `2^e · 2^-8`, so measured
        // against `x` the ratio is `u / (1 + u)` -- the standard statement
        // `|fl(x) − x| ≤ u·|x|` is loose by exactly that factor, and no more.
        // A smaller constant would therefore be wrong rather than conservative.
        assert!(worst <= BF16_U, "{worst:e} exceeded {BF16_U:e}");
        let supremum = BF16_U / (1.0 + BF16_U);
        assert!(
            (worst - supremum).abs() <= f64::EPSILON * supremum,
            "the worst relative error {worst:e} is not the expected u/(1+u) \
             {supremum:e}, so this constant is not the tight one"
        );
    }

    #[test]
    fn two_values_straddling_a_bf16_midpoint_land_a_full_ulp_apart() {
        // Why `expert_error_bound` charges `2·u_b·S` at every BF16 boundary
        // rather than absorbing the rounding: the implementation rounds
        // `bf16(fp32 v)` and the transcription rounds `bf16(fp64 v)`, and when
        // those two straddle a midpoint they land a whole ulp apart however
        // close the inputs were.
        let midpoint = 1.0f32 + 2f32.powi(-8);
        let a = midpoint - 2f32.powi(-20);
        let b = midpoint + 2f32.powi(-20);
        let (ra, rb) = (crate::bf16_round(a), crate::bf16_round(b));
        let separated = (rb as f64 - ra as f64).abs();
        assert_eq!(separated, 2f64.powi(-7), "a full BF16 ulp at 1.0");
        assert!(
            separated > (b as f64 - a as f64),
            "the inputs were {:e} apart and the outputs {separated:e}",
            b as f64 - a as f64
        );
        // The allowance the bound actually uses covers it; the halved constant
        // did not.
        let allowed = (b as f64 - a as f64)
            + BF16_U * (a as f64).abs()
            + BF16_U * (b as f64).abs()
            + 2.0 * BF16_ETA;
        assert!(separated <= allowed, "{separated:e} vs {allowed:e}");
    }

    #[test]
    fn the_underflow_units_are_half_the_smallest_subnormal() {
        // FP32's smallest subnormal is `2^-149` and BF16's is `2^-133`, seven
        // mantissa bits below its `2^-126` smallest normal.
        assert_eq!(FP32_ETA, 2f64.powi(-150));
        assert_eq!(BF16_ETA, 2f64.powi(-134));
        // And the two unit roundoffs are `2^-p` for their own precisions.
        assert_eq!(FP32_U, 2f64.powi(-24));
        assert_eq!(BF16_U, 2f64.powi(-8));
    }

    #[test]
    fn gamma_grows_with_the_number_of_roundings_and_is_never_negative() {
        assert!(gamma(1) > 0.0);
        assert!(gamma(2) > gamma(1));
        assert!(gamma(1000) > gamma(100));
        // A single rounding is about one unit of roundoff.
        assert!((gamma(1) - FP32_U).abs() < 1e-14);
        // The bound degrades honestly rather than wrapping.
        assert!(gamma(1 << 25).is_infinite());
    }

    #[test]
    fn a_dot_product_of_length_k_is_bounded_by_gamma_k() {
        // The property the per-operation bounds rest on, checked against a
        // deliberately ill-conditioned sum where FP32 really does lose bits.
        let k = 512usize;
        let x: Vec<f32> = (0..k)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + i as f32 * 1e-3))
            .collect();
        let w: Vec<f32> = (0..k).map(|i| 1.0 / (1.0 + i as f32)).collect();

        let mut acc = 0f32;
        for i in 0..k {
            acc += x[i] * w[i];
        }
        let mut exact = 0f64;
        let mut scale = 0f64;
        for i in 0..k {
            exact += x[i] as f64 * w[i] as f64;
            scale += (x[i] as f64 * w[i] as f64).abs();
        }

        let s = ErrorSummary::normalized(&[acc], &[exact], &[scale]);
        assert!(
            s.within(gamma(k as u64)),
            "{s} exceeded gamma({k}) = {:.3e}",
            gamma(k as u64)
        );
    }

    #[test]
    fn the_underflow_term_dominates_where_the_relative_term_says_nothing() {
        // A purely relative bound goes to zero with the scale; the real error
        // does not, because gradual underflow is additive.
        let tiny = f32::from_bits(1 << 16) as f64; // 2^-133, a subnormal
        assert!(bound(12, tiny) > 12.0 * FP32_ETA * 0.99);
        assert!(
            bound(12, tiny) > gamma(12) * tiny,
            "the additive term must actually be doing the work here"
        );

        // And it costs essentially nothing at ordinary magnitudes.
        let ordinary = 1.0f64;
        let with = bound(12, ordinary);
        let without = gamma(12) * ordinary;
        assert!((with - without) / without < 1e-30);
    }

    #[test]
    fn summary_reports_all_three_statistics() {
        let got = [1.0f32, 2.0, 3.0];
        let want = [1.0f64, 2.5, 3.0];
        let s = ErrorSummary::absolute(&got, &want);
        assert_eq!(s.count, 3);
        assert!((s.max - 0.5).abs() < 1e-12);
        assert!((s.rms - (0.25f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((s.p99 - 0.5).abs() < 1e-12);
        assert!(s.within(0.5));
        assert!(!s.within(0.4));
        assert!(s.to_string().contains("rms="));
    }

    #[test]
    fn an_exact_match_has_zero_error_and_a_zero_scale_is_not_hidden() {
        let s = ErrorSummary::absolute(&[1.0f32], &[1.0f64]);
        assert_eq!(s.max, 0.0);
        assert!(s.within(0.0));

        // A zero scale with zero error is fine; with non-zero error it must not
        // silently divide by zero into something finite.
        assert_eq!(ErrorSummary::normalized(&[0.0], &[0.0], &[0.0]).max, 0.0);
        assert!(
            ErrorSummary::normalized(&[1.0], &[0.0], &[0.0])
                .max
                .is_infinite()
        );
    }

    #[test]
    fn an_empty_comparison_is_empty_rather_than_a_pass() {
        let s = ErrorSummary::absolute(&[], &[]);
        assert_eq!(s.count, 0);
        // Callers assert on `count` when a test could otherwise vacuously pass.
    }
}
