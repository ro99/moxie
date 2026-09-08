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
