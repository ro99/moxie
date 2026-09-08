//! Gated activations.
//!
//! Document 02 lists "SwiGLU, GeGLU, and parameterized SiLU/SiTU variants" as
//! distinct operations. Only SwiGLU is implemented in task 0003's slice. The
//! others are **absent rather than approximated**: R06 records that Kimi's
//! SiTU-GLU is "explicitly bounded in both gate and up terms, not ordinary
//! SwiGLU", so substituting one for the other would be numerically wrong, not
//! merely slower.
//!
//! Document 02 also notes that "a gate/up tensor's logical order is not its
//! physical interleaved disk layout". These functions take two separate slices;
//! any interleaving is an importer's problem and never reaches the mathematics.

use moxie_types::{Error, Result};

/// The logistic sigmoid, in the overflow-free form.
///
/// `1/(1 + e^{−v})` overflows for `v` around −88 in FP32: `e^{88}` is already
/// past `f32::MAX`, so the denominator becomes `+inf` and the whole expression
/// collapses to zero. The algebraically identical `e^v / (1 + e^v)` has the same
/// problem mirrored, for large positive `v`.
///
/// Choosing by sign keeps the exponent negative in both branches, so the
/// argument to `exp` is never larger than zero and the result is never `inf`.
/// This is the form the pinned legacy source already uses
/// (`src/platform/numerics.cpp:44`), and document 08 says to read those
/// references before inventing a replacement. The first version of this file did
/// not, and the fourth review found the overflow.
pub fn sigmoid(v: f32) -> f32 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

/// `silu(v) = v · sigmoid(v)`, FP32.
///
/// Four rounding steps -- the exponential, the add, the divide, the multiply --
/// which is where task 0003's `γ(4)` bound comes from. The bound is relative and
/// holds while the result is a normal FP32 number; in the subnormal range,
/// gradual underflow costs mantissa bits and the contract falls back to an
/// absolute floor of `f32::MIN_POSITIVE`. See task 0003.
pub fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

/// `y[i] = silu(gate[i]) · up[i]`, FP32, unrounded.
pub fn swiglu_row(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::InvalidArtifact {
            detail: format!("gate has {} elements and up has {}", gate.len(), up.len()),
        });
    }
    if gate.is_empty() {
        return Err(Error::InvalidRequest {
            field: "swiglu",
            detail: "an activation over zero features".into(),
        });
    }
    Ok(gate.iter().zip(up).map(|(g, u)| silu(*g) * u).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    /// The equation, transcribed separately in FP64.
    ///
    /// FP64 has enough exponent range that the naive form does not overflow for
    /// any FP32 input, so this stays the textbook expression -- it is the
    /// specification, and writing the implementation's own trick here would make
    /// the comparison circular.
    fn swiglu_row_f64(gate: &[f32], up: &[f32]) -> Vec<f64> {
        gate.iter()
            .zip(up)
            .map(|(g, u)| {
                let g = *g as f64;
                (g / (1.0 + (-g).exp())) * (*u as f64)
            })
            .collect()
    }

    #[test]
    fn swiglu_matches_the_equation_within_gamma_four() {
        let n = 1024usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 64.0).collect();
        let up: Vec<f32> = (0..n)
            .map(|i| ((i * 31 % 71) as f32 - 35.0) / 9.0)
            .collect();

        let got = swiglu_row(&gate, &up).unwrap();
        let want = swiglu_row_f64(&gate, &up);
        let scale: Vec<f64> = want.iter().map(|v| v.abs()).collect();
        let s = ErrorSummary::normalized(&got, &want, &scale);
        let bound = gamma(4);
        assert_eq!(s.count, n);
        assert!(s.within(bound), "{s} exceeded gamma(4) = {bound:.3e}");
    }

    #[test]
    fn silu_does_not_collapse_to_zero_at_a_large_negative_gate() {
        // Fourth review, reproduced: `1/(1 + e^{-v})` overflows for v around -88,
        // and the naive form returned exactly 0 where the true value is a small
        // negative number. With a large `up` the product is comfortably
        // representable, so the error was 100% of a value that mattered.
        let up = 1e30f32;
        let got = swiglu_row(&[-90.0], &[up]).unwrap()[0];
        let want = swiglu_row_f64(&[-90.0], &[up])[0];
        assert!(want < 0.0 && want.is_finite(), "the reference is {want:e}");
        assert_ne!(got, 0.0, "silu(-90) collapsed to zero");
        assert!(got < 0.0, "and it kept its sign: {got:e}");
        let rel = ((got as f64 - want) / want).abs();
        assert!(
            rel < 1e-6,
            "relative error {rel:e}, got {got:e} want {want:e}"
        );
    }

    #[test]
    fn the_sigmoid_is_finite_across_the_whole_fp32_range() {
        // Both directions: the naive positive-branch form overflows for large
        // negative v, and its mirror for large positive v. Neither branch here
        // ever evaluates `exp` at a positive argument.
        for v in [
            -f32::MAX,
            -1e30,
            -200.0,
            -90.0,
            -88.0,
            -1.0,
            0.0,
            1.0,
            88.0,
            90.0,
            200.0,
            1e30,
            f32::MAX,
        ] {
            let s = sigmoid(v);
            assert!(s.is_finite(), "sigmoid({v:e}) = {s}");
            assert!((0.0..=1.0).contains(&s), "sigmoid({v:e}) = {s} left [0,1]");
            assert!(silu(v).is_finite(), "silu({v:e})");
        }
        assert_eq!(sigmoid(0.0), 0.5);
        // Monotonic across the branch boundary at zero.
        assert!(sigmoid(-1e-6) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(1e-6));
    }

    #[test]
    fn silu_matches_the_reference_into_the_subnormal_range() {
        // Where the relative bound stops holding, and what replaces it: gradual
        // underflow costs mantissa bits, so the contract falls back to an
        // absolute floor rather than pretending to relative accuracy.
        for v in [-100.0f32, -103.0, -110.0, -120.0] {
            let got = silu(v) as f64;
            let want = (v as f64) / (1.0 + (-(v as f64)).exp());
            let abs = (got - want).abs();
            assert!(
                abs <= f32::MIN_POSITIVE as f64,
                "silu({v}) = {got:e}, want {want:e}, absolute error {abs:e}"
            );
        }
    }

    #[test]
    fn silu_has_its_defining_values() {
        // silu(0) = 0 exactly; the sigmoid is 1/2 and the numerator is zero.
        assert_eq!(silu(0.0), 0.0);
        // Large positive: silu(v) -> v.
        assert!((silu(20.0) - 20.0).abs() < 1e-5);
        // Large negative: silu(v) -> 0 from below, and stays negative.
        assert!(silu(-20.0) < 0.0 && silu(-20.0) > -1e-6);
        // It is not monotonic: the minimum sits near v = -1.278.
        assert!(silu(-1.278) < silu(-0.5));
        assert!(silu(-1.278) < silu(-3.0));
    }

    #[test]
    fn swiglu_is_not_a_plain_product_and_not_relu_gated() {
        // The substitutions that would pass a careless test. silu is smooth and
        // negative for negative gates, where both a plain product and a ReLU
        // gate differ.
        let gate = [-2.0f32, -0.5, 0.5, 2.0];
        let up = [1.0f32; 4];
        let got = swiglu_row(&gate, &up).unwrap();

        assert!(
            got[0] < 0.0 && got[0] > -0.3,
            "silu(-2) is small and negative"
        );
        assert_ne!(got[0], gate[0], "not a plain product");
        assert_ne!(got[0], 0.0, "not ReLU-gated");
        assert!(
            got[3] > 1.7 && got[3] < 2.0,
            "silu(2) is close to but under 2"
        );
    }

    #[test]
    fn gate_and_up_are_separate_operands() {
        // Swapping them changes the answer, which is why the graph takes two
        // edges rather than one interleaved tensor.
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        assert_ne!(swiglu_row(&a, &b).unwrap(), swiglu_row(&b, &a).unwrap());
    }

    #[test]
    fn nonfinite_and_shape_problems_are_visible() {
        assert!(swiglu_row(&[], &[]).is_err());
        assert!(swiglu_row(&[1.0], &[1.0, 2.0]).is_err());
        // A non-finite input propagates rather than being silently clamped: the
        // graph's job is to not produce one, and hiding it here would move the
        // failure somewhere harder to attribute.
        assert!(swiglu_row(&[f32::NAN], &[1.0]).unwrap()[0].is_nan());
    }
}
