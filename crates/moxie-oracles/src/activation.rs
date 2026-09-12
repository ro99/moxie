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

/// The sigmoid in FP64, same sign-dependent form.
fn sigmoid_f64(v: f64) -> f64 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

/// `y[i] = silu(gate[i]) · up[i]`, evaluated in FP64 and rounded **once**.
///
/// The intermediate is FP64 because an FP32 one cannot represent the operation's
/// own output range. `σ(−104)` is about `1e−45`, at the very bottom of FP32's
/// subnormals, so `silu(−104)` flushes to zero -- but multiplied by an `up` of
/// `1e30` the true result is `−7.1e−14`, comfortably normal. The fifth review
/// found exactly that: an intermediate underflow producing a 100% error in a
/// perfectly ordinary output.
///
/// Evaluating the whole expression before narrowing is also the more faithful
/// reading of task 0003's rounding table, which puts SwiGLU's single rounding on
/// the product. Rounding the intermediate was an extra boundary the contract
/// never declared.
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
    let mut out = crate::try_vec(gate.len())?;
    out.extend(gate.iter().zip(up).map(|(g, u)| {
        let g = *g as f64;
        ((g * sigmoid_f64(g)) * (*u as f64)) as f32
    }));
    Ok(out)
}

/// The tanh approximation of GELU, FP32 in and out.
///
/// ```text
/// gelu_tanh(v) = 0.5 · v · (1 + tanh(sqrt(2/pi) · (v + 0.044715 · v³)))
/// ```
///
/// `sqrt(2/pi)` is the pinned constant `0.7978845608028654`, transcribed from
/// `src/platform/numerics.cpp:86` rather than recomputed, so this reference and
/// the source it stands for cannot drift apart by a rounding of the constant.
///
/// Evaluated in FP64 and narrowed once. The inner polynomial is why: `v³`
/// overflows FP32 at `|v| > 1.1e13` while the true result there is exactly `v`,
/// and for moderate negative `v` the sum `v + 0.044715·v³` cancels, so an FP32
/// intermediate loses the bits that decide the tanh argument.
pub fn gelu_tanh(v: f32) -> f32 {
    let v = v as f64;
    (0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())) as f32
}

/// `y[i] = bf16(gelu_tanh(gate[i])) · up[i]`.
///
/// **The BF16 rounding of the gate term is part of the equation, not storage.**
/// The pinned source rounds there (`src/models/gemma4/gemma4_ops.cpp:70`), and
/// dropping the boundary would make this reference disagree with the released
/// model by more than the product's own rounding. The second boundary -- on the
/// product -- is the ordinary node-output one and belongs to the interpreter,
/// which is why it is absent here.
///
/// Contrast [`swiglu_row`], whose contract declares a single rounding on the
/// product. The two activations differ in their gate transform *and* in their
/// declared boundaries, which is precisely why document 02 keeps them separate.
pub fn geglu_row(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::InvalidArtifact {
            detail: format!("gate has {} elements and up has {}", gate.len(), up.len()),
        });
    }
    if gate.is_empty() {
        return Err(Error::InvalidRequest {
            field: "geglu",
            detail: "an activation over zero features".into(),
        });
    }
    let mut out = crate::try_vec(gate.len())?;
    out.extend(gate.iter().zip(up).map(|(g, u)| {
        let gated = crate::bf16_round(gelu_tanh(*g)) as f64;
        (gated * (*u as f64)) as f32
    }));
    Ok(out)
}

/// Logit soft capping, `bf16(bf16(tanh(bf16(bf16(x) / cap))) · cap)`.
///
/// **All four BF16 boundaries belong to this operation**, including the one on
/// the result. The pinned kernel's last line is
/// `values[index] = bf16_round(value * softcap)`
/// (`kernels/cuda/detail/backend_kernels.cuh:905`), and unlike every other
/// operation in this crate the caller does not supply that rounding: logits
/// are stored FP32 so the sampler sees the pre-truncation distribution, so a
/// boundary omitted here is a boundary lost.
///
/// The first version left the final multiply unrounded, on the reasoning that
/// the caller rounds node outputs. It does not round this one. For `x = 12.5`
/// and `cap = 30` that returned 11.8359375 where the source gives 11.8125 —
/// a difference in the distribution the sampler draws from. FP32 storage can
/// hold a BF16-rounded value; being able to represent more precision is not a
/// reason to produce it.
///
/// `cap` must be finite and positive. There is no cap value meaning "uncapped";
/// absence is modelled by not calling this.
pub fn softcap(x: f32, cap: f32) -> Result<f32> {
    if !(cap.is_finite() && cap > 0.0) {
        return Err(Error::InvalidRequest {
            field: "softcap",
            detail: format!("cap must be finite and positive, got {cap}"),
        });
    }
    let scaled = crate::bf16_round(crate::bf16_round(x) / cap);
    let bent = crate::bf16_round((scaled as f64).tanh() as f32);
    Ok(crate::bf16_round(bent * cap))
}

/// [`softcap`] over a row.
pub fn softcap_row(x: &[f32], cap: f32) -> Result<Vec<f32>> {
    let mut out = crate::try_vec(x.len())?;
    for v in x {
        out.push(softcap(*v, cap)?);
    }
    Ok(out)
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
    fn an_intermediate_underflow_does_not_zero_a_normal_output() {
        // Fifth review, reproduced: gate -104 makes the FP32 sigmoid flush to
        // zero, but the product with a large `up` is a perfectly ordinary
        // number. Neither the relative bound nor the subnormal floor covered a
        // 100% error on a normal-sized result.
        let up = crate::bf16_round(1e30);
        for gate in [-104.0f32, -110.0, -120.0, -150.0, -200.0] {
            let got = swiglu_row(&[gate], &[up]).unwrap()[0];
            let want = swiglu_row_f64(&[gate], &[up])[0];
            if want.abs() >= f32::MIN_POSITIVE as f64 {
                assert_ne!(got, 0.0, "gate {gate}: output {want:e} collapsed to zero");
                let rel = ((got as f64 - want) / want).abs();
                assert!(rel < 1e-6, "gate {gate}: relative error {rel:e}");
            } else {
                // Genuinely subnormal or zero output: the absolute floor applies.
                assert!((got as f64 - want).abs() <= f32::MIN_POSITIVE as f64);
            }
        }
    }

    #[test]
    fn silu_does_not_collapse_to_zero_at_a_large_negative_gate() {
        // Fourth review, reproduced: `1/(1 + e^{-v})` overflows for v around -88,
        // and the naive form returned exactly 0 where the true value is a small
        // negative number. With a large `up` the product is comfortably
        // representable, so the error was 100% of a value that mattered.
        let up = crate::bf16_round(1e30);
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

    /// The GELU tanh approximation, transcribed separately in FP64 from the
    /// pinned `src/platform/numerics.cpp:86`.
    fn gelu_tanh_f64(v: f64) -> f64 {
        0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v.powi(3))).tanh())
    }

    #[test]
    fn gelu_tanh_matches_the_equation_within_gamma_four() {
        let n = 1024usize;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 48.0).collect();
        let got: Vec<f32> = xs.iter().map(|v| gelu_tanh(*v)).collect();
        let want: Vec<f64> = xs.iter().map(|v| gelu_tanh_f64(*v as f64)).collect();
        let scale: Vec<f64> = want.iter().map(|v| v.abs()).collect();
        let s = ErrorSummary::normalized(&got, &want, &scale);
        let bound = gamma(4);
        assert_eq!(s.count, n);
        assert!(s.within(bound), "{s} exceeded gamma(4) = {bound:.3e}");
    }

    #[test]
    fn gelu_tanh_survives_the_cube_that_overflows_fp32() {
        // `v³` leaves FP32 above about 1.1e13 while the true GELU there is
        // exactly `v`. An FP32 intermediate returns NaN; the FP64 one does not.
        for v in [1e13f32, 1e20, 1e30, f32::MAX] {
            let got = gelu_tanh(v);
            assert!(got.is_finite(), "gelu_tanh({v:e}) = {got}");
            assert_eq!(got, v, "the saturated branch must be the identity");
            let neg = gelu_tanh(-v);
            assert!(neg.is_finite() && neg == 0.0, "gelu_tanh({:e}) = {neg}", -v);
        }
    }

    #[test]
    fn gelu_tanh_cancels_where_fp32_would_lose_the_argument() {
        // Around v = -4.3 the polynomial `v + 0.044715 v³` is a difference of
        // similar magnitudes. FP64 keeps the bits that decide the tanh.
        for v in [-4.2f32, -4.3, -4.35, -4.4] {
            let got = gelu_tanh(v) as f64;
            let want = gelu_tanh_f64(v as f64);
            let rel = ((got - want) / want).abs();
            assert!(rel < 1e-6, "v={v}: relative error {rel:e}");
        }
    }

    #[test]
    fn geglu_rounds_its_gate_term_and_swiglu_does_not() {
        // The declared difference, asserted rather than described: GeGLU's
        // gate passes through BF16 before the multiply. Deleting that rounding
        // changes the result, which is what makes it part of the equation.
        let gate = [0.31f32, -1.7, 2.9, 0.004];
        let up = [1.0f32, 1.0, 1.0, 1.0];
        let got = geglu_row(&gate, &up).unwrap();
        for (i, g) in gate.iter().enumerate() {
            assert_eq!(got[i], crate::bf16_round(gelu_tanh(*g)));
            // And the unrounded value differs, so the boundary is observable.
            if gelu_tanh(*g) != 0.0 {
                assert_ne!(
                    got[i],
                    gelu_tanh(*g),
                    "element {i} happens to be BF16-exact; pick another probe"
                );
            }
        }
    }

    #[test]
    fn geglu_is_not_swiglu() {
        // R06 in one assertion: substituting one gated activation for another
        // is a numerical change, not a naming preference.
        let gate: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let up: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0).collect();
        let g = geglu_row(&gate, &up).unwrap();
        let s = swiglu_row(&gate, &up).unwrap();
        assert_ne!(g, s);
        // They agree at zero, where both gates vanish, so the difference above
        // is not an artefact of comparing unrelated scales.
        assert_eq!(geglu_row(&[0.0], &[5.0]).unwrap(), vec![0.0]);
        assert_eq!(swiglu_row(&[0.0], &[5.0]).unwrap(), vec![0.0]);
    }

    #[test]
    fn geglu_shape_disagreements_are_typed_errors() {
        assert!(geglu_row(&[1.0], &[1.0, 2.0]).is_err());
        assert!(geglu_row(&[], &[]).is_err());
    }

    #[test]
    fn the_softcap_bends_large_logits_and_barely_moves_small_ones() {
        let cap = 30.0f32;
        // Far past the cap, tanh saturates and the output approaches it.
        for x in [300.0f32, 1e4, 1e30] {
            let got = softcap(x, cap).unwrap();
            assert!(
                (got - cap).abs() <= 0.25,
                "softcap({x:e}) = {got}, expected about {cap}"
            );
        }
        assert_eq!(softcap(-1e30, cap).unwrap(), -cap);
        // Near zero it is close to the identity: tanh(x/c)*c ~= x.
        for x in [0.5f32, -0.25, 1.0] {
            let got = softcap(x, cap).unwrap();
            assert!((got - x).abs() < 0.05, "softcap({x}) = {got}");
        }
        // And it is odd.
        assert_eq!(softcap(7.0, cap).unwrap(), -softcap(-7.0, cap).unwrap());
    }

    #[test]
    fn the_softcap_is_the_pinned_four_step_sequence() {
        // `kernels/cuda/detail/backend_kernels.cuh:905`, transcribed here from
        // the source rather than from the implementation -- the earlier version
        // of this fixture repeated the implementation's own missing final
        // rounding and therefore agreed with the bug.
        let cap = 30.0f32;
        for x in [12.5f32, -3.75, 41.0, 0.001, 7.125, -19.5] {
            let a = crate::bf16_round(x);
            let b = crate::bf16_round(a / cap);
            let c = crate::bf16_round((b as f64).tanh() as f32);
            let want = crate::bf16_round(c * cap);
            assert_eq!(softcap(x, cap).unwrap(), want, "x={x}");
        }
        let naive = cap * (12.5f32 / cap).tanh();
        assert_ne!(softcap(12.5, cap).unwrap(), naive);
    }

    #[test]
    fn every_softcap_result_is_a_bf16_value() {
        // Independent review finding, reproduced: the final multiply was left
        // unrounded, so results carried FP32 precision the source does not
        // produce. `x = 12.5, cap = 30` is the exact probe.
        assert_eq!(softcap(12.5, 30.0).unwrap(), 11.8125);
        assert_ne!(softcap(12.5, 30.0).unwrap(), 11.8359375);
        for x in [0.5f32, -2.25, 12.5, 41.0, 1e4, -1e30] {
            for cap in [4.0f32, 30.0, 0.5] {
                let got = softcap(x, cap).unwrap();
                assert_eq!(
                    got,
                    crate::bf16_round(got),
                    "softcap({x}, {cap}) = {got} is not a BF16 value"
                );
            }
        }
    }

    #[test]
    fn a_softcap_must_be_finite_and_positive() {
        for cap in [0.0f32, -1.0, f32::INFINITY, f32::NAN] {
            assert!(softcap(1.0, cap).is_err(), "cap {cap} was accepted");
        }
        assert!(softcap_row(&[1.0, 2.0], 4.0).is_ok());
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
    fn the_operation_is_bounded_by_its_output_not_its_intermediate() {
        // The revised statement: the relative bound is about the *output*. An
        // intermediate that underflows is not an excuse, because the operation's
        // declared rounding boundary is the product.
        let n = 512usize;
        let gate: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.4).collect();
        let up: Vec<f32> = (0..n)
            .map(|i| crate::bf16_round(1e20 * (1.0 + i as f32)))
            .collect();
        let got = swiglu_row(&gate, &up).unwrap();
        let want = swiglu_row_f64(&gate, &up);
        let mut checked = 0;
        for i in 0..n {
            let e = (got[i] as f64 - want[i]).abs();
            if want[i].abs() >= f32::MIN_POSITIVE as f64 {
                assert!(
                    e <= gamma(2) * want[i].abs(),
                    "index {i}: gate {} gives {:e}, want {:e}, error {e:e}",
                    gate[i],
                    got[i],
                    want[i]
                );
                checked += 1;
            } else {
                assert!(e <= f32::MIN_POSITIVE as f64, "index {i}: {e:e}");
            }
        }
        assert!(
            checked > 100,
            "only {checked} normal outputs; fixture is too narrow"
        );
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
