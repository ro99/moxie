//! Residual combination.
//!
//! Trivial arithmetic with a non-trivial rule attached: document 04 requires
//! that "bias and residual terms are applied exactly once, not on every partial
//! output before an unintended sum". That is a partitioning rule rather than an
//! equation, and it is why this is its own operation with its own
//! `PartitionRule::Replicated` rather than an anonymous `+` inside another node.

use moxie_types::{Error, Result};

/// `y[i] = a[i] + b[i]`, FP32, unrounded.
pub fn residual_row(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(Error::InvalidArtifact {
            detail: format!("operands have {} and {} elements", a.len(), b.len()),
        });
    }
    if a.is_empty() {
        return Err(Error::InvalidRequest {
            field: "residual",
            detail: "a residual over zero features".into(),
        });
    }
    let mut out = crate::try_vec(a.len())?;
    out.extend(a.iter().zip(b).map(|(x, y)| x + y));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    #[test]
    fn a_residual_is_exact_when_the_sum_is_representable() {
        let a = [1.0f32, 0.5, -2.0, 0.0];
        let b = [2.0f32, 0.25, 2.0, -0.0];
        assert_eq!(residual_row(&a, &b).unwrap(), vec![3.0, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn a_residual_is_one_rounding_otherwise() {
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) / 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) / 7.0 + 1e-8).collect();
        let got = residual_row(&a, &b).unwrap();
        let want: Vec<f64> = (0..n).map(|i| a[i] as f64 + b[i] as f64).collect();
        let scale: Vec<f64> = (0..n)
            .map(|i| (a[i] as f64).abs() + (b[i] as f64).abs())
            .collect();
        let s = ErrorSummary::normalized(&got, &want, &scale);
        assert!(s.within(gamma(1)), "{s} exceeded gamma(1)");
    }

    #[test]
    fn adding_a_residual_twice_is_not_the_same_as_adding_it_once() {
        // Document 04's rule as an assertion. A partitioned linear that applied
        // the residual on each rank's partial output before the reduction would
        // produce this.
        let x = [1.0f32, 2.0];
        let r = [10.0f32, 20.0];
        let once = residual_row(&x, &r).unwrap();
        let twice = residual_row(&once, &r).unwrap();
        assert_ne!(once, twice);
    }

    #[test]
    fn shape_disagreements_are_typed_errors() {
        assert!(residual_row(&[1.0], &[1.0, 2.0]).is_err());
        assert!(residual_row(&[], &[]).is_err());
    }
}
