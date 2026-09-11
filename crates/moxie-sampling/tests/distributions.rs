use moxie_sampling::{Distribution, Domain, counter, distribution, philox, target_uniform};

fn probabilities(logits: &[f32], mask: Option<&[bool]>, t: f64) -> Vec<f64> {
    let mut bytes = vec![0; logits.len() * 8];
    distribution(logits, mask, t, &mut bytes)
        .unwrap()
        .probabilities()
        .collect()
}

#[test]
fn exhaustive_analytic_distributions_and_masks() {
    for v in 1..=5u32 {
        for mut code in 0..3usize.pow(v) {
            let logits: Vec<f32> = (0..v)
                .map(|_| {
                    let l = [-2., 0., 2.][code % 3];
                    code /= 3;
                    l
                })
                .collect();
            for bits in 0..(1 << v) {
                let mask: Vec<bool> = (0..v).map(|i| bits & (1 << i) != 0).collect();
                for t in [0., 0.25, 1., 2., 10.] {
                    let mut bytes = vec![0; v as usize * 8];
                    let result = distribution(&logits, Some(&mask), t, &mut bytes);
                    if bits == 0 {
                        assert_eq!(result.unwrap_err().kind(), "numerical");
                        continue;
                    }
                    let got: Vec<_> = result.unwrap().probabilities().collect();
                    // Unshifted analytic exponential is an independent equation
                    // on this finite exhaustive domain, avoiding production's max path.
                    let weights: Vec<f64> = logits
                        .iter()
                        .zip(&mask)
                        .map(|(&l, &m)| {
                            if m {
                                (l as f64 / if t == 0. { 1. } else { t }).exp()
                            } else {
                                0.
                            }
                        })
                        .collect();
                    if t == 0. {
                        let best = (0..v as usize)
                            .filter(|&i| mask[i])
                            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
                            .unwrap();
                        for (i, &p) in got.iter().enumerate() {
                            assert_eq!(p, if i == best { 1. } else { 0. });
                        }
                    } else {
                        let total: f64 = weights.iter().sum();
                        for (p, w) in got.iter().zip(weights) {
                            assert!((p - w / total).abs() <= 1e-12);
                        }
                    }
                    assert!((got.iter().sum::<f64>() - 1.).abs() <= 1e-12);
                }
            }
        }
    }
}

#[test]
fn extrema_invalid_inputs_and_existing_oracle() {
    assert_eq!(
        probabilities(&[-f32::MAX, f32::MAX], None, 10.),
        vec![0., 1.]
    );
    for t in [f64::from_bits(1), f32::from_bits(1) as f64, 0.25, 1., 10.] {
        assert_eq!(
            probabilities(&[f32::NEG_INFINITY, 9.], None, t),
            vec![0., 1.]
        );
    }
    for l in [f32::NAN, f32::INFINITY] {
        assert_eq!(
            distribution(&[l, 0.], Some(&[false, true]), 1., &mut [0; 16])
                .unwrap_err()
                .kind(),
            "numerical"
        );
    }
    assert_eq!(
        distribution(&[f32::NEG_INFINITY], None, 0., &mut [0; 8])
            .unwrap_err()
            .kind(),
        "numerical"
    );
    for t in [-1., 10.1, f64::NAN, f64::INFINITY] {
        assert_eq!(
            distribution(&[0.], None, t, &mut [0; 8])
                .unwrap_err()
                .kind(),
            "invalid_request"
        );
    }
    assert!(distribution(&[], None, 1., &mut []).is_err());
    assert!(distribution(&[0.], Some(&[]), 1., &mut [0; 8]).is_err());
    assert!(distribution(&[0.], None, 1., &mut []).is_err());
    for v in [3, 7] {
        let logits: Vec<_> = (0..v).map(|i| i as f32 * 0.25 - 1.).collect();
        for t in [0., 0.5, 1., 2., 10.] {
            let expected = moxie_oracles::sampler::Pipeline::temperature(t as f32)
                .distribution(&logits)
                .unwrap();
            for (p, q) in probabilities(&logits, None, t).iter().zip(expected.probs) {
                assert!((p - q as f64).abs() <= 1e-6);
            }
        }
    }
}

#[test]
fn wide_vocabulary_normalization_does_not_refuse_valid_logits() {
    for v in [32_768, 100_000, 131_072] {
        let logits = vec![0.; v];
        let p = probabilities(&logits, None, 1.);
        let expected = 1. / v as f64;
        assert!(p.iter().all(|&x| (x - expected).abs() <= 1e-12));
        // Independent blockwise reduction avoids replicating production's sum.
        let sum: f64 = p.chunks(32).map(|block| block.iter().sum::<f64>()).sum();
        assert!((sum - 1.).abs() <= 1e-12);
    }
}

#[test]
fn philox_known_answers_and_counter_assignment() {
    assert_eq!(
        philox(
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0]
        ),
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
    );
    // Published Random123 tests/kat_vectors, pinned at
    // 9545ff6413f258be2f04c1d319d99aaef7521150. These were not generated by Moxie.
    assert_eq!(
        philox([0; 4], [0; 2]),
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    );
    assert_eq!(
        philox([u32::MAX; 4], [u32::MAX; 2]),
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
    );
    assert_eq!(
        counter(0x123456789abcdef0, Domain::Target),
        [0x9abcdef0, 0x12345678, 0, 0]
    );
    for (i, domain) in [
        Domain::Target,
        Domain::Proposal,
        Domain::Acceptance,
        Domain::Filtering,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(counter(u64::MAX, domain), [u32::MAX, u32::MAX, i as u32, 0]);
    }
    let want = (0x6627e8d5e169c58du64 >> 11) as f64 / 9007199254740992.;
    assert_eq!(target_uniform(0, 0), want);
}

fn encoded(p: &[f64]) -> Vec<u8> {
    p.iter().flat_map(|p| p.to_le_bytes()).collect()
}

#[test]
fn cdf_intervals_endpoints_and_zero_mass() {
    let greedy_bytes = encoded(&[0.25, 0.375, 0.375]);
    let greedy = Distribution::from_bytes(&greedy_bytes).unwrap();
    assert_eq!(greedy.draw(7, 13, true).unwrap(), 1);
    assert_eq!(greedy.draw(99, 34, true).unwrap(), 1);
    let bytes = encoded(&[0., 0.5, 0., 0.25, 0.25, 0.]);
    let d = Distribution::from_bytes(&bytes).unwrap();
    for (u, want) in [
        (0., 1),
        (0.25, 1),
        (0.5, 3),
        (0.75, 4),
        (f64::from_bits(1.0f64.to_bits() - 1), 4),
    ] {
        assert_eq!(d.draw_uniform(u).unwrap(), want);
    }
    assert!(d.draw_uniform(1.).is_err());
    assert!(d.draw_uniform(f64::NAN).is_err());
    assert!(d.draw(0, u64::MAX, false).is_err());
    // Exhaustive dyadic grid: every interval receives exactly its analytic mass.
    let mut counts = [0; 6];
    for i in 0..1024 {
        counts[d.draw_uniform(i as f64 / 1024.).unwrap() as usize] += 1;
    }
    assert_eq!(counts, [0, 512, 0, 256, 256, 0]);
    for p in [&[0., 0.][..], &[f64::NAN, 1.], &[-1., 2.], &[0.1, 0.1]] {
        assert!(Distribution::from_bytes(&encoded(p)).is_err());
    }
}

#[test]
fn seeded_statistical_bins_under_predeclared_bound() {
    let epsilon = ((2. * 13. / 1e-6f64).ln() / (2. * 100000.)).sqrt();
    for p in [vec![1. / 3.; 3], vec![1. / 7.; 7], vec![0.5, 0.25, 0.25]] {
        let bytes = encoded(&p);
        let d = Distribution::from_bytes(&bytes).unwrap();
        let mut counts = vec![0u64; p.len()];
        for step in 0..100000 {
            counts[d.draw(33377335, step, false).unwrap() as usize] += 1;
        }
        eprintln!(
            "task0014 bins p={p:?} counts={counts:?} n=100000 epsilon={epsilon} seed=33377335"
        );
        for (&n, &want) in counts.iter().zip(&p) {
            assert!((n as f64 / 100000. - want).abs() <= epsilon);
        }
    }
}
