//! Exact dyadic source weights keep the independent task0019 oracle's BF16
//! input contract intact while exercising packed integer loads, maps and tails.
use moxie_graph::ExpertActivation;
use moxie_kernels::{
    affine::AffineWeight,
    cpu_expert::{self, ExpertAssignment, ExpertShape, ExpertTiling},
};
use moxie_types::{GateTransform, Precision};

fn weights(
    outputs: usize,
    inputs: usize,
    bits: usize,
    group: usize,
    mapped: bool,
) -> (Vec<u8>, Vec<u32>, Vec<f32>) {
    let groups = inputs.div_ceil(group);
    let stride = inputs.div_ceil(8 / bits);
    let mut bytes = vec![0; outputs * stride];
    let map: Vec<u32> = (0..inputs)
        .map(|k| {
            if mapped {
                (groups - 1 - k / group) as u32
            } else {
                (k / group) as u32
            }
        })
        .collect();
    let mut values = Vec::new();
    let sign = |o: usize, g: usize| if (o + g).is_multiple_of(2) { 1.0 } else { -1.0 };
    let zero = |o: usize, g: usize| ((o + g) % 3) as i16 - 1;
    for o in 0..outputs {
        for k in 0..inputs {
            let q = ((o * 11 + k * 7) % (1 << bits)) as i32 - (1 << (bits - 1));
            if bits == 4 {
                bytes[o * stride + k / 2] |= (q as u8 & 15) << ((k % 2) * 4);
            } else {
                bytes[o * stride + k] = q as u8;
            }
            values.push(
                (q - i32::from(zero(o, map[k] as usize))) as f32 / 64.0 * sign(o, map[k] as usize),
            );
        }
    }
    for o in 0..outputs {
        for g in 0..groups {
            let bits = if sign(o, g) > 0.0 {
                0x2400u16
            } else {
                0xa400u16
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    } // Signed F16 2^-6.
    for o in 0..outputs {
        for g in 0..groups {
            bytes.extend_from_slice(&zero(o, g).to_le_bytes());
        }
    }
    (bytes, map, values)
}

#[test]
fn packed_experts_match_independent_gate_oracle_with_tails_and_maps() {
    for bits in [4, 8] {
        for group in [32, 128] {
            for mapped in [false, true] {
                let (hidden, intermediate) = (35usize, 67usize);
                let (gu, gm, gv) = weights(2 * intermediate, hidden, bits, group, mapped);
                let (down, dm, dv) = weights(hidden, intermediate, bits, group, mapped);
                let precision = if bits == 4 {
                    Precision::Int4
                } else {
                    Precision::Int8
                };
                let gate_up = AffineWeight::new(
                    &gu,
                    2 * intermediate,
                    hidden,
                    precision,
                    group,
                    Precision::F16,
                    true,
                    mapped.then_some(&gm),
                )
                .unwrap();
                let down = AffineWeight::new(
                    &down,
                    hidden,
                    intermediate,
                    precision,
                    group,
                    Precision::F16,
                    true,
                    mapped.then_some(&dm),
                )
                .unwrap();
                let x: Vec<f32> = (0..3 * hidden)
                    .map(|k| ((k % 13) as f32 - 6.0) / 16.0)
                    .collect();
                let xb: Vec<u8> = x
                    .iter()
                    .flat_map(|&v| cpu_expert::to_bf16_bits(v).to_le_bytes())
                    .collect();
                for (activation, transform) in [
                    (ExpertActivation::GeGlu, GateTransform::GeluTanh),
                    (ExpertActivation::SwiGlu, GateTransform::Silu),
                ] {
                    for tile in [1, 7, 67] {
                        let shape = ExpertShape {
                            hidden: hidden as u32,
                            intermediate: intermediate as u32,
                        };
                        let tiling = ExpertTiling::lanes(tile);
                        let mut workspace = vec![0.; shape.workspace_f32(tiling).unwrap()];
                        let mut output = vec![0; 3 * hidden * 2];
                        cpu_expert::expert_group_affine(
                            &xb,
                            ExpertAssignment {
                                rows: &[2, 0, 1],
                                slots: &[1, 2, 0],
                            },
                            gate_up,
                            down,
                            transform,
                            shape,
                            tiling,
                            &mut workspace,
                            &mut output,
                        )
                        .unwrap();
                        for (row, slot) in [(2, 1), (0, 2), (1, 0)] {
                            let expected = moxie_oracles::route::expert_row(
                                &x[row * hidden..(row + 1) * hidden],
                                &gv,
                                &dv,
                                0,
                                moxie_oracles::route::ExpertSpec {
                                    experts: 1,
                                    hidden,
                                    intermediate,
                                    activation,
                                },
                            )
                            .unwrap();
                            let expected: Vec<u8> = expected
                                .iter()
                                .flat_map(|&v| cpu_expert::to_bf16_bits(v).to_le_bytes())
                                .collect();
                            assert_eq!(
                                &output[slot * hidden * 2..(slot + 1) * hidden * 2],
                                expected,
                                "bits={bits} group={group} mapped={mapped} gate={activation:?} tile={tile}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_packed_operands_and_late_invalid_routes_are_refused() {
    let (gu, gm, _) = weights(6, 35, 4, 32, true);
    let (down, dm, _) = weights(35, 3, 4, 32, true);
    let make_gu = |bytes, map| {
        AffineWeight::new(bytes, 6, 35, Precision::Int4, 32, Precision::F16, true, map)
    };
    assert!(make_gu(&gu[..gu.len() - 1], Some(gm.as_slice())).is_err());
    assert!(make_gu(&gu, Some(&[0; 35])).is_err()); // unused second group
    assert!(make_gu(&gu, Some(&[2; 35])).is_err());
    let gate_up = make_gu(&gu, Some(&gm)).unwrap();
    let down = AffineWeight::new(
        &down,
        35,
        3,
        Precision::Int4,
        32,
        Precision::F16,
        true,
        Some(&dm),
    )
    .unwrap();
    let mut output = vec![0xa5; 140];
    let mut workspace = vec![0.; 9];
    let result = cpu_expert::expert_group_affine(
        &[0; 70],
        ExpertAssignment {
            rows: &[0, 1],
            slots: &[0, 1],
        },
        gate_up,
        down,
        GateTransform::Silu,
        ExpertShape {
            hidden: 35,
            intermediate: 3,
        },
        ExpertTiling::lanes(3),
        &mut workspace,
        &mut output,
    );
    assert!(result.is_err());
    assert_eq!(output, vec![0xa5; 140]);
}
