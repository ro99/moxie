//! Task 0021's first declared gate: the host expert kernel is **bitwise** equal
//! to task 0019's accepted oracle.
//!
//! Not "within a tolerance". Every operation in `cpu_expert` is FP32 or FP64 in
//! a fixed order, and so is every operation in `moxie_oracles::route::expert_row`,
//! so any difference at all is a difference in the equation. A tolerance here
//! would hide a transposed weight, a swapped gate and up block, or a
//! reassociated reduction -- the three mistakes this layout invites.

use moxie_graph::ExpertActivation;
use moxie_kernels::cpu_expert::{
    ExpertAssignment, ExpertShape, ExpertTiling, bf16_round, combine_rows_bf16, expert_group_bf16,
    to_bf16_bits,
};
use moxie_oracles::route;
use moxie_types::GateTransform;

/// A deterministic BF16-representable stream. Fixed seeds so a failure is
/// reproducible from the test name alone.
struct Values(u64);

impl Values {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
        bf16_round(unit * 2.0)
    }

    fn block(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next()).collect()
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16_bits(*v).to_le_bytes());
    }
    out
}

fn gate_of(activation: ExpertActivation) -> GateTransform {
    match activation {
        ExpertActivation::GeGlu => GateTransform::GeluTanh,
        ExpertActivation::SwiGlu => GateTransform::Silu,
    }
}

struct Fixture {
    shape: ExpertShape,
    experts: usize,
    x: Vec<f32>,
    rows: usize,
    gate_up: Vec<f32>,
    down: Vec<f32>,
}

impl Fixture {
    fn new(seed: u64, hidden: u32, intermediate: u32, experts: usize, rows: usize) -> Self {
        let mut v = Values::new(seed);
        let shape = ExpertShape {
            hidden,
            intermediate,
        };
        let x = v.block(rows * hidden as usize);
        let gate_up = v.block(experts * 2 * intermediate as usize * hidden as usize);
        let down = v.block(experts * hidden as usize * intermediate as usize);
        Self {
            shape,
            experts,
            x,
            rows,
            gate_up,
            down,
        }
    }

    fn expert_slice_bytes(&self, expert: usize) -> (Vec<u8>, Vec<u8>) {
        let gu = 2 * self.shape.intermediate as usize * self.shape.hidden as usize;
        let d = self.shape.hidden as usize * self.shape.intermediate as usize;
        (
            to_bytes(&self.gate_up[expert * gu..(expert + 1) * gu]),
            to_bytes(&self.down[expert * d..(expert + 1) * d]),
        )
    }

    fn oracle_slot(&self, row: usize, expert: u32, activation: ExpertActivation) -> Vec<u16> {
        let hidden = self.shape.hidden as usize;
        let y = route::expert_row(
            &self.x[row * hidden..(row + 1) * hidden],
            &self.gate_up,
            &self.down,
            expert,
            route::ExpertSpec {
                experts: self.experts,
                hidden,
                intermediate: self.shape.intermediate as usize,
                activation,
            },
        )
        .expect("oracle");
        y.iter().map(|v| to_bf16_bits(*v)).collect()
    }
}

fn run_group(
    fixture: &Fixture,
    expert: u32,
    activation: ExpertActivation,
    rows: &[u32],
    slots: &[u32],
    tiling: ExpertTiling,
    slot_count: usize,
) -> Vec<u8> {
    let (gate_up, down) = fixture.expert_slice_bytes(expert as usize);
    let mut workspace = vec![0f32; fixture.shape.workspace_f32(tiling).expect("extent")];
    let mut out = vec![0u8; slot_count * fixture.shape.hidden as usize * 2];
    expert_group_bf16(
        &to_bytes(&fixture.x),
        ExpertAssignment { rows, slots },
        &gate_up,
        &down,
        gate_of(activation),
        fixture.shape,
        tiling,
        &mut workspace,
        &mut out,
    )
    .expect("group");
    out
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn every_slot_is_bit_identical_to_the_oracle_for_both_gate_transforms() {
    for activation in [ExpertActivation::GeGlu, ExpertActivation::SwiGlu] {
        let fixture = Fixture::new(0x51ded00d, 24, 10, 4, 5);
        let hidden = fixture.shape.hidden as usize;
        let mut compared = 0usize;
        for expert in 0..fixture.experts as u32 {
            let rows: Vec<u32> = (0..fixture.rows as u32).collect();
            let slots: Vec<u32> = (0..fixture.rows as u32).collect();
            let got = run_group(
                &fixture,
                expert,
                activation,
                &rows,
                &slots,
                ExpertTiling::lanes(3),
                fixture.rows,
            );
            let got = as_u16(&got);
            for row in 0..fixture.rows {
                let want = fixture.oracle_slot(row, expert, activation);
                assert_eq!(
                    &got[row * hidden..(row + 1) * hidden],
                    want.as_slice(),
                    "{activation:?} expert {expert} row {row}"
                );
                compared += hidden;
            }
        }
        // The count is printed because "it matches the oracle" is a claim whose
        // size matters: four experts by five rows by 24 components.
        assert_eq!(compared, 4 * 5 * 24);
        println!("{activation:?}: {compared} BF16 components bit-identical to the oracle");
    }
}

#[test]
fn the_tile_width_changes_nothing_at_all() {
    let fixture = Fixture::new(0xfade1, 20, 12, 2, 3);
    let rows: Vec<u32> = (0..3).collect();
    let slots: Vec<u32> = (0..3).collect();
    let reference = run_group(
        &fixture,
        1,
        ExpertActivation::GeGlu,
        &rows,
        &slots,
        ExpertTiling::lanes(1),
        3,
    );
    for lanes in [2u32, 3, 5, 12, 64] {
        let got = run_group(
            &fixture,
            1,
            ExpertActivation::GeGlu,
            &rows,
            &slots,
            ExpertTiling::lanes(lanes),
            3,
        );
        assert_eq!(got, reference, "tile of {lanes} lanes changed the result");
    }
}

#[test]
fn a_group_scatters_into_the_slots_it_was_given_and_touches_no_other() {
    let fixture = Fixture::new(0x5ca77e2, 16, 8, 3, 4);
    let hidden = fixture.shape.hidden as usize;
    // Rows 3 and 1, in that order, into slot positions 5 and 2 of an eight-slot
    // buffer. Expert-major order and slot-major order are different orders, and
    // this is the scatter between them.
    let out = run_group(
        &fixture,
        2,
        ExpertActivation::SwiGlu,
        &[3, 1],
        &[5, 2],
        ExpertTiling::lanes(4),
        8,
    );
    let got = as_u16(&out);
    assert_eq!(
        &got[5 * hidden..6 * hidden],
        fixture
            .oracle_slot(3, 2, ExpertActivation::SwiGlu)
            .as_slice()
    );
    assert_eq!(
        &got[2 * hidden..3 * hidden],
        fixture
            .oracle_slot(1, 2, ExpertActivation::SwiGlu)
            .as_slice()
    );
    for slot in [0usize, 1, 3, 4, 6, 7] {
        assert!(
            got[slot * hidden..(slot + 1) * hidden]
                .iter()
                .all(|v| *v == 0),
            "slot {slot} was written by a group that was not assigned it"
        );
    }
}

#[test]
fn an_out_of_range_row_or_slot_is_refused_rather_than_read() {
    let fixture = Fixture::new(0xb0117, 8, 4, 1, 2);
    let (gate_up, down) = fixture.expert_slice_bytes(0);
    let mut workspace = vec![0f32; fixture.shape.workspace_f32(ExpertTiling::lanes(2)).unwrap()];
    let mut out = vec![0u8; 2 * 8 * 2];
    let err = expert_group_bf16(
        &to_bytes(&fixture.x),
        ExpertAssignment {
            rows: &[7],
            slots: &[0],
        },
        &gate_up,
        &down,
        GateTransform::GeluTanh,
        fixture.shape,
        ExpertTiling::lanes(2),
        &mut workspace,
        &mut out,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("do not fit"), "{err}");
}

#[test]
fn the_reduction_matches_the_oracle_in_whichever_order_it_is_given() {
    let mut v = Values::new(0xc0b81);
    let (rows, top_k, hidden) = (3usize, 4usize, 6usize);
    let slot_values = v.block(rows * top_k * hidden);
    let weights: Vec<f32> = (0..rows * top_k).map(|_| v.next()).collect();
    let experts: Vec<u32> = vec![9, 2, 7, 4, 3, 8, 1, 6, 5, 0, 11, 10];
    let slots_bytes = to_bytes(&slot_values);

    for order in [
        moxie_graph::CombineOrder::AscendingExpertId,
        moxie_graph::CombineOrder::SelectionOrder,
    ] {
        let mut permutation = Vec::new();
        for r in 0..rows {
            let row_experts = &experts[r * top_k..(r + 1) * top_k];
            permutation.extend(
                route::combine_order(row_experts, order)
                    .expect("order")
                    .into_iter()
                    .map(|j| j as u32),
            );
        }
        let mut out = vec![0u8; rows * hidden * 2];
        combine_rows_bf16(
            &slots_bytes,
            &weights,
            &permutation,
            rows as u32,
            top_k as u32,
            hidden as u32,
            &mut vec![0f32; hidden],
            &mut out,
        )
        .expect("combine");
        let got = as_u16(&out);
        for r in 0..rows {
            let want = route::combine_row(
                &experts[r * top_k..(r + 1) * top_k],
                &weights[r * top_k..(r + 1) * top_k],
                &slot_values[r * top_k * hidden..(r + 1) * top_k * hidden],
                hidden,
                order,
            )
            .expect("oracle");
            let want: Vec<u16> = want.iter().map(|v| to_bf16_bits(*v)).collect();
            assert_eq!(
                &got[r * hidden..(r + 1) * hidden],
                want.as_slice(),
                "{order:?} row {r}"
            );
        }
    }
}
