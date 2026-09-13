//! The host grouped-expert kernel, and the host reduction beside it.
//!
//! Document 03 makes shared CPU expert execution "a planner candidate for
//! low-reuse, disk/PCIe-constrained decode" and constrains how it may be built:
//! "CPU kernels operate on bounded tiles of canonical packed weights; do not
//! materialize the entire model as BF16."
//!
//! Both halves of that sentence are mechanisms here rather than intentions.
//! **Canonical packed weights** means this kernel reads the residency cache's
//! BF16 bytes in place, through a `&[u8]` that is one lease's chunk: it never
//! receives, builds or keeps an FP32 copy of a tensor. **Bounded tiles** means
//! the FP32 working set is `intermediate + 2 * tile` values and is supplied by
//! the caller, so a plan can charge it to `HostTier::CpuWorkspace` before the
//! first byte is touched rather than discovering it afterwards.
//!
//! The numerical contract is not this crate's to choose. Every boundary below
//! is task 0019's, accepted with its FP64 oracle, and the acceptance gate is
//! **bitwise** equality with `moxie_oracles::route::expert_row`: every operation
//! here is FP32 or FP64 in a fixed order, so a tolerance would be hiding
//! something rather than allowing for something.

use moxie_types::{Error, GateTransform, Result};

/// One expert's logical dimensions. The expert axis is absent on purpose: a
/// lease hands this kernel **one** expert's slice, and a kernel that took the
/// fused tensor plus an index would be free to read a neighbouring expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    pub hidden: u32,
    pub intermediate: u32,
}

/// How many gate/up lanes are projected before they are activated.
///
/// The bound document 03 asks for. It changes no result: the tile only decides
/// how many projected lanes exist at once, and every lane is computed by the
/// same fixed reduction whatever the tile is. A test asserts that across several
/// tile sizes, because "the tiling is numerically inert" is exactly the kind of
/// claim that is true until someone reassociates inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertTiling {
    pub lanes: u32,
}

impl ExpertTiling {
    pub const fn lanes(lanes: u32) -> Self {
        Self { lanes }
    }
}

impl ExpertShape {
    /// FP32 values the caller must supply: the activated intermediate, which the
    /// down projection reduces over whole, plus one tile of gate and up lanes.
    pub fn workspace_f32(self, tiling: ExpertTiling) -> Option<usize> {
        let lanes = tiling.lanes.min(self.intermediate);
        (self.intermediate as usize).checked_add(2usize.checked_mul(lanes as usize)?)
    }
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

/// Round to nearest BF16 and back, round-half-to-even.
///
/// Reimplemented rather than depended on: `moxie-format` sits beside this crate
/// in the ownership table, not below it, and this crate's allowlist row is
/// `moxie-types` alone. That is only acceptable while the two are demonstrably
/// the same function, which `agrees_with_the_format_crate_over_every_bf16_pattern`
/// checks over every BF16 value and every midpoint between two of them.
#[inline]
pub fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    if v.is_nan() {
        return v;
    }
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits.wrapping_add(0x7FFF).wrapping_add(lsb)) >> 16) << 16)
}

/// The BF16 bit pattern of `v`. NaN keeps its payload's high half.
#[inline]
pub fn to_bf16_bits(v: f32) -> u16 {
    (bf16_round(v).to_bits() >> 16) as u16
}

#[inline]
fn load_bf16(bytes: &[u8], index: usize) -> f32 {
    let lo = index * 2;
    let bits = u16::from_le_bytes([bytes[lo], bytes[lo + 1]]);
    f32::from_bits((bits as u32) << 16)
}

#[inline]
fn store_bf16(bytes: &mut [u8], index: usize, value: f32) {
    let bits = to_bf16_bits(value).to_le_bytes();
    let lo = index * 2;
    bytes[lo] = bits[0];
    bytes[lo + 1] = bits[1];
}

/// `silu` and `gelu_tanh`, in the exact form task 0019 accepted.
///
/// They differ in more than their curve: GeGLU rounds its gate term to BF16
/// before multiplying because the pinned Gemma 4 source does, and SwiGLU
/// evaluates the whole product in FP64 and rounds once because task 0003's
/// contract says so after a review found an intermediate underflow producing a
/// 100% error. Collapsing them into one function with a flag would erase a
/// declared boundary, which is what document 02 keeps them apart to prevent.
#[inline]
fn gate_times_up(gate: f32, up: f32, transform: GateTransform) -> f32 {
    match transform {
        GateTransform::GeluTanh => {
            let v = gate as f64;
            let g =
                (0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())) as f32;
            (bf16_round(g) as f64 * up as f64) as f32
        }
        GateTransform::Silu => {
            let v = gate as f64;
            let sigmoid = if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            };
            (v * sigmoid * up as f64) as f32
        }
    }
}

/// Every row this launch serves, for one expert.
///
/// `rows[i]` indexes `x`; `slots[i]` indexes `out`. They are separate because
/// the expert-major grouping a grouped kernel wants and the slot-major layout a
/// deterministic reduction needs are different orders, and the scatter between
/// them is this kernel's obligation rather than the caller's.
#[derive(Debug, Clone, Copy)]
pub struct ExpertAssignment<'a> {
    pub rows: &'a [u32],
    pub slots: &'a [u32],
}

/// Compute one expert's contribution for every assigned row.
///
/// `x` is `[any, hidden]` BF16, `gate_up` is one expert's `[2 * intermediate,
/// hidden]` BF16 slice with the whole gate block before the whole up block, and
/// `down` is its `[hidden, intermediate]` BF16 slice. `out` is the slot-major
/// `[any, hidden]` BF16 buffer the reduction later reads.
#[allow(clippy::too_many_arguments)]
pub fn expert_group_bf16(
    x: &[u8],
    assignment: ExpertAssignment<'_>,
    gate_up: &[u8],
    down: &[u8],
    transform: GateTransform,
    shape: ExpertShape,
    tiling: ExpertTiling,
    workspace: &mut [f32],
    out: &mut [u8],
) -> Result<()> {
    let hidden = shape.hidden as usize;
    let intermediate = shape.intermediate as usize;
    if hidden == 0 || intermediate == 0 {
        return Err(invalid(
            "expert_mlp",
            format!("degenerate shape: hidden {hidden}, intermediate {intermediate}"),
        ));
    }
    if tiling.lanes == 0 {
        return Err(invalid(
            "tiling",
            "a tile of zero lanes makes no progress".into(),
        ));
    }
    let lanes = (tiling.lanes as usize).min(intermediate);
    if gate_up.len() != 2 * intermediate * hidden * 2 {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "gate/up slice is {} B, expected {} for [{}, {hidden}] BF16",
                gate_up.len(),
                2 * intermediate * hidden * 2,
                2 * intermediate
            ),
        });
    }
    if down.len() != hidden * intermediate * 2 {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "down slice is {} B, expected {} for [{hidden}, {intermediate}] BF16",
                down.len(),
                hidden * intermediate * 2
            ),
        });
    }
    if assignment.rows.len() != assignment.slots.len() {
        return Err(invalid(
            "assignment",
            format!(
                "{} row(s) against {} slot(s)",
                assignment.rows.len(),
                assignment.slots.len()
            ),
        ));
    }
    let needed = shape
        .workspace_f32(tiling)
        .ok_or_else(|| invalid("workspace", "workspace extent overflows".into()))?;
    if workspace.len() < needed {
        return Err(Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
            requested_bytes: (needed * 4) as u64,
            available_bytes: (workspace.len() * 4) as u64,
        });
    }
    let (activated, lane_tile) = workspace.split_at_mut(intermediate);
    let (gate_tile, up_tile) = lane_tile.split_at_mut(lanes);

    let up_base = intermediate * hidden;
    for (assigned, (&row, &slot)) in assignment
        .rows
        .iter()
        .zip(assignment.slots.iter())
        .enumerate()
    {
        let row = row as usize;
        let slot = slot as usize;
        let x_start = row.checked_mul(hidden).and_then(|v| v.checked_mul(2));
        let out_start = slot.checked_mul(hidden).and_then(|v| v.checked_mul(2));
        match (x_start, out_start) {
            (Some(xs), Some(os)) if xs + hidden * 2 <= x.len() && os + hidden * 2 <= out.len() => {}
            _ => {
                return Err(invalid(
                    "assignment",
                    format!(
                        "entry {assigned} addresses row {row} and slot {slot}, which do not fit \
                         {} B of activations and {} B of slots",
                        x.len(),
                        out.len()
                    ),
                ));
            }
        }
        let x_row = &x[row * hidden * 2..(row + 1) * hidden * 2];

        // The gate/up projection, one bounded tile of lanes at a time. Each
        // lane is a full sequential ascending-k FP32 reduction: the tile bounds
        // how many lanes are live, never how a lane is summed.
        let mut lane0 = 0usize;
        while lane0 < intermediate {
            let count = lanes.min(intermediate - lane0);
            for t in 0..count {
                let lane = lane0 + t;
                let mut gate = 0f32;
                let mut up = 0f32;
                let gate_row = lane * hidden;
                let up_row = up_base + lane * hidden;
                for k in 0..hidden {
                    let xk = load_bf16(x_row, k);
                    gate += xk * load_bf16(gate_up, gate_row + k);
                    up += xk * load_bf16(gate_up, up_row + k);
                }
                gate_tile[t] = bf16_round(gate);
                up_tile[t] = bf16_round(up);
            }
            for t in 0..count {
                activated[lane0 + t] =
                    bf16_round(gate_times_up(gate_tile[t], up_tile[t], transform));
            }
            lane0 += count;
        }

        // The down projection, over the whole activated intermediate. It is not
        // tiled: the reduction is over `intermediate` and splitting it would
        // reassociate the sum the oracle fixes.
        let out_row = &mut out[slot * hidden * 2..(slot + 1) * hidden * 2];
        for o in 0..hidden {
            let mut acc = 0f32;
            let base = o * intermediate;
            for (i, h) in activated.iter().enumerate() {
                acc += *h * load_bf16(down, base + i);
            }
            store_bf16(out_row, o, acc);
        }
    }
    Ok(())
}

/// Reduce every row's `top_k` slots in the order the plan supplies.
///
/// `order[r * top_k + t]` is the slot position, within row `r`, that is added at
/// step `t`. The order is data because floating-point addition is not
/// associative: task 0019 made it a parameter of `Combine` for that reason, and
/// carrying it as an explicit permutation is what lets the executor reduce a
/// plan that mixed a host group and a device group without either candidate's
/// completion order reaching the sum.
///
/// `accumulator` is `hidden` FP32 values, supplied by the caller. It used to be
/// a `vec![0f32; hidden]` allocated here, which put an allocation outside the
/// admitted envelope on the one path every plan takes.
#[allow(clippy::too_many_arguments)]
pub fn combine_rows_bf16(
    slots: &[u8],
    weights: &[f32],
    order: &[u32],
    rows: u32,
    top_k: u32,
    hidden: u32,
    accumulator: &mut [f32],
    out: &mut [u8],
) -> Result<()> {
    let rows = rows as usize;
    let top_k = top_k as usize;
    let hidden = hidden as usize;
    if rows == 0 || top_k == 0 || hidden == 0 {
        return Err(invalid(
            "combine",
            format!("degenerate shape: {rows} rows, top_k {top_k}, hidden {hidden}"),
        ));
    }
    let expected_slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| invalid("combine", "slot count overflows".into()))?;
    if slots.len() != expected_slots * hidden * 2 {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "slot buffer is {} B, expected {} for [{expected_slots}, {hidden}] BF16",
                slots.len(),
                expected_slots * hidden * 2
            ),
        });
    }
    if weights.len() != expected_slots || order.len() != expected_slots {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "{} coefficient(s) and {} order entries for {expected_slots} slot(s)",
                weights.len(),
                order.len()
            ),
        });
    }
    if out.len() != rows * hidden * 2 {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "output is {} B, expected {} for [{rows}, {hidden}] BF16",
                out.len(),
                rows * hidden * 2
            ),
        });
    }
    if accumulator.len() < hidden {
        return Err(Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
            requested_bytes: (hidden * 4) as u64,
            available_bytes: (accumulator.len() * 4) as u64,
        });
    }
    let acc = &mut accumulator[..hidden];
    for r in 0..rows {
        acc.iter_mut().for_each(|v| *v = 0.0);
        let row_order = &order[r * top_k..(r + 1) * top_k];
        let mut seen = 0u64;
        for position in row_order {
            let j = *position as usize;
            if j >= top_k {
                return Err(invalid(
                    "order",
                    format!("row {r} names slot position {j} of {top_k}"),
                ));
            }
            // A permutation, checked. An order that repeated one slot and
            // dropped another would produce a plausible row that no oracle
            // predicts, and the bug would live in the plan rather than here.
            let bit = 1u64 << j;
            if seen & bit != 0 {
                return Err(invalid(
                    "order",
                    format!("row {r} names slot position {j} twice"),
                ));
            }
            seen |= bit;
            let w = weights[r * top_k + j];
            let base = (r * top_k + j) * hidden;
            for (a, o) in acc.iter_mut().enumerate() {
                *o += w * load_bf16(slots, base + a);
            }
        }
        let out_row = &mut out[r * hidden * 2..(r + 1) * hidden * 2];
        for (o, value) in acc.iter().enumerate() {
            store_bf16(out_row, o, *value);
        }
    }
    Ok(())
}

/// Guard against a `top_k` wider than the permutation check can represent.
pub const MAX_TOP_K: u32 = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agrees_with_the_format_crate_over_every_bf16_pattern() {
        for bits in 0u32..=0xFFFF {
            let exact = f32::from_bits(bits << 16);
            for probe in [exact, f32::from_bits((bits << 16) | 0x8000)] {
                if probe.is_nan() {
                    continue;
                }
                let want = moxie_format::bf16::bf16_bits_to_f32(
                    moxie_format::bf16::f32_to_bf16_bits(probe),
                );
                assert_eq!(
                    bf16_round(probe).to_bits(),
                    want.to_bits(),
                    "0x{:08x}",
                    probe.to_bits()
                );
                assert_eq!(
                    to_bf16_bits(probe),
                    moxie_format::bf16::f32_to_bf16_bits(probe),
                    "0x{:08x}",
                    probe.to_bits()
                );
            }
        }
        assert!(bf16_round(f32::NAN).is_nan());
    }

    #[test]
    fn the_workspace_extent_is_the_intermediate_plus_one_tile_of_two_lanes() {
        let shape = ExpertShape {
            hidden: 8,
            intermediate: 6,
        };
        assert_eq!(shape.workspace_f32(ExpertTiling::lanes(2)), Some(6 + 4));
        // A tile wider than the intermediate is clamped rather than charged.
        assert_eq!(shape.workspace_f32(ExpertTiling::lanes(64)), Some(6 + 12));
    }

    #[test]
    fn a_short_workspace_is_refused_as_capacity_rather_than_panicking() {
        let shape = ExpertShape {
            hidden: 4,
            intermediate: 4,
        };
        let mut ws = vec![0f32; 2];
        let err = expert_group_bf16(
            &[0u8; 8],
            ExpertAssignment {
                rows: &[0],
                slots: &[0],
            },
            &[0u8; 4 * 2 * 4 * 2],
            &[0u8; 4 * 4 * 2],
            GateTransform::GeluTanh,
            shape,
            ExpertTiling::lanes(2),
            &mut ws,
            &mut [0u8; 8],
        )
        .unwrap_err();
        assert!(matches!(err, Error::CapacityExceeded { .. }), "{err}");
    }

    #[test]
    fn an_order_that_is_not_a_permutation_is_refused() {
        let mut out = vec![0u8; 4];
        let err = combine_rows_bf16(
            &[0u8; 8],
            &[1.0, 1.0],
            &[0, 0],
            1,
            2,
            2,
            &mut [0f32; 2],
            &mut out,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("twice"), "{err}");
    }
}
