//! GPTQ-v1 integer serialization, shared by externally quantized sources.
//!
//! Pinned oracle: AutoRound 0.15.0, commit
//! 7269a3cd89afbb516d5a25a254ab1faea9bb1aec, `qlinear_torch_zp.py`.
//! This is specifically the zero-minus-one serialization selected by
//! `get_autogptq_packing_qlinear`, not the direct-zero torch serialization.
//! No quantization, I/O or execution occurs here. Input columns stay in logical
//! order; an optional `g_idx` changes their scale group, not their code position.

use std::ops::Range;

use moxie_types::Result;

use crate::affine::{AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints};
use crate::payload;
use crate::scale::ScaleDtype;

/// The serialization declaration; it must come from source provenance/config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptqSpec {
    pub width: IntWidth,
    pub group_size: usize,
    /// A declaration of activation ordering cannot silently lose its map.
    pub requires_group_index: bool,
}

/// Headers and optional I32 map payload. Integer tensor dtypes are checked by
/// the caller's header resolver; these byte APIs interpret little-endian I32.
#[derive(Debug, Clone, Copy)]
pub struct DeclaredSource<'a> {
    /// Logical `(output channels, input channels)`, including unpadded tails.
    pub logical: (usize, usize),
    pub qweight_shape: &'a [u64],
    pub qzeros_shape: &'a [u64],
    pub scales_shape: &'a [u64],
    pub scale_dtype: ScaleDtype,
    pub group_index: Option<&'a [u8]>,
    pub group_index_shape: Option<&'a [u64]>,
}

#[derive(Debug, Clone, Copy)]
pub struct SourceTensors<'a> {
    pub declaration: DeclaredSource<'a>,
    pub qweight: &'a [u8],
    pub qzeros: &'a [u8],
    pub scales: &'a [u8],
}

/// Validated geometry, with an owned map if the source carries one. Construction
/// allocates only that map and its validation scratch, never weight payloads.
#[derive(Debug)]
pub struct GptqPlan {
    descriptor: AffineDescriptor,
    groups: usize,
    pack: usize,
    input_words: usize,
    output_words: usize,
    lengths: (usize, usize, usize),
}

fn bad(detail: &'static str) -> moxie_types::Error {
    crate::invalid_static(detail)
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| bad("GPTQ dimension/byte multiplication overflows"))
}

fn bytes(len: usize) -> Result<Vec<u8>> {
    let mut out = crate::try_vec(len)?;
    out.resize(len, 0);
    Ok(out)
}

fn exact(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(bad(
            "GPTQ payload or tile length disagrees with its declared geometry",
        ));
    }
    Ok(())
}

impl GptqPlan {
    pub fn new(spec: &GptqSpec, source: DeclaredSource<'_>) -> Result<Self> {
        let (outputs, inputs) = source.logical;
        if outputs == 0 || inputs == 0 || !matches!(spec.group_size, 32 | 128) {
            return Err(bad(
                "GPTQ needs positive dimensions and group size 32 or 128",
            ));
        }
        let pack = 32 / spec.width.bits() as usize;
        let groups = inputs.div_ceil(spec.group_size);
        let input_words = inputs.div_ceil(pack);
        let output_words = outputs.div_ceil(pack);
        if source.qweight_shape != [input_words as u64, outputs as u64]
            || source.qzeros_shape != [groups as u64, output_words as u64]
            || source.scales_shape != [groups as u64, outputs as u64]
        {
            return Err(bad(
                "GPTQ source axes must be qweight[input_words,outputs], qzeros[groups,output_words], scales[groups,outputs]",
            ));
        }
        let lengths = (
            product(product(input_words, outputs)?, 4)?,
            product(product(groups, output_words)?, 4)?,
            product(product(groups, outputs)?, source.scale_dtype.bytes())?,
        );
        // Check all canonical sections before allocating even a map.
        product(outputs, spec.width.row_stride(inputs))?;
        product(product(outputs, groups)?, 2)?;
        let group_index = match (source.group_index, source.group_index_shape) {
            (None, None) if !spec.requires_group_index => None,
            (None, None) => return Err(bad("GPTQ activation ordering requires g_idx")),
            (Some(raw), Some(shape)) => {
                if shape != [inputs as u64] {
                    return Err(bad("GPTQ g_idx must have shape [input channels]"));
                }
                exact(raw.len(), product(inputs, 4)?)?;
                let mut map = crate::try_vec(inputs)?;
                for word in raw.chunks_exact(4) {
                    let group = i32::from_le_bytes(word.try_into().expect("four bytes"));
                    if group < 0 || group as usize >= groups {
                        return Err(bad("GPTQ g_idx contains a negative or out-of-range group"));
                    }
                    map.push(group as u32);
                }
                Some(map)
            }
            _ => return Err(bad("GPTQ g_idx header and payload must both be present")),
        };
        let descriptor = AffineDescriptor {
            width: spec.width,
            out_features: outputs,
            in_features: inputs,
            grouping: Grouping::Contiguous {
                size: spec.group_size as u32,
            },
            group_index,
            scale_dtype: source.scale_dtype,
        };
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            groups,
            pack,
            input_words,
            output_words,
            lengths,
        })
    }

    pub fn descriptor(&self) -> &AffineDescriptor {
        &self.descriptor
    }
    pub fn into_descriptor(self) -> AffineDescriptor {
        self.descriptor
    }
    pub fn out_features(&self) -> usize {
        self.descriptor.out_features
    }
    pub fn in_features(&self) -> usize {
        self.descriptor.in_features
    }
    pub fn groups_per_row(&self) -> usize {
        self.groups
    }
    pub fn values_per_word(&self) -> usize {
        self.pack
    }
    pub fn input_word_rows(&self) -> usize {
        self.input_words
    }
    pub fn output_word_columns(&self) -> usize {
        self.output_words
    }
    /// Full source lengths in `(qweight, qzeros, scales)` order.
    pub fn source_lengths(&self) -> Result<(usize, usize, usize)> {
        Ok(self.lengths)
    }
    /// Bytes gathered per selected output for code conversion.
    pub fn source_code_row_bytes(&self) -> usize {
        self.input_words * 4
    }
    pub fn canonical_code_row_bytes(&self) -> usize {
        self.descriptor.width.row_stride(self.in_features())
    }
    pub fn scale_row_bytes(&self) -> usize {
        self.groups * self.descriptor.scale_dtype.bytes()
    }
    pub fn canonical_zero_point_row_bytes(&self) -> usize {
        self.groups * 2
    }
    /// Bytes gathered per selected output word, across every group.
    pub fn source_zero_point_word_row_bytes(&self) -> usize {
        self.groups * 4
    }
    pub fn zero_points_per_word(&self) -> usize {
        self.pack
    }

    fn rows(&self, rows: &Range<usize>) -> Result<usize> {
        if rows.start >= rows.end || rows.end > self.out_features() {
            return Err(bad(
                "GPTQ output-row tile must be nonempty and inside logical bounds",
            ));
        }
        Ok(rows.end - rows.start)
    }

    /// Output words to gather from EACH source group. All internal tile
    /// boundaries are word aligned; the final partial output word is allowed.
    pub fn zero_point_word_rows(&self, rows: Range<usize>) -> Result<Range<usize>> {
        self.rows(&rows)?;
        if !rows.start.is_multiple_of(self.pack)
            || (rows.end != self.out_features() && !rows.end.is_multiple_of(self.pack))
        {
            return Err(bad(
                "GPTQ zero-point tiles require output-word-aligned boundaries",
            ));
        }
        Ok(rows.start / self.pack..rows.end.div_ceil(self.pack))
    }

    /// `source` is a gathered rectangle `[input_words, rows.len()]`, with
    /// contiguous selected output columns from each qweight source row. It is
    /// NOT a contiguous output-row slice of the original source tensor.
    /// `out` becomes canonical packed output-major codes. No allocation.
    pub fn convert_code_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        out: &mut [u8],
    ) -> Result<()> {
        let count = self.rows(&rows)?;
        exact(source.len(), product(count, self.source_code_row_bytes())?)?;
        exact(out.len(), product(count, self.canonical_code_row_bytes())?)?;
        out.fill(0); // deterministic padding in the final odd INT4 nibble
        let bits = self.descriptor.width.bits();
        for o in 0..count {
            for k in 0..self.in_features() {
                let at = (k / self.pack * count + o) * 4;
                let word = u32::from_le_bytes(source[at..at + 4].try_into().expect("four bytes"));
                let unsigned = (word >> ((k % self.pack) as u32 * bits))
                    & self.descriptor.width.max_unsigned();
                let signed = unsigned as i32 - self.descriptor.width.bias();
                let row = o * self.canonical_code_row_bytes();
                match self.descriptor.width {
                    IntWidth::Int4 => out[row + k / 2] |= ((signed as u8) & 15) << ((k % 2) * 4),
                    IntWidth::Int8 => out[row + k] = signed as u8,
                }
            }
        }
        Ok(())
    }

    /// Gathered `[groups, rows.len()]` scalar rectangle, preserving source
    /// encodings while transposing to canonical `[rows.len(), groups]`.
    pub fn convert_scale_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        out: &mut [u8],
    ) -> Result<()> {
        let count = self.rows(&rows)?;
        exact(source.len(), product(count, self.scale_row_bytes())?)?;
        exact(out.len(), source.len())?;
        let width = self.descriptor.scale_dtype.bytes();
        for g in 0..self.groups {
            for o in 0..count {
                let from = (g * count + o) * width;
                let to = (o * self.groups + g) * width;
                payload::write_scale_block(
                    self.descriptor.scale_dtype,
                    &source[from..from + width],
                    &mut out[to..to + width],
                )?;
            }
        }
        Ok(())
    }

    /// Gathered `[groups, selected_output_words]` I32 rectangle, decoded into
    /// canonical i16 `[selected_outputs, groups]`. The +1 happens in a wide
    /// integer BEFORE rebias: stored max (15/255) represents zero 16/256.
    pub fn convert_zero_point_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        out: &mut [u8],
    ) -> Result<()> {
        let count = self.rows(&rows)?;
        let words = self.zero_point_word_rows(rows)?;
        let word_count = words.end - words.start;
        exact(
            source.len(),
            product(word_count, self.source_zero_point_word_row_bytes())?,
        )?;
        exact(
            out.len(),
            product(count, self.canonical_zero_point_row_bytes())?,
        )?;
        for o in 0..count {
            for g in 0..self.groups {
                let at = (g * word_count + o / self.pack) * 4;
                let word = u32::from_le_bytes(source[at..at + 4].try_into().expect("four bytes"));
                let stored = (word >> ((o % self.pack) as u32 * self.descriptor.width.bits()))
                    & self.descriptor.width.max_unsigned();
                let zero = stored as i32 + 1 - self.descriptor.width.bias();
                let to = (o * self.groups + g) * 2;
                payload::write_zero_point(zero as i16, &mut out[to..to + 2])?;
            }
        }
        Ok(())
    }
}

/// Convenience importer for callers already holding bounded source tensors.
/// Produces only packed integer bytes and scale/zero metadata, never a dense
/// floating-point tensor. Streaming publishers use the row conversion methods.
pub fn import(spec: &GptqSpec, source: SourceTensors<'_>) -> Result<AffineTensor> {
    let plan = GptqPlan::new(spec, source.declaration)?;
    let (weights, zeros, scales) = plan.source_lengths()?;
    exact(source.qweight.len(), weights)?;
    exact(source.qzeros.len(), zeros)?;
    exact(source.scales.len(), scales)?;
    let rows = 0..plan.out_features();
    let mut codes = bytes(product(
        plan.out_features(),
        plan.canonical_code_row_bytes(),
    )?)?;
    plan.convert_code_rows(rows.clone(), source.qweight, &mut codes)?;
    let mut scale_bytes = bytes(scales)?;
    plan.convert_scale_rows(rows.clone(), source.scales, &mut scale_bytes)?;
    let entries = product(plan.out_features(), plan.groups)?;
    let scales = payload::decode_scale_section(plan.descriptor.scale_dtype, &scale_bytes, entries)?;
    drop(scale_bytes);
    let mut zero_bytes = bytes(product(entries, 2)?)?;
    plan.convert_zero_point_rows(rows, source.qzeros, &mut zero_bytes)?;
    let mut zero_points = crate::try_vec(entries)?;
    for pair in zero_bytes.chunks_exact(2) {
        zero_points.push(i16::from_le_bytes([pair[0], pair[1]]));
    }
    drop(zero_bytes);
    AffineTensor::new(
        plan.into_descriptor(),
        codes,
        scales,
        ZeroPoints::PerGroup(zero_points),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        spec: GptqSpec,
        outputs: usize,
        inputs: usize,
        shape_w: [u64; 2],
        shape_z: [u64; 2],
        shape_s: [u64; 2],
        shape_map: [u64; 1],
        weights: Vec<u8>,
        zeros: Vec<u8>,
        scales: Vec<u8>,
        map: Option<Vec<u8>>,
        dtype: ScaleDtype,
    }
    impl Fixture {
        fn new(width: IntWidth, group: usize, dtype: ScaleDtype, mapped: bool) -> Self {
            let (outputs, inputs) = (11usize, 259usize);
            let pack = 32 / width.bits() as usize;
            let groups = inputs.div_ceil(group);
            let mut weights = vec![0u32; inputs.div_ceil(pack) * outputs];
            let mut zeros = vec![0u32; groups * outputs.div_ceil(pack)];
            for o in 0..outputs {
                for k in 0..inputs {
                    weights[k / pack * outputs + o] |= ((k + o * 17) as u32 & width.max_unsigned())
                        << ((k % pack) as u32 * width.bits());
                }
                for g in 0..groups {
                    let z = if (o + g) % 2 == 0 {
                        0
                    } else {
                        width.max_unsigned()
                    };
                    zeros[g * outputs.div_ceil(pack) + o / pack] |=
                        z << ((o % pack) as u32 * width.bits());
                }
            }
            let mut scales = Vec::new();
            for g in 0..groups {
                for o in 0..outputs {
                    let n = (o + g) % 3;
                    match dtype {
                        ScaleDtype::F16 => {
                            scales.extend_from_slice(&[0x3800u16, 0x3c00, 0x4000][n].to_le_bytes())
                        }
                        ScaleDtype::Bf16 => {
                            scales.extend_from_slice(&[0x3f00u16, 0x3f80, 0x4000][n].to_le_bytes())
                        }
                        ScaleDtype::F32 => {
                            scales.extend_from_slice(&[0.5f32, 1.0, 2.0][n].to_le_bytes())
                        }
                    }
                }
            }
            Self {
                spec: GptqSpec {
                    width,
                    group_size: group,
                    requires_group_index: mapped,
                },
                outputs,
                inputs,
                shape_w: [inputs.div_ceil(pack) as u64, outputs as u64],
                shape_z: [groups as u64, outputs.div_ceil(pack) as u64],
                shape_s: [groups as u64, outputs as u64],
                shape_map: [inputs as u64],
                weights: weights.iter().flat_map(|x| x.to_le_bytes()).collect(),
                zeros: zeros.iter().flat_map(|x| x.to_le_bytes()).collect(),
                scales,
                map: mapped.then(|| {
                    (0..inputs)
                        .flat_map(|k| ((groups - 1 - k / group) as i32).to_le_bytes())
                        .collect()
                }),
                dtype,
            }
        }
        fn declaration(&self) -> DeclaredSource<'_> {
            DeclaredSource {
                logical: (self.outputs, self.inputs),
                qweight_shape: &self.shape_w,
                qzeros_shape: &self.shape_z,
                scales_shape: &self.shape_s,
                scale_dtype: self.dtype,
                group_index: self.map.as_deref(),
                group_index_shape: self.map.as_ref().map(|_| self.shape_map.as_slice()),
            }
        }
        fn source(&self) -> SourceTensors<'_> {
            SourceTensors {
                declaration: self.declaration(),
                qweight: &self.weights,
                qzeros: &self.zeros,
                scales: &self.scales,
            }
        }
    }

    #[test]
    fn source_equation_all_codes_zero_extremes_scale_dtypes_and_maps() {
        for width in IntWidth::ALL {
            for group in [32, 128] {
                for dtype in ScaleDtype::ALL {
                    for mapped in [false, true] {
                        let f = Fixture::new(*width, group, *dtype, mapped);
                        let map_shape = [f.inputs as u64];
                        let mut src = f.source();
                        src.declaration.group_index_shape = mapped.then_some(&map_shape);
                        let tensor = import(&f.spec, src).unwrap();
                        for o in 0..f.outputs {
                            for (k, value) in tensor.reconstruct_row(o).unwrap().iter().enumerate()
                            {
                                let g = if mapped {
                                    f.inputs.div_ceil(group) - 1 - k / group
                                } else {
                                    k / group
                                };
                                let unsigned = (k + o * 17) as u32 & width.max_unsigned();
                                let stored_zero = if (o + g) % 2 == 0 {
                                    0
                                } else {
                                    width.max_unsigned()
                                };
                                let expected = (unsigned as i32 - stored_zero as i32 - 1) as f32
                                    * [0.5, 1.0, 2.0][(o + g) % 3];
                                assert_eq!(
                                    value.to_bits(),
                                    expected.to_bits(),
                                    "{width:?}/{group}/{dtype:?}/{mapped} [{o},{k}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn gather(
        bytes: &[u8],
        rows: usize,
        columns: usize,
        selected: Range<usize>,
        scalar: usize,
    ) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| {
                bytes
                    [(r * columns + selected.start) * scalar..(r * columns + selected.end) * scalar]
                    .iter()
                    .copied()
            })
            .collect()
    }

    #[test]
    fn gathered_tiles_match_full_conversion_and_reject_bad_extents() {
        for width in IntWidth::ALL {
            let f = Fixture::new(*width, 32, ScaleDtype::F16, false);
            let plan = GptqPlan::new(&f.spec, f.declaration()).unwrap();
            let tensor = import(&f.spec, f.source()).unwrap();
            let mut canonical = vec![
                0;
                payload::length_of(tensor.descriptor(), payload::ZeroPointSection::PerGroup)
                    .unwrap() as usize
            ];
            payload::encode(&tensor, &mut canonical).unwrap();
            let mut tiled = Vec::new();
            for kind in 0..3 {
                let block = if kind == 2 {
                    plan.zero_points_per_word()
                } else {
                    3
                };
                for start in (0..f.outputs).step_by(block) {
                    let end = (start + block).min(f.outputs);
                    let rows = start..end;
                    let mut dest = vec![
                        0;
                        (end - start)
                            * match kind {
                                0 => plan.canonical_code_row_bytes(),
                                1 => plan.scale_row_bytes(),
                                _ => plan.canonical_zero_point_row_bytes(),
                            }
                    ];
                    match kind {
                        0 => plan
                            .convert_code_rows(
                                rows.clone(),
                                &gather(&f.weights, plan.input_word_rows(), f.outputs, rows, 4),
                                &mut dest,
                            )
                            .unwrap(),
                        1 => plan
                            .convert_scale_rows(
                                rows.clone(),
                                &gather(&f.scales, plan.groups_per_row(), f.outputs, rows, 2),
                                &mut dest,
                            )
                            .unwrap(),
                        _ => {
                            let words = plan.zero_point_word_rows(rows.clone()).unwrap();
                            plan.convert_zero_point_rows(
                                rows,
                                &gather(
                                    &f.zeros,
                                    plan.groups_per_row(),
                                    plan.output_word_columns(),
                                    words,
                                    4,
                                ),
                                &mut dest,
                            )
                            .unwrap();
                        }
                    }
                    tiled.extend_from_slice(&dest);
                }
            }
            assert_eq!(tiled, canonical);
            assert!(plan.convert_code_rows(0..1, &[], &mut []).is_err());
            assert!(plan.convert_scale_rows(1..1, &[], &mut []).is_err());
            assert!(plan.zero_point_word_rows(1..f.outputs).is_err());
            let mut wrong = f.declaration();
            wrong.qweight_shape = &[1, 1];
            assert!(GptqPlan::new(&f.spec, wrong).is_err());
            let mut missing = f.spec;
            missing.requires_group_index = true;
            assert!(GptqPlan::new(&missing, f.declaration()).is_err());
            let bad_map = vec![255u8; f.inputs * 4];
            let map_shape = [f.inputs as u64];
            let mut wrong = f.declaration();
            wrong.group_index = Some(&bad_map);
            wrong.group_index_shape = Some(&map_shape);
            assert!(GptqPlan::new(&f.spec, wrong).is_err());
        }
    }
}
