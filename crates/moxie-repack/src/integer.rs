//! Closed source-codec dispatch for one canonical affine publication path.
use moxie_format::{
    affine::AffineDescriptor, compressed_tensors::PackQuantizedPlan, gptq::GptqPlan,
};
use moxie_types::Result;
use std::ops::Range;

#[derive(Debug)]
pub enum IntegerPlan {
    CompressedTensors(PackQuantizedPlan),
    Gptq(GptqPlan),
}
macro_rules! geometry {
    ($($name:ident),*) => {$ (
        pub fn $name(&self) -> usize {
            match self { Self::CompressedTensors(p)=>p.$name(), Self::Gptq(p)=>p.$name() }
        }
    )*};
}
impl IntegerPlan {
    pub fn descriptor(&self) -> &AffineDescriptor {
        match self {
            Self::CompressedTensors(p) => p.descriptor(),
            Self::Gptq(p) => p.descriptor(),
        }
    }
    geometry!(
        out_features,
        groups_per_row,
        zero_points_per_word,
        source_code_row_bytes,
        canonical_code_row_bytes,
        scale_row_bytes,
        source_zero_point_word_row_bytes,
        canonical_zero_point_row_bytes
    );
    pub fn zero_point_word_rows(&self, rows: Range<usize>) -> Result<Range<usize>> {
        match self {
            Self::CompressedTensors(p) => p.zero_point_word_rows(rows),
            Self::Gptq(p) => p.zero_point_word_rows(rows),
        }
    }
    pub fn convert_code_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        scratch: &mut [i32],
        out: &mut [u8],
    ) -> Result<()> {
        match self {
            Self::CompressedTensors(p) => p.convert_code_rows(rows, source, scratch, out),
            Self::Gptq(p) => p.convert_code_rows(rows, source, out),
        }
    }
    pub fn convert_scale_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        out: &mut [u8],
    ) -> Result<()> {
        match self {
            Self::CompressedTensors(p) => p.convert_scale_rows(rows, source, out),
            Self::Gptq(p) => p.convert_scale_rows(rows, source, out),
        }
    }
    pub fn convert_zero_point_rows(
        &self,
        rows: Range<usize>,
        source: &[u8],
        out: &mut [u8],
    ) -> Result<()> {
        match self {
            Self::CompressedTensors(p) => p.convert_zero_point_rows(rows, source, out),
            Self::Gptq(p) => p.convert_zero_point_rows(rows, source, out),
        }
    }
}
