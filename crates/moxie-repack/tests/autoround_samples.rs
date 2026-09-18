//! Read-only, bounded samples of the two task0034 catalog revisions. No artifact
//! is converted or written. These prove affine values, not model output.
use moxie_format::{
    affine::IntWidth,
    checkpoint_config::IntegerSerialization,
    compressed_tensors::Granularity,
    gptq::{DeclaredSource, GptqPlan, GptqSpec},
    scale::ScaleDtype,
};
use moxie_storage::{ByteBudget, HeaderBudget, Shard};
use std::path::Path;

#[test]
fn pinned_autoround_samples_preserve_source_values() {
    for (model, revision, module) in [
        (
            "GLM-5.3-Flash-W4A16-AutoRound",
            "5eee1846f0321058ed73745f9aa16f2aaf0fc0a0",
            "model.language_model.layers.10.mlp.experts.0.down_proj",
        ),
        (
            "Qwen3.8-Flash-Next-W4A16-AutoRound",
            "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897",
            "model.language_model.layers.0.mlp.experts.0.down_proj",
        ),
    ] {
        let root = Path::new("/fast/models/Intel").join(model);
        if !root.is_dir() {
            eprintln!("SKIPPED: {model} absent");
            continue;
        }
        let config = std::fs::read_to_string(root.join("config.json")).unwrap();
        let declaration = moxie_format::checkpoint_config::parse(&config).unwrap();
        let quantization = declaration.quantization.as_ref().unwrap();
        assert_eq!(quantization.bits, 4);
        assert_eq!(quantization.granularity, Granularity::Group { size: 128 });
        assert!(matches!(
            quantization.serialization,
            IntegerSerialization::GptqV1 { .. }
        ));
        assert!(!quantization.passthrough_patterns.is_empty());
        let required_override = if model.starts_with("Qwen") {
            ".*mtp.*"
        } else {
            ".*shared_head.*"
        };
        assert!(
            quantization
                .passthrough_patterns
                .iter()
                .any(|pattern| pattern == required_override),
            "{model} lost its {required_override:?} 16-bit override"
        );
        let index = moxie_format::checkpoint_config::parse_index(
            &std::fs::read_to_string(root.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert!(
            !index.contains_key(&format!("{module}.g_idx")),
            "sample requires contiguous source groups"
        );
        let open = |suffix: &str| {
            let name = format!("{module}.{suffix}");
            let file = index[&name].as_str();
            let metadata = std::fs::read_to_string(
                root.join(".cache/huggingface/download")
                    .join(format!("{file}.metadata")),
            )
            .unwrap();
            assert_eq!(metadata.lines().next().unwrap(), revision);
            let shard = Shard::open_with_limits(
                &root.join(file),
                ByteBudget::new(64 << 10).unwrap(),
                HeaderBudget::new(128 << 20).unwrap(),
            )
            .unwrap();
            (shard, name)
        };
        let (w, wn) = open("qweight");
        let (z, zn) = open("qzeros");
        let (s, sn) = open("scales");
        let we = w.header().get(&wn).unwrap();
        let ze = z.header().get(&zn).unwrap();
        let se = s.header().get(&sn).unwrap();
        assert_eq!(se.dtype, moxie_format::safetensors::Dtype::F16);
        let outputs = we.shape[1] as usize;
        let inputs = we.shape[0] as usize * 8;
        let groups = inputs.div_ceil(128);
        let plan = GptqPlan::new(
            &GptqSpec {
                width: IntWidth::Int4,
                group_size: 128,
                requires_group_index: false,
            },
            DeclaredSource {
                logical: (outputs, inputs),
                qweight_shape: &we.shape,
                qzeros_shape: &ze.shape,
                scales_shape: &se.shape,
                scale_dtype: ScaleDtype::F16,
                group_index: None,
                group_index_shape: None,
            },
        )
        .unwrap();
        let gather = |shard: &Shard,
                      name: &str,
                      rows: usize,
                      columns: usize,
                      first: usize,
                      count: usize,
                      unit: usize| {
            let mut out = vec![0; rows * count * unit];
            for r in 0..rows {
                shard
                    .read_tensor_range(
                        name,
                        ((r * columns + first) * unit) as u64,
                        &mut out[r * count * unit..(r + 1) * count * unit],
                    )
                    .unwrap();
            }
            out
        };
        for first in [0, outputs - 8] {
            let weights = gather(&w, &wn, inputs / 8, outputs, first, 8, 4);
            let zeros = gather(&z, &zn, groups, outputs / 8, first / 8, 1, 4);
            let scales = gather(&s, &sn, groups, outputs, first, 8, 2);
            let mut packed = vec![0; 8 * plan.canonical_code_row_bytes()];
            let mut scale_bytes = vec![0; 8 * plan.scale_row_bytes()];
            let mut zero_bytes = vec![0; 8 * plan.canonical_zero_point_row_bytes()];
            plan.convert_code_rows(first..first + 8, &weights, &mut packed)
                .unwrap();
            plan.convert_scale_rows(first..first + 8, &scales, &mut scale_bytes)
                .unwrap();
            plan.convert_zero_point_rows(first..first + 8, &zeros, &mut zero_bytes)
                .unwrap();
            packed.extend_from_slice(&scale_bytes);
            packed.extend_from_slice(&zero_bytes);
            let mut descriptor = plan.descriptor().clone();
            descriptor.out_features = 8;
            let tensor = moxie_format::payload::decode(
                descriptor,
                moxie_format::payload::ZeroPointSection::PerGroup,
                &packed,
            )
            .unwrap();
            for o in 0..8 {
                let actual = tensor.reconstruct_row(o).unwrap();
                for (k, value) in actual.iter().enumerate() {
                    let at = (k / 8 * 8 + o) * 4;
                    let q = (u32::from_le_bytes(weights[at..at + 4].try_into().unwrap())
                        >> (k % 8 * 4))
                        & 15;
                    let g = k / 128;
                    let zero = ((u32::from_le_bytes(zeros[g * 4..g * 4 + 4].try_into().unwrap())
                        >> (o * 4))
                        & 15)
                        + 1;
                    let at = (g * 8 + o) * 2;
                    let bits = u16::from_le_bytes(scales[at..at + 2].try_into().unwrap());
                    let exponent = (bits >> 10) & 31;
                    assert!(exponent < 31);
                    let scale = if exponent == 0 {
                        f32::from(bits & 1023) * 2f32.powi(-24)
                    } else {
                        f32::from(1024 + (bits & 1023)) * 2f32.powi(i32::from(exponent) - 25)
                    } * if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
                    assert_eq!(
                        value.to_bits(),
                        ((q as i32 - zero as i32) as f32 * scale).to_bits()
                    );
                }
            }
            eprintln!(
                "{model} revision={revision} shape=[{outputs},{inputs}] rows={first}..{} values={} scale=F16 source_header={} gathered_codes={} gathered_scales={} gathered_zeros={}",
                first + 8,
                8 * inputs,
                w.header_sha256(),
                moxie_format::sha256_hex(&weights),
                moxie_format::sha256_hex(&scales),
                moxie_format::sha256_hex(&zeros)
            );
        }
    }
}
