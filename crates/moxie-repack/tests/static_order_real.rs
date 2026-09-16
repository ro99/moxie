//! Read-only bounded samples from the pinned static-order source. No execution
//! or checkpoint publication; comparison is the canonical FP32 equation.
use std::path::Path;

use moxie_format::affine::IntWidth;
use moxie_format::compressed_tensors::{
    self, Granularity, PackQuantizedSpec, SourceTensors, ZeroPointSource,
};
use moxie_format::scale::ScaleDtype;
use moxie_storage::{ByteBudget, HeaderBudget, Shard};

const ROOT: &str = "/fast/models/canada-quant/hy3-w4a16-mtp";
const REVISION: &str = "49228b990c704e4efd67ac420a8e3d5272f820c0";

#[test]
fn pinned_static_order_modules_preserve_sampled_logical_columns() {
    let root = Path::new(ROOT);
    if !root.is_dir() {
        eprintln!("UNMEASURED: pinned static-order checkpoint absent at {ROOT}");
        return;
    }
    let config = moxie_storage::read_text_capped(&root.join("config.json"), 8 << 20).unwrap();
    assert_eq!(
        moxie_format::sha256_hex(config.as_bytes()),
        "8f5f45c43eb2243af93718347626ba1fb4db7abd466c30321b290adcfcce35aa"
    );
    let declaration = moxie_format::checkpoint_config::parse(&config).unwrap();
    let q = declaration.quantization.unwrap();
    assert_eq!(
        (q.bits, q.granularity, q.zero_points),
        (
            4,
            Granularity::Group { size: 128 },
            ZeroPointSource::Symmetric
        )
    );
    let index_text =
        moxie_storage::read_text_capped(&root.join("model.safetensors.index.json"), 64 << 20)
            .unwrap();
    let index = moxie_format::checkpoint_config::parse_index(&index_text).unwrap();
    assert!(!index.keys().any(|k| k.ends_with(".weight_g_idx")));
    for file in index.values().collect::<std::collections::BTreeSet<_>>() {
        // Header parser also checks the declared payload endpoint against file
        // length. This is structural completeness, not a full payload digest.
        Shard::open_with_limits(
            &root.join(file),
            ByteBudget::new(64 << 10).unwrap(),
            HeaderBudget::new(128 << 20).unwrap(),
        )
        .unwrap();
    }
    let spec = PackQuantizedSpec {
        width: IntWidth::Int4,
        granularity: Granularity::Group { size: 128 },
        zero_points: ZeroPointSource::Symmetric,
    };
    let mut checked = 0;
    for projection in ["down_proj", "gate_proj"] {
        let module = format!("model.layers.1.mlp.experts.0.{projection}");
        let packed_name = format!("{module}.weight_packed");
        let scale_name = format!("{module}.weight_scale");
        let shape_name = format!("{module}.weight_shape");
        let file = &index[&packed_name];
        assert_eq!(&index[&scale_name], file);
        assert_eq!(&index[&shape_name], file);
        let metadata = std::fs::read_to_string(
            root.join(".cache/huggingface/download")
                .join(format!("{file}.metadata")),
        )
        .unwrap();
        assert_eq!(metadata.lines().next().unwrap(), REVISION);
        let shard = Shard::open_with_limits(
            &root.join(file),
            ByteBudget::new(64 << 10).unwrap(),
            HeaderBudget::new(128 << 20).unwrap(),
        )
        .unwrap();
        let entries =
            compressed_tensors::source_entries(shard.header(), &module, ZeroPointSource::Symmetric)
                .unwrap();
        assert_eq!(
            compressed_tensors::scale_dtype_of(entries.scale.dtype).unwrap(),
            ScaleDtype::Bf16
        );
        let mut shape_bytes = [0; 16];
        shard.read_tensor(&shape_name, &mut shape_bytes).unwrap();
        let (rows, columns) = compressed_tensors::decode_weight_shape(&shape_bytes).unwrap();
        let packed_row = columns.div_ceil(8) * 4;
        let groups = columns.div_ceil(128);
        assert_eq!(
            entries.packed.shape,
            [rows as u64, columns.div_ceil(8) as u64]
        );
        assert_eq!(entries.scale.shape, [rows as u64, groups as u64]);
        for row in [0, rows - 1] {
            let mut codes = vec![0; packed_row];
            let mut scales = vec![0; groups * 2];
            shard
                .read_tensor_range(&packed_name, (row * packed_row) as u64, &mut codes)
                .unwrap();
            shard
                .read_tensor_range(&scale_name, (row * groups * 2) as u64, &mut scales)
                .unwrap();
            let tensor = compressed_tensors::import(
                &spec,
                SourceTensors {
                    packed: &codes,
                    packed_shape: &[1, columns.div_ceil(8) as u64],
                    scale: &scales,
                    scale_shape: &[1, groups as u64],
                    scale_dtype: ScaleDtype::Bf16,
                    zero_point: None,
                    logical: (1, columns),
                },
            )
            .unwrap();
            for (column, got) in tensor.reconstruct_row(0).unwrap().iter().enumerate() {
                // Independent source-byte expression: low-to-high packed input
                // lanes, symmetric bias8, original column's contiguous group.
                let byte = codes[column / 2];
                let q = ((byte >> (4 * (column % 2))) & 15) as i32 - 8;
                let i = column / 128 * 2;
                let s =
                    f32::from_bits(u32::from(u16::from_le_bytes([scales[i], scales[i + 1]])) << 16);
                assert_eq!(
                    got.to_bits(),
                    (q as f32 * s).to_bits(),
                    "{module} [{row},{column}]"
                );
                checked += 1;
            }
            eprintln!(
                "{module} row {row}: header={} codes={} scales={}",
                shard.header_sha256(),
                moxie_format::sha256_hex(&codes),
                moxie_format::sha256_hex(&scales)
            );
        }
    }
    assert_eq!(checked, 2 * (1536 + 4096));
    eprintln!("static order: {checked} canonical FP32 values exact; no model-output claim");
}
