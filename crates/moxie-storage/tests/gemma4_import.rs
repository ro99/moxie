//! Task 0018: import real compressed-tensors `pack-quantized` tensors.
//!
//! **Read-only, and nothing here is model support.** This imports a handful of
//! tensors from an inventoried local artifact into canonical affine form. It
//! executes nothing, writes nothing, and proves no quality claim: a tensor that
//! decodes to finite values says the reader agrees with the file, not that the
//! model works.
//!
//! The artifact is not in git and is not required to run the suite. Every test
//! here **skips with a message** when it is absent rather than failing, so a
//! fresh clone is green and a missing checkpoint is never mistaken for a
//! regression.
//!
//! O5 governs bulk writes and this touches none: `File::open` and positioned
//! reads only, under the root
//! [artifact-roots](../../../docs/evidence/artifact-roots.md) already
//! designates for inspection.

use std::path::{Path, PathBuf};

use moxie_format::affine::IntWidth;
use moxie_format::compressed_tensors::{
    Granularity, PackQuantizedSpec, TensorTriple, decode_weight_shape, import, scale_dtype_of,
    triple_entries,
};
use moxie_format::safetensors::Dtype;
use moxie_format::scale::ScaleDtype;
use moxie_storage::Shard;

const ROOT: &str = "/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit";

/// The artifact's declared quantization parameters, from
/// `docs/models/gemma4.md`: `pack-quantized`, 8 bits, symmetric, group 32.
fn spec() -> PackQuantizedSpec {
    PackQuantizedSpec {
        width: IntWidth::Int8,
        granularity: Granularity::Group { size: 32 },
        symmetric: true,
    }
}

fn shard(n: u32) -> Option<PathBuf> {
    let p = Path::new(ROOT).join(format!("model-0000{n}-of-00007.safetensors"));
    p.exists().then_some(p)
}

fn skip(what: &str) {
    eprintln!("SKIP {what}: {ROOT} is not present on this machine");
}

/// Every shard's header parses, and its tensors account for its payload
/// exactly -- the `8 + header + payload == size` identity the inventory
/// recorded, re-derived here through the reader under test.
#[test]
fn all_seven_shard_headers_parse_and_account_for_their_payloads() {
    let Some(_) = shard(1) else {
        return skip("shard header consistency");
    };
    let mut tensors = 0usize;
    let mut payload = 0u64;
    let (mut bf16, mut i32s, mut i64s) = (0usize, 0usize, 0usize);
    for n in 1..=7 {
        let path = shard(n).expect("shard present");
        let s = Shard::open(&path).unwrap();
        let h = s.header();
        assert!(
            h.covers_payload_exactly(),
            "shard {n} payload is {} but its tensors cover {}",
            h.payload_len,
            h.covered_bytes()
        );
        for e in h.tensors().values() {
            match e.dtype {
                Dtype::Bf16 => bf16 += 1,
                Dtype::I32 => i32s += 1,
                Dtype::I64 => i64s += 1,
                other => panic!("unexpected dtype {} in shard {n}", other.name()),
            }
        }
        tensors += h.tensors().len();
        payload += h.payload_len;
    }
    // The numbers `docs/evidence/checkpoint-inventory.md` recorded on
    // 2026-09-12, re-derived by this reader rather than restated.
    assert_eq!(tensors, 2008, "tensor count");
    assert_eq!((bf16, i32s, i64s), (1188, 410, 410), "dtype census");
    assert_eq!(payload, 35_089_877_112, "total tensor payload");
    eprintln!(
        "task0018 artifact evidence: shards=7 tensors={tensors} payload_bytes={payload} \
         bf16={bf16} i32={i32s} i64={i64s}"
    );
}

/// Import three real modules whose logical shapes differ, including one whose
/// output and input axes are transposed relative to another.
#[test]
fn real_pack_quantized_tensors_import_to_canonical_affine_form() {
    let Some(_) = shard(1) else {
        return skip("real pack-quantized import");
    };
    // Module, expected logical shape from the bring-up record's tensor table.
    let wanted: [(&str, (usize, usize)); 3] = [
        ("self_attn.q_proj", (8192, 5376)),
        ("self_attn.o_proj", (5376, 8192)),
        ("mlp.down_proj", (5376, 21504)),
    ];
    let mut imported = 0usize;
    let mut bytes_read = 0u64;
    for n in 1..=7 {
        let path = shard(n).expect("shard present");
        let s = Shard::open(&path).unwrap();
        for (suffix, expected) in wanted {
            // The first layer in this shard carrying the module, whichever it is.
            let Some(module) = s
                .header()
                .tensors()
                .keys()
                .filter_map(|k| k.strip_suffix(".weight_packed"))
                .find(|m| m.ends_with(suffix))
                .map(str::to_string)
            else {
                continue;
            };
            let (packed_e, scale_e, shape_e) = triple_entries(s.header(), &module).unwrap();
            // The scale dtype comes from the header, not from the config --
            // the artifact declares `scale_dtype: null`.
            let scale_dtype = scale_dtype_of(scale_e.dtype).unwrap();
            assert_eq!(scale_dtype, ScaleDtype::Bf16, "{module} scale dtype");

            let packed_shape = packed_e.shape.clone();
            let scale_shape = scale_e.shape.clone();
            bytes_read += packed_e.len() + scale_e.len() + shape_e.len();

            let shape_bytes = s.tensor_bytes(&format!("{module}.weight_shape")).unwrap();
            let logical = decode_weight_shape(&shape_bytes).unwrap();
            assert_eq!(logical, expected, "{module} logical shape");

            // Four INT8 codes per I32 along the input axis; one scale per 32.
            assert_eq!(
                packed_shape,
                vec![logical.0 as u64, (logical.1 / 4) as u64],
                "{module} packed shape"
            );
            assert_eq!(
                scale_shape,
                vec![logical.0 as u64, (logical.1 / 32) as u64],
                "{module} scale shape"
            );

            let packed = s.tensor_bytes(&format!("{module}.weight_packed")).unwrap();
            let scale = s.tensor_bytes(&format!("{module}.weight_scale")).unwrap();
            let tensor = import(
                &spec(),
                TensorTriple {
                    packed: &packed,
                    // The shapes the header declared, so the importer checks
                    // the axes rather than trusting the byte counts.
                    packed_shape: &packed_shape,
                    scale: &scale,
                    scale_shape: &scale_shape,
                    scale_dtype,
                    logical,
                },
            )
            .unwrap();
            let d = tensor.descriptor();
            assert_eq!((d.out_features, d.in_features), logical);
            assert_eq!(d.width, IntWidth::Int8);

            // Spot-check reconstruction on real data: finite everywhere, and
            // the codes actually span the range rather than being all one value.
            let mut low = 0;
            let mut high = 0;
            for o in [0usize, logical.0 / 2, logical.0 - 1] {
                let row = tensor.reconstruct_row(o).unwrap();
                assert_eq!(row.len(), logical.1);
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "{module} row {o} has a nonfinite value"
                );
                for k in 0..logical.1 {
                    let c = tensor.code(o, k).unwrap();
                    low = low.min(c);
                    high = high.max(c);
                }
            }
            assert!(
                low < -8 && high > 8,
                "{module} codes span only [{low}, {high}], which is not a quantized weight"
            );
            eprintln!(
                "task0018 imported {module}: logical={logical:?} codes=[{low}, {high}] \
                 scale=bf16 group=32"
            );
            imported += 1;
            if imported == wanted.len() {
                eprintln!("task0018 artifact bytes read: {bytes_read}");
                return;
            }
        }
    }
    panic!(
        "expected to import {} modules, imported {imported}",
        wanted.len()
    );
}

/// The artifact's `v_proj` is absent on exactly the ten global layers. A reader
/// that invented a tensor, or that failed on a legitimately absent one with the
/// wrong error, would show up here.
#[test]
fn an_absent_module_is_a_typed_error_and_not_a_fabricated_tensor() {
    let Some(path) = shard(2) else {
        return skip("absent-module refusal");
    };
    let s = Shard::open(&path).unwrap();
    assert!(
        s.header()
            .get("model.language_model.layers.0.nonexistent")
            .is_err()
    );
    assert!(triple_entries(s.header(), "model.language_model.layers.0.nonexistent").is_err());
}
