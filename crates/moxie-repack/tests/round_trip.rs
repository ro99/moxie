//! Task 0025 acceptance 4 and 8's synthetic half: the values, the bytes, and
//! what comes back out.
//!
//! Every fixture here is built from explicit source values, and the canonical
//! payload each one should produce is computed from those same values by
//! [`common::Module::expected_canonical`] -- an independent statement of ADR
//! 0023's layout that never calls the codec. A test that wrote with the
//! encoder and read with the decoder would agree with itself about a layout
//! neither of them had to get right.
//!
//! Coverage, deliberately enumerated rather than sampled: every signed INT4 and
//! INT8 code, asymmetric zero points across their whole range, all three source
//! scale encodings, group boundaries and tails on both axes, symmetric and
//! asymmetric modules, mixed BF16, and a module whose companions live in
//! different shards.

mod common;

use std::collections::BTreeMap;

use common::{Entry, Module, Scratch, SelectionBuilder, budgets, run, write_shard};

/// Deterministic codes covering the whole unsigned range for a width.
fn codes(rows: usize, columns: usize, bits: u32, seed: usize) -> Vec<Vec<u32>> {
    let span = 1usize << bits;
    (0..rows)
        .map(|o| {
            (0..columns)
                .map(|k| ((o * 7 + k * 3 + seed) % span) as u32)
                .collect()
        })
        .collect()
}

fn scales(rows: usize, groups: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|o| {
            (0..groups)
                .map(|g| match (o + g) % 4 {
                    0 => 0.5,
                    1 => 1.0,
                    2 => 2.0,
                    _ => 0.25,
                })
                .collect()
        })
        .collect()
}

fn zeros(rows: usize, groups: usize, bits: u32) -> Vec<Vec<u32>> {
    let span = 1usize << bits;
    (0..rows)
        .map(|o| {
            (0..groups)
                .map(|g| ((o * 5 + g * 11) % span) as u32)
                .collect()
        })
        .collect()
}

/// Repack one module and compare the published payload with bytes this test
/// built itself.
fn round_trip(label: &str, module: &Module, scale_dtype: &'static str, split_shards: bool) {
    let scratch = Scratch::new(label);
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let name = "model.layers.0.mlp.down_proj";
    let symmetric = module.zeros.is_empty();

    let mut first = vec![
        Entry::new(
            &format!("{name}.weight_packed"),
            "I32",
            module.packed_shape(),
            module.packed(),
        ),
        Entry::new(
            &format!("{name}.weight_shape"),
            "I64",
            vec![2],
            module.weight_shape(),
        ),
    ];
    if !symmetric {
        first.push(Entry::new(
            &format!("{name}.weight_zero_point"),
            "I32",
            module.zero_point_shape(),
            module.zero_point(),
        ));
    }
    let scale_entry = Entry::new(
        &format!("{name}.weight_scale"),
        match scale_dtype {
            "BF16" => "BF16",
            "F16" => "F16",
            _ => "F32",
        },
        module.scale_shape(),
        module.scale_payload(scale_dtype),
    );
    let scale_file = if split_shards {
        write_shard(&src.join("shard-b.safetensors"), &[scale_entry]);
        "shard-b.safetensors"
    } else {
        first.push(scale_entry);
        "shard-a.safetensors"
    };
    write_shard(&src.join("shard-a.safetensors"), &first);

    let mut files = BTreeMap::from([
        ("weight_packed", "shard-a.safetensors"),
        ("weight_shape", "shard-a.safetensors"),
        ("weight_scale", scale_file),
    ]);
    if !symmetric {
        files.insert("weight_zero_point", "shard-a.safetensors");
    }
    let selection = SelectionBuilder::new("synthetic").pack_quantized(
        "model.layers.0.mlp.down_proj.weight",
        name,
        if module.bits == 4 { "int4" } else { "int8" },
        &module.group.to_string(),
        if symmetric {
            "symmetric"
        } else {
            "packed-along-output"
        },
        &files,
    );
    let selection_path = scratch.join("selection.toml");
    selection.write(&selection_path);
    let out = scratch.join("artifact");

    let mut args = vec![
        "repack",
        "--selection",
        selection_path.to_str().expect("a path"),
        "--source-root",
        src.to_str().expect("a path"),
        "--out",
        out.to_str().expect("a path"),
    ];
    args.extend(budgets());
    let result = run(&args);
    assert_eq!(
        result.outcome(),
        "published",
        "{label}: {}{}",
        result.stdout,
        result.stderr
    );

    // The published payload, against bytes this test built from the values it
    // wrote into the shard. The components stream in canonical order, so what
    // comes back is exactly ADR 0023's concatenation -- the same bytes, now
    // carried as three physical safetensors tensors.
    let expected = module.expected_canonical(scale_dtype);
    let artifact = moxie_storage::Artifact::open(&out).expect("the artifact opens");
    let mut published = Vec::new();
    let mut scratch = vec![0u8; 4096];
    artifact
        .stream_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &mut scratch,
            &mut |slice| {
                published.extend_from_slice(slice);
                Ok(())
            },
        )
        .expect("every component verifies against its own checksum");
    assert_eq!(published.len(), expected.len(), "{label}: canonical length");
    assert_eq!(published, expected, "{label}: canonical bytes");

    // And the production reader agrees the bytes are the bytes their
    // checksums name.
    let verify = run(&[
        "verify",
        "--artifact",
        out.to_str().expect("a path"),
        "--scratch-bytes",
        "4096",
    ]);
    assert_eq!(verify.outcome(), "verified", "{label}: {}", verify.stdout);
    assert_eq!(
        verify.field("bytes-verified"),
        Some(expected.len().to_string().as_str()),
        "{label}"
    );
}

#[test]
fn every_int4_code_and_zero_point_round_trips_to_bytes_built_by_hand() {
    // 16 rows so the zero-point word rows are whole; 64 columns so there are
    // two groups; the code generator walks all sixteen values.
    let rows = 16;
    let columns = 64;
    let groups = columns / 32;
    round_trip(
        "int4-asymmetric",
        &Module {
            rows,
            columns,
            group: 32,
            bits: 4,
            codes: codes(rows, columns, 4, 0),
            scales: scales(rows, groups),
            zeros: zeros(rows, groups, 4),
        },
        "BF16",
        false,
    );
}

#[test]
fn every_int8_code_and_zero_point_round_trips() {
    let rows = 8;
    let columns = 256;
    let groups = columns / 128;
    round_trip(
        "int8-asymmetric",
        &Module {
            rows,
            columns,
            group: 128,
            bits: 8,
            codes: codes(rows, columns, 8, 1),
            scales: scales(rows, groups),
            zeros: zeros(rows, groups, 8),
        },
        "F32",
        false,
    );
}

#[test]
fn a_symmetric_module_publishes_no_zero_point_section() {
    let rows = 8;
    let columns = 128;
    let groups = columns / 32;
    let module = Module {
        rows,
        columns,
        group: 32,
        bits: 4,
        codes: codes(rows, columns, 4, 2),
        scales: scales(rows, groups),
        zeros: Vec::new(),
    };
    // Codes plus scales and nothing else: implicit zero has no payload.
    assert_eq!(
        module.expected_canonical("F16").len(),
        rows * columns / 2 + rows * groups * 2
    );
    round_trip("int4-symmetric", &module, "F16", false);
}

/// Tails on both axes at once: a column count that is not a multiple of the
/// group or of the packing factor, and a row count that is not a multiple of
/// the zero-point packing factor.
#[test]
fn group_and_axis_tails_round_trip() {
    for (rows, columns, bits, group) in [(13usize, 66usize, 4u32, 32usize), (5, 130, 8, 128)] {
        let groups = columns.div_ceil(group);
        round_trip(
            &format!("tails-{bits}-{rows}x{columns}"),
            &Module {
                rows,
                columns,
                group,
                bits,
                codes: codes(rows, columns, bits, 3),
                scales: scales(rows, groups),
                zeros: zeros(rows, groups, bits),
            },
            "BF16",
            false,
        );
    }
}

#[test]
fn every_source_scale_encoding_is_preserved_exactly() {
    for dtype in ["BF16", "F16", "F32"] {
        let (rows, columns) = (8, 64);
        let groups = columns / 32;
        round_trip(
            &format!("scale-{dtype}"),
            &Module {
                rows,
                columns,
                group: 32,
                bits: 4,
                codes: codes(rows, columns, 4, 4),
                scales: scales(rows, groups),
                zeros: zeros(rows, groups, 4),
            },
            dtype,
            false,
        );
    }
}

/// Task 0024 measured that a module's four tensors need not share a shard.
#[test]
fn a_module_split_across_shards_round_trips() {
    let (rows, columns) = (16, 64);
    let groups = columns / 32;
    round_trip(
        "split-shards",
        &Module {
            rows,
            columns,
            group: 32,
            bits: 4,
            codes: codes(rows, columns, 4, 5),
            scales: scales(rows, groups),
            zeros: zeros(rows, groups, 4),
        },
        "BF16",
        true,
    );
}

/// Mixed precisions in one artifact, and a BF16 payload preserved bit for bit.
#[test]
fn a_mixed_selection_publishes_both_precisions_with_bf16_bits_unchanged() {
    let scratch = Scratch::new("mixed");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let (rows, columns) = (16usize, 64usize);
    let module = Module {
        rows,
        columns,
        group: 32,
        bits: 4,
        codes: codes(rows, columns, 4, 6),
        scales: scales(rows, columns / 32),
        zeros: zeros(rows, columns / 32, 4),
    };
    let name = "model.layers.0.mlp.down_proj";
    // A BF16 payload with awkward values in it: a subnormal, a negative zero
    // and the largest finite BF16. None of them may change.
    let norm_bits: Vec<u16> = vec![
        0x0001, 0x8000, 0x7F7F, 0x3F80, 0xBF80, 0x0080, 0x3F00, 0x4049,
    ];
    let norm: Vec<u8> = norm_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    write_shard(
        &src.join("shard-a.safetensors"),
        &[
            Entry::new(
                &format!("{name}.weight_packed"),
                "I32",
                module.packed_shape(),
                module.packed(),
            ),
            Entry::new(
                &format!("{name}.weight_scale"),
                "BF16",
                module.scale_shape(),
                module.scale_payload("BF16"),
            ),
            Entry::new(
                &format!("{name}.weight_shape"),
                "I64",
                vec![2],
                module.weight_shape(),
            ),
            Entry::new(
                &format!("{name}.weight_zero_point"),
                "I32",
                module.zero_point_shape(),
                module.zero_point(),
            ),
            Entry::new("model.norm.weight", "BF16", vec![8], norm.clone()),
        ],
    );
    let files = BTreeMap::from([
        ("weight_packed", "shard-a.safetensors"),
        ("weight_scale", "shard-a.safetensors"),
        ("weight_shape", "shard-a.safetensors"),
        ("weight_zero_point", "shard-a.safetensors"),
    ]);
    let selection = SelectionBuilder::new("synthetic")
        .pack_quantized(
            "model.layers.0.mlp.down_proj.weight",
            name,
            "int4",
            "32",
            "packed-along-output",
            &files,
        )
        .bf16(
            "model.norm.weight",
            "model.norm.weight",
            "shard-a.safetensors",
        );
    let selection_path = scratch.join("selection.toml");
    selection.write(&selection_path);
    let out = scratch.join("artifact");
    let mut args = vec![
        "repack",
        "--selection",
        selection_path.to_str().expect("a path"),
        "--source-root",
        src.to_str().expect("a path"),
        "--out",
        out.to_str().expect("a path"),
    ];
    args.extend(budgets());
    let result = run(&args);
    assert_eq!(result.outcome(), "published", "{}", result.stdout);

    let artifact = moxie_storage::Artifact::open(&out).expect("the artifact opens");
    let mut scratch = vec![0u8; 4096];
    let mut quantized_read = Vec::new();
    artifact
        .stream_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &mut scratch,
            &mut |s| {
                quantized_read.extend_from_slice(s);
                Ok(())
            },
        )
        .expect("the quantized tensor verifies");
    assert_eq!(quantized_read, module.expected_canonical("BF16"));
    // The BF16 tensor is its own component, and its bits are exactly the bits
    // that went in -- including the subnormal, the negative zero and the
    // largest finite BF16.
    let mut norm_read = Vec::new();
    artifact
        .stream_tensor("model.norm.weight", &mut scratch, &mut |s| {
            norm_read.extend_from_slice(s);
            Ok(())
        })
        .expect("the BF16 tensor verifies");
    assert_eq!(norm_read, norm);
    let manifest = std::fs::read_to_string(out.join("manifest.toml")).expect("a manifest");
    assert!(
        manifest.contains("precision = \"affine-int4-v1\""),
        "{manifest}"
    );
    assert!(manifest.contains("precision = \"bf16-v1\""), "{manifest}");
}

/// Manifest v1 assigns a tensor to one chunk, so several tensors and a small
/// chunk-file limit must produce several chunk files -- never a tensor split
/// across manifest rows.
#[test]
fn several_tensors_fill_several_chunk_files() {
    let scratch = Scratch::new("chunks");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let mut entries = Vec::new();
    let mut selection = SelectionBuilder::new("synthetic");
    for i in 0..4 {
        let name = format!("model.layers.{i}.norm.weight");
        let payload: Vec<u8> = (0..512u16)
            .flat_map(|v| (v & 0x7F7F).to_le_bytes())
            .collect();
        entries.push(Entry::new(&name, "BF16", vec![512], payload));
        selection = selection.bf16(&name, &name, "shard-a.safetensors");
    }
    write_shard(&src.join("shard-a.safetensors"), &entries);
    let selection_path = scratch.join("selection.toml");
    selection.write(&selection_path);
    let out = scratch.join("artifact");
    // 1600 bytes per shard holds one 1024-byte tensor plus the header entry
    // its long name needs, and not two: a component never spans shards.
    let result = run(&[
        "repack",
        "--selection",
        selection_path.to_str().expect("a path"),
        "--source-root",
        src.to_str().expect("a path"),
        "--out",
        out.to_str().expect("a path"),
        "--total-bytes",
        "128MiB",
        "--header-bytes",
        "64MiB",
        "--scratch-bytes",
        "1MiB",
        "--chunk-file-bytes",
        "1600",
        "--disk-bytes",
        "64MiB",
    ]);
    assert_eq!(result.outcome(), "published", "{}", result.stdout);
    let names: Vec<String> = std::fs::read_dir(&out)
        .expect("the artifact")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    assert_eq!(names.len(), 4, "one shard per tensor: {names:?}");
    let verify = run(&[
        "verify",
        "--artifact",
        out.to_str().expect("a path"),
        "--scratch-bytes",
        "4096",
    ]);
    assert_eq!(verify.outcome(), "verified", "{}", verify.stdout);
}

/// A tensor larger than the payload scratch is streamed in several units --
/// the case the budget exists for.
#[test]
fn a_tensor_larger_than_the_scratch_streams_in_several_units() {
    let scratch = Scratch::new("streaming");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let name = "model.big.weight";
    // 128 KiB of BF16 against a 16 KiB scratch: eight units of eight KiB, two
    // tiles at a time.
    let payload: Vec<u8> = (0..65_536u32)
        .flat_map(|v| ((v & 0x7F7F) as u16).to_le_bytes())
        .collect();
    write_shard(
        &src.join("shard-a.safetensors"),
        &[Entry::new(name, "BF16", vec![65_536], payload.clone())],
    );
    let selection = SelectionBuilder::new("synthetic").bf16(name, name, "shard-a.safetensors");
    let selection_path = scratch.join("selection.toml");
    selection.write(&selection_path);
    let out = scratch.join("artifact");
    let result = run(&[
        "repack",
        "--selection",
        selection_path.to_str().expect("a path"),
        "--source-root",
        src.to_str().expect("a path"),
        "--out",
        out.to_str().expect("a path"),
        "--total-bytes",
        "128MiB",
        "--header-bytes",
        "64MiB",
        "--scratch-bytes",
        "16KiB",
        "--chunk-file-bytes",
        "4MiB",
        "--disk-bytes",
        "64MiB",
    ]);
    assert_eq!(result.outcome(), "published", "{}", result.stdout);
    assert_eq!(
        result.field("units-written"),
        Some("16"),
        "131072 bytes in 8 KiB tiles: {}",
        result.stdout
    );
    let artifact = moxie_storage::Artifact::open(&out).expect("the artifact opens");
    let mut streamed = Vec::new();
    let mut scratch = vec![0u8; 8192];
    artifact
        .stream_tensor(name, &mut scratch, &mut |s| {
            streamed.extend_from_slice(s);
            Ok(())
        })
        .expect("it verifies");
    assert_eq!(streamed, payload, "the streamed bytes are the source bytes");
}
