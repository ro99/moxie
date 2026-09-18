//! Synthetic source-codec integration through the existing bounded publisher.
mod common;
use common::{Entry, Scratch, SelectionBuilder, budgets, run, write_shard};

#[test]
fn gptq_tiles_publish_exact_bytes_and_preserve_noncontiguous_groups() {
    for (bits, group, inputs, mapped) in [(4usize, 32usize, 259usize, true), (8, 128, 257, false)] {
        let outputs = 11usize;
        let pack = 32 / bits;
        let groups = inputs.div_ceil(group);
        let mask = (1u32 << bits) - 1;
        let bias = 1i32 << (bits - 1);
        let scratch = Scratch::new("gptq-publication");
        let source = scratch.join("source");
        std::fs::create_dir(&source).unwrap();
        let mut weights = vec![0u32; inputs.div_ceil(pack) * outputs];
        let mut zeros = vec![0u32; groups * outputs.div_ceil(pack)];
        let mut scales = Vec::new();
        let code = |o: usize, k: usize| ((o * 17 + k) as u32) & mask;
        let zero = |o: usize, g: usize| if (o + g).is_multiple_of(2) { mask } else { 0 };
        let scale = |o: usize, g: usize| {
            (((0.5f32 + (o + g) as f32) * if (o + g).is_multiple_of(2) { 1.0 } else { -1.0 })
                .to_bits()
                >> 16) as u16
        };
        for o in 0..outputs {
            for k in 0..inputs {
                weights[k / pack * outputs + o] |= code(o, k) << ((k % pack) * bits);
            }
        }
        for g in 0..groups {
            for o in 0..outputs {
                zeros[g * outputs.div_ceil(pack) + o / pack] |= zero(o, g) << ((o % pack) * bits);
                scales.extend_from_slice(&scale(o, g).to_le_bytes());
            }
        }
        let mut entries = vec![
            Entry::new(
                "m.qweight",
                "I32",
                vec![inputs.div_ceil(pack) as u64, outputs as u64],
                weights.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            Entry::new(
                "m.qzeros",
                "I32",
                vec![groups as u64, outputs.div_ceil(pack) as u64],
                zeros.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            Entry::new(
                "m.scales",
                "BF16",
                vec![groups as u64, outputs as u64],
                scales,
            ),
        ];
        let map: Vec<u32> = (0..inputs)
            .map(|k| (groups - 1 - k / group) as u32)
            .collect();
        if mapped {
            entries.push(Entry::new(
                "m.g_idx",
                "I32",
                vec![inputs as u64],
                map.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ));
        }
        write_shard(&source.join("source.safetensors"), &entries);
        let mut selection = SelectionBuilder::new("gptq fixture").text();
        selection.push_str(&format!("\n[[tensor]]\nrole = \"m.weight\"\nkind = \"gptq-v1\"\nmodule = \"m\"\nwidth = \"int{bits}\"\ngroup = {group}\nshape = [{outputs},{inputs}]\nrequires_group_index = {mapped}\n[tensor.files]\nqweight = \"source.safetensors\"\nqzeros = \"source.safetensors\"\nscales = \"source.safetensors\"\n"));
        if mapped {
            selection.push_str("g_idx = \"source.safetensors\"\n");
        }
        let selected = scratch.join("selection.toml");
        std::fs::write(&selected, selection).unwrap();
        let output = scratch.join("artifact");
        let mut args = vec![
            "repack",
            "--selection",
            selected.to_str().unwrap(),
            "--source-root",
            source.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ];
        args.extend(budgets());
        // Small source/destination tiles require many independent gathers.
        args.extend(["--scratch-bytes", "1024"]);
        let result = run(&args);
        assert_eq!(result.status, 0, "{}{}", result.stdout, result.stderr);
        let artifact = moxie_storage::Artifact::open(&output).unwrap();
        let mut actual = Vec::new();
        artifact
            .stream_tensor("m.weight", &mut [0; 128], &mut |part| {
                actual.extend_from_slice(part);
                Ok(())
            })
            .unwrap();
        let row_bytes = inputs.div_ceil(8 / bits);
        let mut expected = vec![0u8; outputs * row_bytes];
        for o in 0..outputs {
            for k in 0..inputs {
                let signed = (code(o, k) as i32 - bias) as u8;
                if bits == 4 {
                    expected[o * row_bytes + k / 2] |= (signed & 15) << ((k % 2) * 4);
                } else {
                    expected[o * row_bytes + k] = signed;
                }
            }
        }
        for o in 0..outputs {
            for g in 0..groups {
                expected.extend_from_slice(&scale(o, g).to_le_bytes());
            }
        }
        for o in 0..outputs {
            for g in 0..groups {
                expected.extend_from_slice(&((zero(o, g) as i32 + 1 - bias) as i16).to_le_bytes());
            }
        }
        assert_eq!(actual, expected);
        let tensor = artifact
            .manifest()
            .tensors
            .iter()
            .find(|t| t.role == "m.weight")
            .unwrap();
        assert_eq!(
            tensor.affine.as_ref().unwrap().group_index.as_ref(),
            mapped.then_some(&map)
        );
    }
}

#[test]
fn autoround_generated_plan_keeps_passthrough_and_checks_overrides() {
    let scratch = Scratch::new("autoround-plan");
    let source = scratch.join("source");
    std::fs::create_dir(&source).unwrap();
    write_shard(
        &source.join("model.safetensors"),
        &[
            Entry::new(
                "m.qweight",
                "I32",
                vec![8, 8],
                0x76543210u32.to_le_bytes().repeat(64),
            ),
            Entry::new(
                "m.qzeros",
                "I32",
                vec![2, 1],
                0x77777777u32.to_le_bytes().repeat(2),
            ),
            Entry::new(
                "m.scales",
                "F16",
                vec![2, 8],
                0x3c00u16.to_le_bytes().repeat(16),
            ),
            Entry::new(
                "m.g_idx",
                "I32",
                vec![64],
                (0..64u32)
                    .map(|k| 1 - k / 32)
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            ),
            Entry::new(
                "norm.weight",
                "BF16",
                vec![1],
                0x3f80u16.to_le_bytes().to_vec(),
            ),
            Entry::new(
                "mtp.head.weight",
                "BF16",
                vec![1],
                0x4000u16.to_le_bytes().to_vec(),
            ),
        ],
    );
    let config = r#"{"model_type":"fixture","quantization_config":{"quant_method":"auto-round","packing_format":"auto_round:auto_gptq","autoround_version":"0.15.0","bits":4,"group_size":32,"sym":true,"desc_act":true,"data_type":"int","extra_config":{".*norm.*":{"bits":16,"data_type":"float"},".*mtp.*":{"bits":16,"data_type":"fp"}}}}"#;
    std::fs::write(source.join("config.json"), config).unwrap();
    std::fs::write(source.join("model.safetensors.index.json"),r#"{"weight_map":{"m.qweight":"model.safetensors","m.qzeros":"model.safetensors","m.scales":"model.safetensors","m.g_idx":"model.safetensors","norm.weight":"model.safetensors","mtp.head.weight":"model.safetensors"}}"#).unwrap();
    let plan = scratch.join("plan.toml");
    let output = scratch.join("artifact");
    let planned = run(&[
        "plan",
        "--source-root",
        source.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(planned.status, 0, "{}{}", planned.stdout, planned.stderr);
    let text = std::fs::read_to_string(&plan).unwrap();
    assert!(text.contains("[gptq]"));
    assert!(text.contains("norm.weight"));
    assert!(text.contains("mtp.head.weight"));
    assert!(text.contains("g_idx"));
    let published = run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        output.to_str().unwrap(),
    ]);
    assert_eq!(
        published.status, 0,
        "{}{}",
        published.stdout, published.stderr
    );
    let artifact = moxie_storage::Artifact::open(&output).unwrap();
    let mut norm = [0u8; 2];
    artifact.read_tensor("norm.weight", &mut norm).unwrap();
    assert_eq!(norm, 0x3f80u16.to_le_bytes());
    let mut mtp = [0u8; 2];
    artifact.read_tensor("mtp.head.weight", &mut mtp).unwrap();
    assert_eq!(mtp, 0x4000u16.to_le_bytes());
    let mut raw = Vec::new();
    artifact
        .stream_tensor("m.weight", &mut [0; 128], &mut |part| {
            raw.extend_from_slice(part);
            Ok(())
        })
        .unwrap();
    let descriptor = moxie_format::affine::AffineDescriptor {
        width: moxie_format::affine::IntWidth::Int4,
        out_features: 8,
        in_features: 64,
        grouping: moxie_format::affine::Grouping::Contiguous { size: 32 },
        group_index: Some((0..64u32).map(|k| 1 - k / 32).collect()),
        scale_dtype: moxie_format::scale::ScaleDtype::F16,
    };
    let tensor = moxie_format::payload::decode(
        descriptor,
        moxie_format::payload::ZeroPointSection::PerGroup,
        &raw,
    )
    .unwrap();
    for row in 0..8 {
        for (k, value) in tensor.reconstruct_row(row).unwrap().iter().enumerate() {
            assert_eq!(*value, (k % 8) as f32 - 8.0);
        }
    }
    std::fs::write(
        source.join("config.json"),
        config.replace(".*norm.*", "^m$"),
    )
    .unwrap();
    let refused = run(&[
        "plan",
        "--source-root",
        source.to_str().unwrap(),
        "--out-plan",
        scratch.join("invalid.toml").to_str().unwrap(),
    ]);
    assert_ne!(refused.status, 0);
    assert!(format!("{}{}", refused.stdout, refused.stderr).contains("passthrough override"));
}
