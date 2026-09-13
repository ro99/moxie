//! Task 0024: import real **asymmetric** INT4 `pack-quantized` tensors, whose
//! zero points are packed along the output axis.
//!
//! **Read-only, and nothing here is model support.** This imports tensors from
//! two inventoried local artifacts into canonical affine form. It executes
//! nothing, writes nothing, and proves no quality claim: a tensor that decodes
//! to the source's own arithmetic says the reader agrees with the file, not
//! that the model works. O2 is repack-only and needs paired output against the
//! released model; O5 governs bulk writes and this makes none -- `File::open`
//! and positioned reads only, under the roots
//! [artifact-roots](../../../docs/evidence/artifact-roots.md) designates for
//! inspection.
//!
//! Both artifacts skip with a printed message when absent, so a fresh clone is
//! green and a missing checkpoint is never mistaken for a regression.
//!
//! **A module's four tensors need not live in one shard.** Laguna keeps them
//! together; Qwen3.8 puts every module's codes and zero points in shards 1--2
//! and every scale somewhere else, so **none** of its 256 modules is complete in
//! any single shard. `source_entries` takes one header by design and cannot see
//! across that split; this file resolves each tensor to its own shard, which is
//! index work that M3 item 1's manifest owns. Recorded rather than built here.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use moxie_format::affine::{IntWidth, ZeroPoints};
use moxie_format::compressed_tensors::{
    Granularity, PackQuantizedSpec, PackedZeroPoints, SourceTensors, ZeroPointSource,
    decode_weight_shape, import, scale_dtype_of, source_entries,
};
use moxie_format::safetensors::Dtype;
use moxie_format::scale::ScaleDtype;
use moxie_storage::{HeaderBudget, Shard};

/// 64 MiB of estimated header peak.
///
/// Stated rather than defaulted, and the reason is a fact about these
/// artifacts: Laguna's index holds **140,989** tensors, so one shard's header
/// is about a megabyte serialized and its admitted peak estimate is 16.5 MB --
/// above the 8 MiB default, whose own docstring is calibrated on Gemma 4's
/// 17--64 KB headers. The default refusing it is the budget working. Raising it
/// here is a declared choice for a read-only inspection, not a change to the
/// default.
fn header_budget() -> HeaderBudget {
    HeaderBudget::new(64 << 20).expect("a valid header budget")
}

/// One artifact, with the facts its bring-up record already states.
///
/// `root` is owned rather than `&'static str` so that a **synthetic** artifact
/// in a temporary directory is the same kind of thing as a real one. The second
/// review's finding 2 is why: the regression written for the inventory bug
/// tested the audit's helper directly and never built an inventory, so
/// restoring the original premature filter left all five artifact tests
/// passing. A negative fixture has to enter through the same door the defect
/// was behind.
struct Source {
    name: String,
    root: PathBuf,
    shards: u32,
    /// The compressor version its `quantization_config` declares. Recorded
    /// because it is **not** the version of the library this reading is pinned
    /// to, which is why the reading is also measured below.
    compressor: &'static str,
}

impl Source {
    fn new(name: &str, root: impl Into<PathBuf>, shards: u32, compressor: &'static str) -> Self {
        Self {
            name: name.to_string(),
            root: root.into(),
            shards,
            compressor,
        }
    }
}

fn laguna() -> Source {
    Source::new(
        "Laguna-S-2.1-AWQ-INT4",
        "/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4",
        15,
        "0.1.dev534+gb269f2e",
    )
}

fn qwen() -> Source {
    Source::new(
        "Qwen3.8-27B-AWQ-BF16-INT4",
        "/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4",
        6,
        "0.1.dev535+gdc9611a",
    )
}

fn sources() -> [Source; 2] {
    [laguna(), qwen()]
}

/// The packing every one of these artifacts declares: asymmetric INT4 at group
/// 32, with the zero point packed along the output axis.
const SPEC: PackQuantizedSpec = PackQuantizedSpec {
    width: IntWidth::Int4,
    granularity: Granularity::Group { size: 32 },
    zero_points: ZeroPointSource::PackedAlongOutput,
};

const SUFFIXES: [&str; 4] = [
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
];

impl Source {
    fn shard_path(&self, n: u32) -> PathBuf {
        self.root
            .join(format!("model-{:05}-of-{:05}.safetensors", n, self.shards))
    }

    fn present(&self) -> bool {
        self.shard_path(1).exists()
    }

    fn skip(&self, what: &str) {
        eprintln!(
            "SKIP {what} for {}: {} is not present",
            self.name,
            self.root.display()
        );
    }
}

/// Where each of an artifact's `pack-quantized` tensors actually lives.
///
/// **Every discovered packed module is kept.** An earlier version filtered the
/// incomplete ones out of `modules` *before* the audit that is supposed to find
/// them, so the audit could not fail: an independent review built a shard with
/// four packed modules, one missing its `weight_zero_point`, and the inventory
/// test passed while reporting three. A filter applied to the population being
/// audited removes exactly the rows the audit exists to see -- which is task
/// 0023's omitted layer, in a different file.
struct Inventory<'a> {
    source: &'a Source,
    /// Tensor name -> (shard number, dtype, declared shape).
    located: BTreeMap<String, (u32, Dtype, Vec<u64>)>,
    /// **Every** module carrying `weight_packed`, smallest payload first,
    /// complete or not.
    modules: Vec<(String, u64)>,
    /// Modules whose four tensors are all in one shard, by shard number.
    whole: BTreeMap<u32, Vec<String>>,
    /// One opened shard, kept so a run of reads from the same shard parses its
    /// header once.
    open: RefCell<Option<(u32, Shard)>>,
}

/// A module missing one or more of the four tensors a `pack-quantized` weight
/// is serialized as, anywhere in the artifact.
#[derive(Debug, PartialEq, Eq)]
struct Incomplete {
    module: String,
    missing: Vec<&'static str>,
}

/// Which of the four companions a module lacks, across the whole artifact.
///
/// A pure function of the located names, so the audit it feeds can be tested
/// against a table that **contains** a missing companion -- the case no real
/// artifact here supplies, and the case the check exists for.
fn missing_companions<'n>(module: &str, located: impl Fn(&str) -> bool + 'n) -> Vec<&'static str> {
    SUFFIXES
        .iter()
        .copied()
        .filter(|s| !located(&format!("{module}.{s}")))
        .collect()
}

impl<'a> Inventory<'a> {
    fn build(source: &'a Source) -> Self {
        let mut located = BTreeMap::new();
        let mut sizes: BTreeMap<String, u64> = BTreeMap::new();
        let mut per_shard: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for n in 1..=source.shards {
            let path = source.shard_path(n);
            let shard =
                Shard::open_with_limits(&path, Default::default(), header_budget()).expect("shard");
            let names: Vec<&String> = shard.header().tensors().keys().collect();
            let present: std::collections::BTreeSet<&str> =
                names.iter().map(|s| s.as_str()).collect();
            for (name, entry) in shard.header().tensors() {
                let Some((module, suffix)) = name.rsplit_once('.') else {
                    continue;
                };
                if !SUFFIXES.contains(&suffix) {
                    continue;
                }
                if suffix == "weight_packed" {
                    sizes.insert(module.to_string(), entry.len());
                    if SUFFIXES
                        .iter()
                        .all(|s| present.contains(format!("{module}.{s}").as_str()))
                    {
                        per_shard.entry(n).or_default().push(module.to_string());
                    }
                }
                located.insert(name.clone(), (n, entry.dtype, entry.shape.clone()));
            }
        }
        // Every packed module, complete or not. The audit below is what
        // decides whether an incomplete one is acceptable; this list is its
        // population and may not pre-empt it.
        let mut modules: Vec<(String, u64)> = sizes.into_iter().collect();
        // By size, then by name: a deterministic choice, not whatever the map
        // happened to yield.
        modules.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Self {
            source,
            located,
            modules,
            whole: per_shard,
            open: RefCell::new(None),
        }
    }

    /// Every module lacking one of the four tensors, anywhere in the artifact.
    fn incomplete(&self) -> Vec<Incomplete> {
        self.modules
            .iter()
            .filter_map(|(module, _)| {
                let missing = missing_companions(module, |n| self.located.contains_key(n));
                (!missing.is_empty()).then(|| Incomplete {
                    module: module.clone(),
                    missing,
                })
            })
            .collect()
    }

    /// The modules that can actually be imported: all four tensors located.
    fn importable(&self) -> Vec<&(String, u64)> {
        self.modules
            .iter()
            .filter(|(module, _)| {
                missing_companions(module, |n| self.located.contains_key(n)).is_empty()
            })
            .collect()
    }

    fn entry(&self, name: &str) -> &(u32, Dtype, Vec<u64>) {
        self.located
            .get(name)
            .unwrap_or_else(|| panic!("{name} is in no shard of {}", self.source.name))
    }

    fn read(&self, name: &str) -> Vec<u8> {
        let (n, _, _) = *self.entry(name);
        let mut open = self.open.borrow_mut();
        if open.as_ref().map(|(at, _)| *at) != Some(n) {
            let path = self.source.shard_path(n);
            let shard =
                Shard::open_with_limits(&path, Default::default(), header_budget()).expect("shard");
            *open = Some((n, shard));
        }
        let (_, shard) = open.as_ref().expect("just opened");
        shard.tensor_bytes(name).expect("a declared tensor")
    }
}

/// Everything one module's import needs, read across whatever shards hold it.
struct Payloads {
    packed: Vec<u8>,
    packed_shape: Vec<u64>,
    scale: Vec<u8>,
    scale_shape: Vec<u64>,
    scale_dtype: ScaleDtype,
    zero_point: Vec<u8>,
    zero_point_shape: Vec<u64>,
    logical: (usize, usize),
}

impl Payloads {
    fn read(inventory: &Inventory<'_>, module: &str) -> Self {
        let (_, zp_dtype, zero_point_shape) =
            inventory.entry(&format!("{module}.weight_zero_point"));
        assert_eq!(*zp_dtype, Dtype::I32, "{module} zero-point dtype");
        let (_, packed_dtype, packed_shape) = inventory.entry(&format!("{module}.weight_packed"));
        assert_eq!(*packed_dtype, Dtype::I32, "{module} packed dtype");
        let (_, scale_dtype, scale_shape) = inventory.entry(&format!("{module}.weight_scale"));
        let scale_dtype = scale_dtype_of(*scale_dtype).expect("a canonical scale dtype");
        let shape_bytes = inventory.read(&format!("{module}.weight_shape"));
        let logical = decode_weight_shape(&shape_bytes).expect("a logical shape");
        Self {
            packed_shape: packed_shape.clone(),
            scale_shape: scale_shape.clone(),
            zero_point_shape: zero_point_shape.clone(),
            scale_dtype,
            packed: inventory.read(&format!("{module}.weight_packed")),
            scale: inventory.read(&format!("{module}.weight_scale")),
            zero_point: inventory.read(&format!("{module}.weight_zero_point")),
            logical,
        }
    }

    fn tensors(&self) -> SourceTensors<'_> {
        SourceTensors {
            packed: &self.packed,
            packed_shape: &self.packed_shape,
            scale: &self.scale,
            scale_shape: &self.scale_shape,
            scale_dtype: self.scale_dtype,
            zero_point: Some(PackedZeroPoints {
                payload: &self.zero_point,
                shape: &self.zero_point_shape,
            }),
            logical: self.logical,
        }
    }

    fn groups(&self) -> usize {
        self.logical.1.div_ceil(32)
    }

    /// Unpack the zero points the way the pinned library's `decompress` does:
    /// `unpacked[l::pack_factor, :] = value >> (bits * l)`, built as a matrix
    /// rather than indexed by a closed form.
    ///
    /// Procedurally different from [`import`]'s own `(o / per_word) * groups + g`,
    /// so agreement is two expressions of one convention rather than one
    /// expression compared with itself.
    fn unpack_zero_points(&self) -> Vec<Vec<i32>> {
        let (out_features, _) = self.logical;
        let groups = self.groups();
        let per_word = 8usize;
        let word_rows = out_features.div_ceil(per_word);
        let mut unpacked = vec![vec![0i32; groups]; word_rows * per_word];
        for l in 0..per_word {
            for j in 0..word_rows {
                let destination = &mut unpacked[l + per_word * j];
                for (g, slot) in destination.iter_mut().enumerate() {
                    let at = (j * groups + g) * 4;
                    let word = u32::from_le_bytes(
                        self.zero_point[at..at + 4].try_into().expect("four bytes"),
                    );
                    *slot = ((word >> (4 * l as u32)) & 0xF) as i32 - 8;
                }
            }
        }
        unpacked.truncate(out_features);
        unpacked
    }

    fn scale_at(&self, o: usize, g: usize) -> f32 {
        let index = o * self.groups() + g;
        match self.scale_dtype {
            ScaleDtype::Bf16 => {
                let bits = u16::from_le_bytes(
                    self.scale[index * 2..index * 2 + 2]
                        .try_into()
                        .expect("two bytes"),
                );
                f32::from_bits((bits as u32) << 16)
            }
            ScaleDtype::F16 => moxie_format::scale::f16_bits_to_f32(u16::from_le_bytes(
                self.scale[index * 2..index * 2 + 2]
                    .try_into()
                    .expect("two bytes"),
            )),
            ScaleDtype::F32 => f32::from_le_bytes(
                self.scale[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            ),
        }
    }

    /// The source's reconstruction of `(o, k)`, **with the rounding boundary its
    /// own reference applies**, as a BF16 bit pattern.
    ///
    /// The pinned `_dequantize` is
    /// `(x_q.to(scale.dtype) - zero_point.to(scale.dtype)) * scale`, and
    /// `scale.dtype` is BF16 for both artifacts, so the source's value is a
    /// **BF16** number. The canonical reconstruction is FP32 by document 03's
    /// contract ("scale multiplication is evaluated in FP32 in the host
    /// reconstruction oracle"), and the two are not the same quantity.
    ///
    /// Task 0024's independent review found the first version of this file
    /// claiming an FP32 comparison as agreement with "the source's own declared
    /// arithmetic" and measured **27,501 of 112,640** values where the two
    /// differ. Both are right; only one of them is what the source computes.
    ///
    /// **Why one rounding and not two.** `q - z` is an integer in `[-15, 15]`
    /// and the scale is a BF16 value; both are exact in FP32 and their product
    /// needs at most twelve significant bits, so the FP32 multiply is exact and
    /// the only rounding is the single round-to-nearest-even into BF16. That is
    /// what makes `bf16(canonical)` the source's value rather than an
    /// approximation of it.
    ///
    /// Takes the already-unpacked zero points rather than unpacking them per
    /// value: the first version called `unpack_zero_points` inside the loop and
    /// turned a per-value check into a quadratic one.
    fn source_bf16(&self, zeros: &[Vec<i32>], o: usize, k: usize) -> u16 {
        let q = self.source_code(o, k);
        let z = zeros[o][k / 32];
        moxie_format::bf16::f32_to_bf16_bits((q - z) as f32 * self.scale_at(o, k / 32))
    }

    /// The source's own code at `(o, k)`, read straight from `weight_packed`.
    fn source_code(&self, o: usize, k: usize) -> i32 {
        let packed_columns = self.logical.1.div_ceil(8);
        let at = (o * packed_columns + k / 8) * 4;
        let word = u32::from_le_bytes(self.packed[at..at + 4].try_into().expect("four bytes"));
        (((word >> ((k % 8) as u32 * 4)) & 0xF) as i32) - 8
    }
}

fn config_of(source: &Source) -> serde_json::Value {
    let text =
        std::fs::read_to_string(source.root.join("config.json")).expect("config.json is readable");
    serde_json::from_str(&text).expect("config.json parses")
}

/// The parameters the importer is handed are the parameters the artifact
/// declares -- read from its own config, not assumed for the family.
///
/// The Laguna record says why this is checked per artifact: "That is a
/// quantizer's sensitivity choice, not a format rule, and it must be read from
/// the artifact rather than assumed."
#[test]
fn the_declared_packing_parameters_match_each_artifacts_own_config() {
    let mut checked = 0usize;
    for source in &sources() {
        if !source.present() {
            source.skip("declared packing parameters");
            continue;
        }
        let c = config_of(source);
        let q = &c["quantization_config"];
        assert_eq!(q["quant_method"], "compressed-tensors", "{}", source.name);
        assert_eq!(q["format"], "pack-quantized", "{}", source.name);
        assert_eq!(
            q["version"], source.compressor,
            "{} declares a different compressor version than this test records",
            source.name
        );
        let w = &q["config_groups"]["group_0"]["weights"];
        assert_eq!(w["num_bits"].as_u64().unwrap(), 4, "{}", source.name);
        assert_eq!(w["type"], "int", "{}", source.name);
        assert!(
            !w["symmetric"].as_bool().unwrap(),
            "{} must be asymmetric for this importer path",
            source.name
        );
        assert_eq!(w["strategy"], "group", "{}", source.name);
        assert_eq!(w["group_size"].as_u64().unwrap(), 32, "{}", source.name);
        assert_eq!(w["zp_dtype"], "torch.int8", "{}", source.name);
        // Contiguous grouping, which is what the descriptor's `group_index:
        // None` claims. A permutation here would have to be carried.
        assert!(
            w["actorder"].is_null(),
            "{} declares actorder {:?}; a permutation may not be ignored",
            source.name,
            w["actorder"]
        );
        eprintln!(
            "task0024 {}: int4 asymmetric group 32, compressor {}",
            source.name, source.compressor
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("SKIPPED: no asymmetric artifact is present on this machine");
    }
}

/// Real tensors import, and every reconstructed value is **bitwise** equal to
/// the source's own declared arithmetic computed independently from the raw
/// bytes.
///
/// This is ADR 0018's "bit-identical repack `W=(Q-Z)*S`" at tensor scale. It is
/// **not** a quality claim: a faithful repack of a publisher's weights says
/// nothing about what the model produces, which is O2's and needs paired
/// output.
#[test]
fn real_asymmetric_tensors_import_and_match_the_sources_own_arithmetic() {
    let mut imported = 0usize;
    let mut bytes_read = 0u64;
    let mut values_checked = 0u64;
    // How many sampled values the source's BF16 boundary actually moves. A
    // boundary check on a sample where the boundary never fires would be
    // vacuous, so this is counted and required to be nonzero.
    let mut narrowed = 0u64;
    for source in &sources() {
        if !source.present() {
            source.skip("asymmetric import");
            continue;
        }
        let inventory = Inventory::build(source);
        assert!(
            inventory.importable().len() >= 3,
            "{} holds {} importable asymmetric module(s)",
            source.name,
            inventory.importable().len()
        );
        for (module, _) in inventory.importable().into_iter().take(3) {
            let p = Payloads::read(&inventory, module);
            let (out_features, in_features) = p.logical;
            let groups = p.groups();
            // The three shapes the artifact declared, each against the logical
            // shape rather than against a byte count.
            assert_eq!(
                p.packed_shape,
                vec![out_features as u64, in_features.div_ceil(8) as u64],
                "{module} packed shape"
            );
            assert_eq!(
                p.scale_shape,
                vec![out_features as u64, groups as u64],
                "{module} scale shape"
            );
            assert_eq!(
                p.zero_point_shape,
                vec![out_features.div_ceil(8) as u64, groups as u64],
                "{module} zero-point shape -- packed along the output axis"
            );
            bytes_read += (p.packed.len() + p.scale.len() + p.zero_point.len() + 16) as u64;

            let tensor = import(&SPEC, p.tensors()).unwrap();
            let d = tensor.descriptor();
            assert_eq!((d.out_features, d.in_features), p.logical);
            assert_eq!(d.width, IntWidth::Int4);
            let ZeroPoints::PerGroup(zeros) = tensor.zero_points() else {
                panic!("{module} imported without zero points");
            };
            assert_eq!(zeros.len(), out_features * groups);

            // The independent transcription: the library's unpack procedure,
            // then document 03's equation.
            let unpacked = p.unpack_zero_points();
            for o in 0..out_features {
                for g in 0..groups {
                    assert_eq!(
                        zeros[o * groups + g] as i32,
                        unpacked[o][g],
                        "{module} zero point at ({o},{g})"
                    );
                }
            }
            // Two comparisons, because there are two quantities and they are
            // not the same one. A deterministic sample of rows spanning the
            // whole output axis, including the last -- which is the row a
            // padded zero-point word would get wrong.
            let rows: Vec<usize> = [0, 1, out_features / 3, out_features / 2, out_features - 1]
                .into_iter()
                .filter(|o| *o < out_features)
                .collect();
            let mut low = i32::MAX;
            let mut high = i32::MIN;
            for o in rows {
                let row = tensor.reconstruct_row(o).unwrap();
                assert_eq!(row.len(), in_features);
                for (k, value) in row.iter().enumerate() {
                    let q = p.source_code(o, k);
                    let z = unpacked[o][k / 32];
                    // (a) The **canonical** equation of document 03, in FP32,
                    // over the source's own bytes. This is what `moxie-format`
                    // is contracted to reconstruct.
                    let canonical = (q - z) as f32 * p.scale_at(o, k / 32);
                    assert_eq!(
                        value.to_bits(),
                        canonical.to_bits(),
                        "{module} at ({o},{k}): {value} against the canonical {canonical}"
                    );
                    // (b) The **source's own** arithmetic, with the BF16
                    // boundary its reference applies. The canonical FP32 value
                    // must round to it exactly -- a repack that changed a code,
                    // a zero point or a scale byte could not.
                    let source = p.source_bf16(&unpacked, o, k);
                    assert_eq!(
                        moxie_format::bf16::f32_to_bf16_bits(*value),
                        source,
                        "{module} at ({o},{k}): the canonical value does not round to the \
                         source's own BF16 result"
                    );
                    if canonical.to_bits() != moxie_format::bf16::bf16_bits_to_f32(source).to_bits()
                    {
                        narrowed += 1;
                    }
                    low = low.min(q);
                    high = high.max(q);
                    values_checked += 1;
                }
            }
            assert_eq!(
                (low, high),
                (-8, 7),
                "{module} codes span [{low}, {high}], not the full INT4 range"
            );
            eprintln!(
                "task0024 imported {}::{module}: logical={:?} groups={groups} \
                 zero_point={:?} scale={}",
                source.name,
                p.logical,
                p.zero_point_shape,
                p.scale_dtype.name()
            );
            imported += 1;
        }
    }
    if imported == 0 {
        return eprintln!("SKIPPED: no asymmetric artifact is present on this machine");
    }
    eprintln!(
        "task0024 imported {imported} module(s), read {bytes_read} artifact byte(s), \
         checked {values_checked} reconstructed value(s) against the canonical FP32 \
         equation and against the source's own BF16 arithmetic; the source's rounding \
         boundary moves {narrowed} of them"
    );
    assert!(
        narrowed > 0,
        "the source's BF16 boundary never fired on {values_checked} sampled value(s), so \
         checking it proved nothing"
    );
}

/// The zero points go with the output channels the pinned mapping says they do,
/// and the artifacts' own bytes say so.
///
/// Task 0018 recorded that a *code* word's lane order cannot be checked against
/// an artifact, because all eight lanes fall inside one scale group. A
/// **zero-point** word is different: its lanes are eight different output
/// channels, whose codes have different statistics. So this mapping is
/// measurable, and it is measured rather than taken on the library's word --
/// which matters, because both artifacts declare a compressor version that is
/// not the one this reading is pinned to.
///
/// The statistic, declared before it was run: an asymmetric group's codes are
/// centred near its zero point, so `mean(q) - z` is tight under the correct
/// pairing and is two independent quantities subtracted under a wrong one. The
/// thresholds are 1.0 and 1.2 codes of mean absolute deviation.
///
/// It is corroboration of a **reading**, not a quality claim.
#[test]
fn the_pinned_zero_point_lane_assignment_is_the_one_the_artifacts_bytes_support() {
    let mut measured = 0usize;
    for source in &sources() {
        if !source.present() {
            source.skip("zero-point lane measurement");
            continue;
        }
        let inventory = Inventory::build(source);
        for (module, _) in inventory.importable().into_iter().take(2) {
            let p = Payloads::read(&inventory, module);
            let (out_features, in_features) = p.logical;
            let groups = p.groups();
            let word_rows = out_features.div_ceil(8);

            // The mean code of every (output channel, group), from the codes
            // alone -- no zero point is involved in building it.
            let mut mean_code = vec![0f64; out_features * groups];
            for o in 0..out_features {
                for k in 0..in_features {
                    mean_code[o * groups + k / 32] += p.source_code(o, k) as f64;
                }
            }
            let group_size = |g: usize| (in_features - g * 32).min(32) as f64;
            for o in 0..out_features {
                for g in 0..groups {
                    mean_code[o * groups + g] /= group_size(g);
                }
            }

            // Four assignments the same bytes and the same shape permit.
            let word = |j: usize, g: usize| -> u32 {
                let at = (j * groups + g) * 4;
                u32::from_le_bytes(p.zero_point[at..at + 4].try_into().expect("four bytes"))
            };
            /// One reading of the packed zero points: output channel and
            /// group in, the zero point it assigns out.
            type Assignment<'a> = (&'a str, Box<dyn Fn(usize, usize) -> i32 + 'a>);
            let candidates: [Assignment<'_>; 5] = [
                (
                    "pinned o = 8j + l",
                    Box::new(|o: usize, g: usize| {
                        ((word(o / 8, g) >> (4 * (o % 8) as u32)) & 0xF) as i32 - 8
                    }),
                ),
                (
                    "lane-reversed",
                    Box::new(|o: usize, g: usize| {
                        ((word(o / 8, g) >> (4 * (7 - o % 8) as u32)) & 0xF) as i32 - 8
                    }),
                ),
                (
                    "block o = j + rows*l",
                    Box::new(|o: usize, g: usize| {
                        ((word(o % word_rows, g) >> (4 * (o / word_rows) as u32)) & 0xF) as i32 - 8
                    }),
                ),
                (
                    "block reversed",
                    Box::new(|o: usize, g: usize| {
                        ((word(o % word_rows, g) >> (4 * (7 - o / word_rows) as u32)) & 0xF) as i32
                            - 8
                    }),
                ),
                // The sign. `mean_code - (-z)` is `mean_code + z`, so the
                // pinned assignment negated measures whether the source adds
                // its zero point instead of subtracting it.
                //
                // The first version of this record claimed the statistic could
                // not separate the two, on the grounds that a distribution
                // centred near zero is symmetric. That reasoning ignores the
                // correlation between a group's code mean and its own zero
                // point -- which is the entire premise of this measurement, two
                // paragraphs earlier. An independent review measured the
                // difference and it is large.
                (
                    "sign-flipped z -> -z",
                    Box::new(|o: usize, g: usize| {
                        -(((word(o / 8, g) >> (4 * (o % 8) as u32)) & 0xF) as i32 - 8)
                    }),
                ),
            ];

            let mut deviations = Vec::new();
            for (name, assign) in &candidates {
                let mut total = 0f64;
                let mut n = 0u64;
                for o in 0..out_features {
                    for g in 0..groups {
                        total += (mean_code[o * groups + g] - assign(o, g) as f64).abs();
                        n += 1;
                    }
                }
                deviations.push((*name, total / n as f64));
            }
            eprintln!(
                "task0024 {}::{module} zero-point assignment mean|mean_code - z|: {:?}",
                source.name, deviations
            );
            let pinned = deviations[0].1;
            assert!(
                pinned < 1.0,
                "{}::{module}: the pinned assignment deviates by {pinned:.4}, above the \
                 declared 1.0 threshold",
                source.name
            );
            for (name, d) in &deviations[1..] {
                assert!(
                    *d > 1.2,
                    "{}::{module}: the alternative '{name}' deviates by {d:.4}, below the \
                     declared 1.2 threshold -- this measurement does not separate the two \
                     readings and cannot corroborate either",
                    source.name
                );
                assert!(
                    *d > pinned * 2.0,
                    "{}::{module}: '{name}' at {d:.4} is not clearly worse than the pinned \
                     {pinned:.4}",
                    source.name
                );
            }
            measured += 1;
        }
    }
    if measured == 0 {
        eprintln!("SKIPPED: no asymmetric artifact is present on this machine");
    }
}

/// Every asymmetric module of every shard declares the packed zero-point shape,
/// and the counts are printed rather than assumed.
///
/// This is the inventory the tests above sample three modules from: it checks
/// the declared shapes of every module in the artifact. It also exercises
/// `source_entries` on the modules that are **whole within one shard**, and
/// reports how many are not -- "skipped silently" and "there were none" are
/// different facts, and Qwen3.8 is entirely the second.
#[test]
fn every_asymmetric_module_declares_a_packed_zero_point() {
    let mut inspected = 0usize;
    for source in &sources() {
        if !source.present() {
            source.skip("zero-point shape inventory");
            continue;
        }
        let inventory = Inventory::build(source);
        // The audit's own population: every packed module, before any filter.
        // A module missing a companion is a finding, not a row to drop.
        let incomplete = inventory.incomplete();
        assert!(
            incomplete.is_empty(),
            "{}: {} packed module(s) lack a companion tensor: {:?}",
            source.name,
            incomplete.len(),
            &incomplete[..incomplete.len().min(5)]
        );
        assert_eq!(
            inventory.importable().len(),
            inventory.modules.len(),
            "{}: the audited population and the importable one must be the same              set once no module is incomplete",
            source.name
        );
        for (module, _) in &inventory.modules {
            let shape = |suffix: &str| &inventory.entry(&format!("{module}.{suffix}")).2;
            let packed = shape("weight_packed");
            let scale = shape("weight_scale");
            let zp = shape("weight_zero_point");
            let (out_features, groups) = (packed[0], scale[1]);
            assert_eq!(
                *zp,
                vec![out_features.div_ceil(8), groups],
                "{module} zero-point shape"
            );
            assert_eq!(scale[0], out_features, "{module} scale rows");
            // Eight INT4 codes per word along the input axis, one scale per 32
            // input channels: the two axes are consistent with each other,
            // which is what makes the two packing conventions separable at all.
            assert_eq!(
                packed[1] * 8,
                groups * 32,
                "{module} packed columns and scale groups disagree"
            );
        }
        // And the production entry helper, on the modules it can see: all four
        // tensors in one header, with the declared serialization agreeing with
        // the index in both directions.
        let mut whole = 0usize;
        for (n, modules) in &inventory.whole {
            let path = source.shard_path(*n);
            let shard =
                Shard::open_with_limits(&path, Default::default(), header_budget()).unwrap();
            for module in modules {
                let e = source_entries(shard.header(), module, ZeroPointSource::PackedAlongOutput)
                    .unwrap_or_else(|e| panic!("{module}: {e}"));
                assert!(e.zero_point.is_some(), "{module}");
                assert!(
                    source_entries(shard.header(), module, ZeroPointSource::Symmetric).is_err(),
                    "{module} must not read as symmetric"
                );
                whole += 1;
            }
        }
        assert!(
            !inventory.modules.is_empty(),
            "{} holds no complete packed module",
            source.name
        );
        eprintln!(
            "task0024 {}: {} asymmetric module(s), every one packing its zero points along \
             the output axis; {whole} of them have all four tensors in one shard",
            source.name,
            inventory.modules.len()
        );
        inspected += 1;
    }
    if inspected == 0 {
        eprintln!("SKIPPED: no asymmetric artifact is present on this machine");
    }
}

/// The inventory audit can actually fail — **through `Inventory::build`**.
///
/// The second review's finding 2. The first version of this regression called
/// `missing_companions` directly, so it could not see the defect it was written
/// for: the bug was a filter applied *before* that helper ran, and restoring it
/// left all five artifact tests passing. **A correct helper cannot detect a
/// module that was removed before the helper was called.**
///
/// This builds a real four-module shard with one module's `weight_zero_point`
/// omitted, constructs the inventory over it exactly as the artifact tests do,
/// and asserts both halves the filter would break: the independent
/// packed-module population, and the module the audit must name.
#[test]
fn the_inventory_reports_a_module_missing_a_companion_rather_than_dropping_it() {
    let dir = scratch_dir("inventory-negative");
    write_synthetic_shard(&dir.join("model-00001-of-00001.safetensors"), 4, &["m3"]);
    let source = Source::new("synthetic", &dir, 1, "n/a");
    let inventory = Inventory::build(&source);

    // The population, counted independently of the audit: four modules carry
    // `weight_packed`, and a filter that drops the incomplete one shows up here
    // first.
    assert_eq!(
        inventory.modules.len(),
        4,
        "every packed module must be in the audit's population, complete or not"
    );
    // And the audit names the one that is incomplete.
    assert_eq!(
        inventory.incomplete(),
        vec![Incomplete {
            module: "m3".to_string(),
            missing: vec!["weight_zero_point"],
        }]
    );
    // The other three are importable; the incomplete one is not.
    assert_eq!(inventory.importable().len(), 3);
    assert!(
        !inventory
            .importable()
            .iter()
            .any(|(module, _)| module == "m3")
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// And the audit's decision, over a table that includes cases no artifact here
/// supplies. Complements the fixture above rather than standing in for it.
#[test]
fn missing_companions_names_every_absent_tensor() {
    let present: std::collections::BTreeSet<&str> = [
        "m0.weight_packed",
        "m0.weight_scale",
        "m0.weight_shape",
        "m0.weight_zero_point",
        "m1.weight_packed",
        "m1.weight_scale",
        "m1.weight_shape",
        // m1 has no zero point.
        "m2.weight_packed",
        "m2.weight_zero_point",
        // m2 has neither scale nor shape.
    ]
    .into_iter()
    .collect();
    let located = |n: &str| present.contains(n);

    assert!(missing_companions("m0", located).is_empty());
    assert_eq!(missing_companions("m1", located), vec!["weight_zero_point"]);
    assert_eq!(
        missing_companions("m2", located),
        vec!["weight_scale", "weight_shape"]
    );
    assert_eq!(
        missing_companions("absent", located),
        SUFFIXES.to_vec(),
        "a module with nothing at all names every companion"
    );
}

/// A scratch directory under the system temp root, named for this test.
fn scratch_dir(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "moxie-task0024-{what}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("a clock after 1970")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// A safetensors shard holding `modules` valid asymmetric INT4 modules, with
/// the named ones missing their `weight_zero_point`.
///
/// Logical `[8, 32]`: packed `[8, 4]` I32, one group so the scale is `[8, 1]`,
/// and `ceil(8 / 8) = 1` zero-point word row, `[1, 1]`. Real shapes, so the
/// inventory's own consistency checks pass and the only thing wrong with the
/// fixture is the tensor it is missing.
fn write_synthetic_shard(path: &Path, modules: usize, without_zero_point: &[&str]) {
    struct Tensor {
        name: String,
        dtype: &'static str,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }
    let mut tensors = Vec::new();
    for m in 0..modules {
        let module = format!("m{m}");
        tensors.push(Tensor {
            name: format!("{module}.weight_packed"),
            dtype: "I32",
            shape: vec![8, 4],
            bytes: vec![0x21u8; 8 * 4 * 4],
        });
        tensors.push(Tensor {
            name: format!("{module}.weight_scale"),
            dtype: "BF16",
            shape: vec![8, 1],
            bytes: (0..8)
                .flat_map(|_| ((1.0f32.to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
        });
        let mut shape_bytes = 8i64.to_le_bytes().to_vec();
        shape_bytes.extend_from_slice(&32i64.to_le_bytes());
        tensors.push(Tensor {
            name: format!("{module}.weight_shape"),
            dtype: "I64",
            shape: vec![2],
            bytes: shape_bytes,
        });
        if !without_zero_point.contains(&module.as_str()) {
            tensors.push(Tensor {
                name: format!("{module}.weight_zero_point"),
                dtype: "I32",
                shape: vec![1, 1],
                bytes: vec![0x33u8; 4],
            });
        }
    }
    // Safetensors requires the header's names in any order but the offsets to
    // cover the payload exactly; lay them out in declaration order.
    let mut json = String::from("{");
    let mut at = 0u64;
    let mut payload = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let end = at + t.bytes.len() as u64;
        json.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{at},{end}]}}",
            t.name, t.dtype, t.shape
        ));
        payload.extend_from_slice(&t.bytes);
        at = end;
    }
    json.push('}');
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(&payload);
    std::fs::write(path, out).expect("a synthetic shard");
}
