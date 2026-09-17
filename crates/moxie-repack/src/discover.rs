//! Building a selection from a checkpoint, so nobody has to write one by hand.
//!
//! A selection is what makes a conversion **explicit**: it names every tensor,
//! and nothing is discovered, expanded from a pattern or inferred from a
//! filename ([ADR 0020]). That property is worth keeping. What did not follow,
//! and what this module corrects, is that a *person* should have to author the
//! file. Every fact in it is already in the checkpoint -- the packing
//! parameters in `quantization_config`, which modules are not quantized in its
//! `ignore` list, and which shard holds which tensor in the shard headers.
//!
//! So the file stops being an input somebody types and becomes an **output**
//! they can read, edit and keep. What is generated is reported before anything
//! is converted, and skipped tensors are reported with the reason they were
//! skipped -- a selection that silently omitted something would be worse than
//! one nobody could write.
//!
//! [ADR 0020]: ../../../docs/decisions/adr/0020-user-managed-storage-and-canonical-materialization.md

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use moxie_format::checkpoint_config::{self, CheckpointDeclaration};
use moxie_format::compressed_tensors::{Granularity, PackQuantizedSpec, ZeroPointSource};
use moxie_format::safetensors::Dtype;
use moxie_types::Result;

use crate::source::{Sources, invalid};

/// The four tensors a `pack-quantized` module is made of.
const MODULE_SUFFIXES: [&str; 4] = [
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
];

/// What a generated selection covers, and what it does not.
#[derive(Debug, Clone)]
pub struct Discovery {
    /// The checkpoint's own declaration of itself.
    pub checkpoint: CheckpointDeclaration,
    /// Quantized modules, by module prefix, with the shard each tensor is in.
    pub modules: Vec<(String, BTreeMap<String, String>)>,
    /// GPTQ logical shapes derived from the declared packed input extent.
    pub gptq_shapes: BTreeMap<String, [usize; 2]>,
    /// BF16 tensors, as `(name, file)`.
    pub bf16: Vec<(String, String)>,
    /// What was found and left out, each with the reason.
    pub skipped: Vec<(String, String)>,
    /// The shard files that were read.
    pub files: Vec<String>,
    /// How many tensors the index names. The denominator of every completeness
    /// claim this makes.
    pub index_tensors: usize,
    /// Tensors the index names that no module claimed and that are not BF16.
    pub unaccounted: Vec<String>,
    /// The largest serialized shard header, which is what the header budget has
    /// to admit the parse of.
    pub max_header_bytes: u64,
    /// Bytes of source payload the selected tensors occupy, which bounds what
    /// the conversion writes.
    pub source_payload_bytes: u64,
    /// Source payload bytes of each selected tensor, largest first. Work units
    /// are cut inside a tensor and never across two, so this -- not the total
    /// above -- is what says how many units a tile size implies.
    pub component_bytes: Vec<u64>,
    /// SHA-256 of the **exact** `config.json` bytes this discovery parsed.
    ///
    /// Recorded here rather than re-read when the plan is written: a second
    /// read is a second moment, and a checkpoint that changed in between would
    /// be described by a plan built from the first read and bound to the
    /// second. Independent review reproduced exactly that.
    pub config_sha256: String,
    /// SHA-256 of the **exact** index bytes this discovery parsed, for the same
    /// reason.
    pub index_sha256: String,
}

impl Discovery {
    /// How many tensors a plan built from this would carry.
    pub fn selected(&self) -> usize {
        self.modules.len() + self.bf16.len()
    }

    /// How many of the index's tensors this plan accounts for.
    pub fn accounted(&self) -> usize {
        self.modules
            .iter()
            .map(|(_, files)| files.len())
            .sum::<usize>()
            + self.bf16.len()
    }

    /// Whether every tensor the **index** names is covered.
    ///
    /// Not "nothing was skipped": a suffix scan that recognised nothing skips
    /// nothing and covers nothing. The denominator is the index.
    pub fn is_complete(&self) -> bool {
        self.unaccounted.is_empty() && self.accounted() == self.index_tensors
    }
}

/// Read a checkpoint and report exactly what a plan over it would contain.
///
/// `model.safetensors.index.json` is the **authority** on what the model is.
/// Scanning the directory for shards and matching suffixes answers a different
/// question -- what happens to be on disk and what happens to look familiar --
/// and the two differ in ways that matter: on an auto-round checkpoint a suffix
/// scan finds no modules at all and would report a BF16-only plan with nothing
/// skipped, which is a false claim of completeness rather than a refusal.
///
/// So every tensor the index names is accounted for: part of a module, a
/// supported standalone tensor, or reported as unaccounted with the reason.
pub fn discover(root: &Path, sources: &mut Sources) -> Result<Discovery> {
    let config_path = root.join("config.json");
    let text = moxie_storage::read_text_capped(
        &config_path,
        moxie_format::checkpoint_config::MAX_CONFIG_BYTES,
    )
    .map_err(|e| {
        invalid(format!(
            "cannot read {}: {e}. A plan is built from the checkpoint's own declaration, so a \
             checkpoint without one has to be described by hand",
            config_path.display()
        ))
    })?;
    let checkpoint = checkpoint_config::parse(&text)?;

    let index_path = root.join("model.safetensors.index.json");
    let index_text = moxie_storage::read_text_capped(
        &index_path,
        checkpoint_config::MAX_INDEX_BYTES,
    )
    .map_err(|e| {
        invalid(format!(
            "cannot read {}: {e}. The index is what says which tensors this model has; without \
             it, a plan could only describe what a directory scan happened to find",
            index_path.display()
        ))
    })?;
    let index = checkpoint_config::parse_index(&index_text)?;
    if let Some(name) = index.keys().find(|name| name.ends_with(".weight_g_idx")) {
        return Err(invalid(format!(
            "{name} requires mapped grouping; a contiguous plan cannot discard its column-to-group map"
        )));
    }
    if index.is_empty() {
        return Err(invalid(format!(
            "{} names no tensors",
            index_path.display()
        )));
    }

    // The shards the index refers to, and whether each one is there.
    let mut files: Vec<String> = index.values().cloned().collect();
    files.sort();
    files.dedup();
    let mut missing_shards = Vec::new();
    for f in &files {
        if !root.join(f).is_file() {
            missing_shards.push(f.clone());
        }
    }
    if !missing_shards.is_empty() {
        return Err(invalid(format!(
            "the index names {} shard(s) this checkpoint does not have: {missing_shards:?}. A \
             plan over an incomplete download would describe tensors nobody can read",
            missing_shards.len()
        )));
    }

    // What each present shard declares, checked against the index both ways.
    let mut declared: BTreeMap<String, (String, Dtype)> = BTreeMap::new();
    let mut duplicates: Vec<String> = Vec::new();
    let mut max_header_bytes = 0u64;
    let mut sizes: BTreeMap<String, u64> = BTreeMap::new();
    for file in &files {
        max_header_bytes = max_header_bytes.max(sources.header_bytes(file)?);
        for (name, dtype, len) in sources.header_entries_sized(file)? {
            sizes.insert(name.clone(), len);
            if let Some((first, _)) = declared.insert(name.clone(), (file.clone(), dtype)) {
                duplicates.push(format!("{name} (in both '{first}' and '{file}')"));
            }
        }
    }
    if !duplicates.is_empty() {
        return Err(invalid(format!(
            "{} tensor name(s) are declared by more than one shard: {:?}. Which bytes a name \
             means would depend on read order, so this is refused rather than resolved",
            duplicates.len(),
            &duplicates[..duplicates.len().min(5)]
        )));
    }
    let mut absent: Vec<String> = index
        .keys()
        .filter(|n| !declared.contains_key(*n))
        .cloned()
        .collect();
    absent.sort();
    if !absent.is_empty() {
        return Err(invalid(format!(
            "the index names {} tensor(s) no shard declares, starting with {:?}",
            absent.len(),
            &absent[..absent.len().min(5)]
        )));
    }

    let gptq = checkpoint.quantization.as_ref().is_some_and(|q| {
        matches!(
            q.serialization,
            checkpoint_config::IntegerSerialization::GptqV1 { .. }
        )
    });
    let mut gptq_shapes = BTreeMap::new();
    let symmetric = checkpoint
        .quantization
        .as_ref()
        .map(|q| q.zero_points == ZeroPointSource::Symmetric)
        .unwrap_or(false);
    // A module carries three tensors when it is symmetric and four when it is
    // not: assuming four made the accounting go negative on every symmetric
    // checkpoint measured.
    let suffixes: &[&str] = if gptq {
        &["qweight", "qzeros", "scales"]
    } else if symmetric {
        &MODULE_SUFFIXES[..3]
    } else {
        &MODULE_SUFFIXES
    };

    let ignored: BTreeSet<&str> = checkpoint
        .quantization
        .as_ref()
        .map(|q| q.ignored.iter().map(String::as_str).collect())
        .unwrap_or_default();

    let mut modules: Vec<(String, BTreeMap<String, String>)> = Vec::new();
    let mut bf16: Vec<(String, String)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut claimed: BTreeSet<String> = BTreeSet::new();

    let anchor_suffix = if gptq { ".qweight" } else { ".weight_packed" };
    let mut anchors: Vec<&String> = index
        .keys()
        .filter(|n| n.ends_with(anchor_suffix))
        .collect();
    anchors.sort();
    // Pinned AutoRound matches_any_regex uses re.search over the stored
    // pattern. Compile one bounded matcher at a time; unsupported Python-only
    // syntax refuses instead of silently dropping an override.
    if let Some(q) = &checkpoint.quantization {
        for pattern in &q.passthrough_patterns {
            let raw_pattern = pattern
                .strip_prefix("+:")
                .or_else(|| pattern.strip_prefix("-:"))
                .unwrap_or(pattern);
            let matcher = regex::RegexBuilder::new(raw_pattern)
                .size_limit(1 << 20)
                .build()
                .map_err(|e| {
                    invalid(format!(
                        "invalid or unsupported passthrough override {pattern:?}: {e}"
                    ))
                })?;
            if let Some(name) = anchors
                .iter()
                .find(|n| matcher.is_match(n.strip_suffix(anchor_suffix).expect("anchor")))
            {
                return Err(invalid(format!(
                    "{name} is quantized but matches 16-bit passthrough override {pattern:?}"
                )));
            }
        }
    }
    for anchor in anchors {
        let module = anchor
            .strip_suffix(anchor_suffix)
            .expect("filtered on the suffix");
        if ignored.contains(module) {
            skipped.push((
                module.to_string(),
                "the checkpoint's own `ignore` list says this module is not quantized".into(),
            ));
            continue;
        }
        let mut found: BTreeMap<String, String> = BTreeMap::new();
        let mut missing: Vec<&str> = Vec::new();
        for suffix in suffixes {
            let tensor = format!("{module}.{suffix}");
            match declared.get(&tensor) {
                Some((holder, _)) => {
                    found.insert((*suffix).to_string(), holder.clone());
                }
                None => missing.push(suffix),
            }
        }
        if !missing.is_empty() {
            skipped.push((
                module.to_string(),
                format!("the shards carry no {missing:?} for it"),
            ));
            continue;
        }
        if gptq {
            let q = checkpoint.quantization.as_ref().expect("GPTQ declaration");
            let (file, _) = declared.get(anchor).expect("declared anchor");
            let weight = sources.entry(file, anchor)?;
            if weight.dtype != Dtype::I32 || weight.shape.len() != 2 {
                return Err(invalid(format!(
                    "{anchor}: GPTQ qweight must be rank-two I32"
                )));
            }
            let inputs = weight.shape[0]
                .checked_mul(u64::from(32 / q.bits))
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid("GPTQ input extent overflows".into()))?;
            let outputs = usize::try_from(weight.shape[1])
                .map_err(|_| invalid("GPTQ output extent overflows".into()))?;
            gptq_shapes.insert(module.to_string(), [outputs, inputs]);
            let map_name = format!("{module}.g_idx");
            if let Some((file, dtype)) = declared.get(&map_name) {
                if *dtype != Dtype::I32 {
                    return Err(invalid(format!("{map_name} must be I32")));
                }
                found.insert("g_idx".into(), file.clone());
            } else if matches!(
                q.serialization,
                checkpoint_config::IntegerSerialization::GptqV1 {
                    requires_group_index: true
                }
            ) {
                return Err(invalid(format!(
                    "{module}: activation ordering requires g_idx"
                )));
            }
        }
        for suffix in found.keys() {
            claimed.insert(format!("{module}.{suffix}"));
        }
        modules.push((module.to_string(), found));
    }

    // Everything the index names that no module claimed.
    let mut unaccounted: Vec<String> = Vec::new();
    let mut names: Vec<&String> = index.keys().collect();
    names.sort();
    for name in names {
        if claimed.contains(name) {
            continue;
        }
        let (file, dtype) = declared.get(name).expect("checked above");
        match dtype {
            Dtype::Bf16 => bf16.push((name.clone(), file.clone())),
            other => {
                unaccounted.push(format!("{name} ({})", other.name()));
                skipped.push((
                    name.clone(),
                    format!(
                        "it is {}, and only BF16 tensors are copied through unchanged",
                        other.name()
                    ),
                ));
            }
        }
    }

    // What the selected tensors weigh at the source, which is the basis for an
    // automatic disk budget.
    let mut source_payload_bytes = 0u64;
    for (module, module_files) in &modules {
        for suffix in module_files.keys() {
            source_payload_bytes += sizes
                .get(&format!("{module}.{suffix}"))
                .copied()
                .unwrap_or(0);
        }
    }
    for (name, _) in &bf16 {
        source_payload_bytes += sizes.get(name).copied().unwrap_or(0);
    }
    // The same bytes again, per **canonical component**. A budget derived from
    // the total alone is wrong in the direction that matters: a work unit lives
    // inside one component, so raising the tile cannot merge two of them and
    // the unit count has a floor the total cannot see.
    //
    // Bound each canonical component separately. Packed zero points expand
    // to i16: one entry per scale, regardless of their source packing width.
    let mut component_bytes: Vec<u64> = Vec::with_capacity(modules.len() * 3 + bf16.len());
    for (module, module_files) in &modules {
        for suffix in module_files.keys() {
            if matches!(suffix.as_str(), "weight_shape" | "g_idx") {
                continue;
            }
            let name = format!("{module}.{suffix}");
            let bytes = if matches!(suffix.as_str(), "qzeros" | "weight_zero_point") {
                let scale_name =
                    format!("{module}.{}", if gptq { "scales" } else { "weight_scale" });
                let (_, dtype) = declared.get(&scale_name).expect("selected scale");
                sizes[&scale_name]
                    .checked_div(dtype.bytes() as u64)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| invalid("canonical zero point extent overflows".into()))?
            } else {
                sizes[&name]
            };
            component_bytes.push(bytes);
        }
    }
    for (name, _) in &bf16 {
        component_bytes.push(sizes.get(name).copied().unwrap_or(0));
    }
    component_bytes.sort_unstable_by(|a, b| b.cmp(a));

    Ok(Discovery {
        checkpoint,
        modules,
        gptq_shapes,
        bf16,
        skipped,
        files,
        index_tensors: index.len(),
        unaccounted,
        max_header_bytes,
        source_payload_bytes,
        component_bytes,
        // Hashed from the bytes parsed above, not from a second read of the
        // same paths.
        config_sha256: moxie_format::sha256_hex(text.as_bytes()),
        index_sha256: moxie_format::sha256_hex(index_text.as_bytes()),
    })
}

/// The `pack-quantized` spec a checkpoint's declaration implies.
pub fn spec_of(checkpoint: &CheckpointDeclaration) -> Result<PackQuantizedSpec> {
    let q = checkpoint.quantization.as_ref().ok_or_else(|| {
        invalid("this checkpoint declares no quantization, so it has no packed modules".into())
    })?;
    Ok(PackQuantizedSpec {
        width: match q.bits {
            4 => moxie_format::affine::IntWidth::Int4,
            8 => moxie_format::affine::IntWidth::Int8,
            other => {
                return Err(invalid(format!("{other}-bit weights are not read here")));
            }
        },
        granularity: q.granularity,
        zero_points: q.zero_points,
    })
}

/// Render a discovery as a **plan**: the compact form, version 2.
///
/// Every module that keeps its tensors in one shard is one entry. The measured
/// exception -- one module of 34,740 on Laguna -- is spelled out.
pub fn to_plan_toml(discovery: &Discovery, source: &SourceBinding) -> Result<String> {
    use moxie_format::plan::string;

    let q = discovery.checkpoint.quantization.as_ref();
    let width = match q.map(|q| q.bits) {
        Some(4) => "int4",
        Some(8) => "int8",
        Some(other) => return Err(invalid(format!("{other}-bit weights are not written here"))),
        None => "int4",
    };
    let group = match q.map(|q| q.granularity) {
        Some(Granularity::Group { size }) => size.to_string(),
        Some(Granularity::Channel) => "\"per-channel\"".to_string(),
        None => "32".to_string(),
    };
    let zero_points = match q.map(|q| q.zero_points) {
        Some(ZeroPointSource::Symmetric) => "symmetric",
        _ => "packed-along-output",
    };
    let shard_index: BTreeMap<&str, usize> = discovery
        .files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect();

    let mut out = String::new();
    out.push_str("# A Moxie repack plan, generated by `moxie-repack plan`.\n");
    out.push_str("#\n");
    out.push_str("# Every value came from the checkpoint's own config.json and its safetensors\n");
    out.push_str("# index. Nothing was inferred from a file name. Read it, edit it if you want,\n");
    out.push_str("# then: moxie-repack repack --plan <this file> --out <dir>\n\n");
    out.push_str("version = 2\n\n");

    out.push_str("shards = [\n");
    for f in &discovery.files {
        out.push_str(&format!("  {},\n", string(f)));
    }
    out.push_str("]\n");

    out.push_str(&source.to_toml());

    // What the second command needs so it does not ask again.
    out.push_str("\n[binding]\n");
    out.push_str("# What this plan was generated from. `repack` reads these again and\n");
    out.push_str("# refuses if they have changed: a plan describes a checkpoint as it was,\n");
    out.push_str("# and a checkpoint that has gained or lost tensors is a different model.\n");
    out.push_str(&format!(
        "config_sha256 = {}\n",
        string(&discovery.config_sha256)
    ));
    out.push_str(&format!(
        "index_sha256 = {}\n",
        string(&discovery.index_sha256)
    ));
    out.push_str(&format!("index_tensors = {}\n", discovery.index_tensors));

    out.push_str("\n[options]\n");
    out.push_str("# Chosen automatically from this checkpoint. Any of them can be\n");
    out.push_str("# overridden on the command line, which wins over what is written here.\n");
    out.push_str(&format!("total_bytes = {}\n", source.options.total_bytes));
    out.push_str(&format!("header_bytes = {}\n", source.options.header_bytes));
    out.push_str(&format!(
        "scratch_bytes = {}\n",
        source.options.scratch_bytes
    ));
    out.push_str(&format!(
        "chunk_file_bytes = {}\n",
        source.options.chunk_file_bytes
    ));
    out.push_str(&format!("disk_bytes = {}\n", source.options.disk_bytes));

    out.push_str("\n[tokenizer]\n");
    out.push_str(&format!("name = {}\n", string(&source.tokenizer.name)));
    out.push_str(&format!(
        "version = {}\n",
        string(&source.tokenizer.version)
    ));
    out.push_str(&format!("digest = {}\n", string(&source.tokenizer.digest)));
    out.push_str("\n[template]\n");
    out.push_str(&format!("name = {}\n", string(&source.template.name)));
    out.push_str(&format!("version = {}\n", string(&source.template.version)));
    out.push_str(&format!("digest = {}\n", string(&source.template.digest)));

    out.push_str("\n[architecture]\n");
    out.push_str(&format!(
        "name = {}\n",
        string(&discovery.checkpoint.architecture)
    ));
    out.push_str("version = \"1\"\n[architecture.metadata]\n");
    out.push_str(&format!(
        "model_type = {}\n",
        string(&discovery.checkpoint.model_type)
    ));

    out.push_str("\n[provenance]\n");
    out.push_str("scale_convention = \"affine-v1\"\n");
    out.push_str(&format!("quantizer = {}\n", string(&source.quantizer)));
    out.push_str("calibration = \"not-declared-by-the-checkpoint\"\n");

    out.push_str("\n[completeness]\n");
    if discovery.is_complete() {
        out.push_str("status = \"complete\"\n");
    } else {
        out.push_str("status = \"partial\"\nmissing = [\n");
        for (what, why) in discovery.skipped.iter().take(64) {
            out.push_str(&format!("  {},\n", string(&format!("{what}: {why}"))));
        }
        if discovery.skipped.len() > 64 {
            out.push_str(&format!(
                "  {},\n",
                string(&format!(
                    "... and {} more, of {} index tensors, {} accounted for",
                    discovery.skipped.len() - 64,
                    discovery.index_tensors,
                    discovery.accounted()
                ))
            ));
        }
        out.push_str("]\n");
    }

    if let Some(checkpoint_config::QuantizationDeclaration {
        serialization:
            checkpoint_config::IntegerSerialization::GptqV1 {
                requires_group_index,
            },
        ..
    }) = q
    {
        if !discovery.modules.is_empty() {
            out.push_str(&format!("\n[gptq]\nwidth = \"{width}\"\ngroup = {group}\nrequires_group_index = {requires_group_index}\n"));
            for (module, files) in &discovery.modules {
                let shape = discovery
                    .gptq_shapes
                    .get(module)
                    .ok_or_else(|| invalid(format!("missing GPTQ shape for {module}")))?;
                out.push_str(&format!(
                    "\n[[gptq.modules]]\nmodule = {}\nshape = [{},{}]\n",
                    string(module),
                    shape[0],
                    shape[1]
                ));
                for (suffix, file) in files {
                    out.push_str(&format!("{suffix} = {}\n", shard_index[file.as_str()]));
                }
            }
        }
    } else if !discovery.modules.is_empty() {
        out.push_str("\n[weights]\n");
        out.push_str(&format!("width = \"{width}\"\n"));
        out.push_str(&format!("group = {group}\n"));
        out.push_str(&format!("zero_points = \"{zero_points}\"\n"));
        out.push_str("modules = [\n");
        let mut split: Vec<&(String, BTreeMap<String, String>)> = Vec::new();
        for entry in &discovery.modules {
            let (module, files) = entry;
            let mut homes: Vec<&String> = files.values().collect();
            homes.sort();
            homes.dedup();
            if homes.len() == 1 {
                let i = shard_index[homes[0].as_str()];
                out.push_str(&format!("  [{}, {i}],\n", string(module)));
            } else {
                split.push(entry);
            }
        }
        out.push_str("]\n");
        for (module, files) in split {
            out.push_str("\n[[weights.split]]\n");
            out.push_str(&format!("module = {}\n", string(module)));
            for (suffix, file) in files {
                out.push_str(&format!("{suffix} = {}\n", shard_index[file.as_str()]));
            }
        }
    }

    if !discovery.bf16.is_empty() {
        out.push_str("\n[bf16]\ntensors = [\n");
        for (name, file) in &discovery.bf16 {
            out.push_str(&format!(
                "  [{}, {}],\n",
                string(name),
                shard_index[file.as_str()]
            ));
        }
        out.push_str("]\n");
    }
    Ok(out)
}

/// An identity, as a plan records one.
#[derive(Debug, Clone)]
pub struct AssetIdentity {
    pub name: String,
    pub version: String,
    pub digest: String,
}

/// What a plan says about where its bytes came from.
///
/// Where the downloader recorded a revision and per-file digests, those are
/// carried. Where it did not, that is **stated as absent** rather than filled
/// with a placeholder, and the source is bound by what this program can see:
/// each shard's size and the digest of its header, which is what every tensor
/// offset in the plan depends on. The payload is hashed at repack, which is
/// where it is read anyway.
#[derive(Debug, Clone)]
pub struct SourceBinding {
    /// Where the checkpoint is, so the second command does not need it again.
    pub root: String,
    pub model: String,
    pub revision: Option<String>,
    pub license: String,
    pub quantizer: String,
    pub tokenizer: AssetIdentity,
    pub template: AssetIdentity,
    /// `(file, sha256)` as the downloader recorded them, when it did.
    pub recorded_digests: Vec<(String, String)>,
    /// The budgets this plan resolved, so `repack --plan` needs none of them.
    pub options: crate::Budgets,
    // **No digests here.** They used to be fields a caller filled in, and the
    // caller filled them in by reading `config.json` and the index a second
    // time -- a second moment, describing a checkpoint that may have changed
    // since the one the plan above describes. They now come from the
    // `Discovery` this binding is written beside, where they cannot disagree
    // with it, so there is nothing left for a caller to get wrong.
}

impl SourceBinding {
    fn to_toml(&self) -> String {
        use moxie_format::plan::string;
        let mut out = String::from("\n[source]\n");
        out.push_str(&format!("root = {}\n", string(&self.root)));
        out.push_str(&format!("model = {}\n", string(&self.model)));
        match &self.revision {
            Some(r) => out.push_str(&format!("revision = {}\n", string(r))),
            // Honest absence, under a documented spelling: this is not a
            // revision, it says there is none to record.
            None => out.push_str("revision = \"unrecorded:no-download-metadata\"\n"),
        }
        out.push_str(&format!("license = {}\n", string(&self.license)));
        out
    }
}

/// Read what the downloader recorded beside a checkpoint, if anything.
///
/// `huggingface_hub` writes `<file>.metadata` under `.cache/huggingface/download`
/// carrying the commit the file came from and its SHA-256. That is real
/// provenance, already computed, and carrying it costs one small read per shard.
pub fn recorded_provenance(
    root: &Path,
    files: &[String],
) -> (Option<String>, Vec<(String, String)>) {
    let dir = root.join(".cache/huggingface/download");
    let mut revisions: BTreeSet<String> = BTreeSet::new();
    let mut digests = Vec::new();
    for f in files {
        let meta = dir.join(format!("{f}.metadata"));
        let Ok(text) = std::fs::read_to_string(&meta) else {
            continue;
        };
        let mut lines = text.lines();
        if let Some(rev) = lines.next()
            && rev.len() == 40
            && rev.chars().all(|c| c.is_ascii_hexdigit())
        {
            revisions.insert(rev.to_string());
        }
        if let Some(sha) = lines.next()
            && sha.len() == 64
            && sha.chars().all(|c| c.is_ascii_hexdigit())
        {
            digests.push((f.clone(), sha.to_string()));
        }
    }
    // One revision or none: files from two commits are not one revision, and
    // saying so is better than picking the first.
    let revision = if revisions.len() == 1 {
        revisions.into_iter().next()
    } else {
        None
    };
    (revision, digests)
}

/// The identity of an asset beside the checkpoint, by its content.
///
/// A tokenizer or chat template is either **there**, in which case its digest
/// is what identifies it, or it is not, in which case the plan says so. Neither
/// is a placeholder: `absent` is a fact about this checkpoint.
pub fn asset_identity(root: &Path, file: &str) -> AssetIdentity {
    let path = root.join(file);
    match std::fs::read(&path) {
        Ok(bytes) => AssetIdentity {
            name: file.to_string(),
            version: "content-addressed".into(),
            digest: moxie_format::sha256_hex(&bytes),
        },
        Err(_) => AssetIdentity {
            name: "absent".into(),
            version: "absent".into(),
            // The digest of the empty input: an identity that says "nothing",
            // rather than a made-up one.
            digest: moxie_format::sha256_hex(&[]),
        },
    }
}

/// Budgets derived from what a checkpoint actually contains.
///
/// Five numbers with no defaults was defended as "how much of a user's machine
/// a tool may spend is not a question the tool should answer". That reads well
/// and does not survive contact with a user: the right values follow from the
/// largest shard header, the payload total and the work-unit size that keeps a
/// journal readable, and the program knows all three while the user knows none
/// of them ([ADR 0026]).
///
/// So they are **derived and reported**, never hidden: the plan records what was
/// chosen, the run prints it, and any explicit flag overrides it.
///
/// [ADR 0026]: ../../../docs/decisions/adr/0026-generated-plans-and-automatic-budgets.md
pub fn automatic_budgets(discovery: &Discovery) -> Result<crate::Budgets> {
    // One source header, at the peak its parse costs rather than its serialized
    // length, with room for a checkpoint whose largest header is bigger than
    // the one measured here.
    let header_bytes = moxie_storage::HeaderBudget::estimated_peak(discovery.max_header_bytes)
        .unwrap_or(u64::MAX)
        .saturating_mul(2)
        .max(16 << 20);

    let scratch_bytes = automatic_scratch_bytes(discovery)?;

    // The canonical payload is the source's represented weights, which a
    // bit-identical repack neither grows nor shrinks materially. The margin
    // covers shard headers, the journal, the staged manifest and the
    // replacement a compaction writes beside it.
    let canonical_bytes = discovery
        .component_bytes
        .iter()
        .fold(0u64, |sum, &n| sum.saturating_add(n));
    let disk_bytes = canonical_bytes
        .saturating_add(canonical_bytes / 8)
        .saturating_add(1 << 30);

    // A shard the ecosystem's tools are comfortable with -- but never larger
    // than the disk this run may use, which a small checkpoint would otherwise
    // make it. A shard maximum above the disk budget is a plan whose first
    // shard cannot be written.
    let chunk_file_bytes: u64 = (4u64 << 30).min(disk_bytes);

    // Everything the run holds that is not a payload tile: the parsed plan, the
    // resolved tensors, the output plan, the shard headers and, on a resume,
    // the journal records.
    let entries = discovery.selected() as u64;
    let tiles = 3 * scratch_bytes as u64 / 2;
    let map_metadata = discovery
        .modules
        .iter()
        .filter(|(_, files)| files.contains_key("g_idx"))
        .fold(0u64, |sum, (module, _)| {
            sum.saturating_add((discovery.gptq_shapes[module][1] as u64).saturating_mul(64))
        });
    let metadata = header_bytes
        .saturating_add(map_metadata)
        .saturating_add(moxie_format::manifest::MAX_MANIFEST_BYTES as u64)
        .saturating_add(entries.saturating_mul(8 * 1024));
    let total_bytes = tiles
        .saturating_add(metadata)
        .saturating_add(entries.saturating_mul(4 * 1024))
        .saturating_add(512 << 20);

    Ok(crate::Budgets {
        total_bytes,
        header_bytes,
        scratch_bytes,
        chunk_file_bytes,
        disk_bytes,
    })
}

/// The largest payload tile the reader will accept, and so the largest half of
/// a scratch allowance there is any point choosing.
///
/// Derived from the reader rather than written down again: a planner that
/// picked a tile the reader refuses would be handing out a plan whose very
/// first `open_sources` fails, which is what independent review saw when a
/// 32,000-tensor checkpoint drove this to a 1 GiB scratch.
fn max_tile_bytes() -> u64 {
    let admits = |b: u64| {
        usize::try_from(b)
            .ok()
            .is_some_and(|m| moxie_storage::ByteBudget::new(m).is_some())
    };
    // Doubled until it is refused, then the last accepted power of two. The
    // reader's limit is a power of two, so this lands on it exactly; if it ever
    // stops being one, this is the largest usable tile below it, which is still
    // a tile the reader admits.
    let mut tile: u64 = 1;
    while let Some(next) = tile.checked_mul(2) {
        if !admits(next) {
            break;
        }
        tile = next;
    }
    tile
}

/// How many journal records a tile size implies, through the units that are
/// actually cut.
///
/// A unit lives **inside one tensor**: the converter walks each tensor's
/// sections and splits each into tile-sized pieces, so raising the tile can
/// never combine two tensors into one record. Dividing the payload total by the
/// tile misses that entirely, and the sizing loop that did so kept raising the
/// scratch against a floor it could not move.
fn units_at_tile(component_bytes: &[u64], tile: u64) -> u64 {
    let tile = tile.max(1);
    component_bytes
        .iter()
        .map(|b| b.div_ceil(tile).max(1))
        .fold(0u64, |a, b| a.saturating_add(b))
}

/// A work-unit size whose journal a resume can read back, or a refusal.
///
/// **A refusal is a real outcome here.** Every unit appends one journal line,
/// the journal has a cap checked before it is parsed, and the smallest unit
/// count is one per tensor whatever the tile. A checkpoint with more tensors
/// than that cap allows cannot be planned at these defaults, and saying so is
/// the honest answer -- the previous loop instead raised the scratch to 1 GiB,
/// emitted the plan, and let `repack` fail on its own defaults one command
/// later.
fn automatic_scratch_bytes(discovery: &Discovery) -> Result<usize> {
    // What one record costs, generously: the fixed part of a unit line plus the
    // longest name in this plan.
    let widest = discovery
        .modules
        .iter()
        .map(|(m, _)| m.len())
        .chain(discovery.bf16.iter().map(|(n, _)| n.len()))
        .max()
        .unwrap_or(64) as u64;
    // **The run's own rule**, not a second estimate of it: a planner whose
    // arithmetic differs from the writer's can emit a plan the writer rejects,
    // which is the defect being fixed here.
    let journal_of = |units: u64| crate::StagingEstimate::journal_bound_for(units, widest);
    let cap = moxie_format::journal::MAX_JOURNAL_BYTES as u64;

    let components: &[u64] = &discovery.component_bytes;
    let max_tile = max_tile_bytes();
    let max_scratch = max_tile.saturating_mul(2);

    // The floor: one record per tensor, which no tile size goes below.
    let floor = units_at_tile(components, u64::MAX).max(1);
    if journal_of(floor) > cap {
        return Err(invalid(format!(
            "this checkpoint's {} selected tensor(s) hold {floor} component(s), and a conversion \
             writes at least one resume-journal record for each: {} byte(s), above the {cap} byte \
             cap a resume can read back. A work unit is cut inside a component and never across \
             two, so no scratch size lowers this -- it is a limit of the journal format, not of \
             the budgets. Convert this checkpoint in parts with a hand-written selection",
            discovery.selected().max(1),
            journal_of(floor)
        )));
    }

    let mut scratch: u64 = (64 << 20).min(max_scratch);
    loop {
        let tile = (scratch / 2).max(1);
        if journal_of(units_at_tile(components, tile)) <= cap {
            break;
        }
        if scratch >= max_scratch {
            // Unreachable while the floor check above holds -- the floor is
            // what `max_tile` converges to -- but a sizing rule that could
            // return an unusable number if that ever stopped being true is the
            // bug being fixed, so it refuses instead.
            return Err(invalid(format!(
                "this checkpoint needs a payload tile above the {max_tile} byte(s) a reader                  admits before its resume journal fits. Convert it in parts with a hand-written                  selection"
            )));
        }
        scratch = scratch.saturating_mul(2).min(max_scratch);
    }

    // Everything downstream of this is a `usize`, and the reader is what set
    // the ceiling, so the conversion cannot fail -- but it is checked rather
    // than asserted.
    let tile = (scratch / 2).max(1);
    let tile_usize = usize::try_from(tile)
        .ok()
        .filter(|t| moxie_storage::ByteBudget::new(*t).is_some());
    if tile_usize.is_none() {
        return Err(invalid(format!(
            "{tile} byte(s) is not a payload tile this machine's reader admits"
        )));
    }
    usize::try_from(scratch).map_err(|_| {
        invalid(format!(
            "{scratch} byte(s) of scratch does not fit this machine"
        ))
    })
}

/// Refuse a plan whose checkpoint has changed since it was generated.
///
/// A source shard digest is not enough. Independent review generated a plan,
/// added an indexed BF16 tensor in a new shard, and watched `repack` publish
/// the old subset under `completeness = "complete"` -- every shard the plan
/// named was byte-identical, and the model was still a different model.
///
/// So the plan binds what **defines** the model: its `config.json` and its
/// index, by content, plus how many tensors that index named.
/// `generated` says which route this document arrived by: `--plan` means the
/// program wrote it and every guard below applies, `--selection` means a person
/// did and the binding is optional.
pub fn confirm_binding(
    root: &Path,
    plan_text: &str,
    selection: &moxie_format::selection::Selection,
    generated: bool,
) -> Result<()> {
    let Some((config_sha, index_sha, index_tensors)) =
        moxie_format::plan::declared_binding(plan_text)?
    else {
        // **A `--plan` document without a binding is not a plan.** The binding
        // is what every check below is measured against, so a document missing
        // it silently skipped all of them -- independent review deleted the
        // `[binding]` block, deleted a tensor, and published a `complete`
        // artifact with the tensor missing. The parser leaves the section
        // optional because a hand-written selection has none, and that is the
        // `--selection` route; it is not this one.
        if generated {
            return Err(invalid(
                "this plan has no [binding] section. `--plan` means this program generated the \
                 document, and the binding is what says which checkpoint it describes -- without \
                 it nothing can be checked against anything. Generate a new plan, or pass a \
                 hand-written document with --selection, which is the route that does not bind"
                    .to_string(),
            ));
        }
        // A hand-written selection binds nothing of the kind; the advanced path
        // is the user saying what to convert.
        return Ok(());
    };
    let config = moxie_storage::read_text_capped(
        &root.join("config.json"),
        moxie_format::checkpoint_config::MAX_CONFIG_BYTES,
    )?;
    let now_config = moxie_format::sha256_hex(config.as_bytes());
    if now_config != config_sha {
        return Err(invalid(format!(
            "this checkpoint's config.json has changed since the plan was generated ({config_sha} \
             became {now_config}). What the plan says about packing may no longer describe it, so \
             it is refused rather than applied. Generate a new plan"
        )));
    }
    let index_text = moxie_storage::read_text_capped(
        &root.join("model.safetensors.index.json"),
        moxie_format::checkpoint_config::MAX_INDEX_BYTES,
    )?;
    let now_index = moxie_format::sha256_hex(index_text.as_bytes());
    if now_index != index_sha {
        let now = moxie_format::checkpoint_config::parse_index(&index_text)?;
        return Err(invalid(format!(
            "this checkpoint's tensor index has changed since the plan was generated: it named \
             {index_tensors} tensor(s) and now names {}. A plan describes a checkpoint as it was, \
             and publishing the old subset as complete would be describing a model nobody has. \
             Generate a new plan",
            now.len()
        )));
    }
    let index = moxie_format::checkpoint_config::parse_index(&index_text)?;
    // **And does this plan still cover that index?** The two questions are not
    // the same one, and only the first was being asked. An independent review
    // deleted one tensor entry from a generated plan and published a
    // one-tensor artifact marked `complete`: the checkpoint had not changed, so
    // the digests above matched, and the guard downstream read the plan's own
    // `completeness` field -- which the same edit leaves untouched.
    //
    // **Counting is not comparing.** The first repair counted what the plan
    // accounted for and checked the total against `index_tensors`. Review then
    // replaced one entry with a *different* tensor the index already named,
    // left the total alone, and published an artifact carrying one tensor twice
    // and another not at all -- still marked `complete`. A total is a shadow of
    // a set; two different sets cast the same one.
    //
    // So the exact `(tensor name, shard)` pairs are compared. The index is
    // bound by content above, so it is the index the plan was generated from,
    // and a complete plan has to name every pair in it and no pair outside it.
    let mut named: BTreeMap<String, &str> = BTreeMap::new();
    let mut duplicated: Vec<String> = Vec::new();
    for tensor in &selection.tensors {
        let pairs: Vec<(String, &str)> = match &tensor.kind {
            moxie_format::selection::SelectionKind::Bf16 { name, file } => {
                vec![(name.clone(), file.as_str())]
            }
            moxie_format::selection::SelectionKind::PackQuantized { module, files, .. }
            | moxie_format::selection::SelectionKind::Gptq { module, files, .. } => files
                .iter()
                .map(|(suffix, file)| (format!("{module}.{suffix}"), file.as_str()))
                .collect(),
        };
        for (name, file) in pairs {
            if let Some(first) = named.insert(name.clone(), file) {
                duplicated.push(format!("{name} (in '{first}' and again in '{file}')"));
            }
        }
    }
    if !duplicated.is_empty() {
        return Err(invalid(format!(
            "this plan names {} tensor(s) more than once: {:?}. One source tensor converted twice \
             is not a description of this checkpoint, whatever the total comes to. Generate a new \
             plan",
            duplicated.len(),
            &duplicated[..duplicated.len().min(5)]
        )));
    }
    if !matches!(
        selection.completeness,
        moxie_format::selection::Completeness::Complete
    ) {
        return Ok(());
    }
    let mut missing: Vec<&str> = Vec::new();
    let mut moved: Vec<String> = Vec::new();
    for (name, file) in &index {
        match named.get(name) {
            None => missing.push(name.as_str()),
            Some(planned) if *planned != file.as_str() => {
                moved.push(format!(
                    "{name} (plan says '{planned}', index says '{file}')"
                ));
            }
            Some(_) => {}
        }
    }
    let mut extra: Vec<&str> = named
        .keys()
        .filter(|name| !index.contains_key(name.as_str()))
        .map(String::as_str)
        .collect();
    missing.sort_unstable();
    extra.sort_unstable();
    if !missing.is_empty() || !extra.is_empty() || !moved.is_empty() {
        return Err(invalid(format!(
            "this plan says it is complete and does not describe this checkpoint's index: {} \
             indexed tensor(s) it never names{}, {} tensor(s) the index does not have{}, and {} \
             bound to the wrong shard{}. It covers {} of {index_tensors}. A plan edited since it \
             was generated is not a description of this model, and publishing the difference as \
             complete would describe a model nobody has. Generate a new plan, or declare the \
             subset deliberately with a partial plan and --allow-partial",
            missing.len(),
            sample(&missing),
            extra.len(),
            sample(&extra),
            moved.len(),
            sample_owned(&moved),
            named.len()
        )));
    }
    Ok(())
}

/// The first few of a list, for a refusal that has to stay readable.
fn sample(names: &[&str]) -> String {
    if names.is_empty() {
        return String::new();
    }
    format!(" (starting with {:?})", &names[..names.len().min(3)])
}

fn sample_owned(names: &[String]) -> String {
    if names.is_empty() {
        return String::new();
    }
    format!(" (starting with {:?})", &names[..names.len().min(3)])
}
