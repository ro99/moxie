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

    let symmetric = checkpoint
        .quantization
        .as_ref()
        .map(|q| q.zero_points == ZeroPointSource::Symmetric)
        .unwrap_or(false);
    // A module carries three tensors when it is symmetric and four when it is
    // not: assuming four made the accounting go negative on every symmetric
    // checkpoint measured.
    let suffixes: &[&str] = if symmetric {
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

    let mut anchors: Vec<&String> = index
        .keys()
        .filter(|n| n.ends_with(".weight_packed"))
        .collect();
    anchors.sort();
    for anchor in anchors {
        let module = anchor
            .strip_suffix(".weight_packed")
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
        for suffix in suffixes {
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

    Ok(Discovery {
        checkpoint,
        modules,
        bf16,
        skipped,
        files,
        index_tensors: index.len(),
        unaccounted,
        max_header_bytes,
        source_payload_bytes,
    })
}

/// The `pack-quantized` spec a checkpoint's declaration implies.
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

    if !discovery.modules.is_empty() {
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
pub fn automatic_budgets(discovery: &Discovery) -> crate::Budgets {
    // One source header, at the peak its parse costs rather than its serialized
    // length, with room for a checkpoint whose largest header is bigger than
    // the one measured here.
    let header_bytes = moxie_storage::HeaderBudget::estimated_peak(discovery.max_header_bytes)
        .unwrap_or(u64::MAX)
        .saturating_mul(2)
        .max(16 << 20);

    // A work unit large enough that a whole model does not produce more journal
    // records than a resume can read back, and small enough to stay a bounded
    // allocation. Both tiles come out of this.
    let scratch_bytes: usize = 64 << 20;

    // The canonical payload is the source's represented weights, which a
    // bit-identical repack neither grows nor shrinks materially. The margin
    // covers shard headers, the journal, the staged manifest and the
    // replacement a compaction writes beside it.
    let disk_bytes = discovery
        .source_payload_bytes
        .saturating_add(discovery.source_payload_bytes / 8)
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
    let metadata = header_bytes
        .saturating_add(moxie_format::manifest::MAX_MANIFEST_BYTES as u64)
        .saturating_add(entries.saturating_mul(8 * 1024));
    let total_bytes = tiles
        .saturating_add(metadata)
        .saturating_add(entries.saturating_mul(4 * 1024))
        .saturating_add(512 << 20);

    crate::Budgets {
        total_bytes,
        header_bytes,
        scratch_bytes,
        chunk_file_bytes,
        disk_bytes,
    }
}
