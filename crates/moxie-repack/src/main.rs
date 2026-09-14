//! `moxie-repack` — inspect, repack and verify canonical artifacts offline.
//!
//! Arguments and reporting only: every decision below this line belongs to the
//! shared crates. See the library's module documentation for what a repack is
//! and, more importantly, what it is not evidence of.
//!
//! ## Exit status
//!
//! | Code | Meaning |
//! |---|---|
//! | 0 | The command did what it was asked |
//! | 1 | The command line was wrong |
//! | 2 | Refused or failed; a repack leaves a resumable destination |
//! | 3 | Cancelled before the publication boundary |
//! | 4 | **Published, durability unconfirmed** — the artifact exists and this program could not confirm it survives a power cut |
//!
//! Four is separate from zero and from two on purpose. A publish whose
//! confirming `fsync` failed has not failed: deleting the artifact would be
//! deleting a possibly published output, and calling it success would be
//! claiming a durability nothing observed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use moxie_repack::write::{Faults, Options, Outcome, Site};
use moxie_repack::{Budgets, InspectReport, RepackReport, VerifyReport};

const USAGE: &str = "\
moxie-repack — offline canonical inspection, repack and verification

The usual path, two commands and no other flags:

  moxie-repack plan   --source-root <checkpoint> [--out-plan <file>]
  moxie-repack repack --plan <file> --out <dir>

`plan` reads the checkpoint's own config and safetensors index, writes a plan
you can read, and converts nothing. The plan records where the checkpoint is and
which resource settings it chose, so `repack` needs neither again.

Everything below is advanced: a hand-written selection, or overriding what the
plan chose. A flag always wins over the plan, which wins over the automatic
default.

  moxie-repack inspect --selection <file> --source-root <dir> <budgets>
  moxie-repack repack  --selection <file> --source-root <dir> --out <dir> <budgets> [options]
  moxie-repack verify  --artifact <dir> --scratch-bytes <n>

Budgets (required only on the advanced paths; `plan` chooses them from the
checkpoint and records what it chose):
  --total-bytes <n>        total admitted dynamic working memory
  --header-bytes <n>       peak heap admitted for one source header
  --scratch-bytes <n>      payload scratch: one source tile plus one canonical tile
  --chunk-file-bytes <n>   largest output chunk file (a disk plan, not RAM)
  --disk-bytes <n>         total payload bytes this run may write

Options:
  --out-plan <file>             where `plan` writes the plan (default: ./<model>.plan.toml)
  --plan <file>                 the plan `repack` should convert
  --force                       replace an existing plan
  --allow-partial               convert a plan that does not cover the whole model
  --out-selection <file>        deprecated spelling of --out-plan
  --take-over-interrupted-run   continue a destination whose lock file survived a crash
  --test-fail-at <site>:<n>     fail the nth visit to a named durable boundary
  --test-abort-after-units <n>  abort the process after n units, to exercise restart
  --test-cancel-after-units <n> request cancellation after n units

The three --test- options exist so the restart and failure gates drive this
program rather than a harness that reimplements it. They are named for what
they are.

Sizes accept a plain byte count or a KiB/MiB/GiB suffix (4MiB, 64KiB).
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    let Some(command) = args.first() else {
        eprint!("{USAGE}");
        return 1;
    };
    let mut flags = match Flags::parse(&args[1..]) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("moxie-repack: {e}");
            eprint!("{USAGE}");
            return 1;
        }
    };
    match command.as_str() {
        "plan" => command_plan(&mut flags),
        "inspect" => command_inspect(&mut flags),
        "repack" => command_repack(&mut flags),
        "verify" => command_verify(&mut flags),
        "--help" | "-h" | "help" => {
            print!("{USAGE}");
            0
        }
        other => {
            eprintln!("moxie-repack: unknown command '{other}'");
            eprint!("{USAGE}");
            1
        }
    }
}

#[derive(Debug, Default)]
struct Flags {
    selection: Option<PathBuf>,
    source_root: Option<PathBuf>,
    out: Option<PathBuf>,
    out_selection: Option<PathBuf>,
    out_plan: Option<PathBuf>,
    plan: Option<PathBuf>,
    force: bool,
    allow_partial: bool,
    artifact: Option<PathBuf>,
    total_bytes: Option<u64>,
    header_bytes: Option<u64>,
    scratch_bytes: Option<u64>,
    chunk_file_bytes: Option<u64>,
    disk_bytes: Option<u64>,
    take_over: bool,
    fail_at: Vec<(Site, u64)>,
    abort_after_units: Option<usize>,
    cancel_after_units: Option<usize>,
}

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut out = Flags::default();
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let mut value = || -> Result<String, String> {
                i += 1;
                args.get(i)
                    .cloned()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag {
                "--selection" => out.selection = Some(PathBuf::from(value()?)),
                "--source-root" => out.source_root = Some(PathBuf::from(value()?)),
                "--out" => out.out = Some(PathBuf::from(value()?)),
                "--out-selection" => out.out_selection = Some(PathBuf::from(value()?)),
                "--out-plan" => out.out_plan = Some(PathBuf::from(value()?)),
                "--plan" => out.plan = Some(PathBuf::from(value()?)),
                "--force" => out.force = true,
                "--allow-partial" => out.allow_partial = true,
                "--artifact" => out.artifact = Some(PathBuf::from(value()?)),
                "--total-bytes" => out.total_bytes = Some(bytes(&value()?)?),
                "--header-bytes" => out.header_bytes = Some(bytes(&value()?)?),
                "--scratch-bytes" => out.scratch_bytes = Some(bytes(&value()?)?),
                "--chunk-file-bytes" => out.chunk_file_bytes = Some(bytes(&value()?)?),
                "--disk-bytes" => out.disk_bytes = Some(bytes(&value()?)?),
                "--take-over-interrupted-run" => out.take_over = true,
                "--test-fail-at" => {
                    let raw = value()?;
                    let (name, n) = raw
                        .rsplit_once(':')
                        .ok_or_else(|| format!("--test-fail-at wants <site>:<n>, got {raw:?}"))?;
                    let site = Site::from_name(name).ok_or_else(|| {
                        format!(
                            "unknown failure site {name:?}; known sites: {}",
                            Site::ALL
                                .iter()
                                .map(|s| s.name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                    let n: u64 = n.parse().map_err(|_| format!("{n:?} is not a count"))?;
                    out.fail_at.push((site, n));
                }
                "--test-abort-after-units" => {
                    out.abort_after_units = Some(
                        value()?
                            .parse()
                            .map_err(|_| "--test-abort-after-units wants a count".to_string())?,
                    );
                }
                "--test-cancel-after-units" => {
                    out.cancel_after_units = Some(
                        value()?
                            .parse()
                            .map_err(|_| "--test-cancel-after-units wants a count".to_string())?,
                    );
                }
                other => return Err(format!("unknown flag '{other}'")),
            }
            i += 1;
        }
        Ok(out)
    }

    fn budgets(&self) -> Result<Budgets, String> {
        let need = |name: &str, v: Option<u64>| -> Result<u64, String> {
            v.ok_or_else(|| {
                format!(
                    "{name} is required: a repack states what it may spend rather than assuming it"
                )
            })
        };
        let scratch = need("--scratch-bytes", self.scratch_bytes)?;
        Ok(Budgets {
            total_bytes: need("--total-bytes", self.total_bytes)?,
            header_bytes: need("--header-bytes", self.header_bytes)?,
            scratch_bytes: usize::try_from(scratch)
                .map_err(|_| "--scratch-bytes does not fit this platform".to_string())?,
            chunk_file_bytes: need("--chunk-file-bytes", self.chunk_file_bytes)?,
            disk_bytes: need("--disk-bytes", self.disk_bytes)?,
        })
    }

    fn faults(&self) -> Faults {
        let mut faults = Faults::none();
        for (site, n) in &self.fail_at {
            faults = faults.fail_at(*site, *n);
        }
        faults
    }
}

/// A byte count, with the suffixes a person actually types.
fn bytes(raw: &str) -> Result<u64, String> {
    let raw = raw.trim();
    let (digits, scale) = match raw {
        r if r.ends_with("GiB") => (&r[..r.len() - 3], 1u64 << 30),
        r if r.ends_with("MiB") => (&r[..r.len() - 3], 1u64 << 20),
        r if r.ends_with("KiB") => (&r[..r.len() - 3], 1u64 << 10),
        r => (r, 1),
    };
    let n: u64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("{raw:?} is not a byte count"))?;
    n.checked_mul(scale)
        .ok_or_else(|| format!("{raw:?} overflows a byte count"))
}

/// `plan` — read a checkpoint and write the selection it implies.
///
/// The selection exists so a conversion is explicit, not so a person has to
/// type one: every value in it is already in the checkpoint's `config.json` and
/// its shard headers. This reads those, writes the file, and says what it left
/// out and why. It converts nothing.
/// Refuse a plan whose output would land inside the checkpoint it describes.
///
/// A checkpoint is a **read-only input**. Writing the plan, or the staging file
/// beside it, into that directory would leave this program's own output among
/// the source files it later hashes, and would modify a directory the user did
/// not offer for writing.
///
/// The directory is what is compared, canonically, because the plan file itself
/// does not exist yet and because `../` and symbolic links make a textual
/// comparison say whatever the path was spelled like.
fn plan_output_outside(root: &std::path::Path, out: &std::path::Path) -> Result<(), String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|e| format!("cannot resolve {}: {e}", root.display()))?;
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    let canonical_parent = parent.canonicalize().map_err(|e| {
        format!(
            "cannot resolve {} to write {} into: {e}",
            parent.display(),
            out.display()
        )
    })?;
    if canonical_parent.starts_with(&canonical_root) {
        return Err(format!(
            "{} is inside the checkpoint at {}. A checkpoint is a read-only input here, so the \
             plan is written outside it: run this from another directory, or pass --out-plan with \
             a path outside {}",
            canonical_parent.display(),
            canonical_root.display(),
            canonical_root.display()
        ));
    }
    Ok(())
}

fn command_plan(flags: &mut Flags) -> i32 {
    let Some(root) = flags.source_root.clone() else {
        eprintln!("moxie-repack: plan needs --source-root");
        return 1;
    };
    if flags.selection.is_some() {
        eprintln!("moxie-repack: plan takes no --selection; it writes one");
        return 1;
    }
    // Planning needs to parse headers before it can say what this checkpoint
    // would cost, so it reads them under a bounded bootstrap allowance and then
    // derives the real budgets from what it found.
    let bootstrap = moxie_repack::Budgets {
        total_bytes: 2 << 30,
        header_bytes: 512 << 20,
        scratch_bytes: 1 << 20,
        chunk_file_bytes: 4 << 30,
        disk_bytes: 1 << 30,
    };
    // **Never inside the source root.** A checkpoint is a read-only input, and
    // writing a plan into one would be this program writing where it promised
    // not to.
    let out_selection = flags
        .out_plan
        .clone()
        .or_else(|| flags.out_selection.clone())
        .unwrap_or_else(|| {
            let leaf = root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "checkpoint".into());
            std::path::PathBuf::from(format!("./{leaf}.plan.toml"))
        });
    // **Checked before anything is created or removed.** The comment above says
    // never inside the source root; nothing enforced it. The default resolves
    // against the current directory, so running `plan` from inside a checkpoint
    // wrote the plan into the checkpoint -- independent review did exactly that
    // -- and an explicit `--out-plan` pointing in there had the same gap. The
    // staging file shares this directory, so this one check covers both.
    match plan_output_outside(&root, &out_selection) {
        Ok(()) => {}
        Err(why) => {
            eprintln!("moxie-repack: {why}");
            return 1;
        }
    }
    if out_selection.exists() && !flags.force {
        eprintln!(
            "moxie-repack: {} already exists. Pass --force to replace it, or name another path \
             with --out-plan",
            out_selection.display()
        );
        return 1;
    }

    let result = (|| -> moxie_types::Result<(moxie_repack::discover::Discovery, String)> {
        let mut sources = moxie_repack::open_sources(&root, &bootstrap)?;
        let discovery = moxie_repack::discover::discover(&root, &mut sources)?;
        let org = root
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned());
        let leaf = root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unnamed-checkpoint".into());
        let model = match org {
            Some(o) => format!("{o}/{leaf}"),
            None => leaf,
        };
        let (revision, recorded_digests) =
            moxie_repack::discover::recorded_provenance(&root, &discovery.files);
        let binding = moxie_repack::discover::SourceBinding {
            // **Absolute.** A relative root recorded as given resolves against
            // whatever directory the second command runs from, which is not
            // where the plan was made.
            // A root that will not resolve is an error, not a relative path
            // written down anyway: the plan would name a directory the second
            // command resolves somewhere else.
            root: root
                .canonicalize()
                .map_err(|e| moxie_types::Error::InvalidArtifact {
                    detail: format!("cannot resolve {} to an absolute path: {e}", root.display())
                        .into(),
                })?
                .to_string_lossy()
                .into_owned(),
            model,
            revision,
            license: "see the checkpoint's own licence files".into(),
            quantizer: match &discovery.checkpoint.quantization {
                Some(_) => "declared-by-the-checkpoint".into(),
                None => "none-the-source-is-bf16".into(),
            },
            tokenizer: moxie_repack::discover::asset_identity(&root, "tokenizer.json"),
            template: moxie_repack::discover::asset_identity(&root, "chat_template.jinja"),
            recorded_digests,
            // Derived from the checkpoint, and overridden by any flag the user
            // gave: explicit always wins over automatic.
            // Automatic, with any flag the user gave overriding **that field**.
            // All-or-nothing fallback discarded a lone `--disk-bytes`.
            options: {
                let auto = moxie_repack::discover::automatic_budgets(&discovery)?;
                moxie_repack::Budgets {
                    total_bytes: flags.total_bytes.unwrap_or(auto.total_bytes),
                    header_bytes: flags.header_bytes.unwrap_or(auto.header_bytes),
                    scratch_bytes: flags
                        .scratch_bytes
                        .map(|v| v as usize)
                        .unwrap_or(auto.scratch_bytes),
                    chunk_file_bytes: flags.chunk_file_bytes.unwrap_or(auto.chunk_file_bytes),
                    disk_bytes: flags.disk_bytes.unwrap_or(auto.disk_bytes),
                }
            },
        };
        let text = moxie_repack::discover::to_plan_toml(&discovery, &binding)?;
        Ok((discovery, text))
    })();

    match result {
        Ok((discovery, text)) => {
            println!("architecture: {}", discovery.checkpoint.architecture);
            println!("shards-read: {}", discovery.files.len());
            println!("modules: {}", discovery.modules.len());
            println!("bf16-tensors: {}", discovery.bf16.len());
            println!("selected: {}", discovery.selected());
            println!("skipped: {}", discovery.skipped.len());
            println!("selection-bytes: {}", text.len());
            // Parsed back before it is offered: a selection this program could
            // not read is not a selection, and finding that out at repack time
            // would waste the user's next command.
            if let Err(e) = moxie_format::selection::parse(&text) {
                println!("outcome: refused");
                eprintln!("moxie-repack: the generated selection does not parse: {e}");
                return 2;
            }
            // Published by rename: a plan is either the whole document or it
            // is not there, never a half-written file a second command would
            // read.
            // **Create it, never open what is already there.** `fs::write`
            // follows a symlink, so a staging path that someone has pointed
            // into the read-only checkpoint would be written through before the
            // rename ever happened -- independent review overwrote a
            // `config.json` that way. `create_new` refuses any existing path,
            // symlink included, and `O_NOFOLLOW` closes the window between the
            // check and the open.
            let staging = out_selection.with_extension("toml.partial");
            match std::fs::symlink_metadata(&staging) {
                // A symbolic link here is not a leftover, it is a redirection.
                Ok(m) if m.file_type().is_symlink() => {
                    println!("outcome: refused");
                    eprintln!(
                        "moxie-repack: {} is a symbolic link. This program writes only files it \
                         creates, and following one would write wherever it points",
                        staging.display()
                    );
                    return 2;
                }
                // A leftover from an interrupted plan is ours to clear.
                Ok(_) => {
                    if let Err(e) = std::fs::remove_file(&staging) {
                        println!("outcome: refused");
                        eprintln!("moxie-repack: cannot clear {}: {e}", staging.display());
                        return 2;
                    }
                }
                Err(_) => {}
            }
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                const O_NOFOLLOW: i32 = 0o400000;
                options.custom_flags(O_NOFOLLOW);
            }
            if let Err(e) = options
                .open(&staging)
                .and_then(|mut f| std::io::Write::write_all(&mut f, text.as_bytes()))
                .and_then(|()| std::fs::rename(&staging, &out_selection))
            {
                let _ = std::fs::remove_file(&staging);
                println!("outcome: refused");
                eprintln!(
                    "moxie-repack: cannot write {}: {e}",
                    out_selection.display()
                );
                return 2;
            }
            println!("outcome: planned");
            println!("selection: {}", out_selection.display());
            match &discovery.checkpoint.quantization {
                Some(q) => println!(
                    "quantization: {}-bit, {:?}, {:?}",
                    q.bits, q.granularity, q.zero_points
                ),
                None => println!("quantization: none declared; BF16 tensors only"),
            }
            for (what, why) in discovery.skipped.iter().take(20) {
                println!("  skipped {what}: {why}");
            }
            if discovery.skipped.len() > 20 {
                println!(
                    "  ... and {} more, all in the file",
                    discovery.skipped.len() - 20
                );
            }
            println!();
            println!("Read it, edit it if you want, then:");
            // **The two-command form.** This printed the advanced invocation --
            // `--selection`, `--source-root` and five budgets -- which is the
            // one this task exists to stop a normal user needing. The plan
            // carries the root and the budgets; the second command needs
            // neither.
            println!(
                "  moxie-repack repack --plan {} --out <dir>",
                out_selection.display()
            );
            0
        }
        Err(e) => {
            println!("outcome: refused");
            eprintln!("moxie-repack: {e}");
            2
        }
    }
}

fn command_inspect(flags: &mut Flags) -> i32 {
    let (Some(selection_path), Some(root)) = (flags.selection.clone(), flags.source_root.clone())
    else {
        eprintln!("moxie-repack: inspect needs --selection and --source-root");
        return 1;
    };
    if flags.out.is_some() {
        // Stated rather than ignored: inspection has no output-directory side
        // effects, so accepting a destination would be accepting an argument
        // that changes nothing.
        eprintln!("moxie-repack: inspect takes no --out; it never writes anything");
        return 1;
    }
    let budgets = match flags.budgets() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("moxie-repack: {e}");
            return 1;
        }
    };
    let report = (|| {
        let selection = moxie_repack::read_selection(&selection_path)?;
        let mut sources = moxie_repack::open_sources(&root, &budgets)?;
        // An inspection admits what it is about to build, against the same
        // total a repack would: reporting a cost is itself work with a cost.
        let mut ledger = moxie_repack::ledger_for(&budgets)?;
        moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger)
    })();
    match report {
        Ok(r) => {
            print_inspect(&r, &budgets);
            0
        }
        Err(e) => {
            println!("outcome: refused");
            println!("error: {e}");
            2
        }
    }
}

fn print_inspect(r: &InspectReport, budgets: &Budgets) {
    println!("model: {}", r.model);
    println!("revision: {}", r.revision);
    println!("source-root: {}", r.source_root.display());
    println!("selection-digest: {}", r.selection_digest);
    println!("plan-digest: {}", r.plan_digest);
    println!("source-files: {}", r.files.join(", "));
    println!("headers-parsed: {}", r.headers_parsed);
    println!("header-bytes-read: {}", r.header_bytes_read);
    println!("tensors: {}", r.tensors.len());
    for t in &r.tensors {
        println!(
            "tensor: {} | {} | shape {:?} | canonical {} B | source {} B | {}@{} | {} unit(s)",
            t.role,
            t.profile,
            t.shape,
            t.canonical_bytes,
            t.source_bytes,
            t.chunk,
            t.offset,
            t.units
        );
    }
    println!("canonical-payload-bytes: {}", r.canonical_payload_bytes);
    println!(
        "shard-bytes: {} (the components plus each shard's header)",
        r.shard_bytes
    );
    println!("chunk-files: {}", r.chunk_files);
    println!("largest-chunk-bytes: {}", r.largest_chunk_bytes);
    println!("source-payload-bytes: {}", r.source_payload_bytes);
    println!("staging-disk-peak-bytes: {}", r.staging.total());
    println!(
        "staging-disk-peak-detail: payload {} B exactly, plus at most {} B of journal and {} B of \
         staged manifest, both removed at publication",
        r.staging.payload_bytes, r.staging.journal_bound_bytes, r.staging.manifest_bound_bytes
    );
    println!(
        "note: free space is not checked here; a disk-full condition is a typed failure at the \
         write that hits it, not something an estimate prevents"
    );
    println!("admitted-ram-bytes: {}", budgets.total_bytes);
    println!("payload-scratch-bytes: {}", budgets.scratch_bytes);
    println!("header-budget-bytes: {}", budgets.header_bytes);
    match &r.completeness {
        moxie_format::manifest::Completeness::Complete => {
            println!("completeness: complete");
            println!(
                "note: this selection claims to be the whole model; a reader will load it as one"
            );
        }
        moxie_format::manifest::Completeness::Partial { missing } => {
            println!("completeness: partial");
            println!("missing: {}", missing.join(", "));
            println!(
                "note: a partial artifact opens for inspection and refuses every tensor read; a \
                 completed selection is not a complete model"
            );
        }
    }
    // Time is either an estimate with a rate this program was given, or it is
    // unmeasured. It has not been given one, and inventing a disk rate here
    // would be inventing the number a user would plan with.
    println!("estimated-time: unmeasured (no measured or supplied I/O rate)");
    println!(
        "note: checksums here are header-derived expectations; only a repack reads and hashes \
         payload bytes"
    );
    println!("outcome: inspected");
}

impl Flags {
    /// Budgets from the command line, falling back **field by field** to what a
    /// plan recorded.
    ///
    /// Precedence is what the docs promise: flag, then plan, then the automatic
    /// value the plan already resolved. All-or-nothing fallback made that
    /// promise false.
    fn merged_budgets(&self, plan: Option<[u64; 5]>) -> Result<Budgets, String> {
        let pick = |flag: Option<u64>, index: usize, name: &str| -> Result<u64, String> {
            match (flag, plan) {
                (Some(v), _) => Ok(v),
                (None, Some(p)) => Ok(p[index]),
                (None, None) => Err(format!(
                    "{name} is required: this document records no resolved options, so every \
                     budget has to be given"
                )),
            }
        };
        let scratch = pick(self.scratch_bytes, 2, "--scratch-bytes")?;
        Ok(Budgets {
            total_bytes: pick(self.total_bytes, 0, "--total-bytes")?,
            header_bytes: pick(self.header_bytes, 1, "--header-bytes")?,
            scratch_bytes: usize::try_from(scratch)
                .map_err(|_| "--scratch-bytes does not fit this platform".to_string())?,
            chunk_file_bytes: pick(self.chunk_file_bytes, 3, "--chunk-file-bytes")?,
            disk_bytes: pick(self.disk_bytes, 4, "--disk-bytes")?,
        })
    }
}

fn command_repack(flags: &mut Flags) -> i32 {
    // Two ways in. `--plan` is the normal one: the plan carries where its
    // checkpoint is and what it resolved, so neither is asked for twice.
    // `--selection` with explicit flags is the advanced one, unchanged.
    let from_plan = flags.plan.is_some();
    let selection_path = match (flags.plan.clone(), flags.selection.clone()) {
        (Some(_), Some(_)) => {
            eprintln!("moxie-repack: pass --plan or --selection, not both");
            return 1;
        }
        (Some(p), None) | (None, Some(p)) => p,
        (None, None) => {
            eprintln!(
                "moxie-repack: repack needs --plan <file> --out <dir>\n\
                 \n\
                 Generate a plan first:\n\
                 \x20 moxie-repack plan --source-root <checkpoint> --out-plan <file>"
            );
            return 1;
        }
    };
    let Some(out) = flags.out.clone() else {
        eprintln!("moxie-repack: repack needs --out <dir>");
        return 1;
    };
    let plan_text = match std::fs::read_to_string(&selection_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "moxie-repack: cannot read {}: {e}",
                selection_path.display()
            );
            return 1;
        }
    };
    // The plan's root, unless the command line names one: explicit wins.
    let root = match flags.source_root.clone() {
        Some(r) => r,
        None => match moxie_format::plan::source_root(&plan_text) {
            Ok(Some(r)) => PathBuf::from(r),
            _ => {
                eprintln!(
                    "moxie-repack: this document records no source root, so --source-root is \
                     needed. A plan from `moxie-repack plan` carries one"
                );
                return 1;
            }
        },
    };
    // **Per field, not all or nothing.** `--disk-bytes N` with nothing else used
    // to fall back to every value the plan recorded, silently discarding the one
    // number the user gave. Each field takes the flag when there is one, then
    // the plan's value; a field with neither is refused by name.
    let budgets = match flags
        .merged_budgets(moxie_format::plan::resolved_options(&plan_text).unwrap_or(None))
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("moxie-repack: {e}");
            return 1;
        }
    };
    println!(
        "settings: total={} header={} scratch={} shard-max={} disk={}",
        budgets.total_bytes,
        budgets.header_bytes,
        budgets.scratch_bytes,
        budgets.chunk_file_bytes,
        budgets.disk_bytes
    );
    let faults = flags.faults();
    let options = Options {
        take_over_interrupted_run: flags.take_over,
    };
    // Cancellation and the simulated crash are both counted in units, because
    // a unit is the boundary the state machine is defined over.
    static UNITS: AtomicUsize = AtomicUsize::new(0);
    static CANCEL: AtomicBool = AtomicBool::new(false);
    let cancel_after = flags.cancel_after_units;
    let abort_after = flags.abort_after_units;
    let cancelled = move || CANCEL.load(Ordering::SeqCst);

    let result = (|| {
        let selection = moxie_repack::read_selection(&selection_path)?;
        // **Is this still the checkpoint the plan describes?** Asked before
        // anything is created. A plan binds the config and index it was
        // generated from, so a checkpoint that has gained or lost tensors is
        // refused rather than published as the old subset.
        moxie_repack::discover::confirm_binding(&root, &plan_text)?;
        // **A partial conversion is not a model.** The normal path refuses one,
        // because an artifact missing tensors that opens and verifies is the
        // most expensive kind of wrong. Converting a subset on purpose stays
        // available and has to be asked for.
        // **Only on the generated path.** `--plan` means "convert this model",
        // so a plan that does not cover it is refused. A hand-written selection
        // that declares `status = "partial"` is already a deliberate subset --
        // the person wrote that line -- and refusing it would break the
        // explicit path this flag exists to preserve.
        if from_plan
            && let moxie_format::selection::Completeness::Partial { missing } =
                &selection.completeness
            && !flags.allow_partial
        {
            let shown: Vec<&str> = missing.iter().take(3).map(String::as_str).collect();
            return Err(moxie_types::Error::InvalidArtifact {
                detail: format!(
                    "this plan is partial: {} tensor(s) are not covered, starting with {shown:?}. A partial artifact opens for inspection and refuses every read, so it is not a model. Pass --allow-partial to convert the covered subset deliberately",
                    missing.len()
                )
                .into(),
            });
        }
        let mut sources = moxie_repack::open_sources(&root, &budgets)?;
        let mut ledger = moxie_repack::ledger_for(&budgets)?;
        let mut progress = |line: &str| {
            println!("progress: {line}");
            let seen = if line.starts_with("unit ") {
                UNITS.fetch_add(1, Ordering::SeqCst) + 1
            } else {
                UNITS.load(Ordering::SeqCst)
            };
            if let Some(n) = cancel_after
                && seen >= n
            {
                CANCEL.store(true, Ordering::SeqCst);
            }
            if let Some(n) = abort_after
                && seen >= n
            {
                println!("progress: aborting after {seen} unit(s), as asked");
                // A real crash: no unwinding, no cleanup, no journal line.
                // This is the state a restart has to cope with.
                std::process::abort();
            }
        };
        moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &options,
            &faults,
            &cancelled,
            &mut ledger,
            &mut progress,
        )
    })();

    match result {
        Ok(report) => {
            print_repack(&report);
            match report.outcome {
                Outcome::Published { .. } => 0,
                Outcome::PublishedDurabilityUnconfirmed { .. } => 4,
                Outcome::Cancelled { .. } => 3,
            }
        }
        Err(e) => {
            println!("outcome: failed");
            println!("error: {e}");
            println!(
                "note: the destination holds a resumable private state; run the same command \
                 again to continue it"
            );
            2
        }
    }
}

fn print_repack(r: &RepackReport) {
    for line in &r.resume_detail {
        println!("resume: {line}");
    }
    for (file, digest) in &r.source_digests {
        println!("source-digest: {file} {digest}");
    }
    println!("resumed: {}", r.resumed);
    println!("units-written: {}", r.units_written);
    println!("units-reused: {}", r.units_reused);
    println!("bytes-written: {}", r.bytes_written);
    println!("source-bytes-read: {}", r.source_bytes_read);
    match &r.outcome {
        Outcome::Published { artifact, bytes } => {
            println!("artifact: {}", artifact.display());
            println!("artifact-identity: {}", r.artifact_identity);
            println!("published-bytes: {bytes}");
            println!("outcome: published");
        }
        Outcome::PublishedDurabilityUnconfirmed { artifact, detail } => {
            println!("artifact: {}", artifact.display());
            println!("artifact-identity: {}", r.artifact_identity);
            println!("durability-detail: {detail}");
            println!("outcome: published-durability-unconfirmed");
        }
        Outcome::Cancelled {
            destination,
            bytes_done,
        } => {
            println!("destination: {}", destination.display());
            println!("bytes-done: {bytes_done}");
            println!("outcome: cancelled");
        }
    }
}

fn command_verify(flags: &mut Flags) -> i32 {
    let Some(artifact) = flags.artifact.clone() else {
        eprintln!("moxie-repack: verify needs --artifact");
        return 1;
    };
    let scratch_bytes = match flags.scratch_bytes {
        Some(n) => n,
        None => {
            eprintln!(
                "moxie-repack: verify needs --scratch-bytes: it reads every payload byte through \
                 a buffer you sized"
            );
            return 1;
        }
    };
    let Ok(scratch_bytes) = usize::try_from(scratch_bytes) else {
        eprintln!("moxie-repack: --scratch-bytes does not fit this platform");
        return 1;
    };
    // Reserved fallibly: this size comes from the command line, and an
    // allocator that aborts turns "--scratch-bytes 1TiB" into a crash with no
    // message. It is the one allocation in this program a user sizes directly.
    let want = scratch_bytes.max(1);
    let mut scratch: Vec<u8> = Vec::new();
    if scratch.try_reserve_exact(want).is_err() {
        println!("outcome: refused");
        println!("error: cannot reserve {want} byte(s) of verification scratch");
        return 2;
    }
    scratch.resize(want, 0);
    match moxie_repack::verify(&artifact, &mut scratch) {
        Ok(r) => {
            print_verify(&r);
            0
        }
        Err(e) => {
            println!("outcome: failed");
            println!("error: {e}");
            2
        }
    }
}

fn print_verify(r: &VerifyReport) {
    println!("artifact: {}", r.artifact.display());
    println!("artifact-identity: {}", r.identity);
    println!("tensors: {}", r.tensors);
    println!("bytes-verified: {}", r.bytes_verified);
    println!(
        "unclaimed-bytes: {} (alignment padding between tensors lands here; anything more is a \
         chunk holding bytes no tensor describes)",
        r.unclaimed_bytes
    );
    match &r.completeness {
        moxie_format::manifest::Completeness::Complete => println!("completeness: complete"),
        moxie_format::manifest::Completeness::Partial { missing } => {
            println!("completeness: partial");
            println!("missing: {}", missing.join(", "));
        }
    }
    println!(
        "note: this verifies that the published bytes are the bytes their checksums name. It is \
         not a statement about what any model produces"
    );
    println!("outcome: verified");
}
