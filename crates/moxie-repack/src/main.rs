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

  moxie-repack inspect --selection <file> --source-root <dir> <budgets>
  moxie-repack repack  --selection <file> --source-root <dir> --out <dir> <budgets> [options]
  moxie-repack verify  --artifact <dir> --scratch-bytes <n>

Budgets (all required; there is no default for what a tool may spend on a
user's machine):
  --total-bytes <n>        total admitted dynamic working memory
  --header-bytes <n>       peak heap admitted for one source header
  --scratch-bytes <n>      payload scratch: one source tile plus one canonical tile
  --chunk-file-bytes <n>   largest output chunk file (a disk plan, not RAM)
  --disk-bytes <n>         total payload bytes this run may write

Options:
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
        moxie_repack::inspect(&selection, &mut sources, &budgets)
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

fn command_repack(flags: &mut Flags) -> i32 {
    let (Some(selection_path), Some(root), Some(out)) = (
        flags.selection.clone(),
        flags.source_root.clone(),
        flags.out.clone(),
    ) else {
        eprintln!("moxie-repack: repack needs --selection, --source-root and --out");
        return 1;
    };
    let budgets = match flags.budgets() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("moxie-repack: {e}");
            return 1;
        }
    };
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
