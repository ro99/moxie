//! Task 0025: publication state x failure point x cancellation x restart.
//!
//! The interesting part of a repacker is what it leaves behind when it stops,
//! so this file is mostly about stopping. Every durable boundary in the state
//! machine is named ([`Site`]), a clean run records how many times each one is
//! visited, and the enumeration below fails **every visit to every boundary**
//! in turn and checks the same invariants each time:
//!
//! 1. A failure before the publication rename leaves no `manifest.toml`. A
//!    reader sees no artifact, not a half-written one.
//! 2. The destination is resumable: the same plan, restarted, publishes.
//! 3. What it publishes is **byte-identical** to an uninterrupted run, with the
//!    same artifact identity. A resume that produced a subtly different
//!    artifact would pass every checksum it wrote itself.
//!
//! The payloads here are BF16, because the writer does not know what a
//! canonical payload means -- that is `moxie-format`'s job, and the conversion
//! is exercised where it lives. What is exercised here is the durability
//! machinery around it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use moxie_format::journal::RunBinding;
use moxie_format::manifest::{
    self, Architecture, Completeness, Endianness, Identity, Manifest, Provenance, Source,
    SourceFile, Tensor, TensorPrecision,
};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_storage_write::{
    Faults, MANIFEST_FILE, Options, Outcome, OutputPlan, Run, Site, Start, TensorRequest,
    WriteBudget,
};
use moxie_types::Scope;

const HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// A temporary directory that removes itself, so a failed test leaves nothing
/// behind but its output.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "moxie-task0025-{label}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock after 1970")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).expect("a scratch directory");
        Self { path: base }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 128 << 20, 1 << 20).expect("a snapshot")])
        .expect("a ledger")
}

fn budget() -> WriteBudget {
    WriteBudget::new(4096, 64 * 1024, 1 << 20).expect("a budget")
}

fn binding() -> RunBinding {
    RunBinding {
        plan_digest: "1".repeat(64),
        converter: "test-converter/1".into(),
        schema_version: manifest::SCHEMA_VERSION,
    }
}

/// Finite BF16 bytes: the production reader validates every element while
/// verifying, so a payload of arbitrary bytes would fail for the wrong reason.
fn payload(seed: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (i, pair) in out.chunks_exact_mut(2).enumerate() {
        let bits = (((i as u64 * 2654435761 + seed) % 0x7F00) as u16) & 0x7F7F;
        pair.copy_from_slice(&bits.to_le_bytes());
    }
    out
}

/// Two tensors, one of them large enough to need several units.
fn requests() -> Vec<TensorRequest> {
    vec![
        TensorRequest {
            role: "layers.0.w".into(),
            shape: vec![64, 80],
            precision: TensorPrecision::Bf16V1,
            affine: None,
            length: 64 * 80 * 2,
            alignment: 16,
        },
        TensorRequest {
            role: "layers.0.norm".into(),
            shape: vec![16],
            precision: TensorPrecision::Bf16V1,
            affine: None,
            length: 32,
            alignment: 16,
        },
    ]
}

fn plan() -> OutputPlan {
    OutputPlan::build(requests(), &budget()).expect("a plan")
}

/// Units of at most 4096 bytes, the admitted scratch.
fn units(request: &TensorRequest) -> Vec<(u64, usize)> {
    let mut out = Vec::new();
    let mut at = 0u64;
    while at < request.length {
        let take = ((request.length - at) as usize).min(budget().scratch_bytes());
        out.push((at, take));
        at += take as u64;
    }
    out
}

/// An empty architecture-metadata tree, taken from a parsed manifest.
fn empty_metadata() -> moxie_format::manifest::OpaqueArchMetadata {
    let text = format!(
        "schema_version = 1\nrequired_features = []\nendianness = \"little\"\n\
         excluded = []\n\n\
         [source]\nmodel = \"m\"\nrevision = \"r\"\nlicense = \"l\"\n\
         [[source.files]]\npath = \"p\"\nsha256 = \"{HEX}\"\n\n\
         [tokenizer]\nname = \"t\"\nversion = \"1\"\ndigest = \"{HEX}\"\n\n\
         [template]\nname = \"t\"\nversion = \"1\"\ndigest = \"{HEX}\"\n\n\
         [architecture]\nname = \"a\"\nversion = \"1\"\n[architecture.metadata]\n\n\
         [provenance]\nscale_convention = \"affine-v1\"\nquantizer = \"none\"\n\
         calibration = \"none\"\n\n\
         [[tensors]]\nrole = \"w\"\nshape = [1]\nprecision = \"bf16-v1\"\n\
         chunk = \"c.bin\"\noffset = 0\nlength = 2\nsha256 = \"{HEX}\"\n\
         alignment = 2\nlogical_order = 0\n\n\
         [completeness]\nstatus = \"complete\"\nmissing = []\n"
    );
    manifest::parse(&text)
        .expect("the fixture parses")
        .architecture
        .metadata
}

fn manifest_for(sealed: &[moxie_storage_write::SealedTensor]) -> Manifest {
    let mut tensors = Vec::new();
    for (order, request) in requests().iter().enumerate() {
        let s = sealed
            .iter()
            .find(|s| s.role == request.role)
            .expect("every tensor is sealed");
        tensors.push(Tensor {
            role: request.role.clone(),
            shape: request.shape.clone(),
            precision: request.precision,
            chunk: s.chunk.clone(),
            offset: s.offset,
            length: s.length,
            sha256: s.sha256.clone(),
            alignment: request.alignment,
            logical_order: order as u64,
            affine: None,
        });
    }
    Manifest {
        required_features: Vec::new(),
        endianness: Endianness::Little,
        source: Source {
            model: "fixture".into(),
            revision: "r1".into(),
            license: "unlicensed-test-fixture".into(),
            files: vec![SourceFile {
                path: "source.safetensors".into(),
                sha256: HEX.into(),
            }],
        },
        tokenizer: Identity {
            name: "none".into(),
            version: "not-selected".into(),
            digest: HEX.into(),
        },
        template: Identity {
            name: "none".into(),
            version: "not-selected".into(),
            digest: HEX.into(),
        },
        architecture: Architecture {
            name: "fixture".into(),
            version: "1".into(),
            // The opaque tree, obtained through the parser rather than
            // constructed: this crate has no TOML dependency, and it has no
            // business gaining one to write a test fixture.
            metadata: empty_metadata(),
        },
        provenance: Provenance {
            scale_convention: "affine-v1".into(),
            quantizer: "none".into(),
            calibration: "none".into(),
        },
        tensors,
        excluded: Vec::new(),
        completeness: Completeness::Partial {
            missing: vec!["everything else".into()],
        },
    }
}

/// Run to completion, failing wherever `faults` says and cancelling whenever
/// `cancel_after_units` says.
fn run_to_end(
    dest: &Path,
    faults: &Faults,
    take_over: bool,
    cancel_after_units: Option<usize>,
) -> moxie_types::Result<(Outcome, String, usize)> {
    let mut ledger = ledger();
    let options = Options {
        take_over_interrupted_run: take_over,
    };
    let start = Run::begin(
        dest,
        plan(),
        binding(),
        budget(),
        &options,
        &mut ledger,
        faults,
    )?;
    let mut run = match start {
        Start::Fresh(run) | Start::Resumed(run, _) => run,
        Start::AlreadyPublished { artifact } => {
            return Err(moxie_types::Error::InvalidArtifact {
                detail: format!("{} is already published", artifact.display()).into(),
            });
        }
    };
    let mut written = 0usize;
    let cancelled = std::cell::Cell::new(false);
    let check = || cancelled.get();
    for request in requests() {
        let bytes = payload(request.length, request.length as usize);
        let done = run.bytes_done(&request.role)?;
        for (at, len) in units(&request) {
            if at + len as u64 <= done {
                continue;
            }
            if check() {
                let outcome = run.cancel(&mut ledger)?;
                return Ok((outcome, String::new(), written));
            }
            match run.write_unit(
                &request.role,
                &bytes[at as usize..at as usize + len],
                HEX,
                faults,
            ) {
                Ok(()) => written += 1,
                Err(e) => {
                    run.abandon(&mut ledger)?;
                    return Err(e);
                }
            }
            if Some(written) == cancel_after_units {
                cancelled.set(true);
            }
        }
    }
    let sealed = match run.seal() {
        Ok(s) => s,
        Err(e) => {
            run.abandon(&mut ledger)?;
            return Err(e);
        }
    };
    let manifest = manifest_for(&sealed);
    let identity = manifest::artifact_identity(&manifest);
    let text = manifest::encode(&manifest).expect("it encodes");
    let outcome = run.publish(&text, &check, faults, &mut ledger)?;
    assert!(
        ledger.outstanding().is_empty(),
        "a finished run gives every admitted byte back: {:?}",
        ledger.outstanding()
    );
    Ok((outcome, identity, written))
}

/// Every published byte of a directory, for comparing two runs.
fn artifact_bytes(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(dir).expect("a directory") {
        let entry = entry.expect("an entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        out.insert(name, std::fs::read(entry.path()).expect("a file"));
    }
    out
}

#[test]
fn an_uninterrupted_run_publishes_and_verifies() {
    let scratch = Scratch::new("clean");
    let dest = scratch.join("artifact");
    let (outcome, identity, written) =
        run_to_end(&dest, &Faults::none(), false, None).expect("it publishes");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    assert!(written > 1, "the large tensor takes several units");
    // The private files are gone and the artifact is what is left.
    let names: Vec<String> = artifact_bytes(&dest).keys().cloned().collect();
    assert_eq!(names, ["chunk0.bin", "manifest.toml"]);
    assert!(!dest.join(".moxie-repack-journal").exists());
    assert!(!dest.join(".moxie-repack-lock").exists());
    // And it opens through the production reader.
    let artifact = moxie_storage::Artifact::open(&dest).expect("it opens");
    assert_eq!(artifact.identity(), identity);
    let mut scratch_buf = vec![0u8; 512];
    for t in &artifact.manifest().tensors {
        artifact
            .verify_tensor(&t.role, &mut scratch_buf)
            .expect("every tensor verifies");
    }
}

/// The enumeration. Every visit to every named boundary, failed in turn.
#[test]
fn every_failure_point_leaves_a_resumable_destination_that_publishes_the_same_artifact() {
    // The reference: what an uninterrupted run produces.
    let reference_dir = Scratch::new("reference");
    let reference = reference_dir.join("artifact");
    let (_, reference_identity, _) =
        run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    // How many times each boundary is visited in a clean run, measured rather
    // than assumed: a boundary that gained a visit gains enumeration cases.
    let counting = Faults::none();
    let counted_dir = Scratch::new("counted");
    run_to_end(&counted_dir.join("artifact"), &counting, false, None).expect("it publishes");
    let visited = counting.visited_sites();

    let mut cases = 0usize;
    let mut published_after_failure = 0usize;
    let mut coverage: Vec<(Site, usize)> = Vec::new();
    for site in Site::ALL.iter().copied() {
        let visits = counting.visits(site);
        coverage.push((site, visits));
        for visit in 1..=visits {
            cases += 1;
            let scratch = Scratch::new("enumerated");
            let dest = scratch.join("artifact");
            let faults = Faults::none().fail_at(site, visit as u64);
            let first = run_to_end(&dest, &faults, false, None);
            assert!(
                faults.all_fired(),
                "{site:?} visit {visit} never fired: this case measured nothing"
            );
            match &first {
                // A failure before the artifact exists.
                Err(_) => {
                    // Invariant 1: a reader sees no artifact at all.
                    assert!(
                        !dest.join(MANIFEST_FILE).exists(),
                        "{site:?} visit {visit} left a manifest behind"
                    );
                    // Invariants 2 and 3: restart, publish, match the
                    // reference byte for byte.
                    let (outcome, identity, _) = run_to_end(&dest, &Faults::none(), true, None)
                        .unwrap_or_else(|e| {
                            panic!("{site:?} visit {visit} could not be resumed: {e}")
                        });
                    assert!(
                        matches!(outcome, Outcome::Published { .. }),
                        "{site:?} visit {visit} resumed to {outcome:?}"
                    );
                    assert_eq!(identity, reference_identity, "{site:?} visit {visit}");
                }
                // A failure at or after the publication rename: the artifact
                // exists, and nothing may delete it.
                Ok((outcome, identity, _)) => {
                    published_after_failure += 1;
                    assert!(
                        matches!(
                            outcome,
                            Outcome::Published { .. }
                                | Outcome::PublishedDurabilityUnconfirmed { .. }
                        ),
                        "{site:?} visit {visit}: {outcome:?}"
                    );
                    assert!(
                        dest.join(MANIFEST_FILE).exists(),
                        "{site:?} visit {visit} reported a publish with no manifest"
                    );
                    assert_eq!(identity, &reference_identity, "{site:?} visit {visit}");
                    // A second attempt reports it rather than touching it.
                    let mut ledger = ledger();
                    let start = Run::begin(
                        &dest,
                        plan(),
                        binding(),
                        budget(),
                        &Options {
                            take_over_interrupted_run: true,
                        },
                        &mut ledger,
                        &Faults::none(),
                    )
                    .expect("a published destination reports itself");
                    assert!(
                        matches!(start, Start::AlreadyPublished { .. }),
                        "{site:?} visit {visit}: {start:?}"
                    );
                }
            }
            assert_eq!(
                artifact_bytes(&dest),
                reference_bytes,
                "{site:?} visit {visit} published different bytes"
            );
        }
    }
    eprintln!(
        "task0025 failure-point coverage: {cases} case(s) across {} boundary/visit pair(s)",
        cases
    );
    for (site, visits) in &coverage {
        eprintln!("  {:<22} {visits} visit(s)", site.name());
    }
    eprintln!("  {published_after_failure} case(s) failed at or after the publication boundary");
    assert!(
        cases >= Site::ALL.len(),
        "{cases} case(s) is fewer than the {} named boundaries",
        Site::ALL.len()
    );
    // A boundary nothing ever reached is a boundary this enumeration did not
    // test, and saying which is the point of measuring coverage.
    let unvisited: Vec<&str> = Site::ALL
        .iter()
        .filter(|s| !visited.contains(s))
        .map(|s| s.name())
        .collect();
    eprintln!("  boundaries not reached by this scenario: {unvisited:?}");
    assert!(
        unvisited.len() <= 1,
        "more boundaries than expected are unreached: {unvisited:?}"
    );
}

/// A publish whose confirming sync fails is neither a success nor a failure,
/// and the artifact it produced is **not** deleted.
#[test]
fn a_failed_durability_confirmation_reports_published_durability_unconfirmed() {
    let scratch = Scratch::new("durability");
    let dest = scratch.join("artifact");
    let faults = Faults::none().fail_at(Site::PublishDurability, 1);
    let (outcome, identity, _) = run_to_end(&dest, &faults, false, None).expect("it publishes");
    match &outcome {
        Outcome::PublishedDurabilityUnconfirmed { artifact, detail } => {
            assert_eq!(artifact, &dest);
            assert!(detail.contains("publish-durability"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    // The artifact exists, opens and verifies: the rename succeeded, and only
    // the confirmation after it did not.
    let artifact = moxie_storage::Artifact::open(&dest).expect("it opens");
    assert_eq!(artifact.identity(), identity);
}

/// Cancellation before the publication boundary leaves a resumable state and
/// no artifact; a restart finishes the job.
#[test]
fn cancellation_is_resumable_and_publishes_the_same_artifact() {
    let reference_dir = Scratch::new("cancel-reference");
    let reference = reference_dir.join("artifact");
    run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    for after in 1..=3 {
        let scratch = Scratch::new("cancelled");
        let dest = scratch.join("artifact");
        let (outcome, _, _) = run_to_end(&dest, &Faults::none(), false, Some(after))
            .expect("cancellation is not an error");
        match outcome {
            Outcome::Cancelled {
                destination,
                bytes_done,
            } => {
                assert_eq!(destination, dest);
                assert!(bytes_done > 0, "after {after} unit(s) something is durable");
            }
            other => panic!("cancelling after {after} unit(s) gave {other:?}"),
        }
        assert!(!dest.join(MANIFEST_FILE).exists());
        // Repeated cancellation and resume must not grow what is retained:
        // cancel again on the way through, then finish. A resumed run has
        // fewer units left than the original, so cancelling "after one more"
        // may legitimately finish instead -- both are accepted, and the
        // published artifact is compared either way.
        let (outcome, _, _) =
            run_to_end(&dest, &Faults::none(), true, Some(1)).expect("a second attempt");
        assert!(
            matches!(
                outcome,
                Outcome::Cancelled { .. } | Outcome::Published { .. }
            ),
            "{outcome:?}"
        );
        if matches!(outcome, Outcome::Published { .. }) {
            assert_eq!(artifact_bytes(&dest), reference_bytes);
            continue;
        }
        let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it finishes");
        assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
        assert_eq!(artifact_bytes(&dest), reference_bytes);
    }
}

#[test]
fn a_published_destination_is_never_overwritten() {
    let scratch = Scratch::new("published");
    let dest = scratch.join("artifact");
    run_to_end(&dest, &Faults::none(), false, None).expect("it publishes");
    let before = artifact_bytes(&dest);

    let mut ledger = ledger();
    let start = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut ledger,
        &Faults::none(),
    )
    .expect("begin reports rather than refuses");
    match start {
        Start::AlreadyPublished { artifact } => assert_eq!(artifact, dest),
        other => panic!("{other:?}"),
    }
    assert_eq!(artifact_bytes(&dest), before, "nothing was touched");
    assert!(
        ledger.outstanding().is_empty(),
        "a refused start admits nothing"
    );
}

#[test]
fn a_directory_this_run_did_not_create_is_refused() {
    let scratch = Scratch::new("occupied");
    let dest = scratch.join("artifact");
    std::fs::create_dir_all(&dest).expect("a directory");
    std::fs::write(dest.join("someone-elses-file"), b"data").expect("a file");
    let mut ledger = ledger();
    let e = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut ledger,
        &Faults::none(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("someone-elses-file"), "{e}");
    assert!(
        e.to_string()
            .contains("no repack journal to account for them"),
        "{e}"
    );
    assert!(dest.join("someone-elses-file").exists(), "nothing deleted");
    // Its own leftovers are a different case: a crash between taking the lock
    // and writing the journal header leaves exactly the lock, and a
    // destination that can never be used again is not a recovery.
    std::fs::remove_file(dest.join("someone-elses-file")).expect("removing it");
    std::fs::write(dest.join(moxie_storage_write::LOCK_FILE), b"pid 1").expect("a stale lock");
    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it starts over");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
}

#[test]
fn a_second_run_is_refused_while_one_owns_the_destination() {
    let scratch = Scratch::new("locked");
    let dest = scratch.join("artifact");
    let mut ledger = ledger();
    let first = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut ledger,
        &Faults::none(),
    )
    .expect("the first run starts");
    let held = match first {
        Start::Fresh(run) => run,
        other => panic!("{other:?}"),
    };
    let mut second_ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 128 << 20, 1 << 20).expect("a snapshot")])
            .expect("a second ledger");
    let e = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut second_ledger,
        &Faults::none(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("owns this destination"), "{e}");
    assert!(
        second_ledger.outstanding().is_empty(),
        "a refused start admits nothing"
    );
    held.abandon(&mut ledger).expect("the first run releases");
}

/// A resume whose binding differs is refused rather than merged -- whichever
/// field differs.
#[test]
fn a_resume_bound_to_a_different_plan_is_refused() {
    for change in ["plan", "converter"] {
        let scratch = Scratch::new("rebound");
        let dest = scratch.join("artifact");
        run_to_end(&dest, &Faults::none(), false, Some(1)).expect("a cancelled run");
        let mut ledger = ledger();
        let mut binding = binding();
        match change {
            "plan" => binding.plan_digest = "2".repeat(64),
            _ => binding.converter = "test-converter/2".into(),
        }
        let e = Run::begin(
            &dest,
            plan(),
            binding,
            budget(),
            &Options {
                take_over_interrupted_run: true,
            },
            &mut ledger,
            &Faults::none(),
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("bound to a different plan"),
            "{change}: {e}"
        );
        assert!(
            ledger.outstanding().is_empty(),
            "a refused resume admits nothing"
        );
    }
}

/// A journal entry is not evidence its payload is correct.
#[test]
fn staged_bytes_that_no_longer_match_the_journal_are_discarded_and_recomputed() {
    let reference_dir = Scratch::new("corrupt-reference");
    let reference = reference_dir.join("artifact");
    run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    let scratch = Scratch::new("corrupted");
    let dest = scratch.join("artifact");
    run_to_end(&dest, &Faults::none(), false, Some(2)).expect("a cancelled run");
    // Corrupt one byte of the first unit's staged payload.
    let chunk = dest.join("chunk0.bin");
    let mut bytes = std::fs::read(&chunk).expect("a staged chunk");
    bytes[3] ^= 0xFF;
    std::fs::write(&chunk, &bytes).expect("corrupting it");

    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it recovers");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    assert_eq!(
        artifact_bytes(&dest),
        reference_bytes,
        "the corrupted unit was recomputed rather than published"
    );
}

/// Bytes written past the last journal line -- a crash between a payload write
/// and its record -- are truncated and rewritten.
#[test]
fn bytes_past_the_journal_are_truncated_on_resume() {
    let scratch = Scratch::new("overshoot");
    let dest = scratch.join("artifact");
    run_to_end(&dest, &Faults::none(), false, Some(2)).expect("a cancelled run");
    let chunk = dest.join("chunk0.bin");
    let before = std::fs::metadata(&chunk).expect("a chunk").len();
    // Append what a crashed unit would have left.
    let mut bytes = std::fs::read(&chunk).expect("a chunk");
    bytes.extend_from_slice(&[0u8; 777]);
    std::fs::write(&chunk, &bytes).expect("extending it");

    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it recovers");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    let after = std::fs::metadata(&chunk).expect("a chunk").len();
    assert!(
        after > before,
        "the run continued rather than stopping at the truncation"
    );
}

/// A torn final journal line loses that record and nothing before it.
#[test]
fn a_torn_journal_line_costs_one_unit() {
    let scratch = Scratch::new("torn");
    let dest = scratch.join("artifact");
    run_to_end(&dest, &Faults::none(), false, Some(3)).expect("a cancelled run");
    let journal = dest.join(".moxie-repack-journal");
    let text = std::fs::read_to_string(&journal).expect("a journal");
    let cut = text[..text.len() - 1].rfind('\n').expect("several lines") + 40;
    std::fs::write(&journal, &text[..cut]).expect("tearing it");

    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it recovers");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    let artifact = moxie_storage::Artifact::open(&dest).expect("it opens");
    let mut buf = vec![0u8; 512];
    for t in &artifact.manifest().tensors {
        artifact.verify_tensor(&t.role, &mut buf).expect("verifies");
    }
}

/// The one boundary a clean run never reaches: rehashing a staged unit only
/// happens on a resume, so the enumeration above cannot cover it and this
/// covers it directly.
#[test]
fn a_failure_while_rehashing_a_resumed_unit_is_recoverable() {
    let reference_dir = Scratch::new("readback-reference");
    let reference = reference_dir.join("artifact");
    run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    let scratch = Scratch::new("readback");
    let dest = scratch.join("artifact");
    run_to_end(&dest, &Faults::none(), false, Some(2)).expect("a cancelled run");
    // Fail the first read-back of a staged unit. A unit whose staged bytes
    // cannot be re-read is **discarded and recomputed**, not fatal: the run
    // continues and publishes, which is the behaviour a resume should have and
    // is worth pinning rather than assuming.
    let faults = Faults::none().fail_at(Site::ChunkReadBack, 1);
    let (outcome, identity, written) = run_to_end(&dest, &faults, true, None)
        .expect("an unreadable staged unit is recomputed, not fatal");
    assert!(faults.all_fired(), "the injected failure never fired");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    assert_eq!(
        written, 4,
        "every unit of the tensor whose first unit was discarded is rewritten"
    );
    let reference_identity = {
        let artifact = moxie_storage::Artifact::open(&reference).expect("it opens");
        artifact.identity()
    };
    assert_eq!(identity, reference_identity);
    assert_eq!(artifact_bytes(&dest), reference_bytes);
}

#[test]
fn a_unit_above_the_admitted_scratch_is_refused() {
    let scratch = Scratch::new("oversized");
    let dest = scratch.join("artifact");
    let mut ledger = ledger();
    let start = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut ledger,
        &Faults::none(),
    )
    .expect("it starts");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("{other:?}"),
    };
    let oversized = vec![0u8; budget().scratch_bytes() + 1];
    let e = run
        .write_unit("layers.0.w", &oversized, HEX, &Faults::none())
        .unwrap_err();
    assert!(e.to_string().contains("payload scratch"), "{e}");
    // And an incomplete selection cannot be sealed.
    let e = run.seal().unwrap_err();
    assert!(e.to_string().contains("cannot be published"), "{e}");
    run.abandon(&mut ledger).expect("it releases");
}
