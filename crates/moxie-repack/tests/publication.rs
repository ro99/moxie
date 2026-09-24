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
use moxie_repack::write::{
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
            "moxie-repack-{label}-{}-{:?}",
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
    [("layers.0.w", vec![64u64, 80]), ("layers.0.norm", vec![16])]
        .into_iter()
        .map(|(role, shape)| TensorRequest {
            role: role.into(),
            components: moxie_format::canonical::bf16_components(role, &shape)
                .expect("a bf16 component"),
            shape,
            precision: TensorPrecision::Bf16V1,
            affine: None,
        })
        .collect()
}

/// One request's single component, which is what a unit is written against.
fn component_of(request: &TensorRequest) -> String {
    request.components[0].name.clone()
}

fn request_len(request: &TensorRequest) -> u64 {
    request.payload_bytes()
}

fn plan() -> OutputPlan {
    // The disk budget now covers the journal and the staged manifest too,
    // so the plan is built with the same overhead bound the program uses.
    OutputPlan::build(requests(), &budget(), 8 * 1024).expect("a plan")
}

/// Units of at most 4096 bytes, the admitted scratch.
fn units(request: &TensorRequest) -> Vec<(u64, usize)> {
    let mut out = Vec::new();
    let mut at = 0u64;
    let len = request_len(request);
    while at < len {
        let take = ((len - at) as usize).min(budget().scratch_bytes());
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

fn manifest_for(sealed: &[moxie_repack::write::SealedTensor]) -> Manifest {
    let mut tensors = Vec::new();
    for (order, request) in requests().iter().enumerate() {
        tensors.push(Tensor {
            role: request.role.clone(),
            shape: request.shape.clone(),
            precision: request.precision,
            logical_order: order as u64,
            affine: None,
            placement: moxie_format::manifest::Placement::Components(
                sealed
                    .iter()
                    .filter(|s| s.role == request.role)
                    .map(|s| moxie_format::manifest::Component {
                        kind: s.kind,
                        file: s.file.clone(),
                        name: s.name.clone(),
                        sha256: s.sha256.clone(),
                    })
                    .collect(),
            ),
        });
    }
    Manifest {
        schema_version: manifest::SCHEMA_VERSION,
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
        &|| false,
    )?;
    let mut run = match start {
        Start::Fresh(run) | Start::Resumed(run, _) => run,
        Start::AlreadyPublished { artifact } => {
            return Err(moxie_types::Error::InvalidArtifact {
                detail: format!("{} is already published", artifact.display()).into(),
            });
        }
        Start::Cancelled { destination } => {
            return Ok((
                Outcome::Cancelled {
                    destination,
                    bytes_done: 0,
                },
                String::new(),
                0,
            ));
        }
    };
    let mut written = 0usize;
    let cancelled = std::cell::Cell::new(false);
    let check = || cancelled.get();
    for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        let done = run.bytes_done(&component_of(&request))?;
        for (at, len) in units(&request) {
            if at + len as u64 <= done {
                continue;
            }
            if check() {
                let outcome = run.cancel(&mut ledger)?;
                return Ok((outcome, String::new(), written));
            }
            match run.write_unit(
                &component_of(&request),
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
    assert_eq!(
        names,
        ["manifest.toml", "model-00001-of-00001.safetensors"],
        "a published artifact is a manifest plus conforming shards (ADR 0025)"
    );
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

/// Task 0031: the validation pass is the only thing standing between sealed
/// bytes and a published artifact, and until now nothing made it prove that.
///
/// `published-validation-skipped` -- the substitution that drops the
/// `Site::Validate` check and empties the role list, so not one payload byte is
/// read back -- **survived the battery** on task 0030's tree. It could, because
/// every other test in this file publishes bytes the writer itself just wrote
/// and hashed: skipping a read-back changes nothing when nothing on disk has
/// changed since it was written. The gate can only mean something against bytes
/// that went wrong **after** the writer was finished with them.
///
/// So this corrupts the staged payload in the one window the validation pass
/// owns: after `seal`, before `publish`. The unit checksums were taken as the
/// bytes streamed past, the read-back boundary is only reached on a resume or a
/// rehash, and nothing else looks at the payload again. A run that publishes
/// this artifact has published bytes nobody checked.
///
/// **One bit, in a mantissa's low byte.** Not a byte, and not the high byte:
/// flipping an exponent would also trip the BF16 finiteness check, and then
/// this test would pass for a reason that has nothing to do with checksums --
/// the "two rules that can both fire need a case each" shape experiment 0006
/// already recorded once. The value stays finite, and the only thing wrong with
/// it is that it is not the value that was hashed.
#[test]
fn a_staged_payload_corrupted_after_sealing_is_refused_before_publication() {
    let scratch = Scratch::new("corrupt-after-seal");
    let dest = scratch.join("artifact");
    let mut ledger = ledger();
    let faults = Faults::none();
    let options = Options {
        take_over_interrupted_run: false,
    };
    let start = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &options,
        &mut ledger,
        &faults,
        &|| false,
    )
    .expect("a fresh run");
    let mut run = match start {
        Start::Fresh(run) => run,
        _ => panic!("an empty destination gives a fresh run"),
    };
    for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        for (at, len) in units(&request) {
            run.write_unit(
                &component_of(&request),
                &bytes[at as usize..at as usize + len],
                HEX,
                &faults,
            )
            .expect("every unit is written");
        }
    }
    let sealed = run.seal().expect("it seals");

    // The shard is staged under its final name -- only the manifest is
    // private -- so the bytes a reader would get are already on disk here.
    let shard = dest.join("model-00001-of-00001.safetensors");
    let mut raw = std::fs::read(&shard).expect("the staged shard");
    let header_len = u64::from_le_bytes(raw[..8].try_into().expect("a length prefix")) as usize;
    let payload_start = 8 + header_len;
    assert!(
        payload_start < raw.len(),
        "the staged shard has {} byte(s) and a payload starting at {payload_start}",
        raw.len()
    );
    let before = raw[payload_start];
    // Little-endian BF16: the low byte is mantissa, so no exponent moves and
    // nothing becomes non-finite.
    raw[payload_start] = before ^ 0x01;
    std::fs::write(&shard, &raw).expect("the corruption lands");

    let manifest = manifest_for(&sealed);
    let text = manifest::encode(&manifest).expect("it encodes");
    let error = run
        .publish(&text, &|| false, &faults, &mut ledger)
        .expect_err(
            "one wrong bit in the staged payload must stop publication: the validation pass \
             exists to read those bytes back",
        );
    let detail = error.to_string();
    assert!(
        detail.contains("checksum mismatch"),
        "the refusal does not name a checksum mismatch: {detail}"
    );

    // And nothing is exposed. A reader sees no artifact, not a corrupt one.
    assert!(
        !dest.join(MANIFEST_FILE).exists(),
        "a refused publication left a readable manifest behind"
    );
    assert!(
        moxie_storage::Artifact::open(&dest).is_err(),
        "a destination whose payload failed validation opened as an artifact"
    );
    // The failed path still gives back every admitted byte.
    assert!(
        ledger.outstanding().is_empty(),
        "a refused publication kept a charge: {:?}",
        ledger.outstanding()
    );
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

    // How many times each boundary is visited, measured rather than assumed: a
    // boundary that gained a visit gains enumeration cases.
    //
    // The scenario **includes a resume**, because some boundaries only exist on
    // one: journal compaction runs when an interrupted run is recovered, and
    // enumerating a clean run alone would leave its writes, its sync and its
    // replacement untested. Independent review asked for exactly that after
    // compaction was added.
    let counting = Faults::none();
    let counted_dir = Scratch::new("counted");
    let counted = counted_dir.join("artifact");
    run_to_end(&counted, &counting, false, Some(2)).expect("it cancels");
    run_to_end(&counted, &counting, true, None).expect("it publishes");
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
            // The same shape the visits were counted over: stop part-way, then
            // resume. A boundary that only a resume reaches is failed on the
            // resume.
            let first = run_to_end(&dest, &faults, false, Some(2))
                .and_then(|_| run_to_end(&dest, &faults, true, None));
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
                        &|| false,
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
        "failure-point coverage: {cases} case(s) across {} boundary/visit pair(s)",
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
    //
    // **Every** boundary, with no allowance. This read `unvisited.len() <= 1`
    // until task 0031, from when one boundary was genuinely unreachable by this
    // scenario: a staged unit is only rehashed on a resume, so `chunk-read-back`
    // was never visited. The scenario then grew a resume -- independent review
    // asked for it, so that journal compaction would be covered -- and that
    // boundary started being reached. The coverage table above says so: it
    // reports `chunk-read-back` with **2 visits** and nothing unreached at all.
    // The allowance stayed behind, and nobody re-measured it.
    //
    // What it cost: with zero unreached and one allowed, deleting any single
    // `faults.check(Site::X)` for a once-visited boundary moves the count from
    // 0 to 1 and still passes. That is half of `published-validation-skipped`,
    // the mutation that survived the battery on task 0030's tree -- its other
    // half is caught by the corruption regression above. Slack sized for a gap
    // that has closed is slack sized to hide the next removed check.
    //
    // Tightening this is a strengthening, not a tolerance weakening: the table
    // beside it measures zero unreached boundaries, so nothing legitimate is
    // being excluded, and a boundary that becomes genuinely unreachable again
    // must be argued for in writing rather than absorbed by a spare slot.
    let unvisited: Vec<&str> = Site::ALL
        .iter()
        .filter(|s| !visited.contains(s))
        .map(|s| s.name())
        .collect();
    eprintln!("  boundaries not reached by this scenario: {unvisited:?}");
    assert!(
        unvisited.is_empty(),
        "these boundaries are never reached, so nothing here tests them: {unvisited:?}"
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

    // Cancellation observed at the publication boundary itself -- every unit
    // written, nothing published yet -- is still cancellation. This is the
    // check the per-unit one cannot make: it fires after the last unit, so
    // only the boundary check before the rename can see it.
    let scratch = Scratch::new("cancel-at-the-boundary");
    let dest = scratch.join("artifact");
    let units = requests().iter().map(|r| units(r).len()).sum::<usize>();
    let (outcome, _, written) =
        run_to_end(&dest, &Faults::none(), false, Some(units)).expect("cancellation at the edge");
    assert_eq!(written, units, "every unit was written before cancelling");
    assert!(
        matches!(outcome, Outcome::Cancelled { .. }),
        "cancellation at the publication boundary gave {outcome:?}"
    );
    assert!(
        !dest.join(MANIFEST_FILE).exists(),
        "a run cancelled at the boundary published anyway"
    );
    // And it is still resumable into exactly the reference artifact.
    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it finishes");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    assert_eq!(artifact_bytes(&dest), reference_bytes);
}

/// The order is the contract: a unit's bytes are durable **before** the record
/// that claims them exists. A crash can therefore leave bytes with no record --
/// which a restart discards -- but never a record with no bytes.
#[test]
fn bytes_are_durable_before_the_record_that_claims_them() {
    // Failing the sync of the first chunk write means no unit line may exist.
    let scratch = Scratch::new("order-sync");
    let dest = scratch.join("artifact");
    let faults = Faults::none().fail_at(Site::ChunkSync, 1);
    run_to_end(&dest, &faults, false, None).unwrap_err();
    let journal = std::fs::read_to_string(dest.join(".moxie-repack-journal")).expect("a journal");
    assert!(
        !journal.contains("unit = "),
        "a unit was recorded before its bytes were durable:\n{journal}"
    );

    // Failing the *journal append* for the first unit means its bytes are
    // already on disk: the record is what is missing, not the payload.
    let scratch = Scratch::new("order-journal");
    let dest = scratch.join("artifact");
    // Visit one is the header, whose version and plan lines are appended
    // together; visit two is the first unit's record.
    let faults = Faults::none().fail_at(Site::JournalAppend, 2);
    run_to_end(&dest, &faults, false, None).unwrap_err();
    let staged = std::fs::metadata(dest.join(moxie_repack::write::shard_name(1, 1)))
        .expect("the chunk exists")
        .len();
    assert!(
        staged >= budget().scratch_bytes() as u64,
        "{staged} byte(s) staged: the payload should already be durable"
    );
    let journal = std::fs::read_to_string(dest.join(".moxie-repack-journal")).expect("a journal");
    assert!(!journal.contains("unit = "), "{journal}");
    // And the restart discards those unrecorded bytes and rewrites them.
    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it recovers");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
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
        &|| false,
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
        &|| false,
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
    std::fs::write(dest.join(moxie_repack::write::LOCK_FILE), b"pid 1").expect("a stale lock");
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
        &|| false,
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
        &|| false,
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
            &|| false,
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
    // Corrupt one byte of the first unit's staged **payload**.
    //
    // Past the header, deliberately: a shard's first eight bytes are its
    // header length, which every `begin` rewrites from the plan, so a byte
    // flipped there is repaired before the resume even looks -- and this test
    // would pass while checking nothing. The mutation battery found exactly
    // that when the container changed.
    let chunk = dest.join(moxie_repack::write::shard_name(1, 1));
    let mut bytes = std::fs::read(&chunk).expect("a staged shard");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
    let payload_start = 8 + header_len;
    assert!(
        bytes.len() > payload_start,
        "the staged shard holds no payload yet"
    );
    bytes[payload_start] ^= 0xFF;
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
    let chunk = dest.join(moxie_repack::write::shard_name(1, 1));
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

/// An I/O failure fails the run **naming the boundary it happened at**.
///
/// The enumeration above accepts either outcome at every site, because most of
/// them can legitimately end either way -- and the mutation battery found what
/// that costs: swallowing the error from a payload write survived every gate,
/// because the checksum and the resume machinery repair the damage on the next
/// attempt. Repairing a failure is not the same as reporting it, and a run that
/// silently redoes work it was told had failed is one nobody can debug.
#[test]
fn an_injected_write_failure_fails_the_run_and_names_the_boundary() {
    let scratch = Scratch::new("write-failure");
    let dest = scratch.join("artifact");
    let faults = Faults::none().fail_at(Site::ChunkWrite, 1);
    let e = run_to_end(&dest, &faults, false, None).unwrap_err();
    assert!(
        e.to_string().contains("chunk-write"),
        "the failure does not name the boundary it happened at: {e}"
    );
    assert!(faults.all_fired(), "the injected failure never fired");
    assert!(!dest.join(MANIFEST_FILE).exists());
    // And it is still resumable.
    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it recovers");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
}

/// Cancellation observed **after** validation, in the last moment before the
/// rename, is still cancellation.
///
/// The battery found this one too: with the per-unit check and the pre-staging
/// check both in place, removing the check between validation and the rename
/// changed nothing any test could see. That check is the one that matters most,
/// because it is the last point at which stopping is still free.
#[test]
fn cancellation_observed_after_validation_still_stops_before_the_rename() {
    let scratch = Scratch::new("cancel-after-validation");
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
        &|| false,
    )
    .expect("it starts");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("{other:?}"),
    };
    for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        for (at, len) in units(&request) {
            run.write_unit(
                &component_of(&request),
                &bytes[at as usize..at as usize + len],
                HEX,
                &Faults::none(),
            )
            .expect("every unit writes");
        }
    }
    let sealed = run.seal().expect("it seals");
    let text = manifest::encode(&manifest_for(&sealed)).expect("it encodes");
    // False at the first boundary, true afterwards: cancellation arrives while
    // the artifact is being validated.
    let asked = std::cell::Cell::new(0usize);
    let cancelled = || {
        asked.set(asked.get() + 1);
        asked.get() > 1
    };
    let outcome = run
        .publish(&text, &cancelled, &Faults::none(), &mut ledger)
        .expect("cancellation is not an error");
    assert!(
        matches!(outcome, Outcome::Cancelled { .. }),
        "cancellation after validation gave {outcome:?}"
    );
    assert!(
        !dest.join(MANIFEST_FILE).exists(),
        "it published despite being cancelled"
    );
    assert!(asked.get() >= 2, "the second boundary was never consulted");
    assert!(ledger.outstanding().is_empty());
}

/// Cancellation that arrives at the **last** gate, after everything has been
/// validated and before the rename.
///
/// Two earlier versions of this test were circular and the second review proved
/// it by running the mutation: both learned *when* to cancel from the
/// implementation under test, so deleting the final gate simply moved the
/// boundary they cancelled at and they kept passing.
///
/// The trigger here is **not** a count. The run appends `phase = "validated"`
/// to its journal, durably, as the last thing before the final gate, so the
/// journal on disk says "validation is over" from outside the process. This
/// closure answers no until it sees that line. With the gate deleted there is
/// nothing left to stop, and the run publishes -- which is the failure.
#[test]
fn cancellation_at_the_final_gate_still_stops_before_the_rename() {
    let scratch = Scratch::new("cancel-final-gate");
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
        &|| false,
    )
    .expect("it starts");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("{other:?}"),
    };
    for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        for (at, len) in units(&request) {
            run.write_unit(
                &component_of(&request),
                &bytes[at as usize..at as usize + len],
                HEX,
                &Faults::none(),
            )
            .expect("every unit writes");
        }
    }
    let sealed = run.seal().expect("it seals");
    let text = manifest::encode(&manifest_for(&sealed)).expect("it encodes");

    // The journal, read from disk at every boundary: an observer outside this
    // run, watching for the record that says validation finished.
    let journal = dest.join(".moxie-repack-journal");
    let asked = std::cell::Cell::new(0usize);
    let validated = std::cell::Cell::new(false);
    let cancelled = || {
        asked.set(asked.get() + 1);
        let seen = std::fs::read_to_string(&journal)
            .map(|t| t.contains("phase = \"validated\""))
            .unwrap_or(false);
        if seen {
            validated.set(true);
        }
        seen
    };
    let outcome = run
        .publish(&text, &cancelled, &Faults::none(), &mut ledger)
        .expect("cancellation is not an error");
    assert!(
        validated.get(),
        "the run never recorded that it had validated, so this test never reached the gate it \
         is about: it asked {} time(s)",
        asked.get()
    );
    assert!(
        matches!(outcome, Outcome::Cancelled { .. }),
        "cancellation at the final gate gave {outcome:?}"
    );
    assert!(
        !dest.join(MANIFEST_FILE).exists(),
        "it published despite being cancelled at the last gate"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A private file replaced by a symbolic link is refused, and the link's target
/// is untouched.
///
/// The review's reproduction: cancel a run, point `.moxie-repack-manifest` at
/// an unrelated file, resume. `File::create` followed the link and truncated
/// the target, and validation rejected the escape afterwards -- after the
/// damage. Confinement happens before the mutation now, and this is what says
/// so.
#[test]
fn a_private_file_replaced_by_a_symlink_is_refused_before_anything_is_written() {
    let shard = moxie_repack::write::shard_name(1, 1);
    for victim_name in [
        ".moxie-repack-manifest",
        shard.as_str(),
        ".moxie-repack-journal",
    ] {
        let scratch = Scratch::new("symlink");
        let dest = scratch.join("artifact");
        let outsider = scratch.join("someone-elses-data");
        std::fs::write(&outsider, b"a file this run has no business touching")
            .expect("the outsider exists");
        run_to_end(&dest, &Faults::none(), false, Some(1)).expect("a cancelled run");

        let victim = dest.join(victim_name);
        let _ = std::fs::remove_file(&victim);
        std::os::unix::fs::symlink(&outsider, &victim).expect("the symlink is planted");

        let outcome = run_to_end(&dest, &Faults::none(), true, None);
        // Whatever it does, it does not write through the link.
        assert_eq!(
            std::fs::read(&outsider).expect("the outsider survives"),
            b"a file this run has no business touching",
            "{victim_name}: the run wrote through a symbolic link"
        );
        match outcome {
            Err(e) => assert!(
                e.to_string().contains("symbolic link") || e.to_string().contains("journal"),
                "{victim_name}: refused for an unrelated reason: {e}"
            ),
            Ok((outcome, _, _)) => panic!("{victim_name}: {outcome:?} through a symlink"),
        }
    }
}

/// A torn journal survives being torn, resumed, interrupted and resumed again.
///
/// The review's reproduction: parsing ignored the torn tail and recovery never
/// truncated it, so the next record was appended onto the fragment and the
/// journal became unparseable for good. The existing test ran straight to
/// publication, which deletes the journal and hid it.
#[test]
fn a_torn_journal_survives_repeated_interruption() {
    let reference_dir = Scratch::new("torn-twice-reference");
    let reference = reference_dir.join("artifact");
    run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    let scratch = Scratch::new("torn-twice");
    let dest = scratch.join("artifact");
    let journal = dest.join(".moxie-repack-journal");
    run_to_end(&dest, &Faults::none(), false, Some(1)).expect("a cancelled run");

    // Tear the tail, resume far enough to append a record onto the repair, and
    // stop again.
    let torn = {
        let mut text = std::fs::read_to_string(&journal).expect("a journal");
        text.push_str("unit = { tensor = \"layers.0.w\", index = 9, chunk");
        text
    };
    std::fs::write(&journal, &torn).expect("tearing it");
    run_to_end(&dest, &Faults::none(), true, Some(1)).expect("a second cancelled run");

    // The repair has to have happened: the journal parses, and what follows the
    // repaired tail is a whole record rather than an append onto a fragment.
    let after = std::fs::read_to_string(&journal).expect("a journal");
    assert!(
        !after.contains("index = 9, chunk\nunit"),
        "a record was appended onto the torn fragment:\n{after}"
    );

    // Tear it once more, then finish. This is the sequence that used to leave
    // the destination unusable.
    let mut text = std::fs::read_to_string(&journal).expect("a journal");
    text.push_str("unit = { tensor = ");
    std::fs::write(&journal, &text).expect("tearing it again");
    let (outcome, _, _) = run_to_end(&dest, &Faults::none(), true, None).expect("it publishes");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
    assert_eq!(artifact_bytes(&dest), reference_bytes);
}

/// A record torn **inside a multi-byte character** is still a torn record.
///
/// The ASCII repair passed while the reader still decoded the whole file as
/// UTF-8 before recovery could discard the tail: independent review interrupted
/// a record inside a Unicode string and the resume failed immediately, before
/// the code that exists to handle exactly this ever ran. A newline is 0x0A and
/// never appears inside a multi-byte sequence, so the committed prefix is found
/// on the bytes and only that prefix is decoded.
#[test]
fn a_journal_torn_inside_a_multibyte_character_still_resumes() {
    let reference_dir = Scratch::new("torn-utf8-reference");
    let reference = reference_dir.join("artifact");
    run_to_end(&reference, &Faults::none(), false, None).expect("the reference publishes");
    let reference_bytes = artifact_bytes(&reference);

    let scratch = Scratch::new("torn-utf8");
    let dest = scratch.join("artifact");
    let journal = dest.join(".moxie-repack-journal");
    run_to_end(&dest, &Faults::none(), false, Some(1)).expect("a cancelled run");

    // Append a record and cut it in the middle of a three-byte character, which
    // is what an interrupted write of a record carrying one looks like.
    let mut bytes = std::fs::read(&journal).expect("a journal");
    let fragment = "unit = { tensor = \"layers.0.\u{4e16}";
    let fragment = fragment.as_bytes();
    bytes.extend_from_slice(&fragment[..fragment.len() - 1]);
    assert!(
        String::from_utf8(bytes.clone()).is_err(),
        "this journal is still valid UTF-8, so it does not reproduce the finding"
    );
    std::fs::write(&journal, &bytes).expect("tearing it");

    let (outcome, _, _) =
        run_to_end(&dest, &Faults::none(), true, None).expect("a torn character is recoverable");
    assert!(matches!(outcome, Outcome::Published { .. }), "{outcome:?}");
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
        &|| false,
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

/// The journal's own cap refuses a record even when the plan allowed it.
///
/// The plan bounds the journal, and this is the **second** net: a per-append
/// check that does not depend on the planner having been right. Independent
/// review asked for it "independently of the combined staging allowance", and a
/// run driven through a correct planner can never show it, because a correct
/// plan never lets the append path reach the cap. So the plan here is built
/// with a deliberately generous allowance -- 64 MiB against the journal's
/// 16 MiB -- over a tensor whose units are small enough to produce more records
/// than the cap admits.
#[test]
fn the_journal_cap_refuses_a_record_the_plan_allowed() {
    use moxie_repack::write::{TensorRequest, WriteBudget};

    let scratch = Scratch::new("journal-cap-runtime");
    let dest = scratch.join("artifact");

    // One BF16 tensor, written in 256-byte units: each record is a few hundred
    // bytes, so the cap arrives well before the payload runs out.
    // Enough payload that the cap is reached before the tensor ends: 39,062
    // records was just short of it.
    let elements = 12_000_000u64;
    let role = "cap.probe.weight";
    let components =
        moxie_format::canonical::bf16_components(role, &[elements]).expect("components");
    let request = TensorRequest {
        role: role.to_string(),
        shape: vec![elements],
        precision: moxie_format::manifest::TensorPrecision::Bf16V1,
        affine: None,
        components,
    };
    let write = WriteBudget::new(512, 256 << 20, 256 << 20).expect("a write budget");
    let generous = OutputPlan::build(vec![request], &write, 64 << 20).expect("a plan");

    let mut ledger = ledger();
    let start = Run::begin(
        &dest,
        generous,
        binding(),
        write,
        &Options::default(),
        &mut ledger,
        &Faults::none(),
        &|| false,
    )
    .expect("it starts");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("{other:?}"),
    };

    let component = role.to_string();
    let unit = vec![0u8; 256];
    let mut records = 0usize;
    let refusal = loop {
        match run.write_unit(&component, &unit, HEX, &Faults::none()) {
            Ok(()) => records += 1,
            Err(e) => break e.to_string(),
        }
        assert!(
            records < 200_000,
            "the journal grew past {records} records with no limit"
        );
    };
    assert!(
        refusal.contains("read back on a resume"),
        "the refusal is not the journal's own cap, after {records} record(s): {refusal}"
    );
    eprintln!("journal cap: refused after {records} record(s)");
    run.abandon(&mut ledger).expect("it abandons");
}

/// A logical publication unit must not write and sync its payload twice.
/// The tail mutant adds a duplicate durable write after the journal append;
/// it does not remove the original pre-journal write. The defect measured here
/// is duplicate payload I/O and an unnecessary failure boundary, not proof of
/// a corrupt journal or lost durability.
#[test]
fn write_unit_touches_chunk_write_and_sync_exactly_once() {
    let scratch = Scratch::new("write-once");
    let dest = scratch.join("artifact");
    let mut ledger = ledger();
    let faults = Faults::none();
    let start = Run::begin(
        &dest,
        plan(),
        binding(),
        budget(),
        &Options::default(),
        &mut ledger,
        &faults,
        &|| false,
    )
    .expect("a fresh start");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("expected a fresh start: {other:?}"),
    };
    let request = &requests()[0];
    let (at, len) = units(request)[0];
    let bytes = payload(request_len(request), request_len(request) as usize);
    run.write_unit(
        &component_of(request),
        &bytes[at as usize..at as usize + len],
        HEX,
        &faults,
    )
    .expect("the unit writes");
    assert_eq!(
        faults.visits(Site::ChunkWrite),
        1,
        "one payload write per unit"
    );
    assert_eq!(
        faults.visits(Site::ChunkSync),
        1,
        "one payload sync per unit; duplicate syncs add redundant I/O and failure boundaries"
    );
    run.abandon(&mut ledger).expect("it abandons");
}

/// A resume whose compaction would hold both journals above the plan's
/// overhead allowance is refused, not silently allowed to exceed its own disk
/// budget.
///
/// Compaction writes a replacement journal beside the original and renames
/// over it, so for the length of that write the destination holds both at
/// once -- `peak = overhead_used + journal_used` in `recover`. Nothing the
/// incremental charging in `write_unit` does during ordinary writing ever
/// reaches that peak; only a resume that has to compact does, which is why
/// this drives a real cancel-then-resume rather than asserting on the charge
/// functions directly.
#[test]
fn a_resume_whose_compaction_peak_exceeds_the_overhead_budget_is_refused() {
    // First pass: measure how large the journal actually is after two units,
    // under a budget roomy enough to get there uninterrupted.
    let probe_dir = Scratch::new("compaction-peak-probe");
    let probe_dest = probe_dir.join("artifact");
    let mut probe_ledger = ledger();
    let probe_plan = OutputPlan::build(requests(), &budget(), 8 * 1024).expect("a plan");
    let start = Run::begin(
        &probe_dest,
        probe_plan,
        binding(),
        budget(),
        &Options::default(),
        &mut probe_ledger,
        &Faults::none(),
        &|| false,
    )
    .expect("a fresh start");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("expected a fresh start: {other:?}"),
    };
    let mut written = 0usize;
    'probe: for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        for (at, len) in units(&request) {
            run.write_unit(
                &component_of(&request),
                &bytes[at as usize..at as usize + len],
                HEX,
                &Faults::none(),
            )
            .expect("the unit writes");
            written += 1;
            if written == 2 {
                break 'probe;
            }
        }
    }
    let outcome = run.cancel(&mut probe_ledger).expect("it cancels");
    assert!(matches!(outcome, Outcome::Cancelled { .. }), "{outcome:?}");
    let journal_len = std::fs::metadata(probe_dest.join(".moxie-repack-journal"))
        .expect("a journal")
        .len();
    assert!(journal_len > 0, "the probe wrote no journal");

    // Second pass, in a fresh destination, with the plan's overhead sized to
    // hold one journal comfortably and the peak of two nowhere near.
    let tight_overhead = journal_len + 16;
    assert!(
        tight_overhead < 2 * journal_len,
        "the calibration does not leave room to distinguish one journal from the compaction peak"
    );
    let scratch = Scratch::new("compaction-peak");
    let dest = scratch.join("artifact");
    let mut tight_ledger = ledger();
    let tight_plan = OutputPlan::build(requests(), &budget(), tight_overhead).expect("a plan");
    let start = Run::begin(
        &dest,
        tight_plan,
        binding(),
        budget(),
        &Options::default(),
        &mut tight_ledger,
        &Faults::none(),
        &|| false,
    )
    .expect("a fresh start under the tight budget");
    let mut run = match start {
        Start::Fresh(run) => run,
        other => panic!("expected a fresh start: {other:?}"),
    };
    let mut written = 0usize;
    'tight: for request in requests() {
        let bytes = payload(request_len(&request), request_len(&request) as usize);
        for (at, len) in units(&request) {
            run.write_unit(
                &component_of(&request),
                &bytes[at as usize..at as usize + len],
                HEX,
                &Faults::none(),
            )
            .expect("the unit writes under the tight budget");
            written += 1;
            if written == 2 {
                break 'tight;
            }
        }
    }
    let outcome = run.cancel(&mut tight_ledger).expect("it cancels");
    assert!(matches!(outcome, Outcome::Cancelled { .. }), "{outcome:?}");
    let confirm_len = std::fs::metadata(dest.join(".moxie-repack-journal"))
        .expect("a journal")
        .len();
    assert_eq!(
        confirm_len, journal_len,
        "the tight-budget run produced a differently sized journal than the probe: the \
         calibration above does not apply here"
    );

    // Resume: recovery reads this journal back and compacts it, and the peak
    // of holding both copies at once is double what the tight overhead
    // reserved.
    let mut resume_ledger = ledger();
    let resume_plan = OutputPlan::build(requests(), &budget(), tight_overhead).expect("a plan");
    let e = Run::begin(
        &dest,
        resume_plan,
        binding(),
        budget(),
        &Options {
            take_over_interrupted_run: true,
        },
        &mut resume_ledger,
        &Faults::none(),
        &|| false,
    )
    .expect_err("a compaction peak above the overhead budget is a refusal, not a silent overrun");
    assert!(
        e.to_string().contains("compacting this journal would hold"),
        "the refusal does not name the compaction peak: {e}"
    );
    assert!(resume_ledger.outstanding().is_empty());
}
