//! The publication state machine: stage, checksum, journal, validate, publish.
//!
//! Document 03's offline contract in one sentence -- "convert bounded chunks ->
//! validate -> atomically publish manifest" -- with every boundary in between
//! made explicit, because the interesting part of a repacker is what happens
//! when it stops halfway.
//!
//! ## What is on disk, and when
//!
//! The destination directory **is** the staging area. Chunk files are written
//! under their final names, and what makes the artifact unreadable until it is
//! finished is that `manifest.toml` does not exist yet: the production reader
//! opens the manifest first, so a directory of complete chunks with no manifest
//! is not an artifact at all. Publication is one rename of the staged manifest
//! into place.
//!
//! The alternative -- staging in a subdirectory and moving payloads at publish
//! time -- would copy or rename every payload file at the moment of highest
//! consequence and would need twice the disk in the failure case. This way the
//! bytes never move after they are durable.
//!
//! Two private files live beside the payloads while a run is in flight, both
//! dot-prefixed and both removed when it finishes:
//!
//! * `.moxie-repack-journal` -- the restart record ([ADR 0023]).
//! * `.moxie-repack-lock` -- exclusive ownership, taken with `O_EXCL`.
//! * `.moxie-repack-manifest` -- the staged manifest, validated through the
//!   production reader before it is renamed to `manifest.toml`.
//!
//! ## What a crash leaves
//!
//! | Crash point | What a restart finds | What it does |
//! |---|---|---|
//! | Mid-unit | Journal without that unit; chunk file possibly longer than the journal says | Truncates to the last journaled byte, rehashes it, continues |
//! | After a unit's sync, before its journal line | The same | The same: the unit is recomputed |
//! | After the journal line | A complete unit | Rehashes and keeps it |
//! | After `phase = validated` | Everything durable, no manifest | Revalidates and publishes |
//! | After the rename | `manifest.toml` present | Reports **already published** and touches nothing |
//!
//! The asymmetry is deliberate: a unit is recomputed when in doubt, and a
//! published artifact is never deleted when in doubt.
//!
//! [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_format::journal::{self, CompletedUnit, JournalState, Phase, RunBinding};
use moxie_format::sha256::{StreamingSha256, sha256_hex};
use moxie_memory::{HostBuffer, Ledger};
use moxie_storage::{Artifact, ByteBudget};
use moxie_types::{HostTier, Result};

use crate::fault::{Faults, Site};
use crate::plan::{OutputPlan, PlannedTensor};
use crate::{WriteBudget, invalid};

/// The journal file's name inside the destination.
pub const JOURNAL_FILE: &str = ".moxie-repack-journal";
/// The exclusive-ownership file's name.
pub const LOCK_FILE: &str = ".moxie-repack-lock";
/// The staged manifest's name, before it becomes `manifest.toml`.
pub const STAGED_MANIFEST_FILE: &str = ".moxie-repack-manifest";
/// What a published artifact's manifest is called.
pub const MANIFEST_FILE: &str = "manifest.toml";

/// How a run ended.
///
/// Failures are errors, not outcomes: they leave a resumable destination and
/// say why. These are the ends that are not failures -- including the one that
/// is neither success nor failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The manifest is in place and its durability was confirmed.
    Published { artifact: PathBuf, bytes: u64 },
    /// The publication rename succeeded and the confirmation after it did not.
    ///
    /// **Not a failure, and not a success.** The artifact may well be readable;
    /// what is unknown is whether it survives a power cut. A restart
    /// reconciles. Deleting it here would be deleting a possibly published
    /// output as though it were an uncommitted attempt.
    PublishedDurabilityUnconfirmed { artifact: PathBuf, detail: String },
    /// Cancellation was observed before the publication boundary. The
    /// destination holds a documented, resumable private state.
    Cancelled {
        destination: PathBuf,
        bytes_done: u64,
    },
}

/// What `begin` found.
#[derive(Debug)]
pub enum Start {
    /// A destination that did not exist.
    Fresh(Run),
    /// A destination with a journal this plan binds to.
    Resumed(Run, ResumeReport),
    /// A destination that already holds a published manifest. Nothing was
    /// changed and nothing may be.
    AlreadyPublished { artifact: PathBuf },
}

/// What a resume recovered, measured rather than assumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeReport {
    /// Units the journal recorded and whose staged bytes rehashed correctly.
    pub reused_units: usize,
    pub reused_bytes: u64,
    /// Units the journal recorded that had to be discarded, with the reason.
    pub discarded: Vec<String>,
    /// Bytes past the last journaled unit found in chunk files: a crash
    /// between a payload write and its journal line.
    pub truncated_bytes: u64,
    /// A torn final journal line, in bytes.
    pub torn_journal_bytes: usize,
    /// The phase the journal was left in.
    pub phase: Phase,
}

/// One tensor's progress through its byte range.
#[derive(Debug)]
struct Progress {
    done: u64,
    next_index: u64,
    hasher: StreamingSha256,
}

/// A tensor whose payload is complete and hashed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedTensor {
    pub role: String,
    pub chunk: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

/// One run against one destination.
#[derive(Debug)]
pub struct Run {
    dest: PathBuf,
    plan: OutputPlan,
    budget: WriteBudget,
    binding: RunBinding,
    journal: File,
    progress: BTreeMap<String, Progress>,
    /// Bytes written to chunk files by this process, for the disk budget.
    disk_used: u64,
    /// Read-back scratch, admitted from the ledger like every other byte this
    /// run holds.
    scratch: HostBuffer,
    /// Set once `seal` has run, so `publish` cannot be reached with an
    /// incomplete selection.
    sealed: Option<Vec<SealedTensor>>,
}

/// Options a caller states rather than a run inferring them.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Continue a destination whose lock file is still present.
    ///
    /// A lock file outlives a crashed run, and this program cannot tell a
    /// crashed run from a live one: doing that needs process liveness, which
    /// means reading this machine's own telemetry, and ADR 0006 gives that to
    /// exactly one crate. So the choice is the user's, it is explicit, and the
    /// limitation is documented rather than guessed around.
    pub take_over_interrupted_run: bool,
}

impl Run {
    /// Open a destination: fresh, resumed, or already published.
    pub fn begin(
        dest: &Path,
        plan: OutputPlan,
        binding: RunBinding,
        budget: WriteBudget,
        options: &Options,
        ledger: &mut Ledger,
        faults: &Faults,
    ) -> Result<Start> {
        let dest = dest.to_path_buf();
        if dest.join(MANIFEST_FILE).exists() {
            // Never an overwrite, and never an in-place model update: this
            // slice publishes new artifacts only.
            return Ok(Start::AlreadyPublished { artifact: dest });
        }
        let journal_path = dest.join(JOURNAL_FILE);
        // A journal that exists but carries no plan line is a run interrupted
        // between creating the file and recording what it was for: it accounts
        // for nothing, because a unit cannot be recorded before the header.
        // Removing it and starting over is the only reading that does not
        // strand the destination.
        let resuming = journal_path.exists() && journal_binds(&journal_path)?;
        if journal_path.exists() && !resuming {
            std::fs::remove_file(&journal_path).map_err(|e| {
                invalid(format!(
                    "cannot remove the empty journal {}: {e}",
                    journal_path.display()
                ))
            })?;
        }
        if dest.exists() {
            if !resuming {
                // A directory holding only this program's own private files is
                // one of its own runs that stopped before it recorded
                // anything -- a crash between taking the lock and writing the
                // journal header leaves exactly that. Anything else is
                // somebody's data, and this never writes into it.
                let mut foreign = Vec::new();
                for entry in std::fs::read_dir(&dest)
                    .map_err(|e| invalid(format!("cannot read {}: {e}", dest.display())))?
                {
                    let entry = entry.map_err(|e| invalid(format!("cannot read an entry: {e}")))?;
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if !matches!(
                        name.as_str(),
                        LOCK_FILE | JOURNAL_FILE | STAGED_MANIFEST_FILE
                    ) {
                        foreign.push(name);
                    }
                }
                if !foreign.is_empty() {
                    foreign.sort();
                    return Err(invalid(format!(
                        "{} already exists and holds {foreign:?} with no repack journal to account \
                         for them: refusing to write into a directory this run did not create",
                        dest.display()
                    )));
                }
            }
        } else {
            faults.check(Site::DestinationCreate)?;
            std::fs::create_dir_all(&dest)
                .map_err(|e| invalid(format!("cannot create {}: {e}", dest.display())))?;
        }

        take_lock(&dest, options, faults)?;

        faults.check(Site::Allocate)?;
        let mut scratch = HostBuffer::allocate_in(
            ledger,
            "repack read-back scratch",
            HostTier::Pageable,
            budget.scratch_bytes(),
            0,
        )?;

        // The journal handle before the run, so there is no moment at which a
        // `Run` exists without the file that records what it did.
        let journal = match open_journal(&journal_path, resuming, &binding, faults) {
            Ok(file) => file,
            Err(e) => {
                // An admitted buffer outlives a failure only if nobody
                // releases it; releasing here is what keeps a refused start
                // from leaving a charge behind.
                scratch.release(ledger)?;
                let _ = std::fs::remove_file(dest.join(LOCK_FILE));
                return Err(e);
            }
        };

        let mut run = Run {
            dest: dest.clone(),
            plan,
            budget,
            binding,
            journal,
            progress: BTreeMap::new(),
            disk_used: 0,
            scratch,
            sealed: None,
        };
        for t in run.plan.tensors() {
            run.progress.insert(
                t.request.role.clone(),
                Progress {
                    done: 0,
                    next_index: 0,
                    hasher: StreamingSha256::new(),
                },
            );
        }

        if resuming {
            match run.recover(&journal_path, faults) {
                Ok(report) => Ok(Start::Resumed(run, report)),
                Err(e) => {
                    // Same rule as above: a start that refuses releases what
                    // it admitted.
                    run.abandon(ledger)?;
                    Err(e)
                }
            }
        } else {
            Ok(Start::Fresh(run))
        }
    }

    /// Rebuild state from the journal, rehashing every byte it claims.
    ///
    /// A journal entry is never evidence its payload is correct. Every unit is
    /// re-read from the staged chunk file and rehashed; a mismatch discards the
    /// unit and everything after it in that tensor, because a tensor's bytes
    /// are a sequence and a hole in it cannot be filled out of order.
    fn recover(&mut self, journal_path: &Path, faults: &Faults) -> Result<ResumeReport> {
        let text = read_capped(journal_path, journal::MAX_JOURNAL_BYTES)?;
        let state: JournalState = journal::parse(&text)?;
        let recorded = state.binding.clone().ok_or_else(|| {
            invalid("this journal records no plan: `begin` should have started over".into())
        })?;
        if recorded != self.binding {
            return Err(invalid(format!(
                "this destination holds a run bound to a different plan: recorded {:?}, requested \
                 {:?}. A resume binds source content, converter version, selection and output \
                 plan; a difference is a refusal, not a merge",
                recorded, self.binding
            )));
        }
        let mut report = ResumeReport {
            reused_units: 0,
            reused_bytes: 0,
            discarded: Vec::new(),
            truncated_bytes: 0,
            torn_journal_bytes: state.torn_tail_bytes,
            phase: state.phase,
        };
        let mut stopped: BTreeMap<String, bool> = BTreeMap::new();
        let mut staged_bytes: u64 = 0;
        for unit in &state.units {
            let Some(planned) = self.plan.tensor(&unit.tensor).cloned() else {
                return Err(invalid(format!(
                    "the journal records a unit for tensor '{}', which this plan does not \
                     describe",
                    unit.tensor
                )));
            };
            if *stopped.get(&unit.tensor).unwrap_or(&false) {
                report.discarded.push(format!(
                    "{} unit {}: follows a discarded unit",
                    unit.tensor, unit.index
                ));
                continue;
            }
            match self.reuse_unit(&planned, unit, faults) {
                Ok(()) => {
                    report.reused_units += 1;
                    report.reused_bytes += unit.len;
                }
                Err(e) => {
                    stopped.insert(unit.tensor.clone(), true);
                    report
                        .discarded
                        .push(format!("{} unit {}: {e}", unit.tensor, unit.index));
                }
            }
        }
        // Bytes past the last journaled unit are a crash between a payload
        // write and its journal line: the chunk file is truncated back to what
        // the journal accounts for, so the next unit writes where it expects.
        for (name, _) in self.plan.chunks() {
            let path = self.dest.join(name);
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            let accounted = self
                .plan
                .tensors()
                .iter()
                .filter(|t| &t.chunk == name)
                .map(|t| t.offset + self.progress[&t.request.role].done)
                .max()
                .unwrap_or(0);
            if meta.len() > accounted {
                report.truncated_bytes += meta.len() - accounted;
                let file = OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .map_err(|e| invalid(format!("cannot truncate {}: {e}", path.display())))?;
                file.set_len(accounted)
                    .map_err(|e| invalid(format!("cannot truncate {}: {e}", path.display())))?;
                file.sync_all()
                    .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;
            }
            // Across chunks this is a sum, not a maximum: the disk budget is
            // about the run's total footprint, and a resumed run already owns
            // every byte the journal accounts for.
            staged_bytes += accounted;
        }
        self.disk_used = staged_bytes;
        Ok(report)
    }

    /// Re-read one journaled unit's staged bytes and fold them into the
    /// tensor's running hash, or refuse it.
    fn reuse_unit(
        &mut self,
        planned: &PlannedTensor,
        unit: &CompletedUnit,
        faults: &Faults,
    ) -> Result<()> {
        let progress = self
            .progress
            .get(&unit.tensor)
            .ok_or_else(|| invalid(format!("no progress for tensor '{}'", unit.tensor)))?;
        if unit.index != progress.next_index {
            return Err(invalid(format!(
                "unit {} arrives after {} unit(s): a tensor's bytes are a sequence",
                unit.index, progress.next_index
            )));
        }
        if unit.chunk != planned.chunk || unit.offset != planned.offset + progress.done {
            return Err(invalid(format!(
                "unit {} claims {}@{} but the plan puts those bytes at {}@{}",
                unit.index,
                unit.chunk,
                unit.offset,
                planned.chunk,
                planned.offset + progress.done
            )));
        }
        if progress.done + unit.len > planned.request.length {
            return Err(invalid(format!(
                "unit {} would take tensor '{}' past its {} byte(s)",
                unit.index, unit.tensor, planned.request.length
            )));
        }
        // The bytes themselves, rehashed from the file in one pass that feeds
        // both hashes: the unit's own, to check the journal's claim, and the
        // tensor's running hash, which has to see every byte in order.
        //
        // The running hash is **cloned** first and only committed if the unit
        // is accepted. Folding first and checking afterwards is how a
        // discarded unit still ends up in the tensor's checksum -- and a
        // checksum computed over bytes that were then rewritten matches
        // nothing on disk. The corruption regression found exactly that.
        let path = self.dest.join(&unit.chunk);
        faults.check(Site::ChunkReadBack)?;
        let mut running = self
            .progress
            .get(&unit.tensor)
            .expect("checked above")
            .hasher
            .clone();
        let mut unit_hasher = StreamingSha256::new();
        read_range(
            &path,
            unit.offset,
            unit.len,
            self.scratch.bytes_mut(),
            &mut |slice| {
                running.update(slice);
                unit_hasher.update(slice);
                Ok(())
            },
        )?;
        let got = unit_hasher.finalize_hex();
        if got != unit.sha256 {
            return Err(invalid(format!(
                "the staged bytes hash to {got}, the journal recorded {}: a journal entry is not \
                 evidence its payload is correct",
                unit.sha256
            )));
        }
        let progress = self.progress.get_mut(&unit.tensor).expect("checked above");
        progress.hasher = running;
        progress.done += unit.len;
        progress.next_index += 1;
        Ok(())
    }

    /// The source hashes of the units a resume kept, so the caller can compare
    /// them against the source it is about to read.
    pub fn destination(&self) -> &Path {
        &self.dest
    }

    pub fn plan(&self) -> &OutputPlan {
        &self.plan
    }

    /// How many of a tensor's payload bytes are already durable.
    pub fn bytes_done(&self, role: &str) -> Result<u64> {
        self.progress
            .get(role)
            .map(|p| p.done)
            .ok_or_else(|| invalid(format!("tensor '{role}' is not in this plan")))
    }

    /// Whether every tensor's payload is complete.
    pub fn is_complete(&self) -> bool {
        self.plan
            .tensors()
            .iter()
            .all(|t| self.progress[&t.request.role].done == t.request.length)
    }

    /// Write one bounded work unit: append, sync, then journal.
    ///
    /// The order is the contract. The bytes are durable before the record that
    /// claims them exists, so a crash can leave bytes with no record -- which a
    /// restart discards -- but never a record with no bytes.
    pub fn write_unit(
        &mut self,
        role: &str,
        bytes: &[u8],
        source_sha256: &str,
        faults: &Faults,
    ) -> Result<()> {
        let planned = self
            .plan
            .tensor(role)
            .cloned()
            .ok_or_else(|| invalid(format!("tensor '{role}' is not in this plan")))?;
        let progress = self
            .progress
            .get(role)
            .ok_or_else(|| invalid(format!("tensor '{role}' has no progress")))?;
        if bytes.is_empty() {
            return Err(invalid(format!(
                "tensor '{role}': an empty unit is not work"
            )));
        }
        if bytes.len() > self.budget.scratch_bytes() {
            return Err(invalid(format!(
                "tensor '{role}': a {}-byte unit is above the admitted {}-byte payload scratch",
                bytes.len(),
                self.budget.scratch_bytes()
            )));
        }
        let done = progress.done;
        let end = done + bytes.len() as u64;
        if end > planned.request.length {
            return Err(invalid(format!(
                "tensor '{role}': this unit ends at {end} of a {} byte payload",
                planned.request.length
            )));
        }
        let offset = planned.offset + done;
        let index = progress.next_index;

        let new_disk = self.disk_used + bytes.len() as u64;
        if new_disk > self.budget.disk_bytes() {
            return Err(invalid(format!(
                "writing {} more byte(s) would take this run to {new_disk}, above its admitted \
                 disk budget of {}",
                bytes.len(),
                self.budget.disk_bytes()
            )));
        }

        let path = self.dest.join(&planned.chunk);
        faults.check(Site::ChunkCreate)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| invalid(format!("cannot open chunk {}: {e}", path.display())))?;
        write_at(&mut file, offset, bytes, faults)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;

        let unit = CompletedUnit {
            tensor: role.to_string(),
            index,
            chunk: planned.chunk.clone(),
            offset,
            len: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            source_sha256: source_sha256.to_string(),
        };
        append_durably(&mut self.journal, &journal::unit_line(&unit), faults)?;

        let progress = self.progress.get_mut(role).expect("checked above");
        progress.hasher.update(bytes);
        progress.done = end;
        progress.next_index += 1;
        self.disk_used = new_disk;
        Ok(())
    }

    /// Finish every tensor's checksum. Refuses an incomplete selection.
    pub fn seal(&mut self) -> Result<Vec<SealedTensor>> {
        if let Some(sealed) = &self.sealed {
            return Ok(sealed.clone());
        }
        let mut out = Vec::with_capacity(self.plan.tensors().len());
        for t in self.plan.tensors() {
            let p = &self.progress[&t.request.role];
            if p.done != t.request.length {
                return Err(invalid(format!(
                    "tensor '{}' has {} of {} byte(s): an incomplete tensor cannot be published",
                    t.request.role, p.done, t.request.length
                )));
            }
            let mut hasher = StreamingSha256::new();
            core::mem::swap(
                &mut hasher,
                &mut self
                    .progress
                    .get_mut(&t.request.role)
                    .expect("present")
                    .hasher,
            );
            let sha256 = hasher.finalize_hex();
            // The hasher is consumed, so put an equivalent one back: sealing
            // twice must not return two different digests.
            out.push(SealedTensor {
                role: t.request.role.clone(),
                chunk: t.chunk.clone(),
                offset: t.offset,
                length: t.request.length,
                sha256,
            });
        }
        self.sealed = Some(out.clone());
        Ok(out)
    }

    /// Validate the staged artifact through the production reader and publish
    /// it with one rename.
    ///
    /// `manifest_text` must be the encoding of a manifest describing exactly
    /// the sealed tensors; the reader is what checks that, and it is the
    /// **production** reader rather than a second validator written for the
    /// writer's benefit.
    pub fn publish(
        mut self,
        manifest_text: &str,
        cancelled: &dyn Fn() -> bool,
        faults: &Faults,
        ledger: &mut Ledger,
    ) -> Result<Outcome> {
        let result = self.publish_inner(manifest_text, cancelled, faults, ledger);
        if result.is_err() {
            // A failed publication still gives back what it admitted. Dropping
            // the buffer would free the bytes and leave the charge standing,
            // which is the leak `Ledger::outstanding` exists to make visible --
            // and this is the one path that reaches the end of a run without
            // going through publish or cancel.
            let _ = self.scratch.release(ledger);
            let _ = std::fs::remove_file(self.dest.join(LOCK_FILE));
        }
        result
    }

    fn publish_inner(
        &mut self,
        manifest_text: &str,
        cancelled: &dyn Fn() -> bool,
        faults: &Faults,
        ledger: &mut Ledger,
    ) -> Result<Outcome> {
        let sealed = self.seal()?;
        if cancelled() {
            let bytes = self.progress.values().map(|p| p.done).sum();
            return self.cancel_with(bytes, ledger);
        }
        let staged = self.dest.join(STAGED_MANIFEST_FILE);
        faults.check(Site::ManifestWrite)?;
        {
            let mut file = File::create(&staged)
                .map_err(|e| invalid(format!("cannot create {}: {e}", staged.display())))?;
            file.write_all(manifest_text.as_bytes())
                .map_err(|e| invalid(format!("cannot write {}: {e}", staged.display())))?;
            faults.check(Site::ManifestSync)?;
            file.sync_all()
                .map_err(|e| invalid(format!("cannot sync {}: {e}", staged.display())))?;
        }

        // Through the production reader, before anything is exposed: it parses
        // and validates the manifest, stats every chunk, and verifies every
        // tensor's checksum against the bytes on disk.
        faults.check(Site::Validate)?;
        let artifact = Artifact::open_unpublished(&self.dest, &staged, ByteBudget::default())?;
        for t in &sealed {
            let read = artifact.verify_tensor(&t.role, self.scratch.bytes_mut())?;
            if read != t.length {
                return Err(invalid(format!(
                    "tensor '{}' verified {read} byte(s) against a planned {}",
                    t.role, t.length
                )));
            }
        }
        drop(artifact);
        append_durably(
            &mut self.journal,
            &journal::phase_line(Phase::Validated),
            faults,
        )?;
        if cancelled() {
            // The last point at which cancellation can be honoured: after this
            // the artifact exists.
            let bytes = self.progress.values().map(|p| p.done).sum();
            return self.cancel_with(bytes, ledger);
        }

        let final_path = self.dest.join(MANIFEST_FILE);
        faults.check(Site::Publish)?;
        std::fs::rename(&staged, &final_path).map_err(|e| {
            invalid(format!(
                "cannot publish {} as {}: {e}",
                staged.display(),
                final_path.display()
            ))
        })?;
        // Past this point the artifact exists. Nothing below may delete it.
        let bytes = self.plan.payload_bytes();
        let durability = self.confirm_durability(faults);
        append_durably(
            &mut self.journal,
            &journal::phase_line(Phase::Published),
            &Faults::none(),
        )
        .ok();
        let outcome = match durability {
            Ok(()) => Outcome::Published {
                artifact: self.dest.clone(),
                bytes,
            },
            Err(e) => Outcome::PublishedDurabilityUnconfirmed {
                artifact: self.dest.clone(),
                detail: e.to_string(),
            },
        };
        // The journal and the lock go last, and their removal failing does not
        // unpublish anything.
        if faults.check(Site::JournalRemove).is_ok() {
            let _ = std::fs::remove_file(self.dest.join(JOURNAL_FILE));
        }
        let _ = std::fs::remove_file(self.dest.join(LOCK_FILE));
        self.scratch.release(ledger)?;
        Ok(outcome)
    }

    /// `fsync` the destination directory, which is what makes the rename
    /// durable on Linux.
    fn confirm_durability(&self, faults: &Faults) -> Result<()> {
        faults.check(Site::PublishDurability)?;
        faults.check(Site::DirectorySync)?;
        let dir = File::open(&self.dest).map_err(|e| {
            invalid(format!(
                "cannot open {} to sync it: {e}",
                self.dest.display()
            ))
        })?;
        dir.sync_all().map_err(|e| {
            invalid(format!(
                "cannot sync the directory {}: {e}",
                self.dest.display()
            ))
        })
    }

    /// Stop, leaving a documented resumable state.
    pub fn cancel(mut self, ledger: &mut Ledger) -> Result<Outcome> {
        let bytes = self.progress.values().map(|p| p.done).sum();
        self.cancel_with(bytes, ledger)
    }

    fn cancel_with(&mut self, bytes_done: u64, ledger: &mut Ledger) -> Result<Outcome> {
        // The staged manifest is the one thing a cancelled run removes: it is
        // the only file whose presence could later be mistaken for a validated
        // state, and it is cheap to rebuild.
        let _ = std::fs::remove_file(self.dest.join(STAGED_MANIFEST_FILE));
        let _ = std::fs::remove_file(self.dest.join(LOCK_FILE));
        self.scratch.release(ledger)?;
        Ok(Outcome::Cancelled {
            destination: self.dest.clone(),
            bytes_done,
        })
    }

    /// Release the admitted scratch without publishing or cancelling: the
    /// path a failure takes, so a failed run does not leave a charge behind.
    pub fn abandon(mut self, ledger: &mut Ledger) -> Result<()> {
        let _ = std::fs::remove_file(self.dest.join(LOCK_FILE));
        self.scratch.release(ledger)
    }
}

/// Whether a journal file carries the plan line that binds a run.
fn journal_binds(path: &Path) -> Result<bool> {
    let text = read_capped(path, journal::MAX_JOURNAL_BYTES)?;
    Ok(journal::parse(&text)?.binding.is_some())
}

/// Open the journal: append to an existing one, or create it with its header.
fn open_journal(
    path: &Path,
    resuming: bool,
    binding: &RunBinding,
    faults: &Faults,
) -> Result<File> {
    if resuming {
        return OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| invalid(format!("cannot open {}: {e}", path.display())));
    }
    faults.check(Site::JournalCreate)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| invalid(format!("cannot create {}: {e}", path.display())))?;
    append_durably(&mut file, &journal::header_lines(binding), faults)?;
    Ok(file)
}

/// Take the exclusive run lock, or explain who has it.
fn take_lock(dest: &Path, options: &Options, faults: &Faults) -> Result<()> {
    let lock = dest.join(LOCK_FILE);
    faults.check(Site::LockAcquire)?;
    match OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(mut file) => {
            let _ = writeln!(file, "moxie-repack pid {}", std::process::id());
            let _ = file.sync_all();
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if options.take_over_interrupted_run {
                std::fs::remove_file(&lock).map_err(|e| {
                    invalid(format!(
                        "cannot remove the stale lock {}: {e}",
                        lock.display()
                    ))
                })?;
                take_lock(dest, &Options::default(), faults)
            } else {
                Err(invalid(format!(
                    "{} exists: another run owns this destination, or one was interrupted. This \
                     program cannot tell those apart -- that needs process liveness, which one \
                     crate owns (ADR 0006) -- so continuing an interrupted run is an explicit \
                     choice",
                    lock.display()
                )))
            }
        }
        Err(e) => Err(invalid(format!("cannot take {}: {e}", lock.display()))),
    }
}

fn append_durably(file: &mut File, line: &str, faults: &Faults) -> Result<()> {
    faults.check(Site::JournalAppend)?;
    file.write_all(line.as_bytes())
        .map_err(|e| invalid(format!("cannot append to the journal: {e}")))?;
    faults.check(Site::JournalSync)?;
    file.sync_data()
        .map_err(|e| invalid(format!("cannot sync the journal: {e}")))
}

/// Write at an absolute offset without touching the file's own cursor.
fn write_at(file: &mut File, offset: u64, bytes: &[u8], faults: &Faults) -> std::io::Result<()> {
    faults
        .check(Site::ChunkWrite)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(bytes, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(bytes)
    }
}

/// Read one byte range in scratch-sized slices.
fn read_range(
    path: &Path,
    offset: u64,
    len: u64,
    scratch: &mut [u8],
    sink: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let file =
        File::open(path).map_err(|e| invalid(format!("cannot open {}: {e}", path.display())))?;
    let meta = file
        .metadata()
        .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?;
    if offset + len > meta.len() {
        return Err(invalid(format!(
            "{} holds {} byte(s); the range {offset}..{} is past its end",
            path.display(),
            meta.len(),
            offset + len
        )));
    }
    let slice = scratch.len().max(1);
    let mut done = 0u64;
    while done < len {
        let want = ((len - done) as usize).min(slice);
        let buf = &mut scratch[..want];
        read_exact_at(&file, offset + done, buf)
            .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
        sink(buf)?;
        done += want as u64;
    }
    Ok(())
}

fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut owned = file.try_clone()?;
        owned.seek(SeekFrom::Start(offset))?;
        owned.read_exact(buf)
    }
}

fn read_capped(path: &Path, cap: usize) -> Result<String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?;
    if meta.len() > cap as u64 {
        return Err(invalid(format!(
            "{} is {} byte(s), above the {cap} byte cap checked before reading",
            path.display(),
            meta.len()
        )));
    }
    let bytes =
        std::fs::read(path).map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
    String::from_utf8(bytes).map_err(|e| invalid(format!("{} is not UTF-8: {e}", path.display())))
}
