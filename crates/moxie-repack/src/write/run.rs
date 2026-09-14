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

use crate::write::fault::{Faults, Site};
use crate::write::plan::{OutputPlan, PlacedComponent};
use crate::write::{WriteBudget, invalid};

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
    pub kind: moxie_format::canonical::ComponentKind,
    /// The tensor's name inside its shard.
    pub name: String,
    /// The shard file it lives in.
    pub file: String,
    pub length: u64,
    /// SHA-256 of this component's payload bytes, which is the scope
    /// [ADR 0025] records.
    ///
    /// [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md
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
    /// Journal and staged-manifest bytes written so far, charged against the
    /// plan's own allowance so the destination cannot quietly exceed the disk
    /// budget the plan was accepted under.
    overhead_used: u64,
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
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        dest: &Path,
        plan: OutputPlan,
        binding: RunBinding,
        budget: WriteBudget,
        options: &Options,
        ledger: &mut Ledger,
        faults: &Faults,
        cancelled: &dyn Fn() -> bool,
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
        // strand the destination -- but the removal is a **mutation**, so it
        // waits for the lock below. Reading is all that happens here.
        let empty_journal = journal_path.exists();
        let resuming = empty_journal && journal_binds(&journal_path)?;
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
            // Every directory this creates, deepest last, so each one's parent
            // can be synced below: on Linux a new directory entry is durable
            // only once the directory holding it is synced, and reporting
            // `Published` while the destination's own existence is unconfirmed
            // would be the same overstatement the durability outcome exists to
            // avoid.
            let mut created: Vec<PathBuf> = Vec::new();
            let mut ancestor = dest.clone();
            while !ancestor.exists() {
                created.push(ancestor.clone());
                match ancestor.parent() {
                    Some(parent) => ancestor = parent.to_path_buf(),
                    None => break,
                }
            }
            std::fs::create_dir_all(&dest)
                .map_err(|e| invalid(format!("cannot create {}: {e}", dest.display())))?;
            for made in created.iter().rev() {
                if let Some(parent) = made.parent() {
                    let handle = File::open(parent).map_err(|e| {
                        invalid(format!("cannot open {} to sync it: {e}", parent.display()))
                    })?;
                    handle
                        .sync_all()
                        .map_err(|e| invalid(format!("cannot sync {}: {e}", parent.display())))?;
                }
            }
        }

        take_lock(&dest, options, faults)?;

        // Under the lock, and not before it: removing another run's file is a
        // mutation, and the whole point of the lock is that one run at a time
        // decides what this destination holds.
        if empty_journal && !resuming {
            std::fs::remove_file(&journal_path).map_err(|e| {
                invalid(format!(
                    "cannot remove the empty journal {}: {e}",
                    journal_path.display()
                ))
            })?;
        }

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
            overhead_used: 0,
            scratch,
            sealed: None,
        };
        for c in run.plan.components() {
            run.progress.insert(
                c.name.clone(),
                Progress {
                    done: 0,
                    next_index: 0,
                    hasher: StreamingSha256::new(),
                },
            );
        }
        // Every shard exists with its header written and synced before any
        // payload byte is placed: the offsets this plan fixed are offsets past
        // that header, and a resumed run writes into the same file it would
        // have. Rewriting an identical header is idempotent, and cheap -- a
        // header is kilobytes.
        //
        // On a **resume** it happens after `recover`, never before: recovery is
        // where the journal's binding is checked, and independent review
        // resumed with a different selection, was correctly refused, and found
        // the existing shard already rewritten. A destination that is not this
        // run's is a destination this run does not touch.
        //
        // A start that refuses releases what it admitted. `begin` never hands
        // back a `Run` it failed to build, so nothing above can release the
        // scratch on its behalf.
        if resuming {
            let report = match run.recover(&journal_path, faults, cancelled) {
                Ok(report) => report,
                Err(e) => {
                    run.abandon(ledger)?;
                    return Err(e);
                }
            };
            if let Err(e) = run.write_shard_headers(faults) {
                run.abandon(ledger)?;
                return Err(e);
            }
            Ok(Start::Resumed(run, report))
        } else {
            if let Err(e) = run.write_shard_headers(faults) {
                run.abandon(ledger)?;
                return Err(e);
            }
            Ok(Start::Fresh(run))
        }
    }

    /// Rebuild state from the journal, rehashing every byte it claims.
    ///
    /// A journal entry is never evidence its payload is correct. Every unit is
    /// re-read from the staged chunk file and rehashed; a mismatch discards the
    /// unit and everything after it in that tensor, because a tensor's bytes
    /// are a sequence and a hole in it cannot be filled out of order.
    fn recover(
        &mut self,
        journal_path: &Path,
        faults: &Faults,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<ResumeReport> {
        // Bytes, not text: a record torn inside a multi-byte character makes
        // the file invalid UTF-8 past its last committed newline, and that is
        // the state this recovery exists to repair.
        let bytes = moxie_storage::read_bytes_capped(journal_path, journal::MAX_JOURNAL_BYTES)?;
        let state: JournalState = journal::parse_bytes(&bytes)?;
        // **Whose run is this?** Asked before anything is changed. A journal
        // bound to a different plan means this destination is not ours to
        // repair, rewrite or append to, and independent review found the tear
        // repair and the shard-header pass both running ahead of this question.
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
        // **Repair the tear before anything appends to it.** Parsing ignores a
        // torn final line, but leaving it on disk means the next record is
        // written onto the fragment and the journal becomes unparseable for
        // good. An independent review reproduced exactly that: cancel, tear,
        // resume, cancel, and the next resume could never read the journal
        // again. The existing test ran straight through to publication, which
        // deletes the journal and hid it.
        if state.torn_tail_bytes > 0 {
            let keep = bytes.len() - state.torn_tail_bytes;
            let file = open_confined(&self.dest, JOURNAL_FILE, false)?;
            file.set_len(keep as u64).map_err(|e| {
                invalid(format!(
                    "cannot truncate the torn journal {}: {e}",
                    journal_path.display()
                ))
            })?;
            file.sync_all().map_err(|e| {
                invalid(format!(
                    "cannot sync the repaired journal {}: {e}",
                    journal_path.display()
                ))
            })?;
            // **And move this run's own handle back to the repaired end.** It
            // was positioned at the length the file had when it was opened,
            // which is now past EOF, so the next record would be written after
            // a hole of zero bytes -- a journal that parses as neither the old
            // text nor the new one. The regression that found this tore the
            // tail, resumed, and tore it again.
            use std::io::Seek;
            self.journal
                .seek(std::io::SeekFrom::Start(keep as u64))
                .map_err(|e| {
                    invalid(format!(
                        "cannot reposition {} after repairing it: {e}",
                        journal_path.display()
                    ))
                })?;
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
            // Recovery rehashes every byte it reuses, which for a large run is
            // minutes of reading. A cancellation only observed afterwards is a
            // cancellation nobody experiences.
            if cancelled() {
                return Err(invalid(
                    "cancelled while rehashing the staged units of an interrupted run".into(),
                ));
            }
            let Some(planned) = self.plan.component(&unit.tensor).cloned() else {
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
        for shard in self.plan.shards() {
            let name = &shard.file;
            let path = self.dest.join(name);
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            // Never below the header: those bytes are the shard's index, not
            // payload, and truncating them would leave a file no reader can
            // open.
            let accounted = self
                .plan
                .components()
                .iter()
                .filter(|c| &c.file == name)
                .map(|c| c.file_offset + self.progress[&c.name].done)
                .max()
                .unwrap_or(0)
                .max(shard.layout.header_bytes().len() as u64);
            if meta.len() > accounted {
                report.truncated_bytes += meta.len() - accounted;
                let file = open_confined(&self.dest, name, false)?;
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
        planned: &PlacedComponent,
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
        if unit.chunk != planned.file || unit.offset != planned.file_offset + progress.done {
            return Err(invalid(format!(
                "unit {} claims {}@{} but the plan puts those bytes at {}@{}",
                unit.index,
                unit.chunk,
                unit.offset,
                planned.file,
                planned.file_offset + progress.done
            )));
        }
        if progress.done + unit.len > planned.len {
            return Err(invalid(format!(
                "unit {} would take tensor '{}' past its {} byte(s)",
                unit.index, unit.tensor, planned.len
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
        moxie_storage::read_range(
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
    pub fn bytes_done(&self, component: &str) -> Result<u64> {
        self.progress
            .get(component)
            .map(|p| p.done)
            .ok_or_else(|| invalid(format!("component '{component}' is not in this plan")))
    }

    /// Write every shard's header, once, durably.
    fn write_shard_headers(&mut self, faults: &Faults) -> Result<()> {
        for shard in self.plan.shards() {
            faults.check(Site::ShardCreate)?;
            let mut file = open_confined(&self.dest, &shard.file, false)?;
            let header = shard.layout.header_bytes();
            write_at(&mut file, 0, header, faults, Site::ShardHeaderWrite).map_err(|e| {
                invalid(format!("cannot write the header of '{}': {e}", shard.file))
            })?;
            faults.check(Site::ShardHeaderSync)?;
            file.sync_all()
                .map_err(|e| invalid(format!("cannot sync '{}': {e}", shard.file)))?;
        }
        Ok(())
    }

    /// Charge `bytes` against the plan's staging allowance, or refuse.
    ///
    /// The journal and the staged manifest are the destination's other two
    /// files, and the disk budget was checked against payload **plus** a bound
    /// on them. Independent review found that bound was a per-record constant
    /// rather than a function of the names in the records, so a long role
    /// overran it: a 160,000-byte budget retained 166,026 bytes. The bound is
    /// proportional now, and this is what makes it a limit rather than a guess.
    fn charge_overhead(&mut self, bytes: u64, what: &str) -> Result<()> {
        let used = self.overhead_used + bytes;
        if used > self.plan.overhead_bytes() {
            return Err(invalid(format!(
                "writing {bytes} more byte(s) of {what} would take this run's staging files to \
                 {used}, above the {} byte(s) its disk plan reserved for them",
                self.plan.overhead_bytes()
            )));
        }
        self.overhead_used = used;
        Ok(())
    }

    /// Whether every tensor's payload is complete.
    pub fn is_complete(&self) -> bool {
        self.plan
            .components()
            .iter()
            .all(|c| self.progress[&c.name].done == c.len)
    }

    /// Write one bounded work unit: append, sync, then journal.
    ///
    /// The order is the contract. The bytes are durable before the record that
    /// claims them exists, so a crash can leave bytes with no record -- which a
    /// restart discards -- but never a record with no bytes.
    pub fn write_unit(
        &mut self,
        component: &str,
        bytes: &[u8],
        source_sha256: &str,
        faults: &Faults,
    ) -> Result<()> {
        let planned = self
            .plan
            .component(component)
            .cloned()
            .ok_or_else(|| invalid(format!("component '{component}' is not in this plan")))?;
        let progress = self
            .progress
            .get(component)
            .ok_or_else(|| invalid(format!("component '{component}' has no progress")))?;
        if bytes.is_empty() {
            return Err(invalid(format!(
                "component '{component}': an empty unit is not work"
            )));
        }
        if bytes.len() > self.budget.scratch_bytes() {
            return Err(invalid(format!(
                "component '{component}': a {}-byte unit is above the admitted {}-byte payload \
                 scratch",
                bytes.len(),
                self.budget.scratch_bytes()
            )));
        }
        let done = progress.done;
        let end = done + bytes.len() as u64;
        if end > planned.len {
            return Err(invalid(format!(
                "component '{component}': this unit ends at {end} of a {} byte payload",
                planned.len
            )));
        }
        let offset = planned.file_offset + done;
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

        let path = self.dest.join(&planned.file);
        faults.check(Site::ChunkCreate)?;
        let mut file = open_confined(&self.dest, &planned.file, false)?;
        write_at(&mut file, offset, bytes, faults, Site::ChunkWrite)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;

        let unit = CompletedUnit {
            tensor: component.to_string(),
            index,
            chunk: planned.file.clone(),
            offset,
            len: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            source_sha256: source_sha256.to_string(),
        };
        let line = journal::unit_line(&unit);
        self.charge_overhead(line.len() as u64, "journal")?;
        append_durably(&mut self.journal, &line, faults)?;

        let progress = self.progress.get_mut(component).expect("checked above");
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
        let mut out = Vec::with_capacity(self.plan.components().len());
        for c in self.plan.components() {
            let p = &self.progress[&c.name];
            if p.done != c.len {
                return Err(invalid(format!(
                    "component '{}' has {} of {} byte(s): an incomplete tensor cannot be published",
                    c.name, p.done, c.len
                )));
            }
            let mut hasher = StreamingSha256::new();
            core::mem::swap(
                &mut hasher,
                &mut self.progress.get_mut(&c.name).expect("present").hasher,
            );
            out.push(SealedTensor {
                role: c.role.clone(),
                kind: c.kind,
                name: c.name.clone(),
                file: c.file.clone(),
                length: c.len,
                sha256: hasher.finalize_hex(),
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
        self.charge_overhead(manifest_text.len() as u64, "staged manifest")?;
        faults.check(Site::ManifestWrite)?;
        {
            let mut file = open_confined(&self.dest, STAGED_MANIFEST_FILE, false)?;
            file.set_len(0)
                .map_err(|e| invalid(format!("cannot truncate {}: {e}", staged.display())))?;
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
        let mut roles: Vec<&str> = sealed.iter().map(|t| t.role.as_str()).collect();
        roles.dedup();
        for role in roles {
            // Validation reads every published byte, and a cancellation only
            // observed after all of them is a cancellation nobody experiences.
            // Asked **inside** the read as well as before it: one tensor can be
            // gigabytes, and the scratch buffer is what bounds how long a
            // cancellation still waits.
            let want: u64 = sealed
                .iter()
                .filter(|t| t.role == role)
                .map(|t| t.length)
                .sum();
            let Some(read) =
                artifact.verify_tensor_cancellable(role, self.scratch.bytes_mut(), cancelled)?
            else {
                drop(artifact);
                let bytes = self.progress.values().map(|p| p.done).sum();
                return self.cancel_with(bytes, ledger);
            };
            if read != want {
                return Err(invalid(format!(
                    "tensor '{role}' verified {read} byte(s) against a planned {want}"
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
    let bytes = moxie_storage::read_bytes_capped(path, journal::MAX_JOURNAL_BYTES)?;
    Ok(journal::parse_bytes(&bytes)?.binding.is_some())
}

/// Open the journal: append to an existing one, or create it with its header.
fn open_journal(
    path: &Path,
    resuming: bool,
    binding: &RunBinding,
    faults: &Faults,
) -> Result<File> {
    let dest = path
        .parent()
        .ok_or_else(|| invalid("the journal has no destination directory".into()))?;
    if resuming {
        let file = open_confined(dest, JOURNAL_FILE, false)?;
        let len = file
            .metadata()
            .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?
            .len();
        // Append by position rather than by `O_APPEND`, so that the torn-tail
        // repair below can shorten the file and the next record still lands
        // where the repair left off.
        use std::io::Seek;
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(len))
            .map_err(|e| invalid(format!("cannot seek {}: {e}", path.display())))?;
        return Ok(file);
    }
    faults.check(Site::JournalCreate)?;
    let mut file = open_confined(dest, JOURNAL_FILE, true)?;
    append_durably(&mut file, &journal::header_lines(binding), faults)?;
    Ok(file)
}

/// Open a file inside the destination for writing, refusing every escape
/// **before** a byte is written.
///
/// Three checks, and all three are needed. The name must be a single component
/// of the destination, checked on the string. The path must not already be a
/// symlink, checked with `symlink_metadata`, which does not follow one. And the
/// open itself passes `O_NOFOLLOW`, which closes the window between the two.
///
/// An independent review cancelled a run, replaced `.moxie-repack-manifest`
/// with a symlink to an unrelated file and resumed: `File::create` followed it
/// and truncated the target, and the escape was rejected afterwards -- by
/// validation, after the damage. Confinement has to happen before the mutation,
/// not after it.
fn open_confined(dest: &Path, name: &str, create_new: bool) -> Result<File> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return Err(invalid(format!(
            "'{name}' is not a single component of the destination"
        )));
    }
    let path = dest.join(name);
    match std::fs::symlink_metadata(&path) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(invalid(format!(
                    "{} is a symbolic link: this run writes only files it created, and following \
                     one would write outside the destination it was given",
                    path.display()
                )));
            }
            if !meta.is_file() {
                return Err(invalid(format!("{} is not a regular file", path.display())));
            }
            if create_new {
                return Err(invalid(format!("{} already exists", path.display())));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(invalid(format!("cannot stat {}: {e}", path.display())));
        }
    }
    let mut options = OpenOptions::new();
    options.write(true).truncate(false);
    if create_new {
        options.create_new(true);
    } else {
        options.create(true);
    }
    #[cfg(unix)]
    {
        // `O_NOFOLLOW`: if the final component is a symbolic link, the open
        // fails rather than following it. The stat above can go stale between
        // the check and the open; this cannot.
        use std::os::unix::fs::OpenOptionsExt;
        const O_NOFOLLOW: i32 = 0o400000;
        options.custom_flags(O_NOFOLLOW);
    }
    let file = options
        .open(&path)
        .map_err(|e| invalid(format!("cannot open {}: {e}", path.display())))?;
    // The check that `O_NOFOLLOW` cannot make: a **hard link**. A regular file
    // inside the destination can be a second name for a file anywhere on the
    // same filesystem, and writing through it writes there. Independent review
    // cancelled a run, hard-linked an unrelated file over a private name, and
    // watched publication overwrite it.
    //
    // Made on the opened handle rather than on the path, so nothing can be
    // swapped in between the question and the write. A file this run created
    // has exactly one name; anything else is an alias it did not make.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = file
            .metadata()
            .map_err(|e| invalid(format!("cannot stat the open {}: {e}", path.display())))?;
        if !meta.is_file() {
            return Err(invalid(format!(
                "{} is not a regular file once opened",
                path.display()
            )));
        }
        if meta.nlink() != 1 {
            return Err(invalid(format!(
                "{} has {} names: a hard link makes it a second name for a file this run did not \
                 create, and writing through it writes outside the destination",
                path.display(),
                meta.nlink()
            )));
        }
    }
    Ok(file)
}

/// Take the exclusive run lock, or explain who has it.
fn take_lock(dest: &Path, options: &Options, faults: &Faults) -> Result<()> {
    let lock = dest.join(LOCK_FILE);
    faults.check(Site::LockAcquire)?;
    // `symlink_metadata` does not follow a link, so a lock file replaced by one
    // is a refusal rather than a write somewhere else.
    let held = match std::fs::symlink_metadata(&lock) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(invalid(format!(
                    "{} is a symbolic link: a run lock is a file this program created",
                    lock.display()
                )));
            }
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(invalid(format!("cannot stat {}: {e}", lock.display()))),
    };
    if held {
        if !options.take_over_interrupted_run {
            return Err(invalid(format!(
                "{} exists: another run owns this destination, or one was interrupted. This \
                 program cannot tell those apart -- that needs process liveness, which one \
                 crate owns (ADR 0006) -- so continuing an interrupted run is an explicit \
                 choice",
                lock.display()
            )));
        }
        std::fs::remove_file(&lock).map_err(|e| {
            invalid(format!(
                "cannot remove the stale lock {}: {e}",
                lock.display()
            ))
        })?;
    }
    let mut file = open_confined(dest, LOCK_FILE, true)?;
    let _ = writeln!(file, "moxie-repack pid {}", std::process::id());
    let _ = file.sync_all();
    Ok(())
}

fn append_durably(file: &mut File, line: &str, faults: &Faults) -> Result<()> {
    faults.check(Site::JournalAppend)?;
    file.write_all(line.as_bytes())
        .map_err(|e| invalid(format!("cannot append to the journal: {e}")))?;
    faults.check(Site::JournalSync)?;
    file.sync_data()
        .map_err(|e| invalid(format!("cannot sync the journal: {e}")))
}

/// Write at an absolute offset without touching the file's own cursor, failing
/// first at the boundary the caller names.
///
/// The site is a parameter because a shard's header and a shard's payload are
/// two different durable boundaries that happen to share a syscall: the header
/// pass runs on every `begin`, so a fault injected for a payload unit would
/// otherwise always fire at a header first and measure the wrong thing.
fn write_at(
    file: &mut File,
    offset: u64,
    bytes: &[u8],
    faults: &Faults,
    site: Site,
) -> std::io::Result<()> {
    faults
        .check(site)
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
