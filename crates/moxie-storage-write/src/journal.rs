//! The private restart journal: one TOML document per line, append-only.
//!
//! [ADR 0023] fixes the format and says why it is not a manifest field: a
//! resumable run is not a property of a published artifact. The journal lives
//! in the destination directory as a dot-file, is never part of what a reader
//! opens, and is removed when the run publishes or is abandoned.
//!
//! Every line is a complete TOML document with exactly one key, parsed with the
//! parser this repository already builds with. Two properties follow, and both
//! are load-bearing:
//!
//! * **A torn line is discarded, and nothing before it is.** A crash between
//!   `write` and `fsync` can leave a partial final line. A line without a
//!   trailing newline is dropped on read, so the last durable record is the
//!   last complete one.
//! * **No hand-rolled parser.** ADR 0004 is four reviews of what parsing
//!   structured text by hand costs here.
//!
//! A `unit` line may be appended only **after** its payload bytes are durable,
//! and it is never evidence that they are correct: [`Journal::read`] returns
//! what was recorded and the caller rehashes the bytes it names.
//!
//! [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_types::{Error, Result};

use crate::fault::{Faults, Site};

/// The journal format this writer emits and this reader accepts.
pub const JOURNAL_VERSION: u32 = 1;

/// A journal larger than this is refused before it is parsed, exactly as the
/// manifest reader caps `manifest.toml`: the bound is on the parse, not on the
/// file that happens to be there.
pub const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

/// What binds a run to its inputs. Resume with any field different is a
/// refusal, not a merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunBinding {
    /// Digest over the selection, the output plan and every source range's
    /// identity. Computed by the caller, which is the only layer that knows
    /// what was selected.
    pub plan_digest: String,
    /// The converter that produced the canonical bytes. A different importer
    /// version may produce different bytes and may not continue a run.
    pub converter: String,
    /// The canonical schema the output is being written against.
    pub schema_version: u32,
}

/// One durable, checksummed work unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedUnit {
    /// The tensor role these bytes belong to.
    pub tensor: String,
    /// The unit's position in that tensor's byte sequence, counted from zero.
    pub index: u64,
    pub chunk: String,
    pub offset: u64,
    pub len: u64,
    /// SHA-256 of the bytes this unit wrote.
    pub sha256: String,
    /// SHA-256 of the **source** bytes it converted, so a source that changed
    /// under a resumed run is caught rather than half-converted.
    pub source_sha256: String,
}

/// How far the run had got when it last wrote to the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Units are still being written.
    Writing,
    /// Every unit is durable and the staged manifest has been validated
    /// through the production reader. The only step left is the publication
    /// rename.
    Validated,
    /// The manifest rename returned success. Restart reconciles rather than
    /// deleting anything.
    Published,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Writing => "writing",
            Phase::Validated => "validated",
            Phase::Published => "published",
        }
    }
}

/// The state one journal file describes.
#[derive(Debug, Clone)]
pub struct JournalState {
    pub binding: RunBinding,
    pub units: Vec<CompletedUnit>,
    pub phase: Phase,
    /// Lines that were present but incomplete: a torn final write. Reported
    /// rather than silently dropped, because "the journal was truncated" is
    /// something a restart should be able to say out loud.
    pub torn_tail_bytes: usize,
}

/// An open journal, appended to under an explicit fault-injection plan.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: File,
}

fn invalid(detail: String) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

impl Journal {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create a journal, refusing to overwrite one.
    pub fn create(path: &Path, binding: &RunBinding, faults: &Faults) -> Result<Self> {
        faults.check(Site::JournalCreate)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| invalid(format!("cannot create journal {}: {e}", path.display())))?;
        let mut journal = Self {
            path: path.to_path_buf(),
            file,
        };
        journal.append_line(&format!("version = {JOURNAL_VERSION}\n"), faults)?;
        let line = format!(
            "plan = {{ plan_digest = \"{}\", converter = \"{}\", schema_version = {} }}\n",
            escape(&binding.plan_digest),
            escape(&binding.converter),
            binding.schema_version
        );
        journal.append_line(&line, faults)?;
        Ok(journal)
    }

    /// Open an existing journal for appending.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| invalid(format!("cannot open journal {}: {e}", path.display())))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Append one completed unit. The caller must already have made the bytes
    /// it names durable.
    pub fn record_unit(&mut self, unit: &CompletedUnit, faults: &Faults) -> Result<()> {
        let line = format!(
            "unit = {{ tensor = \"{}\", index = {}, chunk = \"{}\", offset = {}, len = {}, \
             sha256 = \"{}\", source_sha256 = \"{}\" }}\n",
            escape(&unit.tensor),
            unit.index,
            escape(&unit.chunk),
            unit.offset,
            unit.len,
            escape(&unit.sha256),
            escape(&unit.source_sha256),
        );
        self.append_line(&line, faults)
    }

    /// Append a phase marker.
    pub fn record_phase(&mut self, phase: Phase, faults: &Faults) -> Result<()> {
        self.append_line(&format!("phase = \"{}\"\n", phase.name()), faults)
    }

    fn append_line(&mut self, line: &str, faults: &Faults) -> Result<()> {
        faults.check(Site::JournalAppend)?;
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| invalid(format!("cannot append to {}: {e}", self.path.display())))?;
        faults.check(Site::JournalSync)?;
        self.file
            .sync_data()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", self.path.display())))?;
        Ok(())
    }

    /// Read and validate a journal.
    ///
    /// Refused by version before any other line is interpreted, exactly as a
    /// future manifest is: a journal written by a later repacker is not a
    /// journal this one may continue.
    pub fn read(path: &Path) -> Result<JournalState> {
        let bytes = std::fs::read(path)
            .map_err(|e| invalid(format!("cannot read journal {}: {e}", path.display())))?;
        if bytes.len() > MAX_JOURNAL_BYTES {
            return Err(invalid(format!(
                "journal {} is {} byte(s), above the {MAX_JOURNAL_BYTES} cap checked before parsing",
                path.display(),
                bytes.len()
            )));
        }
        let text = String::from_utf8(bytes)
            .map_err(|e| invalid(format!("journal {} is not UTF-8: {e}", path.display())))?;
        // A final line without its newline is a torn append: the bytes were
        // written and the crash arrived before the rest of the record.
        let (complete, torn) = match text.rfind('\n') {
            Some(at) => (&text[..=at], text.len() - at - 1),
            None => ("", text.len()),
        };

        let mut version: Option<u32> = None;
        let mut binding: Option<RunBinding> = None;
        let mut units: Vec<CompletedUnit> = Vec::new();
        let mut phase = Phase::Writing;
        for (n, line) in complete.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: toml::Value = toml::from_str(line).map_err(|e| {
                invalid(format!(
                    "journal {} line {}: not one TOML document: {e}",
                    path.display(),
                    n + 1
                ))
            })?;
            let table = value.as_table().ok_or_else(|| {
                invalid(format!("journal {} line {}: not a table", path.display(), n + 1))
            })?;
            let (key, value) = match table.iter().next() {
                Some(kv) if table.len() == 1 => kv,
                _ => {
                    return Err(invalid(format!(
                        "journal {} line {}: a journal line carries exactly one key",
                        path.display(),
                        n + 1
                    )));
                }
            };
            match key.as_str() {
                "version" => {
                    let v = value.as_integer().and_then(|v| u32::try_from(v).ok());
                    let v = v.ok_or_else(|| {
                        invalid(format!("journal {}: version is not a u32", path.display()))
                    })?;
                    if v != JOURNAL_VERSION {
                        return Err(invalid(format!(
                            "journal {} is version {v}; this repacker writes and continues only \
                             version {JOURNAL_VERSION}: refused by version before any other field",
                            path.display()
                        )));
                    }
                    version = Some(v);
                }
                "plan" => {
                    if version.is_none() {
                        return Err(invalid(format!(
                            "journal {}: the plan line precedes the version line",
                            path.display()
                        )));
                    }
                    binding = Some(RunBinding {
                        plan_digest: string_field(value, "plan_digest", path)?,
                        converter: string_field(value, "converter", path)?,
                        schema_version: u32::try_from(integer_field(value, "schema_version", path)?)
                            .map_err(|_| {
                                invalid(format!(
                                    "journal {}: schema_version is out of range",
                                    path.display()
                                ))
                            })?,
                    });
                }
                "unit" => {
                    if binding.is_none() {
                        return Err(invalid(format!(
                            "journal {}: a unit precedes the plan it belongs to",
                            path.display()
                        )));
                    }
                    units.push(CompletedUnit {
                        tensor: string_field(value, "tensor", path)?,
                        index: integer_field(value, "index", path)?,
                        chunk: string_field(value, "chunk", path)?,
                        offset: integer_field(value, "offset", path)?,
                        len: integer_field(value, "len", path)?,
                        sha256: string_field(value, "sha256", path)?,
                        source_sha256: string_field(value, "source_sha256", path)?,
                    });
                }
                "phase" => {
                    phase = match value.as_str() {
                        Some("writing") => Phase::Writing,
                        Some("validated") => Phase::Validated,
                        Some("published") => Phase::Published,
                        other => {
                            return Err(invalid(format!(
                                "journal {}: unknown phase {other:?}",
                                path.display()
                            )));
                        }
                    };
                }
                other => {
                    return Err(invalid(format!(
                        "journal {}: unknown line key '{other}'",
                        path.display()
                    )));
                }
            }
        }
        let binding = binding.ok_or_else(|| {
            invalid(format!(
                "journal {} records no plan: it cannot bind a resume",
                path.display()
            ))
        })?;
        Ok(JournalState {
            binding,
            units,
            phase,
            torn_tail_bytes: torn,
        })
    }
}

fn string_field(value: &toml::Value, field: &str, path: &Path) -> Result<String> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            invalid(format!(
                "journal {}: a record is missing the string field '{field}'",
                path.display()
            ))
        })
}

fn integer_field(value: &toml::Value, field: &str, path: &Path) -> Result<u64> {
    let raw = value.get(field).and_then(|v| v.as_integer()).ok_or_else(|| {
        invalid(format!(
            "journal {}: a record is missing the integer field '{field}'",
            path.display()
        ))
    })?;
    u64::try_from(raw).map_err(|_| {
        invalid(format!(
            "journal {}: '{field}' is {raw}, which is not a byte count",
            path.display()
        ))
    })
}

/// Escape a value for a TOML basic string.
///
/// Only the four things a role or a digest could contain that TOML basic
/// strings cannot: a quote, a backslash, a newline and a carriage return. It
/// is not a general serializer -- every field written through it is either a
/// hex digest, a chunk name the manifest validator has already constrained, or
/// a role, and the round trip is checked by the journal's own tests.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
