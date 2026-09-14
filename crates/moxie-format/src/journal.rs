//! The repack restart journal: schema, encoding and parsing, I/O-free.
//!
//! [ADR 0023] fixes the format and says why it is not a manifest field: a
//! resumable run is not a property of a published artifact. The journal lives
//! in the destination directory as a dot-file, is never part of what a reader
//! opens, and is removed when the run publishes or is abandoned.
//!
//! This module is its schema, exactly as [`crate::manifest`] is the manifest's:
//! it turns records into lines and a `&str` into validated records, and never
//! learns that a file exists. The offline repacker's `write` module owns the
//! file half -- creating it, appending to it, syncing it, and reading it back
//! under a cap through `moxie-storage`.
//!
//! Every line is one complete TOML document with exactly one key. Two
//! properties follow, and both are load-bearing:
//!
//! * **A torn line is discarded, and nothing before it is.** A crash between
//!   the write and the sync can leave a partial final line. A line with no
//!   trailing newline is dropped by [`parse`], so the last record it returns is
//!   the last complete one.
//! * **No hand-rolled parser.** ADR 0004 is four reviews of what parsing
//!   structured text by hand costs in this repository.
//!
//! A `unit` line may be written only **after** the bytes it names are durable,
//! and it is never evidence that they are correct: the caller rehashes them.
//!
//! [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md

use moxie_types::{Error, Result};

/// The journal format this writer emits and this reader continues.
pub const JOURNAL_VERSION: u32 = 1;

/// Text above this is refused before it is parsed, exactly as the manifest
/// reader caps `manifest.toml`: the bound is on the parse.
pub const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed repack journal (detail unavailable: out of memory)",
        detail,
    )
}

fn invalid_static(detail: &'static str) -> Error {
    crate::invalid_static(detail)
}

/// What binds a run to its inputs. A resume with any field different is a
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
    /// The canonical schema the output is written against.
    pub schema_version: u32,
}

/// One durable, checksummed work unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedUnit {
    /// The tensor role these bytes belong to.
    pub tensor: String,
    /// The unit's position in that tensor's byte sequence, from zero.
    pub index: u64,
    pub chunk: String,
    pub offset: u64,
    pub len: u64,
    /// SHA-256 of the bytes this unit wrote.
    pub sha256: String,
    /// SHA-256 of the **source** bytes this unit converted.
    ///
    /// **Recorded provenance, not a check.** What refuses a source that changed
    /// under a resumed run is the run binding, whose digest covers every source
    /// file **whole** and is recomputed on every start; re-reading each
    /// completed unit's source range to compare this field would buy nothing
    /// and cost the reads a resume exists to avoid. It is written down because
    /// a journal that says which bytes produced which output is worth having
    /// when something has gone wrong, and task 0025's mutation battery carries
    /// an independence control that says out loud that nothing compares it.
    pub source_sha256: String,
}

/// How far a run had got when it last wrote to its journal.
///
/// **Recorded and reported, never branched on.** A restart revalidates and
/// republishes whatever the phase says, because the alternative is trusting a
/// marker about bytes that may have changed since it was written -- and the
/// work it would save is one pass of checksums the run has to be able to do
/// anyway. What the phase is for is telling a person what happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Units are still being written.
    Writing,
    /// Every unit is durable and the staged manifest has been validated
    /// through the production reader. Only the publication rename is left.
    Validated,
    /// The publication rename returned success. A restart reconciles; it never
    /// deletes a possibly published output.
    Published,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Writing => "writing",
            Phase::Validated => "validated",
            Phase::Published => "published",
        }
    }
}

/// What one journal's text describes.
#[derive(Debug, Clone)]
pub struct JournalState {
    /// `None` when the text carries no `plan` line.
    ///
    /// That is a real state, not a corrupt one: the header is written before
    /// any unit can be, so a journal without a binding is a run that was
    /// interrupted between creating the file and recording what it was for.
    /// It records nothing, and the caller starts over rather than refusing a
    /// destination forever.
    pub binding: Option<RunBinding>,
    pub units: Vec<CompletedUnit>,
    pub phase: Phase,
    /// Bytes after the last newline: a torn final append. Reported rather than
    /// silently dropped, because "the journal was truncated" is something a
    /// restart should be able to say out loud.
    pub torn_tail_bytes: usize,
}

/// The first two lines of a journal: its version, then the binding.
pub fn header_lines(binding: &RunBinding) -> String {
    format!(
        "version = {JOURNAL_VERSION}\nplan = {{ plan_digest = \"{}\", converter = \"{}\", \
         schema_version = {} }}\n",
        escape(&binding.plan_digest),
        escape(&binding.converter),
        binding.schema_version
    )
}

/// One `unit` line.
pub fn unit_line(unit: &CompletedUnit) -> String {
    format!(
        "unit = {{ tensor = \"{}\", index = {}, chunk = \"{}\", offset = {}, len = {}, \
         sha256 = \"{}\", source_sha256 = \"{}\" }}\n",
        escape(&unit.tensor),
        unit.index,
        escape(&unit.chunk),
        unit.offset,
        unit.len,
        escape(&unit.sha256),
        escape(&unit.source_sha256),
    )
}

/// One `phase` line.
pub fn phase_line(phase: Phase) -> String {
    format!("phase = \"{}\"\n", phase.name())
}

/// Parse and validate a journal's text.
///
/// Refused by version before any other line is interpreted, exactly as a future
/// manifest is: a journal written by a later repacker is not one this repacker
/// may continue.
pub fn parse(text: &str) -> Result<JournalState> {
    if text.len() > MAX_JOURNAL_BYTES {
        return Err(invalid(format_args!(
            "journal is {} byte(s), above the {MAX_JOURNAL_BYTES} cap checked before parsing",
            text.len()
        )));
    }
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
        let n = n + 1;
        if line.trim().is_empty() {
            continue;
        }
        let value: toml::Value = toml::from_str(line).map_err(|e| {
            invalid(format_args!(
                "journal line {n} is not one TOML document: {e}"
            ))
        })?;
        let table = value
            .as_table()
            .ok_or_else(|| invalid(format_args!("journal line {n} is not a table")))?;
        if table.len() != 1 {
            return Err(invalid(format_args!(
                "journal line {n} carries {} keys; a journal line carries exactly one",
                table.len()
            )));
        }
        let (key, value) = table.iter().next().expect("exactly one key");
        match key.as_str() {
            "version" => {
                let v = value
                    .as_integer()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| invalid_static("journal version is not a u32"))?;
                if v != JOURNAL_VERSION {
                    return Err(invalid(format_args!(
                        "journal is version {v}; this repacker writes and continues only version \
                         {JOURNAL_VERSION}: refused by version before any other field"
                    )));
                }
                version = Some(v);
            }
            "plan" => {
                if version.is_none() {
                    return Err(invalid_static(
                        "the journal's plan line precedes its version line",
                    ));
                }
                binding = Some(RunBinding {
                    plan_digest: string_field(value, "plan_digest")?,
                    converter: string_field(value, "converter")?,
                    schema_version: u32::try_from(integer_field(value, "schema_version")?)
                        .map_err(|_| invalid_static("journal schema_version is out of range"))?,
                });
            }
            "unit" => {
                if binding.is_none() {
                    return Err(invalid_static(
                        "a journal unit precedes the plan it belongs to",
                    ));
                }
                units.push(CompletedUnit {
                    tensor: string_field(value, "tensor")?,
                    index: integer_field(value, "index")?,
                    chunk: string_field(value, "chunk")?,
                    offset: integer_field(value, "offset")?,
                    len: integer_field(value, "len")?,
                    sha256: hex_field(value, "sha256")?,
                    source_sha256: hex_field(value, "source_sha256")?,
                });
            }
            "phase" => {
                phase = match value.as_str() {
                    Some("writing") => Phase::Writing,
                    Some("validated") => Phase::Validated,
                    Some("published") => Phase::Published,
                    other => {
                        return Err(invalid(format_args!("unknown journal phase {other:?}")));
                    }
                };
            }
            other => {
                return Err(invalid(format_args!("unknown journal line key '{other}'")));
            }
        }
    }
    Ok(JournalState {
        binding,
        units,
        phase,
        torn_tail_bytes: torn,
    })
}

fn string_field(value: &toml::Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            invalid(format_args!(
                "a journal record is missing the string field '{field}'"
            ))
        })
}

/// A 64-hex-digit field. Checked here rather than where it is compared: a
/// digest that is not a digest must not reach a comparison that could pass.
fn hex_field(value: &toml::Value, field: &str) -> Result<String> {
    let s = string_field(value, field)?;
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format_args!(
            "journal field '{field}' is {s:?}, not 64 hex digits"
        )));
    }
    Ok(s.to_ascii_lowercase())
}

fn integer_field(value: &toml::Value, field: &str) -> Result<u64> {
    let raw = value
        .get(field)
        .and_then(|v| v.as_integer())
        .ok_or_else(|| {
            invalid(format_args!(
                "a journal record is missing the integer field '{field}'"
            ))
        })?;
    u64::try_from(raw).map_err(|_| {
        invalid(format_args!(
            "journal field '{field}' is {raw}, which is not a byte count"
        ))
    })
}

/// Escape a value for a TOML basic string.
///
/// Every field written through it is a hex digest, a chunk name the manifest
/// validator has already constrained, or a tensor role -- and a role is
/// arbitrary text, which is why this exists at all. It is not a general
/// serializer; `journal_lines_round_trip` is what keeps it honest, including
/// for roles carrying quotes, backslashes and control characters.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> RunBinding {
        RunBinding {
            plan_digest: "a".repeat(64),
            converter: "moxie-repack/0.0.0".into(),
            schema_version: 1,
        }
    }

    fn unit(tensor: &str, index: u64) -> CompletedUnit {
        CompletedUnit {
            tensor: tensor.into(),
            index,
            chunk: "chunk0.bin".into(),
            offset: index * 16,
            len: 16,
            sha256: "b".repeat(64),
            source_sha256: "c".repeat(64),
        }
    }

    fn journal(units: &[CompletedUnit], phase: Option<Phase>) -> String {
        let mut text = header_lines(&binding());
        for u in units {
            text.push_str(&unit_line(u));
        }
        if let Some(p) = phase {
            text.push_str(&phase_line(p));
        }
        text
    }

    #[test]
    fn journal_lines_round_trip() {
        // Roles a hand-written quoter gets wrong, including the control
        // characters TOML basic strings cannot carry literally.
        for role in [
            "layers.0.q",
            "a\"quoted\"role",
            "back\\slash",
            "new\nline",
            "tab\there",
            "bell\u{7}",
            "delete\u{7f}",
            "unicode-\u{4e2d}",
        ] {
            let units = [unit(role, 0), unit(role, 1)];
            let state = parse(&journal(&units, Some(Phase::Validated))).expect("it parses");
            assert_eq!(state.binding, Some(binding()));
            assert_eq!(state.units, units);
            assert_eq!(state.phase, Phase::Validated);
            assert_eq!(state.torn_tail_bytes, 0);
        }
    }

    #[test]
    fn a_torn_final_line_is_dropped_and_nothing_before_it_is() {
        let full = journal(&[unit("w", 0), unit("w", 1)], None);
        // Every truncation inside the last line: the first unit survives, the
        // torn one does not, and the tear is reported.
        let last = full[..full.len() - 1].rfind('\n').expect("two lines") + 1;
        for cut in last..full.len() - 1 {
            let state = parse(&full[..cut]).expect("a torn journal still parses");
            assert_eq!(state.units, [unit("w", 0)], "cut at {cut}");
            assert_eq!(state.torn_tail_bytes, cut - last);
        }
        // And the untorn text keeps both.
        assert_eq!(parse(&full).unwrap().units.len(), 2);
    }

    /// A journal with no plan line is a run that never recorded what it was
    /// for. It parses, it says so, and the caller starts over: refusing it
    /// would strand a destination on a crash between two writes.
    #[test]
    fn a_journal_with_no_plan_line_records_no_run() {
        let state = parse("version = 1\n").expect("it parses");
        assert!(state.binding.is_none());
        assert!(state.units.is_empty());
        let state = parse("").expect("an empty journal parses");
        assert!(state.binding.is_none());
        // A unit without a plan is still a contradiction: the header comes
        // first, so a unit that precedes it cannot have been written by this
        // writer.
        let e = parse(&format!("version = 1\n{}", unit_line(&unit("w", 0)))).unwrap_err();
        assert!(e.to_string().contains("precedes the plan"), "{e}");
    }

    #[test]
    fn a_future_journal_is_refused_by_version() {
        let text = journal(&[unit("w", 0)], None).replace("version = 1", "version = 2");
        let e = parse(&text).unwrap_err();
        assert!(e.to_string().contains("refused by version"), "{e}");
    }

    #[test]
    fn structurally_impossible_journals_are_refused() {
        for (text, needle) in [
            (
                journal(&[], None).replace("version = 1\n", ""),
                "precedes its version line",
            ),
            (
                format!("version = 1\n{}", unit_line(&unit("w", 0))),
                "precedes the plan",
            ),
            (
                journal(&[], None).replace("plan = ", "plans = "),
                "unknown journal line key",
            ),
            (
                journal(&[unit("w", 0)], None).replace("sha256 = \"bbbb", "sha256 = \"zzzz"),
                "not 64 hex digits",
            ),
            (
                journal(&[unit("w", 0)], None).replace("len = 16", "len = -1"),
                "not a byte count",
            ),
            (
                journal(&[], None) + "phase = \"midway\"\n",
                "unknown journal phase",
            ),
            (
                journal(&[], None) + "version = 1\nplan = { plan_digest = \"x\" }\n",
                "missing the string field",
            ),
        ] {
            let e = parse(&text).unwrap_err();
            assert!(e.to_string().contains(needle), "{needle:?} not in {e}");
        }
    }

    /// The one-key rule is what makes a torn tail recoverable, and TOML's own
    /// grammar is most of the enforcement: two key-value pairs cannot share a
    /// line. What this checks is the case the grammar allows -- a line that is
    /// valid TOML but is not one record -- and that a journal whose lines are
    /// separated normally is unaffected.
    #[test]
    fn a_line_that_is_not_one_record_is_refused() {
        let mut text = journal(&[], None);
        text.push_str("unit = { tensor = \"w\" } # trailing comment\n");
        let e = parse(&text).unwrap_err();
        assert!(e.to_string().contains("missing the integer field"), "{e}");
        // A blank line between records is not a record and is skipped.
        let spaced = journal(&[unit("w", 0)], None).replace("unit = ", "\nunit = ");
        assert_eq!(parse(&spaced).unwrap().units.len(), 1);
    }
}
