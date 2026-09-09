//! The host sensor: what this machine says about its own memory.
//!
//! [ADR 0006](../../../docs/decisions/adr/0006-host-telemetry-owner.md) makes
//! this the one component that may read machine telemetry, and `arch-check`
//! enforces it: no other production source may name a `/proc` or `/sys` path,
//! and `moxie-memory` may not depend on this crate, because the ledger never
//! probes.
//!
//! This crate reads and reports. It decides nothing about budgets -- turning a
//! reading into an admissible figure is `moxie-memory`'s rule, and joining the
//! two is the composition root's job. The same split as `moxie-cuda` and a
//! device, one tier over.
//!
//! Everything here is testable without touching `/proc`: [`read_under`] takes
//! the root to read from, so a cgroup limit that does not exist on this machine
//! is still exercised.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use moxie_types::{Error, HostLimit, MeasuredHost, Result};

/// Fields required from `meminfo`. A missing one is an error, never a default:
/// a budget computed from an absent number is a guess wearing a measurement's
/// clothes.
const REQUIRED: &[&str] = &[
    "MemTotal",
    "MemAvailable",
    "MemFree",
    "Buffers",
    "Cached",
    "SwapTotal",
    "SwapFree",
];

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

/// Measure the host, reading under `/`.
pub fn read() -> Result<MeasuredHost> {
    read_under(Path::new("/"))
}

/// Measure the host, reading under `root`.
///
/// The injectable root is what makes every rule in this module testable: the
/// benchmark machine has no cgroup limit at any level, so the limited paths
/// would otherwise be reasoned about rather than run.
pub fn read_under(root: &Path) -> Result<MeasuredHost> {
    let meminfo = slurp(&root.join("proc/meminfo"), "meminfo")?;
    let fields = parse_meminfo(&meminfo)?;
    let get = |name: &'static str| -> Result<u64> {
        fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
            .ok_or_else(|| {
                invalid(
                    "meminfo",
                    match name {
                        // The one absence worth explaining, because the
                        // temptation is to substitute a formula for it.
                        "MemAvailable" => "MemAvailable is missing (kernels before 3.14). The \
                                           substitute formula is a reclaim implementation detail \
                                           that has changed; this refuses rather than guess a \
                                           budget"
                            .to_string(),
                        other => format!("{other} is missing from meminfo"),
                    },
                )
            })
    };

    let machine_total_bytes = get("MemTotal")?;
    let machine_available_bytes = get("MemAvailable")?;
    if machine_available_bytes > machine_total_bytes {
        return Err(invalid(
            "meminfo",
            format!(
                "MemAvailable {machine_available_bytes} B exceeds MemTotal {machine_total_bytes} B"
            ),
        ));
    }

    let limit = cgroup_limit(root);
    let (total_bytes, available_bytes) = match &limit {
        HostLimit::Machine => (machine_total_bytes, machine_available_bytes),
        HostLimit::Cgroup {
            limit_bytes,
            current_bytes,
            ..
        } => (
            machine_total_bytes.min(*limit_bytes),
            machine_available_bytes.min(limit_bytes.saturating_sub(*current_bytes)),
        ),
    };

    Ok(MeasuredHost {
        total_bytes,
        available_bytes,
        machine_total_bytes,
        machine_available_bytes,
        free_bytes: get("MemFree")?,
        buffers_bytes: get("Buffers")?,
        cached_bytes: get("Cached")?,
        swap_total_bytes: get("SwapTotal")?,
        swap_free_bytes: get("SwapFree")?,
        limit,
    })
}

fn slurp(path: &Path, what: &'static str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| invalid(what, format!("{}: {e}", path.display())))
}

/// `MemTotal:  264005080 kB` -> `("MemTotal", 264005080 * 1024)`.
///
/// Only the fields this crate needs are converted, but every line is inspected,
/// so a malformed value in a required field is an error rather than an absence.
fn parse_meminfo(text: &str) -> Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if !REQUIRED.contains(&key) {
            continue;
        }
        let rest = rest.trim();
        let (value, unit) = match rest.split_once(char::is_whitespace) {
            Some((v, u)) => (v, u.trim()),
            // A unitless field exists in meminfo (HugePages_Total). None of the
            // required ones are, so an absent unit is malformed here.
            None => (rest, ""),
        };
        let value: u64 = value.parse().map_err(|e| {
            invalid(
                "meminfo",
                format!("{key}: {value:?} is not a byte count: {e}"),
            )
        })?;
        let bytes = match unit {
            "kB" => value.checked_mul(1024).ok_or_else(|| {
                invalid(
                    "meminfo",
                    format!("{key}: {value} kB overflows a byte count"),
                )
            })?,
            other => {
                return Err(invalid(
                    "meminfo",
                    format!("{key}: unit {other:?} is not the kB this format uses"),
                ));
            }
        };
        out.push((key.to_string(), bytes));
    }
    Ok(out)
}

/// The binding cgroup v2 limit, or [`HostLimit::Machine`].
///
/// A limit set on an ancestor binds just as hard as one on the leaf, so this
/// walks up and takes the smallest. Absence at every step is not an error: a
/// cgroup v1 machine, an unmounted hierarchy or an unreadable file all mean the
/// machine's own figures govern, and the descriptor records which view was used.
fn cgroup_limit(root: &Path) -> HostLimit {
    let Ok(text) = std::fs::read_to_string(root.join("proc/self/cgroup")) else {
        return HostLimit::Machine;
    };
    // cgroup v2 is the single `0::<path>` line. A v1-only machine has none.
    let Some(rel) = text
        .lines()
        .find_map(|l| l.strip_prefix("0::").map(str::trim))
    else {
        return HostLimit::Machine;
    };

    let base = root.join("sys/fs/cgroup");
    let mut here = PathBuf::from(rel.trim_start_matches('/'));
    let mut binding: Option<(String, u64, u64)> = None;
    loop {
        let dir = base.join(&here);
        if let Some(limit) = read_u64_or_max(&dir.join("memory.max")) {
            let current = read_u64_or_max(&dir.join("memory.current")).unwrap_or(0);
            let path = format!("/{}", here.display());
            let path = if path == "/." { "/".to_string() } else { path };
            // The smallest limit on the chain is the one that binds.
            if binding.as_ref().is_none_or(|(_, l, _)| limit < *l) {
                binding = Some((path, limit, current));
            }
        }
        if !here.pop() {
            break;
        }
    }

    match binding {
        None => HostLimit::Machine,
        Some((path, limit_bytes, current_bytes)) => HostLimit::Cgroup {
            path,
            limit_bytes,
            current_bytes,
        },
    }
}

/// A cgroup byte file: a number, or `max` meaning no limit at this level.
fn read_u64_or_max(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text == "max" {
        return None;
    }
    text.parse().ok()
}
