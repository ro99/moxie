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

    let limit = cgroup_limit(root)?;
    let (total_bytes, available_bytes) = match &limit {
        HostLimit::Machine => (machine_total_bytes, machine_available_bytes),
        HostLimit::Cgroup {
            limit_bytes,
            avail_limit_bytes,
            avail_current_bytes,
            ..
        } => (
            machine_total_bytes.min(*limit_bytes),
            machine_available_bytes.min(avail_limit_bytes.saturating_sub(*avail_current_bytes)),
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
/// The binding cgroup v2 limits, or [`HostLimit::Machine`].
///
/// Total and headroom minimise independently over the process's own cgroup and
/// every ancestor: the smallest `memory.max` caps the total, and the smallest
/// saturating `memory.max - memory.current` caps new allocation. Either can
/// bind at a different level -- a sibling scope holding most of an ancestor
/// slice leaves the leaf with the smaller limit and the ancestor with almost
/// no room -- so both minima are tracked and both binding paths recorded.
///
/// Absence of a hierarchy is not an error: a cgroup v1 machine (no `0::` line)
/// or an unmounted hierarchy (no membership file, `NotFound`) means the
/// machine's own figures govern. An incomplete hierarchy is an error: an
/// unreadable membership file for any other reason, a finite `memory.max`
/// with no readable `memory.current`, or a `memory.max` that is neither a
/// number nor `max`, refuses rather than guessing spendable capacity in the
/// optimistic direction. A missing file at a level (`NotFound`) means no
/// limit there; any other I/O failure or a malformed value is a refused
/// measurement.
fn cgroup_limit(root: &Path) -> Result<HostLimit> {
    // Absence is the machine view: no hierarchy mounted, or nothing to read
    // under this root. Any other failure refuses: inability to discover a
    // limit is not evidence that none applies, and falling back to the
    // machine view would present up to hundreds of GiB that the kernel will
    // refuse as spendable.
    let text = match std::fs::read_to_string(root.join("proc/self/cgroup")) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HostLimit::Machine),
        Err(e) => {
            return Err(invalid(
                "cgroup",
                format!(
                    "proc/self/cgroup unreadable ({}): {e}",
                    root.join("proc/self/cgroup").display()
                ),
            ));
        }
    };
    // cgroup v2 is the single `0::<path>` line. A v1-only machine has none.
    let Some(rel) = text
        .lines()
        .find_map(|l| l.strip_prefix("0::").map(str::trim))
    else {
        return Ok(HostLimit::Machine);
    };

    let base = root.join("sys/fs/cgroup");
    let mut here = PathBuf::from(rel.trim_start_matches('/'));
    let mut total_binding: Option<(String, u64, u64)> = None;
    let mut headroom_binding: Option<(String, u64, u64, u64)> = None;
    loop {
        let dir = base.join(&here);
        let cgroup_path = {
            let path = format!("/{}", here.display());
            if path == "/." { "/".to_string() } else { path }
        };
        if let Some(limit) = read_limit(&dir.join("memory.max"), &cgroup_path)? {
            let current = read_current(&dir.join("memory.current"), &cgroup_path)?;
            let headroom = limit.saturating_sub(current);
            // Strictly smaller wins, so ties keep the leaf-most level: the walk
            // runs leaf to root.
            if total_binding.as_ref().is_none_or(|(_, l, _)| limit < *l) {
                total_binding = Some((cgroup_path.clone(), limit, current));
            }
            if headroom_binding
                .as_ref()
                .is_none_or(|(_, _, _, h)| headroom < *h)
            {
                headroom_binding = Some((cgroup_path, limit, current, headroom));
            }
        }
        if !here.pop() {
            break;
        }
    }

    match (total_binding, headroom_binding) {
        (None, None) => Ok(HostLimit::Machine),
        (
            Some((path, limit_bytes, current_bytes)),
            Some((avail_path, avail_limit_bytes, avail_current_bytes, _)),
        ) => Ok(HostLimit::Cgroup {
            path,
            limit_bytes,
            current_bytes,
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
        }),
        // Unreachable: every finite limit yields a headroom candidate at the
        // same level, so both are set together. Fail closed if it ever happens.
        _ => Err(invalid(
            "cgroup",
            "cgroup limits disagree: a total bound without a headroom bound".to_string(),
        )),
    }
}

/// A `memory.max` file: a number (a finite limit at this level), or `max`
/// (no limit here, `None`). A missing file is also no limit here; anything
/// else unreadable or malformed refuses the measurement rather than silently
/// treating a limit as absent.
fn read_limit(path: &Path, cgroup_path: &str) -> Result<Option<u64>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(invalid(
                "cgroup",
                format!(
                    "{cgroup_path} memory.max unreadable ({}): {e}",
                    path.display()
                ),
            ));
        }
    };
    let text = text.trim();
    if text == "max" {
        return Ok(None);
    }
    match text.parse::<u64>() {
        Ok(n) => Ok(Some(n)),
        Err(e) => Err(invalid(
            "cgroup",
            format!("{cgroup_path} memory.max is neither a byte count nor `max` ({text:?}): {e}"),
        )),
    }
}

/// A `memory.current` file where a finite limit applies. Always required:
/// treating unreadable usage as zero would report the whole limit as
/// available, the same optimistic guess this crate refuses for `MemAvailable`.
fn read_current(path: &Path, cgroup_path: &str) -> Result<u64> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        invalid(
            "cgroup",
            format!(
                "{cgroup_path} memory.current unreadable with a finite memory.max ({}): {e}",
                path.display()
            ),
        )
    })?;
    text.trim().parse::<u64>().map_err(|e| {
        invalid(
            "cgroup",
            format!(
                "{cgroup_path} memory.current is not a byte count ({:?}): {e}",
                text.trim()
            ),
        )
    })
}
