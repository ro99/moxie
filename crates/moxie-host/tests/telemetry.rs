//! Acceptance tests for task 0008's host sensor.
//!
//! Every case reads a committed fixture tree, never `/proc`. That is the whole
//! point of the injectable root: the benchmark machine has no cgroup limit at
//! any level, so the limited paths would otherwise be reasoned about instead of
//! run.

use std::path::PathBuf;

use moxie_types::HostLimit;

const KB: u64 = 1024;
const GIB: u64 = 1024 * 1024 * 1024;

/// This machine's real figures, as read on 2026-09-09.
const MEM_TOTAL: u64 = 264_005_080 * KB;
const MEM_AVAILABLE: u64 = 256_824_960 * KB;
const MEM_FREE: u64 = 251_609_192 * KB;

fn tree(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn read(name: &str) -> moxie_types::MeasuredHost {
    moxie_host::read_under(&tree(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn the_budget_input_is_available_and_not_free() {
    // They differ by about 5 GiB here and by far more on a machine with a warm
    // cache. `MemFree` is what is untouched; `MemAvailable` is what a new
    // allocation can actually obtain, reclaimable page cache included.
    let host = read("machine");
    assert_eq!(host.total_bytes, MEM_TOTAL);
    assert_eq!(host.available_bytes, MEM_AVAILABLE);
    assert_eq!(host.free_bytes, MEM_FREE);
    assert!(
        host.available_bytes > host.free_bytes,
        "available {} is not above free {}",
        host.available_bytes,
        host.free_bytes
    );
    // Every level reads `max`, so the machine's own figures govern.
    assert_eq!(host.limit, HostLimit::Machine);
    assert_eq!(host.machine_total_bytes, host.total_bytes);
    assert_eq!(host.machine_available_bytes, host.available_bytes);
}

#[test]
fn a_leaf_cgroup_limit_binds_and_max_ancestors_are_skipped() {
    let host = read("cgroup-leaf-limit");
    match &host.limit {
        HostLimit::Cgroup {
            path,
            limit_bytes,
            current_bytes,
        } => {
            assert_eq!(path, "/user.slice/session.scope");
            assert_eq!(*limit_bytes, 8 * GIB);
            assert_eq!(*current_bytes, GIB);
        }
        other => panic!("expected a cgroup limit, got {other:?}"),
    }
    assert_eq!(host.total_bytes, 8 * GIB, "the limit caps the total");
    assert_eq!(host.available_bytes, 7 * GIB, "limit less what it is using");
    // The machine view survives for the report, so a reader can see the cost.
    assert_eq!(host.machine_total_bytes, MEM_TOTAL);
    assert_eq!(host.machine_available_bytes, MEM_AVAILABLE);
}

#[test]
fn the_smallest_limit_on_the_chain_binds_even_when_it_is_an_ancestor() {
    // The leaf allows 16 GiB and the slice above it allows 4 GiB. Reading only
    // the leaf would over-promise by four times.
    let host = read("cgroup-ancestor-limit");
    match &host.limit {
        HostLimit::Cgroup {
            path,
            limit_bytes,
            current_bytes,
        } => {
            assert_eq!(path, "/user.slice");
            assert_eq!(*limit_bytes, 4 * GIB);
            assert_eq!(*current_bytes, 2 * GIB);
        }
        other => panic!("expected the ancestor's limit, got {other:?}"),
    }
    assert_eq!(host.total_bytes, 4 * GIB);
    assert_eq!(host.available_bytes, 2 * GIB);
}

#[test]
fn the_smallest_limit_binds_when_it_is_the_leaf_too() {
    // The mirror of the case above, and it is not redundant: walking the chain
    // from leaf to root and keeping the *last* limit found agrees with keeping
    // the smallest whenever the ancestor happens to be tighter. Only a fixture
    // where the leaf is tighter tells the two rules apart -- a bite check on the
    // ancestor case alone failed nothing.
    let host = read("cgroup-leaf-tighter");
    match &host.limit {
        HostLimit::Cgroup {
            path, limit_bytes, ..
        } => {
            assert_eq!(path, "/user.slice/session.scope");
            assert_eq!(*limit_bytes, 4 * GIB, "the 16 GiB ancestor must not win");
        }
        other => panic!("expected the leaf's limit, got {other:?}"),
    }
    assert_eq!(host.total_bytes, 4 * GIB);
    assert_eq!(host.available_bytes, 3 * GIB);
}

#[test]
fn an_absent_hierarchy_is_the_machine_view_and_not_an_error() {
    // A cgroup v1 machine, or none at all: the machine's figures govern, and the
    // descriptor says which view was used rather than leaving it inferable.
    for name in ["no-cgroup", "cgroup-v1"] {
        let host = read(name);
        assert_eq!(host.limit, HostLimit::Machine, "{name}");
        assert_eq!(host.total_bytes, MEM_TOTAL, "{name}");
        assert_eq!(host.available_bytes, MEM_AVAILABLE, "{name}");
    }
}

#[test]
fn swap_is_reported_and_changes_no_budget_figure() {
    // Document 03: do not rely on uncontrolled swap as an invisible fourth
    // execution tier. 512 GiB of it must move nothing.
    let plain = read("machine");
    let swapped = read("big-swap");
    assert_eq!(swapped.swap_total_bytes, 536_870_912 * KB);
    assert!(swapped.swap_total_bytes > swapped.total_bytes);
    assert_eq!(swapped.total_bytes, plain.total_bytes);
    assert_eq!(swapped.available_bytes, plain.available_bytes);
}

#[test]
fn page_cache_is_reported_because_pressure_is_visible_before_it_is_managed() {
    let host = read("machine");
    assert_eq!(host.cached_bytes, 6_152_148 * KB);
    assert_eq!(host.buffers_bytes, 415_680 * KB);
}

#[test]
fn a_missing_mem_available_is_refused_rather_than_computed() {
    // The substitute formula is a reclaim implementation detail that has
    // changed. Guessing it would be a private model of the kernel presented as
    // a measurement.
    let e = moxie_host::read_under(&tree("no-mem-available")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("MemAvailable"), "{e}");
    assert!(e.to_string().contains("refuses rather than guess"), "{e}");
}

#[test]
fn a_malformed_field_is_refused_and_the_error_names_it() {
    for (name, needle) in [
        ("bad-unit", "MemTotal"),
        ("non-numeric", "MemAvailable"),
        ("truncated", "SwapTotal"),
    ] {
        let e = moxie_host::read_under(&tree(name))
            .err()
            .unwrap_or_else(|| panic!("{name} must be refused"));
        assert_eq!(e.kind(), "invalid_request", "{name}");
        assert!(e.to_string().contains(needle), "{name}: {e}");
    }
}

#[test]
fn an_unreadable_meminfo_is_refused() {
    let e = moxie_host::read_under(&tree("does-not-exist")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("meminfo"), "{e}");
}

#[test]
fn this_machine_answers_and_its_answer_is_coherent() {
    // The one case that touches the real `/proc`. It asserts shape, not values:
    // the numbers move between runs, and a test that pinned them would be
    // pinning the weather.
    let host = moxie_host::read().expect("this machine has a readable /proc/meminfo");
    assert!(host.total_bytes > 0);
    assert!(host.available_bytes <= host.total_bytes);
    assert!(host.free_bytes <= host.total_bytes);
    assert!(host.machine_available_bytes <= host.machine_total_bytes);
    assert!(host.total_bytes <= host.machine_total_bytes);
}
