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
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
        } => {
            assert_eq!(path, "/user.slice/session.scope");
            assert_eq!(*limit_bytes, 8 * GIB);
            assert_eq!(*current_bytes, GIB);
            assert_eq!(avail_path, "/user.slice/session.scope");
            assert_eq!(*avail_limit_bytes, 8 * GIB);
            assert_eq!(*avail_current_bytes, GIB);
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
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
        } => {
            assert_eq!(path, "/user.slice");
            assert_eq!(*limit_bytes, 4 * GIB);
            assert_eq!(*current_bytes, 2 * GIB);
            assert_eq!(avail_path, "/user.slice");
            assert_eq!(*avail_limit_bytes, 4 * GIB);
            assert_eq!(*avail_current_bytes, 2 * GIB);
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
            path,
            limit_bytes,
            current_bytes,
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
        } => {
            assert_eq!(path, "/user.slice/session.scope");
            assert_eq!(*limit_bytes, 4 * GIB, "the 16 GiB ancestor must not win");
            assert_eq!(*current_bytes, GIB);
            assert_eq!(avail_path, "/user.slice/session.scope");
            assert_eq!(*avail_limit_bytes, 4 * GIB);
            assert_eq!(*avail_current_bytes, GIB);
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

fn assert_coherent(host: &moxie_types::MeasuredHost) {
    // Effective figures are capped by the cgroup chain; machine-wide figures
    // are not. `free_bytes` is `MemFree` for the whole machine, so it is only
    // comparable against `machine_total_bytes`, never against a capped total.
    assert!(host.total_bytes > 0);
    assert!(host.available_bytes <= host.total_bytes);
    assert!(host.free_bytes <= host.machine_total_bytes);
    assert!(host.machine_available_bytes <= host.machine_total_bytes);
    assert!(host.total_bytes <= host.machine_total_bytes);
    assert!(host.available_bytes <= host.machine_available_bytes);
}

#[test]
fn this_machine_answers_and_its_answer_is_coherent() {
    // The one case that touches the real `/proc`. It asserts shape, not values:
    // the numbers move between runs, and a test that pinned them would be
    // pinning the weather.
    let host = moxie_host::read().expect("this machine has a readable /proc/meminfo");
    assert_coherent(&host);
}

#[test]
fn coherence_holds_under_a_cgroup_cap() {
    // The 8 GiB leaf cap leaves ~240 GiB of machine-wide free memory: the old
    // `free_bytes <= total_bytes` comparison failed here despite correct
    // telemetry, because the two quantities live in different scopes.
    let host = read("cgroup-leaf-limit");
    assert_coherent(&host);
    assert!(
        host.free_bytes > host.total_bytes,
        "machine-wide free {} exceeds the capped total {}",
        host.free_bytes,
        host.total_bytes
    );
}

#[test]
fn headroom_is_minimised_over_the_chain_and_not_read_off_the_tightest_limit() {
    // Review finding P1. Two different quantities are minimised over the chain,
    // and they can bind at different levels. Here the leaf has the smaller
    // *limit* (20 GiB against 100 GiB) while the slice above it has almost no
    // room left, because a sibling scope holds 90 GiB of its 100 GiB.
    //
    // A cgroup's `memory.current` includes its descendants', so this is an
    // ordinary shape on any machine with more than one scope under a slice --
    // not a contrived one.
    let host = read("cgroup-sibling-pressure");
    match &host.limit {
        HostLimit::Cgroup {
            path,
            limit_bytes,
            avail_path,
            avail_limit_bytes,
            avail_current_bytes,
            ..
        } => {
            assert_eq!(
                path, "/user.slice/session.scope",
                "smallest limit binds total"
            );
            assert_eq!(*limit_bytes, 20 * GIB);
            assert_eq!(
                avail_path, "/user.slice",
                "tightest headroom binds available"
            );
            assert_eq!(*avail_limit_bytes, 100 * GIB);
            assert_eq!(*avail_current_bytes, 106_300_440_576);
        }
        other => panic!("expected a cgroup limit, got {other:?}"),
    }
    assert_eq!(
        host.total_bytes,
        20 * GIB,
        "the smallest limit caps the total"
    );
    assert_eq!(
        host.available_bytes, GIB,
        "the ancestor has 1 GiB left; reading headroom off the tightest limit \
         would promise 11 GiB and the eleventh would die in the kernel"
    );
}

#[test]
fn an_unreadable_membership_file_is_refused_rather_than_unlimited() {
    // `proc/self/cgroup` as a directory fails with an I/O error other than
    // NotFound. Falling back to the machine view here would report ~245 GiB
    // available against an 8 GiB cap: inability to discover a limit is not
    // evidence that none applies.
    let e = moxie_host::read_under(&tree("cgroup-membership-unreadable")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("self/cgroup"), "{e}");
}

#[test]
fn a_non_utf8_membership_file_is_refused_rather_than_unlimited() {
    // `read_to_string` fails with InvalidData on non-UTF-8 bytes. Same
    // direction, same refusal: the hierarchy cannot be walked, so no bound is
    // known.
    let e = moxie_host::read_under(&tree("cgroup-membership-invalid-utf8")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("self/cgroup"), "{e}");
}

#[test]
fn a_level_with_a_limit_but_no_readable_usage_is_refused() {
    // Review finding P3. Treating an unreadable `memory.current` as zero is the
    // same guess this crate refuses for `MemAvailable`, in the same optimistic
    // direction: it would report the whole 8 GiB limit as available.
    let e = moxie_host::read_under(&tree("cgroup-current-missing")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("memory.current"), "{e}");
}

#[test]
fn a_malformed_limit_is_refused_rather_than_treated_as_max() {
    // A `memory.max` that is neither a number nor `max` must not collapse into
    // "no limit": that spelling would silently admit against the machine view.
    let e = moxie_host::read_under(&tree("cgroup-bad-limit")).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("memory.max"), "{e}");
}
