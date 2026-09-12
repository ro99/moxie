//! Task 0017: per-layer retention, ring reclamation and the two refusals.
//!
//! Storage only. No attention runs here and nothing is model support; the
//! numerical parity of a windowed graph against a full-retention one is the
//! interpreter's test, not this one.
use std::sync::atomic::AtomicBool;

use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, LayerKv, PagedSequence, ROOT, Retention};
use moxie_types::{Error, Precision, Scope, StateTransactionId};

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 24, 1 << 16).unwrap()]).unwrap()
}

/// Two layers that agree about nothing: a wide full-retention layer and a
/// narrow windowed one. Gemma 4's shape in miniature -- its global layers are
/// 4 heads of 512 and full causal, its sliding layers 16 of 256 and windowed.
fn mixed(window: usize, tentative: usize, page_tokens: usize, max_tokens: usize) -> KvGeometry {
    KvGeometry {
        layers: vec![
            LayerKv {
                kv_heads: 2,
                key_dim: 3,
                value_dim: 3,
                retention: Retention::All,
            },
            LayerKv {
                kv_heads: 1,
                key_dim: 2,
                value_dim: 1,
                retention: Retention::Window { window },
            },
        ],
        precision: Precision::Bf16,
        page_tokens,
        max_tokens,
        tentative_rows: tentative,
    }
}

/// Distinct bytes per layer and position, at each layer's own width.
fn encoded(g: &KvGeometry, position: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let element = g.precision.bits() as usize / 8;
    g.layers
        .iter()
        .enumerate()
        .map(|(layer, l)| {
            let bytes = |width, salt| {
                (0..l.kv_heads * width * element)
                    .map(|c| ((position * 41 + layer * 23 + c * 5 + salt) % 256) as u8)
                    .collect()
            };
            (bytes(l.key_dim, 11), bytes(l.value_dim, 191))
        })
        .collect()
}

fn append(
    s: &mut PagedSequence,
    txn: StateTransactionId,
    position: usize,
) -> moxie_types::Result<()> {
    let data = encoded(&s.geometry().clone(), position);
    let rows: Vec<_> = data
        .iter()
        .map(|(key, value)| KvRow { key, value })
        .collect();
    s.append(txn, position as u64, &rows, &AtomicBool::new(false))
}

/// Every readable row holds its own bytes, and every reclaimed one is refused
/// as reclaimed rather than served stale or reported missing.
fn check(s: &PagedSequence, rows: usize) {
    let g = s.geometry().clone();
    for (layer, l) in g.layers.iter().enumerate() {
        let retained = s.retained_range(layer).unwrap();
        assert_eq!(retained.end, rows as u64);
        match l.retention {
            Retention::All => assert_eq!(retained.start, 0),
            // Exactly `rows - window`, pinned here rather than inferred: the
            // interpreter reads a layer's history *before* appending a chunk's
            // own rows, so a store one row short would still answer every
            // query correctly and the numerical parity test would not notice.
            // This is the assertion that holds the frontier in place.
            Retention::Window { window } => {
                assert_eq!(retained.start, rows.saturating_sub(window) as u64)
            }
        }
        for position in 0..rows {
            let expected = &encoded(&g, position)[layer];
            match s.row(layer, position as u64) {
                Ok(row) => {
                    assert!(retained.contains(&(position as u64)));
                    assert_eq!(row.key, expected.0, "layer {layer} position {position} key");
                    assert_eq!(
                        row.value, expected.1,
                        "layer {layer} position {position} value"
                    );
                }
                Err(Error::Reclaimed {
                    layer: l,
                    position: p,
                    retained_from,
                }) => {
                    assert!(!retained.contains(&(position as u64)));
                    assert_eq!(
                        (l as usize, p, retained_from),
                        (layer, position as u64, retained.start)
                    );
                }
                Err(other) => panic!("unexpected {other} at layer {layer} position {position}"),
            }
        }
    }
}

#[test]
fn a_windowed_layer_serves_its_window_and_refuses_below_it_across_every_wrap() {
    // Several page sizes, so the wrap lands on, before and after a page edge.
    for page_tokens in [1, 2, 3, 4, 5] {
        for window in [1, 2, 5] {
            let g = mixed(window, 2, page_tokens, 40);
            let mut owner = ledger();
            let mut s = PagedSequence::new(&mut owner, g).unwrap();
            for start in (0..40).step_by(2) {
                let txn = s.begin().unwrap();
                s.append_prompt(2).unwrap();
                append(&mut s, txn, start).unwrap();
                append(&mut s, txn, start + 1).unwrap();
                s.commit_prefix(txn, 0).unwrap();
                check(&s, start + 2);
            }
            s.close(&mut owner).unwrap();
            assert!(owner.outstanding().is_empty());
        }
    }
}

#[test]
fn a_full_retention_layer_beside_a_windowed_one_never_reclaims() {
    let g = mixed(3, 2, 2, 32);
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    for position in 0..32 {
        let txn = s.begin().unwrap();
        s.append_prompt(1).unwrap();
        append(&mut s, txn, position).unwrap();
        s.commit_prefix(txn, 0).unwrap();
    }
    assert_eq!(s.retained_range(0).unwrap(), 0..32);
    assert_eq!(s.retained_range(1).unwrap(), 29..32);
    // Layer 0 still answers position zero after the ring of layer 1 wrapped
    // ten times over.
    assert_eq!(
        s.row(0, 0).unwrap().key,
        encoded(&s.geometry().clone(), 0)[0].0
    );
    check(&s, 32);
    s.close(&mut owner).unwrap();
}

#[test]
fn a_transaction_longer_than_the_admitted_headroom_is_refused_before_it_writes() {
    let tentative = 3;
    let g = mixed(4, tentative, 2, 64);
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    let txn = s.begin().unwrap();
    s.append_prompt(tentative as u64 + 1).unwrap();
    for position in 0..tentative {
        append(&mut s, txn, position).unwrap();
    }
    // The row past the headroom is refused, and refused by name.
    let refused = append(&mut s, txn, tentative).unwrap_err();
    assert!(
        matches!(&refused, Error::InvalidRequest { field, .. } if *field == "tentative_rows"),
        "expected a tentative_rows refusal, got {refused}"
    );
    // The refusal aborted the transaction, as every append failure does, and
    // left nothing behind.
    assert!(s.state().open_transactions().is_empty());
    assert_eq!(s.usage().rows, 0);
    // And the sequence is still usable: commit in headroom-sized pieces.
    for start in (0..12).step_by(tentative) {
        let txn = s.begin().unwrap();
        s.append_prompt(tentative as u64).unwrap();
        for position in start..start + tentative {
            append(&mut s, txn, position).unwrap();
        }
        s.commit_prefix(txn, 0).unwrap();
    }
    check(&s, 12);
    s.close(&mut owner).unwrap();
}

/// Full retention is always exactly restorable, so the bound does not apply to
/// it -- a sequence that cannot reclaim keeps the freedom it had before.
#[test]
fn a_sequence_that_cannot_reclaim_is_not_bound_by_the_headroom() {
    let mut g = KvGeometry::uniform(2, 1, 2, 1, Precision::Bf16, 2, 64);
    g.tentative_rows = 1;
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    let txn = s.begin().unwrap();
    s.append_prompt(20).unwrap();
    for position in 0..20 {
        append(&mut s, txn, position).unwrap();
    }
    s.commit_prefix(txn, 0).unwrap();
    assert_eq!(s.usage().rows, 20);
    s.close(&mut owner).unwrap();
}

/// The headroom's whole purpose: with the ring full, aborting the longest legal
/// transaction restores every row the window can see.
#[test]
fn abort_with_the_ring_full_restores_every_readable_row() {
    for page_tokens in [1, 2, 3] {
        for tentative in [1, 2, 3] {
            let window = 4;
            let g = mixed(window, tentative, page_tokens, 64);
            let mut owner = ledger();
            let mut s = PagedSequence::new(&mut owner, g).unwrap();
            // Fill well past the ring so reclamation is actually happening.
            for start in (0..24).step_by(tentative) {
                let txn = s.begin().unwrap();
                s.append_prompt(tentative as u64).unwrap();
                for position in start..start + tentative {
                    append(&mut s, txn, position).unwrap();
                }
                s.commit_prefix(txn, 0).unwrap();
            }
            check(&s, 24);
            let frontiers = s.state().frontiers(ROOT).unwrap();
            let lineage = s.state().lineage_at(ROOT, 24).unwrap();

            let txn = s.begin().unwrap();
            s.append_prompt(tentative as u64).unwrap();
            for position in 24..24 + tentative {
                append(&mut s, txn, position).unwrap();
            }
            s.abort(txn).unwrap();

            assert_eq!(s.state().frontiers(ROOT).unwrap(), frontiers);
            assert_eq!(s.state().lineage_at(ROOT, 24).unwrap(), lineage);
            // Every readable byte is back, on both layers.
            check(&s, 24);
            s.close(&mut owner).unwrap();
        }
    }
}

/// The negative control for the test above. Widening the transaction past the
/// admitted headroom -- which the append guard is what prevents -- destroys
/// rows the window still needs, and the check notices.
#[test]
fn without_the_headroom_bound_an_abort_would_lose_readable_rows() {
    let window = 4;
    // Headroom of 1, but eight rows appended in one transaction. The guard
    // refuses this, so reach the same physical situation the long way: commit
    // each row, then roll back past what the ring kept.
    let g = mixed(window, 1, 2, 64);
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    // One prompt token, then twenty-three accepted completion tokens: a
    // rollback target has to be an accepted position outside the prompt.
    let txn = s.begin().unwrap();
    s.append_prompt(1).unwrap();
    append(&mut s, txn, 0).unwrap();
    s.commit_prefix(txn, 0).unwrap();
    for position in 1..24 {
        let txn = s.begin().unwrap();
        append(&mut s, txn, position).unwrap();
        s.commit_prefix(txn, 1).unwrap();
    }
    // Rolling back eight rows would need rows 12..16 on the windowed layer,
    // and the ring overwrote them at row 21. Refused, not served.
    let refused = s.rollback_to(16).unwrap_err();
    assert!(
        matches!(&refused, Error::Reclaimed { layer: 1, .. }),
        "expected the windowed layer to refuse, got {refused}"
    );
    // Nothing moved.
    assert_eq!(s.usage().rows, 24);
    check(&s, 24);
    s.close(&mut owner).unwrap();
}

#[test]
fn a_rollback_whose_window_survives_is_served_and_one_that_does_not_is_refused() {
    let window = 4;
    let g = mixed(window, 4, 2, 64);
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    let txn = s.begin().unwrap();
    s.append_prompt(1).unwrap();
    append(&mut s, txn, 0).unwrap();
    s.commit_prefix(txn, 0).unwrap();
    for position in 1..20 {
        let txn = s.begin().unwrap();
        append(&mut s, txn, position).unwrap();
        s.commit_prefix(txn, 1).unwrap();
    }
    // capacity is ceil((4 + 4)/2)*2 = 8, high_water 20, so rows below 12 are
    // gone. A target of 16 needs rows from 12: exactly reachable.
    s.rollback_to(16).unwrap();
    assert_eq!(s.usage().rows, 16);
    check(&s, 16);
    // One row further back needs row 11, which is not.
    let refused = s.rollback_to(15).unwrap_err();
    assert!(matches!(
        refused,
        Error::Reclaimed {
            layer: 1,
            position: 11,
            retained_from: 12
        }
    ));
    assert_eq!(s.usage().rows, 16);
    // The refusal is about reclamation, not about the frontier: a full
    // retention sequence rolls to the same prefix happily.
    let mut plain = PagedSequence::new(
        &mut owner,
        KvGeometry::uniform(2, 1, 2, 1, Precision::Bf16, 2, 64),
    )
    .unwrap();
    let txn = plain.begin().unwrap();
    plain.append_prompt(1).unwrap();
    append(&mut plain, txn, 0).unwrap();
    plain.commit_prefix(txn, 0).unwrap();
    for position in 1..20 {
        let txn = plain.begin().unwrap();
        append(&mut plain, txn, position).unwrap();
        plain.commit_prefix(txn, 1).unwrap();
    }
    plain.rollback_to(15).unwrap();
    assert_eq!(plain.usage().rows, 15);
    plain.close(&mut owner).unwrap();
    s.close(&mut owner).unwrap();
}

#[test]
fn a_window_at_or_above_the_context_retains_everything_and_costs_the_same() {
    let mut owner = ledger();
    let full = KvGeometry::uniform(1, 1, 2, 1, Precision::Bf16, 4, 16);
    let wide = KvGeometry {
        layers: vec![LayerKv {
            kv_heads: 1,
            key_dim: 2,
            value_dim: 1,
            retention: Retention::Window { window: 99 },
        }],
        ..full.clone()
    };
    let a = PagedSequence::new(&mut owner, full).unwrap();
    let b = PagedSequence::new(&mut owner, wide).unwrap();
    assert_eq!(a.usage().backing_bytes, b.usage().backing_bytes);
    assert_eq!(b.retained_range(0).unwrap(), 0..0);
    a.close(&mut owner).unwrap();
    b.close(&mut owner).unwrap();
}

#[test]
fn rows_of_the_wrong_layers_width_are_refused() {
    let g = mixed(4, 2, 2, 16);
    let mut owner = ledger();
    let mut s = PagedSequence::new(&mut owner, g).unwrap();
    let txn = s.begin().unwrap();
    s.append_prompt(1).unwrap();
    // Layer 0 is 2 heads of 3; layer 1 is 1 head of 2/1. Swapping the two
    // rows keeps the total byte count wrong for both.
    let wide = vec![0u8; 2 * 3 * 2];
    let narrow_k = vec![0u8; 2 * 2];
    let narrow_v = vec![0u8; 2];
    let swapped = [
        KvRow {
            key: &narrow_k,
            value: &narrow_v,
        },
        KvRow {
            key: &wide,
            value: &wide,
        },
    ];
    assert!(s.append(txn, 0, &swapped, &AtomicBool::new(false)).is_err());
    assert!(s.state().open_transactions().is_empty());
    s.close(&mut owner).unwrap();
}
