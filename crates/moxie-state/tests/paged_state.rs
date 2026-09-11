//! Byte oracle and public-API consumers for task 0013. No attention is executed.
use std::sync::atomic::AtomicBool;

use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, PagedSequence, ROOT, SequenceState, StateKind};
use moxie_types::{Error, HostTier, Precision, Scope, StateTransactionId, Tier};

fn ledger(capacity: u64) -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, capacity + 1, 1).unwrap()]).unwrap()
}

fn geometry() -> KvGeometry {
    KvGeometry {
        layers: 2,
        kv_heads: 1,
        key_dim: 2,
        value_dim: 3,
        precision: Precision::Bf16,
        page_tokens: 3,
        max_tokens: 17,
    }
}

// A flat, separately generated logical oracle, with distinct bytes at every
// layer/position/channel and in K versus V. Layout equations are not reused.
fn encoded(g: KvGeometry, position: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let element_bytes = g.precision.bits() as usize / 8;
    (0..g.layers)
        .map(|layer| {
            let bytes = |width, salt| {
                (0..g.kv_heads * width * element_bytes)
                    .map(|channel| ((position * 37 + layer * 19 + channel * 7 + salt) % 256) as u8)
                    .collect()
            };
            (bytes(g.key_dim, 17), bytes(g.value_dim, 173))
        })
        .collect()
}

fn append(s: &mut PagedSequence, txn: StateTransactionId, position: usize) {
    let data = encoded(s.geometry(), position);
    let rows: Vec<_> = data
        .iter()
        .map(|(key, value)| KvRow { key, value })
        .collect();
    s.append(txn, position as u64, &rows, &AtomicBool::new(false))
        .unwrap();
}

fn check_rows(s: &PagedSequence, count: usize) {
    assert_eq!(s.usage().rows, count);
    assert_eq!(s.state().frontiers(ROOT).unwrap().executed, count as u64);
    for position in 0..count {
        for (layer, (key, value)) in encoded(s.geometry(), position).iter().enumerate() {
            let actual = s.row(layer, position as u64).unwrap();
            assert_eq!(actual.key, key);
            assert_eq!(actual.value, value);
        }
    }
    assert!(s.row(0, count as u64).is_err());
    assert!(s.row(s.geometry().layers, 0).is_err());
    assert!(s.row(0, u64::MAX).is_err());
}

#[test]
fn whole_and_every_chunk_width_preserve_all_bytes_across_two_geometries_and_encodings() {
    for precision in [Precision::Bf16, Precision::F16, Precision::F32] {
        for g in [
            KvGeometry {
                precision,
                ..geometry()
            },
            KvGeometry {
                layers: 3,
                kv_heads: 2,
                key_dim: 3,
                value_dim: 1,
                page_tokens: 4,
                precision,
                ..geometry()
            },
        ] {
            for chunk in 1..=g.max_tokens {
                let mut owner = ledger(1 << 20);
                let mut sequence = PagedSequence::new(&mut owner, g).unwrap();
                for start in (0..g.max_tokens).step_by(chunk) {
                    let end = (start + chunk).min(g.max_tokens);
                    // Admission of later prompt chunks participates in abort.
                    let txn = sequence.begin().unwrap();
                    sequence.append_prompt((end - start) as u64).unwrap();
                    for position in start..end {
                        append(&mut sequence, txn, position);
                    }
                    sequence.commit_prefix(txn, 0).unwrap();
                    check_rows(&sequence, end);
                }
                let usage = sequence.usage();
                let pages = g.max_tokens.div_ceil(g.page_tokens);
                let row_bytes =
                    g.layers * g.kv_heads * (g.key_dim + g.value_dim) * g.precision.bits() as usize
                        / 8;
                assert_eq!(usage.live_pages, pages);
                assert_eq!(usage.backing_bytes, pages * (g.page_tokens * row_bytes + 8));
                assert_eq!(usage.logical_kv_bytes, g.max_tokens * row_bytes);
                assert_eq!(
                    owner.committed(Scope::Host, Tier::Host(HostTier::StateSpill)),
                    usage.backing_bytes as u64
                );
                assert_eq!(
                    owner.scope_committed(Scope::Host),
                    (usage.backing_bytes + usage.control_reserve_bytes) as u64
                );
                assert!(!sequence.state().next_logits_valid(ROOT));
                sequence.close(&mut owner).unwrap();
                assert_eq!(owner.scope_committed(Scope::Host), 0);
            }
        }
    }
}

#[test]
fn every_partial_acceptance_truncates_to_replay_and_materializes_a_pending_bonus() {
    for accepted in 0..=7 {
        let mut owner = ledger(1 << 20);
        let mut sequence = PagedSequence::new(&mut owner, geometry()).unwrap();
        sequence.append_prompt(2).unwrap();
        let txn = sequence.begin().unwrap();
        append(&mut sequence, txn, 0);
        append(&mut sequence, txn, 1);
        sequence.commit_prefix(txn, 0).unwrap();
        let txn = sequence.begin().unwrap();
        for position in 2..9 {
            append(&mut sequence, txn, position);
        }
        assert_eq!(sequence.state().frontiers(ROOT).unwrap().tentative(), 7);
        assert!(sequence.rollback_to(2).is_err());
        assert!(sequence.emit(1).is_err());
        sequence.commit_prefix(txn, accepted).unwrap();
        // Task 0004 keeps materialization; the rejected suffix is explicitly
        // removed at the resolved boundary, using its existing rollback rules.
        let keep = 2 + accepted;
        sequence.rollback_to(keep).unwrap();
        check_rows(&sequence, keep as usize);
        sequence.accept(1).unwrap();
        sequence.emit(1).unwrap();
        assert_eq!(
            sequence
                .state()
                .frontiers(ROOT)
                .unwrap()
                .pending_execution(),
            1
        );
        let txn = sequence.begin().unwrap();
        append(&mut sequence, txn, keep as usize);
        sequence.commit_prefix(txn, 0).unwrap();
        check_rows(&sequence, keep as usize + 1);
        assert!(
            sequence.rollback_to(2).is_err(),
            "cannot retract emitted text"
        );
        sequence.close(&mut owner).unwrap();
    }
}

#[test]
fn abort_restores_the_prefix_lineage_and_tentative_prompt_across_page_reuse() {
    let mut owner = ledger(1 << 20);
    let mut sequence = PagedSequence::new(&mut owner, geometry()).unwrap();
    let txn = sequence.begin().unwrap();
    sequence.append_prompt(2).unwrap();
    for position in 0..2 {
        append(&mut sequence, txn, position);
    }
    sequence.commit_prefix(txn, 0).unwrap();
    let frontiers = sequence.state().frontiers(ROOT).unwrap();
    let lineage = sequence.state().lineage_at(ROOT, 2).unwrap();
    let charge = owner.scope_committed(Scope::Host);
    for count in 1..=15 {
        let txn = sequence.begin().unwrap();
        assert!(sequence.begin().is_err());
        sequence.append_prompt(count).unwrap();
        for position in 2..2 + count {
            append(&mut sequence, txn, position as usize);
        }
        sequence.abort(txn).unwrap();
        assert_eq!(sequence.state().frontiers(ROOT).unwrap(), frontiers);
        assert_eq!(sequence.state().lineage_at(ROOT, 2).unwrap(), lineage);
        assert_eq!(sequence.state().lineage_at(ROOT, 3).unwrap(), None);
        assert_eq!(owner.scope_committed(Scope::Host), charge);
        assert!(sequence.state().open_transactions().is_empty());
        check_rows(&sequence, 2);
    }
    sequence.close(&mut owner).unwrap();
}

#[test]
fn transaction_ids_cannot_resolve_another_sequences_journal() {
    let mut a = SequenceState::new([StateKind::KvPages]);
    let mut b = SequenceState::new([StateKind::KvPages]);
    let ta = a.begin(ROOT).unwrap();
    let tb = b.begin(ROOT).unwrap();
    assert_ne!(ta, tb);
    b.execute(ROOT, 1).unwrap();
    assert!(b.abort(ta).is_err());
    assert!(b.commit_prefix(ta, 1).is_err());
    assert_eq!(b.frontiers(ROOT).unwrap().executed, 1);
    assert_eq!(b.open_transactions(), [(tb, ROOT)]);
    b.abort(tb).unwrap();
    a.abort(ta).unwrap();
}

#[test]
fn foreign_resolved_and_unknown_ids_do_not_abort_a_live_paged_transaction() {
    let mut owner = ledger(1 << 20);
    let mut a = PagedSequence::new(&mut owner, geometry()).unwrap();
    let mut b = PagedSequence::new(&mut owner, geometry()).unwrap();
    let ta = a.begin().unwrap();
    let stale = b.begin().unwrap();
    b.abort(stale).unwrap();
    let tb = b.begin().unwrap();
    append(&mut b, tb, 0);
    for wrong in [ta, stale, StateTransactionId(u64::MAX)] {
        assert!(b.append(wrong, 1, &[], &AtomicBool::new(false)).is_err());
        assert!(b.abort(wrong).is_err());
        assert!(b.commit_prefix(wrong, 0).is_err());
        check_rows(&b, 1);
        assert_eq!(b.state().open_transactions(), [(tb, ROOT)]);
    }
    b.abort(tb).unwrap();
    b.close(&mut owner).unwrap();
    a.close(&mut owner).unwrap(); // abandoned transaction also releases
    assert!(owner.outstanding().is_empty());
}

#[test]
fn malformed_rows_positions_and_cancellation_abort_all_prior_appends() {
    for fault in 0..6 {
        let mut owner = ledger(1 << 20);
        let mut sequence = PagedSequence::new(&mut owner, geometry()).unwrap();
        let txn = sequence.begin().unwrap();
        append(&mut sequence, txn, 0);
        sequence.commit_prefix(txn, 1).unwrap();
        let txn = sequence.begin().unwrap();
        append(&mut sequence, txn, 1);
        let data = encoded(geometry(), 2);
        let mut rows: Vec<_> = data
            .iter()
            .map(|(key, value)| KvRow { key, value })
            .collect();
        match fault {
            0 => rows[0].key = &[],
            1 => rows[1].value = &[],
            2 => {
                rows.pop();
            }
            _ => {}
        }
        let position = match fault {
            3 => 0,
            4 => u64::MAX,
            _ => 2,
        };
        let error = sequence
            .append(txn, position, &rows, &AtomicBool::new(fault == 5))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            if fault == 5 {
                "cancelled"
            } else {
                "invalid_request"
            }
        );
        check_rows(&sequence, 1);
        assert!(sequence.state().open_transactions().is_empty());
        let retry = sequence.begin().unwrap();
        append(&mut sequence, retry, 1);
        sequence.commit_prefix(retry, 1).unwrap();
        sequence.close(&mut owner).unwrap();
    }
}

#[test]
fn full_context_failed_commit_and_unsupported_fork_never_change_capacity_or_precision() {
    let g = KvGeometry {
        max_tokens: 4,
        ..geometry()
    };
    let mut owner = ledger(1 << 20);
    let mut sequence = PagedSequence::new(&mut owner, g).unwrap();
    let txn = sequence.begin().unwrap();
    for position in 0..4 {
        append(&mut sequence, txn, position);
    }
    assert!(sequence.commit_prefix(txn, 5).is_err());
    assert_eq!(sequence.state().open_transactions(), [(txn, ROOT)]);
    sequence.commit_prefix(txn, 4).unwrap();
    let txn = sequence.begin().unwrap();
    let data = encoded(g, 4);
    let rows: Vec<_> = data
        .iter()
        .map(|(key, value)| KvRow { key, value })
        .collect();
    assert!(
        sequence
            .append(txn, 4, &rows, &AtomicBool::new(false))
            .is_err()
    );
    check_rows(&sequence, 4);
    assert!(sequence.append_prompt(1).is_err());
    assert!(sequence.accept(u64::MAX).is_err());
    assert!(matches!(sequence.fork(4), Err(Error::Unsupported { .. })));
    assert_eq!(sequence.geometry(), g);
    sequence.close(&mut owner).unwrap();
}

#[test]
fn invalid_geometry_and_insufficient_budget_fail_before_retaining_any_charge() {
    let mut owner = ledger(1 << 20);
    let g = geometry();
    for bad in [
        KvGeometry { layers: 0, ..g },
        KvGeometry { kv_heads: 0, ..g },
        KvGeometry { key_dim: 0, ..g },
        KvGeometry { value_dim: 0, ..g },
        KvGeometry {
            page_tokens: 0,
            ..g
        },
        KvGeometry { max_tokens: 0, ..g },
        KvGeometry {
            precision: Precision::Int4,
            ..g
        },
        KvGeometry {
            precision: Precision::Int8,
            ..g
        },
        KvGeometry {
            layers: usize::MAX,
            ..g
        },
        KvGeometry {
            page_tokens: usize::MAX,
            ..g
        },
        KvGeometry {
            max_tokens: usize::MAX,
            ..g
        },
    ] {
        assert!(PagedSequence::new(&mut owner, bad).is_err());
        assert!(owner.outstanding().is_empty());
    }
    let sequence = PagedSequence::new(&mut owner, g).unwrap();
    let u = sequence.usage();
    let cost = (u.backing_bytes + u.control_reserve_bytes) as u64;
    sequence.close(&mut owner).unwrap();
    let mut exact = ledger(cost);
    let sequence = PagedSequence::new(&mut exact, g).unwrap();
    assert!(PagedSequence::new(&mut exact, g).is_err());
    sequence.close(&mut exact).unwrap();
    let mut short = ledger(cost - 1);
    assert!(matches!(
        PagedSequence::new(&mut short, g),
        Err(Error::CapacityExceeded { .. })
    ));
    assert!(short.outstanding().is_empty());
}

#[test]
fn wrong_ledger_close_returns_the_live_owner_and_next_sequence_can_use_the_budget() {
    let mut owner = ledger(1 << 20);
    let mut wrong = ledger(1 << 20);
    let mut sequence = PagedSequence::new(&mut owner, geometry()).unwrap();
    let txn = sequence.begin().unwrap();
    append(&mut sequence, txn, 0);
    let refused = sequence.close(&mut wrong).unwrap_err();
    assert_eq!(refused.error.kind(), "invalid_request");
    let sequence = refused.sequence;
    check_rows(&sequence, 1);
    sequence.close(&mut owner).unwrap();
    assert!(owner.outstanding().is_empty());
    let sequence = PagedSequence::new(&mut owner, geometry()).unwrap();
    check_rows(&sequence, 0);
    sequence.close(&mut owner).unwrap();
}
