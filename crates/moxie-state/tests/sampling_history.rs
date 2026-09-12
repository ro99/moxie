use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, PagedSequence, ROOT, RestoreCapability, StateKind};
use moxie_types::{Precision, Scope, StateTransactionId};
use std::sync::atomic::AtomicBool;

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 24, 1024).unwrap()]).unwrap()
}
fn geometry(v: usize) -> KvGeometry {
    KvGeometry::uniform(2, 1, v, 1, Precision::Bf16, if v == 3 { 2 } else { 3 }, 64)
}
fn append(s: &mut PagedSequence, txn: StateTransactionId, p: u64, v: usize) {
    let key = vec![p as u8; v * 2];
    let val = (p as u16).to_le_bytes();
    s.append(
        txn,
        p,
        &[KvRow {
            key: &key,
            value: &val,
        }; 2],
        &AtomicBool::new(false),
    )
    .unwrap();
}
fn prompt(s: &mut PagedSequence, n: u64, v: usize, chunk: u64) {
    for start in (0..n).step_by(chunk as usize) {
        let txn = s.begin().unwrap();
        let end = (start + chunk).min(n);
        s.append_prompt(end - start).unwrap();
        assert!(
            s.history(false).unwrap().is_empty(),
            "prompt tokens never enter even tentative generated history"
        );
        for p in start..end {
            append(s, txn, p, v);
        }
        s.commit_prefix(txn, 0).unwrap();
    }
}
fn stage(s: &mut PagedSequence, txn: StateTransactionId, p: u64, v: usize) -> u32 {
    let logits: Vec<_> = (0..v).map(|i| i as f32 * 0.1).collect();
    let prepared = s.prepare_sample(txn, p, &logits, None, 1.).unwrap();
    let token = prepared.draw().unwrap();
    assert_eq!(
        prepared.draw().unwrap(),
        token,
        "queries don't consume draws"
    );
    assert_eq!(prepared.stage(&AtomicBool::new(false)).unwrap(), token);
    token
}
#[derive(Debug, PartialEq)]
struct Snapshot {
    frontiers: moxie_state::Frontiers,
    lineage: Vec<Option<moxie_state::PrefixLineage>>,
    entries: Vec<(u64, u32)>,
    counts: Vec<u64>,
    working: Vec<(u64, u32)>,
    working_counts: Vec<u64>,
    rows: Vec<u8>,
}
fn snapshot(s: &PagedSequence, v: usize) -> Snapshot {
    let f = s.state().frontiers(ROOT).unwrap();
    Snapshot {
        frontiers: f,
        lineage: (0..=f.executed.max(f.accepted))
            .map(|p| s.state().lineage_at(ROOT, p).unwrap())
            .collect(),
        entries: s.history(true).unwrap().entries(0).collect(),
        counts: (0..v)
            .map(|i| s.history(true).unwrap().count(i as u32).unwrap())
            .collect(),
        working: s.history(false).unwrap().entries(0).collect(),
        working_counts: (0..v)
            .map(|i| s.history(false).unwrap().count(i as u32).unwrap())
            .collect(),
        rows: (0..f.executed)
            .flat_map(|p| {
                (0..2).flat_map(move |l| {
                    let row = s.row(l, p).unwrap();
                    row.key.iter().chain(row.value).copied()
                })
            })
            .collect(),
    }
}

#[test]
fn chunking_prompt_length_replay_and_pending_tokens() {
    for v in [3, 7] {
        let mut ledger = ledger();
        let mut a =
            PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 33377335).unwrap();
        let mut b =
            PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 33377335).unwrap();
        prompt(&mut a, 5, v, 5);
        prompt(&mut b, 5, v, 2);
        // Independent sequences have different lineage identities; bytes/counters agree.
        assert_eq!(snapshot(&a, v).rows, snapshot(&b, v).rows);
        assert!(a.history(true).unwrap().is_empty());
        for p in 5..10 {
            let ta = a.begin().unwrap();
            let tb = b.begin().unwrap();
            assert_eq!(stage(&mut a, ta, p, v), stage(&mut b, tb, p, v));
            assert_eq!(a.history(true).unwrap().len(), (p - 5) as usize);
            a.commit_prefix(ta, 1).unwrap();
            b.commit_prefix(tb, 1).unwrap();
            assert_eq!(a.state().frontiers(ROOT).unwrap().pending_execution(), 1);
            assert!(!a.state().next_logits_valid(ROOT));
            let ta = a.begin().unwrap();
            let tb = b.begin().unwrap();
            assert!(a.prepare_sample(ta, p + 1, &vec![0.; v], None, 1.).is_err());
            append(&mut a, ta, p, v);
            append(&mut b, tb, p, v);
            a.commit_prefix(ta, 0).unwrap();
            b.commit_prefix(tb, 0).unwrap();
        }
        assert_eq!(
            a.history(true).unwrap().entries(0).collect::<Vec<_>>(),
            b.history(true).unwrap().entries(0).collect::<Vec<_>>()
        );
        assert_eq!(a.history(true).unwrap().entries(2).count(), 2);
        assert_eq!(
            (0..v)
                .map(|i| a.history(true).unwrap().count(i as u32).unwrap())
                .sum::<u64>(),
            5
        );
        let mut c =
            PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 33377335).unwrap();
        prompt(&mut c, 1, v, 1);
        let tc = c.begin().unwrap();
        let first = stage(&mut c, tc, 1, v);
        assert_eq!(first, a.history(true).unwrap().entries(0).next().unwrap().1);
        c.abort(tc).unwrap();
        a.close(&mut ledger).unwrap();
        b.close(&mut ledger).unwrap();
        c.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}

#[test]
fn partial_acceptance_then_resolved_rollback_restores_counts_and_rng() {
    assert_eq!(
        StateKind::SamplerHistory.restore_capability(),
        RestoreCapability::Explicit
    );
    for v in [3, 7] {
        let mut ledger = ledger();
        let mut s = PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 77).unwrap();
        prompt(&mut s, 2, v, 1);
        let txn = s.begin().unwrap();
        let mut tokens = Vec::new();
        for p in 2..6 {
            tokens.push(stage(&mut s, txn, p, v));
            append(&mut s, txn, p, v);
        }
        s.commit_prefix(txn, 2).unwrap();
        assert_eq!(s.state().frontiers(ROOT).unwrap().tentative(), 2);
        assert_eq!(s.history(true).unwrap().len(), 2);
        assert_eq!(s.history(false).unwrap().len(), 2);
        s.rollback_to(4).unwrap();
        assert_eq!(s.state().frontiers(ROOT).unwrap().executed, 4);
        let before = snapshot(&s, v);
        for _ in 0..10 {
            let txn = s.begin().unwrap();
            assert_eq!(stage(&mut s, txn, 4, v), tokens[2]);
            append(&mut s, txn, 4, v);
            s.abort(txn).unwrap();
            assert_eq!(snapshot(&s, v), before);
        }
        // Destructive committed-history restoration undoes accumulated counts.
        s.rollback_to(3).unwrap();
        assert_eq!(
            s.history(true).unwrap().entries(0).collect::<Vec<_>>(),
            vec![(2, tokens[0])]
        );
        for i in 0..v {
            assert_eq!(
                s.history(true).unwrap().count(i as u32).unwrap(),
                u64::from(i as u32 == tokens[0])
            );
        }
        let txn = s.begin().unwrap();
        assert_eq!(stage(&mut s, txn, 3, v), tokens[1]);
        s.abort(txn).unwrap();
        s.close(&mut ledger).unwrap();
    }
}

#[test]
fn foreign_resolved_failure_and_cancellation_are_isolated() {
    let v = 3;
    let mut ledger = ledger();
    let mut a = PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 3).unwrap();
    let mut b = PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 3).unwrap();
    prompt(&mut a, 2, v, 2);
    prompt(&mut b, 2, v, 2);
    let before = snapshot(&a, v);
    let ta = a.begin().unwrap();
    let tb = b.begin().unwrap();
    assert!(a.prepare_sample(tb, 2, &[0.; 3], None, 1.).is_err());
    assert!(a.commit_prefix(tb, 0).is_err());
    assert!(a.abort(tb).is_err());
    assert_eq!(snapshot(&a, v), before);
    stage(&mut a, ta, 2, v);
    assert!(a.commit_prefix(ta, 2).is_err());
    assert!(a.accept(1).is_err());
    assert!(a.append_prompt(1).is_err());
    assert!(a.emit(1).is_err());
    assert!(a.rollback_to(1).is_err());
    a.abort(ta).unwrap();
    assert_eq!(snapshot(&a, v), before);
    assert!(a.prepare_sample(ta, 2, &[0.; 3], None, 1.).is_err());
    // The physical append itself aborts prior sampler mutations on failure.
    let ta = a.begin().unwrap();
    stage(&mut a, ta, 2, v);
    assert!(a.append(ta, 2, &[], &AtomicBool::new(false)).is_err());
    assert_eq!(snapshot(&a, v), before);
    assert!(a.state().open_transactions().is_empty());
    let ta = a.begin().unwrap();
    stage(&mut a, ta, 2, v);
    assert!(a.append(ta, 2, &[], &AtomicBool::new(true)).is_err());
    assert_eq!(snapshot(&a, v), before);
    let ta = a.begin().unwrap();
    assert!(
        a.prepare_sample(ta, 2, &[0.; 3], None, 1.)
            .unwrap()
            .stage(&AtomicBool::new(true))
            .is_err()
    );
    assert_eq!(snapshot(&a, v), before);
    b.abort(tb).unwrap();
    // Refused cleanup returns every owner; retry after an unfinished transaction.
    let ta = a.begin().unwrap();
    stage(&mut a, ta, 2, v);
    let mut foreign =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 24, 1024).unwrap()]).unwrap();
    let refused = a.close(&mut foreign).unwrap_err();
    refused.sequence.close(&mut ledger).unwrap();
    b.close(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
    let second = PagedSequence::with_sampling(&mut ledger, geometry(v), v, 32, 3).unwrap();
    assert!(second.history(true).unwrap().is_empty());
    second.close(&mut ledger).unwrap();
}

#[test]
fn admission_capacity_emission_and_stale_prefix_refuse_without_damage() {
    let mut ledger = ledger();
    let mut s = PagedSequence::with_sampling(&mut ledger, geometry(3), 3, 1, 3).unwrap();
    prompt(&mut s, 2, 3, 2);
    let txn = s.begin().unwrap();
    assert!(s.prepare_sample(txn, 1, &[0.; 3], None, 1.).is_err());
    assert!(
        s.prepare_sample(txn, 2, &[f32::NAN, 0., 0.], None, 1.)
            .is_err()
    );
    assert!(s.history(false).unwrap().is_empty());
    stage(&mut s, txn, 2, 3);
    append(&mut s, txn, 2, 3);
    s.commit_prefix(txn, 1).unwrap();
    s.emit(1).unwrap();
    let before = snapshot(&s, 3);
    assert!(s.rollback_to(2).is_err());
    assert_eq!(snapshot(&s, 3), before);
    let txn = s.begin().unwrap();
    assert_eq!(
        s.prepare_sample(txn, 3, &[0.; 3], None, 1.)
            .unwrap()
            .stage(&AtomicBool::new(false))
            .unwrap_err()
            .kind(),
        "capacity_exceeded"
    );
    assert_eq!(snapshot(&s, 3), before);
    s.close(&mut ledger).unwrap();
    let mut poor = Ledger::new([CapacitySnapshot::new(Scope::Host, 1024, 1023).unwrap()]).unwrap();
    assert!(PagedSequence::with_sampling(&mut poor, geometry(3), 3, 32, 0).is_err());
    assert!(poor.outstanding().is_empty());
    for (v, h) in [(0, 1), (3, 0), (3, 65), (usize::MAX, 1)] {
        assert!(PagedSequence::with_sampling(&mut ledger, geometry(3), v, h, 0).is_err());
    }
    assert!(ledger.outstanding().is_empty());
}
