//! The host key/value cache.
//!
//! One [`KvHistory`] per layer, indexed by absolute sequence position. This is
//! the M1 reference form: a dense `Vec` per layer, no pages, no eviction, no
//! device memory. Document 06 puts paged device state in M4 and the real
//! residency authority in M2, and this must not grow into either -- what it is
//! for is giving the interpreter somewhere to append so the state contracts have
//! a consumer.
//!
//! `StateKind::KvPages` is `RestoreCapability::Truncate`, and [`KvCache::rollback_to`]
//! is what that means physically.

use moxie_oracles::attention::KvHistory;
use moxie_state::{PrefixLineage, SequenceId, SequenceState};
use moxie_types::{BranchId, Error, Result};

/// Which sequence, branch and version of the prefix a cache holds.
///
/// The fourth review substituted one sequence's cache into another's execution:
/// the lengths matched, so the step succeeded, produced the other history's
/// answer, and `next_logits_valid` returned true. Physical bytes had no identity,
/// so all the provenance work in `moxie-state` guarded a number while the data
/// it described came from somewhere else.
///
/// The identity reuses what task 0002 already built rather than inventing a
/// second scheme. `PrefixLineage::root` mixes in the sequence and the branch, and
/// the chain changes when a suffix is replaced, so one value distinguishes
/// "another sequence", "another branch" and "the prefix that used to be here".
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheOwner {
    sequence: SequenceId,
    branch: BranchId,
    /// `stamps[n]` is the lineage of prefix `n` **as it stood when the cache
    /// reached that length**, one entry per prefix from 0 up to `len()`.
    ///
    /// A single "current" stamp is not enough, and the fifth review showed why:
    /// re-stamping it from the state certifies whatever bytes happen to be
    /// there. Recording the lineage at the moment each prefix was written means
    /// a rollback can compare what the cache *remembers* about a prefix against
    /// what the state now says, and refuse when those positions were rewritten
    /// behind its back.
    stamps: Vec<PrefixLineage>,
}

/// What a cache transaction must put back if it aborts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheJournal {
    lengths: Vec<usize>,
    stamps: usize,
}

/// Per-layer key/value history, bound to one branch of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    layers: Vec<KvHistory>,
    owner: CacheOwner,
}

impl KvCache {
    /// A cache for `layers` attention layers, bound to `branch`.
    ///
    /// The branch must not have executed anything yet: a cache starts empty, and
    /// there is no operation that copies an existing history into a new one. When
    /// forking gets its copy-on-write implementation it will need one, and it
    /// will have to carry the stamps across with the bytes.
    pub fn for_branch(layers: usize, state: &SequenceState, branch: BranchId) -> Result<Self> {
        let at = state.frontiers(branch)?.executed;
        if at != 0 {
            return Err(Error::InvalidRequest {
                field: "branch",
                detail: format!(
                    "{branch} has already executed {at} token(s); an empty cache cannot \
                     stand in for that history"
                ),
            });
        }
        let lineage = state.lineage_at(branch, 0)?.ok_or(Error::InvalidRequest {
            field: "branch",
            detail: format!("prefix 0 is not occupied on {branch}"),
        })?;
        Ok(Self {
            layers: vec![KvHistory::new(); layers],
            owner: CacheOwner {
                sequence: state.id(),
                branch,
                stamps: vec![lineage],
            },
        })
    }

    /// Check that this cache is the one `branch` of `state` should be reading.
    ///
    /// Four ways it can be wrong, all of them silent before this existed:
    /// another sequence, another branch, a length that disagrees with the
    /// frontier, and the right branch and length after those positions were
    /// rewritten.
    pub fn check_owner(&self, state: &SequenceState, branch: BranchId) -> Result<()> {
        if self.owner.sequence != state.id() {
            return Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: format!(
                    "this cache belongs to sequence {}, not {}; physical cache contents \
                     are not interchangeable between sequences",
                    self.owner.sequence.get(),
                    state.id().get()
                ),
            });
        }
        if self.owner.branch != branch {
            return Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: format!("this cache belongs to {}, not {branch}", self.owner.branch),
            });
        }
        let executed = state.frontiers(branch)?.executed;
        if self.len() as u64 != executed {
            return Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: format!(
                    "the cache holds {} position(s) but the branch has executed {executed}",
                    self.len()
                ),
            });
        }
        self.check_stamp(state, branch, executed)
    }

    /// Compare what the cache remembers about `prefix` with what the state says.
    fn check_stamp(&self, state: &SequenceState, branch: BranchId, prefix: u64) -> Result<()> {
        let remembered = self
            .owner
            .stamps
            .get(prefix as usize)
            .ok_or(Error::InvalidRequest {
                field: "kv_cache",
                detail: format!("the cache has no record of prefix {prefix}"),
            })?;
        match state.lineage_at(branch, prefix)? {
            Some(l) if l == *remembered => Ok(()),
            _ => Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: format!(
                    "the cache holds a different version of prefix {prefix}: those \
                     positions have been replaced since it was written"
                ),
            }),
        }
    }

    /// Record that the cache now holds the branch's executed prefix.
    ///
    /// Called by the interpreter once a step has succeeded and the state has
    /// advanced, and **only** then: the stamp for a prefix is written when the
    /// bytes for it are, which is what makes a later comparison mean anything.
    /// There is deliberately no public way to re-stamp a cache without adding
    /// the contents that justify it -- the fifth review found that a public
    /// re-stamp certifies whatever bytes happen to be there.
    pub(crate) fn commit(&mut self, state: &SequenceState, branch: BranchId) -> Result<()> {
        if self.owner.sequence != state.id() || self.owner.branch != branch {
            return Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: "committing to a cache that belongs to another sequence or branch".into(),
            });
        }
        let executed = state.frontiers(branch)?.executed;
        // Both halves of "the cache describes exactly this prefix". `len` reads
        // the first layer only, so without the coherence test a cache whose
        // layers had drifted apart would be stamped as current on the strength
        // of layer 0. The interpreter refuses a layer-count mismatch before it
        // writes anything, and a graph's attention layers are dense by
        // construction, so reaching either arm means a caller supplied a cache
        // that was already ragged. Inside a transaction that is simply an
        // error; it does not have to be unreachable to be safe.
        if self.len() as u64 != executed || !self.is_coherent() {
            let lens: Vec<usize> = self.layers.iter().map(KvHistory::len).collect();
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "the cache layers hold {lens:?} position(s), not {executed} each,                      which is what the branch has executed"
                ),
            });
        }
        while (self.owner.stamps.len() as u64) <= executed {
            let p = self.owner.stamps.len() as u64;
            let l = state.lineage_at(branch, p)?.ok_or(Error::InvalidArtifact {
                detail: format!("prefix {p} is not occupied on {branch}"),
            })?;
            self.owner.stamps.push(l);
        }
        self.owner.stamps.truncate(executed as usize + 1);
        Ok(())
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Open a transaction: record what an abort would have to put back.
    ///
    /// The same journal shape `moxie-state` uses, for the same reason -- the
    /// mutations are monotone, so recording where they started is exact and
    /// cheap. The cache is a *second participant* rather than living inside the
    /// state crate, because `moxie-state` owns sequence state and must not reach
    /// into a consumer's buffers; `moxie-interp` is the composition point that
    /// opens and resolves both together.
    pub fn begin(&self) -> CacheJournal {
        CacheJournal {
            lengths: self.layers.iter().map(KvHistory::len).collect(),
            stamps: self.owner.stamps.len(),
        }
    }

    /// Restore exactly what `begin` recorded. Infallible.
    pub fn abort(&mut self, journal: &CacheJournal) {
        for (l, n) in self.layers.iter_mut().zip(&journal.lengths) {
            l.truncate(*n as u64);
        }
        self.owner.stamps.truncate(journal.stamps);
    }

    /// The stored histories, without the ownership stamp.
    ///
    /// For comparing what two caches *hold*. Two caches on different sequences
    /// legitimately have different owners while holding identical bytes -- that
    /// is the whole point of the stamp -- so a test that means "the same data"
    /// must say so rather than comparing the whole value.
    pub fn contents(&self) -> &[KvHistory] {
        &self.layers
    }

    pub fn history(&self, layer: u32) -> Result<&KvHistory> {
        self.layers
            .get(layer as usize)
            .ok_or(Error::InvalidRequest {
                field: "layer",
                detail: format!("layer {layer} of {}", self.layers.len()),
            })
    }

    pub fn append(
        &mut self,
        layer: u32,
        position: u64,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        let n = self.layers.len();
        self.layers
            .get_mut(layer as usize)
            .ok_or(Error::InvalidRequest {
                field: "layer",
                detail: format!("layer {layer} of {n}"),
            })?
            .append(position, key, value)
    }

    /// Drop everything at or after `prefix`, on every layer, and re-stamp the
    /// identity from the state that authorised it.
    ///
    /// The physical half of a rollback. `moxie-state` decides *whether* the
    /// rollback is allowed -- published output, restore evidence, lineage -- and
    /// this performs it once that has been agreed. Taking the state as an
    /// argument is what keeps the two halves from drifting apart.
    pub fn rollback_to(
        &mut self,
        state: &SequenceState,
        branch: BranchId,
        prefix: u64,
    ) -> Result<()> {
        if self.owner.sequence != state.id() || self.owner.branch != branch {
            return Err(Error::InvalidRequest {
                field: "kv_cache",
                detail: "rolling back a cache that belongs to another sequence or branch".into(),
            });
        }
        if prefix > self.len() as u64 {
            return Err(Error::InvalidRequest {
                field: "prefix",
                detail: format!(
                    "cannot roll back to {prefix}; the cache holds {} position(s)",
                    self.len()
                ),
            });
        }
        // Check *before* mutating: the stamp the cache recorded for this prefix
        // must still be the state's. Truncating first and re-stamping afterwards
        // is the bypass the fifth review found -- it certifies whatever survived
        // the truncation, including bytes for positions that were rewritten.
        self.check_stamp(state, branch, prefix)?;
        for l in &mut self.layers {
            l.truncate(prefix);
        }
        self.owner.stamps.truncate(prefix as usize + 1);
        Ok(())
    }

    /// The number of positions held, which must agree across layers.
    pub fn len(&self) -> usize {
        self.layers.first().map(KvHistory::len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether every layer holds the same number of positions.
    ///
    /// A layer that fell behind means a step wrote some layers and not others,
    /// which is precisely what staging appends is meant to prevent.
    pub fn is_coherent(&self) -> bool {
        let n = self.len();
        self.layers.iter().all(|l| l.len() == n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_state::{ROOT, StateKind};

    fn state_and_cache(layers: usize) -> (SequenceState, KvCache) {
        let state = SequenceState::new([StateKind::KvPages]);
        let kv = KvCache::for_branch(layers, &state, ROOT).unwrap();
        (state, kv)
    }

    #[test]
    fn a_fresh_cache_is_empty_and_coherent() {
        let (_state, kv) = state_and_cache(3);
        assert_eq!(kv.layers(), 3);
        assert!(kv.is_empty());
        assert!(kv.is_coherent());
    }

    #[test]
    fn appends_are_per_layer_and_positions_must_be_dense() {
        let (_state, mut kv) = state_and_cache(2);
        kv.append(0, 0, vec![1.0], vec![2.0]).unwrap();
        assert!(!kv.is_coherent(), "layer 1 has not been written yet");
        kv.append(1, 0, vec![3.0], vec![4.0]).unwrap();
        assert!(kv.is_coherent());
        assert_eq!(kv.len(), 1);

        assert!(kv.append(0, 2, vec![1.0], vec![2.0]).is_err(), "gap");
        assert!(
            kv.append(9, 1, vec![1.0], vec![2.0]).is_err(),
            "no such layer"
        );
        assert!(kv.history(9).is_err());
    }

    #[test]
    fn rollback_truncates_every_layer_to_the_same_prefix() {
        let (mut state, mut kv) = state_and_cache(2);
        state.append_prompt(ROOT, 2).unwrap();
        state.accept(ROOT, 3).unwrap();
        for p in 0..5u64 {
            kv.append(0, p, vec![p as f32], vec![-(p as f32)]).unwrap();
            kv.append(1, p, vec![p as f32 * 2.0], vec![0.0]).unwrap();
            state.execute(ROOT, 1).unwrap();
            kv.commit(&state, ROOT).unwrap();
        }
        assert_eq!(kv.len(), 5);
        state.rollback_to(ROOT, 3, &[]).unwrap();
        kv.rollback_to(&state, ROOT, 3).unwrap();
        assert_eq!(kv.len(), 3);
        assert!(kv.is_coherent());
        kv.check_owner(&state, ROOT).unwrap();

        // Truncation restores exactly the earlier layer contents, which is what
        // RestoreCapability::Truncate claims. The owner stamp is not compared:
        // a rolled-back cache legitimately knows it has been rolled back.
        let (mut fresh_state, mut fresh) = state_and_cache(2);
        fresh_state.append_prompt(ROOT, 2).unwrap();
        fresh_state.accept(ROOT, 1).unwrap();
        fresh_state.execute(ROOT, 3).unwrap();
        let _ = &fresh_state;
        for p in 0..3u64 {
            fresh
                .append(0, p, vec![p as f32], vec![-(p as f32)])
                .unwrap();
            fresh.append(1, p, vec![p as f32 * 2.0], vec![0.0]).unwrap();
        }
        assert_eq!(kv.contents(), fresh.contents());
    }

    #[test]
    fn a_cache_belongs_to_one_branch_of_one_sequence() {
        // Fourth review, reproduced: a cache from another sequence was accepted
        // because only the lengths were compared.
        let (a, kv_a) = state_and_cache(1);
        let (b, _kv_b) = state_and_cache(1);
        assert_ne!(a.id(), b.id());
        kv_a.check_owner(&a, ROOT).unwrap();
        let e = kv_a.check_owner(&b, ROOT).unwrap_err();
        assert!(e.to_string().contains("sequence"), "{e}");

        // Another branch of the same sequence is refused too.
        let mut a = a;
        let child = a.fork(ROOT, 0).unwrap();
        assert!(kv_a.check_owner(&a, child).is_err());
    }

    #[test]
    fn a_cache_that_lags_or_leads_the_frontier_is_refused() {
        let (mut state, mut kv) = state_and_cache(1);
        state.append_prompt(ROOT, 2).unwrap();
        state.execute(ROOT, 2).unwrap();
        // The state executed two tokens; the cache holds none.
        assert!(kv.check_owner(&state, ROOT).is_err());
        kv.append(0, 0, vec![1.0], vec![1.0]).unwrap();
        kv.append(0, 1, vec![1.0], vec![1.0]).unwrap();
        kv.commit(&state, ROOT).unwrap();
        kv.check_owner(&state, ROOT).unwrap();
    }

    #[test]
    fn a_cache_whose_layers_disagree_cannot_be_stamped_as_current() {
        // `len` reads layer 0, so a ragged cache would otherwise be stamped as
        // describing the executed prefix on the strength of that one layer, and
        // every later `check_owner` would agree with it. Until publication
        // became a transaction this was checked in the interpreter, before the
        // state advanced; it belongs with the stamp it protects.
        let (mut state, mut kv) = state_and_cache(2);
        state.append_prompt(ROOT, 1).unwrap();
        state.execute(ROOT, 1).unwrap();
        kv.append(0, 0, vec![1.0], vec![1.0]).unwrap();
        // Layer 1 is left behind. Layer 0 alone matches `executed`.
        assert_eq!(kv.len(), 1);
        let e = kv.commit(&state, ROOT).unwrap_err();
        assert!(e.to_string().contains("[1, 0]"), "{e}");
        kv.append(1, 0, vec![1.0], vec![1.0]).unwrap();
        kv.commit(&state, ROOT).unwrap();
    }

    #[test]
    fn a_cache_holding_a_replaced_prefix_is_refused() {
        // Same sequence, same branch, same length -- but those positions were
        // rewritten, so the bytes describe a prefix that no longer exists.
        let (mut state, mut kv) = state_and_cache(1);
        state.append_prompt(ROOT, 1).unwrap();
        state.accept(ROOT, 1).unwrap();
        state.execute(ROOT, 2).unwrap();
        kv.append(0, 0, vec![1.0], vec![1.0]).unwrap();
        kv.append(0, 1, vec![2.0], vec![2.0]).unwrap();
        kv.commit(&state, ROOT).unwrap();
        kv.check_owner(&state, ROOT).unwrap();

        // Roll the state back and re-execute a different token at position 1,
        // without telling the cache.
        state.rollback_to(ROOT, 1, &[]).unwrap();
        state.accept(ROOT, 1).unwrap();
        state.execute(ROOT, 1).unwrap();
        let e = kv.check_owner(&state, ROOT).unwrap_err();
        assert!(e.to_string().contains("different version"), "{e}");
    }
}
