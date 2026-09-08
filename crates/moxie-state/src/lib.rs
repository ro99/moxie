//! Sequence state: schema, branches, frontiers and output provenance.
//!
//! Document 02: this crate owns "Paged sequence state, forks, transactions,
//! rollback and prefix reuse".
//!
//! M0 scope: the *counters and provenance rules* from document 04, expressed as
//! types and tested. Paging, COW page tables and real recurrent snapshots land
//! in M1/M2/M4. The rules are here first because document 06 warns that M9
//! "must not retrofit incompatible cache ownership", and because getting them
//! wrong is the off-by-one that silently corrupts a cache.
//!
//! ## What the M0 review corrected
//!
//! The first draft had two counters, `committed` and `materialized`, and three
//! claims that the type could not support:
//!
//! 1. It **forbade** materializing past the committed frontier. But document 04
//!    says "a speculative branch may materialize unaccepted candidates beyond
//!    the committed prefix. That is valid tentative state, not corruption, and
//!    must never require publishing candidates before verification." The old
//!    rejection test could only be written by committing every proposal first --
//!    that is, by publishing tokens before verifying them.
//! 2. It conflated accepted history with what the user was shown. Those are
//!    separate counters: history includes the prompt, usage does not.
//! 3. It treated **counter equality as proof that logits exist**. A fresh state
//!    reported valid next-token logits before any forward pass, and rolling back
//!    from 20 to 12 reported valid logits at 12 although nothing was retained or
//!    recomputed there.
//!
//! So there are now four counters per branch, execution may legitimately run
//! ahead of acceptance, and logit validity is a *retained result* qualified by
//! `(branch, prefix, generation)` rather than an arithmetic coincidence.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use moxie_types::{BranchId, Error, Result};

/// The kinds of state a schema can declare (document 04).
///
/// Listed so that a rollback test can enumerate them: "A rejection rolls back
/// every state kind, including grammar/history/recurrent state."
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum StateKind {
    KvPages,
    MlaLatent,
    SparseIndex,
    RecurrentAccumulator,
    ConvolutionHistory,
    PositionCounter,
    SamplerHistory,
    ConstraintState,
}

/// How an earlier state of one component can be recovered.
///
/// Deliberately not a boolean. R20 records Kimi's documented inability to
/// recover recurrent state by decrementing a position, and the M0 review adds
/// the subtler half: a *mutable compressed or accumulated* structure is not
/// truncatable either, however "history"-like its name sounds. A sparse index
/// that is rebuilt incrementally and a frequency histogram both fall in the
/// second case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreCapability {
    /// Append-only: dropping the suffix leaves exactly the earlier state.
    Truncate,
    /// Needs explicit evidence -- a bounded snapshot at or before the target
    /// prefix, or a replay from a saved earlier state. The cost is real and has
    /// to be accounted for (document 04).
    Explicit,
}

impl StateKind {
    /// What it takes to restore this kind to an earlier prefix.
    pub const fn restore_capability(self) -> RestoreCapability {
        match self {
            // Paged KV, its MLA latent equivalent and a position counter are
            // append-only per token: the tentative suffix can be dropped.
            StateKind::KvPages | StateKind::MlaLatent | StateKind::PositionCounter => {
                RestoreCapability::Truncate
            }
            // Everything else is accumulated or compressed in place.
            // `SparseIndex` is here because a model-defined index (GLM-5.3's DSA
            // indexer) is maintained incrementally, not appended; `SamplerHistory`
            // because the sampling pipeline's presence/frequency counters and DRY
            // windows are accumulations over the emitted tokens, not a list that
            // can simply be shortened.
            StateKind::SparseIndex
            | StateKind::RecurrentAccumulator
            | StateKind::ConvolutionHistory
            | StateKind::SamplerHistory
            | StateKind::ConstraintState => RestoreCapability::Explicit,
        }
    }

    /// Every kind, so a rollback test can enumerate the schema exhaustively.
    pub const ALL: &'static [StateKind] = &[
        StateKind::KvPages,
        StateKind::MlaLatent,
        StateKind::SparseIndex,
        StateKind::RecurrentAccumulator,
        StateKind::ConvolutionHistory,
        StateKind::PositionCounter,
        StateKind::SamplerHistory,
        StateKind::ConstraintState,
    ];
}

/// Evidence that one `Explicit` state kind really was restored.
///
/// The point of requiring it is that a rollback cannot be *asserted*. Something
/// has to have taken a snapshot or replayed a prefix, and the prefix it covers
/// has to be at or before the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restore {
    /// A bounded snapshot of this component, taken at logical prefix `taken_at`.
    Snapshot { taken_at: u64 },
    /// Recomputation of this component from a saved state at prefix `from`.
    Replay { from: u64 },
}

impl Restore {
    fn covers(self, target_prefix: u64) -> bool {
        match self {
            Restore::Snapshot { taken_at } => taken_at <= target_prefix,
            Restore::Replay { from } => from <= target_prefix,
        }
    }

    fn at(self) -> u64 {
        match self {
            Restore::Snapshot { taken_at } => taken_at,
            Restore::Replay { from } => from,
        }
    }
}

/// Identity of the executing configuration.
///
/// Document 04: "changing prefix/config invalidates outputs unless an exact
/// saved result is restored." Anything that changes what a forward pass would
/// compute -- graph, precision, positional configuration, tokenizer/template,
/// checkpoint -- bumps this, and every retained logit result from an earlier
/// generation becomes stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateGeneration(pub u64);

/// A retained forward result: logits that predict the token *after* `prefix`.
///
/// All three fields are load-bearing. The same prefix on a different branch is a
/// different computation; the same branch and prefix under a different
/// generation is a different model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogitsHandle {
    pub branch: BranchId,
    pub prefix: u64,
    pub generation: StateGeneration,
}

/// The counters of one branch (document 04).
///
/// Four, not two, and none of them is derivable from the others:
///
/// * `prompt` -- prompt tokens inside the accepted prefix. Usage reports them
///   separately from completion tokens (document 05).
/// * `accepted` -- the accepted logical prefix, prompt included. Only verified
///   tokens enter it.
/// * `emitted` -- completion tokens actually published to the user. Never more
///   than the accepted completion tokens, and often fewer: a stop string can be
///   held back mid-token.
/// * `executed` -- tokens whose forward pass has run and whose state exists.
///
/// `executed` may exceed `accepted`: that is a speculative or entropy branch
/// executing candidates that have not been verified. `accepted` may exceed
/// `executed`: that is a bonus or mismatch token committed to history whose
/// forward pass is still pending. Both are normal; neither is corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Frontiers {
    pub prompt: u64,
    pub accepted: u64,
    pub emitted: u64,
    pub executed: u64,
}

impl Frontiers {
    /// Accepted tokens whose forward pass has not run. These must be executed,
    /// or an equivalent saved forward result used, before continuation.
    pub fn pending_execution(self) -> u64 {
        self.accepted.saturating_sub(self.executed)
    }

    /// Executed tokens beyond the accepted prefix: tentative branch state.
    pub fn tentative(self) -> u64 {
        self.executed.saturating_sub(self.accepted)
    }

    /// Completion tokens accepted into history.
    pub fn completion(self) -> u64 {
        self.accepted.saturating_sub(self.prompt)
    }
}

#[derive(Debug, Clone)]
struct Branch {
    frontiers: Frontiers,
    parent: Option<BranchId>,
    /// The one retained forward result for this branch, if any.
    ///
    /// One is enough for M0: decode and verification need the result at the
    /// current prefix. A retention policy over several prefixes is a memory
    /// authority question and arrives with it.
    logits: Option<LogitsHandle>,
}

/// One sequence's state: its schema, its branches and its retained outputs.
#[derive(Debug, Clone)]
pub struct SequenceState {
    schema: Vec<StateKind>,
    generation: StateGeneration,
    branches: BTreeMap<BranchId, Branch>,
    next_branch: u64,
}

/// The branch every sequence starts with.
pub const ROOT: BranchId = BranchId(0);

impl SequenceState {
    /// Create a sequence whose components are exactly `schema`.
    pub fn new(schema: impl IntoIterator<Item = StateKind>) -> Self {
        let mut schema: Vec<StateKind> = schema.into_iter().collect();
        schema.sort_unstable();
        schema.dedup();
        let mut branches = BTreeMap::new();
        branches.insert(
            ROOT,
            Branch {
                frontiers: Frontiers::default(),
                parent: None,
                logits: None,
            },
        );
        Self {
            schema,
            generation: StateGeneration(0),
            branches,
            next_branch: 1,
        }
    }

    pub fn schema(&self) -> &[StateKind] {
        &self.schema
    }

    pub fn generation(&self) -> StateGeneration {
        self.generation
    }

    pub fn branch_ids(&self) -> Vec<BranchId> {
        self.branches.keys().copied().collect()
    }

    pub fn frontiers(&self, branch: BranchId) -> Result<Frontiers> {
        Ok(self.get(branch)?.frontiers)
    }

    /// The retained forward result for `branch`, if one is still valid.
    pub fn retained_logits(&self, branch: BranchId) -> Result<Option<LogitsHandle>> {
        Ok(self.get(branch)?.logits)
    }

    fn get(&self, branch: BranchId) -> Result<&Branch> {
        self.branches.get(&branch).ok_or(Error::InvalidRequest {
            field: "branch",
            detail: format!("no such branch {branch}"),
        })
    }

    fn get_mut(&mut self, branch: BranchId) -> Result<&mut Branch> {
        self.branches.get_mut(&branch).ok_or(Error::InvalidRequest {
            field: "branch",
            detail: format!("no such branch {branch}"),
        })
    }

    /// Admit `n` prompt tokens into the accepted prefix.
    ///
    /// Only before any completion token has been accepted: a prompt cannot grow
    /// underneath generated text.
    pub fn append_prompt(&mut self, branch: BranchId, n: u64) -> Result<()> {
        let b = self.get_mut(branch)?;
        if b.frontiers.accepted != b.frontiers.prompt {
            return Err(Error::InvalidRequest {
                field: "prompt",
                detail: format!(
                    "cannot extend the prompt after {} completion token(s) were accepted",
                    b.frontiers.completion()
                ),
            });
        }
        b.frontiers.prompt = add(b.frontiers.prompt, n, "prompt")?;
        b.frontiers.accepted = add(b.frontiers.accepted, n, "accepted")?;
        Ok(())
    }

    /// Record that `n` more tokens have had their forward pass executed on this
    /// branch.
    ///
    /// Deliberately unbounded above the accepted prefix. Verification executes
    /// proposals before it knows whether they will be accepted; requiring
    /// acceptance first would mean publishing unverified tokens, which document
    /// 04 forbids.
    pub fn execute(&mut self, branch: BranchId, n: u64) -> Result<()> {
        let b = self.get_mut(branch)?;
        b.frontiers.executed = add(b.frontiers.executed, n, "executed")?;
        Ok(())
    }

    /// Accept `n` verified tokens into this branch's history.
    ///
    /// Accepting is not executing. A bonus token accepted after an all-accepted
    /// chain has no state yet, and `pending_execution` will report it.
    pub fn accept(&mut self, branch: BranchId, n: u64) -> Result<()> {
        let b = self.get_mut(branch)?;
        b.frontiers.accepted = add(b.frontiers.accepted, n, "accepted")?;
        Ok(())
    }

    /// Publish `n` completion tokens to the user and to usage.
    ///
    /// Refused beyond the accepted completion tokens: publication follows
    /// verification, never precedes it.
    pub fn emit(&mut self, branch: BranchId, n: u64) -> Result<()> {
        let b = self.get_mut(branch)?;
        let next = add(b.frontiers.emitted, n, "emitted")?;
        if next > b.frontiers.completion() {
            return Err(Error::InvalidRequest {
                field: "emitted",
                detail: format!(
                    "would publish {next} completion token(s) with only {} accepted",
                    b.frontiers.completion()
                ),
            });
        }
        b.frontiers.emitted = next;
        Ok(())
    }

    /// Retain the forward result that predicts the token after `prefix`.
    ///
    /// Refused for a prefix that has not been executed: there is no such result
    /// to retain.
    pub fn record_logits(&mut self, branch: BranchId, prefix: u64) -> Result<LogitsHandle> {
        let generation = self.generation;
        let b = self.get_mut(branch)?;
        if prefix > b.frontiers.executed {
            return Err(Error::InvalidRequest {
                field: "prefix",
                detail: format!(
                    "no forward pass has run at prefix {prefix}; only {} token(s) executed",
                    b.frontiers.executed
                ),
            });
        }
        let handle = LogitsHandle {
            branch,
            prefix,
            generation,
        };
        b.logits = Some(handle);
        Ok(handle)
    }

    /// Whether the next-token logits for `branch` are valid *right now*.
    ///
    /// True only when a retained result exists for exactly this branch, exactly
    /// the accepted prefix, and the current generation. Equality of counters
    /// proves nothing: an empty state has executed nothing, and a rollback
    /// discards the result it had.
    pub fn next_logits_valid(&self, branch: BranchId) -> bool {
        let Ok(b) = self.get(branch) else {
            return false;
        };
        match b.logits {
            Some(h) => {
                h.branch == branch
                    && h.prefix == b.frontiers.accepted
                    && h.generation == self.generation
            }
            None => false,
        }
    }

    /// Restore a previously saved forward result after a rollback or a
    /// configuration change.
    ///
    /// This is the "or use a saved equivalent forward result" half of document
    /// 04. The caller must actually hold the result; the handle it presents is
    /// checked against the branch and the current generation, so a stale one is
    /// refused rather than silently re-blessed.
    pub fn restore_logits(&mut self, branch: BranchId, handle: LogitsHandle) -> Result<()> {
        let generation = self.generation;
        let b = self.get_mut(branch)?;
        if handle.branch != branch {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: format!("result belongs to {} not {branch}", handle.branch),
            });
        }
        if handle.generation != generation {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: "result was produced under a different graph/state generation".into(),
            });
        }
        if handle.prefix > b.frontiers.executed {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: format!(
                    "result is at prefix {} but only {} token(s) are executed",
                    handle.prefix, b.frontiers.executed
                ),
            });
        }
        b.logits = Some(handle);
        Ok(())
    }

    /// Declare that the executing configuration changed.
    ///
    /// Every retained result becomes stale. Nothing is deleted -- a caller that
    /// genuinely saved an exact result may present it to `restore_logits` under
    /// the new generation -- but nothing is valid by default either.
    pub fn invalidate_generation(&mut self) {
        self.generation = StateGeneration(self.generation.0 + 1);
        for b in self.branches.values_mut() {
            b.logits = None;
        }
    }

    /// Roll back `branch` to an accepted prefix, as `abort` and a speculative
    /// rejection do.
    ///
    /// `restores` must cover every schema component whose
    /// [`RestoreCapability`] is `Explicit`, with evidence at or before
    /// `prefix`. Truncatable components need no entry. A rollback that cannot
    /// name how the recurrent state got back is not a rollback.
    ///
    /// The retained forward result is **discarded**. Truncating KV does not
    /// bring back logits that were computed at the shorter prefix and then
    /// thrown away; if the caller saved one, it says so with `restore_logits`.
    pub fn rollback_to(
        &mut self,
        branch: BranchId,
        prefix: u64,
        restores: &[(StateKind, Restore)],
    ) -> Result<()> {
        let schema = self.schema.clone();
        {
            let b = self.get(branch)?;
            if prefix > b.frontiers.accepted {
                return Err(Error::InvalidRequest {
                    field: "prefix",
                    detail: format!(
                        "cannot roll back to {prefix}, only {} accepted",
                        b.frontiers.accepted
                    ),
                });
            }
            if prefix < b.frontiers.prompt {
                return Err(Error::InvalidRequest {
                    field: "prefix",
                    detail: format!(
                        "cannot roll back to {prefix}, inside the {}-token prompt; \
                         that is a re-prefill, not a rollback",
                        b.frontiers.prompt
                    ),
                });
            }
        }

        for kind in &schema {
            if kind.restore_capability() != RestoreCapability::Explicit {
                continue;
            }
            match restores.iter().find(|(k, _)| k == kind) {
                None => {
                    return Err(Error::InvalidRequest {
                        field: "restores",
                        detail: format!(
                            "{kind:?} cannot be restored by truncating a counter and no \
                             snapshot or replay was supplied"
                        ),
                    });
                }
                Some((_, r)) if !r.covers(prefix) => {
                    return Err(Error::InvalidRequest {
                        field: "restores",
                        detail: format!(
                            "{kind:?} restore is at prefix {} which is after the rollback \
                             target {prefix}",
                            r.at()
                        ),
                    });
                }
                Some(_) => {}
            }
        }

        let b = self.get_mut(branch)?;
        b.frontiers.accepted = prefix;
        b.frontiers.executed = b.frontiers.executed.min(prefix);
        b.frontiers.emitted = b.frontiers.emitted.min(b.frontiers.completion());
        b.logits = None;
        Ok(())
    }

    /// Fork a copy-on-write branch sharing the prefix `at`.
    ///
    /// Used by speculation and by future-entropy lookahead. The child starts
    /// with no retained forward result even though it shares the prefix: a
    /// result is identified by `(branch, prefix, generation)`, and quietly
    /// re-labelling the parent's result as the child's is exactly the kind of
    /// provenance shortcut this module exists to prevent.
    pub fn fork(&mut self, parent: BranchId, at: u64) -> Result<BranchId> {
        let p = self.get(parent)?;
        if at > p.frontiers.accepted {
            return Err(Error::InvalidRequest {
                field: "at",
                detail: format!(
                    "cannot fork at {at}, parent has only {} accepted",
                    p.frontiers.accepted
                ),
            });
        }
        let frontiers = Frontiers {
            prompt: p.frontiers.prompt.min(at),
            accepted: at,
            // Usage is per generation, not per branch: a branch that is
            // discarded published nothing.
            emitted: 0,
            executed: p.frontiers.executed.min(at),
        };
        let child = BranchId(self.next_branch);
        self.next_branch += 1;
        self.branches.insert(
            child,
            Branch {
                frontiers,
                parent: Some(parent),
                logits: None,
            },
        );
        Ok(child)
    }

    pub fn parent_of(&self, branch: BranchId) -> Result<Option<BranchId>> {
        Ok(self.get(branch)?.parent)
    }

    /// Discard a branch and everything tentative on it.
    ///
    /// Document 05: entropy branches are "discarded after consumption", and
    /// document 04 requires the parent to be unchanged by their execution.
    /// Nothing here touches the parent, which is the property the tests assert.
    pub fn discard_branch(&mut self, branch: BranchId) -> Result<()> {
        if branch == ROOT {
            return Err(Error::InvalidRequest {
                field: "branch",
                detail: "the root branch cannot be discarded".into(),
            });
        }
        self.branches
            .remove(&branch)
            .map(|_| ())
            .ok_or(Error::InvalidRequest {
                field: "branch",
                detail: format!("no such branch {branch}"),
            })
    }
}

fn add(base: u64, n: u64, field: &'static str) -> Result<u64> {
    base.checked_add(n).ok_or(Error::InvalidRequest {
        field,
        detail: "token counter overflow".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A schema with one component of each kind, so a rollback has to satisfy
    /// every restore rule at once.
    fn full_schema() -> SequenceState {
        SequenceState::new(StateKind::ALL.iter().copied())
    }

    /// The restore evidence a full-schema rollback to `prefix` needs.
    fn restores_for(prefix: u64) -> Vec<(StateKind, Restore)> {
        StateKind::ALL
            .iter()
            .filter(|k| k.restore_capability() == RestoreCapability::Explicit)
            .map(|k| (*k, Restore::Snapshot { taken_at: prefix }))
            .collect()
    }

    fn kv_only() -> SequenceState {
        SequenceState::new([StateKind::KvPages, StateKind::PositionCounter])
    }

    #[test]
    fn empty_state_has_no_logits() {
        // The plainest form of the F2 defect: `Frontiers::new()` used to report
        // valid next-token logits before any forward pass had run.
        let s = kv_only();
        assert!(!s.next_logits_valid(ROOT));
        assert_eq!(s.retained_logits(ROOT).unwrap(), None);
        assert_eq!(s.frontiers(ROOT).unwrap(), Frontiers::default());
    }

    #[test]
    fn logits_become_valid_only_after_a_forward_pass_at_that_prefix() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 10).unwrap();
        assert!(!s.next_logits_valid(ROOT));

        // Cannot retain a result for a prefix nothing executed.
        assert!(s.record_logits(ROOT, 10).is_err());

        s.execute(ROOT, 10).unwrap();
        assert!(
            !s.next_logits_valid(ROOT),
            "executing does not retain a result by itself"
        );
        s.record_logits(ROOT, 10).unwrap();
        assert!(s.next_logits_valid(ROOT));
    }

    #[test]
    fn a_committed_bonus_token_is_pending_execution_and_invalidates_the_logits() {
        // Document 04's bonus-token case: committed to history, forward pass not
        // yet run.
        let mut s = kv_only();
        s.append_prompt(ROOT, 10).unwrap();
        s.execute(ROOT, 10).unwrap();
        s.record_logits(ROOT, 10).unwrap();
        assert!(s.next_logits_valid(ROOT));

        s.accept(ROOT, 1).unwrap(); // bonus token
        s.emit(ROOT, 1).unwrap();
        let f = s.frontiers(ROOT).unwrap();
        assert_eq!(f.pending_execution(), 1);
        assert_eq!(f.completion(), 1);
        assert_eq!(f.emitted, 1);
        assert!(
            !s.next_logits_valid(ROOT),
            "the retained result is at prefix 10; the accepted prefix is now 11"
        );

        s.execute(ROOT, 1).unwrap();
        s.record_logits(ROOT, 11).unwrap();
        assert!(s.next_logits_valid(ROOT));
    }

    #[test]
    fn a_branch_may_execute_tokens_it_has_not_accepted() {
        // The invariant the old type had backwards. Verification runs the
        // proposals *before* deciding which are accepted, and must not have to
        // publish them first.
        let mut s = kv_only();
        s.append_prompt(ROOT, 100).unwrap();
        s.execute(ROOT, 100).unwrap();

        let draft = s.fork(ROOT, 100).unwrap();
        s.execute(draft, 4).unwrap(); // four unverified proposals
        let f = s.frontiers(draft).unwrap();
        assert_eq!(f.tentative(), 4);
        assert_eq!(f.accepted, 100);
        assert_eq!(f.emitted, 0, "nothing was published to the user");
    }

    #[test]
    fn publication_can_never_run_ahead_of_acceptance() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 5).unwrap();
        assert!(
            s.emit(ROOT, 1).is_err(),
            "the prompt is not completion text"
        );
        s.accept(ROOT, 2).unwrap();
        assert!(s.emit(ROOT, 3).is_err());
        assert!(s.emit(ROOT, 2).is_ok());
    }

    #[test]
    fn rejection_at_every_depth_leaves_consistent_state_and_no_logits() {
        // Verification proposed 4 tokens on a branch; each rejection depth must
        // land on consistent counters, and none of them may claim logits that
        // were never retained at that prefix.
        for accepted in 0..=4u64 {
            let mut s = kv_only();
            s.append_prompt(ROOT, 100).unwrap();
            s.execute(ROOT, 100).unwrap();
            s.record_logits(ROOT, 100).unwrap();

            let draft = s.fork(ROOT, 100).unwrap();
            s.execute(draft, 4).unwrap();
            s.accept(draft, accepted).unwrap();

            s.rollback_to(draft, 100 + accepted, &[]).unwrap();
            let f = s.frontiers(draft).unwrap();
            assert_eq!(f.accepted, 100 + accepted);
            assert_eq!(f.executed, 100 + accepted);
            assert_eq!(f.tentative(), 0);
            assert!(
                !s.next_logits_valid(draft),
                "depth {accepted}: rollback discarded the result; nothing recomputed it"
            );

            // The parent is untouched by the whole episode.
            assert!(s.next_logits_valid(ROOT));
            assert_eq!(s.frontiers(ROOT).unwrap().accepted, 100);
        }
    }

    #[test]
    fn rollback_loses_logits_unless_an_exact_saved_result_is_restored() {
        // The second F2 observation: 20 -> 12 used to report valid logits at 12
        // although nothing was retained or recomputed there.
        let mut s = kv_only();
        s.append_prompt(ROOT, 8).unwrap();
        s.accept(ROOT, 12).unwrap();
        s.execute(ROOT, 20).unwrap();
        s.record_logits(ROOT, 20).unwrap();
        assert!(s.next_logits_valid(ROOT));

        s.rollback_to(ROOT, 12, &[]).unwrap();
        assert!(!s.next_logits_valid(ROOT));
        assert_eq!(s.retained_logits(ROOT).unwrap(), None);

        // A caller that genuinely saved the prefix-12 result may present it.
        let saved = LogitsHandle {
            branch: ROOT,
            prefix: 12,
            generation: s.generation(),
        };
        s.restore_logits(ROOT, saved).unwrap();
        assert!(s.next_logits_valid(ROOT));
    }

    #[test]
    fn a_stale_or_foreign_saved_result_is_refused() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 8).unwrap();
        s.execute(ROOT, 8).unwrap();
        let handle = s.record_logits(ROOT, 8).unwrap();

        // Configuration change: same counters, different model.
        s.invalidate_generation();
        assert!(!s.next_logits_valid(ROOT));
        assert!(
            s.restore_logits(ROOT, handle).is_err(),
            "a result from the previous generation must not be re-blessed"
        );

        let other = s.fork(ROOT, 8).unwrap();
        let foreign = LogitsHandle {
            branch: other,
            prefix: 8,
            generation: s.generation(),
        };
        assert!(s.restore_logits(ROOT, foreign).is_err());
    }

    #[test]
    fn rollback_refuses_state_it_cannot_restore() {
        // R20: recurrent state does not come back by decrementing a counter.
        let mut s = SequenceState::new([StateKind::KvPages, StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 10).unwrap();
        s.execute(ROOT, 10).unwrap();
        s.accept(ROOT, 5).unwrap();
        s.execute(ROOT, 5).unwrap();

        let e = s.rollback_to(ROOT, 10, &[]).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");

        // A snapshot taken *after* the target is no help either.
        assert!(
            s.rollback_to(
                ROOT,
                10,
                &[(
                    StateKind::RecurrentAccumulator,
                    Restore::Snapshot { taken_at: 12 }
                )]
            )
            .is_err()
        );

        // Replay from a saved earlier prefix works.
        s.rollback_to(
            ROOT,
            10,
            &[(StateKind::RecurrentAccumulator, Restore::Replay { from: 8 })],
        )
        .unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().accepted, 10);
    }

    #[test]
    fn every_explicit_kind_in_the_schema_must_be_covered() {
        let mut s = full_schema();
        s.append_prompt(ROOT, 4).unwrap();
        s.execute(ROOT, 4).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 4).unwrap();

        // Drop one entry at a time: each omission must be refused by name.
        let full = restores_for(4);
        for i in 0..full.len() {
            let mut partial = full.clone();
            let missing = partial.remove(i);
            let mut s2 = s.clone();
            let e = s2.rollback_to(ROOT, 4, &partial).unwrap_err();
            assert!(
                e.to_string().contains(&format!("{:?}", missing.0)),
                "omitting {:?} was not reported: {e}",
                missing.0
            );
        }
        s.rollback_to(ROOT, 4, &full).unwrap();
    }

    #[test]
    fn a_sparse_index_and_sampler_history_are_not_truncatable() {
        // The M0 review's addition to R20: a mutable compressed index and an
        // accumulated penalty history are not restored by shortening a counter,
        // however "history"-like the name is.
        assert_eq!(
            StateKind::SparseIndex.restore_capability(),
            RestoreCapability::Explicit
        );
        assert_eq!(
            StateKind::SamplerHistory.restore_capability(),
            RestoreCapability::Explicit
        );
        assert_eq!(
            StateKind::RecurrentAccumulator.restore_capability(),
            RestoreCapability::Explicit
        );
        assert_eq!(
            StateKind::ConvolutionHistory.restore_capability(),
            RestoreCapability::Explicit
        );
        assert_eq!(
            StateKind::KvPages.restore_capability(),
            RestoreCapability::Truncate
        );
    }

    #[test]
    fn a_synthetic_recurrent_replay_reproduces_the_accepted_prefix() {
        // Model-independent: a recurrent accumulator over a token stream, where
        // the state at prefix n genuinely cannot be derived from the state at
        // n+k. Replay from a saved prefix is the only way back, and this checks
        // that the state machine's rule matches that arithmetic.
        fn step(state: u64, token: u64) -> u64 {
            // Order-dependent and non-invertible: dropping the last token cannot
            // undo it.
            (state.wrapping_mul(1_000_003).wrapping_add(token)) % 1_000_000_007
        }
        let tokens: Vec<u64> = (1..=12).collect();
        let mut states = vec![0u64];
        for t in &tokens {
            let next = step(*states.last().unwrap(), *t);
            states.push(next);
        }
        // The saved snapshot is at prefix 8; the accepted prefix after rejection
        // is 10. Replay covers the gap.
        let mut replayed = states[8];
        for t in &tokens[8..10] {
            replayed = step(replayed, *t);
        }
        assert_eq!(replayed, states[10]);

        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 8).unwrap();
        s.execute(ROOT, 12).unwrap();
        s.rollback_to(
            ROOT,
            10,
            &[(StateKind::RecurrentAccumulator, Restore::Replay { from: 8 })],
        )
        .unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 10);
    }

    #[test]
    fn an_entropy_branch_leaves_its_parent_unchanged() {
        // Document 05: entropy branches are executed, read and discarded, and
        // "confirm parent state remains unchanged".
        let mut s = kv_only();
        s.append_prompt(ROOT, 30).unwrap();
        s.execute(ROOT, 30).unwrap();
        s.record_logits(ROOT, 30).unwrap();
        let before = s.frontiers(ROOT).unwrap();
        let before_logits = s.retained_logits(ROOT).unwrap();

        let mut children = Vec::new();
        for _ in 0..4 {
            let c = s.fork(ROOT, 30).unwrap();
            s.execute(c, 1).unwrap(); // one candidate token each
            s.record_logits(c, 31).unwrap();
            children.push(c);
        }
        assert_eq!(s.frontiers(ROOT).unwrap(), before);
        assert_eq!(s.retained_logits(ROOT).unwrap(), before_logits);
        assert!(s.next_logits_valid(ROOT));

        for c in children {
            s.discard_branch(c).unwrap();
        }
        assert_eq!(s.branch_ids(), vec![ROOT]);
        assert_eq!(s.frontiers(ROOT).unwrap(), before);
        assert!(s.next_logits_valid(ROOT));
    }

    #[test]
    fn a_fork_does_not_inherit_the_parents_retained_result() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 6).unwrap();
        s.execute(ROOT, 6).unwrap();
        s.record_logits(ROOT, 6).unwrap();

        let child = s.fork(ROOT, 6).unwrap();
        assert_eq!(s.retained_logits(child).unwrap(), None);
        assert!(!s.next_logits_valid(child));
        assert_eq!(s.parent_of(child).unwrap(), Some(ROOT));
    }

    #[test]
    fn boundaries_and_overflow_are_refused_rather_than_wrapped() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 5).unwrap();
        assert!(
            s.rollback_to(ROOT, 9, &[]).is_err(),
            "past the accepted prefix"
        );
        assert!(
            s.rollback_to(ROOT, 2, &[]).is_err(),
            "into the prompt: that is a re-prefill"
        );
        s.accept(ROOT, u64::MAX - 5).unwrap();
        assert!(s.accept(ROOT, 1).is_err());
        assert!(s.execute(ROOT, u64::MAX).is_ok());
        assert!(s.execute(ROOT, 1).is_err());
        assert!(s.frontiers(BranchId(999)).is_err());
    }

    #[test]
    fn the_prompt_cannot_grow_under_generated_text() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 1).unwrap();
        assert!(s.append_prompt(ROOT, 1).is_err());
    }
}
