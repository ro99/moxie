//! Sequence state: schema, branches, frontiers and output provenance.
//!
//! Document 02: this crate owns "Paged sequence state, forks, transactions,
//! rollback and prefix reuse".
//!
//! M0 scope: the *counters, identities and provenance rules* from document 04,
//! expressed as types and tested. Paging, COW page tables and the buffers that
//! hold real KV, recurrent and logit data land in M1/M2/M4. The rules are here
//! first because document 06 warns that M9 "must not retrofit incompatible cache
//! ownership", and because getting them wrong is the off-by-one that silently
//! corrupts a cache.
//!
//! ## What the M0 reviews corrected
//!
//! The first draft had two counters, `committed` and `materialized`, and three
//! claims the type could not support:
//!
//! 1. It **forbade** materializing past the committed frontier. Document 04 says
//!    "a speculative branch may materialize unaccepted candidates beyond the
//!    committed prefix. That is valid tentative state, not corruption, and must
//!    never require publishing candidates before verification." The old
//!    rejection test could only be written by committing every proposal first --
//!    that is, by publishing tokens before verifying them.
//! 2. It conflated accepted history with what the user was shown.
//! 3. It treated **counter equality as proof that logits exist**.
//!
//! The second review found that fixing (3) with a `(branch, prefix, generation)`
//! triple was still not identity. Every fresh sequence starts at the same root
//! branch and generation, so a handle from one sequence was accepted by another;
//! and a handle for a *discarded* suffix became valid again once the branch was
//! re-executed to the same length. Both are reproduced as tests below. It also
//! found that a snapshot taken at prefix 4 was accepted as restoration to prefix
//! 6, and that a rollback silently reduced the count of tokens already published
//! to the user.
//!
//! So: retained results now carry an opaque, sequence-issued identity and a
//! **prefix lineage** that changes when a suffix is replaced; restore evidence
//! must name the prefix it actually completed; and publication is monotonic.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{BranchId, Error, Result, StateTransactionId};

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
    /// Needs explicit evidence -- a snapshot *of* the target prefix, or a replay
    /// that ran forward *to* it. The cost is real and has to be accounted for
    /// (document 04).
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

/// How one `Explicit` component reached the rollback target.
///
/// The distinction the second review required: an *available source* is not a
/// *completed restoration*. A snapshot taken at prefix 4 holds the state at
/// prefix 4. Rolling back to prefix 6 with only that snapshot in hand leaves the
/// component two tokens behind, and accepting it advances the execution frontier
/// past state that does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreMethod {
    /// An exact snapshot **of the target prefix** was reloaded.
    Snapshot { of_prefix: u64 },
    /// The component was recomputed from a saved state at `from` and replayed
    /// forward, finishing at `to`. Both ends are named: `from` alone says where
    /// the work started, not where it ended.
    Replay { from: u64, to: u64 },
}

impl RestoreMethod {
    /// The prefix this restoration actually finished at.
    pub const fn completed_prefix(self) -> u64 {
        match self {
            RestoreMethod::Snapshot { of_prefix } => of_prefix,
            RestoreMethod::Replay { to, .. } => to,
        }
    }

    const fn is_coherent(self) -> bool {
        match self {
            RestoreMethod::Snapshot { .. } => true,
            RestoreMethod::Replay { from, to } => from <= to,
        }
    }
}

/// Evidence that one `Explicit` component really was restored.
///
/// The point of requiring it is that a rollback cannot be *asserted*. Something
/// has to have reloaded a snapshot or replayed a prefix -- for this component,
/// on this sequence, on this branch, at this version of that prefix, under this
/// graph generation -- and finished at the target.
///
/// Built only by [`SequenceState::restore_evidence`], which stamps the identity
/// from the state as it stands when the restoration is performed. The rollback
/// then checks that identity again. **What this proves is a binding, not a
/// deed**: it cannot establish that the work happened, only that the evidence
/// names this exact place and that nothing has moved underneath it since. A
/// caller that mints evidence and does nothing still passes, and the buffers
/// that would make it checkable belong to the memory authority, which does not
/// exist yet.
///
/// What it does catch, and what the third M0 review found it did not:
///
/// * evidence from another branch -- a root snapshot satisfying a child's
///   rollback, because branch identity was simply absent;
/// * evidence for a version of the prefix that has since been replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Restore {
    kind: StateKind,
    sequence: SequenceId,
    branch: BranchId,
    /// The lineage of the completed prefix, as it stood when this evidence was
    /// minted. Replacing that prefix's contents changes it.
    lineage: PrefixLineage,
    generation: StateGeneration,
    method: RestoreMethod,
}

impl Restore {
    pub const fn kind(&self) -> StateKind {
        self.kind
    }
    pub const fn sequence(&self) -> SequenceId {
        self.sequence
    }
    pub const fn branch(&self) -> BranchId {
        self.branch
    }
    pub const fn lineage(&self) -> PrefixLineage {
        self.lineage
    }
    pub const fn generation(&self) -> StateGeneration {
        self.generation
    }
    pub const fn method(&self) -> RestoreMethod {
        self.method
    }
}

/// Process-unique identity of one sequence's state.
///
/// Two `SequenceState` values never share one, so a retained result cannot cross
/// between them. Allocated from a process counter rather than supplied by the
/// caller, because a caller that supplies its own can duplicate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Opaque identity of one retained forward result.
///
/// There is deliberately no public constructor. Only [`SequenceState::record_logits`]
/// mints one, so a caller -- including a test -- cannot fabricate a handle for a
/// result that was never computed. The first correction pass got this wrong: its
/// rollback test built a "saved" handle out of struct literals and then asserted
/// that the state accepted it, which tested the assertion rather than provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResultId(u64);

impl ResultId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identity of the executing configuration.
///
/// Document 04: "changing prefix/config invalidates outputs unless an exact
/// saved result is restored." Anything that changes what a forward pass would
/// compute -- graph, precision, positional configuration, tokenizer/template,
/// checkpoint -- bumps this, and every retained result from an earlier
/// generation becomes stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateGeneration(u64);

impl StateGeneration {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The execution lineage of a prefix.
///
/// A chain value over the positions `0..prefix`, where each position carries the
/// *epoch* it was written in. Rolling back bumps the branch's epoch, so a
/// position re-executed after a rollback contributes differently from the one it
/// replaced. Two prefixes therefore share a lineage only when they are the same
/// positions written by the same executions.
///
/// It is a lineage, **not** a content digest: it distinguishes "this suffix was
/// replaced" from "this suffix is unchanged". Document 04's prefix-reuse key --
/// checkpoint, tokenizer/template, configuration and token IDs -- is a separate
/// identity that composes with this one and is not implemented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixLineage(u64);

impl PrefixLineage {
    fn root(sequence: SequenceId, branch: BranchId) -> Self {
        Self(mix(mix(0xcbf2_9ce4_8422_2325, sequence.0), branch.get()))
    }

    fn extend(self, epoch: u64, position: u64) -> Self {
        Self(mix(mix(self.0, epoch), position))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// FNV-1a over one `u64`. A cache/identity key, never an integrity checksum.
fn mix(h: u64, v: u64) -> u64 {
    let mut h = h;
    for b in v.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A retained forward result: logits that predict the token *after* `prefix`.
///
/// Every field is load-bearing, and all of them are private. The same prefix
/// length on a different sequence, a different branch, a different graph
/// generation or a *replaced* suffix are four different computations, and each
/// of them was accepted by an earlier version of this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogitsHandle {
    id: ResultId,
    sequence: SequenceId,
    branch: BranchId,
    prefix: u64,
    lineage: PrefixLineage,
    generation: StateGeneration,
}

impl LogitsHandle {
    pub const fn id(&self) -> ResultId {
        self.id
    }
    pub const fn sequence(&self) -> SequenceId {
        self.sequence
    }
    pub const fn branch(&self) -> BranchId {
        self.branch
    }
    pub const fn prefix(&self) -> u64 {
        self.prefix
    }
    pub const fn lineage(&self) -> PrefixLineage {
        self.lineage
    }
    pub const fn generation(&self) -> StateGeneration {
        self.generation
    }
}

/// The counters of one branch (document 04).
///
/// Four, not two, and none of them is derivable from the others:
///
/// * `prompt` -- prompt tokens inside the accepted prefix. Usage reports them
///   separately from completion tokens (document 05).
/// * `accepted` -- the accepted logical prefix, prompt included. Only verified
///   tokens enter it. **This is what usage counts**: document 05 bills "prompt
///   tokens and committed completion tokens only".
/// * `emitted` -- completion tokens actually released to the client as deltas.
///   Never more than the accepted completion tokens, and often fewer: a stop
///   string is held back until it is known not to be one.
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

    /// Committed completion tokens. **This is the usage number**, not `emitted`.
    pub fn completion(self) -> u64 {
        self.accepted.saturating_sub(self.prompt)
    }

    /// Committed completion tokens not yet released to the client, because a
    /// stop string might still be forming across them.
    pub fn withheld(self) -> u64 {
        self.completion().saturating_sub(self.emitted)
    }
}

#[derive(Debug, Clone)]
struct Branch {
    frontiers: Frontiers,
    parent: Option<BranchId>,
    /// `lineage[n]` is the lineage of prefix `n` on this branch. Its length is
    /// always `max(accepted, executed) + 1`.
    lineage: Vec<PrefixLineage>,
    /// Bumped on every rollback, so positions written afterwards differ from the
    /// ones they replaced.
    epoch: u64,
    /// The result this branch is currently continuing from, if any.
    logits: Option<LogitsHandle>,
}

impl Branch {
    fn high_water(&self) -> u64 {
        self.frontiers.accepted.max(self.frontiers.executed)
    }

    /// Give every occupied position a lineage entry.
    fn extend_lineage(&mut self) {
        while (self.lineage.len() as u64) <= self.high_water() {
            let position = self.lineage.len() as u64 - 1;
            let next = self.lineage[position as usize].extend(self.epoch, position);
            self.lineage.push(next);
        }
    }
}

/// What one transaction must put back if it aborts.
///
/// A journal, not a copy. Every mutation inside a transaction is monotone --
/// counters only rise, the lineage vector only grows, results are only added --
/// so recording where each of them started is exact *and* `O(1)`. Cloning the
/// branch would be equally exact and `O(context)` per step, which makes a
/// sequence quadratic; the cheap-looking option is the wrong one here.
///
/// That "only" is a rule, not an observation. A transaction is **append-only**:
/// while one is open on a branch, `rollback_to` and `emit` are refused, and
/// `invalidate_generation` is refused outright. The fifth review reproduced
/// what the alternative costs -- a rollback inside a transaction discards
/// lineage entries *and* live results, and an earlier version of this journal
/// restored the first but not the second, so aborting left a branch whose
/// counters said one thing and whose retained result was gone. Journaling every
/// discarded thing is possible; refusing the destructive operation for the
/// duration is exact, cheap, and has no use case against it, because document
/// 04's mechanism for keeping part of a transaction's work is
/// `commit_prefix(n)`.
#[derive(Debug, Clone)]
struct Journal {
    branch: BranchId,
    frontiers: Frontiers,
    lineage_len: usize,
    epoch: u64,
    logits: Option<LogitsHandle>,
    /// The result counter at `begin`. Results with an id at or above this and
    /// this journal's branch were minted inside the transaction.
    ///
    /// It is a *lower bound for identification*, never a value to restore. The
    /// fifth review reproduced why: rewinding the counter let the next attempt
    /// mint a handle identical in every field to one that had been discarded,
    /// and `restore_logits` accepted it. Identities are not recycled, so an
    /// aborted result stays dead.
    results_from: u64,
}

/// One sequence's state: its schema, its branches and its retained outputs.
///
/// **Deliberately not `Clone`.** The third M0 review reproduced the reason: a
/// derived `Clone` copied the sequence id, the result counter, the lineages and
/// the issued-result ledger into a second, independently mutable object. Two
/// execution authorities then minted identical `ResultId`s, and a handle from
/// one restored into the other -- exactly the isolation the identities exist to
/// provide, undone by a derive.
///
/// Branching is what this type offers instead: [`SequenceState::fork`] creates a
/// copy-on-write branch *inside* one authority, with its own identity. If whole-
/// state copying is ever needed, it has to arrive as an operation that assigns a
/// fresh sequence id and states what happens to retained results, not as a
/// derive.
///
/// ```compile_fail
/// use moxie_state::{SequenceState, StateKind};
/// let a = SequenceState::new([StateKind::KvPages]);
/// let b = a.clone(); // a second authority would issue the same identities
/// drop(b);
/// ```
#[derive(Debug)]
pub struct SequenceState {
    id: SequenceId,
    schema: Vec<StateKind>,
    generation: StateGeneration,
    branches: BTreeMap<BranchId, Branch>,
    /// Results this sequence has issued and not invalidated.
    ///
    /// Membership is what makes a handle unforgeable: a value that never came
    /// from `record_logits`, or one whose result has since been discarded, is
    /// not in here. The buffer the result lives in belongs to the memory
    /// authority, which does not exist yet; this is the ledger it will consult.
    live: BTreeMap<ResultId, LogitsHandle>,
    next_branch: u64,
    next_result: u64,
    /// Open transactions, by id. At most one per branch: two overlapping
    /// journals cannot both be the truth about what to restore.
    open: BTreeMap<StateTransactionId, Journal>,
    next_transaction: u64,
}

/// The branch every sequence starts with.
pub const ROOT: BranchId = BranchId(0);

impl SequenceState {
    /// Create a sequence whose components are exactly `schema`.
    pub fn new(schema: impl IntoIterator<Item = StateKind>) -> Self {
        let mut schema: Vec<StateKind> = schema.into_iter().collect();
        schema.sort_unstable();
        schema.dedup();
        let id = SequenceId::next();
        let mut branches = BTreeMap::new();
        branches.insert(
            ROOT,
            Branch {
                frontiers: Frontiers::default(),
                parent: None,
                lineage: vec![PrefixLineage::root(id, ROOT)],
                epoch: 0,
                logits: None,
            },
        );
        Self {
            id,
            schema,
            generation: StateGeneration::default(),
            branches,
            live: BTreeMap::new(),
            next_branch: 1,
            next_result: 1,
            open: BTreeMap::new(),
            next_transaction: 1,
        }
    }

    pub fn id(&self) -> SequenceId {
        self.id
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

    /// Results this sequence has issued and not invalidated.
    pub fn live_results(&self) -> Vec<LogitsHandle> {
        self.live.values().copied().collect()
    }

    pub fn frontiers(&self, branch: BranchId) -> Result<Frontiers> {
        Ok(self.get(branch)?.frontiers)
    }

    /// The lineage of `prefix` on `branch`, if that prefix is occupied.
    pub fn lineage_at(&self, branch: BranchId, prefix: u64) -> Result<Option<PrefixLineage>> {
        Ok(self.get(branch)?.lineage.get(prefix as usize).copied())
    }

    /// The result `branch` is currently continuing from, if it is still valid.
    pub fn retained_logits(&self, branch: BranchId) -> Result<Option<LogitsHandle>> {
        let b = self.get(branch)?;
        Ok(match b.logits {
            Some(h) if self.handle_is_live(h).is_ok() => Some(h),
            _ => None,
        })
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
        b.extend_lineage();
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
        b.extend_lineage();
        Ok(())
    }

    /// Accept `n` verified tokens into this branch's history.
    ///
    /// Accepting is not executing. A bonus token accepted after an all-accepted
    /// chain has no state yet, and `pending_execution` will report it.
    pub fn accept(&mut self, branch: BranchId, n: u64) -> Result<()> {
        let b = self.get_mut(branch)?;
        b.frontiers.accepted = add(b.frontiers.accepted, n, "accepted")?;
        b.extend_lineage();
        Ok(())
    }

    /// Release `n` completion tokens to the client.
    ///
    /// Refused beyond the accepted completion tokens: publication follows
    /// verification, never precedes it. **Monotonic within a response** -- see
    /// `rollback_to`, which refuses to move the accepted prefix below what has
    /// already been released.
    ///
    /// Also refused while a transaction is open on the branch. Emission is the
    /// one mutation here that leaves the process -- once text has reached the
    /// client, no counter can retract it -- so tentative work must not be
    /// emitted. The fifth review reproduced the alternative: `emit` inside a
    /// transaction moved `emitted` from 0 to 1, and `abort` moved it back,
    /// which is a restored counter describing output the client already has.
    pub fn emit(&mut self, branch: BranchId, n: u64) -> Result<()> {
        if let Some(id) = self.open_on(branch) {
            return Err(Error::InvalidRequest {
                field: "emitted",
                detail: format!(
                    "{branch} has transaction {} open; tentative work cannot be released \
                     to the client, because an abort cannot retract it",
                    id.get()
                ),
            });
        }
        let b = self.get_mut(branch)?;
        let next = add(b.frontiers.emitted, n, "emitted")?;
        if next > b.frontiers.completion() {
            return Err(Error::InvalidRequest {
                field: "emitted",
                detail: format!(
                    "would release {next} completion token(s) with only {} accepted",
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
    /// to retain. The returned handle is the only way to name this result again.
    pub fn record_logits(&mut self, branch: BranchId, prefix: u64) -> Result<LogitsHandle> {
        let generation = self.generation;
        let sequence = self.id;
        let id = ResultId(self.next_result);
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
        let lineage = b.lineage[prefix as usize];
        let handle = LogitsHandle {
            id,
            sequence,
            branch,
            prefix,
            lineage,
            generation,
        };
        b.logits = Some(handle);
        self.next_result += 1;
        self.live.insert(id, handle);
        Ok(handle)
    }

    /// Whether a handle still names a result this sequence holds at a prefix
    /// whose lineage is unchanged.
    fn handle_is_live(&self, handle: LogitsHandle) -> Result<()> {
        if handle.sequence != self.id {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: format!(
                    "result belongs to sequence {} not {}; a prefix length is not an identity",
                    handle.sequence.0, self.id.0
                ),
            });
        }
        if handle.generation != self.generation {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: "result was produced under a different graph/state generation".into(),
            });
        }
        match self.live.get(&handle.id) {
            Some(known) if *known == handle => {}
            Some(_) => {
                return Err(Error::InvalidRequest {
                    field: "logits",
                    detail: format!("result {} does not match the one issued", handle.id.0),
                });
            }
            None => {
                return Err(Error::InvalidRequest {
                    field: "logits",
                    detail: format!(
                        "result {} was never issued by this sequence, or has been discarded",
                        handle.id.0
                    ),
                });
            }
        }
        let b = self.get(handle.branch)?;
        if handle.prefix > b.frontiers.executed {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: format!(
                    "result is at prefix {} but only {} token(s) are executed",
                    handle.prefix, b.frontiers.executed
                ),
            });
        }
        match b.lineage.get(handle.prefix as usize) {
            Some(l) if *l == handle.lineage => Ok(()),
            _ => Err(Error::InvalidRequest {
                field: "logits",
                detail: format!(
                    "prefix {} has been re-executed since this result was produced; \
                     the tokens at those positions are not the ones it saw",
                    handle.prefix
                ),
            }),
        }
    }

    /// Whether the next-token logits for `branch` are valid *right now*.
    ///
    /// True only when the branch holds a live result for exactly the accepted
    /// prefix, on this sequence, under the current generation, at an unchanged
    /// lineage. Equality of counters proves nothing.
    pub fn next_logits_valid(&self, branch: BranchId) -> bool {
        let Ok(b) = self.get(branch) else {
            return false;
        };
        match b.logits {
            Some(h) => {
                h.branch == branch
                    && h.prefix == b.frontiers.accepted
                    && self.handle_is_live(h).is_ok()
            }
            None => false,
        }
    }

    /// Continue from a result this sequence issued earlier.
    ///
    /// This is the "or use a saved equivalent forward result" half of document
    /// 04. Every provenance rule applies: the handle must be one this sequence
    /// minted, still live, on this branch, under the current generation, at a
    /// prefix whose lineage has not changed.
    pub fn restore_logits(&mut self, branch: BranchId, handle: LogitsHandle) -> Result<()> {
        if handle.branch != branch {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: format!("result belongs to {} not {branch}", handle.branch),
            });
        }
        self.handle_is_live(handle)?;
        self.get_mut(branch)?.logits = Some(handle);
        Ok(())
    }

    /// Declare that the executing configuration changed.
    ///
    /// Every retained result becomes stale and is discarded. Nothing survives:
    /// a result computed under a different graph, precision or positional
    /// configuration is a different computation, whatever its prefix.
    /// Refused while any transaction is open: it clears every branch's retained
    /// result, and no journal records what it removed.
    pub fn invalidate_generation(&mut self) -> Result<()> {
        if let Some((id, journal)) = self.open.iter().next() {
            return Err(Error::InvalidRequest {
                field: "generation",
                detail: format!(
                    "transaction {} is open on {}; resolve it before invalidating every \
                     retained result",
                    id.get(),
                    journal.branch
                ),
            });
        }
        self.generation = StateGeneration(self.generation.0 + 1);
        self.live.clear();
        for b in self.branches.values_mut() {
            b.logits = None;
        }
        Ok(())
    }

    /// The transaction open on `branch`, if any.
    ///
    /// A transaction is append-only, so every destructive operation asks this
    /// first. See [`Journal`] for why refusing beats journaling the discarded
    /// content.
    fn open_on(&self, branch: BranchId) -> Option<StateTransactionId> {
        self.open
            .iter()
            .find(|(_, j)| j.branch == branch)
            .map(|(id, _)| *id)
    }

    /// Open a transaction on `branch`.
    ///
    /// Document 04: "`begin` creates tentative state". Everything done between
    /// here and `commit_prefix` or `abort` can be undone exactly, which is what
    /// makes it safe for a step -- or a speculative branch -- to materialise
    /// work it has not accepted.
    ///
    /// At most one transaction may be open per branch.
    pub fn begin(&mut self, branch: BranchId) -> Result<StateTransactionId> {
        let b = self.get(branch)?;
        if let Some(id) = self.open_on(branch) {
            return Err(Error::InvalidRequest {
                field: "transaction",
                detail: format!(
                    "{branch} already has transaction {} open; two journals cannot both \
                     describe what to restore",
                    id.get()
                ),
            });
        }
        let journal = Journal {
            branch,
            frontiers: b.frontiers,
            lineage_len: b.lineage.len(),
            epoch: b.epoch,
            logits: b.logits,
            results_from: self.next_result,
        };
        let id = StateTransactionId(self.next_transaction);
        self.next_transaction += 1;
        self.open.insert(id, journal);
        Ok(id)
    }

    /// Publish `accept` token transitions and close the transaction.
    ///
    /// Document 04: "`commit_prefix(n)` publishes exactly n accepted token
    /// transitions". `n = 0` is the ordinary decode case -- the executor
    /// materialises state, and acceptance is the caller's decision after
    /// sampling.
    ///
    /// The tentative work done inside the transaction is kept.
    pub fn commit_prefix(&mut self, txn: StateTransactionId, accept: u64) -> Result<()> {
        let journal = self.open.get(&txn).cloned().ok_or(Error::InvalidRequest {
            field: "transaction",
            detail: format!("no open transaction {}", txn.get()),
        })?;
        if accept > 0 {
            // Accepting can overflow, and a failed commit must not close the
            // transaction -- the caller can still abort it.
            let b = self.get(journal.branch)?;
            b.frontiers
                .accepted
                .checked_add(accept)
                .ok_or(Error::InvalidRequest {
                    field: "accepted",
                    detail: "token counter overflow".into(),
                })?;
            self.accept(journal.branch, accept)?;
        }
        self.open.remove(&txn);
        Ok(())
    }

    /// Restore the branch to exactly its state at `begin`, and close the
    /// transaction.
    ///
    /// Document 04: "`abort` restores the committed prefix". Infallible apart
    /// from an unknown id: restoration is assignment and truncation, and nothing
    /// in it can run out of anything. That is what lets a caller abort on a
    /// failure path without a second failure to handle.
    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        let journal = self.open.remove(&txn).ok_or(Error::InvalidRequest {
            field: "transaction",
            detail: format!("no open transaction {}", txn.get()),
        })?;
        // Results minted inside the transaction describe work that is being
        // undone; they cannot survive it. Both halves of the predicate matter,
        // and the fifth review reproduced what each one costs when it is
        // missing:
        //
        // - **the branch.** `live` holds every branch's results. Filtering the
        //   whole table by id alone deleted results another branch had already
        //   committed, because those ids are also above this journal's mark.
        // - **the counter.** `next_result` is *not* restored. Rewinding it let
        //   the next attempt mint a handle equal in every field to one this
        //   abort discarded, which `restore_logits` then accepted.
        self.live
            .retain(|id, h| id.get() < journal.results_from || h.branch != journal.branch);
        let b = self
            .branches
            .get_mut(&journal.branch)
            .expect("a branch with an open transaction still exists");
        b.frontiers = journal.frontiers;
        // Truncation is a complete inverse because a transaction is
        // append-only: `rollback_to` is refused while one is open, so no entry
        // below `lineage_len` can have been replaced.
        b.lineage.truncate(journal.lineage_len);
        b.epoch = journal.epoch;
        b.logits = journal.logits;
        Ok(())
    }

    /// Transactions that were opened and never resolved.
    ///
    /// `Drop` cannot reach the state that owns a journal, so an unresolved
    /// transaction cannot abort itself. It is made *visible* instead: the branch
    /// refuses a further `begin`, and this reports what is outstanding. A leak
    /// the tests can assert on beats a silent corruption.
    pub fn open_transactions(&self) -> Vec<(StateTransactionId, BranchId)> {
        self.open.iter().map(|(id, j)| (*id, j.branch)).collect()
    }

    /// Mint restoration evidence for one `Explicit` component.
    ///
    /// Call this at the point the restoration actually happens; the identity it
    /// stamps -- sequence, branch, the lineage of the completed prefix, and the
    /// generation -- is what `rollback_to` checks again. See [`Restore`] for
    /// what that binding does and does not prove.
    pub fn restore_evidence(
        &self,
        kind: StateKind,
        branch: BranchId,
        method: RestoreMethod,
    ) -> Result<Restore> {
        if !self.schema.contains(&kind) {
            return Err(Error::InvalidRequest {
                field: "restore",
                detail: format!("{kind:?} is not part of this sequence's schema"),
            });
        }
        if kind.restore_capability() != RestoreCapability::Explicit {
            return Err(Error::InvalidRequest {
                field: "restore",
                detail: format!(
                    "{kind:?} is restored by truncation; evidence for it would mean \
                     something else was restored instead"
                ),
            });
        }
        if !method.is_coherent() {
            return Err(Error::InvalidRequest {
                field: "restore",
                detail: format!("{kind:?} replay runs backwards: {method:?}"),
            });
        }
        let prefix = method.completed_prefix();
        let lineage = self
            .lineage_at(branch, prefix)?
            .ok_or(Error::InvalidRequest {
                field: "restore",
                detail: format!(
                    "prefix {prefix} is not occupied on {branch}; there is no such state to \
                 have restored"
                ),
            })?;
        Ok(Restore {
            kind,
            sequence: self.id,
            branch,
            lineage,
            generation: self.generation,
            method,
        })
    }

    /// Roll back `branch` to an accepted prefix, as `abort` and a speculative
    /// rejection do.
    ///
    /// `restores` must cover every schema component whose [`RestoreCapability`]
    /// is `Explicit`, exactly once each, with evidence that finished **at**
    /// `prefix` -- not merely evidence that some earlier state is available.
    /// Truncatable components need no entry, and an entry for one is refused as
    /// a sign that the caller is confused about what it restored.
    ///
    /// Refused when it would move the accepted prefix below text already
    /// released to the client: published output cannot be unpublished by
    /// shortening a counter. Regenerating is a new response, not a rollback.
    ///
    /// Every result at a prefix beyond the target is discarded, and the branch's
    /// epoch advances so that re-executing those positions produces a different
    /// lineage. Results at or before the target survive, because those positions
    /// did not change.
    pub fn rollback_to(
        &mut self,
        branch: BranchId,
        prefix: u64,
        restores: &[Restore],
    ) -> Result<()> {
        if let Some(id) = self.open_on(branch) {
            return Err(Error::InvalidRequest {
                field: "prefix",
                detail: format!(
                    "{branch} has transaction {} open; a transaction is append-only, so \
                     resolve it before discarding executed positions",
                    id.get()
                ),
            });
        }
        let schema = self.schema.clone();
        let generation = self.generation;
        let sequence = self.id;
        // The lineage of the target prefix as it stands now. A rollback keeps
        // positions at or before the target, so evidence minted before this call
        // still matches -- unless those positions were replaced in between.
        let target_lineage = self.lineage_at(branch, prefix)?;
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
            let published_through = b.frontiers.prompt + b.frontiers.emitted;
            if prefix < published_through {
                return Err(Error::InvalidRequest {
                    field: "prefix",
                    detail: format!(
                        "cannot roll back to {prefix}: {} completion token(s) have already \
                         been released to the client, through prefix {published_through}. \
                         Published output cannot be unpublished; regeneration is a new response",
                        b.frontiers.emitted
                    ),
                });
            }
        }

        // Every restore must name a component of this schema, and no component
        // may be restored twice.
        let mut seen: BTreeSet<StateKind> = BTreeSet::new();
        for r in restores {
            if !schema.contains(&r.kind) {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!("{:?} is not part of this sequence's schema", r.kind),
                });
            }
            if r.kind.restore_capability() != RestoreCapability::Explicit {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} is restored by truncation; supplying evidence for it means \
                         something else was restored instead",
                        r.kind
                    ),
                });
            }
            if !seen.insert(r.kind) {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!("{:?} has more than one restore", r.kind),
                });
            }
            if r.sequence != sequence {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} evidence names sequence {}, not {}",
                        r.kind, r.sequence.0, sequence.0
                    ),
                });
            }
            if r.branch != branch {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} evidence was minted on {}, not {branch}. A snapshot of one \
                         branch is not a restoration of another, however alike their \
                         prefixes look; sharing one needs a stated equivalence rule",
                        r.kind, r.branch
                    ),
                });
            }
            if r.generation != generation {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} evidence is from generation {}, not the current {}",
                        r.kind, r.generation.0, generation.0
                    ),
                });
            }
            if !r.method.is_coherent() {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!("{:?} replay runs backwards: {:?}", r.kind, r.method),
                });
            }
            if r.method.completed_prefix() != prefix {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} restoration finished at prefix {}, not the rollback target \
                         {prefix}. A source at another prefix is not a completed restoration",
                        r.kind,
                        r.method.completed_prefix()
                    ),
                });
            }
            if Some(r.lineage) != target_lineage {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{:?} evidence describes a different version of prefix {prefix}: \
                         those positions have been replaced since it was minted",
                        r.kind
                    ),
                });
            }
        }
        for kind in &schema {
            if kind.restore_capability() == RestoreCapability::Explicit && !seen.contains(kind) {
                return Err(Error::InvalidRequest {
                    field: "restores",
                    detail: format!(
                        "{kind:?} cannot be restored by truncating a counter and no \
                         snapshot or replay was supplied"
                    ),
                });
            }
        }

        // If a transaction is open on this branch, the entries about to be
        // discarded are the ones an abort could not otherwise reconstruct.

        // Results beyond the target describe state that no longer exists.
        self.live
            .retain(|_, h| !(h.branch == branch && h.prefix > prefix));

        let b = self.get_mut(branch)?;
        b.frontiers.accepted = prefix;
        b.frontiers.executed = b.frontiers.executed.min(prefix);
        b.lineage.truncate(prefix as usize + 1);
        b.epoch += 1;
        if b.logits.is_some_and(|h| h.prefix > prefix) {
            b.logits = None;
        }
        Ok(())
    }

    /// Fork a copy-on-write branch sharing the prefix `at`.
    ///
    /// Used by speculation and by future-entropy lookahead. The child starts
    /// with no retained result even though it shares the prefix: a result is
    /// identified by its branch among other things, and quietly re-labelling the
    /// parent's result as the child's is the provenance shortcut this module
    /// exists to prevent.
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
            // Usage is per response, not per branch: a branch that is discarded
            // released nothing.
            emitted: 0,
            executed: p.frontiers.executed.min(at),
        };
        let lineage = p.lineage[..=at as usize].to_vec();
        let epoch = p.epoch;
        let child = BranchId(self.next_branch);
        self.next_branch += 1;
        self.branches.insert(
            child,
            Branch {
                frontiers,
                parent: Some(parent),
                lineage,
                epoch,
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
        if let Some((id, _)) = self.open.iter().find(|(_, j)| j.branch == branch) {
            return Err(Error::InvalidRequest {
                field: "branch",
                detail: format!(
                    "{branch} has transaction {} open; resolve it before discarding the \
                     branch its journal describes",
                    id.get()
                ),
            });
        }
        if self.branches.remove(&branch).is_none() {
            return Err(Error::InvalidRequest {
                field: "branch",
                detail: format!("no such branch {branch}"),
            });
        }
        self.live.retain(|_, h| h.branch != branch);
        Ok(())
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

    fn kv_only() -> SequenceState {
        SequenceState::new([StateKind::KvPages, StateKind::PositionCounter])
    }

    fn full_schema() -> SequenceState {
        SequenceState::new(StateKind::ALL.iter().copied())
    }

    /// Snapshot evidence for every `Explicit` component, finishing at `prefix`.
    fn snapshots_at(s: &SequenceState, branch: BranchId, prefix: u64) -> Vec<Restore> {
        s.schema()
            .iter()
            .filter(|k| k.restore_capability() == RestoreCapability::Explicit)
            .map(|k| {
                s.restore_evidence(*k, branch, RestoreMethod::Snapshot { of_prefix: prefix })
                    .expect("evidence for a schema component at an occupied prefix")
            })
            .collect()
    }

    /// A full-schema state advanced to `accepted`/`executed` of 8.
    fn advanced_full_schema() -> SequenceState {
        let mut s = full_schema();
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 8).unwrap();
        s
    }

    #[test]
    fn empty_state_has_no_logits() {
        // The plainest form of the F2 defect: `Frontiers::new()` used to report
        // valid next-token logits before any forward pass had run.
        let s = kv_only();
        assert!(!s.next_logits_valid(ROOT));
        assert_eq!(s.retained_logits(ROOT).unwrap(), None);
        assert_eq!(s.frontiers(ROOT).unwrap(), Frontiers::default());
        assert!(s.live_results().is_empty());
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
    fn a_result_from_another_sequence_is_refused() {
        // Second review, reproduced: two fresh sequences share the root branch
        // and generation zero, so a `(branch, prefix, generation)` triple made a
        // handle from A valid in B. Both had `next_logits_valid == true`.
        let mut a = kv_only();
        a.append_prompt(ROOT, 5).unwrap();
        a.execute(ROOT, 5).unwrap();
        let from_a = a.record_logits(ROOT, 5).unwrap();

        let mut b = kv_only();
        b.append_prompt(ROOT, 5).unwrap();
        b.execute(ROOT, 5).unwrap();

        assert_ne!(a.id(), b.id());
        let e = b.restore_logits(ROOT, from_a).unwrap_err();
        assert!(e.to_string().contains("sequence"), "{e}");
        assert!(!b.next_logits_valid(ROOT));

        // Identical counters on both sides; only the identity differs.
        assert_eq!(a.frontiers(ROOT).unwrap(), b.frontiers(ROOT).unwrap());
    }

    #[test]
    fn a_result_for_a_replaced_suffix_does_not_come_back() {
        // Second review, reproduced: rolling back and re-executing to the same
        // length made the discarded result valid again, because a prefix length
        // is not a prefix identity.
        let mut s = kv_only();
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 8).unwrap();
        let at_eight = s.record_logits(ROOT, 8).unwrap();
        assert!(s.next_logits_valid(ROOT));

        s.rollback_to(ROOT, 4, &[]).unwrap();
        s.accept(ROOT, 4).unwrap(); // four *different* tokens
        s.execute(ROOT, 4).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().accepted, 8);
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 8);

        let e = s.restore_logits(ROOT, at_eight).unwrap_err();
        assert!(!s.next_logits_valid(ROOT));
        assert!(
            e.to_string().contains("never issued") || e.to_string().contains("re-executed"),
            "{e}"
        );

        // The lineage of prefix 8 really did change; prefix 4 did not.
        let mut fresh = kv_only();
        fresh.append_prompt(ROOT, 4).unwrap();
        fresh.accept(ROOT, 4).unwrap();
        fresh.execute(ROOT, 8).unwrap();
        assert_ne!(
            s.lineage_at(ROOT, 8).unwrap(),
            fresh.lineage_at(ROOT, 8).unwrap()
        );
    }

    #[test]
    fn a_result_at_an_unchanged_prefix_survives_a_rollback() {
        // The other half of the requirement: replacing a suffix must not
        // invalidate results for prefixes it did not touch.
        let mut s = kv_only();
        s.append_prompt(ROOT, 8).unwrap();
        s.accept(ROOT, 4).unwrap(); // accepted prefix 12
        s.execute(ROOT, 12).unwrap();
        let at_twelve = s.record_logits(ROOT, 12).unwrap();
        assert!(s.next_logits_valid(ROOT));

        s.accept(ROOT, 8).unwrap(); // accepted prefix 20
        s.execute(ROOT, 8).unwrap();
        s.record_logits(ROOT, 20).unwrap();
        assert!(s.next_logits_valid(ROOT));

        s.rollback_to(ROOT, 12, &[]).unwrap();
        assert!(!s.next_logits_valid(ROOT), "the prefix-20 result is gone");
        assert_eq!(s.retained_logits(ROOT).unwrap(), None);

        // The prefix-12 result was genuinely recorded earlier and is still live.
        s.restore_logits(ROOT, at_twelve).unwrap();
        assert!(s.next_logits_valid(ROOT));
        assert_eq!(s.live_results().len(), 1);
    }

    #[test]
    fn a_handle_cannot_be_fabricated() {
        // `LogitsHandle`'s fields are private and only `record_logits` mints one,
        // so a test cannot assert its way past provenance. What is checkable
        // here is that a handle whose result has been dropped stops working.
        let mut s = kv_only();
        s.append_prompt(ROOT, 6).unwrap();
        s.execute(ROOT, 6).unwrap();
        let h = s.record_logits(ROOT, 6).unwrap();
        assert!(s.restore_logits(ROOT, h).is_ok());

        let child = s.fork(ROOT, 6).unwrap();
        s.execute(child, 2).unwrap();
        let child_result = s.record_logits(child, 8).unwrap();
        s.discard_branch(child).unwrap();
        assert!(s.restore_logits(ROOT, child_result).is_err());
        assert_eq!(s.live_results().len(), 1);
    }

    #[test]
    fn a_committed_bonus_token_is_pending_execution_and_invalidates_the_logits() {
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
        let mut s = kv_only();
        s.append_prompt(ROOT, 100).unwrap();
        s.execute(ROOT, 100).unwrap();

        let draft = s.fork(ROOT, 100).unwrap();
        s.execute(draft, 4).unwrap(); // four unverified proposals
        let f = s.frontiers(draft).unwrap();
        assert_eq!(f.tentative(), 4);
        assert_eq!(f.accepted, 100);
        assert_eq!(f.emitted, 0, "nothing was released to the client");
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
    fn published_output_cannot_be_unpublished_by_a_rollback() {
        // Second review, reproduced: accepting and releasing one completion
        // token, then rolling back, silently reset `emitted` to zero. The client
        // already has that text.
        let mut s = kv_only();
        s.append_prompt(ROOT, 3).unwrap();
        s.accept(ROOT, 2).unwrap();
        s.execute(ROOT, 5).unwrap();
        s.emit(ROOT, 1).unwrap();

        let e = s.rollback_to(ROOT, 3, &[]).unwrap_err();
        assert!(e.to_string().contains("already"), "{e}");
        assert_eq!(s.frontiers(ROOT).unwrap().emitted, 1);

        // Rolling back over *unreleased* completion tokens is still fine.
        s.rollback_to(ROOT, 4, &[]).unwrap();
        let f = s.frontiers(ROOT).unwrap();
        assert_eq!(f.accepted, 4);
        assert_eq!(f.emitted, 1);
        assert_eq!(f.completion(), 1);
    }

    #[test]
    fn usage_counts_committed_completion_tokens_not_released_ones() {
        // Document 05 bills "prompt tokens and committed completion tokens
        // only". Stop-string withholding makes released text lag behind, and
        // the two numbers must not be conflated.
        let mut s = kv_only();
        s.append_prompt(ROOT, 7).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 11).unwrap();
        s.emit(ROOT, 2).unwrap(); // two more held while a stop string forms

        let f = s.frontiers(ROOT).unwrap();
        assert_eq!(
            f.completion(),
            4,
            "usage counts committed completion tokens"
        );
        assert_eq!(f.emitted, 2, "the client has seen two");
        assert_eq!(f.withheld(), 2);
        assert_ne!(f.completion(), f.emitted);
    }

    #[test]
    fn rejection_at_every_depth_leaves_consistent_state_and_no_logits() {
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
    fn a_stale_or_foreign_saved_result_is_refused() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 8).unwrap();
        s.execute(ROOT, 8).unwrap();
        let handle = s.record_logits(ROOT, 8).unwrap();

        // Configuration change: same counters, different model.
        s.invalidate_generation().unwrap();
        assert!(!s.next_logits_valid(ROOT));
        assert!(
            s.restore_logits(ROOT, handle).is_err(),
            "a result from the previous generation must not be re-blessed"
        );
        assert!(s.live_results().is_empty());

        // A result from another branch, presented for this one.
        s.execute(ROOT, 0).unwrap();
        let other = s.fork(ROOT, 8).unwrap();
        s.execute(other, 1).unwrap();
        let other_result = s.record_logits(other, 9).unwrap();
        assert!(s.restore_logits(ROOT, other_result).is_err());
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

        // A replay that ran to the target restores it.
        let e = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 8, to: 10 },
            )
            .unwrap();
        s.rollback_to(ROOT, 10, &[e]).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().accepted, 10);
    }

    #[test]
    fn an_earlier_source_is_not_a_completed_restoration() {
        // Second review, reproduced: a snapshot taken at prefix 4 was accepted
        // as restoration to prefix 6, and the execution frontier advanced to 6
        // over state that stopped at 4.
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 2).unwrap();
        s.accept(ROOT, 8).unwrap();
        s.execute(ROOT, 10).unwrap();

        let early = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 4 },
            )
            .unwrap();
        let e = s.rollback_to(ROOT, 6, &[early]).unwrap_err();
        assert!(e.to_string().contains("finished at prefix 4"), "{e}");
        assert_eq!(
            s.frontiers(ROOT).unwrap().executed,
            10,
            "a refused rollback changes nothing"
        );

        // A replay whose source is earlier but which ran *to* the target is
        // exactly what the earlier snapshot was missing.
        let replayed = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 4, to: 6 },
            )
            .unwrap();
        s.rollback_to(ROOT, 6, &[replayed]).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 6);

        // A replay that runs backwards is incoherent.
        let mut t = SequenceState::new([StateKind::RecurrentAccumulator]);
        t.append_prompt(ROOT, 2).unwrap();
        t.accept(ROOT, 8).unwrap();
        t.execute(ROOT, 10).unwrap();
        assert!(
            t.restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 9, to: 6 },
            )
            .is_err(),
            "a backwards replay is incoherent and cannot even be minted"
        );
    }

    #[test]
    fn restore_evidence_must_name_this_sequence_and_generation() {
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 2).unwrap();
        s.accept(ROOT, 6).unwrap();
        s.execute(ROOT, 8).unwrap();

        let method = RestoreMethod::Snapshot { of_prefix: 6 };
        let good = s
            .restore_evidence(StateKind::RecurrentAccumulator, ROOT, method)
            .unwrap();

        // Evidence minted by a different sequence, at the same prefix.
        let mut other = SequenceState::new([StateKind::RecurrentAccumulator]);
        other.append_prompt(ROOT, 2).unwrap();
        other.accept(ROOT, 6).unwrap();
        other.execute(ROOT, 8).unwrap();
        let foreign = other
            .restore_evidence(StateKind::RecurrentAccumulator, ROOT, method)
            .unwrap();
        assert_ne!(s.id(), other.id());
        assert!(
            s.rollback_to(ROOT, 6, &[foreign]).is_err(),
            "evidence from another sequence is not evidence about this one"
        );

        // Evidence minted before a configuration change is stale after it.
        let mut stale_holder = SequenceState::new([StateKind::RecurrentAccumulator]);
        stale_holder.append_prompt(ROOT, 2).unwrap();
        stale_holder.accept(ROOT, 6).unwrap();
        stale_holder.execute(ROOT, 8).unwrap();
        let before = stale_holder
            .restore_evidence(StateKind::RecurrentAccumulator, ROOT, method)
            .unwrap();
        stale_holder.invalidate_generation().unwrap();
        assert!(
            stale_holder.rollback_to(ROOT, 6, &[before]).is_err(),
            "a snapshot from a different configuration is stale"
        );

        s.rollback_to(ROOT, 6, &[good]).unwrap();
    }

    #[test]
    fn every_explicit_kind_in_the_schema_must_be_covered_exactly_once() {
        // Each case builds its own state: `SequenceState` is not `Clone`, and
        // copying one for test convenience is what defeated identity before.
        let base = advanced_full_schema();
        let full = snapshots_at(&base, ROOT, 8);
        assert_eq!(full.len(), 5, "five explicit kinds in the full schema");

        // Drop one at a time: each omission must be refused by name.
        for i in 0..full.len() {
            let mut s = advanced_full_schema();
            let mut partial = snapshots_at(&s, ROOT, 8);
            let missing = partial.remove(i);
            let e = s.rollback_to(ROOT, 8, &partial).unwrap_err();
            assert!(
                e.to_string().contains(&format!("{:?}", missing.kind())),
                "omitting {:?} was not reported: {e}",
                missing.kind()
            );
        }

        // A duplicate is a sign the caller does not know what it restored.
        let mut s = advanced_full_schema();
        let mut duplicated = snapshots_at(&s, ROOT, 8);
        duplicated.push(duplicated[0]);
        assert!(s.rollback_to(ROOT, 8, &duplicated).is_err());

        // Evidence for a truncatable component cannot even be minted.
        let s = advanced_full_schema();
        assert!(
            s.restore_evidence(
                StateKind::KvPages,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 8 }
            )
            .is_err()
        );

        let mut s = advanced_full_schema();
        let full = snapshots_at(&s, ROOT, 8);
        s.rollback_to(ROOT, 8, &full).unwrap();
    }

    #[test]
    fn restore_evidence_from_another_branch_is_refused() {
        // Third review, reproduced: root-branch snapshot evidence satisfied a
        // rollback on a child branch, because branch identity was absent.
        // Sequence, generation and prefix length were all equal.
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 8).unwrap();

        let from_root = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 6 },
            )
            .unwrap();

        let child = s.fork(ROOT, 8).unwrap();
        s.accept(child, 2).unwrap();
        s.execute(child, 2).unwrap();

        let e = s.rollback_to(child, 6, &[from_root]).unwrap_err();
        assert!(e.to_string().contains("minted on"), "{e}");
        assert_eq!(
            s.frontiers(child).unwrap().executed,
            10,
            "a refused rollback changes nothing"
        );

        // The child's own evidence works, so the rule is about identity rather
        // than about forks being unrollbackable.
        let own = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                child,
                RestoreMethod::Snapshot { of_prefix: 6 },
            )
            .unwrap();
        s.rollback_to(child, 6, &[own]).unwrap();
        assert_eq!(s.frontiers(child).unwrap().executed, 6);
    }

    #[test]
    fn restore_evidence_for_a_replaced_suffix_is_refused() {
        // Third review, reproduced: evidence was reused after rolling back and
        // replacing the very suffix it described. Same sequence, same branch,
        // same generation, same prefix length.
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 6).unwrap();
        s.execute(ROOT, 10).unwrap();

        let stale = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 6, to: 10 },
            )
            .unwrap();

        // Replace positions 8 and 9 with different tokens.
        let to_eight = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 8 },
            )
            .unwrap();
        s.rollback_to(ROOT, 8, &[to_eight]).unwrap();
        s.accept(ROOT, 2).unwrap();
        s.execute(ROOT, 2).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 10);

        let e = s.rollback_to(ROOT, 10, &[stale]).unwrap_err();
        assert!(e.to_string().contains("different version"), "{e}");

        // Freshly minted evidence for the prefix that now exists is accepted.
        let current = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 8, to: 10 },
            )
            .unwrap();
        s.rollback_to(ROOT, 10, &[current]).unwrap();
    }

    #[test]
    fn evidence_survives_a_rollback_that_does_not_touch_its_prefix() {
        // The positive half: minting evidence, then rolling back *to* that
        // prefix, must work. Positions at or before the target keep their
        // lineage, so a legitimate flow -- snapshot, then abort back to it --
        // is not caught by the replacement rule.
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 2).unwrap();
        s.accept(ROOT, 8).unwrap();
        s.execute(ROOT, 10).unwrap();
        let e = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 6 },
            )
            .unwrap();
        s.rollback_to(ROOT, 6, &[e]).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 6);
    }

    #[test]
    fn a_restore_for_a_component_outside_the_schema_is_refused() {
        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 2).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 6).unwrap();
        assert!(
            s.restore_evidence(
                StateKind::ConvolutionHistory,
                ROOT,
                RestoreMethod::Snapshot { of_prefix: 4 }
            )
            .is_err(),
            "a component outside the schema has no state to restore"
        );
        let _ = &mut s;
    }

    #[test]
    fn abort_restores_every_counter_including_executed() {
        // The thing that was impossible before task 0004: `executed` had no
        // inverse, so a failure after it could not be unwound and correctness
        // rested on having enumerated every later failure.
        let mut s = kv_only();
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 2).unwrap();
        s.execute(ROOT, 6).unwrap();
        s.emit(ROOT, 1).unwrap();
        s.record_logits(ROOT, 6).unwrap();
        let before = s.frontiers(ROOT).unwrap();
        let before_logits = s.retained_logits(ROOT).unwrap();
        let before_lineage = s.lineage_at(ROOT, 6).unwrap();
        let before_live: Vec<u64> = s.live_results().iter().map(|h| h.prefix()).collect();

        let txn = s.begin(ROOT).unwrap();
        s.execute(ROOT, 3).unwrap();
        s.accept(ROOT, 3).unwrap();
        // Not `emit`. The fifth review reproduced what an earlier version of
        // this test enshrined: emitting inside a transaction and restoring the
        // counter on abort is a retraction of output the client already has.
        // `emitted` is in `before` and must come back **because it never
        // moved**, which is a different guarantee from being rewound.
        let e = s.emit(ROOT, 2).unwrap_err();
        assert!(e.to_string().contains("cannot be released"), "{e}");
        s.record_logits(ROOT, 9).unwrap();
        assert_ne!(s.frontiers(ROOT).unwrap(), before, "the fixture must move");

        s.abort(txn).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap(), before);
        assert_eq!(s.retained_logits(ROOT).unwrap(), before_logits);
        assert_eq!(s.lineage_at(ROOT, 6).unwrap(), before_lineage);
        assert_eq!(
            s.live_results()
                .iter()
                .map(|h| h.prefix())
                .collect::<Vec<_>>(),
            before_live,
            "results minted inside the transaction describe undone work"
        );
        assert!(
            s.lineage_at(ROOT, 7).unwrap().is_none(),
            "lineage truncated"
        );
        assert!(s.next_logits_valid(ROOT));
    }

    #[test]
    fn aborting_one_branch_leaves_another_branch_s_results_alone() {
        // Fifth review, reproduced: `live` holds every branch's results, and
        // filtering it by result id alone deleted results a *different* branch
        // had already committed, because their ids are also above this
        // journal's mark.
        let mut s = kv_only();
        s.append_prompt(ROOT, 2).unwrap();
        s.execute(ROOT, 2).unwrap();
        let child = s.fork(ROOT, 2).unwrap();

        let t_root = s.begin(ROOT).unwrap();
        let t_child = s.begin(child).unwrap();
        s.execute(child, 1).unwrap();
        let h = s.record_logits(child, 3).unwrap();
        s.commit_prefix(t_child, 0).unwrap();
        assert!(s.restore_logits(child, h).is_ok());

        s.abort(t_root).unwrap();
        assert!(
            s.restore_logits(child, h).is_ok(),
            "the child committed this result; the root's abort does not describe it"
        );
    }

    #[test]
    fn an_aborted_result_identity_is_never_reissued() {
        // Fifth review, reproduced: `abort` used to restore `next_result`, so
        // re-executing the same prefix minted a handle equal in every field to
        // the discarded one -- and `restore_logits` accepted it. A restored
        // logical state must not recycle the identities of discarded results.
        let mut s = kv_only();
        s.append_prompt(ROOT, 2).unwrap();

        let t1 = s.begin(ROOT).unwrap();
        s.execute(ROOT, 2).unwrap();
        let discarded = s.record_logits(ROOT, 2).unwrap();
        s.abort(t1).unwrap();
        assert!(s.restore_logits(ROOT, discarded).is_err());

        let t2 = s.begin(ROOT).unwrap();
        s.execute(ROOT, 2).unwrap();
        let fresh = s.record_logits(ROOT, 2).unwrap();
        s.commit_prefix(t2, 0).unwrap();

        assert_ne!(fresh.id(), discarded.id());
        assert_ne!(fresh, discarded);
        assert!(
            s.restore_logits(ROOT, discarded).is_err(),
            "the discarded handle stays dead after a successful retry"
        );
    }

    #[test]
    fn tentative_work_cannot_be_released_to_the_client() {
        // Fifth review, reproduced: `emit` inside a transaction moved `emitted`
        // from 0 to 1 and `abort` moved it back -- a restored counter
        // describing text the client already has. Emission is the one mutation
        // that leaves the process, so it is refused for the duration instead.
        let mut s = kv_only();
        s.append_prompt(ROOT, 1).unwrap();
        s.execute(ROOT, 1).unwrap();
        s.accept(ROOT, 1).unwrap();

        let txn = s.begin(ROOT).unwrap();
        let e = s.emit(ROOT, 1).unwrap_err();
        assert!(e.to_string().contains("cannot be released"), "{e}");
        s.abort(txn).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().emitted, 0);

        // And it works once the transaction is resolved, in either direction.
        s.emit(ROOT, 1).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().emitted, 1);
        let txn = s.begin(ROOT).unwrap();
        s.commit_prefix(txn, 0).unwrap();
        s.emit(ROOT, 0).unwrap();
    }

    #[test]
    fn a_transaction_is_append_only() {
        // The fifth review reproduced both halves of this. A rollback inside a
        // transaction discards lineage entries *and* live results, and an
        // earlier journal restored the first but not the second -- so aborting
        // left a branch whose counters said one thing and whose retained result
        // was gone. Journaling every discarded thing is possible; refusing the
        // destructive operation for the duration is exact and cheap, and
        // document 04's way to keep part of a transaction's work is
        // `commit_prefix(n)`, not a rollback inside it.
        let mut s = kv_only();
        s.append_prompt(ROOT, 2).unwrap();
        s.accept(ROOT, 4).unwrap();
        s.execute(ROOT, 6).unwrap();
        let kept = s.record_logits(ROOT, 6).unwrap();
        let lineage_before = s.lineage_at(ROOT, 6).unwrap();

        let txn = s.begin(ROOT).unwrap();
        let e = s.rollback_to(ROOT, 3, &[]).unwrap_err();
        assert!(e.to_string().contains("append-only"), "{e}");
        let e = s.invalidate_generation().unwrap_err();
        assert!(e.to_string().contains("is open"), "{e}");
        s.execute(ROOT, 3).unwrap();
        s.abort(txn).unwrap();

        // Nothing below the journal's mark could have been replaced, so
        // truncation is a complete inverse and the retained result survives.
        assert_eq!(s.lineage_at(ROOT, 6).unwrap(), lineage_before);
        assert!(s.restore_logits(ROOT, kept).is_ok());
        assert!(s.next_logits_valid(ROOT));

        // Both operations work again once the transaction is resolved.
        s.rollback_to(ROOT, 3, &[]).unwrap();
        s.invalidate_generation().unwrap();
    }

    #[test]
    fn commit_prefix_keeps_the_work_and_publishes_what_it_is_told_to() {
        // Document 04: "commit_prefix(n) publishes exactly n accepted token
        // transitions". n = 0 is the ordinary decode case -- the executor
        // materialises state and acceptance is the caller's later decision.
        let mut s = kv_only();
        s.append_prompt(ROOT, 3).unwrap();

        let txn = s.begin(ROOT).unwrap();
        s.execute(ROOT, 3).unwrap();
        s.record_logits(ROOT, 3).unwrap();
        s.commit_prefix(txn, 0).unwrap();
        let f = s.frontiers(ROOT).unwrap();
        assert_eq!((f.accepted, f.executed), (3, 3));
        assert!(s.next_logits_valid(ROOT), "the tentative work was kept");
        assert!(s.open_transactions().is_empty());

        // And a commit that accepts publishes exactly that many.
        let txn = s.begin(ROOT).unwrap();
        s.execute(ROOT, 2).unwrap();
        s.commit_prefix(txn, 2).unwrap();
        let f = s.frontiers(ROOT).unwrap();
        assert_eq!((f.accepted, f.executed), (5, 5));
        assert_eq!(f.completion(), 2);
    }

    #[test]
    fn a_branch_has_at_most_one_open_transaction() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 1).unwrap();
        let txn = s.begin(ROOT).unwrap();
        let e = s.begin(ROOT).unwrap_err();
        assert!(e.to_string().contains("already has transaction"), "{e}");

        // A different branch may have its own.
        let child = s.fork(ROOT, 1).unwrap();
        let other = s.begin(child).unwrap();
        assert_eq!(s.open_transactions().len(), 2);

        s.abort(txn).unwrap();
        s.commit_prefix(other, 0).unwrap();
        assert!(s.open_transactions().is_empty());
        // ... and the branch is free again.
        assert!(s.begin(ROOT).is_ok());
    }

    #[test]
    fn an_unknown_or_resolved_transaction_is_refused() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 1).unwrap();
        assert!(s.abort(StateTransactionId(999)).is_err());
        assert!(s.commit_prefix(StateTransactionId(999), 0).is_err());

        let txn = s.begin(ROOT).unwrap();
        s.commit_prefix(txn, 0).unwrap();
        assert!(s.abort(txn).is_err(), "already resolved");
        assert!(s.commit_prefix(txn, 0).is_err());
    }

    #[test]
    fn an_unresolved_transaction_is_visible_rather_than_silent() {
        // `Drop` cannot reach the state that owns the journal, so an unresolved
        // transaction cannot abort itself. It is made detectable instead: the
        // branch is locked and the leak is reportable.
        let mut s = kv_only();
        s.append_prompt(ROOT, 1).unwrap();
        let txn = s.begin(ROOT).unwrap();
        s.execute(ROOT, 1).unwrap();
        // The id is `Copy`, so "dropping" it is not what leaks -- simply never
        // resolving it is. That is the case being tested.
        let _ = txn;
        assert_eq!(s.open_transactions().len(), 1);
        assert_eq!(s.open_transactions()[0].1, ROOT);
        assert!(s.begin(ROOT).is_err(), "the branch stays locked");
    }

    #[test]
    fn a_failed_commit_leaves_the_transaction_open_to_abort() {
        // A commit that cannot publish must not close the transaction, or the
        // caller loses its only way back.
        let mut s = kv_only();
        s.append_prompt(ROOT, 1).unwrap();
        let b = s.branches.get_mut(&ROOT).unwrap();
        b.frontiers.accepted = u64::MAX - 1;
        let txn = s.begin(ROOT).unwrap();
        assert!(s.commit_prefix(txn, 5).is_err(), "overflow");
        assert_eq!(s.open_transactions().len(), 1);
        s.abort(txn).unwrap();
        assert!(s.open_transactions().is_empty());
    }

    #[test]
    fn a_branch_with_an_open_transaction_cannot_be_discarded() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 2).unwrap();
        let child = s.fork(ROOT, 2).unwrap();
        let txn = s.begin(child).unwrap();
        assert!(s.discard_branch(child).is_err());
        s.abort(txn).unwrap();
        assert!(s.discard_branch(child).is_ok());
    }

    #[test]
    fn abort_restores_after_every_prefix_of_a_step_s_mutations() {
        // The generalisation of the four holes four review passes found: whatever
        // subset of a step's work has happened, aborting puts all of it back.
        let baseline = |s: &SequenceState| {
            (
                s.frontiers(ROOT).unwrap(),
                s.retained_logits(ROOT).unwrap(),
                s.live_results().len(),
                s.lineage_at(ROOT, 4).unwrap(),
            )
        };
        for stop_after in 0..4 {
            let mut s = kv_only();
            s.append_prompt(ROOT, 2).unwrap();
            s.accept(ROOT, 2).unwrap();
            s.execute(ROOT, 4).unwrap();
            s.record_logits(ROOT, 4).unwrap();
            let before = baseline(&s);

            let txn = s.begin(ROOT).unwrap();
            if stop_after > 0 {
                s.execute(ROOT, 2).unwrap();
            }
            if stop_after > 1 {
                s.record_logits(ROOT, 6).unwrap();
            }
            if stop_after > 2 {
                s.accept(ROOT, 2).unwrap();
            }
            s.abort(txn).unwrap();
            assert_eq!(baseline(&s), before, "stopped after {stop_after}");
            assert!(s.next_logits_valid(ROOT), "stopped after {stop_after}");
        }
    }

    #[test]
    fn a_sparse_index_and_sampler_history_are_not_truncatable() {
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
            (state.wrapping_mul(1_000_003).wrapping_add(token)) % 1_000_000_007
        }
        let tokens: Vec<u64> = (1..=12).collect();
        let mut states = vec![0u64];
        for t in &tokens {
            let next = step(*states.last().unwrap(), *t);
            states.push(next);
        }
        // Saved snapshot at prefix 8, accepted prefix after rejection is 10.
        let mut replayed = states[8];
        for t in &tokens[8..10] {
            replayed = step(replayed, *t);
        }
        assert_eq!(replayed, states[10]);
        // ... and the snapshot alone, without the replay, is a different state.
        assert_ne!(states[8], states[10]);

        let mut s = SequenceState::new([StateKind::RecurrentAccumulator]);
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 8).unwrap();
        s.execute(ROOT, 12).unwrap();
        let e = s
            .restore_evidence(
                StateKind::RecurrentAccumulator,
                ROOT,
                RestoreMethod::Replay { from: 8, to: 10 },
            )
            .unwrap();
        s.rollback_to(ROOT, 10, &[e]).unwrap();
        assert_eq!(s.frontiers(ROOT).unwrap().executed, 10);
    }

    #[test]
    fn an_entropy_branch_leaves_its_parent_unchanged() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 30).unwrap();
        s.execute(ROOT, 30).unwrap();
        s.record_logits(ROOT, 30).unwrap();
        let before = s.frontiers(ROOT).unwrap();
        let before_logits = s.retained_logits(ROOT).unwrap();
        let before_lineage = s.lineage_at(ROOT, 30).unwrap();

        let mut children = Vec::new();
        for _ in 0..4 {
            let c = s.fork(ROOT, 30).unwrap();
            s.execute(c, 1).unwrap(); // one candidate token each
            s.record_logits(c, 31).unwrap();
            children.push(c);
        }
        assert_eq!(s.frontiers(ROOT).unwrap(), before);
        assert_eq!(s.retained_logits(ROOT).unwrap(), before_logits);
        assert_eq!(s.lineage_at(ROOT, 30).unwrap(), before_lineage);
        assert!(s.next_logits_valid(ROOT));

        for c in children {
            s.discard_branch(c).unwrap();
        }
        assert_eq!(s.branch_ids(), vec![ROOT]);
        assert_eq!(s.frontiers(ROOT).unwrap(), before);
        assert!(s.next_logits_valid(ROOT));
        assert_eq!(s.live_results().len(), 1, "the children's results are gone");
    }

    #[test]
    fn a_fork_does_not_inherit_the_parents_retained_result() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 6).unwrap();
        s.execute(ROOT, 6).unwrap();
        let parent_result = s.record_logits(ROOT, 6).unwrap();

        let child = s.fork(ROOT, 6).unwrap();
        assert_eq!(s.retained_logits(child).unwrap(), None);
        assert!(!s.next_logits_valid(child));
        assert_eq!(s.parent_of(child).unwrap(), Some(ROOT));
        // ... and the parent's result cannot be re-labelled as the child's.
        assert!(s.restore_logits(child, parent_result).is_err());
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
        assert!(s.frontiers(BranchId(999)).is_err());
        assert!(s.lineage_at(BranchId(999), 0).is_err());
    }

    #[test]
    fn counter_overflow_is_an_error_not_a_wrap() {
        // Checked separately from the boundary cases: `u64::MAX` positions would
        // make the lineage vector unrepresentable, so this uses a fresh state
        // and only exercises the arithmetic.
        let mut s = kv_only();
        s.append_prompt(ROOT, 4).unwrap();
        let b = s.branches.get_mut(&ROOT).unwrap();
        b.frontiers.accepted = u64::MAX - 2;
        assert!(s.accept(ROOT, 5).is_err());
        let b = s.branches.get_mut(&ROOT).unwrap();
        b.frontiers.executed = u64::MAX;
        assert!(s.execute(ROOT, 1).is_err());
    }

    #[test]
    fn the_prompt_cannot_grow_under_generated_text() {
        let mut s = kv_only();
        s.append_prompt(ROOT, 4).unwrap();
        s.accept(ROOT, 1).unwrap();
        assert!(s.append_prompt(ROOT, 1).is_err());
    }
}
