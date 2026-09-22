# ADR 0036 — A tensor-parallel reduction is exact: its accumulation order is part of the contract

- **ID / date / author / status:** 0036 / 2026-09-22 / recorded by the coordinator on the owner's ruling / **accepted**
- **Classification:** **owner requirement.** AGENTS.md reserves declaring a numerical gate to the owner. The owner was offered a tolerance (reusing ADR 0028's gate) and chose this instead: "my decision is strata approach of course."
- **Scope and owning shared component:** every operation that M5 splits along its reduction axis, where each rank computes a partial sum and the ranks add them. Today that means a row-parallel `Linear` (`o_proj`, `down_proj`). It also governs any later reduction that TP, PP or expert partitioning introduces, and the single-rank path those are compared against.
- **Supersedes:** nothing. ADR 0028 still governs a kernel compared against an FP64/FP32 host oracle. This ADR governs a partitioned execution compared against the unpartitioned one.

## Problem and mechanism

A column-parallel split, or a split by whole heads, computes disjoint output
elements on each rank and concatenates them. It is bit-identical to one rank
by construction (tasks 0056 and 0057). A row-parallel split is different: each
rank sums over its slice of the input axis and the partials are added. FP32
addition is commutative but not associative, so adding the partials does not
reproduce a single sequential sum over the whole axis. An unmodified
single-rank reference and a TP run would therefore differ in the last bits.

## Options examined

- **A tolerance** (ADR 0028's two clauses applied to TP). The single-rank path
  stays as it is, and TP passes if it lands within 2 BF16 ULP or within
  `2^-8 · Σ|x·W|`. This was rejected. It hides TP defects smaller than the
  tolerance and lets generated text drift from single-rank output. Legacy
  Strata measured exactly that drift: a reassociated expert kernel "held to
  5.960e-07 rather than to an output hash … eventually flips an argmax, measured
  at token 7 of a 32-token generation" (`strata/docs/models/glm53.md`).
- **Declared accumulation order, Strata's approach, chosen.** Legacy Strata's
  DeepSeek TP2 path (`strata/docs/dsv4-rank-local-architecture.md`, "Tensor
  sharding", "Collectives", "NUMA and CPU") computed "arithmetic, route,
  coefficients, precision, and accumulation order … identical to the
  centralized path — only the ownership partition differs". Its centralized
  path produced the same two per-shard FP32 partials and added them. The rank
  partials crossed as FP32 (`ncclSum` over `ncclFloat`, "exact-order for the
  declared contract"), and the gathered logits were byte-identical by hash.

## Decision

1. **The reduction order is part of the operation's declared contract.** A
   partitioned reduction declares its split count and split boundaries. Each
   shard's partial is accumulated in FP32 in that shard's own fixed order.
   Partials are combined in FP32 in a fixed, declared order, and the result
   is rounded to BF16 **once**, after the combine. A partial is never rounded
   to BF16 before crossing ranks.
2. **The single-rank path executes the same declared order** when it serves as
   the reference for, or as a peer of, a partitioned plan. TP is then
   **bit-identical** to single-rank, and every TP acceptance test compares bits.
   No tolerance applies.
3. **The combine order is fixed by us, not by a library.** With two partials
   the order is irrelevant, because `a + b == b + a` exactly. With more than
   two, the collective must combine in the declared order, such as a fixed
   tree or rank order, and must never use an unspecified ring order. Moxie's
   collectives are its own (task 0057), so this is enforceable. A third-party
   collective adopted later must preserve the order or is not used for data
   reductions.
4. **The split count is part of plan identity.** A plan declared with split 2
   and one with split 1 may legitimately differ in the last bits. Plan-cache
   keys and any recorded reference output name the declared split.

## Consequences and costs

- A row-parallel kernel accumulates per shard and writes FP32 partials. The
  existing `moxie_bf16_linear_v1` sums the whole `k` axis in one ascending
  loop and rounds to BF16, so the row-parallel slice adds a partial-sum path
  and does not change that kernel.
- Output quality against the released model is still judged by document 07's
  quality gates. This ADR fixes only how partitioned and unpartitioned
  execution agree with each other.
- Failure propagation follows the same Strata design. Its status collective
  used MAX over a status word, and a failing rank still entered every
  collective, so no peer waited forever. That is carried as an obligation of
  the thread-per-rank slice, which is where it becomes necessary.

## Enforcement and re-evaluation

Enforced by the TP tests' bit comparison. There is no tolerance constant to
drift. Re-evaluate if a required kernel cannot expose FP32 partials at a
declared boundary. Relaxing to a tolerance would then again be an owner
decision.
