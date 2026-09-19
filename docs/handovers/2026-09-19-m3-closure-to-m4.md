# Handover — accepted M3 boundary and first M4 slice

## Workspace identity

- Writable root `/home/rodrigo/Developer/moxie`, branch `main`. M3 implementation
  is `85daa05` plus ABI repair `0672d24`; closure evidence is `54b52e3`. Fetch
  the pushed branch and recheck HEAD before implementation.
- `coordinator.md` is unrelated carried work. Preserve it and do not include it
  in M4 commits without a separate assignment.
- Legacy `/home/rodrigo/Developer/strata` remains read-only at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. `/models` and `/fast/models` are
  read-only inputs under ADR0020.

## Completed facts

The repository owner accepted M3.1/M3.2/M3.3 on 2026-09-19 through task0036.
The frozen candidate provides bounded canonical safetensors publication,
compressed-tensors and GPTQ/AutoRound integer import, and common dense/grouped
W4A16/W8A16 execution. Dense and grouped mapped paths run on both SM86 GPUs and
SM120 over synthetic activations/routes; one real republished module supplies a
dense weight. Nothing composes a checkpoint-backed block or generates a token.

Final T0006 at `0672d24` caught 63/63 mutants and held three controls. Final
T0028 caught 24/24 and held one control. Both had stable three-repetition clean
baselines before and after, zero survivor/unstable/invalid/broken/skipped
verdicts, exit zero and a restored clean detached tree. Architecture79/21/13,
specification10, formatting, affected host/driver checks, the six-case dense
device suite and the 11-test grouped-device suite pass. Task0036 carries the
full evidence table and log hashes.

## Decisions

Owner instruction on 2026-09-19: “document and push. we are ready to m4”. M3 is
accepted within task0036's stated boundary and M4 is authorized. O6/O7 remain
open. Offline repacking remains provisional under ADR0027 and experiment0007;
M3 acceptance is not measured inference benefit. The signed-scale and ADR0028
numerical rulings remain unchanged.

## Remaining hypotheses and blockers

M4 has no accepted device-attention implementation yet. Existing attention and
mask oracles define semantic behavior, and paged host state exists, but neither
proves persistent device KV, Flash-style execution, actual 32K attention or
state-growth admission. Legacy short-context measurements are source lessons,
not baselines for this engine. FlashAttention and FlashInfer are pinned source
candidates; architecture, shape, linkage and license fit must be audited before
adoption. MLA, host-backed streaming, COW/recurrent rollback and prefix reuse
remain later M4 work rather than hidden requirements in the first kernel.

## Next task

[Task0037](../tasks/0037-m4-paged-device-attention.md) is the bounded first M4
deliverable: common BF16 paged device attention for prefill, append and decode;
causal/sliding MHA/GQA semantics; persistent admitted state; both SM86 GPUs and
SM120; and one **actual 32,768-row** whole/chunked/continuation gate. Start with
the pinned-source capability audit and the existing oracle/state ownership
contracts. Stop before code if the chosen upstream cannot cover the declared
hardware/shapes without a license exception or model-private fallback.
