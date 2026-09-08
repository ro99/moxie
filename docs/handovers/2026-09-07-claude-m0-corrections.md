# Copy/paste assignment for Claude — M0 corrections and integer weights

The following is the implementation prompt. The review changed documents only; do not assume the requested Rust/CUDA corrections already exist.

```text
Work in /home/rodrigo/Developer/moxie. Preserve unrelated edits and treat /home/rodrigo/Developer/strata as read-only reference. Read AGENTS.md, ADR 0003, the revised documents 01–04/06–09, docs/evidence/quantization-candidates.md, and docs/tasks/0002-m0-review-and-integer-transition.md before editing.

Keep the M0 scaffold; do not restart the project. Close the review's M0 issues before implementing the M1 inference slice. Execute separate bounded subtasks with tests and evidence:

1. Make host CI and arch-check runnable without nvcc, libcuda or a GPU. Separate opt-in CUDA build/run dependencies; retain real CUDA gates. Add the missing cargo xtask alias or consistently document the actual command. No skipped GPU lane counts as passing.
2. Fix architecture enforcement for target-specific dependencies, dependency aliases/package identity, workspace inheritance and custom build/source paths. Add the three reproduced negative cases from the review and assert their intended violation. Use existing Cargo/Rust parsing mechanisms rather than expanding a fragile substring parser. No model may own CUDA, I/O, state, sampling or execution.
3. Correct the safe CUDA image-loading boundary: arbitrary &[u8] cannot establish cuModuleLoadData's valid-image/PTX contract. Keep the raw loader unsafe or expose narrowly validated/trusted typed images. Audit context guards and retain-failure cleanup; keep the real smoke tests.
4. Correct state semantics: accepted history, emitted usage and tentative branch execution are distinct. Verification may execute unaccepted candidates. Counter equality does not prove logits exist; use branch/prefix/generation-qualified retained outputs or recomputation. Test empty-state invalidity, rejection at every depth, pending bonus tokens, rollback/replay and entropy parent isolation. Do not implement full paging just to hide a bad frontier abstraction.
5. Replace the active NVFP4 family with the shared affine INT4/INT8/BF16 design in document 03. Do not rename E2M1 decoding to INT4. Use full signed code ranges, explicit group size/index maps, zero points, and source-preserving scale dtypes. Refactor INT8 away from the symmetric-only/per-channel-only restriction; -128 is a valid INT8 decoder input. Keep BF16 tests. Remove obsolete active NVFP4 exports/capability claims once replacement tests pass; preserve historical evidence/Git history.
6. Finish the required small M0 source-linked synthetic fixtures and bounded stream/event smoke. Missing large checkpoints do not block tiny routing, masks, recurrent equations, sampler distributions or protocol fixtures. Before M1 lowering, replace caller-set oracle booleans with registered evidence and distinguish weight/activation/cache/accumulator roles.

INT4/INT8 are weight-only W4A16/W8A16 initially, with BF16 preferred and explicitly qualified FP16 where needed. W4A16 is not native INT4×INT4 MMA. AWQ/AutoRound are methods, not reasons for separate runtimes. Implement no new model runtime, no FP4/FP8/W4A4/W8A8 path, no cache below 16 bits, and no full checkpoint download/conversion in this task.

The ten candidates have inspected metadata, not proven tensor compatibility or quality. Prioritize shared compressed-tensors integer import, then AutoRound/AutoGPTQ-style packing in later bounded tasks. Preserve asymmetric group-32, symmetric group-128 and group-32 INT8 semantics, exclusions, activation-order mappings and MTP heads. Prefer lossless repacking over requantization. O1/O2/O4/O5 gates still apply; do not ask again whether NVFP4 should be the initial family.

Update task 0001's current closure status, task 0002 subtask results, support claims and handover with exact commands, source/binary identities and passed/failed/skipped results. Record the required local specification version/digest without changing its owner-directed publication policy. End with the smallest M1 BF16 graph/interpreter task proposed next. Do not claim checkpoint support or throughput from codec/smoke tests.
```
