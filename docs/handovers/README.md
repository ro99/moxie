# Handovers

## Active handover

[M3 accepted; begin M4](2026-09-19-m3-closure-to-m4.md) is the current
continuation. The owner accepted M3.1/M3.2/M3.3 on 2026-09-19 and authorized
M4. The next bounded deliverable is [task
0037](../tasks/0037-m4-paged-device-attention.md): common BF16 paged device
attention with persistent admitted state, full/sliding MHA/GQA,
prefill/append/decode, both SM86 GPUs and SM120, and an actual 32,768-row gate.

The accepted M3 boundary includes canonical publication, compressed-tensors and
GPTQ/AutoRound import, and shared dense/grouped W4A16/W8A16 operation execution.
It does not include checkpoint token generation, model-output quality or a
performance claim. O6/O7 and experiment0007 remain open.

## Historical handovers

- [M3 recovery direction](2026-09-16-m3-recovery-direction.md)
- [Task0030 repack correctness](2026-09-16-task0030-repack-correctness.md)
- [Task0029 fallible admission](2026-09-15-task0029-fallible-admission.md)
- [Task0028 shared quantized execution](2026-09-14-task0028-shared-quantized-execution.md)
- [Task0027 two-command repack](2026-09-14-task0027-two-command-repack.md)
- [Task0026 safetensors publication](2026-09-14-task0026-safetensors-publication.md)
- [Task0025 offline repack publication](2026-09-13-task0025-offline-repack-publication.md)
- [Task0024 asymmetric INT4 import](2026-09-13-task0024-asymmetric-int4-import.md)
- [Task0023 whole-working-set trace](2026-09-13-task0023-whole-working-set-trace.md)
- [M1 closure to M2](2026-09-12-m1-closure-to-m2.md)

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md). Name the writable root,
branch/commit, dirty files and legacy snapshot. Keep facts, hypotheses and
decisions separate, record failed/skipped work, and end with one bounded next
task.
