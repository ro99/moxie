# ADR 0032 — write the first paged attention kernel here, adopt no upstream source yet

- **Date / status:** 2026-09-19 / decided before device code; no kernel exists yet.
- **Classification:** implementation choice under the authorized task0037, from a
  **source audit**. Nothing here is measured: O6 and O7 are open and this ADR
  makes no speed claim in either direction.
- **Owner:** `moxie-kernels` device implementation, `moxie-executor` binding.
- **Re-evaluated by:** the first slice authorized to make a performance claim,
  and any shape or visibility rule the local kernel cannot serve.

## Why the question is open

Roadmap M4 item 1 says to "integrate a pinned suitable upstream implementation
**or existing kernel** through the common ABI", and task0037 names two candidates
to audit before adopting code: FlashAttention at
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` and FlashInfer at
`91bda04c66f7cb851e1ab3b78b9fecea644b9844`. Document 08 warns in advance that a
generation name is not proof of a kernel for a 3090 or a 5060 Ti. The audit read
the pinned revisions; no upstream file was copied into this tree, and none is.

## What the audit found

**FlashAttention, BSD 3-Clause.** Its runtime guard is `cc_major >= 8`
(`csrc/flash_attn/flash_api.cpp:392`), which admits both SM86 and SM120, and
`setup.py:74` defaults to archs `80;90;100;110;120`, emitting `compute_80/sm_80`
for Ampere and a `sm_120` gencode only on CUDA ≥ 12.8 — this machine's toolkit
is 13.0. An `sm_80` cubin does reach the 3090s under CUDA's minor-version binary
compatibility rule. But the FA2 sources are the `*_sm80.cu` generation, their
launch template guards on `__CUDA_ARCH__ >= 800`, and their own unsupported-arch
message reads "requires building with sm version **sm80-sm90**"
(`csrc/flash_attn/src/flash_fwd_launch_template.h:17,25`). Compiling them with
`compute_120` satisfies the guard arithmetically; the source does not claim
Blackwell for that generation, so SM120 would be ours to qualify with no upstream
statement behind it.

Two harder facts. The host API is a PyTorch extension, not a C ABI: `flash_api.cpp`
includes `torch/python.h` and `c10/cuda/CUDAStream.h` and every entry point takes
`at::Tensor` and checks with `TORCH_CHECK`. Adopting it means linking libtorch
into this engine — which Rust owns the control plane of — or rewriting the entire
dispatch layer, which is porting rather than adopting. And its paged path
requires `page_block_size % 256 == 0` (`flash_api.cpp:613`), a page geometry
constraint on `moxie-state` that arrives from outside. The CUTLASS submodule
(`setup.py:279`) comes with the kernels.

**FlashInfer, Apache-2.0.** Its GPU table lists SM 8.6 and SM 12.0 explicitly,
with the caveat that not every feature covers every capability, and its core is
genuinely free of PyTorch: `include/flashinfer/attention/decode.cuh` includes
CUDA headers and its own, nothing else. That makes it the stronger long-term
candidate and it is named as such. What it also brings is a page-table ABI:
`paged_kv_t` (`include/flashinfer/page.cuh:38`) fixes the layout, the
`indptr`/`indices`/`last_page_len` representation and the NHD/HND choice, and the
plan/run scheduling and workspace ownership around it live in the Python and
JIT-packaged layer rather than in those headers. Task0037 assigns page identity
and admission to `moxie-state` and `moxie-memory`; importing that type now would
settle a contract this repository has not written yet, in the first slice, to
get code whose benefit cannot be measured until O6/O7 are resolved.

## Decision

Task0037's kernel is written in `moxie-kernels` against the existing common ABI,
beside the BF16 chain, expert and affine-linear images that already compile for
SM86 and SM120 and are qualified per architecture. No FlashAttention or
FlashInfer source enters this tree under this ADR. The FP64 oracle, the
`attention_error_bound` gate and the device-qualification rules the existing
kernels are held to apply unchanged; correctness, bounded state and hardware
qualification are what the first slice proves.

This is not a judgement that the local kernel is faster, and nothing in this
repository may claim it is. It is the ordering: own the semantics, the page
identity and the admission first, where they are this engine's to define, and
take on an upstream dependency when there is a measurement that says what it
buys.

## Enforcement and re-evaluation

No upstream attention file is in the tree, so there is nothing to enforce beyond
that absence; `arch-check` already forbids a model crate owning CUDA code, which
is the ownership rule this decision does not change. Adoption later is not
blocked by this ADR and needs no new one — it needs what task0037 already
requires of any adopted source: pinned revision, license, the exact files and
modifications, and the hardware and shapes claimed, recorded before copying. For
FlashInfer that includes Apache-2.0 attribution and a NOTICE obligation this
repository does not carry today.

Re-evaluate when a slice is authorized to make a performance claim, when a paged
or masked shape the local kernel refuses is actually needed, or when a qualified
SM120 attention kernel exists upstream and says so in its own source.
