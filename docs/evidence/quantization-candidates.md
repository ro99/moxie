# Requested integer checkpoints — metadata evidence

Read-only inspection on 2026-09-07. Sources: each public Hugging Face model API and the `config.json` at the immutable revision below. These candidates may be downloaded under the owner-designated roots [`/models` and `/fast/models`](artifact-roots.md); inspect exact local files and hashes before treating a candidate as an artifact. No remote checkpoint code may be executed. These are candidates supplied by the owner, not a release catalog or a quality endorsement.

## Observed schemas

| Candidate | Revision | Declared quantization, not inferred from name |
|---|---|---|
| [canada-quant/glm-5.3-w4a16-mtp](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp/blob/1c86622dfd7ecca80909ff1524ac1b3618b8da6f/config.json) | `1c86622dfd7ecca80909ff1524ac1b3618b8da6f` | compressed-tensors `pack-quantized`; INT4, symmetric, group 128, `actorder=static`; model type `glm5_next` |
| [canada-quant/hy3-w4a16-mtp](https://huggingface.co/canada-quant/hy3-w4a16-mtp/blob/49228b990c704e4efd67ac420a8e3d5272f820c0/config.json) | `49228b990c704e4efd67ac420a8e3d5272f820c0` | compressed-tensors `pack-quantized`; INT4, symmetric, group 128, `actorder=static`; model type `hy_v3` |
| [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound/blob/5eee1846f0321058ed73745f9aa16f2aaf0fc0a0/config.json) | `5eee1846f0321058ed73745f9aa16f2aaf0fc0a0` | AutoRound 0.15.0; `auto_round:auto_gptq`; INT4, symmetric, group 128; model type `glm5_next` |
| [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/config.json) | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` | AutoRound 0.15.0; `auto_round:auto_gptq`; INT4, symmetric, group 128; model type `qwen4_exp` |
| [Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound](https://huggingface.co/Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound/blob/c838af0996ae78d27a0184d3da9772f46eb34e25/config.json) | `c838af0996ae78d27a0184d3da9772f46eb34e25` | AutoRound 0.15.0; `auto_round:auto_gptq`; INT4, symmetric, group 128; `model_free=true`, `iters=0`; model type `deepseek_v4` |
| [cyankiwi/Laguna-S-2.1-AWQ-INT4](https://huggingface.co/cyankiwi/Laguna-S-2.1-AWQ-INT4/blob/bc59f497520b23759ce61cc5164ca28bcc4f53bc/config.json) | `bc59f497520b23759ce61cc5164ca28bcc4f53bc` | compressed-tensors `pack-quantized`; INT4, **asymmetric**, group **32**, `zp_dtype=torch.int8`; model type `laguna` |
| [cyankiwi/Muse-Glimmer-30B-AWQ-INT4](https://huggingface.co/cyankiwi/Muse-Glimmer-30B-AWQ-INT4/blob/cba01edf73e0f0f4f013615cc01281ea04e79f85/config.json) | `cba01edf73e0f0f4f013615cc01281ea04e79f85` | compressed-tensors `pack-quantized`; INT4, asymmetric, group 32; general `dtype=float16`; model type `muse_glimmer` |
| [cyankiwi/Inkling-Small-AWQ-INT4](https://huggingface.co/cyankiwi/Inkling-Small-AWQ-INT4/blob/599a903386348a364fe30ab0a67dcc61ff9e8008/config.json) | `599a903386348a364fe30ab0a67dcc61ff9e8008` | compressed-tensors `pack-quantized`; INT4, asymmetric, group 32; model type `inkling_mm_model` |
| [cyankiwi/gemma-4-31B-it-AWQ-8bit](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-8bit/blob/34ca187d836de874b2c7e3edf48f439b9f583772/config.json) | `34ca187d836de874b2c7e3edf48f439b9f583772` | compressed-tensors `pack-quantized`; **INT8**, symmetric, **group 32**, not per-channel; model type `gemma4` |
| [cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4/blob/6fcdc07bfd6e872632f79680e786c5d17fcafaee/config.json) | `6fcdc07bfd6e872632f79680e786c5d17fcafaee` | compressed-tensors `pack-quantized`; INT4, asymmetric, group 32; top-level model type `qwen3_5` |

The compressed-tensors configurations declare null input/output activation quantization and null KV-cache schemes. That supports a weight-only import direction; it does not establish the dtype or encoding of every actual tensor. `scale_dtype` is null in the inspected group definitions: inspect the tensor headers before choosing a scale decoder. All sources have per-module exclusions or overrides; counts observed were 765/641 for canada-quant, 679/2340/242 extra entries for Intel, and 1788/315/141/192/456 exclusions for the cyankiwi rows in table order. These lists cannot be ignored or replaced by “quantize every linear.”

## Config content hashes

SHA-256 of the fetched raw `config.json` bytes, in table order:

```text
92ae6f37ca659fe95b2032d1b680a0dffebf3b367b7122de45eba1dafc436acc
8f5f45c43eb2243af93718347626ba1fb4db7abd466c30321b290adcfcce35aa
d4deaf40c47b2ff49f1d8e0c306032d7a8b84f90b6a2743e694b712d87dd5692
0fabfa21fab8bfe69f02234f7ae8df4ed91b785fca2e22011c1230f6a07e5329
0e2e107106244b8db5b3bf93a6f354ab2fba286a873f192bc5b21c924ccd2ecc
28e19fc9b58eb6d39b6c24ba21e5feb7dede13c01ddc0f93a5719a3a0aec2f1d
78ca1b494f2c0d2ba7862f2ebbb4a392a034f8eca1fed5fa70d14121ac0b7ca4
edd625735d6be60cfed9c8c9d8ad72e5e73b7052f2b066311d55001b7a9e2bc7
f9f7b7c592c98a99018843aa242b05cbd19f98a423ce43241d3ef208ec444b76
c85a8abaad496b6fc430c4e3560a55efc32322a6260aea63e33c5d607c4affb4
```

## Consequences and limits

- **Use one affine algorithm, not one guessed packing.** Symmetric and asymmetric weights share `(q-z)*s`. Group size and zero point are meaningful source parameters. AWQ/AutoRound do not require separate model runtimes.
- **Do not force group-32 INT8 into the old per-channel INT8 v1.** That would alter scales/weights or expand storage. Replace the profile before a real importer depends on it.
- **Preserve source scalar dtypes.** Muse explicitly declares FP16 generally. This does not justify rounding all source scales to BF16 without a measured conversion decision. A small scale/activation dtype contract is not broad format proliferation.
- **Do not infer a graph from an artifact name.** Several listed model types differ from the seven legacy family identifiers. HY3, Muse and Qwen variants require semantic inventories and common-op extensions as necessary; they are not supported merely because the integer decoder accepts their bytes.
- **MTP in the name is insufficient.** Inspect the tensor index, head shapes, quantization exclusions, tokenizer and released proposer semantics. Lossless repacking must retain auxiliary heads; it does not implement a verifier or guarantee speedup.
- **Metadata is not quality evidence.** In particular, `model_free=true` / `iters=0` and a recognizable publisher are not proofs of calibration quality. Quantized-source import parity and released-model quality remain separate checks.

Not determined here: actual scale tensor dtypes, actual zero-point packing/offsets, presence and semantics of every `g_idx`/permutation, complete MTP tensor coverage, model licenses/calibration datasets, numerical quality, exact CUDA kernel compatibility, artifact storage needed, and throughput. Pinned tensor indexes/headers, exporter source and small licensed tensor fixtures settle import questions; full reference/quality/performance gates settle the rest. No weights need to be downloaded just to establish these metadata distinctions.
