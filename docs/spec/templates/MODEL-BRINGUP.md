# Model bring-up contract

## Identity

Family, exact checkpoint revision/checksum/license, reference implementation/version, canonical artifact profile/quantizer, tokenizer/template, positional range, catalog decision O1, quality decision O2, storage authorization O5.

## Mathematical inventory

| Component | Source equation / code / line | Common op and options | Gap task if missing | Oracle fixture |
|---|---|---|---|---|
| Embedding / output head | | | | |
| Norm / activation / residual | | | | |
| Attention / position / masks | | | | |
| Routing / experts / shared experts | | | | |
| Recurrent / convolution / index / compression state | | | | |
| Native proposal heads | | | | |
| Image or other supported modality | | | | |

Unknown math is a blocker for that path. “Standard transformer” is not an equation reference.

## Import and resources

- Logical tensor-role mapping and exact scale/layout conversion:
- Sensitive unquantized tensors and quality rationale:
- State schema/dtype, bytes at actual context tiers, rollback method:
- Legal TP/PP/expert partitions and unsupported cases:
- Admitted host/device/disk resources and fallback disclosures:

## Integration proof

- Adapter changes contain metadata/graph only:
- New shared ops/kernels and independent second-consumer tests:
- No private runtime/cache/transfer/sampler/branch evaluator:
- Reference quality, actual-context prefill/decode and continuation:
- CPU/GPU precision, state, cancellation and distributed tests:
- Common sampling, future entropy and speculation capability results:
- HTTP/CLI/template/tokenizer/modality parity:
- Temporary code deleted; support matrix updated:

## Bring-up cost

Engineering hours/elapsed dates (reported, not guessed), files/lines changed by ownership, new shared operations, reused operations, initial performance without family-specific tuning, remaining bottleneck and owner O7 review.
