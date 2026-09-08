# Local checkpoint artifact roots

The owner has designated these local directories as the places where downloaded
model checkpoints may be found on the benchmark machine:

- `/models`
- `/fast/models`

These are **read-only inputs for investigation and bring-up**. They are not
the release catalog, and their presence does not by itself resolve owner gates
O1 (catalog/order), O2 (quality) or O5 (storage, conversion and retention
authority). A directory may contain partial downloads, superseded revisions,
or artifacts whose license and format have not yet been verified.

Before using a checkpoint, an agent must:

1. locate the exact model directory below one of the roots;
2. inspect its configuration, tokenizer/template files, tensor index and shard
   completeness without executing checkpoint-provided code;
3. record the source repository and immutable revision when available;
4. record a content hash or an explicit manifest of the files used; and
5. keep the artifact's status separate from Moxie's support matrix until the
   importer, quality, context, memory and performance gates pass.

Do not assume `/models` and `/fast/models` are one filesystem or that a path
under one root is interchangeable with a path under the other. Do not copy,
convert, delete, or requantize a large artifact without a task that explicitly
authorizes that operation. Large raw checkpoints remain outside git; records
cite their absolute access location, hashes and retention policy.

The inventory in [checkpoint-inventory.md](checkpoint-inventory.md) records what
was observed at a particular date. Update it with exact paths and hashes after
inspection; do not turn this location contract into an assertion that every
candidate has finished downloading.
