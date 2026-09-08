# Specification version and digests

The normative reference documents are kept local and untracked at the owner's
direction; see [docs/README.md](../README.md). This file is tracked, so a fresh
clone can tell whether it has the specification and whether the copy it has is the
one the living records were written against.

`cargo xtask spec-check` verifies it. It reads and hashes; it never fetches,
copies or generates a document, and it does not change the publication policy.

Regenerate with `cargo xtask spec-check --update` **after** an ADR amends a
document, and commit the diff alongside that ADR. A digest that changes with no
ADR is an amendment by edit, which the placement contract forbids.

SHA-256 of the raw file bytes:

| Document | SHA-256 |
|---|---|
| `docs/spec/01-product-and-decisions.md` | `ac14455eea4aead0ce3e6f038b818609741194cbdfeba9d32fcdba540242ab3b` |
| `docs/spec/02-architecture-and-common-api.md` | `7a458c89c4e12ad87eaa67cbf4aea2ec8381e0575e380beb046728355b460cfe` |
| `docs/spec/03-memory-formats-and-cuda.md` | `1432619b97750185cd2679902df14ab902470601a3a66d87425a95f4529e4c5b` |
| `docs/spec/04-attention-parallelism-and-speculation.md` | `f13963ae01f14ae18ae9e8eedb0f3d1807ca560e4182ee887c79c5c38322c612` |
| `docs/spec/05-sampling-api-and-cli.md` | `a5eb36d86efaf6532d40a18791e96a2dca4dc288f0ecfb77e63637b6c6c5f61a` |
| `docs/spec/06-implementation-roadmap.md` | `5fb3fb474138f14e5d1ed0953d32eb5122a7fbfba5da1c4d3d692c0506256bce` |
| `docs/spec/07-validation-and-performance.md` | `1838291d4b53baeefd98353ef383986ce922669ecab396eb99e97c123517442a` |
| `docs/spec/08-strata-reference-map.md` | `a21e754a2d1aebcc215b9a5a50b4ebccdbd5761325cea2918f9f5ec8cd6774a6` |
| `docs/spec/09-agent-playbooks.md` | `96efdaa3d9297d5214c223b49c4101db827e6ba46c79578d88ba6f75ab47d631` |
| `docs/spec/strata-arch-diagnosis.md` | `3983f5177a5b2d480caaf09d6aa11b18a3e7360693475bbe176ea0ae868408ca` |
