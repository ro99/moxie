# Experiment drivers

One file lives here, and it is here for a specific reason rather than by habit.

**Tooling belongs in `xtask`.** The mutation batteries used to be Python scripts
under `docs/`; they are now `cargo xtask mutation-check`, where the substitution
table is type-checked, the driver is covered by `fmt`, `clippy` and the
workspace suite, and there is no second language for a job this repository
already has a crate for.

`0007-reference-reader.py` stays outside the workspace because its whole purpose
is to run an implementation this repository **does not link**. Adding
`safetensors` to `Cargo.toml` would breach task 0026's "no new third-party
dependency" and weaken the check itself: our reader agreeing with a crate we
vendored is a weaker statement than our bytes being accepted by an
implementation that has never heard of us. Its Python binding is how the
reference is installed here without entering `Cargo.lock`.

It is still invoked through xtask, so every gate in this repository is a
`cargo xtask` command:

```
cargo xtask reference-check --artifact <dir>
```

| What | Where |
|---|---|
| Mutation batteries | `cargo xtask mutation-check` ([xtask/src/mutationcheck.rs](../../xtask/src/mutationcheck.rs)) |
| Reference conformance | `cargo xtask reference-check`, which runs `0007-reference-reader.py` |

Experiment 0005's battery was **retired** rather than ported: thirteen of its
twenty-four anchors already matched nothing, because the importer it measured
was rewritten by tasks 0025 and 0026. Its record keeps what it measured.
