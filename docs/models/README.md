# Model bring-up contracts

Two families have a record. [gemma4.md](gemma4.md) covers the dense 31B and the
designated M2 routed 26B-A4B; [laguna.md](laguna.md) covers Laguna S 2.1, whose
**routed block** is composable and whose **attention tower is not** — `softplus`
output gating has no shared operation, and the yarn rotary ramp is implemented
nowhere the artifact ships. Neither family executes a checkpoint.

One file per family, `<family>.md`. Use [MODEL-BRINGUP.md](../spec/templates/MODEL-BRINGUP.md).

Unknown mathematics is a blocker for that path; "standard transformer" is not an equation reference.
A contract is complete only when its integration proof shows the adapter contains metadata and graph
composition and nothing else.
