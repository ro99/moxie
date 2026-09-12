//! A model adapter that wants to own where its experts live.
//!
//! The graph composition is legal; the residency is not. Task 0019 composes
//! `Route -> ExpertMlp -> Combine` in `moxie-models::gemma4` and binds the
//! fused expert tensors as ordinary weights, leaving every question about
//! which of them is resident to task 0020's single authority.

pub fn admit_experts(_ledger: &moxie_memory::Ledger) -> usize {
    // A model crate deciding what to keep. Exactly the second cache owner
    // document 02 forbids, and the reason the edge above is refused.
    0
}
