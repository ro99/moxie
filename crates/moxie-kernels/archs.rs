// Compute capabilities this build targets. Single source of truth: included by
// both `build.rs` (to drive nvcc) and `src/lib.rs` (to report them without
// needing the toolkit). Changing this list is a support matrix change
// (document 07), not a build tweak.
pub const ARCHS: &[&str] = &["86", "120"];
