//! Link against the CUDA driver library -- only when the `driver` feature is on.
//!
//! We link `libcuda` (the *driver*), not `libcudart` (the runtime): the driver
//! API is what lets a rank own its context explicitly (document 01). The driver
//! library ships with the display driver, not the toolkit, so it lives in the
//! system library path rather than under CUDA_HOME.
//!
//! Without the feature this script emits no link directive at all, which is what
//! lets `cargo test --workspace` and `arch-check` build on a runner that has no
//! NVIDIA driver installed (document 07).

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_DRIVER").is_none() {
        return;
    }

    // The real driver library. The toolkit also ships a stub under
    // `lib64/stubs` for link-time-only use; linking the stub would build fine
    // and then fail at run time, so it is deliberately not searched here.
    let mut found = false;
    for dir in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"] {
        if std::path::Path::new(dir).join("libcuda.so.1").exists() {
            println!("cargo:rustc-link-search=native={dir}");
            found = true;
        }
    }
    if !found {
        // Loud and specific: the feature was asked for and cannot be honoured.
        // Document 07 forbids turning an unavailable GPU lane into a pass, and
        // that starts with not silently producing an unlinkable build.
        panic!(
            "moxie-cuda/driver was enabled but libcuda.so.1 was not found in \
             /usr/lib/x86_64-linux-gnu, /usr/lib64 or /usr/lib. Install the \
             NVIDIA driver, or build without the `driver` feature for the \
             host-only lane."
        );
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
