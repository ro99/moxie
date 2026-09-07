//! Link against the CUDA driver library.
//!
//! We link `libcuda` (the *driver*), not `libcudart` (the runtime): the driver
//! API is what lets a rank own its context explicitly (document 01). The driver
//! library ships with the display driver, not the toolkit, so it lives in the
//! system library path rather than under CUDA_HOME.

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=build.rs");

    // The real driver library. The toolkit also ships a stub under
    // `lib64/stubs` for link-time-only use; linking the stub would build fine
    // and then fail at run time, so it is deliberately not searched here.
    for dir in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"] {
        if std::path::Path::new(dir).join("libcuda.so.1").exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
