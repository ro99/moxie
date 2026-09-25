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

    if std::env::var_os("CARGO_FEATURE_CUBLAS").is_some() {
        let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let lib_dir = std::path::Path::new(&cuda_home).join("lib64");
        let library = lib_dir.join("libcublas.so.13");
        if !library.is_file() {
            panic!(
                "moxie-cuda/cublas was enabled but {} is absent",
                library.display()
            );
        }
        println!("cargo:rerun-if-changed={}", library.display());
        let resolved = std::fs::canonicalize(&library).unwrap_or_else(|error| {
            panic!(
                "failed to resolve cuBLAS library {}: {error}",
                library.display()
            )
        });
        println!("cargo:rerun-if-changed={}", resolved.display());
        let filename = resolved
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_else(|| panic!("resolved cuBLAS library has no UTF-8 filename"));
        let version = filename.strip_prefix("libcublas.so.").unwrap_or_else(|| {
            panic!("resolved cuBLAS library filename {filename:?} has no version suffix")
        });
        let components: Vec<_> = version.split('.').collect();
        if components.len() != 4 || components.iter().any(|part| part.parse::<u32>().is_err()) {
            panic!("resolved cuBLAS filename has an unexpected version: {filename}");
        }
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-env=MOXIE_CUBLAS_FILE_VERSION={version}");
    }
}
