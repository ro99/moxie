//! Workspace command index.
//!
//! Document 07 defines the command contracts M0 must create. Names live here and
//! are documented in one place; changing one is a versioned change, not a rename.
//!
//! Two builds, two aliases (see `.cargo/config.toml`):
//!
//! * `cargo xtask <cmd>` -- the host lane. No nvcc, no libcuda, no GPU. This is
//!   what `arch-check` needs, and document 07 requires it to run "without a
//!   checkpoint, NVIDIA driver library or CUDA toolkit".
//! * `cargo xtask-cuda <cmd>` -- the device lane, with `--features cuda`. It
//!   compiles the fatbins and links the driver.
//!
//! A device command invoked from the host build does **not** print a friendly
//! nothing and exit zero. It fails, loudly, with the command that would work:
//! document 07's rule is that a lane which did not run is unmeasured, never
//! passing, and that starts here.

mod archcheck;
#[cfg(feature = "cuda")]
mod capacity;
#[cfg(feature = "cuda")]
mod gpu;
mod mutationcheck;
#[cfg(feature = "cuda")]
mod probe;
mod speccheck;

const USAGE: &str = "\
cargo xtask <command>            host lane: no CUDA toolkit or driver needed
cargo xtask-cuda <command>       device lane: --features cuda

Host lane:
  arch-check              Dependency direction, model ownership, negative fixtures
  spec-check [--update]   Normative specification present and unmodified
  mutation-check [--battery <tag>] [--self-test] [name ...]
                          Can the gates fail? One substitution at a time against
                          tracked source, every lane run, only `caught` counts.
                          Edits source in place and restores it, including after
                          a kill. Run it on a clean tree
  reference-check --artifact <dir>
                          Open every published shard with the **reference**
                          safetensors implementation and hold each component to
                          the descriptor. Needs python3 with `safetensors`: the
                          reference is deliberately not linked by this workspace
  index                   List command contracts and their required lane

Device lane (needs `cargo xtask-cuda`):
  test-gpu [--profile sm_NN]
                          Real CUDA launches on every visible device; a required
                          architecture with no passing case fails the gate
  test-bf16-chain         Reduced H8/H17 semantic chain for CUDA sanitizers
  probe [--out <path>] [--costs <path>]
                          Hardware/topology inventory; --costs writes measured planner costs
  capacity                Measure every device and admit a plan against the ledger;
                          allocates nothing

Not an `xtask` command -- a user program, built by this workspace:
  moxie-repack inspect|repack|verify
                          Offline canonical inspection, bounded repack and
                          verification (ADR 0021/0022). `cargo run -p
                          moxie-repack -- --help` for its arguments; every
                          budget it spends is required on the command line

Not yet implemented; they land with the milestone that defines them:
  test-topology           M5   TP/PP/expert transport, failure and cancellation
  quality                 M3   Pinned artifact/reference comparison
  bench                   M6   Registered paired workload, machine-readable results
  support-matrix --verify M7   Capability claims linked to passing gate IDs
";

/// What a device command prints when this binary was built for the host lane.
#[cfg(not(feature = "cuda"))]
fn device_command_unavailable(cmd: &str) -> i32 {
    eprintln!(
        "`{cmd}` needs the device lane, and this binary was built without it.\n\
         \n\
         Run:  cargo xtask-cuda {cmd}\n\
         \n\
         The host build deliberately links no CUDA driver and compiles no fatbin \
         (document 07). This is not a skip: nothing was measured, so nothing passed."
    );
    2
}

/// The writable repository root. `CARGO_MANIFEST_DIR` is `xtask/`.
pub fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent")
        .to_path_buf()
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    // Only the device commands take flags; in the host build nothing reads it.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };

    let code = match cmd {
        "arch-check" => archcheck::run(),
        "spec-check" => speccheck::run(args.iter().any(|a| a == "--update")),
        "mutation-check" => mutationcheck::run(&args[1..]),
        "reference-check" => mutationcheck::reference_check(&args[1..]),

        #[cfg(feature = "cuda")]
        "test-gpu" => gpu::run(flag("--profile")),
        #[cfg(feature = "cuda")]
        "test-bf16-chain" => gpu::run_chain(),
        #[cfg(feature = "cuda")]
        "probe" => probe::run(flag("--out"), flag("--costs")),
        #[cfg(feature = "cuda")]
        "capacity" => capacity::run(),

        #[cfg(not(feature = "cuda"))]
        "test-gpu" | "test-bf16-chain" | "probe" | "capacity" => device_command_unavailable(cmd),

        "index" | "help" | "-h" | "--help" => {
            print!("{USAGE}");
            0
        }
        other => {
            eprintln!("unknown command: {other}\n");
            eprint!("{USAGE}");
            // An unimplemented command must fail loudly. Document 06: "Tests may
            // initially assert unsupported operations; placeholder success is
            // forbidden."
            2
        }
    };
    std::process::ExitCode::from(code as u8)
}
