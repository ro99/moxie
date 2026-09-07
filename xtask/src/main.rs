//! Workspace command index.
//!
//! Document 07 defines the command contracts M0 must create. Names live here and
//! are documented in one place; changing one is a versioned change, not a rename.

mod archcheck;
mod gpu;
mod probe;

const USAGE: &str = "\
cargo xtask <command>

  arch-check              Dependency direction, model ownership, negative fixtures
  test-gpu                Real CUDA launches on every visible device
  probe [--out <path>]    Hardware/topology inventory; writes markdown when --out given
  index                   List command contracts and their required lane

Not yet implemented; they land with the milestone that defines them:
  test-topology           M5   TP/PP/expert transport, failure and cancellation
  quality                 M3   Pinned artifact/reference comparison
  bench                   M6   Registered paired workload, machine-readable results
  support-matrix --verify M7   Capability claims linked to passing gate IDs
";

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");

    let code = match cmd {
        "arch-check" => archcheck::run(),
        "test-gpu" => gpu::run(),
        "probe" => {
            let out = args
                .iter()
                .position(|a| a == "--out")
                .and_then(|i| args.get(i + 1));
            probe::run(out.map(String::as_str))
        }
        "index" => {
            print!("{USAGE}");
            0
        }
        "help" | "-h" | "--help" => {
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
