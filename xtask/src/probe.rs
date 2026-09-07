//! `cargo xtask probe` -- bounded hardware and topology inventory.
//!
//! Document 06 M0.2: capture device identity, capabilities, peer access and
//! topology "using bounded probes; do not consume all RAM or convert/download
//! models". Nothing here allocates a large fraction of any tier.

use std::fmt::Write as _;

use moxie_cuda::{DeviceBuffer, DeviceContext, query_device};

/// Bytes moved per host-to-device timing sample. 64 MiB is large enough to leave
/// per-call overhead behind and small enough to be harmless on a 16 GiB card.
const XFER_BYTES: usize = 64 * 1024 * 1024;
const XFER_REPS: usize = 5;

pub fn run(out: Option<&str>) -> i32 {
    let count = match moxie_cuda::device_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("cannot enumerate devices: {e}");
            return 2;
        }
    };

    let mut md = String::new();
    let _ = writeln!(
        md,
        "| Ordinal | Name | SM | UUID | VRAM MiB | SMs | PCI bus |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|");

    let mut caps = Vec::new();
    for i in 0..count {
        match query_device(i) {
            Ok(c) => {
                let _ = writeln!(
                    md,
                    "| {} | {} | {} | `{}` | {} | {} | `{}` |",
                    c.ordinal,
                    c.name,
                    c.sm(),
                    c.uuid,
                    c.total_memory_bytes / (1024 * 1024),
                    c.multiprocessor_count,
                    c.pci_bus_id
                );
                caps.push(c);
            }
            Err(e) => {
                eprintln!("device {i}: {e}");
                return 1;
            }
        }
    }

    let _ = writeln!(md, "\n### Peer access (driver `cuDeviceCanAccessPeer`)\n");
    let _ = write!(md, "| from \\ to |");
    for c in &caps {
        let _ = write!(md, " {} |", c.ordinal);
    }
    let _ = writeln!(md);
    let _ = write!(md, "|---|");
    for _ in &caps {
        let _ = write!(md, "---|");
    }
    let _ = writeln!(md);
    for c in &caps {
        let _ = write!(md, "| **{}** |", c.ordinal);
        for peer in &caps {
            if peer.ordinal == c.ordinal {
                let _ = write!(md, " -- |");
            } else if c.can_access_peer(peer.ordinal) {
                let _ = write!(md, " yes |");
            } else {
                let _ = write!(md, " no |");
            }
        }
        let _ = writeln!(md);
    }

    let _ = writeln!(
        md,
        "\n### Host-to-device, pageable source, {} MiB x {} reps\n",
        XFER_BYTES / (1024 * 1024),
        XFER_REPS
    );
    let _ = writeln!(md, "| Ordinal | Median GB/s | Min | Max |");
    let _ = writeln!(md, "|---|---|---|---|");
    for c in &caps {
        match measure_h2d(c.ordinal) {
            Ok((med, lo, hi)) => {
                let _ = writeln!(md, "| {} | {med:.2} | {lo:.2} | {hi:.2} |", c.ordinal);
            }
            Err(e) => {
                let _ = writeln!(md, "| {} | failed: {e} | | |", c.ordinal);
            }
        }
    }

    let _ = writeln!(
        md,
        "\nMeasured with a pageable source and a synchronous copy, one device at a \
         time. Document 03 requires simultaneous-transfer behaviour and pinned \
         alternatives to be measured separately before a planner cost model uses \
         them; this probe does neither and is not that model."
    );

    print!("{md}");
    if let Some(path) = out {
        if let Err(e) = std::fs::write(path, &md) {
            eprintln!("cannot write {path}: {e}");
            return 1;
        }
        eprintln!("wrote {path}");
    }
    0
}

fn measure_h2d(ordinal: u32) -> Result<(f64, f64, f64), moxie_types::Error> {
    let ctx = DeviceContext::new(ordinal)?;
    let host = vec![0u8; XFER_BYTES];
    let mut dev = DeviceBuffer::alloc(&ctx, XFER_BYTES)?;

    // One warm-up: the first copy pays context and mapping costs that are not
    // part of the steady-state rate.
    dev.copy_from_host(&ctx, &host)?;

    let mut rates = Vec::with_capacity(XFER_REPS);
    for _ in 0..XFER_REPS {
        let t = std::time::Instant::now();
        dev.copy_from_host(&ctx, &host)?;
        let secs = t.elapsed().as_secs_f64();
        rates.push((XFER_BYTES as f64) / secs / 1e9);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((rates[rates.len() / 2], rates[0], rates[rates.len() - 1]))
}
