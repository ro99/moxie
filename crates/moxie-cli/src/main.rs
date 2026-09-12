use moxie_cli::{Options, USAGE, render};
use moxie_engine::{Cancel, service::GenerationService};
use moxie_memory::{CapacitySnapshot, Ledger};

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(true);
    }
    let options = Options::parse(&args)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let fixture = options.shape.build()?;
    let reduction = options.shape.reduction();
    if !reduction.is_empty() {
        // Printed before admission, so a reader sees what the graph is not
        // before they see anything it produced.
        println!(
            "event=reduced shape={} reduced={reduction} model_support=false",
            options.shape.name()
        );
    }
    let host = moxie_host::read()?;
    let snapshot = CapacitySnapshot::measured_host(&host, 1 << 30)?;
    let mut ledger = Ledger::new([snapshot])?;
    let mut service = GenerationService::new(&mut ledger, fixture.program());
    service.start(options.request())?;
    let cancel = options
        .cancel_after
        .map_or_else(Cancel::never, Cancel::after);
    Ok(render(
        &mut service,
        &cancel,
        &mut std::io::stdout().lock(),
    )?)
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::from(2)
        }
    }
}
