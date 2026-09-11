//! Presentation-only diagnostic client for the shared generation service.
#![forbid(unsafe_code)]
pub mod fixture;

use moxie_engine::{Cancel, GenerationEvent, GenerationRequest, service::GenerationService};
use std::io::Write;

pub const USAGE: &str = "moxie diagnostic [--shape a|b] [--prompt 0,1,2] [--max-new 4] [--chunk 2] [--temperature 0] [--seed 0] [--cancel-after N]\nExplicit host-reference synthetic token-ID diagnostics; context <=256. No checkpoint or GPU attention.";

/// Render the same typed events an API client would consume. Broken output is a
/// disconnect: the owning caller drops the service and releases the generation.
pub fn render(
    service: &mut GenerationService<'_, '_>,
    cancel: &Cancel,
    out: &mut impl Write,
) -> std::io::Result<bool> {
    let mut success = true;
    while let Some(event) = service.next_event(cancel) {
        match event {
            GenerationEvent::Admitted {
                profile,
                sampler,
                temperature,
                seed,
                context,
                reserved_bytes,
            } => writeln!(
                out,
                "event=admitted profile={profile} synthetic=true sampler={sampler} temperature={temperature} seed={seed} context={context} reserved_bytes={reserved_bytes}"
            )?,
            GenerationEvent::Prefill { processed, total } => {
                writeln!(out, "event=prefill processed={processed} total={total}")?
            }
            GenerationEvent::Token { id, position } => {
                writeln!(out, "event=token id={id} position={position}")?
            }
            GenerationEvent::Finished { usage } => writeln!(
                out,
                "event=finished reason=length prompt_tokens={} completion_tokens={}",
                usage.prompt_tokens, usage.completion_tokens
            )?,
            GenerationEvent::Cancelled { usage } => {
                success = false;
                writeln!(
                    out,
                    "event=cancelled prompt_tokens={} completion_tokens={}",
                    usage.prompt_tokens, usage.completion_tokens
                )?;
            }
            GenerationEvent::Failed { error, usage } => {
                success = false;
                writeln!(
                    out,
                    "event=failed kind={} prompt_tokens={} completion_tokens={} detail={error}",
                    error.kind(),
                    usage.prompt_tokens,
                    usage.completion_tokens
                )?;
            }
        }
        out.flush()?;
    }
    Ok(success)
}

#[derive(Debug)]
pub struct Options {
    pub shape_b: bool,
    pub prompt: Vec<u32>,
    pub maximum: usize,
    pub chunk: usize,
    pub temperature: f64,
    pub seed: u64,
    pub cancel_after: Option<u64>,
}
impl Options {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        if args.first().map(String::as_str) != Some("diagnostic") || args.len() > 17 {
            return Err(USAGE.into());
        }
        let mut options = Self {
            shape_b: false,
            prompt: vec![0, 1, 2],
            maximum: 4,
            chunk: 2,
            temperature: 0.0,
            seed: 0,
            cancel_after: None,
        };
        let mut seen = std::collections::BTreeSet::new();
        for pair in args[1..].chunks(2) {
            if pair.len() != 2 || !seen.insert(pair[0].as_str()) {
                return Err("each option requires one value and may appear only once".into());
            }
            let value = &pair[1];
            let malformed = || format!("invalid {}", pair[0]);
            match pair[0].as_str() {
                "--shape" => {
                    options.shape_b = match value.as_str() {
                        "a" => false,
                        "b" => true,
                        _ => return Err(malformed()),
                    }
                }
                "--prompt" => {
                    if value.len() > 4096 {
                        return Err("prompt argument exceeds 4096 bytes".into());
                    }
                    options.prompt = value
                        .split(',')
                        .map(|t| t.parse().map_err(|_| malformed()))
                        .collect::<Result<_, _>>()?;
                    if options.prompt.len() > 256 {
                        return Err("prompt exceeds diagnostic context".into());
                    }
                }
                "--max-new" => options.maximum = value.parse().map_err(|_| malformed())?,
                "--chunk" => options.chunk = value.parse().map_err(|_| malformed())?,
                "--temperature" => options.temperature = value.parse().map_err(|_| malformed())?,
                "--seed" => options.seed = value.parse().map_err(|_| malformed())?,
                "--cancel-after" => {
                    options.cancel_after = Some(value.parse().map_err(|_| malformed())?)
                }
                _ => return Err(format!("unsupported option {}", pair[0])),
            }
        }
        Ok(options)
    }
    pub fn request(&self) -> GenerationRequest<'_> {
        GenerationRequest {
            prompt: &self.prompt,
            max_new_tokens: self.maximum,
            prefill_chunk: self.chunk,
            temperature: self.temperature,
            seed: self.seed,
        }
    }
}
