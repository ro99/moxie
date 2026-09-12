//! Presentation-only diagnostic client for the shared generation service.
#![forbid(unsafe_code)]
pub mod fixture;
pub mod gemma;

use moxie_engine::{Cancel, GenerationEvent, GenerationRequest, service::GenerationService};
use std::io::Write;

pub const USAGE: &str = "moxie diagnostic [--shape a|b|gemma-a|gemma-b] [--prompt 0,1,2] [--max-new 4] [--chunk 2] [--temperature 0] [--seed 0] [--cancel-after N]\nExplicit host-reference synthetic token-ID diagnostics; context <=256. No checkpoint or GPU attention.\nThe gemma shapes are reduced Gemma-4-like graphs over synthetic BF16 weights: contract fixtures, not model support.";

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

/// Which diagnostic graph to build.
///
/// Named rather than a boolean: there are four now, and a `shape_b: bool` that
/// grew a second flag beside it is how a composition root starts making
/// decisions it should not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Task 0015's synthetic graph: interleaved RoPE, `1/sqrt(head_dim)`
    /// scores, multi-head attention, SwiGLU, no scales and no cap. It is the
    /// independent second consumer of every parameter the Gemma shapes set
    /// differently.
    SyntheticA,
    SyntheticB,
    GemmaA,
    GemmaB,
}

impl Shape {
    pub const fn name(self) -> &'static str {
        match self {
            Shape::SyntheticA => "a",
            Shape::SyntheticB => "b",
            Shape::GemmaA => gemma::Shape::A.name(),
            Shape::GemmaB => gemma::Shape::B.name(),
        }
    }

    pub fn build(self) -> moxie_types::Result<fixture::Fixture> {
        match self {
            Shape::SyntheticA => fixture::build(2, 4, 16, 1),
            Shape::SyntheticB => fixture::build(3, 4, 7, 2),
            Shape::GemmaA => gemma::build(gemma::Shape::A),
            Shape::GemmaB => gemma::build(gemma::Shape::B),
        }
    }

    /// The reduction disclosure line, empty for the non-model fixtures.
    pub fn reduction(self) -> String {
        match self {
            Shape::SyntheticA | Shape::SyntheticB => String::new(),
            Shape::GemmaA | Shape::GemmaB => {
                gemma::reduction_line(moxie_models::gemma4::Reduction::all())
            }
        }
    }
}

#[derive(Debug)]
pub struct Options {
    pub shape: Shape,
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
            shape: Shape::SyntheticA,
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
                    options.shape = match value.as_str() {
                        "a" => Shape::SyntheticA,
                        "b" => Shape::SyntheticB,
                        "gemma-a" => Shape::GemmaA,
                        "gemma-b" => Shape::GemmaB,
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
