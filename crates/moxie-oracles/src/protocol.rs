//! Protocol frames and usage accounting.
//!
//! Document 05 defines the streaming contract: "SSE uses correct content type,
//! stable request IDs, role delta, text/reasoning/tool deltas, finish reason,
//! optional usage event, and `[DONE]`", and "Usage counts prompt tokens and
//! committed completion tokens only, not draft proposals, rejected verification
//! rows or entropy branches."
//!
//! Pinned here: the legal frame order for one generation, the finish reasons,
//! the usage arithmetic including everything it must exclude, and the rule that
//! a stop string is held back across token boundaries rather than leaked.
//!
//! Explicitly **not** pinned here: the HTTP surface itself, JSON shapes, request
//! validation, backpressure and cancellation semantics. Those are M8 and belong
//! with the server; this fixture is the state machine underneath them.

use moxie_types::{Error, Result};

/// One streamed event.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// The role delta that opens a response.
    Role,
    /// Committed, publishable text.
    TextDelta(String),
    /// Reasoning text, where the model's metadata declares the delimiters.
    ReasoningDelta(String),
    /// Terminal, exactly once.
    Finish(FinishReason),
    /// Optional, after `Finish`.
    Usage(Usage),
    /// The stream terminator.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

impl FinishReason {
    pub const fn name(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
            FinishReason::Error => "error",
        }
    }
}

/// Token accounting for one generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    /// Committed completion tokens. Not proposals, not rejected verification
    /// rows, not entropy branches.
    pub completion_tokens: u64,
}

impl Usage {
    pub fn total(self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// What a generation actually did, including the work that is billed to nobody.
///
/// The separation is the point: speculation and entropy lookahead cost real time
/// and real bytes, and document 07 requires them reported as diagnostics. They
/// are not tokens the user asked for and must never enter `Usage`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationWork {
    pub prompt_tokens: u64,
    pub committed_completion_tokens: u64,
    pub draft_proposals: u64,
    pub rejected_proposals: u64,
    pub entropy_branch_steps: u64,
}

impl GenerationWork {
    /// Usage, which counts only two of the five numbers above.
    pub fn usage(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.committed_completion_tokens,
        }
    }
}

/// Validate one generation's frame sequence.
///
/// The rules, from document 05: exactly one `Role`, first; deltas only between
/// `Role` and `Finish`; exactly one `Finish`; `Usage` only after `Finish`;
/// exactly one `Done`, last.
pub fn validate_stream(frames: &[Frame]) -> Result<()> {
    let bad = |detail: String| {
        Err(Error::InvalidRequest {
            field: "stream",
            detail,
        })
    };
    if frames.is_empty() {
        return bad("an empty stream is not a response".into());
    }
    if frames[0] != Frame::Role {
        return bad("a stream must open with the role delta".into());
    }
    if *frames.last().unwrap() != Frame::Done {
        return bad("a stream must end with [DONE]".into());
    }

    let mut finished = false;
    let mut roles = 0;
    let mut finishes = 0;
    let mut dones = 0;
    for (i, f) in frames.iter().enumerate() {
        match f {
            Frame::Role => {
                roles += 1;
                if i != 0 {
                    return bad(format!("role delta at position {i}"));
                }
            }
            Frame::TextDelta(_) | Frame::ReasoningDelta(_) => {
                if finished {
                    return bad(format!("delta at position {i} after the finish reason"));
                }
            }
            Frame::Finish(_) => {
                finishes += 1;
                finished = true;
            }
            Frame::Usage(_) => {
                if !finished {
                    return bad(format!("usage at position {i} before the finish reason"));
                }
            }
            Frame::Done => {
                dones += 1;
                if i + 1 != frames.len() {
                    return bad(format!("[DONE] at position {i} is not last"));
                }
            }
        }
    }
    if roles != 1 || finishes != 1 || dones != 1 {
        return bad(format!(
            "expected exactly one role, finish and done; got {roles}, {finishes}, {dones}"
        ));
    }
    Ok(())
}

/// Buffers text so a stop string is never emitted, even across chunks.
///
/// Document 05: "Stop strings may cross token/UTF-8 boundaries; hold the
/// required suffix and do not leak stop text."
#[derive(Debug, Clone)]
pub struct StopFilter {
    stop: String,
    held: String,
    stopped: bool,
}

impl StopFilter {
    pub fn new(stop: &str) -> Result<Self> {
        if stop.is_empty() {
            return Err(Error::InvalidRequest {
                field: "stop",
                detail: "an empty stop string would stop immediately".into(),
            });
        }
        Ok(Self {
            stop: stop.to_string(),
            held: String::new(),
            stopped: false,
        })
    }

    /// Feed a chunk; returns the text that is safe to publish now.
    pub fn push(&mut self, chunk: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.held.push_str(chunk);
        if let Some(at) = self.held.find(&self.stop) {
            self.stopped = true;
            let out = self.held[..at].to_string();
            self.held.clear();
            return out;
        }
        // Hold back any suffix that could still become the stop string.
        let keep = longest_prefix_suffix(&self.held, &self.stop);
        let split = self.held.len() - keep;
        let out = self.held[..split].to_string();
        self.held = self.held[split..].to_string();
        out
    }

    /// End of generation: release whatever was held, unless the stop fired.
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of `stop`.
fn longest_prefix_suffix(text: &str, stop: &str) -> usize {
    let max = stop.len().saturating_sub(1).min(text.len());
    for n in (1..=max).rev() {
        let start = text.len() - n;
        if text.is_char_boundary(start) && stop.starts_with(&text[start..]) {
            return n;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Frame {
        Frame::TextDelta(s.to_string())
    }

    #[test]
    fn a_well_formed_stream_is_accepted() {
        let s = vec![
            Frame::Role,
            text("Hel"),
            text("lo"),
            Frame::Finish(FinishReason::Stop),
            Frame::Usage(Usage {
                prompt_tokens: 4,
                completion_tokens: 2,
            }),
            Frame::Done,
        ];
        validate_stream(&s).unwrap();

        // Usage is optional.
        let without = vec![
            Frame::Role,
            text("hi"),
            Frame::Finish(FinishReason::Length),
            Frame::Done,
        ];
        validate_stream(&without).unwrap();
    }

    #[test]
    fn every_ordering_rule_is_enforced() {
        assert!(validate_stream(&[]).is_err());
        // No role delta.
        assert!(
            validate_stream(&[text("hi"), Frame::Finish(FinishReason::Stop), Frame::Done]).is_err()
        );
        // No terminator.
        assert!(validate_stream(&[Frame::Role, Frame::Finish(FinishReason::Stop)]).is_err());
        // Two finish reasons.
        assert!(
            validate_stream(&[
                Frame::Role,
                Frame::Finish(FinishReason::Stop),
                Frame::Finish(FinishReason::Length),
                Frame::Done
            ])
            .is_err()
        );
        // A delta after the finish reason: text the client will never attribute.
        assert!(
            validate_stream(&[
                Frame::Role,
                Frame::Finish(FinishReason::Stop),
                text("late"),
                Frame::Done
            ])
            .is_err()
        );
        // Usage before the finish reason.
        assert!(
            validate_stream(&[
                Frame::Role,
                Frame::Usage(Usage::default()),
                Frame::Finish(FinishReason::Stop),
                Frame::Done
            ])
            .is_err()
        );
        // Anything after [DONE].
        assert!(
            validate_stream(&[
                Frame::Role,
                Frame::Finish(FinishReason::Stop),
                Frame::Done,
                text("after")
            ])
            .is_err()
        );
        // A second role delta.
        assert!(
            validate_stream(&[
                Frame::Role,
                Frame::Role,
                Frame::Finish(FinishReason::Stop),
                Frame::Done
            ])
            .is_err()
        );
    }

    #[test]
    fn a_cancelled_generation_still_produces_a_complete_stream() {
        // R08: resources are released at cancellation even though no next token
        // arrives, and the client still gets a terminated stream.
        validate_stream(&[
            Frame::Role,
            text("partial"),
            Frame::Finish(FinishReason::Cancelled),
            Frame::Done,
        ])
        .unwrap();
    }

    #[test]
    fn reasoning_deltas_are_separate_from_text_deltas() {
        // Document 05: reasoning and text are distinct deltas, so a client can
        // present them differently. Collapsing them loses that.
        let s = vec![
            Frame::Role,
            Frame::ReasoningDelta("thinking".into()),
            text("answer"),
            Frame::Finish(FinishReason::Stop),
            Frame::Done,
        ];
        validate_stream(&s).unwrap();
        assert_ne!(
            Frame::ReasoningDelta("x".into()),
            Frame::TextDelta("x".into())
        );
    }

    #[test]
    fn usage_excludes_drafts_rejections_and_entropy_branches() {
        // The accounting rule in one assertion. A generation that proposed 40
        // draft tokens, had 12 rejected and ran 60 entropy branch steps still
        // bills exactly the prompt and the committed completion.
        let work = GenerationWork {
            prompt_tokens: 100,
            committed_completion_tokens: 28,
            draft_proposals: 40,
            rejected_proposals: 12,
            entropy_branch_steps: 60,
        };
        assert_eq!(
            work.usage(),
            Usage {
                prompt_tokens: 100,
                completion_tokens: 28
            }
        );
        assert_eq!(work.usage().total(), 128);

        // Doing more speculative work does not change usage at all.
        let harder = GenerationWork {
            draft_proposals: 4000,
            entropy_branch_steps: 9999,
            ..work
        };
        assert_eq!(harder.usage(), work.usage());
    }

    #[test]
    fn finish_reasons_are_distinct_and_stable() {
        let all = [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::Cancelled,
            FinishReason::Error,
        ];
        let mut names: Vec<_> = all.iter().map(|r| r.name()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(n, names.len());
    }

    #[test]
    fn a_stop_string_split_across_chunks_is_never_leaked() {
        // The failure this guards: emitting "END" one character at a time
        // because no single chunk contained the whole stop string.
        let mut f = StopFilter::new("END").unwrap();
        assert_eq!(f.push("hello E"), "hello ");
        assert_eq!(f.push("N"), "");
        assert_eq!(f.push("D and more"), "");
        assert!(f.stopped());
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn text_that_only_looks_like_the_stop_string_is_released() {
        let mut f = StopFilter::new("END").unwrap();
        assert_eq!(f.push("EN"), "", "held: EN could still become END");
        assert_eq!(f.push("OUGH"), "ENOUGH");
        assert!(!f.stopped());
        assert_eq!(f.push(" done"), " done");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn a_held_suffix_is_released_at_the_end_of_generation() {
        let mut f = StopFilter::new("END").unwrap();
        assert_eq!(f.push("the E"), "the ");
        assert_eq!(
            f.finish(),
            "E",
            "a held suffix that never completed is text"
        );
    }

    #[test]
    fn a_stop_string_inside_one_chunk_truncates_it() {
        let mut f = StopFilter::new("END").unwrap();
        assert_eq!(f.push("before END after"), "before ");
        assert!(f.stopped());
        assert_eq!(f.push("more"), "");
    }

    #[test]
    fn the_filter_holds_at_a_character_boundary_not_a_byte_boundary() {
        // A multi-byte stop string, fed one byte-ish chunk at a time. Slicing
        // mid-character would panic; the filter must hold instead.
        let mut f = StopFilter::new("→END").unwrap();
        assert_eq!(f.push("go "), "go ");
        assert_eq!(f.push("→"), "");
        assert_eq!(f.push("EN"), "");
        assert_eq!(f.push("D"), "");
        assert!(f.stopped());

        let mut g = StopFilter::new("🌍").unwrap();
        assert_eq!(g.push("hi 🌍 there"), "hi ");
        assert!(g.stopped());
    }

    #[test]
    fn an_empty_stop_string_is_refused() {
        assert!(StopFilter::new("").is_err());
    }
}
