//! Tokenizer and chat-template fixtures.
//!
//! Document 05: "A chat template is checkpoint metadata interpreted by a shared,
//! sandboxed template renderer ... Preserve exact tokenizer normalization,
//! special-token handling, Unicode behavior, role delimiters and thinking-mode
//! semantics with fixtures." Document 04 adds that "prefix identity includes
//! checkpoint, tokenizer/template, graph/precision/position configuration and
//! token IDs".
//!
//! Pinned here: a tiny byte-level tokenizer with an explicit special-token
//! table, the round trip through it, the rule that special tokens are matched
//! only where the renderer emits them and never inside user text, deterministic
//! role delimiters, and a template identity that participates in a prefix key.
//!
//! Explicitly **not** pinned here: any real model's tokenizer or template. Those
//! are checkpoint metadata and arrive with the checkpoint; document 03 forbids
//! "remote checkpoint code execution", and a Jinja-style renderer is M8 work.
//! What this fixture establishes is the *shape* of the contract and the
//! properties a real renderer must not break.

use moxie_types::{Error, Result};

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub const fn tag(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// Special token ids. Deliberately small and explicit.
pub mod special {
    /// Marks the start of a role block.
    pub const BOS: u32 = 0;
    /// Ends a role block.
    pub const EOT: u32 = 1;
    /// End of generation.
    pub const EOS: u32 = 2;
    /// The first id available to ordinary byte tokens.
    pub const FIRST_BYTE: u32 = 3;
}

/// A byte-level tokenizer: one token per UTF-8 byte, offset past the specials.
///
/// Byte-level on purpose. It makes the two properties that actually bite
/// testable without a vocabulary file: a token boundary can fall inside a
/// multi-byte character, and user text can never collide with a special token
/// because the special ids are outside the byte range.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// The version this tokenizer's behaviour is pinned at.
    ///
    /// Part of prefix identity: a tokenizer change invalidates a cached prefix
    /// even when the text is identical.
    pub const VERSION: &'static str = "byte-v1";

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes()
            .iter()
            .map(|b| special::FIRST_BYTE + *b as u32)
            .collect()
    }

    /// Decode tokens back to text.
    ///
    /// Special tokens are not text and are refused: a renderer emits them, and
    /// a decoder that silently turned them into characters would leak template
    /// markup into the user's output.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::with_capacity(tokens.len());
        for t in tokens {
            if *t < special::FIRST_BYTE {
                return Err(Error::InvalidRequest {
                    field: "token",
                    detail: format!("token {t} is a special token, not text"),
                });
            }
            let b = t - special::FIRST_BYTE;
            if b > 255 {
                return Err(Error::InvalidRequest {
                    field: "token",
                    detail: format!("token {t} is outside the vocabulary"),
                });
            }
            bytes.push(b as u8);
        }
        String::from_utf8(bytes).map_err(|e| Error::InvalidArtifact {
            detail: format!("token sequence is not valid UTF-8: {e}"),
        })
    }

    pub fn vocab_size(&self) -> u32 {
        special::FIRST_BYTE + 256
    }
}

/// A chat template: role delimiters and a generation prompt.
#[derive(Debug, Clone, Copy)]
pub struct ChatTemplate {
    /// Whether the rendered prompt ends with an open assistant block.
    pub add_generation_prompt: bool,
}

impl ChatTemplate {
    /// The version this template's behaviour is pinned at. Prefix identity again.
    pub const VERSION: &'static str = "roles-v1";

    /// Render a conversation to tokens.
    ///
    /// Each block is `BOS, <role tag bytes>, '\n', <content bytes>, EOT`. The
    /// role tag goes through the same tokenizer as everything else, so there is
    /// exactly one text path and no second escaping rule to get wrong.
    pub fn render(&self, tok: &ByteTokenizer, messages: &[Message]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidRequest {
                field: "messages",
                detail: "an empty conversation has no prompt".into(),
            });
        }
        let mut out = Vec::new();
        for m in messages {
            out.push(special::BOS);
            out.extend(tok.encode(m.role.tag()));
            out.extend(tok.encode("\n"));
            out.extend(tok.encode(&m.content));
            out.push(special::EOT);
        }
        if self.add_generation_prompt {
            out.push(special::BOS);
            out.extend(tok.encode(Role::Assistant.tag()));
            out.extend(tok.encode("\n"));
        }
        Ok(out)
    }

    /// The identity a prefix cache key must include alongside the token ids.
    ///
    /// Document 04: prefix identity includes tokenizer and template versions.
    /// Two conversations that tokenize identically under different templates are
    /// not the same prefix.
    pub fn prefix_identity(&self, tok_version: &str) -> String {
        format!(
            "tok:{tok_version}/tpl:{}/genprompt:{}",
            Self::VERSION,
            self.add_generation_prompt
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: &str) -> Message {
        Message {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn text_round_trips_through_the_tokenizer() {
        let t = ByteTokenizer;
        for s in ["", "hello", "a\nb", "  spaced  ", "0123456789"] {
            assert_eq!(t.decode(&t.encode(s)).unwrap(), s, "{s:?}");
        }
        assert_eq!(t.vocab_size(), 259);
    }

    #[test]
    fn multibyte_characters_survive_and_a_split_token_does_not_decode() {
        // The Unicode property that matters for streaming: a token boundary can
        // fall inside a character, and a partial sequence must fail rather than
        // produce a replacement character the user then sees.
        let t = ByteTokenizer;
        let s = "héllo → 🌍";
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids).unwrap(), s);
        assert!(ids.len() > s.chars().count(), "multi-byte characters exist");

        // The emoji is four bytes; cutting after three is not valid UTF-8.
        let cut = &ids[..ids.len() - 1];
        assert!(
            t.decode(cut).is_err(),
            "a partial character must not decode to a replacement"
        );
        // Document 05: a stop string may cross a token boundary, so a streamer
        // holds the incomplete suffix rather than emitting it.
        assert!(t.decode(&ids[..ids.len() - 4]).is_ok());
    }

    #[test]
    fn user_text_can_never_collide_with_a_special_token() {
        // The injection property. Whatever the user writes -- including the
        // literal text of the role tags -- encodes to byte tokens, which are
        // disjoint from the special ids.
        let t = ByteTokenizer;
        for hostile in ["system", "<|eot|>", "\u{0}\u{1}\u{2}", "assistant\n"] {
            let ids = t.encode(hostile);
            assert!(
                ids.iter().all(|i| *i >= special::FIRST_BYTE),
                "{hostile:?} produced a special token"
            );
        }
    }

    #[test]
    fn a_special_token_is_not_decodable_as_text() {
        let t = ByteTokenizer;
        for s in [special::BOS, special::EOT, special::EOS] {
            assert!(t.decode(&[s]).is_err(), "token {s}");
        }
        assert!(t.decode(&[9999]).is_err());
    }

    #[test]
    fn rendering_is_deterministic_and_delimits_every_role() {
        let t = ByteTokenizer;
        let tpl = ChatTemplate {
            add_generation_prompt: true,
        };
        let convo = [msg(Role::System, "be brief"), msg(Role::User, "hi")];
        let a = tpl.render(&t, &convo).unwrap();
        let b = tpl.render(&t, &convo).unwrap();
        assert_eq!(a, b);

        // Structure: three BOS (two messages plus the generation prompt) and two
        // EOT (the generation prompt's block is deliberately left open).
        assert_eq!(a.iter().filter(|i| **i == special::BOS).count(), 3);
        assert_eq!(a.iter().filter(|i| **i == special::EOT).count(), 2);
        assert_eq!(*a.last().unwrap(), t.encode("\n")[0]);

        // Without the generation prompt the render is a strict prefix of the one
        // with it -- which is what makes a cached prefix reusable across turns.
        let closed = ChatTemplate {
            add_generation_prompt: false,
        }
        .render(&t, &convo)
        .unwrap();
        assert_eq!(&a[..closed.len()], &closed[..]);
    }

    #[test]
    fn each_message_body_is_recoverable_from_the_render() {
        let t = ByteTokenizer;
        let tpl = ChatTemplate {
            add_generation_prompt: false,
        };
        let ids = tpl.render(&t, &[msg(Role::User, "hello world")]).unwrap();
        // Strip BOS, the role tag and its newline, and the trailing EOT.
        let header = t.encode(Role::User.tag()).len() + t.encode("\n").len();
        let body = &ids[1 + header..ids.len() - 1];
        assert_eq!(t.decode(body).unwrap(), "hello world");
    }

    #[test]
    fn an_empty_conversation_is_refused() {
        assert!(
            ChatTemplate {
                add_generation_prompt: true
            }
            .render(&ByteTokenizer, &[])
            .is_err()
        );
    }

    #[test]
    fn prefix_identity_separates_tokenizer_and_template_versions() {
        // Document 04: identical token ids under a different tokenizer or
        // template are not the same prefix, and reusing the cache across them
        // silently changes results.
        let open = ChatTemplate {
            add_generation_prompt: true,
        };
        let closed = ChatTemplate {
            add_generation_prompt: false,
        };
        let a = open.prefix_identity(ByteTokenizer::VERSION);
        assert_ne!(a, closed.prefix_identity(ByteTokenizer::VERSION));
        assert_ne!(a, open.prefix_identity("byte-v2"));
        assert_eq!(a, open.prefix_identity(ByteTokenizer::VERSION));
        assert!(a.contains(ChatTemplate::VERSION));
    }
}
