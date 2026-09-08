//! Attention masks, and the exact small attention they feed.
//!
//! Document 04 requires a tested capability matrix for "full and sliding
//! attention ... variable and partial prefill chunks; paged decode and append;
//! page boundaries/ring wrap". Document 07 puts "masks/indices" among the
//! required exact checks.
//!
//! Pinned here: causal visibility, sliding windows, chunked prefill at position
//! zero *and* at a later chunk, partial and tail chunks, page boundaries, and an
//! exact softmax attention that proves a masked position contributes nothing.
//!
//! Explicitly **not** pinned here: GQA head mapping, attention sinks and biases,
//! MLA latent projections, model-defined sparse selection, and the online-
//! softmax merge for streamed pages. The merge in particular is document 04
//! mathematics that deserves its own fixture when the streaming path exists.

use moxie_types::{Error, Result};

/// Which keys a query position may attend to.
///
/// Defined in `moxie-graph`, because a visibility rule is part of the attention
/// operation's contract rather than of the reference that evaluates it
/// (document 04's descriptor fields). Re-exported here so that this module stays
/// the place its behaviour is pinned.
pub use moxie_graph::Visibility;

/// A prefill chunk: `len` query positions starting at absolute position `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub start: u64,
    pub len: u64,
}

/// The mask for one chunk against a history of `visible_keys` absolute
/// positions, row-major `[chunk row][key position]`.
pub fn chunk_mask(chunk: Chunk, visible_keys: u64, vis: Visibility) -> Result<Vec<Vec<bool>>> {
    let end = chunk
        .start
        .checked_add(chunk.len)
        .ok_or(Error::InvalidRequest {
            field: "chunk",
            detail: "chunk end overflows".into(),
        })?;
    if end > visible_keys {
        return Err(Error::InvalidRequest {
            field: "chunk",
            detail: format!(
                "chunk covers positions {}..{end} but only {visible_keys} key(s) exist; \
                 a query cannot precede its own key",
                chunk.start
            ),
        });
    }
    Ok((0..chunk.len)
        .map(|r| {
            let q = chunk.start + r;
            (0..visible_keys).map(|k| vis.allows(q, k)).collect()
        })
        .collect())
}

/// Split a sequence into prefill chunks of at most `width`, leaving a short tail.
///
/// R21: "test position zero AND later chunks, partial widths, short tails".
pub fn chunk_plan(total: u64, width: u64) -> Result<Vec<Chunk>> {
    if width == 0 {
        return Err(Error::InvalidRequest {
            field: "chunk_width",
            detail: "zero-width prefill chunk".into(),
        });
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let len = width.min(total - start);
        out.push(Chunk { start, len });
        start += len;
    }
    Ok(out)
}

/// Exact single-head attention over f32, for one query row.
///
/// The oracle: full precision, no tiling, no online softmax, masked entries
/// removed from the sum rather than given a large negative bias. Document 07
/// wants an "independent FP32/FP64 oracle", and the simplest correct thing is
/// the one to compare a Flash-style kernel against.
///
/// Returns an error rather than a uniform draw when no key is visible: document
/// 05 forbids inheriting a silent fallback for an empty candidate set, and the
/// same applies here.
pub fn attend_row(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    allowed: &[bool],
    scale: f32,
) -> Result<Vec<f32>> {
    if keys.len() != values.len() || keys.len() != allowed.len() {
        return Err(Error::InvalidRequest {
            field: "attention",
            detail: format!(
                "{} key(s), {} value(s), {} mask entries",
                keys.len(),
                values.len(),
                allowed.len()
            ),
        });
    }
    let visible: Vec<usize> = (0..keys.len()).filter(|i| allowed[*i]).collect();
    if visible.is_empty() {
        return Err(Error::Numerical {
            detail: "no visible key for this query position".into(),
        });
    }
    let value_dim = values[visible[0]].len();

    let mut scores = Vec::with_capacity(visible.len());
    for i in &visible {
        if keys[*i].len() != query.len() {
            return Err(Error::InvalidRequest {
                field: "key",
                detail: format!("key {i} has dimension {}", keys[*i].len()),
            });
        }
        let dot: f32 = query.iter().zip(keys[*i].iter()).map(|(a, b)| a * b).sum();
        scores.push(dot * scale);
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let denom: f32 = exps.iter().sum();

    let mut out = vec![0f32; value_dim];
    for (n, i) in visible.iter().enumerate() {
        if values[*i].len() != value_dim {
            return Err(Error::InvalidRequest {
                field: "value",
                detail: format!("value {i} has dimension {}", values[*i].len()),
            });
        }
        let w = exps[n] / denom;
        for (o, v) in out.iter_mut().zip(values[*i].iter()) {
            *o += w * v;
        }
    }
    Ok(out)
}

/// Which page a position lives on, for a paged cache with `page` positions each.
pub const fn page_of(position: u64, page: u64) -> u64 {
    position / page
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        v[i] = 1.0;
        v
    }

    #[test]
    fn causal_visibility_is_exactly_at_or_before() {
        for q in 0..8u64 {
            for k in 0..8u64 {
                assert_eq!(Visibility::Causal.allows(q, k), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn a_sliding_window_sees_exactly_its_own_width() {
        let w = Visibility::SlidingWindow { window: 3 };
        // Position 10 sees 8, 9, 10 and nothing else.
        assert!(!w.allows(10, 7));
        assert!(w.allows(10, 8));
        assert!(w.allows(10, 9));
        assert!(w.allows(10, 10));
        assert!(!w.allows(10, 11));

        // A window of 1 is self-attention only.
        let one = Visibility::SlidingWindow { window: 1 };
        assert!(one.allows(5, 5));
        assert!(!one.allows(5, 4));

        // Early positions are not clipped into negative territory.
        assert!(w.allows(0, 0));
        assert!(!w.allows(0, 1));
    }

    #[test]
    fn a_masked_position_contributes_nothing_at_all() {
        // The property that matters: not "contributes a little", but exactly
        // zero. A large-negative-bias implementation is only approximately this.
        let query = vec![1.0f32, 0.0];
        let keys = vec![unit(0, 2), unit(0, 2), unit(0, 2)];
        let values = vec![vec![1.0f32], vec![100.0], vec![3.0]];

        let with_all = attend_row(&query, &keys, &values, &[true, true, true], 1.0).unwrap();
        let without_middle = attend_row(&query, &keys, &values, &[true, false, true], 1.0).unwrap();

        // Keys are identical, so visible values are averaged uniformly.
        assert!((with_all[0] - 104.0 / 3.0).abs() < 1e-4, "{with_all:?}");
        assert!((without_middle[0] - 2.0).abs() < 1e-6, "{without_middle:?}");

        // And the masked value can be anything at all without moving the result.
        let mut absurd = values.clone();
        absurd[1] = vec![1e30];
        let again = attend_row(&query, &keys, &absurd, &[true, false, true], 1.0).unwrap();
        assert_eq!(again, without_middle);
    }

    #[test]
    fn attention_weights_sum_to_one_over_the_visible_set() {
        let query = vec![0.3f32, -1.2];
        let keys: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 0.25, 1.0 - i as f32])
            .collect();
        // Each value is a distinct basis vector, so the output *is* the weights.
        let values: Vec<Vec<f32>> = (0..5).map(|i| unit(i, 5)).collect();
        let out = attend_row(
            &query,
            &keys,
            &values,
            &[true, true, false, true, false],
            0.7,
        )
        .unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to {sum}");
        assert_eq!(out[2], 0.0);
        assert_eq!(out[4], 0.0);
        assert!(out[0] > 0.0 && out[1] > 0.0 && out[3] > 0.0);
    }

    #[test]
    fn an_empty_visible_set_is_a_typed_failure_not_a_uniform_draw() {
        let e = attend_row(&[1.0], &[vec![1.0]], &[vec![1.0]], &[false], 1.0).unwrap_err();
        assert_eq!(e.kind(), "numerical");
    }

    #[test]
    fn chunked_prefill_matches_whole_prefill_row_for_row() {
        // Document 06 M1's exit gate in miniature: "whole-versus-chunked prefill
        // parity on small fixtures".
        let total = 7u64;
        let whole = chunk_mask(
            Chunk {
                start: 0,
                len: total,
            },
            total,
            Visibility::Causal,
        )
        .unwrap();

        for width in 1..=total {
            let mut rebuilt: Vec<Vec<bool>> = Vec::new();
            for c in chunk_plan(total, width).unwrap() {
                rebuilt.extend(chunk_mask(c, total, Visibility::Causal).unwrap());
            }
            assert_eq!(rebuilt, whole, "width {width}");
        }
    }

    #[test]
    fn a_later_chunk_is_not_the_first_chunk_shifted() {
        // R21: the failure mode is indexing within a chunk instead of by
        // absolute position. Chunk 1 of a causal sequence sees the whole of
        // chunk 0; a within-chunk implementation would produce a lower triangle
        // again and lose that history.
        let first = chunk_mask(Chunk { start: 0, len: 4 }, 8, Visibility::Causal).unwrap();
        let later = chunk_mask(Chunk { start: 4, len: 4 }, 8, Visibility::Causal).unwrap();

        assert_eq!(
            first[0],
            vec![true, false, false, false, false, false, false, false]
        );
        assert_eq!(
            later[0],
            vec![true, true, true, true, true, false, false, false]
        );
        assert_ne!(first[0], later[0]);
        // Every row of the later chunk sees all four earlier positions.
        for row in &later {
            assert!(row[..4].iter().all(|v| *v));
        }
    }

    #[test]
    fn a_short_tail_and_a_partial_width_are_handled() {
        let plan = chunk_plan(10, 4).unwrap();
        assert_eq!(
            plan,
            vec![
                Chunk { start: 0, len: 4 },
                Chunk { start: 4, len: 4 },
                Chunk { start: 8, len: 2 },
            ]
        );
        let tail = chunk_mask(plan[2], 10, Visibility::Causal).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[1].iter().filter(|v| **v).count(), 10);

        // A single-position chunk, which is what decode is.
        let decode = chunk_mask(Chunk { start: 9, len: 1 }, 10, Visibility::Causal).unwrap();
        assert_eq!(decode.len(), 1);
        assert!(decode[0].iter().all(|v| *v));

        assert!(chunk_plan(10, 0).is_err());
        assert_eq!(chunk_plan(0, 4).unwrap(), vec![]);
    }

    #[test]
    fn a_chunk_beyond_the_visible_history_is_refused() {
        // A query whose own key has not been written yet is a scheduling defect,
        // not something to mask around.
        assert!(chunk_mask(Chunk { start: 4, len: 4 }, 6, Visibility::Causal).is_err());
        assert!(chunk_mask(Chunk { start: 4, len: 4 }, 8, Visibility::Causal).is_ok());
        assert!(
            chunk_mask(
                Chunk {
                    start: u64::MAX,
                    len: 2
                },
                8,
                Visibility::Causal
            )
            .is_err()
        );
    }

    #[test]
    fn a_sliding_window_chunk_drops_history_the_window_excludes() {
        let m = chunk_mask(
            Chunk { start: 4, len: 2 },
            6,
            Visibility::SlidingWindow { window: 3 },
        )
        .unwrap();
        assert_eq!(m[0], vec![false, false, true, true, true, false]);
        assert_eq!(m[1], vec![false, false, false, true, true, true]);
    }

    #[test]
    fn page_boundaries_fall_where_the_page_size_says() {
        // Document 04 asks for page boundary and ring-wrap coverage. The
        // boundary itself is the arithmetic; the ring is the cache's business.
        for page in [1u64, 2, 8, 16] {
            for pos in 0..40u64 {
                assert_eq!(page_of(pos, page), pos / page);
            }
            // The last position of one page and the first of the next differ.
            assert_ne!(page_of(page - 1, page), page_of(page, page));
        }
        // A chunk that straddles a boundary spans two pages, which is the case a
        // one-page fast path gets wrong.
        let c = Chunk { start: 6, len: 4 };
        assert_ne!(page_of(c.start, 8), page_of(c.start + c.len - 1, 8));
    }
}
