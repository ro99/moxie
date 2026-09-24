//! FP64 reference algebra for the unabsorbed GLM-5.2-style MLA path.
//!
//! The source contract is `src/models/glm52/glm52_runtime.cpp:46-49` for the
//! ranks and head slices, `:824-845` for query down projection/RMSNorm/up
//! projection, `:846-855` for query RoPE, `:857-882` for KV down
//! projection/RMSNorm/RoPE, `:884-892` for cache append, and
//! `src/models/glm52/glm52_ops.cpp:85-114` for the split-write RoPE
//! permutation. Latent KV decompression is `glm52_runtime.cpp:977-1001`.
//! The implementation intentionally keeps those stages visible: it does not
//! absorb `kv_b_proj` into the query path and it does not include GLM-5.2's
//! separate sparse indexer.
//!
//! All arithmetic here is FP64. Inputs are flat row-major tensors with the
//! same `[out, in]` weight convention as the graph descriptor. This module is a
//! host reference only; it reads no checkpoint data and has no device path.

use moxie_graph::{MlaAttentionDescriptor, RopeLayout};
use moxie_types::{Error, Result};

/// The seven projection/norm tensors named by the GLM-5.2 MLA checkpoint
/// metadata. Matrices are row-major `[out, in]`; norm gains are one-dimensional.
#[derive(Debug, Clone, Copy)]
pub struct MlaWeights<'a> {
    pub q_a_proj: &'a [f64],
    pub q_a_layernorm: &'a [f64],
    pub q_b_proj: &'a [f64],
    pub kv_a_proj_with_mqa: &'a [f64],
    pub kv_a_layernorm: &'a [f64],
    pub kv_b_proj: &'a [f64],
    pub o_proj: &'a [f64],
}

/// One projected query and one latent/positional row produced from one input
/// row.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaProjection {
    pub position: u64,
    pub query: Vec<f64>,
    pub cached_latent: Vec<f64>,
    pub cached_rope: Vec<f64>,
}

impl MlaProjection {
    /// Turn this projection into the two-part row retained by latent state.
    pub fn cached_token(&self) -> Result<MlaCachedToken> {
        Ok(MlaCachedToken {
            position: self.position,
            latent: crate::try_clone_slice(&self.cached_latent)?,
            rope: crate::try_clone_slice(&self.cached_rope)?,
        })
    }
}

/// One token in the MLA latent cache. The rope part is shared by all query
/// heads when it is reconstructed.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaCachedToken {
    pub position: u64,
    pub latent: Vec<f64>,
    pub rope: Vec<f64>,
}

/// The result of one causal MLA attention row.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaAttentionResult {
    /// One row per query head, with masked entries represented by `-∞`.
    pub scores: Vec<Vec<f64>>,
    /// One row per query head, with zero at masked entries.
    pub probabilities: Vec<Vec<f64>>,
    /// Concatenated per-head value outputs, before `o_proj`.
    pub head_output: Vec<f64>,
    /// The final residual-stream width after `o_proj`.
    pub output: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Dims {
    hidden: usize,
    q_lora: usize,
    kv_lora: usize,
    nope: usize,
    rope: usize,
    value: usize,
    heads: usize,
    query_head: usize,
    query_width: usize,
    kv_width: usize,
    decompressed_width: usize,
    output_width: usize,
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

fn usize_dim(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(field, format!("dimension {value} exceeds usize")))
}

fn dims(descriptor: MlaAttentionDescriptor) -> Result<Dims> {
    descriptor.validate()?;
    let hidden = usize_dim(descriptor.hidden, "hidden")?;
    let q_lora = usize_dim(descriptor.q_lora_rank, "q_lora_rank")?;
    let kv_lora = usize_dim(descriptor.kv_lora_rank, "kv_lora_rank")?;
    let nope = usize_dim(descriptor.qk_nope_head_dim, "qk_nope_head_dim")?;
    let rope = usize_dim(descriptor.qk_rope_head_dim, "qk_rope_head_dim")?;
    let value = usize_dim(descriptor.v_head_dim, "v_head_dim")?;
    let heads = usize_dim(descriptor.heads, "heads")?;
    let query_head = nope
        .checked_add(rope)
        .ok_or(moxie_types::DimError::Overflow)?;
    let query_width = heads
        .checked_mul(query_head)
        .ok_or(moxie_types::DimError::Overflow)?;
    let kv_width = kv_lora
        .checked_add(rope)
        .ok_or(moxie_types::DimError::Overflow)?;
    let decompressed_width = heads
        .checked_mul(
            nope.checked_add(value)
                .ok_or(moxie_types::DimError::Overflow)?,
        )
        .ok_or(moxie_types::DimError::Overflow)?;
    let output_width = heads
        .checked_mul(value)
        .ok_or(moxie_types::DimError::Overflow)?;
    Ok(Dims {
        hidden,
        q_lora,
        kv_lora,
        nope,
        rope,
        value,
        heads,
        query_head,
        query_width,
        kv_width,
        decompressed_width,
        output_width,
    })
}

fn finite(values: &[f64], field: &'static str) -> Result<()> {
    for (i, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(invalid(
                field,
                format!("element {i} is nonfinite ({value})"),
            ));
        }
    }
    Ok(())
}

fn exact_len(values: &[f64], expected: usize, field: &'static str) -> Result<()> {
    if values.len() != expected {
        return Err(invalid(
            field,
            format!("got {} elements, expected {expected}", values.len()),
        ));
    }
    Ok(())
}

fn matvec(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    input: &[f64],
    field: &'static str,
) -> Result<Vec<f64>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(moxie_types::DimError::Overflow)?;
    exact_len(matrix, expected, field)?;
    exact_len(input, cols, "mla_input")?;
    finite(matrix, field)?;
    finite(input, "mla_input")?;

    let mut output = crate::try_vec(rows)?;
    for row in matrix.chunks_exact(cols) {
        let mut sum = 0.0f64;
        for (weight, value) in row.iter().zip(input) {
            sum += weight * value;
        }
        if !sum.is_finite() {
            return Err(Error::Numerical {
                detail: format!("{field} produced a nonfinite FP64 result"),
            });
        }
        output.push(sum);
    }
    Ok(output)
}

/// Apply `o_proj` in contiguous head groups, narrowing each FP64 partial to
/// FP32 before adding the groups in order.
pub fn o_proj_grouped(
    o_proj: &[f64],
    hidden: usize,
    width: usize,
    head_output: &[f64],
    groups: usize,
) -> Result<Vec<f32>> {
    if groups == 0 {
        return Err(invalid("groups", "must be nonzero"));
    }
    if !width.is_multiple_of(groups) {
        return Err(invalid("width", "must divide evenly into head groups"));
    }
    let expected = hidden
        .checked_mul(width)
        .ok_or(moxie_types::DimError::Overflow)?;
    exact_len(o_proj, expected, "o_proj")?;
    exact_len(head_output, width, "mla_input")?;
    finite(o_proj, "o_proj")?;
    finite(head_output, "mla_input")?;

    let group_width = width / groups;
    let mut output = crate::try_vec(hidden)?;
    for row in 0..hidden {
        let weights = &o_proj[row * width..(row + 1) * width];
        let mut total = 0.0f32;
        for group in 0..groups {
            let first = group * group_width;
            let end = first + group_width;
            let mut sum = 0.0f64;
            for column in first..end {
                sum += weights[column] * head_output[column];
            }
            if !sum.is_finite() {
                return Err(Error::Numerical {
                    detail: "o_proj produced a nonfinite FP64 partial".into(),
                });
            }
            let partial = sum as f32;
            if !partial.is_finite() {
                return Err(Error::Numerical {
                    detail: "o_proj produced a nonfinite FP32 partial".into(),
                });
            }
            if group == 0 {
                total = partial;
            } else {
                total += partial;
            }
        }
        if !total.is_finite() {
            return Err(Error::Numerical {
                detail: "o_proj produced a nonfinite FP32 result".into(),
            });
        }
        output.push(total);
    }
    Ok(output)
}

fn rms_norm(input: &[f64], gain: &[f64], eps: f32, field: &'static str) -> Result<Vec<f64>> {
    if input.is_empty() {
        return Err(invalid(field, "normalization width is zero"));
    }
    exact_len(gain, input.len(), "mla_norm_gain")?;
    finite(input, field)?;
    finite(gain, "mla_norm_gain")?;
    let mut sum = 0.0f64;
    for value in input {
        sum += value * value;
    }
    let denominator = (sum / input.len() as f64 + eps as f64).sqrt();
    if !denominator.is_finite() || denominator == 0.0 {
        return Err(Error::Numerical {
            detail: format!("{field} produced an invalid RMS denominator"),
        });
    }
    let mut output = crate::try_vec(input.len())?;
    for (value, weight) in input.iter().zip(gain) {
        let result = value * weight / denominator;
        if !result.is_finite() {
            return Err(Error::Numerical {
                detail: format!("{field} produced a nonfinite FP64 result"),
            });
        }
        output.push(result);
    }
    Ok(output)
}

fn rotate(values: &[f64], position: u64, descriptor: MlaAttentionDescriptor) -> Result<Vec<f64>> {
    finite(values, "mla_rope_input")?;
    if !values.len().is_multiple_of(2) {
        return Err(invalid(
            "mla_rope_input",
            format!("RoPE width {} is not even", values.len()),
        ));
    }
    let mut output = crate::try_clone_slice(values)?;
    let half = values.len() / 2;
    let frequency_dim = values.len() as f64;
    for j in 0..half {
        let (input_first, input_second, output_first, output_second) = match descriptor.rope_layout
        {
            RopeLayout::Interleaved => (2 * j, 2 * j + 1, j, half + j),
            RopeLayout::HalfSplit => (j, j + half, j, j + half),
        };
        let inverse_frequency = (descriptor.rope_base as f64).powf(-2.0 * j as f64 / frequency_dim);
        let theta = position as f64 * inverse_frequency;
        let (sin, cos) = theta.sin_cos();
        let (a, b) = (values[input_first], values[input_second]);
        output[output_first] = a * cos - b * sin;
        output[output_second] = b * cos + a * sin;
    }
    finite(&output, "mla_rope_output")?;
    Ok(output)
}

/// Project one FP64 hidden row through the unfused MLA query and KV paths.
pub fn project(
    descriptor: MlaAttentionDescriptor,
    input: &[f64],
    weights: MlaWeights<'_>,
    position: u64,
) -> Result<MlaProjection> {
    let d = dims(descriptor)?;
    exact_len(input, d.hidden, "mla_input")?;
    finite(input, "mla_input")?;

    let q_a = matvec(weights.q_a_proj, d.q_lora, d.hidden, input, "q_a_proj")?;
    let q_a_norm = rms_norm(
        &q_a,
        weights.q_a_layernorm,
        descriptor.rms_norm_eps,
        "q_a_layernorm",
    )?;
    let query_before_rope = matvec(
        weights.q_b_proj,
        d.query_width,
        d.q_lora,
        &q_a_norm,
        "q_b_proj",
    )?;
    let mut query = crate::try_clone_slice(&query_before_rope)?;
    for head in 0..d.heads {
        let start = head * d.query_head;
        let rope_start = start + d.nope;
        let rotated = rotate(
            &query_before_rope[rope_start..start + d.query_head],
            position,
            descriptor,
        )?;
        query[rope_start..start + d.query_head].copy_from_slice(&rotated);
    }

    let kv_a = matvec(
        weights.kv_a_proj_with_mqa,
        d.kv_width,
        d.hidden,
        input,
        "kv_a_proj_with_mqa",
    )?;
    let cached_latent = rms_norm(
        &kv_a[..d.kv_lora],
        weights.kv_a_layernorm,
        descriptor.rms_norm_eps,
        "kv_a_layernorm",
    )?;
    let cached_rope = rotate(&kv_a[d.kv_lora..], position, descriptor)?;

    Ok(MlaProjection {
        position,
        query,
        cached_latent,
        cached_rope,
    })
}

fn validate_cached_token(token: &MlaCachedToken, d: Dims) -> Result<()> {
    exact_len(&token.latent, d.kv_lora, "mla_cached_latent")?;
    exact_len(&token.rope, d.rope, "mla_cached_rope")?;
    finite(&token.latent, "mla_cached_latent")?;
    finite(&token.rope, "mla_cached_rope")?;
    Ok(())
}

/// Reconstruct all per-head nope keys and values from one cached latent row.
pub fn decompress(
    descriptor: MlaAttentionDescriptor,
    token: &MlaCachedToken,
    kv_b_proj: &[f64],
) -> Result<MlaReconstructedToken> {
    let d = dims(descriptor)?;
    validate_cached_token(token, d)?;
    exact_len(
        kv_b_proj,
        d.decompressed_width
            .checked_mul(d.kv_lora)
            .ok_or(moxie_types::DimError::Overflow)?,
        "kv_b_proj",
    )?;
    finite(kv_b_proj, "kv_b_proj")?;

    let per_head = d.nope + d.value;
    let mut keys = crate::try_vec(d.heads)?;
    let mut values = crate::try_vec(d.heads)?;
    for head in 0..d.heads {
        let base = head * per_head * d.kv_lora;
        let key_nope = matvec(
            &kv_b_proj[base..base + d.nope * d.kv_lora],
            d.nope,
            d.kv_lora,
            &token.latent,
            "kv_b_proj_key",
        )?;
        let value_start = base + d.nope * d.kv_lora;
        let value = matvec(
            &kv_b_proj[value_start..value_start + d.value * d.kv_lora],
            d.value,
            d.kv_lora,
            &token.latent,
            "kv_b_proj_value",
        )?;
        let mut key = crate::try_vec(d.nope + d.rope)?;
        key.extend_from_slice(&key_nope);
        key.extend_from_slice(&token.rope);
        keys.push(key);
        values.push(value);
    }
    Ok(MlaReconstructedToken { keys, values })
}

/// Reconstructed dense rows for one cached token.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaReconstructedToken {
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
}

fn dot(left: &[f64], right: &[f64], field: &'static str) -> Result<f64> {
    if left.len() != right.len() {
        return Err(invalid(
            field,
            format!("{} elements versus {}", left.len(), right.len()),
        ));
    }
    let mut sum = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        sum += a * b;
    }
    if !sum.is_finite() {
        return Err(Error::Numerical {
            detail: format!("{field} produced a nonfinite FP64 score"),
        });
    }
    Ok(sum)
}

fn validate_projection(projection: &MlaProjection, d: Dims) -> Result<()> {
    exact_len(&projection.query, d.query_width, "mla_query")?;
    exact_len(&projection.cached_latent, d.kv_lora, "mla_cached_latent")?;
    exact_len(&projection.cached_rope, d.rope, "mla_cached_rope")?;
    finite(&projection.query, "mla_query")?;
    finite(&projection.cached_latent, "mla_cached_latent")?;
    finite(&projection.cached_rope, "mla_cached_rope")?;
    Ok(())
}

/// Run standard causal FP64 attention over reconstructed keys and values, then
/// apply the unfused output projection.
pub fn attend(
    descriptor: MlaAttentionDescriptor,
    query: &MlaProjection,
    history: &[MlaCachedToken],
    kv_b_proj: &[f64],
    o_proj: &[f64],
) -> Result<MlaAttentionResult> {
    let d = dims(descriptor)?;
    validate_projection(query, d)?;
    if history.is_empty() {
        return Err(invalid(
            "mla_history",
            "causal attention needs at least one token",
        ));
    }
    let mut reconstructed = crate::try_vec(history.len())?;
    for token in history {
        reconstructed.push(decompress(descriptor, token, kv_b_proj)?);
    }

    let scale = 1.0f64 / (d.query_head as f64).sqrt();
    let mut scores = crate::try_vec(d.heads)?;
    let mut probabilities = crate::try_vec(d.heads)?;
    let mut head_output = crate::try_vec(d.output_width)?;
    for head in 0..d.heads {
        let query_start = head * d.query_head;
        let query_row = &query.query[query_start..query_start + d.query_head];
        let mut row_scores = crate::try_vec(history.len())?;
        let mut maximum = f64::NEG_INFINITY;
        for (index, token) in history.iter().enumerate() {
            if descriptor.visibility.allows(query.position, token.position) {
                let score = dot(query_row, &reconstructed[index].keys[head], "mla_score")? * scale;
                if !score.is_finite() {
                    return Err(Error::Numerical {
                        detail: "MLA score became nonfinite after scaling".into(),
                    });
                }
                maximum = maximum.max(score);
                row_scores.push(score);
            } else {
                row_scores.push(f64::NEG_INFINITY);
            }
        }
        if maximum == f64::NEG_INFINITY {
            return Err(Error::Numerical {
                detail: "no visible MLA key for this query position".into(),
            });
        }
        let mut row_probabilities = crate::try_vec(history.len())?;
        let mut denominator = 0.0f64;
        for score in &row_scores {
            let weight = if score.is_finite() {
                (*score - maximum).exp()
            } else {
                0.0
            };
            denominator += weight;
            row_probabilities.push(weight);
        }
        if !denominator.is_finite() || denominator == 0.0 {
            return Err(Error::Numerical {
                detail: "MLA softmax denominator is invalid".into(),
            });
        }
        for weight in &mut row_probabilities {
            *weight /= denominator;
        }
        let mut output = crate::try_vec(d.value)?;
        output.resize(d.value, 0.0);
        for (index, weight) in row_probabilities.iter().enumerate() {
            for (out, value) in output
                .iter_mut()
                .zip(reconstructed[index].values[head].iter())
            {
                *out += weight * value;
            }
        }
        finite(&output, "mla_head_output")?;
        scores.push(row_scores);
        probabilities.push(row_probabilities);
        head_output.extend_from_slice(&output);
    }

    let output = matvec(o_proj, d.hidden, d.output_width, &head_output, "o_proj")?;
    Ok(MlaAttentionResult {
        scores,
        probabilities,
        head_output,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_graph::{MlaAttentionDescriptor, Visibility};
    use moxie_types::{CachePrecision, Precision};

    fn descriptor() -> MlaAttentionDescriptor {
        MlaAttentionDescriptor {
            hidden: 6,
            q_lora_rank: 3,
            kv_lora_rank: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 4,
            v_head_dim: 3,
            heads: 2,
            rms_norm_eps: 1e-5,
            rope_base: 10_000.0,
            rope_layout: RopeLayout::Interleaved,
            visibility: Visibility::Causal,
            layer: 0,
            cache_precision: CachePrecision::expect(Precision::Bf16),
        }
    }

    fn glm52_descriptor() -> MlaAttentionDescriptor {
        MlaAttentionDescriptor {
            hidden: 6144,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_head_dim: 192,
            qk_rope_head_dim: 64,
            v_head_dim: 256,
            heads: 64,
            rms_norm_eps: 1e-5,
            rope_base: 8_000_000.0,
            rope_layout: RopeLayout::Interleaved,
            visibility: Visibility::Causal,
            layer: 0,
            cache_precision: CachePrecision::expect(Precision::Bf16),
        }
    }

    fn values(len: usize, seed: f64) -> Vec<f64> {
        (0..len)
            .map(|i| ((i * 13 + 7) % 29) as f64 / 17.0 - seed)
            .collect()
    }

    struct OwnedWeights {
        q_a_proj: Vec<f64>,
        q_a_layernorm: Vec<f64>,
        q_b_proj: Vec<f64>,
        kv_a_proj_with_mqa: Vec<f64>,
        kv_a_layernorm: Vec<f64>,
        kv_b_proj: Vec<f64>,
        o_proj: Vec<f64>,
    }

    impl OwnedWeights {
        fn refs(&self) -> MlaWeights<'_> {
            MlaWeights {
                q_a_proj: &self.q_a_proj,
                q_a_layernorm: &self.q_a_layernorm,
                q_b_proj: &self.q_b_proj,
                kv_a_proj_with_mqa: &self.kv_a_proj_with_mqa,
                kv_a_layernorm: &self.kv_a_layernorm,
                kv_b_proj: &self.kv_b_proj,
                o_proj: &self.o_proj,
            }
        }
    }

    fn weights(d: MlaAttentionDescriptor) -> OwnedWeights {
        let q_head = d.query_head_dim().unwrap() as usize;
        let kv_width = d.kv_projection_width().unwrap() as usize;
        let decompressed = d.decompressed_kv_width().unwrap() as usize;
        let output = d.output_width().unwrap() as usize;
        let q_a = values(d.q_lora_rank as usize * d.hidden as usize, 0.3);
        let q_gain = vec![1.0; d.q_lora_rank as usize];
        let q_b = values(d.heads as usize * q_head * d.q_lora_rank as usize, 0.2);
        let kv_a = values(kv_width * d.hidden as usize, 0.4);
        let kv_gain = vec![1.0; d.kv_lora_rank as usize];
        let kv_b = values(decompressed * d.kv_lora_rank as usize, 0.1);
        let o = values(d.hidden as usize * output, 0.5);
        OwnedWeights {
            q_a_proj: q_a,
            q_a_layernorm: q_gain,
            q_b_proj: q_b,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_gain,
            kv_b_proj: kv_b,
            o_proj: o,
        }
    }

    // Independent transcription of glm_rope_interleaved_f32's input-pair and
    // split-output indexing, kept in the test so the production rotate cannot
    // validate itself.
    fn legacy_interleaved(values: &[f64], position: u64, base: f64) -> Vec<f64> {
        let half = values.len() / 2;
        let mut output = values.to_vec();
        for index in 0..half {
            let inverse_frequency = base.powf(-2.0 * index as f64 / values.len() as f64);
            let (sin, cos) = (position as f64 * inverse_frequency).sin_cos();
            let first = values[2 * index];
            let second = values[2 * index + 1];
            output[index] = first * cos - second * sin;
            output[half + index] = second * cos + first * sin;
        }
        output
    }

    fn rope_fixture(d: MlaAttentionDescriptor) -> (OwnedWeights, Vec<f64>, f64) {
        let mut owned = weights(d);
        owned.q_a_proj.fill(0.0);
        owned.q_a_proj[0] = 1.0;
        owned.q_b_proj.fill(0.0);
        for row in 0..d.query_width().unwrap() as usize {
            owned.q_b_proj[row * d.q_lora_rank as usize] = row as f64 + 1.0;
        }
        owned.kv_a_proj_with_mqa.fill(0.0);
        for row in 0..d.kv_projection_width().unwrap() as usize {
            owned.kv_a_proj_with_mqa[row * d.hidden as usize] = row as f64 + 13.0;
        }
        let mut input = vec![0.0; d.hidden as usize];
        input[0] = 1.0;
        let normalization = 1.0 / (1.0 / d.q_lora_rank as f64 + d.rms_norm_eps as f64).sqrt();
        (owned, input, normalization)
    }

    fn expected_kv_rope(
        d: MlaAttentionDescriptor,
        input: &[f64],
        weights: MlaWeights<'_>,
        position: u64,
    ) -> Vec<f64> {
        let hidden = d.hidden as usize;
        let start = d.kv_lora_rank as usize;
        let rope = d.qk_rope_head_dim as usize;
        let raw = (0..rope)
            .map(|offset| {
                weights.kv_a_proj_with_mqa[(start + offset) * hidden..(start + offset + 1) * hidden]
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect::<Vec<_>>();
        legacy_interleaved(&raw, position, d.rope_base as f64)
    }

    #[test]
    fn both_checkpoint_and_small_shapes_declare_the_latent_chain() {
        for (d, expected) in [
            (glm52_descriptor(), (256, 16_384, 576, 28_672, 16_384)),
            (descriptor(), (6, 12, 6, 10, 6)),
        ] {
            d.validate().unwrap();
            assert_eq!(d.query_head_dim().unwrap(), expected.0);
            assert_eq!(d.query_width().unwrap(), expected.1);
            assert_eq!(d.cache_width().unwrap(), expected.2);
            assert_eq!(d.decompressed_kv_width().unwrap(), expected.3);
            assert_eq!(d.output_width().unwrap(), expected.4);
            assert_eq!(d.q_a_proj_shape().unwrap(), (d.q_lora_rank, d.hidden));
            assert_eq!(d.q_b_proj_shape().unwrap(), (expected.1, d.q_lora_rank));
            assert_eq!(d.kv_a_proj_shape().unwrap(), (expected.2, d.hidden));
            assert_eq!(d.kv_b_proj_shape().unwrap(), (expected.3, d.kv_lora_rank));
            assert_eq!(d.o_proj_shape().unwrap(), (d.hidden, expected.4));
        }
    }

    #[test]
    fn projection_matches_legacy_rope_for_every_head_and_cached_row() {
        let d = descriptor();
        let (owned, input, normalization) = rope_fixture(d);
        let w = owned.refs();
        let at_zero = project(d, &input, w, 0).unwrap();
        let query_head = d.query_head_dim().unwrap() as usize;
        assert_eq!(
            &at_zero.query[..query_head],
            &[
                normalization,
                2.0 * normalization,
                3.0 * normalization,
                5.0 * normalization,
                4.0 * normalization,
                6.0 * normalization,
            ]
        );
        assert_eq!(
            &at_zero.query[query_head..2 * query_head],
            &[
                7.0 * normalization,
                8.0 * normalization,
                9.0 * normalization,
                11.0 * normalization,
                10.0 * normalization,
                12.0 * normalization,
            ]
        );
        assert_eq!(at_zero.cached_rope, vec![15.0, 17.0, 16.0, 18.0]);

        let at_one = project(d, &input, w, 1).unwrap();
        for head in 0..d.heads as usize {
            let start = head * query_head;
            let first = (3 + head * query_head) as f64;
            let raw = [
                first * normalization,
                (first + 1.0) * normalization,
                (first + 2.0) * normalization,
                (first + 3.0) * normalization,
            ];
            let expected = legacy_interleaved(&raw, 1, d.rope_base as f64);
            assert_eq!(
                &at_one.query[start + d.qk_nope_head_dim as usize..start + query_head],
                expected.as_slice()
            );
            assert_eq!(
                &at_one.query[start..start + d.qk_nope_head_dim as usize],
                &[(first - 2.0) * normalization, (first - 1.0) * normalization,]
            );
        }
        assert_eq!(
            at_one.cached_rope,
            legacy_interleaved(&[15.0, 16.0, 17.0, 18.0], 1, d.rope_base as f64)
        );

        let token = at_one.cached_token().unwrap();
        let reconstructed = decompress(d, &token, w.kv_b_proj).unwrap();
        assert_eq!(at_one.query.len(), d.query_width().unwrap() as usize);
        assert_eq!(at_one.cached_latent.len(), d.kv_lora_rank as usize);
        assert_eq!(
            at_one.cached_latent.len() + at_one.cached_rope.len(),
            d.cache_width().unwrap() as usize
        );
        assert_eq!(reconstructed.keys.len(), d.heads as usize);
        assert_eq!(reconstructed.values.len(), d.heads as usize);
        assert!(
            reconstructed
                .keys
                .iter()
                .all(|key| key.len() == d.query_head_dim().unwrap() as usize)
        );
        assert!(
            reconstructed
                .values
                .iter()
                .all(|value| value.len() == d.v_head_dim as usize)
        );
    }

    #[test]
    fn scores_match_an_independent_dense_causal_calculation() {
        let d = descriptor();
        let owned = weights(d);
        let w = owned.refs();
        let mut history = Vec::new();
        let mut expected_ropes = Vec::new();
        for position in 0..3 {
            let input = values(d.hidden as usize, 0.7 + position as f64);
            let projection = project(d, &input, w, position).unwrap();
            expected_ropes.push(expected_kv_rope(d, &input, w, position));
            history.push(projection.cached_token().unwrap());
        }
        // A future token exercises the causal refusal without contributing to
        // the direct dense scores.
        let future_input = values(d.hidden as usize, 1.7);
        let future = project(d, &future_input, w, 4)
            .unwrap()
            .cached_token()
            .unwrap();
        expected_ropes.push(expected_kv_rope(d, &future_input, w, 4));
        history.push(future);
        let query = project(d, &values(d.hidden as usize, 0.1), w, 2).unwrap();
        let result = attend(d, &query, &history, w.kv_b_proj, w.o_proj).unwrap();

        // Reconstruct the expected keys independently instead of calling the
        // oracle's decompressor on both sides of the comparison. This direct
        // FP64 transcription is the guard against a shared bug in score and
        // decompression code cancelling itself out.
        let per_head = d.qk_nope_head_dim as usize + d.v_head_dim as usize;
        let latent = d.kv_lora_rank as usize;
        let nope = d.qk_nope_head_dim as usize;
        let rope = d.qk_rope_head_dim as usize;
        let dense_keys: Vec<Vec<Vec<f64>>> = (0..d.heads as usize)
            .map(|head| {
                history
                    .iter()
                    .enumerate()
                    .map(|(index, token)| {
                        let mut key = Vec::with_capacity(nope + rope);
                        for lane in 0..nope {
                            let row = (head * per_head + lane) * latent;
                            key.push(
                                (0..latent)
                                    .map(|rank| w.kv_b_proj[row + rank] * token.latent[rank])
                                    .sum(),
                            );
                        }
                        key.extend_from_slice(&expected_ropes[index]);
                        key
                    })
                    .collect()
            })
            .collect();
        let scale = 1.0 / (d.query_head_dim().unwrap() as f64).sqrt();
        for ((query_row, result_scores), expected_keys) in query
            .query
            .chunks_exact(d.query_head_dim().unwrap() as usize)
            .zip(&result.scores)
            .zip(&dense_keys)
        {
            for ((score, token), expected_key) in
                result_scores.iter().zip(&history).zip(expected_keys)
            {
                let expected = if token.position <= query.position {
                    query_row
                        .iter()
                        .zip(expected_key)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        * scale
                } else {
                    f64::NEG_INFINITY
                };
                assert_eq!(*score, expected);
            }
        }
        assert_eq!(result.output.len(), d.hidden as usize);
    }

    #[test]
    fn malformed_geometry_and_inputs_are_typed_refusals() {
        let mut bad = descriptor();
        bad.q_lora_rank = 0;
        assert!(matches!(bad.validate(), Err(Error::InvalidRequest { .. })));

        let d = descriptor();
        let owned = weights(d);
        let w = owned.refs();
        let mut input = values(d.hidden as usize, 0.7);
        input[0] = f64::NAN;
        assert!(matches!(
            project(d, &input, w, 0),
            Err(Error::InvalidRequest { .. })
        ));

        let projection = project(d, &values(d.hidden as usize, 0.7), w, 0).unwrap();
        let token = projection.cached_token().unwrap();
        assert!(matches!(
            decompress(d, &token, &w.kv_b_proj[..w.kv_b_proj.len() - 1]),
            Err(Error::InvalidRequest { .. })
        ));
    }

    #[test]
    fn grouped_o_proj_matches_matvec_with_one_group() {
        let matrix = values(15, 0.3);
        let input = values(5, 0.9);
        let expected = matvec(&matrix, 3, 5, &input, "o_proj")
            .unwrap()
            .into_iter()
            .map(|value| (value as f32).to_bits())
            .collect::<Vec<_>>();
        let actual = o_proj_grouped(&matrix, 3, 5, &input, 1)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
