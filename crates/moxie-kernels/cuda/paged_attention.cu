// Task 0037's common BF16 paged attention: prefill, append and decode over one
// layer's paged key/value state.
//
// One symbol serves whole prefill, a chunk of prefill and a single decode row,
// because they are the same operation over different row counts. They are not
// three kernels here for the same reason W4A16 and W8A16 are not two in
// `affine_linear.cu`: a kernel per calling pattern is how seven engines grow
// back (document 02). Multi-head and grouped-query attention are likewise one
// path, since GQA *is* MHA when `kv_heads == heads`.
//
// **Visibility is decided on absolute positions, never on an index within a
// chunk.** R21 is the family of bugs where a chunked prefill is right at
// position zero and wrong at every later chunk. A query row's absolute position
// is `first_position + row`; a stored row's is `history_base + logical`, where
// `history_base` is above zero for a sliding layer that has reclaimed what its
// window can no longer see. A masked key is *excluded* -- its score never
// enters the maximum, the denominator or the sum -- rather than biased by a
// large negative number that a later exponential turns into a small positive
// weight.
//
// The algebra is `moxie_oracles::online_softmax`, which the host fixtures pin:
// a running maximum, a running denominator, a running weighted value sum, and a
// rescale whenever a later tile raises the maximum. The empty partial is the
// identity and is handled *around* the arithmetic, because an empty tile's
// maximum is -inf and `exp(-inf - -inf)` is NaN. That is the same edge the host
// fixture `a_fully_masked_block_contributes_nothing_and_never_a_nan` pins, and
// a sliding window reaches it whenever the window has moved past a whole tile.
//
// Scores, the maximum subtraction, the exponentials, the partials and the
// weighted value accumulation are all FP32 (`AccumulationPolicy::Bf16InF32Acc`),
// and the only narrowing is the single `RoundingProfile::FinalBf16Rne` at the
// output. Products are exact: a BF16 times a BF16 has at most 16 significand
// bits and FP32 holds 24, so the only difference from the FP64 oracle is the
// order of the adds and the exponential's own rounding -- which is what
// `moxie_oracles::attention::attention_error_bound` bounds, at the layer's
// *declared* score scale. `expf` is the accurate device exponential, not
// `__expf`: this image is compiled without `--use_fast_math`, and a fast
// exponential carrying 2^-21 of relative error is not covered by that bound.
//
// No tensor cores in this version, and no tuning: O6 and O7 are open, so this
// kernel exists to be *correct* on both SM86 and SM120 and to make no claim
// about speed. A tiled tensor-core version is a later slice with a measurement
// attached to it.
#include <cuda_bf16.h>

// Keys per tile. One tile is what the block scores, maximizes and folds into
// the running partial before it looks at the next one, so it is the block width
// of the online softmax rather than a page size: pages are a *storage* fact and
// this is a *scheduling* one. They are deliberately independent -- a tile spans
// page boundaries and a page tail leaves a tile partly masked, which is exactly
// the case a page-shaped tile would never exercise.
#define MOXIE_ATTN_TILE 128
// Threads per block: four warps. Phase one gives each warp one key at a time and
// splits the head dimension across its lanes; phase two gives each thread one
// output component.
#define MOXIE_ATTN_THREADS 128
// The largest head dimension this build serves. The host refuses anything wider
// rather than this kernel reading past the end of a row.
#define MOXIE_ATTN_MAX_HEAD_DIM 256
#define MOXIE_ATTN_ACC_SLOTS (MOXIE_ATTN_MAX_HEAD_DIM / MOXIE_ATTN_THREADS)
#define MOXIE_ATTN_WARPS (MOXIE_ATTN_THREADS / 32)

static __device__ __forceinline__ float moxie_attn_ninf_v1() {
    return __int_as_float(0xff800000);
}

static __device__ __forceinline__ float moxie_attn_failure_v1() {
    return __uint_as_float(0x7fc00000U);
}

// out[row, head, d] = sum_k p_k * V[k, head/group, d], with
// p = softmax over visible k of (scale * dot(Q[row, head], K[k, head/group])).
//
// Keys and values live in pages of `page_tokens` rows; row `r` of the logical
// history is slot `r % page_tokens` of physical page `page_table[r /
// page_tokens]`, and within a row the layout is `[kv_head][head_dim]`. The page
// table is the only thing that knows where a page physically is, so a logical
// history stays contiguous while its pages do not.
//
// `window` is zero for full causal visibility and the window width otherwise,
// matching `moxie_graph::Visibility`. A query with no visible key writes a
// quiet NaN in every component: the host oracle refuses that case with a typed
// error, and the device says the same thing the only way it can. It is never a
// uniform draw over the cache and never a zero row.
static __device__ __forceinline__ void moxie_bf16_paged_attention_body_v1(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages,
    const unsigned int* __restrict__ page_table,
    __nv_bfloat16* __restrict__ output,
    unsigned long long rows,
    unsigned long long first_position,
    unsigned long long history_base,
    unsigned long long history_rows,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int page_tokens,
    unsigned int window,
    float scale) {
    const unsigned long long row = blockIdx.x;
    const unsigned head = blockIdx.y;
    if (row >= rows || head >= heads) return;

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31U;
    const unsigned warp = tid >> 5;

    __shared__ float q_sh[MOXIE_ATTN_MAX_HEAD_DIM];
    __shared__ float score_sh[MOXIE_ATTN_TILE];
    // The element offset of each tile row's key/value block, resolved through
    // the page table once per tile rather than once per output component.
    __shared__ unsigned long long base_sh[MOXIE_ATTN_TILE];
    __shared__ float reduce_sh[MOXIE_ATTN_WARPS];
    __shared__ float tile_sh[2];

    const unsigned long long position = first_position + row;
    const unsigned kv_head = head / (heads / kv_heads);
    // Query and output have the same geometry, `[rows][heads][head_dim]`, so
    // one offset addresses this block's row in both.
    const unsigned long long row_head_base =
        (row * heads + head) * static_cast<unsigned long long>(head_dim);

    // The visible range, in absolute positions. A causal query sees everything
    // through its own position, which the caller has already appended; a
    // sliding one sees the `window` most recent, inclusive of itself.
    bool empty = history_rows == 0ULL;
    unsigned long long lo_abs = history_base;
    unsigned long long hi_abs = 0ULL;
    if (!empty) {
        if (window != 0U) {
            const unsigned long long w = window;
            const unsigned long long floor_abs =
                (position + 1ULL > w) ? (position + 1ULL - w) : 0ULL;
            if (floor_abs > lo_abs) lo_abs = floor_abs;
        }
        const unsigned long long last = history_base + history_rows - 1ULL;
        hi_abs = (position < last) ? position : last;
        empty = hi_abs < lo_abs;
    }
    if (empty) {
        for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
            output[row_head_base + d] = __float2bfloat16_rn(moxie_attn_failure_v1());
        }
        return;
    }

    const unsigned long long lo = lo_abs - history_base;
    const unsigned long long hi = hi_abs - history_base;

    for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
        q_sh[d] = __bfloat162float(query[row_head_base + d]);
    }

    // One accumulator slot per output component this thread owns. The loop
    // bound is a compile-time constant so the array stays in registers: a
    // runtime bound would put the running value sum in local memory, which is
    // the accumulator spilling to DRAM once per tile.
    float acc[MOXIE_ATTN_ACC_SLOTS];
#pragma unroll
    for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) acc[s] = 0.0F;
    float run_max = moxie_attn_ninf_v1();
    float run_sum = 0.0F;
    __syncthreads();

    const unsigned long long row_stride =
        static_cast<unsigned long long>(kv_heads) * static_cast<unsigned long long>(head_dim);
    const unsigned long long head_off =
        static_cast<unsigned long long>(kv_head) * static_cast<unsigned long long>(head_dim);

    for (unsigned long long tile = (lo / MOXIE_ATTN_TILE) * MOXIE_ATTN_TILE; tile <= hi;
         tile += MOXIE_ATTN_TILE) {
        // Phase one: one warp per key, lanes split the head dimension. A tile
        // entry outside the visible range scores -inf, which is exclusion
        // rather than a bias: it contributes nothing to the maximum below, and
        // its weight becomes exactly zero.
        for (unsigned t = warp; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_WARPS) {
            const unsigned long long logical = tile + t;
            float score = moxie_attn_ninf_v1();
            unsigned long long base = 0ULL;
            if (logical >= lo && logical <= hi) {
                const unsigned long long page = logical / page_tokens;
                const unsigned long long slot = logical % page_tokens;
                base = ((static_cast<unsigned long long>(page_table[page]) * page_tokens) + slot) *
                           row_stride +
                       head_off;
                float partial = 0.0F;
                for (unsigned d = lane; d < head_dim; d += 32U) {
                    partial = fmaf(q_sh[d], __bfloat162float(key_pages[base + d]), partial);
                }
                for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
                    partial += __shfl_down_sync(0xffffffffU, partial, offset);
                }
                score = partial * scale;
            }
            if (lane == 0U) {
                score_sh[t] = score;
                base_sh[t] = base;
            }
        }
        __syncthreads();

        // Phase two: this tile's maximum, by block reduction over the scores.
        float local = moxie_attn_ninf_v1();
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            local = fmaxf(local, score_sh[t]);
        }
        for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
            local = fmaxf(local, __shfl_down_sync(0xffffffffU, local, offset));
        }
        if (lane == 0U) reduce_sh[warp] = local;
        __syncthreads();
        if (tid == 0U) {
            float m = reduce_sh[0];
            for (unsigned w = 1; w < MOXIE_ATTN_WARPS; ++w) m = fmaxf(m, reduce_sh[w]);
            tile_sh[0] = m;
        }
        __syncthreads();
        const float tile_max = tile_sh[0];

        if (tile_max == moxie_attn_ninf_v1()) {
            // A tile with nothing visible in it: the identity of the merge, and
            // the reason it is skipped outright rather than folded in. With the
            // running maximum also still -inf, `exp(run_max - tile_max)` is
            // `exp(-inf + inf)`, a NaN that would poison every later tile. A
            // sliding window walks over whole tiles like this one.
            __syncthreads();
            continue;
        }

        const float new_max = fmaxf(run_max, tile_max);
        // An empty running partial rescales to zero, not through `exp` at all.
        const float correction =
            (run_max == moxie_attn_ninf_v1()) ? 0.0F : expf(run_max - new_max);

        // The tile's weights, in place. A masked entry's -inf score becomes
        // exactly zero here, which is the property the host fixtures assert
        // about the oracle: a masked position contributes nothing at all, not
        // "almost nothing".
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            const float s = score_sh[t];
            score_sh[t] = (s == moxie_attn_ninf_v1()) ? 0.0F : expf(s - new_max);
        }
        __syncthreads();

        float sum_local = 0.0F;
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            sum_local += score_sh[t];
        }
        for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
            sum_local += __shfl_down_sync(0xffffffffU, sum_local, offset);
        }
        if (lane == 0U) reduce_sh[warp] = sum_local;
        __syncthreads();
        if (tid == 0U) {
            float s = reduce_sh[0];
            for (unsigned w = 1; w < MOXIE_ATTN_WARPS; ++w) s += reduce_sh[w];
            tile_sh[1] = s;
        }
        __syncthreads();

        // The weighted value sum. Each thread owns one output component, so the
        // lanes of a warp read consecutive elements of one value row.
#pragma unroll
        for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) {
            const unsigned d = tid + s * MOXIE_ATTN_THREADS;
            if (d >= head_dim) continue;
            float sum = acc[s] * correction;
            for (unsigned t = 0; t < MOXIE_ATTN_TILE; ++t) {
                const float w = score_sh[t];
                if (w != 0.0F) {
                    sum = fmaf(w, __bfloat162float(value_pages[base_sh[t] + d]), sum);
                }
            }
            acc[s] = sum;
        }
        run_sum = run_sum * correction + tile_sh[1];
        run_max = new_max;
        __syncthreads();
    }

    if (!(run_sum > 0.0F)) {
        for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
            output[row_head_base + d] = __float2bfloat16_rn(moxie_attn_failure_v1());
        }
        return;
    }

#pragma unroll
    for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) {
        const unsigned d = tid + s * MOXIE_ATTN_THREADS;
        if (d >= head_dim) continue;
        const float value = acc[s] / run_sum;
        output[row_head_base + d] =
            __float2bfloat16_rn(isfinite(value) ? value : moxie_attn_failure_v1());
    }
}

extern "C" __global__ void moxie_bf16_paged_attention_v1(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages,
    const unsigned int* __restrict__ page_table,
    __nv_bfloat16* __restrict__ output,
    unsigned long long rows,
    unsigned long long first_position,
    unsigned long long history_base,
    unsigned long long history_rows,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int page_tokens,
    unsigned int window,
    float scale) {
    moxie_bf16_paged_attention_body_v1(
        query,
        key_pages,
        value_pages,
        page_table,
        output,
        rows,
        first_position,
        history_base,
        history_rows,
        heads,
        kv_heads,
        head_dim,
        page_tokens,
        window,
        scale);
}

extern "C" __global__ void moxie_bf16_paged_attention_indirect_v1(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages,
    const unsigned int* __restrict__ page_table,
    __nv_bfloat16* __restrict__ output,
    const unsigned long long* __restrict__ step,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int page_tokens,
    unsigned int window,
    float scale) {
    __shared__ unsigned long long step_values[4];
    if (threadIdx.x == 0U) {
        step_values[0] = step[0];
        step_values[1] = step[1];
        step_values[2] = step[2];
        step_values[3] = step[3];
    }
    __syncthreads();
    moxie_bf16_paged_attention_body_v1(
        query,
        key_pages,
        value_pages,
        page_table,
        output,
        step_values[0],
        step_values[1],
        step_values[2],
        step_values[3],
        heads,
        kv_heads,
        head_dim,
        page_tokens,
        window,
        scale);
}

extern "C" __global__ void moxie_kv_append_indirect_v1(
    const unsigned char* __restrict__ keys,
    const unsigned char* __restrict__ values,
    unsigned char* __restrict__ key_pages,
    unsigned char* __restrict__ value_pages,
    const unsigned long long* __restrict__ step,
    const unsigned long long* __restrict__ offsets,
    unsigned long long row_bytes) {
    const unsigned long long row = blockIdx.x;
    __shared__ unsigned long long rows;
    __shared__ unsigned long long key_offset;
    __shared__ unsigned long long value_offset;
    if (threadIdx.x == 0U) {
        rows = step[0];
    }
    __syncthreads();
    if (row >= rows) return;
    if (threadIdx.x == 0U) {
        key_offset = offsets[2ULL * row];
        value_offset = offsets[2ULL * row + 1ULL];
    }
    __syncthreads();

    const unsigned char* key_source = keys + row * row_bytes;
    const unsigned char* value_source = values + row * row_bytes;
    unsigned char* key_destination = key_pages + key_offset;
    unsigned char* value_destination = value_pages + value_offset;
    const bool aligned =
        (row_bytes & 15ULL) == 0ULL &&
        ((reinterpret_cast<unsigned long long>(key_source) |
          reinterpret_cast<unsigned long long>(value_source) |
          reinterpret_cast<unsigned long long>(key_destination) |
          reinterpret_cast<unsigned long long>(value_destination)) &
         15ULL) == 0ULL;
    if (aligned) {
        const unsigned long long vectors = row_bytes / 16ULL;
        for (unsigned long long i = threadIdx.x; i < vectors; i += blockDim.x) {
            const uint4 key = reinterpret_cast<const uint4*>(key_source)[i];
            const uint4 value = reinterpret_cast<const uint4*>(value_source)[i];
            reinterpret_cast<uint4*>(key_destination)[i] = key;
            reinterpret_cast<uint4*>(value_destination)[i] = value;
        }
    } else {
        for (unsigned long long i = threadIdx.x; i < row_bytes; i += blockDim.x) {
            key_destination[i] = key_source[i];
            value_destination[i] = value_source[i];
        }
    }
}

// The host-backed two-block path uses the same arithmetic and visibility rules
// but needs the partial state before the final division. This is a separate
// symbol and ABI on purpose: the accepted single-shot kernel above keeps its
// output contract and its qualified launch untouched.
//
// Outputs are laid out as `[rows][heads]` for max/sum and
// `[rows][heads][head_dim]` for weighted. All three are FP32, matching the
// accumulator precision of the existing kernel; the host widens them to the
// oracle's FP64 Partial representation before calling Partial::merge.
extern "C" __global__ void moxie_bf16_paged_attention_partial_v1(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages,
    const unsigned int* __restrict__ page_table,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_weighted,
    unsigned long long rows,
    unsigned long long first_position,
    unsigned long long history_base,
    unsigned long long history_rows,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int page_tokens,
    unsigned int window,
    float scale) {
    const unsigned long long row = blockIdx.x;
    const unsigned head = blockIdx.y;
    if (row >= rows || head >= heads) return;

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31U;
    const unsigned warp = tid >> 5;
    const unsigned long long partial_index = row * heads + head;
    const unsigned long long weighted_base = partial_index * head_dim;

    __shared__ float q_sh[MOXIE_ATTN_MAX_HEAD_DIM];
    __shared__ float score_sh[MOXIE_ATTN_TILE];
    __shared__ unsigned long long base_sh[MOXIE_ATTN_TILE];
    __shared__ float reduce_sh[MOXIE_ATTN_WARPS];
    __shared__ float tile_sh[2];

    const unsigned long long position = first_position + row;
    const unsigned kv_head = head / (heads / kv_heads);
    const unsigned long long row_head_base =
        (row * heads + head) * static_cast<unsigned long long>(head_dim);

    bool empty = history_rows == 0ULL;
    unsigned long long lo_abs = history_base;
    unsigned long long hi_abs = 0ULL;
    if (!empty) {
        if (window != 0U) {
            const unsigned long long w = window;
            const unsigned long long floor_abs =
                (position + 1ULL > w) ? (position + 1ULL - w) : 0ULL;
            if (floor_abs > lo_abs) lo_abs = floor_abs;
        }
        const unsigned long long last = history_base + history_rows - 1ULL;
        hi_abs = (position < last) ? position : last;
        empty = hi_abs < lo_abs;
    }
    if (empty) {
        if (tid == 0U) {
            partial_max[partial_index] = moxie_attn_ninf_v1();
            partial_sum[partial_index] = 0.0F;
        }
        for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
            partial_weighted[weighted_base + d] = 0.0F;
        }
        return;
    }

    const unsigned long long lo = lo_abs - history_base;
    const unsigned long long hi = hi_abs - history_base;

    for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
        q_sh[d] = __bfloat162float(query[row_head_base + d]);
    }

    float acc[MOXIE_ATTN_ACC_SLOTS];
#pragma unroll
    for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) acc[s] = 0.0F;
    float run_max = moxie_attn_ninf_v1();
    float run_sum = 0.0F;
    __syncthreads();

    const unsigned long long row_stride =
        static_cast<unsigned long long>(kv_heads) * static_cast<unsigned long long>(head_dim);
    const unsigned long long head_off =
        static_cast<unsigned long long>(kv_head) * static_cast<unsigned long long>(head_dim);

    for (unsigned long long tile = (lo / MOXIE_ATTN_TILE) * MOXIE_ATTN_TILE; tile <= hi;
         tile += MOXIE_ATTN_TILE) {
        for (unsigned t = warp; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_WARPS) {
            const unsigned long long logical = tile + t;
            float score = moxie_attn_ninf_v1();
            unsigned long long base = 0ULL;
            if (logical >= lo && logical <= hi) {
                const unsigned long long page = logical / page_tokens;
                const unsigned long long slot = logical % page_tokens;
                base = ((static_cast<unsigned long long>(page_table[page]) * page_tokens) + slot) *
                           row_stride +
                       head_off;
                float partial = 0.0F;
                for (unsigned d = lane; d < head_dim; d += 32U) {
                    partial = fmaf(q_sh[d], __bfloat162float(key_pages[base + d]), partial);
                }
                for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
                    partial += __shfl_down_sync(0xffffffffU, partial, offset);
                }
                score = partial * scale;
            }
            if (lane == 0U) {
                score_sh[t] = score;
                base_sh[t] = base;
            }
        }
        __syncthreads();

        float local = moxie_attn_ninf_v1();
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            local = fmaxf(local, score_sh[t]);
        }
        for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
            local = fmaxf(local, __shfl_down_sync(0xffffffffU, local, offset));
        }
        if (lane == 0U) reduce_sh[warp] = local;
        __syncthreads();
        if (tid == 0U) {
            float m = reduce_sh[0];
            for (unsigned w = 1; w < MOXIE_ATTN_WARPS; ++w) m = fmaxf(m, reduce_sh[w]);
            tile_sh[0] = m;
        }
        __syncthreads();
        const float tile_max = tile_sh[0];
        if (tile_max == moxie_attn_ninf_v1()) {
            __syncthreads();
            continue;
        }

        const float new_max = fmaxf(run_max, tile_max);
        const float correction =
            (run_max == moxie_attn_ninf_v1()) ? 0.0F : expf(run_max - new_max);
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            const float s = score_sh[t];
            score_sh[t] = (s == moxie_attn_ninf_v1()) ? 0.0F : expf(s - new_max);
        }
        __syncthreads();

        float sum_local = 0.0F;
        for (unsigned t = tid; t < MOXIE_ATTN_TILE; t += MOXIE_ATTN_THREADS) {
            sum_local += score_sh[t];
        }
        for (unsigned offset = 16U; offset > 0U; offset >>= 1) {
            sum_local += __shfl_down_sync(0xffffffffU, sum_local, offset);
        }
        if (lane == 0U) reduce_sh[warp] = sum_local;
        __syncthreads();
        if (tid == 0U) {
            float s = reduce_sh[0];
            for (unsigned w = 1; w < MOXIE_ATTN_WARPS; ++w) s += reduce_sh[w];
            tile_sh[1] = s;
        }
        __syncthreads();

#pragma unroll
        for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) {
            const unsigned d = tid + s * MOXIE_ATTN_THREADS;
            if (d >= head_dim) continue;
            float sum = acc[s] * correction;
            for (unsigned t = 0; t < MOXIE_ATTN_TILE; ++t) {
                const float w = score_sh[t];
                if (w != 0.0F) {
                    sum = fmaf(w, __bfloat162float(value_pages[base_sh[t] + d]), sum);
                }
            }
            acc[s] = sum;
        }
        run_sum = run_sum * correction + tile_sh[1];
        run_max = new_max;
        __syncthreads();
    }

    if (!(run_sum > 0.0F)) {
        if (tid == 0U) {
            partial_max[partial_index] = moxie_attn_ninf_v1();
            partial_sum[partial_index] = 0.0F;
        }
        for (unsigned d = tid; d < head_dim; d += MOXIE_ATTN_THREADS) {
            partial_weighted[weighted_base + d] = 0.0F;
        }
        return;
    }

    if (tid == 0U) {
        partial_max[partial_index] = run_max;
        partial_sum[partial_index] = run_sum;
    }
#pragma unroll
    for (unsigned s = 0; s < MOXIE_ATTN_ACC_SLOTS; ++s) {
        const unsigned d = tid + s * MOXIE_ATTN_THREADS;
        if (d >= head_dim) continue;
        partial_weighted[weighted_base + d] = acc[s];
    }
}
