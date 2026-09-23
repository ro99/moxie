// Shared BF16 dense-graph operations.  The image is deliberately generic:
// dispatch is by the semantic descriptor, shape and layout, never by model.
#include <cuda_bf16.h>

static __device__ __forceinline__ float moxie_dense_failure_v1() {
    return __uint_as_float(0x7fc00000U);
}

static __device__ __forceinline__ bool moxie_dense_subnormal_v1(float value) {
    return value != 0.0F && fabsf(value) < 0x1p-126F;
}

static __device__ __forceinline__ float moxie_dense_bf16_v1(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// The single-device declared-S reference.  Each input-axis block starts from
// zero in FP32, blocks are combined in ascending order, and only the final
// result crosses the BF16 boundary.  This is intentionally a separate symbol
// from the TP partial kernel below: the reference must not share its arithmetic
// path with the pair's partial/reduce implementation.
extern "C" __global__ void moxie_dense_linear_split_v1(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned long long rows,
    unsigned long long input_width, unsigned long long output_width,
    unsigned long long blocks) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * output_width;
    if (index >= count || blocks == 0 || input_width % blocks != 0) return;
    const unsigned long long row = index / output_width;
    const unsigned long long column = index % output_width;
    const unsigned long long block_width = input_width / blocks;
    float total = 0.0F;
    for (unsigned long long block = 0; block < blocks; ++block) {
        float partial = 0.0F;
        const unsigned long long first = block * block_width;
        for (unsigned long long k = 0; k < block_width; ++k) {
            const unsigned long long offset = row * input_width + first + k;
            partial = __fadd_rn(partial, __fmul_rn(
                __bfloat162float(x[offset]),
                __bfloat162float(weight[column * input_width + first + k])));
        }
        total = __fadd_rn(total, partial);
    }
    output[index] = __float2bfloat16_rn(total);
}

// The TP row-parallel partial.  It deliberately stores FP32 and does not
// round; RankGroup::reduce owns the one declared output rounding boundary.
extern "C" __global__ void moxie_dense_linear_partial_v1(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    float* output, unsigned long long rows,
    unsigned long long input_width, unsigned long long output_width) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * output_width;
    if (index >= count) return;
    const unsigned long long row = index / output_width;
    const unsigned long long column = index % output_width;
    float sum = 0.0F;
    for (unsigned long long k = 0; k < input_width; ++k) {
        sum = __fadd_rn(sum, __fmul_rn(
            __bfloat162float(x[row * input_width + k]),
            __bfloat162float(weight[column * input_width + k])));
    }
    output[index] = sum;
}

// Two-rank exact reduce: the arguments are already FP32 partials.  The order
// is explicit even though two-term addition is commutative, because the ABI is
// the two-rank specialization of the declared ascending-rank contract.
extern "C" __global__ void moxie_tp_reduce_f32_v1(
    const float* rank_zero, const float* rank_one,
    __nv_bfloat16* output, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16_rn(__fadd_rn(rank_zero[index], rank_one[index]));
}

extern "C" __global__ void moxie_dense_embedding_v1(
    const unsigned long long* token_ids, const __nv_bfloat16* table,
    __nv_bfloat16* output, unsigned long long rows, unsigned long long vocab,
    unsigned long long hidden, float scale) {
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned long long token = token_ids[row];
    if (token >= vocab) {
        for (unsigned long long k = 0; k < hidden; ++k) {
            output[row * hidden + k] = __float2bfloat16_rn(moxie_dense_failure_v1());
        }
        return;
    }
    const __nv_bfloat16* source = table + token * hidden;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float value = __bfloat162float(source[k]);
        if (scale == 1.0F) {
            output[row * hidden + k] = source[k];
        } else {
            // The host oracle evaluates the scale in FP64, narrows to FP32,
            // then applies the node's BF16 boundary.
            const float scaled = static_cast<float>(
                static_cast<double>(value) * static_cast<double>(scale));
            output[row * hidden + k] = __float2bfloat16_rn(scaled);
        }
    }
}

// One thread owns one row/group.  That is intentionally slower than a warp
// reduction: the host oracle's ascending group order is the device contract.
extern "C" __global__ void moxie_dense_grouped_rms_v1(
    const __nv_bfloat16* input, const __nv_bfloat16* gain,
    __nv_bfloat16* output, unsigned long long rows, unsigned long long hidden,
    unsigned long long groups, float epsilon) {
    const unsigned long long group_index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * groups;
    if (group_index >= count || groups == 0 || hidden == 0 || hidden % groups != 0) {
        return;
    }
    const unsigned long long row = group_index / groups;
    const unsigned long long group = group_index % groups;
    const unsigned long long width = hidden / groups;
    const unsigned long long start = row * hidden + group * width;
    float sum = 0.0F;
    bool invalid = false;
    for (unsigned long long k = 0; k < width; ++k) {
        const float value = __bfloat162float(input[start + k]);
        const float square = __fmul_rn(value, value);
        invalid = invalid || (value != 0.0F &&
            (square == 0.0F || moxie_dense_subnormal_v1(square)));
        sum = __fadd_rn(sum, square);
    }
    const float mean = __fdiv_rn(sum, static_cast<float>(width));
    const float denominator = sqrtf(__fadd_rn(mean, epsilon));
    invalid = invalid || !isfinite(sum) || !isfinite(mean) ||
        !isfinite(denominator) || denominator <= 0.0F ||
        (sum != 0.0F && (mean == 0.0F || moxie_dense_subnormal_v1(mean)));
    for (unsigned long long k = 0; k < width; ++k) {
        const unsigned long long index = start + k;
        const float value = __bfloat162float(input[index]);
        const float scaled = __fmul_rn(value, __bfloat162float(gain[k]));
        const float normalized = __fdiv_rn(scaled, denominator);
        if (invalid || !isfinite(scaled) || !isfinite(normalized) ||
            (scaled != 0.0F &&
             (normalized == 0.0F || moxie_dense_subnormal_v1(normalized)))) {
            output[index] = __float2bfloat16_rn(moxie_dense_failure_v1());
        } else {
            output[index] = __float2bfloat16_rn(normalized);
        }
    }
}

// Angles are laid out as [row][rotary_pair][cos, sin].  HalfSplit pairing is
// the only layout admitted by the reduced Gemma graph; the table is shared by
// query and key Rope nodes with the same geometry and positions.
extern "C" __global__ void moxie_dense_rope_v1(
    const __nv_bfloat16* input, const float* angles, __nv_bfloat16* output,
    unsigned long long rows, unsigned long long heads,
    unsigned long long head_dim, unsigned long long rotary_dim) {
    const unsigned long long head_index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * heads;
    if (head_index >= count || head_dim == 0 || rotary_dim > head_dim ||
        rotary_dim % 2 != 0 || head_dim % 2 != 0) {
        return;
    }
    const unsigned long long row = head_index / heads;
    const unsigned long long head = head_index % heads;
    const unsigned long long row_start = (row * heads + head) * head_dim;
    for (unsigned long long k = 0; k < head_dim; ++k) {
        output[row_start + k] = input[row_start + k];
    }
    const unsigned long long pairs = rotary_dim / 2;
    const unsigned long long half = head_dim / 2;
    for (unsigned long long j = 0; j < pairs; ++j) {
        const float cos_theta = angles[(row * pairs + j) * 2];
        const float sin_theta = angles[(row * pairs + j) * 2 + 1];
        const unsigned long long p = row_start + j;
        const unsigned long long q = row_start + j + half;
        const float a = __bfloat162float(input[p]);
        const float b = __bfloat162float(input[q]);
        output[p] = __float2bfloat16_rn(__fsub_rn(
            __fmul_rn(a, cos_theta), __fmul_rn(b, sin_theta)));
        output[q] = __float2bfloat16_rn(__fadd_rn(
            __fmul_rn(b, cos_theta), __fmul_rn(a, sin_theta)));
    }
}

static __device__ __forceinline__ float moxie_dense_gelu_tanh_v1(float value) {
    // Strata's device path uses the single-precision tanh approximation.  The
    // host oracle uses f64 tanh, so this operation is qualified to the task's
    // one-BF16-ULP gate rather than claiming bit identity here.
    const float cube = __fmul_rn(__fmul_rn(value, value), value);
    const float polynomial = __fadd_rn(value, __fmul_rn(0.044715F, cube));
    const float argument = __fmul_rn(0.7978845608028654F, polynomial);
    return __fmul_rn(__fmul_rn(0.5F, value), __fadd_rn(1.0F, tanhf(argument)));
}

extern "C" __global__ void moxie_dense_geglu_v1(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float activated = moxie_dense_bf16_v1(
        moxie_dense_gelu_tanh_v1(__bfloat162float(gate[index])));
    const float product = __fmul_rn(activated, __bfloat162float(up[index]));
    output[index] = __float2bfloat16_rn(product);
}

extern "C" __global__ void moxie_dense_residual_scaled_v1(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    __nv_bfloat16* output, unsigned long long elements, float scale) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float sum = moxie_dense_bf16_v1(__fadd_rn(
        __bfloat162float(left[index]), __bfloat162float(right[index])));
    output[index] = __float2bfloat16_rn(__fmul_rn(sum, scale));
}

extern "C" __global__ void moxie_dense_vocab_projection_v1(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, float* output,
    unsigned long long rows, unsigned long long hidden,
    unsigned long long vocab, float softcap) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * vocab;
    if (index >= count) return;
    const unsigned long long row = index / vocab;
    const unsigned long long column = index % vocab;
    float sum = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        sum = __fadd_rn(sum, __fmul_rn(
            __bfloat162float(input[row * hidden + k]),
            __bfloat162float(weight[column * hidden + k])));
    }
    if (softcap > 0.0F) {
        const float scaled = moxie_dense_bf16_v1(__fdiv_rn(moxie_dense_bf16_v1(sum), softcap));
        const float bent = moxie_dense_bf16_v1(static_cast<float>(tanh(static_cast<double>(scaled))));
        sum = moxie_dense_bf16_v1(__fmul_rn(bent, softcap));
    }
    output[index] = sum;
}
