// Task 0012's deliberately small correctness kernels. They preserve the
// semantic BF16 boundaries and fixed ascending reductions; they make no
// throughput claim and allocate no memory.
#include <cuda_bf16.h>

static __device__ __forceinline__ float moxie_numerical_failure_v1() {
    return __uint_as_float(0x7fc00000U);
}

static __device__ __forceinline__ bool moxie_fp32_subnormal_v1(float value) {
    return value != 0.0F && fabsf(value) < 0x1p-126F;
}

extern "C" __global__ void moxie_bf16_linear_v1(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned long long rows,
    unsigned long long input_width, unsigned long long output_width) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * output_width;
    if (index >= count) return;
    const unsigned long long row = index / output_width;
    const unsigned long long column = index % output_width;
    float sum = 0.0F;
    for (unsigned long long k = 0; k < input_width; ++k) {
        const float product = __fmul_rn(
            __bfloat162float(x[row * input_width + k]),
            __bfloat162float(weight[column * input_width + k]));
        sum = __fadd_rn(sum, product);
    }
    output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void moxie_bf16_rms_sum_v1(
    const __nv_bfloat16* input, float* row_sums,
    unsigned long long rows, unsigned long long hidden) {
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float value = __bfloat162float(input[row * hidden + k]);
        const float square = __fmul_rn(value, value);
        if (value != 0.0F &&
            (square == 0.0F || moxie_fp32_subnormal_v1(square))) {
            // The fixed relative-error proof excludes FP32 underflow. Mark
            // this row so apply/readback fails closed rather than accepting a
            // finite result outside the predeclared bound.
            row_sums[row] = moxie_numerical_failure_v1();
            return;
        }
        sum = __fadd_rn(sum, square);
    }
    row_sums[row] = sum;
}

extern "C" __global__ void moxie_bf16_rms_apply_v1(
    const __nv_bfloat16* input, const __nv_bfloat16* gain,
    const float* row_sums, __nv_bfloat16* output,
    unsigned long long rows, unsigned long long hidden, float epsilon) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = rows * hidden;
    if (index >= count) return;
    const unsigned long long row = index / hidden;
    const float row_sum = row_sums[row];
    const float mean = __fdiv_rn(row_sum, static_cast<float>(hidden));
    const float denom = sqrtf(__fadd_rn(mean, epsilon));
    const float input_value = __bfloat162float(input[index]);
    const float gain_value = __bfloat162float(gain[index % hidden]);
    const float scaled = __fmul_rn(input_value, gain_value);
    const bool mean_underflow = row_sum != 0.0F &&
        (mean == 0.0F || moxie_fp32_subnormal_v1(mean));
    const bool scaling_underflow = input_value != 0.0F && gain_value != 0.0F &&
        (scaled == 0.0F || moxie_fp32_subnormal_v1(scaled));
    if (!isfinite(row_sum) || mean_underflow || !isfinite(denom) ||
        denom <= 0.0F || !isfinite(scaled) || scaling_underflow) {
        // Finite BF16 operands can overflow or underflow the declared FP32
        // intermediates. Preserve that failure through residual/readback
        // instead of returning a finite value outside the fixed bound.
        output[index] = __float2bfloat16_rn(moxie_numerical_failure_v1());
        return;
    }
    const float normalized = __fdiv_rn(scaled, denom);
    const bool normalization_underflow = scaled != 0.0F &&
        (normalized == 0.0F || moxie_fp32_subnormal_v1(normalized));
    output[index] = __float2bfloat16_rn(
        isfinite(normalized) && !normalization_underflow
            ? normalized
            : moxie_numerical_failure_v1());
}

extern "C" __global__ void moxie_bf16_residual_v1(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    __nv_bfloat16* output, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16_rn(__fadd_rn(
        __bfloat162float(left[index]), __bfloat162float(right[index])));
}
