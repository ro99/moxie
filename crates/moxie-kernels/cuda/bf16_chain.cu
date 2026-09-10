// Task 0012's deliberately small correctness kernels. They preserve the
// semantic BF16 boundaries and fixed ascending reductions; they make no
// throughput claim and allocate no memory.
#include <cuda_bf16.h>

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
        sum = __fadd_rn(sum, __fmul_rn(value, value));
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
    const float mean = __fdiv_rn(row_sums[row], static_cast<float>(hidden));
    const float denom = sqrtf(__fadd_rn(mean, epsilon));
    const float scaled = __fmul_rn(
        __bfloat162float(input[index]),
        __bfloat162float(gain[index % hidden]));
    if (!isfinite(row_sums[row]) || !isfinite(denom) || denom <= 0.0F ||
        !isfinite(scaled)) {
        // A finite BF16 input can overflow the FP32 reduction or scaling.
        // Preserve that failure through the residual and bounded final
        // readback instead of turning finite / infinity into a silent zero.
        output[index] = __float2bfloat16_rn(__uint_as_float(0x7fc00000U));
        return;
    }
    const float normalized = __fdiv_rn(scaled, denom);
    output[index] = __float2bfloat16_rn(
        isfinite(normalized) ? normalized : __uint_as_float(0x7fc00000U));
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
