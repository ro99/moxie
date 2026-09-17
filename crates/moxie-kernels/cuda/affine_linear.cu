// Task 0028's shared W4A16 / W8A16 dense linear.
//
// One symbol serves both widths. INT4 and INT8 differ only in how a code is
// read out of the packed row; the group rule, the zero-point section and the
// scale encoding are *runtime* parameters, because a kernel per checkpoint
// shape is how seven engines grow back (document 02, ADR 0003).
//
// What this kernel deliberately does **not** do is dequantize the weight into
// a BF16 tensor and call the BF16 linear. The roadmap forbids that by name for
// M3 item 3, and the reason is memory rather than taste: a 16-bit copy of a
// quantized weight is the thing quantization exists to avoid, and a path that
// materializes one has already spent the saving before the first launch. The
// codes are unpacked and dequantized **into a 16x16 tensor-core tile in shared
// memory**, one k-tile at a time. Nothing wider than that tile is ever 16-bit.
//
// The scale of a group is converted **once per group**, not once per element:
// group sizes are 32 or 128 and the k-tile is 16 wide at a 16-aligned offset,
// so the eight columns a lane dequantizes always lie inside one group. The host
// checks that invariant before launching; this kernel assumes it.
//
// Accumulation is FP32 inside the tensor core (`AccumulationPolicy::Bf16InF32Acc`)
// and the output boundary is one `RoundingProfile::FinalBf16Rne`. The products
// are exact: a BF16 times a BF16 has at most 16 significand bits and FP32 holds
// 24, so the only difference from the host oracle is the order of the adds.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#define MOXIE_AFFINE_TILE 16

static __device__ __forceinline__ float moxie_affine_failure_v1() {
    return __uint_as_float(0x7fc00000U);
}

#include "affine_decode.cuh"

// y[m, n] = sum_k x[m, k] * W[n, k], with W[n, k] = (Q[n, k] - Z[n, g]) * S[n, g].
//
// One warp per 16x16 output tile. `zero_points` is null for a symmetric tensor,
// which is the absent section rather than a buffer of zeros.
extern "C" __global__ void moxie_affine_linear_v1(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const short* __restrict__ zero_points,
    __nv_bfloat16* __restrict__ output,
    unsigned long long rows,
    unsigned long long in_features,
    unsigned long long out_features,
    unsigned long long row_stride,
    unsigned long long groups_per_row,
    unsigned int code_bits,
    unsigned int group_size,
    unsigned int scale_kind) {
    namespace wmma = nvcuda::wmma;

    const unsigned long long n0 =
        static_cast<unsigned long long>(blockIdx.x) * MOXIE_AFFINE_TILE;
    const unsigned long long m0 =
        static_cast<unsigned long long>(blockIdx.y) * MOXIE_AFFINE_TILE;
    if (n0 >= out_features || m0 >= rows) return;

    __shared__ __nv_bfloat16 xs[MOXIE_AFFINE_TILE * MOXIE_AFFINE_TILE];
    __shared__ __nv_bfloat16 ws[MOXIE_AFFINE_TILE * MOXIE_AFFINE_TILE];
    __shared__ float os[MOXIE_AFFINE_TILE * MOXIE_AFFINE_TILE];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0F);

    // Lane `t` owns half a tile row: eight consecutive columns of row `t / 2`.
    // That assignment, not a strided one, is what makes the group constant
    // across a lane's eight columns.
    const unsigned lane = threadIdx.x;
    const unsigned slot = lane >> 1;
    const unsigned half = (lane & 1U) * 8U;
    const __nv_bfloat16 zero_bf16 = __float2bfloat16_rn(0.0F);

    for (unsigned long long k0 = 0; k0 < in_features; k0 += MOXIE_AFFINE_TILE) {
        const unsigned long long m = m0 + slot;
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned column = half + j;
            const unsigned long long k = k0 + column;
            xs[slot * MOXIE_AFFINE_TILE + column] =
                (m < rows && k < in_features)
                    ? x[m * in_features + k]
                    : zero_bf16;
        }

        const unsigned long long n = n0 + slot;
        const bool row_live = n < out_features;
        float scale = 0.0F;
        int zero = 0;
        if (row_live) {
            const unsigned long long group =
                (groups_per_row == 1ULL)
                    ? 0ULL
                    : ((k0 + half) / static_cast<unsigned long long>(group_size));
            const unsigned long long entry = n * groups_per_row + group;
            scale = moxie_affine_scale_v1(scales, entry, scale_kind);
            zero = (zero_points == nullptr)
                       ? 0
                       : static_cast<int>(zero_points[entry]);
        }
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned column = half + j;
            const unsigned long long k = k0 + column;
            float value = 0.0F;
            if (row_live && k < in_features) {
                const int code =
                    moxie_affine_code_v1(codes, n * row_stride, k, code_bits);
                value = __fmul_rn(static_cast<float>(code - zero), scale);
                if (!isfinite(value)) {
                    // A finite code, zero point and scale can still multiply to
                    // something that is not. The host reader rejects such a
                    // tensor; if one reaches here anyway, poison the result
                    // rather than accumulate a wrong finite number.
                    value = moxie_affine_failure_v1();
                }
            }
            ws[slot * MOXIE_AFFINE_TILE + column] = __float2bfloat16_rn(value);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, xs, MOXIE_AFFINE_TILE);
        // Column-major over the same buffer: element (k, n) of B is
        // `ws[n * 16 + k]`, which is W[n0 + n][k0 + k] as written above. B is
        // the transpose of the weight, and the transpose is free here.
        wmma::load_matrix_sync(b, ws, MOXIE_AFFINE_TILE);
        wmma::mma_sync(acc, a, b, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(os, acc, MOXIE_AFFINE_TILE, wmma::mem_row_major);
    __syncthreads();

    for (unsigned i = lane; i < MOXIE_AFFINE_TILE * MOXIE_AFFINE_TILE; i += 32) {
        const unsigned long long m = m0 + i / MOXIE_AFFINE_TILE;
        const unsigned long long n = n0 + i % MOXIE_AFFINE_TILE;
        if (m < rows && n < out_features) {
            const float value = os[i];
            output[m * out_features + n] = __float2bfloat16_rn(
                isfinite(value) ? value : moxie_affine_failure_v1());
        }
    }
}
