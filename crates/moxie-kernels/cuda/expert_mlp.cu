// Task 0021's grouped expert kernels.
//
// Two launches per group, for the same reason task 0012 split RMSNorm into a
// sum and an apply: the gated intermediate is `intermediate` wide, is reduced
// over by every output component, and does not fit in registers at the
// designated artifact's 704. It is a declared FP32 workspace, not a hidden
// allocation inside a launch.
//
// Every boundary below is task 0019's accepted contract, and the reductions are
// **sequential ascending** with explicit `__fmul_rn` / `__fadd_rn`: nvcc
// contracts `a * b + c` into an FMA by default, which is a different rounding
// pattern and therefore a different answer from the oracle. The FP64 gate
// transform uses `__dmul_rn` / `__dadd_rn` for exactly the same reason.
//
// The gate transforms are two symbols rather than one with a flag. Document 02
// keeps GeGLU and SwiGLU as distinct operations because they have distinct
// declared rounding boundaries -- GeGLU rounds its gate term to BF16 before
// multiplying and SwiGLU evaluates the whole product in FP64 and rounds once --
// and a flag would invite one implementation to drift into serving both.
#include <cuda_bf16.h>

// sqrt(2/pi), transcribed from the pinned source, not recomputed.
static __device__ __forceinline__ double moxie_gelu_c0_v1() {
    return 0.7978845608028654;
}

static __device__ __forceinline__ float moxie_gelu_tanh_v1(float x) {
    const double v = static_cast<double>(x);
    const double cubed =
        __dmul_rn(__dmul_rn(__dmul_rn(0.044715, v), v), v);
    const double inner = __dmul_rn(moxie_gelu_c0_v1(), __dadd_rn(v, cubed));
    return static_cast<float>(
        __dmul_rn(__dmul_rn(0.5, v), __dadd_rn(1.0, tanh(inner))));
}

static __device__ __forceinline__ double moxie_sigmoid_v1(double v) {
    if (v >= 0.0) {
        return __ddiv_rn(1.0, __dadd_rn(1.0, exp(-v)));
    }
    const double e = exp(v);
    return __ddiv_rn(e, __dadd_rn(1.0, e));
}

// One lane of the gate/up projection: two sequential FP32 reductions over the
// row, each rounded to BF16, then the gate transform and its own BF16 boundary.
static __device__ __forceinline__ void moxie_expert_lanes_v1(
    const __nv_bfloat16* x_row, const __nv_bfloat16* gate_up,
    unsigned long long hidden, unsigned long long intermediate,
    unsigned long long lane, float* gate_out, float* up_out) {
    const unsigned long long gate_row = lane * hidden;
    const unsigned long long up_row = (intermediate + lane) * hidden;
    float gate = 0.0F;
    float up = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float xk = __bfloat162float(x_row[k]);
        gate = __fadd_rn(gate, __fmul_rn(xk, __bfloat162float(gate_up[gate_row + k])));
        up = __fadd_rn(up, __fmul_rn(xk, __bfloat162float(gate_up[up_row + k])));
    }
    *gate_out = __bfloat162float(__float2bfloat16_rn(gate));
    *up_out = __bfloat162float(__float2bfloat16_rn(up));
}

extern "C" __global__ void moxie_bf16_expert_project_gelu_v1(
    const __nv_bfloat16* x, const unsigned int* row_index,
    const __nv_bfloat16* gate_up, float* activated,
    unsigned long long assignments, unsigned long long hidden,
    unsigned long long intermediate) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * intermediate) return;
    const unsigned long long slot = index / intermediate;
    const unsigned long long lane = index % intermediate;
    const __nv_bfloat16* x_row =
        x + static_cast<unsigned long long>(row_index[slot]) * hidden;
    float gate = 0.0F;
    float up = 0.0F;
    moxie_expert_lanes_v1(x_row, gate_up, hidden, intermediate, lane, &gate, &up);
    // bf16(gelu_tanh(gate)) * up, the product in FP64 and narrowed once, then
    // the activation's own BF16 boundary.
    const double gated =
        static_cast<double>(__bfloat162float(__float2bfloat16_rn(moxie_gelu_tanh_v1(gate))));
    const float h = static_cast<float>(__dmul_rn(gated, static_cast<double>(up)));
    activated[index] = __bfloat162float(__float2bfloat16_rn(h));
}

extern "C" __global__ void moxie_bf16_expert_project_silu_v1(
    const __nv_bfloat16* x, const unsigned int* row_index,
    const __nv_bfloat16* gate_up, float* activated,
    unsigned long long assignments, unsigned long long hidden,
    unsigned long long intermediate) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * intermediate) return;
    const unsigned long long slot = index / intermediate;
    const unsigned long long lane = index % intermediate;
    const __nv_bfloat16* x_row =
        x + static_cast<unsigned long long>(row_index[slot]) * hidden;
    float gate = 0.0F;
    float up = 0.0F;
    moxie_expert_lanes_v1(x_row, gate_up, hidden, intermediate, lane, &gate, &up);
    // (g * sigmoid(g)) * up, all in FP64, narrowed once -- then the activation's
    // BF16 boundary, which belongs to the expert operation rather than to SwiGLU.
    const double g = static_cast<double>(gate);
    const float h = static_cast<float>(
        __dmul_rn(__dmul_rn(g, moxie_sigmoid_v1(g)), static_cast<double>(up)));
    activated[index] = __bfloat162float(__float2bfloat16_rn(h));
}

// The down projection, over the whole activated intermediate. Not tiled and not
// split: the reduction is over `intermediate` in ascending order and splitting
// it would reassociate the sum the oracle fixes.
extern "C" __global__ void moxie_bf16_expert_down_v1(
    const float* activated, const __nv_bfloat16* down,
    const unsigned int* slot_index, __nv_bfloat16* slots,
    unsigned long long assignments, unsigned long long hidden,
    unsigned long long intermediate) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * hidden) return;
    const unsigned long long slot = index / hidden;
    const unsigned long long component = index % hidden;
    const float* h = activated + slot * intermediate;
    const __nv_bfloat16* row = down + component * intermediate;
    float acc = 0.0F;
    for (unsigned long long i = 0; i < intermediate; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(h[i], __bfloat162float(row[i])));
    }
    slots[static_cast<unsigned long long>(slot_index[slot]) * hidden + component] =
        __float2bfloat16_rn(acc);
}

// Canonical integer operands share the decoder with dense affine execution.
// The payload is codes/scales/i16 zeros; map is an optional admitted operand.
// Packed sections may be unaligned, so metadata loads are byte-addressed.
#include "affine_decode.cuh"

static __device__ __forceinline__ float moxie_expert_affine_v1(
    const unsigned char* weight, const unsigned int* map,
    unsigned long long output, unsigned long long input,
    unsigned long long outputs, unsigned long long inputs,
    unsigned int bits, unsigned int group, unsigned int scale_kind,
    unsigned int has_zeros) {
    if (bits == 16U) {
        const unsigned char* p = weight + (output * inputs + input) * 2;
        return __uint_as_float((static_cast<unsigned int>(p[0]) | (static_cast<unsigned int>(p[1]) << 8)) << 16);
    }
    const unsigned long long stride = (inputs + (8 / bits) - 1) / (8 / bits);
    const unsigned long long groups = (inputs + group - 1) / group;
    unsigned long long logical_group = input / group;
    if (map) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(map) + input * 4;
        logical_group = static_cast<unsigned int>(p[0]) | (static_cast<unsigned int>(p[1]) << 8) | (static_cast<unsigned int>(p[2]) << 16) | (static_cast<unsigned int>(p[3]) << 24);
    }
    const unsigned long long entry = output * groups + logical_group;
    const unsigned long long scale_offset = outputs * stride;
    const unsigned long long zero_offset = scale_offset + outputs * groups * (scale_kind == 2 ? 4 : 2);
    int zero = 0;
    if (has_zeros) {
        const unsigned char* p = weight + zero_offset + entry * 2;
        zero = static_cast<short>(static_cast<unsigned short>(p[0]) | (static_cast<unsigned short>(p[1]) << 8));
    }
    const int code = moxie_affine_code_v1(weight, output * stride, input, bits);
    const float scale = moxie_affine_scale_v1(weight + scale_offset, entry, scale_kind);
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(static_cast<float>(code - zero), scale)));
}

static __device__ __forceinline__ void moxie_affine_expert_lanes_v1(
    const __nv_bfloat16* x_row, const unsigned char* gate_up, const unsigned int* map,
    unsigned long long hidden, unsigned long long intermediate, unsigned long long lane,
    unsigned int bits, unsigned int group, unsigned int scale_kind, unsigned int has_zeros,
    float* gate_out, float* up_out) {
    float gate = 0.0F, up = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float xk = __bfloat162float(x_row[k]);
        gate = __fadd_rn(gate, __fmul_rn(xk, moxie_expert_affine_v1(gate_up, map, lane, k, 2 * intermediate, hidden, bits, group, scale_kind, has_zeros)));
        up = __fadd_rn(up, __fmul_rn(xk, moxie_expert_affine_v1(gate_up, map, intermediate + lane, k, 2 * intermediate, hidden, bits, group, scale_kind, has_zeros)));
    }
    *gate_out = __bfloat162float(__float2bfloat16_rn(gate));
    *up_out = __bfloat162float(__float2bfloat16_rn(up));
}

extern "C" __global__ void moxie_affine_expert_project_gelu_v1(
    const __nv_bfloat16* x, const unsigned int* row_index,
    const unsigned char* gate_up, const unsigned int* map, float* activated,
    unsigned long long assignments, unsigned long long hidden, unsigned long long intermediate,
    unsigned int bits, unsigned int group, unsigned int scale_kind, unsigned int has_zeros) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * intermediate) return;
    float gate, up;
    moxie_affine_expert_lanes_v1(x + static_cast<unsigned long long>(row_index[index / intermediate]) * hidden,
        gate_up, map, hidden, intermediate, index % intermediate, bits, group, scale_kind, has_zeros, &gate, &up);
    const double gated = static_cast<double>(__bfloat162float(__float2bfloat16_rn(moxie_gelu_tanh_v1(gate))));
    const float h = static_cast<float>(__dmul_rn(gated, static_cast<double>(up)));
    activated[index] = __bfloat162float(__float2bfloat16_rn(h));
}

extern "C" __global__ void moxie_affine_expert_project_silu_v1(
    const __nv_bfloat16* x, const unsigned int* row_index,
    const unsigned char* gate_up, const unsigned int* map, float* activated,
    unsigned long long assignments, unsigned long long hidden, unsigned long long intermediate,
    unsigned int bits, unsigned int group, unsigned int scale_kind, unsigned int has_zeros) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * intermediate) return;
    float gate, up;
    moxie_affine_expert_lanes_v1(x + static_cast<unsigned long long>(row_index[index / intermediate]) * hidden,
        gate_up, map, hidden, intermediate, index % intermediate, bits, group, scale_kind, has_zeros, &gate, &up);
    const double g = static_cast<double>(gate);
    const float h = static_cast<float>(__dmul_rn(__dmul_rn(g, moxie_sigmoid_v1(g)), static_cast<double>(up)));
    activated[index] = __bfloat162float(__float2bfloat16_rn(h));
}

extern "C" __global__ void moxie_affine_expert_down_v1(
    const float* activated, const unsigned char* down, const unsigned int* map,
    const unsigned int* slot_index, __nv_bfloat16* slots,
    unsigned long long assignments, unsigned long long hidden, unsigned long long intermediate,
    unsigned int bits, unsigned int group, unsigned int scale_kind, unsigned int has_zeros) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * hidden) return;
    const unsigned long long slot = index / hidden;
    const unsigned long long component = index % hidden;
    float acc = 0.0F;
    for (unsigned long long k = 0; k < intermediate; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(activated[slot * intermediate + k], moxie_expert_affine_v1(down, map, component, k, hidden, intermediate, bits, group, scale_kind, has_zeros)));
    }
    slots[static_cast<unsigned long long>(slot_index[slot]) * hidden + component] = __float2bfloat16_rn(acc);
}
