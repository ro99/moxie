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
