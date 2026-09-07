// Smoke kernels.
//
// These exist to prove the toolchain, the fatbin packaging and the launch path,
// on every architecture the product claims. They are not the engine's kernels
// and carry no performance intent.
//
// Document 03 requires a feature matrix built "from device queries and launch
// probes, not GPU marketing names". `smoke_axpy_f32` is the launch probe.

#include <cuda_bf16.h>

extern "C" __global__ void moxie_smoke_axpy_f32(const float* x, float* y, float a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Round f32 -> bf16 on the device and return the raw bit pattern.
//
// Document 03 pins "BF16 little-endian weights with specified round-to-nearest-
// even conversion from FP32". The host oracle in moxie-format implements that
// rule independently; this kernel lets the GPU lane check that the device agrees
// on the same inputs, including the ties and the non-finite cases.
extern "C" __global__ void moxie_smoke_f32_to_bf16_bits(const float* src, unsigned short* dst, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 h = __float2bfloat16(src[i]);
        dst[i] = *reinterpret_cast<unsigned short*>(&h);
    }
}
