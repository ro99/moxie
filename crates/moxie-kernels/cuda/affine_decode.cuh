#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Scale encodings, in the source's own dtype. `kind` is the closed
// `ScaleDtype` tag: 0 = f16, 1 = bf16, 2 = f32. Reading the source encoding
// here is what keeps an FP16 scale from being rounded to BF16 on the way to
// the device, which document 03 classifies as a value change.
static __device__ __forceinline__ float moxie_affine_scale_v1(
    const unsigned char* scales, unsigned long long entry, unsigned int kind) {
    if (kind == 2U) {
        const unsigned char* p = scales + entry * 4;
        return __uint_as_float(static_cast<unsigned int>(p[0]) | (static_cast<unsigned int>(p[1]) << 8) | (static_cast<unsigned int>(p[2]) << 16) | (static_cast<unsigned int>(p[3]) << 24));
    }
    const unsigned char* p = scales + entry * 2;
    const unsigned short bits = static_cast<unsigned short>(p[0]) | (static_cast<unsigned short>(p[1]) << 8);
    if (kind == 1U) {
        return __uint_as_float(static_cast<unsigned int>(bits) << 16);
    }
    return __half2float(__ushort_as_half(bits));
}

// The signed code at logical column `k` of a row that starts at `row_base`
// bytes. INT4 packs two per byte, **low nibble first**: the low nibble is the
// earlier logical input column. Reading the pair the other way round transposes
// every adjacent column in the tensor, which no accuracy tolerance would catch.
static __device__ __forceinline__ int moxie_affine_code_v1(
    const unsigned char* codes, unsigned long long row_base,
    unsigned long long k, unsigned int bits) {
    if (bits == 8U) {
        return static_cast<int>(static_cast<signed char>(codes[row_base + k]));
    }
    const unsigned char byte = codes[row_base + (k >> 1)];
    const int nibble = static_cast<int>(
        (k & 1ULL) ? (byte >> 4) : (byte & 0x0FU));
    return nibble >= 8 ? nibble - 16 : nibble;
}

