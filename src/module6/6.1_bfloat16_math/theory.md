# 6.1 BFloat16 (`__nv_bfloat16`) Math & Mixed Precision

## 1. Why BFloat16 in LLM Training?

In modern LLM training (LLaMA, GPT-4, Mistral), **Brain Floating Point (BFloat16)** has largely superseded standard IEEE FP16 (Half Precision).

### Bit Allocations Comparison

| Format | Total Bits | Sign Bits | Exponent Bits | Mantissa (Fraction) Bits | Dynamic Range | Relative Precision ($\epsilon$) |
|---|---|---|---|---|---|---|
| **FP32** | 32 | 1 | 8 | 23 | $\approx 10^{\pm 38}$ | $\approx 1.2 \times 10^{-7}$ |
| **FP16** | 16 | 1 | 5 | 10 | $\approx 10^{\pm 4.8}$ (max 65,504) | $\approx 9.8 \times 10^{-4}$ |
| **BF16** | 16 | 1 | 8 | 7 | $\approx 10^{\pm 38}$ | $\approx 7.8 \times 10^{-3}$ |

### The Critical Advantage of BF16
* **Identical Exponent to FP32**: Because BF16 possesses 8 exponent bits, its dynamic range matches FP32 exactly.
* **Elimination of Loss Scaling**: Standard FP16 easily overflows at $65504$ or underflows at $6 \times 10^{-5}$, requiring dynamic loss scaling algorithms in optimizers. BF16 eliminates gradient underflow/overflow without needing loss scaling.
* **Trivial Conversion to/from FP32**: Truncating the lower 16 bits of an FP32 value yields a valid BF16 value (with truncation rounding), whereas rounding to nearest even (`_rn`) rounds the 7-bit mantissa.

---

## 2. CUDA Types and Headers

CUDA provides native support via `<cuda_bf16.h>`:
```cpp
#include <cuda_bf16.h>

// Types:
__nv_bfloat16   // Scalar 16-bit float
__nv_bfloat162  // Packed pair of two 16-bit floats (32 bits total)
```

### Conversions
```cpp
// Float to BF16:
float f = 3.14159265f;
__nv_bfloat16 b = __float2bfloat16(f);           // Round to nearest even
__nv_bfloat16 b_rz = __float2bfloat16_rz(f);     // Round toward zero

// BF16 to Float:
float f_out = __bfloat162float(b);

// Packed float2 to __nv_bfloat162:
float2 f2 = make_float2(1.0f, 2.0f);
__nv_bfloat162 b2 = __float22bfloat162_rn(f2);
```

---

## 3. Packed Vectorization (`__nv_bfloat162`)

By pairing two 16-bit numbers into a 32-bit register (`__nv_bfloat162`), GPU compute pipelines can perform SIMD arithmetic on 2 numbers simultaneously per ALU cycle:

```cpp
#if __CUDA_ARCH__ >= 800
__device__ __nv_bfloat162 fast_fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c); // 2 FMAs in a single instruction!
}
#endif
```

---

## 4. Mixed-Precision Accumulation Pattern

While storing tensors and activations in BF16 cuts memory bandwidth in half (16 bytes per 8 tokens instead of 32 bytes), **accumulating** reductions (dot products, softmax sums, RMSNorm squared sums) in BF16 quickly exhausts the 7-bit mantissa precision.

**Gold Standard Pattern**:
1. Load inputs as `__nv_bfloat16` or `__nv_bfloat162`.
2. Convert elements to `float` (FP32).
3. Accumulate sum / variance / product in 32-bit registers.
4. Perform final normalization / activation in FP32.
5. Convert final result back to `__nv_bfloat16` for global memory writeback.
