# Topic 6.1: BFloat16 (`__nv_bfloat16`) Math & Vectorized Types — Cheat Sheet

A concise reference for 16-bit brain float format, dynamic range, and packed arithmetic.

---

### 1. FP32 vs. FP16 vs. BF16 Bit Representation
- **FP32**: 1 sign bit, 8 exponent bits, 23 mantissa bits.
- **FP16**: 1 sign bit, 5 exponent bits, 10 mantissa bits. (Prone to underflow/overflow!).
- **BF16**: 1 sign bit, 8 exponent bits, 7 mantissa bits.
  - **Identical dynamic range to FP32** (`~1e-38` to `~3e38`), eliminating loss scaling!

---

### 2. Packed 2-Way Math (`__nv_bfloat162`)
- NVIDIA Ampere/Hopper/Blackwell execute two 16-bit operations simultaneously in a single 32-bit register:
  ```cpp
  #include <cuda_bf16.h>

  __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&in_a[idx]);
  __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&in_b[idx]);

  // Fused Multiply-Add on 2 packed BF16 values simultaneously
  __nv_bfloat162 res = __hfma2(a, b, c);
  ```

---

### 3. Conversion Intrinsics
- `__float2bfloat16(val_f32)`: Converts float to BF16 with round-to-nearest-even.
- `__bfloat162float(val_bf16)`: Converts BF16 to float.

---

### 4. LLM Systems Context
- **Standard LLM Format**: LLaMA, Mistral, and Gemma models train and infer natively in BF16 to save 50% memory bandwidth and VRAM over FP32.
