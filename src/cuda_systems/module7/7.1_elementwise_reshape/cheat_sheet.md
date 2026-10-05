# Topic 7.1: Elementwise Operations & Fused Reshaping — Cheat Sheet

A concise reference for fused activations, residual additions, and memory bandwidth saturation.

---

### 1. Fused Residual Add & Bias
- **Unfused (Slow)**:
  `A = X + Bias` (Write to DRAM) -> `B = A + Residual` (Read/Write DRAM).
- **Fused (Peak Speed)**:
  ```cpp
  __global__ void fused_add_residual(const float4* x, const float4* res, float4* out, int n4) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n4) {
          float4 v_x = x[idx];
          float4 v_r = res[idx];
          float4 ans;
          ans.x = v_x.x + v_r.x;
          ans.y = v_x.y + v_r.y;
          ans.z = v_x.z + v_r.z;
          ans.w = v_x.w + v_r.w;
          out[idx] = ans;
      }
  }
  ```

---

### 2. Fused SwiGLU Activation
```text
SwiGLU(G, U) = (G * sigmoid(G)) * U = (G / (1 + exp(-G))) * U
```
- Simultaneously streams Gate and Up projection tensors with 128-bit loads, computing SiLU and Hadamard product in registers.

---

### 3. LLM Systems Context
- **Transformer Block Skips**: Every Transformer layer executes 2 residual additions and 1 SwiGLU feed-forward activation.
