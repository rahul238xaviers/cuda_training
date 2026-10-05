# Topic 3.4: Grids, SM Occupancy & Grid-Stride Loops — Cheat Sheet

A concise reference for launch configurations, SM occupancy tuning, and scalable grid-stride loops.

---

### 1. The Grid-Stride Loop Pattern
- Instead of launching 1 thread per element (`idx < N`), threads loop with a stride equal to the total grid width:
  ```cpp
  __global__ void grid_stride_scale(const float* in, float* out, float scale, int n) {
      int stride = blockDim.x * gridDim.x;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
          out[i] = in[i] * scale;
      }
  }
  ```

---

### 2. Why Grid-Stride Loops Dominate Production CUDA
1. **Hardware Agnostic**: Runs correctly whether N=10, N=1,000,000, or N=1,000,000,000 without exceeding grid limits.
2. **Cache Locality**: Consecutive loop iterations maintain coalesced DRAM access patterns.
3. **Debugging Safety**: Allows running kernels on 1 block / 1 thread for debugging with identical semantics.

---

### 3. SM Theoretical Occupancy
```text
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)
```
- Limited by:
  - Registers per thread (too many registers limits active warps).
  - Shared memory per block (large tiles limit active blocks per SM).
  - Block size (too small, e.g. 64 threads, wastes warp slots).

---

### 4. LLM Systems Context
- **Fused Elementwise Kernels**: Rotary position embeddings, SwiGLU activations, and residual additions run with fixed-size persistent grid-stride kernels.
