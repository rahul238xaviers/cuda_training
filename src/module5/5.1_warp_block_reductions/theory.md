# 5.1 Two-Pass Block & Warp Reductions

## 1. Reduction Architecture in High-Performance GPU Computing

Computing the sum, maximum, or norm across thousands of elements is at the heart of ML layers (LayerNorm, RMSNorm, Softmax, Cross-Entropy Loss).

A naive approach using `atomicAdd` on a single global variable causes extreme memory serialization (thousands of threads stalling on a single cache line).

### The Canonical Two-Pass Block Reduction
Instead of global atomics, high-performance kernels use a hierarchical two-pass reduction:
```
All Threads (e.g. 256 threads = 8 warps)
   │
   ▼
[Phase 1]: Intra-Warp Shuffle Reductions (5 clock cycles in registers)
   │
   ▼
[Phase 2]: Lane 0 of each warp writes to Shared Scratchpad (8 floats)
   │
   ▼  <-- __syncthreads() barrier
[Phase 3]: Warp 0 reads Shared Scratchpad (8 elements)
   │
   ▼
[Phase 4]: Warp 0 performs final intra-warp reduction
   │
   ▼
Lane 0 holds the exact sum for all 256 threads!
```

---

## 2. Multi-Value Reductions for Normalization

In RMSNorm and LayerNorm, kernels must compute multiple statistics across a row simultaneously:
- For RMSNorm: $\sum x_i^2$
- For LayerNorm: $\sum x_i$ and $\sum x_i^2$ (to compute mean $\mu$ and variance $\sigma^2$)

Instead of running two separate reduction passes, threads accumulate a pair `float2 (sum, sum_sq)` in registers and reduce both simultaneously using vectorized shuffles or dual shuffles:
```cpp
__device__ inline float2 warp_reduce_sum2(float2 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}
```
This halves the reduction overhead and cuts shared memory traffic by 50%.
