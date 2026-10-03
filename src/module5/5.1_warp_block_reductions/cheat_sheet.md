# Topic 5.1: Warp & Block-Level Tree Reductions — Cheat Sheet

A concise reference for tree reductions across warps and blocks with minimal communication.

---

### 1. Two-Level Reduction Architecture
1. **Level 1 (Intra-Warp)**: Each warp of 32 threads reduces its values using `__shfl_down_sync` in 5 clock cycles with zero shared memory.
2. **Level 2 (Inter-Warp)**: The first lane of each warp (lane 0) writes its warp sum into shared memory `__shared__ float warp_sums[32]`.
3. **Level 3 (Final Aggregate)**: Warp 0 reads `warp_sums` and executes one final warp reduction.

---

### 2. Complete Block Reduction Function
```cpp
__device__ inline float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // 1 slot per warp (up to 1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val); // Warp reduction

    if (lane == 0) shared[wid] = val; // Store warp sum
    __syncthreads();

    // First warp reduces the warp sums
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val; // Lane 0 of Block holds overall total
}
```

---

### 3. LLM Systems Context
- **Global Loss Computation**: Computing cross-entropy loss across all sequence tokens.
- **Norm Layers**: Sum of squares reduction in RMSNorm.
