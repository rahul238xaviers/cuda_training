# Topic 5.2: Atomic Operations & Contention Reduction — Cheat Sheet

A concise reference for hardware atomic intrinsics, fences, and contention mitigation.

---

### 1. Hardware Atomic Intrinsics
- Read-Modify-Write performed at the L2 cache or memory controller in a single transaction:
  ```cpp
  atomicAdd(float* address, float val);
  atomicMax(int* address, int val);
  atomicCAS(int* address, int compare, int val);
  ```

---

### 2. Reducing Atomic Contention (Hierarchical Aggregation)
- **Bad (High Contention)**: 100,000 threads simultaneously calling `atomicAdd(&global_sum, val)`. Hardware serializes all 100,000 transactions!
- **Good (Hierarchical)**:
  1. Threads reduce locally within their block using `block_reduce_sum`.
  2. Exactly **1 thread per block** executes `atomicAdd(&global_sum, block_sum)`.
  3. Contention reduced by 256x to 1024x!

---

### 3. LLM Systems Context
- **Gradient Accumulation**: Accumulating parameter gradients across multiple micro-batches in Distributed Data Parallel (DDP) training.
- **Sparse MoE Routing**: Aggregating token counts assigned to each expert in Mixture-of-Experts routing.
