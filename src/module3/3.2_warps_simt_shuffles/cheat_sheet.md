# Topic 3.2: Warps, SIMT & Shuffle Intrinsics — Cheat Sheet

A concise reference for 32-thread lockstep execution, warp divergence, and register shuffles.

---

### 1. The Warp (32 Threads Lockstep)
- A **warp** consists of 32 consecutive threads in a block (`threadIdx.x` from `0` to `31`, `32` to `63`, etc.).
- The Warp Scheduler issues 1 instruction for all 32 threads simultaneously (SIMT).

---

### 2. Warp Divergence
- If threads within the same warp take different execution paths (`if (threadIdx.x % 2 == 0)`):
  - The hardware serializes the two branches.
  - Total time = Time(Branch A) + Time(Branch B).
- **Rule**: Branch on warp boundary (`blockIdx` or `threadIdx / 32`), never on thread lane ID.

---

### 3. Warp Shuffle Down (`__shfl_down_sync`)
- Exchange registers directly across threads without Shared Memory or DRAM trips:
  ```cpp
  #define FULL_MASK 0xffffffff
  // Thread i receives value from thread (i + delta)
  float val = __shfl_down_sync(FULL_MASK, val, delta);
  ```

---

### 4. 32-Thread Warp Reduction Tree (Log2(32) = 5 Steps)
```cpp
__device__ inline float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val; // Lane 0 holds the total sum of all 32 threads
}
```

---

### 5. LLM Systems Context
- **RMSNorm & Softmax**: Sum-of-squares and max reductions execute in 5 clock cycles inside registers with warp shuffles instead of hundreds of cycles in DRAM.
