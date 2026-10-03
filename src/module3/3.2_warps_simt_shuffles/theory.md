# Chapter 3.2: Warps, SIMT Execution Model & Warp Shuffle Primitives

If Chapter 3.1 covers what threads are, Chapter 3.2 covers how 32 threads collaborate without going through memory. Warp shuffle intrinsics are one of the most powerful CUDA primitives available — they allow register-to-register data exchange across 32 threads in a single clock cycle, completely bypassing shared memory, global memory, and even L1. Understanding them deeply is the gateway to writing state-of-the-art reduction kernels, softmax, and layer normalization.

---

## 1. Recap: The Warp as the Atomic Execution Unit

An NVIDIA SM does not execute individual threads. It executes **warps**: groups of exactly 32 threads that share one program counter and execute one instruction per cycle across all 32 SIMD lanes simultaneously.

```text
SM Warp Scheduler View (4 schedulers, each can issue 1 instruction/cycle):

Cycle  1: Scheduler 0 issues ADD to Warp 0  (lanes 0-31 compute simultaneously)
Cycle  2: Scheduler 1 issues LDG to Warp 4  (lanes 0-31 issue memory loads)
Cycle  3: Scheduler 2 issues FMA to Warp 8  (lanes 0-31 multiply-accumulate)
Cycle  4: Scheduler 3 issues SHF to Warp 12 (lanes 0-31 shuffle registers)
...
Cycle 200+: Warp 4 finally returns from memory → Scheduler 1 can re-issue to it
```

The critical property: **All 32 threads in a warp share a single instruction counter**. They must all execute the same instruction on each cycle. Divergence is the violation of this principle.

---

## 2. SIMT vs. SIMD: Why CUDA Is More Flexible

SIMD (Single Instruction Multiple Data) — as in AVX-512 — requires all lanes to execute identically with no conditional masking.

CUDA's SIMT (Single Instruction Multiple Thread) is more flexible:
- Each lane can predicate itself (be "inactive") on a per-lane basis.
- Divergent branches are handled by executing both paths with different predicate masks.
- Lanes can even have separate stack frames for function calls (though this is expensive).

```text
Branch execution with SIMT:

if (lane_id < 16) {
    do_A();   // → PC at instruction A
} else {
    do_B();   // → PC at instruction B
}

Hardware execution:
  Pass 1: Predicate mask = [1,1,...,1,0,0,...,0] (lanes 0-15 active)
          All 32 lanes execute do_A(); lanes 16-31 are masked → results discarded
  Pass 2: Predicate mask = [0,0,...,0,1,1,...,1] (lanes 16-31 active)
          All 32 lanes execute do_B(); lanes 0-15 are masked → results discarded
  Total: 2x the latency of a non-divergent warp.
```

---

## 3. Warp-Level Primitives: Cooperative Thread Arrays Within a Warp

Before the shuffle intrinsics, communicating between threads required shared memory:

```cpp
// Old pattern (pre-Kepler): slow shared memory exchange
__shared__ float smem[32];
smem[threadIdx.x % 32] = my_val;
__syncthreads();              // Wait for all threads to write
float neighbor_val = smem[(threadIdx.x + 1) % 32];  // Read neighbor's value
```

This requires a full round-trip to on-chip SRAM: write, synchronize, read. Latency: ~23 cycles. Plus the `__syncthreads()` barrier halts the entire block.

Since CUDA 9.0 and sm_70 (Volta), warp shuffle intrinsics bypass all of this:
- Registers are moved directly between lanes **in the register file** — no SRAM involved.
- Completes in **1 clock cycle**.
- No synchronization barrier needed (the synchronization is implicit in the shuffle mask).

---

## 4. The Four Shuffle Primitives

All shuffles take a **mask** as the first argument. The mask is a 32-bit bitmask specifying which lanes are participating. For a full warp, use `0xffffffff`.

### 4.1 `__shfl_sync`: Lane Broadcast

```cpp
// Broadcast lane `src_lane`'s value to all participating lanes
float result = __shfl_sync(mask, val, src_lane);
```

Use case: All 32 threads need to know lane 0's computed value (e.g., the block's reduction result).

```cpp
// Example: Broadcast lane 0's computed scale to all lanes in the warp
float scale = (lane_id == 0) ? compute_scale(data) : 0.0f;
scale = __shfl_sync(0xffffffff, scale, 0);  // All lanes now have scale
output[idx] = input[idx] * scale;
```

### 4.2 `__shfl_down_sync`: Downward Shift

```cpp
// Lane i receives the value from lane (i + delta)
float result = __shfl_down_sync(mask, val, delta);
// Lanes in the top `delta` positions get undefined/own values
```

Use case: Tree reductions. This is the most commonly used shuffle primitive.

```text
Example: __shfl_down_sync(0xffffffff, val, 4)

Before:                        After:
Lane  0: val=10   Lane  0: receives lane 4's val=50 → val=50
Lane  1: val=20   Lane  1: receives lane 5's val=60 → val=60
Lane  2: val=30   Lane  2: receives lane 6's val=70 → val=70
Lane  3: val=40   Lane  3: receives lane 7's val=80 → val=80
Lane  4: val=50   Lane  4: receives lane 8's val=...
...
Lane 28: val=290  Lane 28: receives lane 32 (undefined) → unchanged
...
Lane 31: val=320  Lane 31: unchanged (no lane 32 exists)
```

### 4.3 `__shfl_up_sync`: Upward Shift

```cpp
// Lane i receives the value from lane (i - delta)
float result = __shfl_up_sync(mask, val, delta);
```

Use case: Prefix sum (exclusive scan) — each lane needs the accumulated value from all preceding lanes.

### 4.4 `__shfl_xor_sync`: Butterfly Exchange

```cpp
// Lane i exchanges with lane (i XOR laneMask)
float result = __shfl_xor_sync(mask, val, laneMask);
```

Use case: Butterfly reduction networks (used in all-reduce across sub-groups).

```text
Example: __shfl_xor_sync(0xffffffff, val, 1)  (swap adjacent pairs)

Lane 0 ↔ Lane 1    (0 XOR 1 = 1)
Lane 2 ↔ Lane 3    (2 XOR 1 = 3)
Lane 4 ↔ Lane 5    (4 XOR 1 = 5)
...
Lane 30 ↔ Lane 31
```

---

## 5. Canonical Warp Reduction: 5 Instructions, 32 Threads → 1 Sum

The most important use of shuffle intrinsics is the tree reduction. In 5 shuffle-down steps (log2(32) = 5), all 32 thread values collapse into a single sum in lane 0.

```cpp
// Warp-level sum reduction using shuffle down:
__device__ inline float warp_reduce_sum(float val) {
    // Step 1: Each lane adds the value from lane+16
    val += __shfl_down_sync(0xffffffff, val, 16);
    // Step 2: Each lane adds the value from lane+8
    val += __shfl_down_sync(0xffffffff, val, 8);
    // Step 3: Each lane adds the value from lane+4
    val += __shfl_down_sync(0xffffffff, val, 4);
    // Step 4: Each lane adds the value from lane+2
    val += __shfl_down_sync(0xffffffff, val, 2);
    // Step 5: Each lane adds the value from lane+1
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
    // Lane 0 now holds the sum of all 32 original values
    // All other lanes hold partial sums (lane 1 holds sum of lanes 1-31, etc.)
}

__device__ inline float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}
```

### Visualizing the Tree Reduction Steps

```text
Initial values: [10, 3, 7, 2, 8, 5, 1, 4, ...]  (32 values, showing 8)

Step 1 (delta=4): lane i += lane (i+4)
  Lane 0: 10+8=18    Lane 1: 3+5=8    Lane 2: 7+1=8    Lane 3: 2+4=6    ...

Step 2 (delta=2): lane i += lane (i+2)
  Lane 0: 18+8=26    Lane 1: 8+6=14   ...

Step 3 (delta=1): lane i += lane (i+1)
  Lane 0: 26+14=40   → Total sum of first 8 elements

(Repeated for the full 32 elements across all 5 steps)
```

Each step takes 1 clock cycle. Total: 5 cycles to reduce 32 values. Compare to sequential summation: 31 additions, 31 cycles. **6x faster in register, with no memory traffic**.

---

## 6. Block-Level Reduction via Warps + Shared Memory

A full block of 256 threads has 8 warps. To reduce all 256 values to one scalar:

```cpp
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // Phase 1: Each warp reduces its 32 lanes to 1 value
    val = warp_reduce_sum(val);

    // Phase 2: Lane 0 of each warp writes its partial result to shared memory
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();  // Wait for all warp results to land in smem

    // Phase 3: Thread 0 (or first warp) loads and reduces the warp results
    if (threadIdx.x < num_warps) {
        val = smem[threadIdx.x];
    } else {
        val = 0.0f;
    }

    // Phase 4: Final warp reduction over the 8 warp results (only first warp active)
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    // Thread 0 now holds the total sum of all 256 values
    return val;
}
```

This pattern — warp shuffle first, then one shared memory write per warp, then final warp shuffle — is the standard block reduction template used in PyTorch, FlashAttention, and cuBLAS.

---

## 7. Prefix Sum (Exclusive Scan) Within a Warp

Prefix sum is the building block for dynamic allocation patterns, CUDA CSR graph algorithms, and online normalization (like Online Softmax's normalizer tracking).

```cpp
// Warp-level inclusive prefix sum (each lane gets sum of lanes 0..lane):
__device__ float warp_prefix_sum(float val) {
    float result = val;

    for (int delta = 1; delta <= 16; delta <<= 1) {
        float prev = __shfl_up_sync(0xffffffff, result, delta);
        if (threadIdx.x % 32 >= delta) result += prev;
    }
    return result;
}

// Example result for input [1, 1, 1, 1, 1, ...] (all ones):
// Lane 0: 1, Lane 1: 2, Lane 2: 3, ..., Lane 31: 32
```

---

## 8. Inter-Warp Synchronization: `__syncthreads()` vs. `cooperative_groups`

### `__syncthreads()` — Block Barrier
Waits for **all threads in the block** to reach the barrier. No thread proceeds until every thread has reached `__syncthreads()`.

```cpp
__shared__ float smem[256];
smem[threadIdx.x] = val;  // Write phase
__syncthreads();            // Barrier: wait for all writes to complete
float neighbor = smem[(threadIdx.x + 1) % 256];  // Safe to read
```

### `__syncwarp()` — Warp Barrier (Volta+)
Waits only for threads within the same warp. Cheaper, but only safe within a single warp.

```cpp
// After warp shuffle, values are consistent within the warp automatically.
// __syncwarp() is needed if one lane writes to shared memory and another reads it:
smem[lane_id] = my_val;
__syncwarp();               // Sync only the 32 warp threads
float read_val = smem[(lane_id + 1) % 32];
```

### `cooperative_groups` — Fine-Grained Sync

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_example(float* data) {
    // Thread block group (all threads in block)
    cg::thread_block block = cg::this_thread_block();

    // Warp-level group (all 32 threads in current warp)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Built-in warp reduce via cooperative groups:
    float sum = cg::reduce(warp, data[warp.thread_rank()], cg::plus<float>());
}
```

---

## 9. Applied Pattern: Online Softmax in a Single Warp Pass

Softmax of a vector `x[32]` (one value per lane) requires:
1. Find max of all 32 values
2. Compute `exp(x[i] - max)` for each lane
3. Sum all `exp(...)` values
4. Divide each by the sum

Without shuffles: 3 passes through shared memory. With shuffles: done in registers, no memory.

```cpp
__device__ float warp_softmax(float x, float* smem_out) {
    int lane = threadIdx.x % 32;

    // Step 1: Warp max
    float max_val = warp_reduce_max(x);
    max_val = __shfl_sync(0xffffffff, max_val, 0);  // Broadcast to all lanes

    // Step 2: Compute exp(x - max)
    float exp_val = expf(x - max_val);

    // Step 3: Warp sum of exp values
    float sum_exp = warp_reduce_sum(exp_val);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);  // Broadcast to all lanes

    // Step 4: Normalize
    return exp_val / sum_exp;
}
```

This is the essential building block of the FlashAttention numerically stable online softmax algorithm.

---

## 10. Summary: Shuffle Primitives Cheat Sheet

| Intrinsic | Direction | Use Case |
| :--- | :--- | :--- |
| `__shfl_sync(mask, val, src_lane)` | Broadcast | Distribute one lane's result to all |
| `__shfl_down_sync(mask, val, delta)` | Downward | Tree reduction (sum, max, min) |
| `__shfl_up_sync(mask, val, delta)` | Upward | Prefix sum / scan |
| `__shfl_xor_sync(mask, val, laneMask)` | Butterfly | All-reduce, butterfly sort |

**Key rule**: Use `0xffffffff` as mask when all 32 threads are unconditionally active. For divergent warps (e.g., after an `if` branch), construct the mask from `__activemask()` to avoid undefined behavior.
