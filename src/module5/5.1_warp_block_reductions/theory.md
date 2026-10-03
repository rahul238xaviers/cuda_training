# Chapter 5.1: Warp & Block Reductions — From Primitives to Production Patterns

Reduction is everywhere in deep learning: LayerNorm, RMSNorm, Softmax, Cross-Entropy Loss, and attention score scaling all require reducing thousands of values to one scalar (or a small set of scalars) per row. The difference between a naive reduction and a production one is not just speed — it is numerical stability, register efficiency, and the ability to co-locate the reduction with other computation in a single kernel pass.

---

## 1. Why Not Just Use `atomicAdd` Into a Global Variable?

The naive approach:

```cpp
// SLOW: global atomic reduction
__global__ void naive_sum(const float* input, float* total, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) atomicAdd(total, input[gid]);  // All threads hammer one location!
}
```

### The Hardware Problem

`atomicAdd` on a global memory address works by:
1. Acquiring exclusive ownership of the cache line containing `total`.
2. Reading the current value.
3. Adding.
4. Writing back.
5. Releasing ownership.

When 2,048 threads all atomicAdd to the same address simultaneously:
- The memory controller serializes all 2,048 requests one by one.
- Each atomic operation stalls all other threads.
- Total time = 2,048 × (cache line latency) — completely sequential execution despite massive parallelism.

**Rule**: Never use global atomics as the primary reduction mechanism. Use them only at the final step after a hierarchical block-level reduction.

---

## 2. The Hierarchical Reduction Principle

The correct approach builds reductions in layers, from fastest to slowest memory:

```text
Level 1: Intra-Warp Reduction (Registers)
  Tool: __shfl_down_sync
  Latency: 5 steps × 1 cycle = 5 cycles
  Memory: Zero — entirely in registers

Level 2: Warp-to-Block Communication (Shared Memory)
  Tool: smem[warp_id] = warp_result
  Latency: ~23 cycles per write/read + __syncthreads()
  Memory: 1 float per warp = 8 floats for 256-thread block

Level 3: Block-to-Grid Communication (Global Memory Atomic)
  Tool: atomicAdd(global_result, block_result)
  Latency: ~400 cycles
  Memory: 1 float for the entire grid
  Frequency: Once per block (only 128 atomics for 128-block grid)
```

```text
256 threads (8 warps):

Warp 0: threads 0-31  ──→ shuffle_reduce ──→ partial_sum_0  ──┐
Warp 1: threads 32-63 ──→ shuffle_reduce ──→ partial_sum_1  ──│
Warp 2: threads 64-95 ──→ shuffle_reduce ──→ partial_sum_2  ──│──→ smem[0..7]
Warp 3: threads 96-127──→ shuffle_reduce ──→ partial_sum_3  ──│
Warp 4-7: ...          ──→ shuffle_reduce ──→ partial_sum_4-7─┘

  __syncthreads()   ← Wait for all smem writes to land

Warp 0 reads smem[0..7] ──→ shuffle_reduce over 8 values ──→ block_total (lane 0)

Lane 0: atomicAdd(global_result, block_total);  ← one atomic per block
```

---

## 3. The Complete Block Reduction Template

```cpp
// Reusable block reduction for sum. Call with any value per thread.
// Result is in thread 0 after the call.
__device__ float block_reduce_sum(float val) {
    // Phase 1: smem scratch for warp partial results
    __shared__ float warp_sums[32];  // At most 32 warps per block (1024 threads)

    int lane    = threadIdx.x & 31;       // Thread's lane within its warp
    int warp_id = threadIdx.x >> 5;       // Which warp this thread belongs to
    int n_warps = (blockDim.x + 31) >> 5; // Number of warps in the block

    // Phase 1: Each warp reduces its 32 lanes to 1 value (5 shuffle steps)
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    // Phase 2: Lane 0 of each warp deposits its partial sum
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();   // All warp deposits visible before any reads

    // Phase 3: Load the warp partial sums (only first warp participates)
    val = (threadIdx.x < n_warps) ? warp_sums[threadIdx.x] : 0.0f;

    // Phase 4: Warp 0 reduces the warp partial sums to a single block total
    if (warp_id == 0) {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);
    }

    return val;  // Only thread 0 (lane 0 of warp 0) has the correct total
}

// Usage in a reduction kernel:
__global__ void sum_reduction_kernel(const float* input, float* output, int N) {
    float local_sum = 0.0f;

    // Grid-stride accumulation (each thread sums multiple elements)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        local_sum += input[i];

    // Block reduction (result in thread 0 of each block)
    float block_sum = block_reduce_sum(local_sum);

    // Block 0's thread 0 writes the final global result
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum);   // Only 1 atomic per block!
}
```

---

## 4. Multi-Value Simultaneous Reductions

LayerNorm needs both mean and variance in one pass, requiring two reductions. Instead of running the block twice, compute both statistics in a single reduction with a `float2` accumulator:

```cpp
__device__ float2 warp_reduce_float2(float2 val) {
    // Simultaneously reduce both x (sum) and y (sum of squares)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

__device__ float2 block_reduce_float2(float2 val) {
    __shared__ float2 warp_partials[32];
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_float2(val);
    if (lane == 0) warp_partials[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < n_warps) ? warp_partials[threadIdx.x] : make_float2(0.f, 0.f);
    if (warp_id == 0) val = warp_reduce_float2(val);
    return val;
}

// LayerNorm kernel using dual reduction:
__global__ void layernorm_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int D)
{
    int row = blockIdx.x;
    const float* row_x = x + row * D;
    float* row_out = out + row * D;

    // Accumulate sum and sum_sq simultaneously:
    float2 stats = make_float2(0.0f, 0.0f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = row_x[i];
        stats.x += v;          // sum of values
        stats.y += v * v;      // sum of squared values
    }

    // One block reduction → both mean and variance:
    stats = block_reduce_float2(stats);
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) {
        float mean = stats.x / D;
        float var  = stats.y / D - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + 1e-5f);
    }
    __syncthreads();

    // Normalize using broadcast from shared memory:
    float mean = s_mean, rstd = s_rstd;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_out[i] = (row_x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}
```

Single kernel, one HBM read of the input row (two passes over it in shared memory / L1 cache), correct LayerNorm computation.

---

## 5. The Tree Reduction: Visualized Step by Step

For a block of 8 threads (simplified), reducing values `[3, 1, 4, 1, 5, 9, 2, 6]`:

```text
Initial:  [3, 1, 4, 1, 5, 9, 2, 6]   (lane 0..7)

delta=4:  lane i += shfl_down(val, 4)
          lane 0: 3 + 5 = 8
          lane 1: 1 + 9 = 10
          lane 2: 4 + 2 = 6
          lane 3: 1 + 6 = 7
          lane 4..7: unchanged (delta=4 has no source)
Values:   [8, 10, 6, 7, 5, 9, 2, 6]

delta=2:  lane i += shfl_down(val, 2)
          lane 0: 8 + 6 = 14
          lane 1: 10 + 7 = 17
Values:   [14, 17, 6, 7, 5, 9, 2, 6]

delta=1:  lane i += shfl_down(val, 1)
          lane 0: 14 + 17 = 31   ← Total! (3+1+4+1+5+9+2+6 = 31 ✓)
Values:   [31, 17, 6, 7, 5, 9, 2, 6]

Lane 0 holds the exact sum. Only 3 shuffle instructions needed.
```

---

## 6. Reduction for Max and Min

The same pattern generalizes to any associative binary operator:

```cpp
// Warp max reduction:
__device__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val,  1));
    return val;
}

// Online Softmax needs BOTH max and sum in one pass:
// (Reduces memory bandwidth by avoiding a separate max-finding pass)
struct MaxSum { float max_val; float sum_exp; };

__device__ MaxSum warp_reduce_max_sum(MaxSum v) {
    // Combine two warps' max+sum statistics using the log-sum-exp merge formula
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, v.max_val,  offset);
        float other_sum = __shfl_down_sync(0xffffffff, v.sum_exp, offset);
        // Merge: adjust the smaller max's sum by exp(smaller_max - larger_max)
        if (other_max > v.max_val) {
            v.sum_exp = v.sum_exp * expf(v.max_val - other_max) + other_sum;
            v.max_val = other_max;
        } else {
            v.sum_exp += other_sum * expf(other_max - v.max_val);
        }
    }
    return v;
}
```

---

## 7. When to Use Each Reduction Strategy

| Scenario | Strategy | Why |
| :--- | :--- | :--- |
| Warp-level (≤32 threads) | Shuffle reduction only | Zero smem, 5 cycles |
| Block-level (≤1024 threads) | Shuffle + smem scratchpad | 1 smem round-trip |
| Grid-level, small N (≤32M) | Block reduction + 1 global atomic per block | Minimal atomic contention |
| Grid-level, large N | Multi-level kernel (block reduces to temp buffer, second kernel reduces buffer) | Eliminates all atomic contention |
| Streaming/online reduction | Two-pass online algorithm (e.g. Online Softmax) | Avoids multi-pass memory reads |

---

## 8. Numerical Stability: Kahan Summation and Log-Sum-Exp

### Kahan Compensated Summation (High Precision)

```cpp
// When summing millions of floats, sequential accumulation loses precision.
// Kahan summation tracks the error term and compensates:
__device__ float kahan_block_sum(float val, float* smem) {
    float sum = 0.0f;
    float compensation = 0.0f;  // Tracks accumulated rounding error

    // Each thread maintains its own compensated partial sum:
    float y = val - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;

    // Then block-reduce the compensated sums:
    return block_reduce_sum(sum);
}
```

### Numerically Stable Softmax (Log-Sum-Exp Trick)

Standard softmax `exp(x_i) / sum(exp(x_j))` overflows for large logits (e.g. x=1000 causes `exp(1000)` = infinity). The stable version subtracts the max first:

```cpp
// Numerically stable softmax — max subtraction prevents overflow:
// exp(x_i - max) / sum(exp(x_j - max)) = exp(x_i) / sum(exp(x_j))  (mathematically identical)

__global__ void stable_softmax_kernel(float* logits, float* probs, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Pass 1: Find max over the block (prevents exp() overflow)
    float local_max = (gid < N) ? logits[gid] : -INFINITY;
    float global_max = block_reduce_max(local_max);  // Uses warp_reduce_max + smem

    // Pass 2: Compute exp(x - max) and sum
    float exp_val = (gid < N) ? expf(logits[gid] - global_max) : 0.0f;
    float sum_exp = block_reduce_sum(exp_val);

    // Pass 3: Normalize
    if (gid < N) probs[gid] = exp_val / sum_exp;
}
```

---

## 9. Summary

| Reduction Component | Tool | Latency | Memory Traffic |
| :--- | :--- | :--- | :--- |
| 32-thread warp sum | 5x `__shfl_down_sync` | 5 cycles | 0 bytes |
| 256-thread block sum | Warp reduce + 8-float smem write/read | ~30 cycles | 32 bytes |
| Grid sum (128 blocks) | Block reduce + 128 `atomicAdd` | ~400 cycles (parallel) | 4 bytes final |
| Dual stat (mean+var) | float2 warp reduce | 10 cycles | 0 bytes |
| Online softmax | MaxSum struct reduce | 10 cycles | 0 bytes extra |
