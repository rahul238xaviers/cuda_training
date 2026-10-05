# Chapter 5.2: Atomic Operations, Contention & Lock-Free GPU Patterns

Atomic operations are the backbone of parallel coordination in CUDA. They allow thousands of threads to safely write to shared locations — but they come with severe performance penalties when misused. Understanding how atomics work at the hardware level, how to measure and minimize contention, and when to replace them with hierarchical reductions is essential to writing production-grade GPU code.

---

## 1. What Is an Atomic Operation?

An **atomic** operation executes an indivisible Read-Modify-Write (RMW) sequence on a memory address:

```text
Atomic lifecycle:
  1. LOCK: Hardware acquires exclusive ownership of the target cache line
  2. READ: Loads the current value
  3. MODIFY: Applies the operation (add, max, swap, etc.)
  4. WRITE: Stores the new value
  5. UNLOCK: Releases ownership, signals waiting threads

No other thread can observe an intermediate state.
```

On modern NVIDIA GPUs (Pascal sm_60+), atomics are executed natively at the **L2 cache controller** or the **shared memory controller** — not in slow global DRAM. This makes L2-hitting atomics dramatically faster than on older architectures.

---

## 2. The Full Atomic API

### Integer Atomics

```cpp
// Arithmetic
int old = atomicAdd(int* addr, int val);     // *addr += val; returns old value
int old = atomicSub(int* addr, int val);     // *addr -= val
int old = atomicMin(int* addr, int val);     // *addr = min(*addr, val)
int old = atomicMax(int* addr, int val);     // *addr = max(*addr, val)

// Bitwise
int old = atomicAnd(int* addr, int val);     // *addr &= val
int old = atomicOr (int* addr, int val);     // *addr |= val
int old = atomicXor(int* addr, int val);     // *addr ^= val

// Swap
int old = atomicExch(int* addr, int val);    // *addr = val; returns old

// Compare-And-Swap (the foundation of all lock-free algorithms)
int old = atomicCAS(int* addr, int expected, int desired);
// If *addr == expected: *addr = desired, return expected
// Otherwise:            *addr unchanged,  return current *addr
```

### Float Atomics

```cpp
// Native since Pascal (sm_60):
float old = atomicAdd(float* addr, float val);  // Works directly on sm_60+

// For atomicMin/Max on floats (no native hardware support):
// Use atomicCAS with integer reinterpretation:
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float current = __int_as_float(old);
        if (current >= val) break;  // Already max, no update needed
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);  // Retry until our CAS succeeds
    return __int_as_float(old);
}
```

---

## 3. The Atomic Bottleneck: Contention at Scale

```cpp
// CATASTROPHIC PATTERN: 65,536 threads all hammer the same 4-byte address
__global__ void loss_sum_naive(const float* loss_per_element, float* total_loss, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N)
        atomicAdd(total_loss, loss_per_element[gid]);  // ALL threads serialized here!
}
```

### What Happens at the L2 Controller

The L2 cache controller has a limited-size **atomic replay queue**. When thousands of atomics arrive simultaneously:

```text
L2 Controller (for address 0x40000000):
  Queue: [thread 0] [thread 1] [thread 2] ... [thread 65535]

  Process thread 0: read 0.0, add 1.5 → write 1.5
  Process thread 1: read 1.5, add 2.3 → write 3.8
  Process thread 2: read 3.8, add 0.7 → write 4.5
  ...
  (65,536 sequential operations. Parallelism = 0.)
```

Every thread after thread 0 **blocks** (its warp stalls) waiting for the L2 to finish the preceding atomic. This eliminates all GPU parallelism.

### The 32x Contention Reduction via Warp-Level Aggregation

```cpp
// CORRECT: Aggregate within the warp first (reduces atomics by 32x):
__global__ void loss_sum_warp_reduced(const float* loss, float* total, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < N) ? loss[gid] : 0.0f;

    // Reduce 32 threads → 1 value (5 register instructions, no memory):
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    // Only lane 0 issues the atomic (32x fewer atomics than the naive version):
    if ((threadIdx.x & 31) == 0)
        atomicAdd(total, val);
}
```

### The 1024x Reduction via Block-Level Aggregation

```cpp
// OPTIMAL: Full block reduction (256 or 1024x fewer global atomics):
__global__ void loss_sum_block_reduced(const float* loss, float* total, int N) {
    __shared__ float smem[32];  // One entry per warp
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n_warps = (blockDim.x + 31) >> 5;

    // Grid-stride accumulation:
    float local = 0.0f;
    for (int i = gid; i < N; i += gridDim.x * blockDim.x)
        local += loss[i];

    // Warp reduction:
    local += __shfl_down_sync(0xffffffff, local, 16);
    local += __shfl_down_sync(0xffffffff, local,  8);
    local += __shfl_down_sync(0xffffffff, local,  4);
    local += __shfl_down_sync(0xffffffff, local,  2);
    local += __shfl_down_sync(0xffffffff, local,  1);
    if (lane == 0) smem[warp_id] = local;
    __syncthreads();

    // Final warp reduction:
    local = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        local += __shfl_down_sync(0xffffffff, local, 16);
        local += __shfl_down_sync(0xffffffff, local,  8);
        local += __shfl_down_sync(0xffffffff, local,  4);
        local += __shfl_down_sync(0xffffffff, local,  2);
        local += __shfl_down_sync(0xffffffff, local,  1);
    }

    // Only ONE atomic per block (256x reduction from naive):
    if (threadIdx.x == 0)
        atomicAdd(total, local);
}
```

---

## 4. Shared Memory Atomics: Blazing-Fast Intra-Block Coordination

When multiple threads in the same block need to coordinate writes to a shared table (like a histogram), use **shared memory atomics** which go through the shared memory controller — not L2 or HBM — providing ~23-cycle latency instead of ~400-cycle latency:

```cpp
__global__ void histogram_kernel(const int* input, int* global_hist, int N, int num_bins) {
    // Shared histogram accumulator (per-block private copy):
    extern __shared__ int local_hist[];

    // Initialize shared histogram to zero:
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    // Accumulate into fast shared memory atomics:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        int bin = input[i];
        atomicAdd(&local_hist[bin], 1);  // Shared memory atomic! ~23-cycle latency
    }
    __syncthreads();

    // Merge per-block histograms into global (one atomic per bin per block):
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        atomicAdd(&global_hist[i], local_hist[i]);  // Reduced global atomics: num_bins per block
}
// Launch: shared_bytes = num_bins * sizeof(int)
```

For a 256-bin histogram and 128 blocks: only 128 × 256 = 32,768 global atomics instead of N (millions). Each global atomic is now a different bin, so there is almost no contention!

---

## 5. Compare-And-Swap (CAS): The Foundation of Lock-Free Structures

`atomicCAS` is the primitive from which all complex atomic patterns are built. It is the hardware equivalent of the compare_exchange operation in C++ memory model.

### Pattern 1: Custom Float atomicMin

```cpp
__device__ float atomicMinFloat(float* addr, float val) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int assumed, old = *addr_i;
    do {
        assumed = old;
        float old_f = __int_as_float(old);
        if (old_f <= val) return old_f;  // Already smaller, done
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);  // CAS failed → retry with fresh value
    return __int_as_float(old);
}
```

### Pattern 2: Lock-Free Work Counter

```cpp
// Multiple blocks compete to claim the next chunk of work:
__global__ void dynamic_partition_kernel(float* data, int* work_counter, int total_work) {
    while (true) {
        // Atomically grab the next unclaimed chunk:
        int chunk_id = atomicAdd(work_counter, 1);
        if (chunk_id >= total_work) break;   // All chunks claimed

        // Process this block's chunk:
        int start = chunk_id * CHUNK_SIZE;
        process_chunk(data, start, CHUNK_SIZE);
    }
}
```

### Pattern 3: Spin-Lock (AVOID in Kernel Code — For Understanding Only)

```cpp
// Spin-lock pattern (educational — avoid in GPU kernels due to warp deadlock risk):
__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {}  // Spin until we swap 0 → 1
}
__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);  // Release: write 0
}
```

**Warning**: Spin-locks in GPU kernels can deadlock if a holding warp is de-scheduled while other warps in the same block spin indefinitely waiting for the lock. Use lock-free patterns with CAS retry loops instead.

---

## 6. Atomics in Shared vs. L2 vs. Global Memory: Performance Table

| Location | Latency | Bandwidth | Use Case |
| :--- | :--- | :--- | :--- |
| Shared Memory | ~23 cycles | ~30 TB/s per SM | Intra-block histograms, counters, partial sums |
| L2 Cache | ~100-200 cycles | ~3.35 TB/s | Final global accumulation (few unique addresses) |
| HBM Global | ~400+ cycles | 3.35 TB/s | Fallback when L2 evicted; avoid if possible |

**Rule**: Minimize unique atomic addresses that collide across warps. The fewer threads competing for the same address, the better. Hierarchical reduction (warp → block → 1 global atomic) achieves this.

---

## 7. Summary: Atomic Best Practices

| Pattern | Speedup vs Naive | When To Use |
| :--- | :--- | :--- |
| Raw global atomic | 1x (baseline) | Almost never at scale |
| Warp-level aggregation first | 32x | Minimum acceptable |
| Block-level aggregation first | 256x-1024x | Standard production pattern |
| Shared memory atomics | 10x-50x latency improvement | Intra-block coordination (histograms) |
| CAS loop | Enables custom operations | Custom float min/max, lock-free counters |
| Persistent kernel with atomic work queue | Eliminates launch overhead | Streaming workloads |
