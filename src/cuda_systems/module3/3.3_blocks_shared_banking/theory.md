# Chapter 3.3: Thread Blocks, Shared Memory Architecture & Bank Conflict Analysis

Shared memory is the most powerful tool in the CUDA programmer's arsenal. It sits on the SM itself — same silicon, same clock domain — and delivers roughly 15x lower latency than HBM. The difference between a naive kernel and a production kernel often reduces to one question: are you loading the hot data into shared memory and reusing it?

Understanding the bank structure is equally critical. A single poorly-designed shared memory access pattern can reduce shared memory throughput from 30 TB/s to under 1 TB/s — fully eliminating the advantage of having on-chip memory at all.

---

## 1. The Block's Place in the Execution Hierarchy

When you launch a kernel, the CUDA runtime divides your grid into blocks and schedules each block on an SM. The block is the unit of scheduling, and more importantly, it is the **unit of shared memory allocation**.

```text
Grid (entire launch)
 ├── Block (0,0)  → SM 0  [gets its own shared memory pool]
 ├── Block (1,0)  → SM 1  [gets its own shared memory pool]
 ├── Block (2,0)  → SM 0  [queued, waits for Block (0,0) to finish or SM to have room]
 ├── Block (3,0)  → SM 2
 └── ...

Within Block (0,0) on SM 0:
  ├── Warp 0: threads [0 .. 31]   — assigned to scheduler 0
  ├── Warp 1: threads [32 .. 63]  — assigned to scheduler 1
  ├── Warp 2: threads [64 .. 95]  — assigned to scheduler 2
  └── ...
  └── Shared Memory: 48 KB of on-chip SRAM (private to this block)
```

The shared memory allocated to a block lives for the block's lifetime. When the block finishes, its shared memory is reclaimed and assigned to the next waiting block.

---

## 2. The Shared Memory Physical Design

Shared memory is implemented as a bank of 32 parallel SRAM arrays (banks), each independently addressable. This parallel design is what gives shared memory its high throughput.

```text
Shared Memory Bank Layout (32 banks × 4 bytes each):

Word Index:    0    1    2    3    4    ...   31   32   33   ...
Bank:          0    1    2    3    4    ...   31    0    1   ...
Address:       0    4    8   12   16   ...  124  128  132   ...
```

**Bank assignment formula:**
```
bank_id = (byte_address / 4) % 32
       = (word_index) % 32
```

Each bank can service exactly one request per clock cycle. If no two threads in a warp access the same bank, all 32 requests are serviced simultaneously in 1 cycle (32 banks × 1 request each).

---

## 3. The Three Access Patterns

### Pattern 1: Conflict-Free (Sequential) Access — Peak Throughput

Thread `i` accesses word `i`. All 32 threads map to 32 different banks.

```cpp
__shared__ float s_data[32];
float val = s_data[threadIdx.x % 32];  // Thread i → Bank i
```

```text
Thread 0 → Word 0 → Bank 0  ✓
Thread 1 → Word 1 → Bank 1  ✓
Thread 2 → Word 2 → Bank 2  ✓
...
Thread 31 → Word 31 → Bank 31  ✓
All 32 serviced in 1 cycle. Full throughput.
```

### Pattern 2: Broadcast — Also Peak Throughput

All threads access the exact same word (same address). The SRAM controller recognizes this as a broadcast: reads the value once and delivers it to all 32 threads in 1 cycle.

```cpp
float val = s_data[0];  // All 32 threads read address 0 → 1 cycle broadcast
```

This is what makes warp-uniform reads of constants (like RMSNorm's scale `gamma`) from shared memory free.

### Pattern 3: Bank Conflict — Throughput Killer

Multiple threads access different addresses that map to the same bank.

```cpp
__shared__ float s_data[64];
// Thread i reads element i * 2 (stride-2 access):
float val = s_data[threadIdx.x * 2];

// Bank mapping:
// Thread 0: word 0  → Bank 0
// Thread 1: word 2  → Bank 2
// Thread 2: word 4  → Bank 4
// ...
// Thread 16: word 32 → Bank 0  ← CONFLICT with Thread 0!
// Thread 17: word 34 → Bank 2  ← CONFLICT with Thread 1!
// → 2-way conflict: 2 cycles instead of 1
```

For stride-32 access:
```cpp
float val = s_data[threadIdx.x * 32];
// All 32 threads access Bank 0 → 32-way conflict → 32 cycles → same as sequential!
```

---

## 4. The Matrix Transpose Bank Conflict Analysis

Matrix transpose is the canonical bank conflict example. It appears constantly in attention kernels and GEMM tiling.

```cpp
// Naive transpose — introduces 32-way bank conflicts during shared memory read:
__global__ void transpose_naive(const float* in, float* out, int N) {
    __shared__ float tile[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    // LOAD from global: coalesced (threadIdx.x varies → consecutive addresses → good)
    tile[threadIdx.y][threadIdx.x] = in[row * N + col];
    __syncthreads();

    // STORE to global (transposed): coalesced (threadIdx.x drives output column → good)
    // READ from shared: threadIdx.x drives ROW → stride 32 → 32-way bank conflict!
    out[col * N + row] = tile[threadIdx.x][threadIdx.y];
    //                          ^^^^^^^^^^^^ drives row index: all threads in row threadIdx.y
    //                          access the same column → stride 32 within bank → CONFLICT
}
```

The read `tile[threadIdx.x][threadIdx.y]` is the problem:
- Thread 0: reads `tile[0][0]` → Word 0  → Bank 0
- Thread 1: reads `tile[1][0]` → Word 32 → Bank 0 (32 % 32 = 0)
- Thread 2: reads `tile[2][0]` → Word 64 → Bank 0 (64 % 32 = 0)
- All 32 threads hit Bank 0 → 32-way serialization.

### The Padding Fix

```cpp
// Add 1 dummy column to shift rows to different banks:
__shared__ float tile[32][33];  // ← key change: 33 instead of 32

// Now tile[r][c] is at word offset: r * 33 + c
// Thread 0: tile[0][0] → Word 0  → Bank 0
// Thread 1: tile[1][0] → Word 33 → Bank 1  (33 % 32 = 1) ← shifted by 1!
// Thread 2: tile[2][0] → Word 66 → Bank 2  (66 % 32 = 2)
// All 32 threads hit different banks → 0 conflicts!

// Storage cost: 32 * 33 * 4 = 4224 bytes instead of 32 * 32 * 4 = 4096 bytes.
// Extra: 128 bytes (3.1% overhead) to eliminate 32x throughput collapse.
```

---

## 5. Dynamic Shared Memory

When the shared memory size depends on a runtime parameter (like block size or tile size chosen by the caller), use dynamic shared memory:

```cpp
// Kernel declaration: extern means "size determined at launch time"
__global__ void layernorm_kernel(const float* x, float* out, int D) {
    extern __shared__ float smem[];  // Size declared at launch

    float* s_x   = smem;              // First D floats for input
    float* s_var = smem + D;          // Next 32 floats for warp-level variances
    float* s_mu  = smem + D + 32;     // Next 32 floats for warp-level means

    // ... kernel body ...
}

// Host-side launch with 3 sections in shared memory:
size_t shared_bytes = (D + 32 + 32) * sizeof(float);
layernorm_kernel<<<grid, block, shared_bytes>>>(d_x, d_out, D);
```

Dynamic shared memory is essential when you want to write a single kernel that handles different problem sizes without recompiling, and when the shared memory layout depends on hyperparameters.

---

## 6. Shared Memory Configuration: Prefer Larger Shared Memory

On Ampere and Hopper, the L1 cache and shared memory share the same physical SRAM pool. You can configure how this pool is split:

```cpp
// Set maximum shared memory per block to 98 KB (leaving minimum for L1):
cudaFuncSetAttribute(my_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     98 * 1024);

// Or: prefer larger shared memory (less L1 cache):
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

// Hopper H100 allows up to 228 KB of shared memory per SM when configured:
// Use cudaFuncSetAttribute with cudaFuncAttributeMaxDynamicSharedMemorySize
// set to 227 * 1024 for nearly the full SRAM pool.
```

### When to Use More Shared Memory

- **GEMM / Attention kernels**: Tile A and B matrices for dot product accumulation
- **1D/2D reduction kernels**: Store partial sums per warp
- **Convolution kernels**: Tile input activation patches
- **LayerNorm / RMSNorm**: Cache the input row for two-pass normalization

---

## 7. The Shared Memory Tiling Pattern for LayerNorm

LayerNorm requires two passes over the input: first to compute mean/variance, then to normalize. Shared memory allows caching the input for the second pass without re-reading from HBM:

```cpp
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int D)   // D = embedding dimension (row width)
{
    extern __shared__ float s_x[];   // Cache the entire row [D floats]

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float eps = 1e-5f;

    // --- Pass 1: Load row into shared memory ---
    for (int i = tid; i < D; i += blockDim.x) {
        s_x[i] = x[row * D + i];
    }
    __syncthreads();

    // --- Compute mean (each thread accumulates a partial sum) ---
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += s_x[i];
    }
    local_sum = block_reduce_sum(local_sum);  // Warp+smem reduction

    float mean = (threadIdx.x == 0) ? local_sum / D : 0.0f;
    mean = __shfl_sync(0xffffffff, mean, 0);  // Broadcast mean to all threads

    // --- Compute variance ---
    float local_var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = s_x[i] - mean;
        local_var += diff * diff;
    }
    local_var = block_reduce_sum(local_var);
    float rstd = (threadIdx.x == 0) ? rsqrtf(local_var / D + eps) : 0.0f;
    rstd = __shfl_sync(0xffffffff, rstd, 0);

    // --- Pass 2: Normalize from shared memory (no HBM access!) ---
    for (int i = tid; i < D; i += blockDim.x) {
        out[row * D + i] = (s_x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}
// Launch: one block per row, shared_bytes = D * sizeof(float)
// layernorm_kernel<<<batch_size, 256, D * sizeof(float)>>>(x, gamma, beta, out, D);
```

Without shared memory, `s_x[i]` in pass 2 would re-read from HBM — doubling memory bandwidth. With shared memory, the row data is reused from on-chip SRAM with ~23-cycle latency.

---

## 8. `__syncthreads()` Deadlock: The Most Common Bug

`__syncthreads()` inside a conditional branch is one of the most common bugs in CUDA kernels:

```cpp
// WRONG: __syncthreads() inside a conditional → deadlock!
if (threadIdx.x < D) {
    smem[threadIdx.x] = x[idx];
    __syncthreads();   // Some threads reach here, some never do → deadlock!
}

// CORRECT: Load conditionally, but sync unconditionally
smem[threadIdx.x] = (threadIdx.x < D) ? x[idx] : 0.0f;
__syncthreads();   // ALL threads in block reach this point
```

**Rule**: `__syncthreads()` must be reached by **every thread** in the block on every execution path, unconditionally.

---

## 9. Summary: Block & Shared Memory Mental Model

| Concept | Physical Fact |
| :--- | :--- |
| Block assigned to SM | Exclusively — block never moves to another SM |
| Shared memory scope | Block-private (not visible to other blocks or SMs) |
| Shared memory size | Up to 228 KB per SM (Hopper), configurable |
| Shared memory banks | 32 banks × 4 bytes = 128 bytes per bank row |
| Conflict-free access | Each thread in warp accesses a different bank |
| Broadcast | All threads same address → 1 cycle, no conflict |
| Bank conflict cost | K-way conflict → K serialized cycles (1/K throughput) |
| Padding trick | Add 1 word per row to rotate bank assignments |
| `__syncthreads()` | Must be reached by ALL threads unconditionally |
