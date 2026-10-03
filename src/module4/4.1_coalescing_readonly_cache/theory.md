# Chapter 4.1: Memory Coalescing, Read-Only Cache & GPU Memory Hierarchy

Memory access is the number one bottleneck in virtually every GPU kernel. The difference between a coalesced and an uncoalesced kernel on the same hardware can be 10x to 32x in execution time. This chapter builds the complete mental model from the physical memory bus up to the application-level patterns that saturate or destroy bandwidth.

---

## 1. The GPU Memory Hierarchy: Speed vs. Capacity

Every byte your kernel reads travels through a layered hierarchy. Understanding the latency and bandwidth at each level is the foundation of memory optimization.

```text
GPU Memory Hierarchy (Hopper H100):

  +----------------------------------------------------------+  Latency    | Bandwidth
  | Registers (256 KB per SM, private per thread)            |  0 cycles   | Unlimited
  +----------------------------------------------------------+
  | Shared Memory / L1 Cache (up to 228 KB per SM, shared)  |  ~23 cycles | ~30 TB/s per SM
  +----------------------------------------------------------+
  | L2 Cache (50 MB, shared across all SMs)                  |  ~180 cycles| ~3.35 TB/s
  +----------------------------------------------------------+
  | HBM3 Global Memory (80 GB)                               |  ~400 cycles| 3.35 TB/s
  +----------------------------------------------------------+
  | PCIe / NVLink to CPU DRAM                                |  ~10,000 ns | ~64 GB/s
  +----------------------------------------------------------+
```

The critical insight: the memory bandwidth ratio from registers to HBM is roughly 1000:1. An instruction that hits registers completes in 1 cycle; an instruction that misses to HBM stalls for 400 cycles. This is why memory access patterns determine performance, not raw FLOPs.

---

## 2. The 128-Byte Transaction: Hardware's Atomic Unit

The HBM memory controller does not fetch individual bytes or individual floats. It always fetches memory in **128-byte aligned chunks** called transactions.

When a warp of 32 threads issues a global memory load, the memory controller:
1. Collects all 32 requested addresses from the warp.
2. Groups them into 128-byte aligned sectors.
3. Issues one transaction per distinct sector.

```text
Warp Load Analysis:

Thread 0 reads address: 0x1000  (byte 0 of a 128-byte sector)
Thread 1 reads address: 0x1004  (byte 4 of the same sector)
Thread 2 reads address: 0x1008
...
Thread 31 reads address: 0x107C  (byte 124 of the same sector)

All 32 addresses fall within ONE 128-byte sector (0x1000 to 0x107F).
Result: 1 transaction → 128 bytes fetched → 128 bytes used.
Bus Efficiency: 128/128 = 100%
```

This is **coalesced access** — the ideal case.

---

## 3. Coalesced vs. Uncoalesced: The 32x Performance Gap

### Coalesced Access (Ideal)

Each thread in a warp reads a consecutive 4-byte float. Thread `i` reads `input[gid + i]`:

```text
Thread 0: reads input[0]   → address 0x1000
Thread 1: reads input[1]   → address 0x1004
Thread 2: reads input[2]   → address 0x1008
...
Thread 31: reads input[31] → address 0x107C

All addresses within 0x1000-0x107F.
→ 1 memory transaction (128 bytes)
→ 32 useful floats in 1 transaction
→ Bus efficiency: 100%
```

### Strided Access (Catastrophic)

Each thread reads with stride 32: thread `i` reads `input[i * 32]`:

```text
Thread 0:  reads input[0]  → address 0x1000  (sector 0x1000-0x107F)
Thread 1:  reads input[32] → address 0x1080  (sector 0x1080-0x10FF)
Thread 2:  reads input[64] → address 0x1100  (sector 0x1100-0x117F)
...
Thread 31: reads input[992] → address 0x1F80 (sector 0x1F80-0x1FFF)

32 different sectors → 32 memory transactions
→ 32 * 128 = 4,096 bytes fetched
→ Only 32 * 4 = 128 bytes actually used
→ Bus efficiency: 128/4096 = 3.1%
```

The same number of threads reads the same amount of data, but the strided version uses 32x more memory bandwidth and issues 32x more transactions — each stalling the warp scheduler.

```cpp
// Kernel that demonstrates both patterns:
__global__ void coalesced_vs_strided(const float* input, float* output_coalesced,
                                      float* output_strided, int N, int stride) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Coalesced: thread i reads element i (consecutive)
    if (gid < N)
        output_coalesced[gid] = input[gid];   // Adjacent threads → adjacent addresses

    // Strided: thread i reads element i * stride (widely scattered)
    if (gid * stride < N)
        output_strided[gid] = input[gid * stride];  // Adjacent threads → scattered addresses
}
```

### How to Detect Coalescing Issues

Using Nsight Compute profiler:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
               l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./my_kernel
```

- `sectors / requests` ratio close to **4** means good coalescing (128 bytes / 32 bytes = 4 sectors per request is ideal)
- A ratio of **32** means every thread triggered its own sector — completely uncoalesced

---

## 4. The Read-Only Data Cache: `__ldg` and `const __restrict__`

Modern NVIDIA GPUs (Kepler and later) include a dedicated **read-only data cache** that is separate from the L1/shared memory pool. This cache is:
- **Non-coherent**: It assumes data it caches will not be modified by the current kernel. It is safe for read-only inputs.
- **Higher effective bandwidth**: It has its own tags and does not evict shared memory data.

### Enabling the Read-Only Cache

```cpp
// Method 1: Mark the pointer as const and __restrict__
// The compiler automatically emits LDG (Load Global via read-only cache)
__global__ void kernel(const float* __restrict__ weights, float* out, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) out[gid] = weights[gid] * 2.0f;  // weights accessed via LDG automatically
}

// Method 2: Explicit __ldg intrinsic (works regardless of pointer annotations)
__global__ void kernel_explicit(const float* weights, float* out, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        float w = __ldg(&weights[gid]);   // Forces LDG instruction
        out[gid] = w * 2.0f;
    }
}
```

### When to Use the Read-Only Cache

| Data Type | Use Read-Only Cache? | Reason |
| :--- | :--- | :--- |
| Model weights (inference) | Yes | Never modified during forward pass |
| Embedding lookup table | Yes | Read-only during forward pass |
| Activation inputs | Yes (for read pointer) | Written by previous layer, read-only in current |
| Attention key/value matrices | Yes | Read multiple times by different query positions |
| Gradient buffers | No | Written by backward pass, must be coherent |
| Atomic targets | No | Require coherent write-back cache |

---

## 5. Constant Memory: Hardware Broadcast Across the Warp

NVIDIA GPUs provide 64 KB of special on-chip **Constant Memory**. It sits in a dedicated cache with a unique property: when all 32 threads in a warp read the **same address**, the constant cache services all 32 reads in **a single clock cycle via hardware broadcast**.

```cpp
// Declare constant memory (globally, outside any function):
__constant__ float c_weights[1024];        // 1024 floats = 4 KB
__constant__ float c_gamma_beta[512];      // For LayerNorm scale and bias

// Copy to constant memory before kernel launch (host-side):
cudaMemcpyToSymbol(c_weights, host_weights, sizeof(host_weights));

// In kernel: all threads read the same address → 1 cycle broadcast:
__global__ void attention_scale_kernel(float* qk, int N, int head_dim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        // c_weights[0] is the same address for all 32 threads → broadcast!
        float scale = c_weights[0];   // Served in 1 cycle for entire warp
        qk[gid] *= scale;
    }
}
```

### Constant Memory vs. Read-Only Cache

| Feature | Constant Memory | Read-Only Cache |
| :--- | :--- | :--- |
| Capacity | 64 KB | Part of L1 (varies) |
| Update mechanism | cudaMemcpyToSymbol | Normal global memory writes |
| Access pattern | Uniform (same address per warp) = 1 cycle | Any pattern, cached reads |
| Best for | Broadcast scalars, small lookup tables | Large read-only weight tensors |
| Strided access | Serialized (slow) | Cached but not broadcast |

---

## 6. Shared Memory: The Programmable L1 Cache

Shared memory (`__shared__`) is the on-chip SRAM that lives inside the SM, shared by all threads in a block. It provides ~23-cycle access latency — roughly 15x faster than HBM.

The key pattern: **load from slow global memory once, store in fast shared memory, compute from shared memory multiple times**.

```cpp
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    const int TILE = 32;

    // Shared memory tiles for A and B blocks
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K / TILE; ++t) {
        // Phase 1: Cooperative loading from HBM into shared SRAM
        s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();  // Wait for all threads to finish loading

        // Phase 2: Compute from fast SRAM (no HBM traffic during this phase)
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();  // Wait before overwriting shared tiles
    }

    C[row * N + col] = acc;
}
```

### Why This Dramatically Outperforms Naive Matmul

Without tiling: Each output element `C[row][col]` requires reading an entire row of `A` and column of `B` from HBM. If K = 4096, that's 2 * 4096 * 4 = 32 KB of HBM reads per output element.

With TILE=32 tiling: The 32x32 tile of A (4 KB) and B (4 KB) are loaded once into shared memory. All 32*32 = 1024 output elements in the tile reuse these tiles. HBM reads reduce by a factor of TILE = 32x.

```formula
Without tiling: HBM reads per element = 2 * K * sizeof(float)
With TILE tiling: HBM reads per element = 2 * K * sizeof(float) / TILE
Speedup factor = TILE (theoretical)
```

---

## 7. Shared Memory Bank Conflicts: The Hidden Throughput Killer

Shared memory is organized into 32 banks, each 4 bytes wide. Bank index of an address: `bank = (addr / 4) % 32`.

When multiple threads in a warp access different addresses that map to the **same bank**, those accesses are serialized. This is a **bank conflict**.

```text
No conflict (ideal):
Thread 0 → s_data[0]  → Bank 0    ✓
Thread 1 → s_data[1]  → Bank 1    ✓
Thread 2 → s_data[2]  → Bank 2    ✓
...
Thread 31 → s_data[31] → Bank 31  ✓
All 32 reads served simultaneously in 1 cycle.

2-way bank conflict:
Thread 0  → s_data[0]  → Bank 0  |
Thread 16 → s_data[16] → Bank 0  |  Both on Bank 0 → 2 serial cycles!
Thread 1  → s_data[1]  → Bank 1  ✓
...
Half-throughput!

32-way bank conflict (catastrophic):
Thread i → s_data[i * 32]  All map to Bank 0 → 32 serial cycles!
→ Same as sequential execution, zero parallelism.
```

### Bank Conflict in Matrix Transpose

```cpp
__global__ void transpose_with_conflict(const float* in, float* out, int N) {
    __shared__ float tile[32][32];

    // Load: coalesced (row-major threads → row-major shared memory)
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in[row * N + col];  // No conflict
    __syncthreads();

    // Store transposed: coalesced global write but bank conflict in shared memory!
    // threadIdx.x drives the row index → stride 32 within bank → 32-way conflict!
    out[col * N + row] = tile[threadIdx.x][threadIdx.y];  // ← CONFLICT!
}
```

### Bank Conflict Fix: Padding

```cpp
// Add 1 element of padding per row to shift each row to a different bank:
__shared__ float tile[32][33];  // ← +1 padding column

// Now tile[0][k] and tile[1][k] are in different banks:
// tile[0][k]: bank = k % 32
// tile[1][k]: bank = (33 + k) % 32 = (k+1) % 32  ← offset by 1!
// No conflicts.
```

---

## 8. The Memory Access Checklist for Production Kernels

Before shipping any memory-bound kernel, verify all of the following:

```text
✅ Coalescing:
   - Consecutive thread IDs access consecutive global memory addresses?
   - No transposition happening in global memory access?

✅ Vectorized Loads:
   - Using float4 / __nv_bfloat162 / int4 for 128-bit transactions?
   - Base pointer is 16-byte aligned?

✅ Read-Only Hint:
   - All read-only pointers annotated with const __restrict__?
   - Or __ldg() used explicitly?

✅ Shared Memory:
   - Hot data loaded once into shared, computed from shared multiple times?
   - __syncthreads() placed correctly (after loads, before reads)?

✅ Bank Conflicts:
   - No stride-32 access patterns in shared memory?
   - Padding applied where needed?

✅ Constant Memory:
   - Broadcast scalars (scale, epsilon, alpha) stored in __constant__?
   - All threads in warp reading the same constant address?

✅ Profiler Verification:
   - ncu shows sectors/request ratio near 4?
   - lmem = 0 in ptxas output?
```

---

## 9. Applied Pattern: Optimized Memory Coalescing in an Embedding Lookup

LLM embedding lookup is a classic memory-bandwidth-bound kernel. Each token ID indexes into a large weight table and retrieves a fixed-size embedding vector.

```cpp
// Embedding lookup: token_ids[batch] → weight_table[token_id * embed_dim ... ]
// Each block handles one token. Each thread handles embed_dim / 4 elements (float4).

__launch_bounds__(256)
__global__ void embedding_lookup_kernel(
    const int* __restrict__   token_ids,     // [batch_size]
    const float* __restrict__ weight_table,  // [vocab_size * embed_dim]
    float* __restrict__       output,        // [batch_size * embed_dim]
    int embed_dim)
{
    // One block per token:
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;  // Thread index within block

    int token_id = token_ids[token_idx];
    const float* src_row = weight_table + token_id * embed_dim;
    float*       dst_row = output + token_idx * embed_dim;

    // Each thread processes 4 consecutive floats (128-bit load/store):
    for (int i = tid * 4; i < embed_dim; i += blockDim.x * 4) {
        // Coalesced: threads 0,1,...,63 read elements 0,4,8,...,252 consecutively
        float4 v = *reinterpret_cast<const float4*>(&src_row[i]);
        *reinterpret_cast<float4*>(&dst_row[i]) = v;
    }
}

// Launch: one block per batch element, 256 threads per block
// embedding_lookup_kernel<<<batch_size, 256>>>(d_token_ids, d_weights, d_output, embed_dim);
```

### Why This Is Fast:
1. Block = token → all 256 threads are loading from the same HBM row → hardware prefetcher can identify the sequential stream
2. `float4` → 128-bit loads, 4x fewer instructions
3. `const __restrict__` → compiler routes through read-only cache (LDG)
4. Sequential memory access across consecutive thread IDs → fully coalesced

---

## 10. Summary

| Memory Type | Location | Latency | Bandwidth | Lifetime | Capacity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Registers | On-chip SM | 0 cycles | Unlimited | Per thread | 256 KB per SM |
| Shared Memory | On-chip SM | ~23 cycles | ~30 TB/s per SM | Per block | Up to 228 KB per SM |
| L1 Cache | On-chip SM | ~30 cycles | Combined with smem | Automatic | Combined with smem |
| Read-Only Cache | On-chip | ~30 cycles | High | Per kernel | L2 region |
| Constant Memory | On-chip cache | 1 cycle (broadcast) | Broadcast | Global | 64 KB total |
| L2 Cache | On-chip GPU | ~180 cycles | 3.35 TB/s | Automatic | 50 MB (H100) |
| HBM / Global Memory | Off-chip DRAM | ~400 cycles | 3.35 TB/s | Lifetime of allocation | 80 GB (H100) |
