# Chapter 3.1: CUDA Threads, Registers & The Hardware Execution Model

In CUDA, everything starts with a single thread. Understanding how threads are created, how they map to physical silicon, what state they own, and how the hardware scheduler juggles them is the single most important mental model for writing fast kernels. Without this foundation, you cannot reason about register pressure, occupancy, warp divergence, or instruction-level parallelism.

---

## 1. The Physical Reality: Streaming Multiprocessors (SMs)

Before writing a single line of kernel code, you must understand the physical hardware your threads run on.

An NVIDIA GPU is a collection of **Streaming Multiprocessors (SMs)**. An H100 has 132 SMs. An A100 has 108 SMs. Each SM is an independent execution engine. SMs do not share registers, shared memory, or the L1 cache.

### What Lives Inside One SM:

```text
One Streaming Multiprocessor (SM) — Ampere/Hopper Architecture:
+---------------------------------------------------------------+
| Register File:       65,536 x 32-bit registers (256 KB)      |
| Shared Memory:       Up to 228 KB (configurable)              |
| L1 Cache:            Combined with Shared Memory (up to 256KB)|
| Warp Schedulers:     4 independent hardware schedulers        |
| Dispatch Units:      4 (can issue 4 instructions per cycle)   |
| CUDA Cores (FP32):   128 cores                                |
| Tensor Core Units:   4 (FP16/BF16/TF32 matrix engines)       |
| Load/Store Units:    32                                        |
+---------------------------------------------------------------+
```

The register file is the fastest memory on the chip — zero latency, zero energy cost per access. Every local variable in your kernel competes for a slice of those 65,536 registers. This competition is the root of all occupancy and register pressure analysis.

---

## 2. Thread Launching & The Three-Level Hierarchy

When you launch a kernel, you supply a grid of thread blocks:

```cpp
dim3 gridDim(128, 1, 1);      // 128 blocks in the grid
dim3 blockDim(256, 1, 1);     // 256 threads per block
my_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
```

### Hardware Mapping:
- The **Grid** is a logical concept. It is divided by the CUDA runtime into blocks.
- Each **Block** is dispatched to exactly one SM and stays on that SM until all its threads complete. A block is never split across SMs.
- One SM can run multiple blocks simultaneously if the SM has enough registers and shared memory to satisfy all of them.

```text
Grid (128 blocks total)
  |-- Block 0  --> SM 0  (runs to completion)
  |-- Block 1  --> SM 1  (runs to completion)
  |-- Block 2  --> SM 0  (queued, starts when Block 0 finishes)
  |-- ...
```

### Thread Identification Built-ins:
```cpp
// 1D grid/block:
int gid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid/block (e.g., image processing):
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Global linear index from 2D:
int gid = row * gridDim.x * blockDim.x + col;
```

---

## 3. The Warp: The Atomic Unit of Execution

This is the most critical fact in GPU programming:

**The GPU hardware does not execute individual threads. It executes warps of exactly 32 threads simultaneously.**

When the SM receives a block of 256 threads, it immediately divides them into 256/32 = 8 warps. Threads 0-31 form warp 0, threads 32-63 form warp 1, and so on. From that point, the hardware never schedules individual threads — it always schedules entire warps.

### Why 32?

The 32-thread warp width matches the width of the SIMD execution units inside the SM. All 32 threads in a warp share a single program counter and execute the same instruction simultaneously — one instruction, 32 data elements. This is the SIMT (Single Instruction Multiple Thread) model.

```text
Warp 0: threads [0, 1, 2, ..., 31]   → All execute:  fmaf.f32  reg5, reg1, reg2, reg3
Warp 1: threads [32, 33, ..., 63]    → All execute:  fmaf.f32  reg5, reg1, reg2, reg3
Warp 2: threads [64, 65, ..., 95]    → Waiting for memory load to return
Warp 3: threads [96, 97, ..., 127]   → Executing:    ld.global.f32  reg1, [ptr + offset]
```

### Zero-Overhead Warp Switching

When a warp issues a memory load (which takes 200-400 ns on HBM), the warp scheduler immediately switches to another warp that is ready to execute arithmetic. This **latency hiding** is the GPU's primary mechanism for achieving high throughput. The registers for all active warps are always resident in the register file — no context switch, no register saving/restoring.

The SM can have up to 64 warps active (2,048 threads) simultaneously on Ampere/Hopper. The more warps are available to switch between, the more effectively the SM hides memory latency.

---

## 4. Registers: The Private State of Each Thread

Every thread in a CUDA kernel has its own private register state. When the compiler processes your kernel, it analyzes every local variable and assigns it a register slot.

```cpp
__global__ void my_kernel(const float* input, float* output, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;  // → register r0
    if (gid >= N) return;

    float val = input[gid];    // val → register r1
    float result = val * val + 2.0f * val + 1.0f;  // result → register r2
    output[gid] = result;
}
```

The PTX assembly will look roughly like:
```text
mov.u32     %r0, %ctaid.x;
mad.lo.u32  %r1, %r0, %ntid.x, %tid.x;   // gid = blockIdx.x * blockDim.x + threadIdx.x
ld.global.f32  %f0, [%rd1 + %r1*4];      // val = input[gid]
fma.rn.f32  %f1, %f0, %f0, 2.0;          // result = val * val + 2.0*val
fadd.rn.f32 %f2, %f1, 1.0;
st.global.f32  [%rd2 + %r1*4], %f2;      // output[gid] = result
```

Each of the 32 threads in a warp holds its own copy of `%r0`, `%f0`, `%f1`, `%f2` in its private register slice. The SM's register file is partitioned by the hardware so each thread gets its portion.

### How Many Registers Per Thread?

The SM's 65,536 registers are shared among all active threads on the SM. If each thread in your kernel uses 32 registers, and you want 2,048 concurrent threads (maximum occupancy):

```formula
Max threads on SM        = 2,048
Registers per thread     = 32
Total registers needed   = 2,048 * 32 = 65,536 (uses the whole register file!)

If registers per thread = 64:
Total needed = 2,048 * 64 = 131,072 (exceeds the register file)
→ The hardware limits active warps to: 65,536 / 64 = 1,024 threads (50% occupancy)
```

This is occupancy: the ratio of active warps the SM is running versus the maximum it is physically capable of running. Lower occupancy means fewer warps available to hide memory latency.

---

## 5. Register Pressure & Spilling: The Silent Performance Killer

When the compiler determines your kernel uses more registers per thread than what the hardware allows (max 255 registers per thread), or when occupancy requirements force a lower-than-needed register budget, the compiler **spills** registers to **local memory**.

Local memory is physically located in the same HBM/GDDR as global memory. It has 200-400 ns latency — identical to slow global memory. A spilled register access is as slow as a global memory access.

### Detecting Register Spilling:

```bash
nvcc --ptxas-options=-v -O3 -arch=sm_90 kernel.cu
```

Output you want to see:
```text
ptxas info : Compiling entry function 'my_kernel' for 'sm_90'
ptxas info : Used 32 registers, 0 bytes smem, 0 bytes lmem  ← lmem = 0 is good!
```

Output that indicates trouble:
```text
ptxas info : Used 74 registers, 0 bytes smem, 1024 bytes lmem  ← 1 KB of spilled registers!
```

### Controlling Register Allocation:

```cpp
// Force compiler to cap registers at 32 per thread:
__launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerSM=4
__global__ void my_kernel(float* data, int N) { ... }

// Or use the nvcc flag:
//   nvcc --maxrregcount=32 kernel.cu
```

`__launch_bounds__` tells the compiler the exact block size you will use, allowing it to make more accurate register allocation decisions without sacrificing too many registers.

---

## 6. Warp Divergence: When 32 Threads Cannot Agree

Because a warp executes one instruction at a time for all 32 threads, conditional branches where different threads take different paths are handled by **serialization**:

```cpp
__global__ void divergent_kernel(float* data, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid % 2 == 0) {
        data[gid] = data[gid] * 2.0f;  // Even threads: path A
    } else {
        data[gid] = data[gid] + 1.0f;  // Odd threads: path B
    }
}
```

Since threads 0, 2, 4... and threads 1, 3, 5... alternate between path A and B within the same warp, the hardware executes **both paths sequentially** with some threads disabled via a predicate mask:
- Cycle 1: Execute `data[gid] * 2.0f` — even threads active, odd threads masked off (doing nothing)
- Cycle 2: Execute `data[gid] + 1.0f` — odd threads active, even threads masked off

The warp takes twice as long. **Warp divergence halves throughput when 50% of threads take each branch.**

### Eliminating Divergence: Thread-Coherent Branching

Design kernels so threads within the same warp (consecutive thread IDs) always take the same branch:

```cpp
// Divergent: alternating threads branch differently
if (gid % 2 == 0) { ... }

// Coherent: entire warp takes same branch (first half of threads vs second half)
if (gid < N / 2) { ... }

// Even better: branch at warp granularity
int warp_id = gid / 32;
if (warp_id % 2 == 0) { ... }  // All 32 threads in warp agree!
```

---

## 7. Occupancy: Balancing Warps vs. Resources

Occupancy is the percentage of the SM's maximum possible warp count that is actually active. Higher occupancy gives the warp scheduler more warps to switch between when some warps are stalled on memory.

### The Three Occupancy Limiters:

**Limiter 1 — Registers:**
```formula
Active threads = min(SM max threads, floor(SM registers / registers per thread) * 32)
Occupancy from registers = Active threads / SM max threads
```

**Limiter 2 — Shared Memory:**
```formula
Max blocks from smem = floor(SM shared memory / shared memory per block)
Active threads from smem = Max blocks from smem * threads per block
Occupancy from smem = Active threads from smem / SM max threads
```

**Limiter 3 — Block Size:**
```formula
Max blocks from block size = floor(SM max threads / threads per block)
```

The final occupancy is the minimum across all three limiters.

### Practical Guidance:

```cpp
// Query at runtime what resources your kernel uses:
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, my_kernel, 0, 0);
printf("Optimal block size: %d, Grid size: %d\n", blockSize, minGridSize);

// Check actual occupancy:
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, my_kernel, blockSize, 0);
float occupancy = (maxActiveBlocks * blockSize) / (float)2048;
printf("Theoretical occupancy: %.1f%%\n", occupancy * 100);
```

### The Occupancy Myth

High occupancy does not always equal high performance. A kernel that does very little memory access and lots of arithmetic can be perfectly fast at 25% occupancy — the ALUs are busy every cycle regardless of how many warps are waiting. The goal is to have enough occupancy to hide the **specific** latency bottleneck in your kernel (arithmetic latency vs. memory latency).

---

## 8. The Kernel Launch Pipeline: From CPU to SM

Understanding what happens between `my_kernel<<<G, B>>>()` and the first instruction executing on silicon:

```text
Host CPU                        GPU Hardware
    |                               |
    |-- cudaLaunchKernel() -------->|
    |   (via PCIe or NVLink)        |-- GPC (GPU Processing Cluster) receives launch
    |                               |   Splits grid into blocks, assigns blocks to SMs
    |                               |
    |                               SM receives a block:
    |                               |-- Allocates registers from register file
    |                               |-- Allocates shared memory from SRAM pool
    |                               |-- Divides block into warps
    |                               |-- Loads warp context into warp scheduler slots
    |                               |
    |                               Warp Scheduler issues first instruction:
    |                               |-- Fetch PTX/SASS instruction from I-cache
    |                               |-- Decode and dispatch to execution unit
    |                               |-- Execute (arithmetic: 4-6 cycles, memory: 200-400 ns)
    |                               |-- Switch to next ready warp (zero overhead)
```

---

## 9. Applied Pattern: The Perfect 1D Element-Wise Kernel

Bringing all these concepts together, here is an optimal element-wise kernel with proper bounds handling, vectorized loads, and no register waste:

```cpp
#include <cuda_runtime.h>

// Optimal element-wise scale + bias kernel:
// - 128-bit vectorized loads (float4)
// - __restrict__ to eliminate aliasing assumptions
// - __launch_bounds__ to give compiler register budget guidance
// - Coalesced warp access (consecutive thread IDs touch consecutive memory)

__launch_bounds__(256)
__global__ void scale_bias_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    float bias,
    int N)
{
    // Process 4 elements per thread using float4 (128-bit load/store)
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (gid + 3 < N) {
        // One 128-bit load: fetches 4 contiguous floats in a single instruction
        float4 in4 = *reinterpret_cast<const float4*>(&input[gid]);

        float4 out4;
        out4.x = in4.x * scale + bias;
        out4.y = in4.y * scale + bias;
        out4.z = in4.z * scale + bias;
        out4.w = in4.w * scale + bias;

        // One 128-bit store: writes 4 floats in a single instruction
        *reinterpret_cast<float4*>(&output[gid]) = out4;
    } else {
        // Scalar tail: handles remaining elements when N is not divisible by 4
        for (int k = gid; k < N; ++k) {
            output[k] = input[k] * scale + bias;
        }
    }
}

// Launch:
// dim3 block(256);
// dim3 grid((N/4 + 255) / 256);  // N/4 because each thread handles 4 elements
// scale_bias_kernel<<<grid, block>>>(d_input, d_output, scale, bias, N);
```

### Why This Kernel Is Fast:
1. **Vectorized loads** — 1 LDG.128 instruction per 4 floats, 4x fewer memory instructions
2. **Coalesced access** — consecutive thread IDs load consecutive addresses → single 128-byte transaction per warp
3. **Register efficient** — only `gid`, `in4`, `out4`, `scale`, `bias` in registers (6 registers used)
4. **No divergence** — all threads in warp take the same branch (fast path vs tail)
5. **`__launch_bounds__`** — compiler knows the max block size and can tune register allocation

---

## 10. Summary: Thread & Register Mental Model

| Concept | Physical Fact |
| :--- | :--- |
| SM Count | H100: 132 SMs, A100: 108 SMs |
| Register file per SM | 65,536 × 32-bit registers (256 KB) |
| Shared memory per SM | Up to 228 KB (Hopper) |
| Max threads per SM | 2,048 threads (64 warps) |
| Warp size | Always 32 threads |
| Register latency | 0 cycles (on-chip) |
| Local memory latency | Same as global DRAM (200-400 ns) |
| Warp switch overhead | 0 cycles (all warps live in register file) |
| Optimal block size | 128 or 256 (multiples of 32, fills SM evenly) |
| Divergence cost | Proportional to number of distinct paths taken |
