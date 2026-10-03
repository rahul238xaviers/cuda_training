# Chapter 3.4: Grids, SM Occupancy & Grid-Stride Loops

Launching too few blocks leaves GPU hardware idle. Launching too many creates scheduling overhead. Using one-thread-per-element indexing limits flexibility and breaks cache locality. This chapter covers how the GPU's work distribution system works, how to reason about occupancy precisely, and how the grid-stride loop pattern replaces all of those problems with a single elegant construct.

---

## 1. The Grid: The Full Problem Launch

A **grid** is the complete collection of thread blocks you launch for one kernel invocation.

```cpp
dim3 grid(128, 1, 1);    // 128 blocks along X
dim3 block(256, 1, 1);   // 256 threads per block
kernel<<<grid, block>>>(args...);
// Total threads: 128 * 256 = 32,768 threads
```

The grid can be 1D, 2D, or 3D. For image processing you might use:
```cpp
dim3 grid(width / 32, height / 32, 1);
dim3 block(32, 32, 1);    // 32x32 = 1024 threads per block
```

For LLM token-level processing (one block per token, threads across embedding dim):
```cpp
dim3 grid(batch_size, 1, 1);
dim3 block(256, 1, 1);
```

---

## 2. The Hardware Work Distribution Engine

When a kernel launches, a dedicated **hardware GPC (Graphics Processing Cluster) Work Distributor** manages block assignment to SMs. This is not the CUDA runtime — it is a physical circuit that operates every few cycles.

```text
Work Distributor Logic:

1. Kernel launches with 1,024 blocks.
2. GPU has 108 SMs (A100).
3. Work distributor scans: which SMs have enough free registers and shared memory
   for one more block of this kernel?
4. First wave: assigns blocks 0-107 to SMs 0-107 (one per SM, if resources allow).
5. As each SM finishes a block, it reports completion.
6. Work distributor immediately assigns the next pending block to that SM.
7. Continues until all 1,024 blocks complete.
```

Critically: **blocks must be fully independent**. CUDA provides no ordering guarantees between blocks. Block 500 may complete before Block 1. If your kernel has any inter-block dependency (e.g., block 1 reads results written by block 0), you must launch separate kernels and rely on the implicit barrier between kernel launches.

---

## 3. Occupancy: The Quantitative Framework

Occupancy is the ratio of active warps on an SM to the SM's maximum possible active warps.

```formula
Occupancy = Active warps per SM / Maximum warps per SM
```

On H100, the maximum is 64 warps (2,048 threads) per SM.
On A100, the maximum is 64 warps (2,048 threads) per SM.

### Why Occupancy Matters

The SM has 4 warp schedulers. When one warp stalls (waiting for a memory load to return from HBM — 400 cycles), the scheduler must immediately switch to another ready warp. If there are only 4 warps active total (minimum), the SM will have nothing to do during those 400 cycles. With 32+ warps active, the scheduler always has something to execute while warps wait for memory.

**Rule of thumb**: Kernels bottlenecked by memory latency benefit greatly from higher occupancy. Kernels bottlenecked by arithmetic throughput benefit less — the ALUs are busy regardless.

---

## 4. The Three Occupancy Limiters

### Limiter 1: Register Count

```formula
Max threads from registers = floor(SM register file / registers per thread)
                           = floor(65,536 / R) where R = registers per thread

Occupancy from registers = min(Max threads, SM max threads) / SM max threads
```

Examples (H100: 65,536 registers, 2,048 max threads):

| Registers/thread | Max threads | Occupancy |
| :--- | :--- | :--- |
| 16 | 65536/16 = 4,096 → capped at 2,048 | 100% |
| 32 | 65536/32 = 2,048 | 100% |
| 64 | 65536/64 = 1,024 | 50% |
| 128 | 65536/128 = 512 | 25% |
| 255 | 65536/255 = 257 → rounded to 256 | 12.5% |

### Limiter 2: Shared Memory

```formula
Max blocks from smem = floor(SM smem capacity / smem per block)

Active threads from smem = Max blocks * threads per block
Occupancy from smem = Active threads / SM max threads
```

Example: H100 with 228 KB shared memory, 48 KB per block, 256 threads per block:
```
Max blocks = floor(228 / 48) = 4 blocks
Active threads = 4 * 256 = 1,024
Occupancy = 1,024 / 2,048 = 50%
```

### Limiter 3: Block Size

Thread blocks must be multiples of 32 (the warp size). Non-multiples waste execution lanes:

```text
Block size 50 threads → 2 warps = 64 threads allocated
14 threads (28%) sit idle in the second warp, consuming register/smem resources.

Rule: Always use block sizes that are multiples of 32.
Common choices: 128, 256, 512, 1024.
```

**Final occupancy** = the minimum across all three limiters.

---

## 5. Measuring Occupancy: The API

```cpp
// Method 1: At compile time, inspect ptxas output:
// nvcc --ptxas-options=-v -arch=sm_90 kernel.cu
// Output: "Used 48 registers, 16384 bytes smem" → plug into occupancy formula

// Method 2: At runtime, query theoretically achievable occupancy:
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxActiveBlocks,
    my_kernel,      // Kernel function
    blockSize,      // Threads per block
    sharedMemBytes  // Dynamic shared memory bytes
);
float occupancy = (float)(maxActiveBlocks * blockSize) / 2048.0f;  // H100 max
printf("Theoretical occupancy: %.1f%%\n", occupancy * 100.0f);

// Method 3: Let CUDA choose the optimal block size:
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &optimalBlockSize,
    my_kernel,
    0,    // dynamic shared memory (0 = none)
    0     // max block size limit (0 = use hardware max)
);
printf("Optimal block size: %d\n", optimalBlockSize);
```

---

## 6. Naive Indexing vs. Grid-Stride Loop

### The Naive Pattern (Beginner)

```cpp
// Launch one thread per element:
int gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid < N) {
    out[gid] = in[gid] * 2.0f;
}
// Grid: (N + blockDim.x - 1) / blockDim.x blocks
```

This works for small N but has serious problems at scale:
- N = 1 billion → 4 million blocks. The hardware work distributor becomes a bottleneck.
- Each block does almost no work. The amortized overhead of block setup/teardown dominates.
- The grid is too large for the work distributor to schedule efficiently.

### The Grid-Stride Loop (Production)

Instead of one thread per element, each thread handles multiple elements:

```cpp
// Fixed grid: always launch exactly enough blocks to fill the GPU
int total_threads = gridDim.x * blockDim.x;    // e.g. 108 SMs * 256 * 2 = 55,296 threads
int gid = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = gid; i < N; i += total_threads) {
    out[i] = in[i] * 2.0f;
}
```

Or written idiomatically:
```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += gridDim.x * blockDim.x) {
    out[i] = in[i] * 2.0f;
}
```

### Choosing the Grid Size

```cpp
// Rule: Launch 2-4 blocks per SM (provides enough warps to hide latency while
// keeping scheduling overhead near zero):
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
int blocks_per_sm = 2;   // Start here; profile and tune
int grid_size = num_sms * blocks_per_sm;
int block_size = 256;

element_wise_kernel<<<grid_size, block_size>>>(d_in, d_out, N);
```

### Visual Comparison: 3 Threads, N=9 Elements

```text
Naive (one-thread-per-element): Launches 3 blocks, each block does 1 element
  Block 0: Thread 0 processes i=0, Thread 1 processes i=1, Thread 2 processes i=2
  Block 1: Thread 0 processes i=3, Thread 1 processes i=4, Thread 2 processes i=5
  Block 2: Thread 0 processes i=6, Thread 1 processes i=7, Thread 2 processes i=8
  Total: 3 block launches.

Grid-Stride (1 block, stride = total threads = 3):
  Block 0: Thread 0 processes i=0,3,6 (stride over 3 iterations)
           Thread 1 processes i=1,4,7
           Thread 2 processes i=2,5,8
  Total: 1 block launch. Same work, fewer scheduling roundtrips.
```

---

## 7. Grid-Stride Loop for 2D Tensors (Common in Attention Kernels)

Many LLM kernels operate on 2D data `[batch, seq_len]` or `[rows, cols]`:

```cpp
__global__ void scale_2d_kernel(float* data, float scale, int rows, int cols) {
    // Grid-stride over both dimensions:
    for (int row = blockIdx.y * blockDim.y + threadIdx.y;
             row < rows;
             row += gridDim.y * blockDim.y) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x;
                 col < cols;
                 col += gridDim.x * blockDim.x) {
            data[row * cols + col] *= scale;
        }
    }
}

// Launch: fixed grid covering the GPU regardless of actual data size
dim3 block(32, 8);           // 256 threads, 2D arrangement
dim3 grid(min(32, (cols + 31) / 32),   // Cover columns, capped at 32
           min(16, (rows + 7) / 8));    // Cover rows, capped at 16
scale_2d_kernel<<<grid, block>>>(d_data, scale, rows, cols);
```

---

## 8. Persistent Kernels: The Most Extreme Grid-Stride Pattern

In throughput-critical pipelines (like FlashAttention or FMHA), the overhead of re-launching kernels between attention layers is non-trivial. **Persistent kernels** launch once and never return:

```cpp
__global__ void persistent_transform_kernel(
    const WorkQueue* queue,   // A lock-free task queue in global memory
    float* output,
    int* done_flag)
{
    while (true) {
        // Try to dequeue a task atomically:
        int task_id = atomicAdd(&queue->head, 1);
        if (task_id >= queue->total_tasks) break;   // No more work, exit

        // Process the task:
        process_task(queue, task_id, output);
    }
    // Last thread to finish signals the host:
    if (atomicAdd(done_flag, 1) == gridDim.x * blockDim.x - 1) {
        *done_flag = -1;   // Signal completion
    }
}

// Launch once, process millions of tasks:
persistent_transform_kernel<<<num_sms * 2, 256>>>(d_queue, d_output, d_done);
```

This pattern completely eliminates kernel launch overhead for iterative workloads.

---

## 9. Summary: Grid Design Decision Tree

```text
Q: How large is N?
├── N <= 1 million: Use naive one-thread-per-element, blockSize=256
├── N <= 100 million: Use grid-stride, gridSize = num_SMs * 2
└── N >= 1 billion: Use persistent kernel or double-buffer streaming

Q: Is the kernel memory-bound or compute-bound?
├── Memory-bound: Maximize occupancy (reduce registers/smem per thread)
└── Compute-bound: Reduce occupancy if needed to keep register pressure low

Q: Does the kernel use shared memory?
├── No: Max block size = 1024; registers are the only limiter
└── Yes: Size smem per block for 2-4 concurrent blocks per SM

Q: Is the problem 1D, 2D, or 3D?
├── 1D: dim3 grid(N / blockSize); dim3 block(blockSize)
├── 2D: dim3 grid(cols/BX, rows/BY); dim3 block(BX, BY)
└── Token batched: dim3 grid(batch, 1); dim3 block(head_threads, 1)
```
