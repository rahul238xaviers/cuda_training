# 3.4 Grids, SM Occupancy & Grid-Stride Loops

## 1. The Grid Abstraction & SM Scheduling

A **Grid** represents the entirety of thread blocks launched for a CUDA kernel:
```cpp
dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);
dim3 block(threads_x, threads_y, threads_z);
kernel<<<grid, block>>>(...);
```

When a kernel is launched:
1. The GPU's hardware Work Distribution Engine assigns thread blocks to available Streaming Multiprocessors (SMs).
2. Each SM executes one or more thread blocks concurrently.
3. Blocks must be completely independent — CUDA makes **no guarantees** regarding execution order of blocks!

---

## 2. Theoretical vs. Achieved Occupancy

**Occupancy** is the ratio of active warps executing concurrently on an SM to the maximum theoretical warps supported by the hardware:
$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}$$

On Ampere / Hopper / Turing:
- Maximum Warps per SM: **32 to 64 warps** (1024 to 2048 threads per SM).
- Maximum Blocks per SM: **16 to 32 blocks**.
- Maximum Registers per SM: **64K 32-bit registers**.
- Maximum Shared Memory per SM: **64KB to 228KB**.

### Occupancy Bottlenecks
1. **Register Count**: If each thread uses 64 registers, an SM with 64K registers can only host $\frac{65536}{64} = 1024$ threads (32 warps $\rightarrow$ 50% occupancy).
2. **Shared Memory Allocation**: If each block requests 48KB shared memory and the SM capacity is 64KB, only 1 block can be scheduled simultaneously on that SM, regardless of available registers!
3. **Block Size**: Block sizes that are not multiples of 32 (e.g. 50 threads) waste warp execution units (32-thread granularity).

---

## 3. The Grid-Stride Loop Pattern

In beginner CUDA tutorials, kernels often assume `numBlocks = (N + blockSize - 1) / blockSize`, allocating one thread per element:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) { out[i] = in[i] * 2.0f; }
```
In production LLM engines, this is replaced by the **Grid-Stride Loop**:
```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = in[i] * 2.0f;
}
```

### Why Grid-Stride Loops are Crucial:
1. **Hardware Saturation Without Overhead**: Launch exactly enough blocks to saturate all SMs (e.g. `32 * num_SMs`). Launching millions of tiny blocks creates severe hardware scheduling overhead.
2. **Cache Line Reusability**: When a warp loops through memory, successive iterations leverage L1/L2 cache locality.
3. **Arbitrary Problem Sizes**: The exact same kernel handles $N = 100$ or $N = 10^{9}$ without changing grid dimensions or risking integer overflow in index calculations.
