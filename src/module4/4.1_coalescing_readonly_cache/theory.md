# 4.1 Memory Coalescing & Read-Only Cache

## 1. Global Memory Architecture & Transaction Granularity

Global Memory (GPU DRAM / HBM / GDDR) is connected to the GPU SMs via a wide 128-bit to 4096-bit memory bus.
When threads in a warp access global memory:
- The hardware checks the memory addresses requested by all 32 threads.
- If all 32 threads access a contiguous 128-byte chunk of memory aligned to a 128-byte boundary, the hardware services the entire warp request in **a single 128-byte transaction**!
- This is called **Coalesced Memory Access** (100% memory bus efficiency).

### 1.1 Uncoalesced / Strided Access Penalty
If threads in a warp access non-contiguous memory (e.g. strided access `in[threadIdx.x * stride]` where `stride = 32`):
- Each of the 32 threads accesses a different 128-byte memory segment.
- The GPU must issue **32 separate memory transactions** to read 32 floats (128 bytes of useful data requires $32 \times 128 = 4096$ bytes transferred!).
- Efficiency drops to $\frac{128}{4096} = 3.125\%$! The kernel becomes severely memory-bandwidth bound.

---

## 2. The Read-Only Data Cache (`__ldg` and `const __restrict__`)

Modern NVIDIA GPUs feature a dedicated Read-Only Data Cache (bypassing the standard L1 write-back cache):
- Higher bandwidth and non-coherent caching for immutable data (weights, embedding matrices).
- When a pointer is marked `const __restrict__`:
  ```cpp
  __global__ void my_kernel(const float* __restrict__ weights, float* out)
  ```
  The compiler automatically emits `LDG` instructions (Load Global via Read-Only Cache).
- You can explicitly force read-only cache loading via the CUDA intrinsic:
  ```cpp
  float w = __ldg(&weights[idx]);
  ```

---

## 3. Constant Memory (`__constant__`)

NVIDIA GPUs provide 64KB of fast **Constant Memory**:
```cpp
__constant__ float c_filter[256];
```
- Backed by a specialized on-chip constant cache.
- **Optimal access pattern**: When all 32 threads in a warp read the **same memory address** simultaneously, constant memory services all 32 threads in **1 clock cycle** via hardware broadcast!
- Ideal for: Model hyperparameters, normalization epsilon, layer scale weights.
