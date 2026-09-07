# 3.1 Threads & Registers

## 1. The Thread Execution Model

In NVIDIA's CUDA programming model, a **thread** is the finest grain of execution. When a kernel is launched:
```cpp
my_kernel<<<numBlocks, threadsPerBlock>>>(args...);
```
Each thread executes the identical kernel code on different data elements (SPMD: Single Program, Multiple Data).

### 1.1 Thread Identification
Within a block of threads, each thread is uniquely identified by the built-in 3D vector variable `threadIdx`:
- 1D block: index is simply `threadIdx.x`
- 2D block: index is `threadIdx.y * blockDim.x + threadIdx.x`
- 3D block: index is `(threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x`

Combined with `blockIdx` and `blockDim`, the unique global thread ID across the entire grid in 1D is:
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 2. Registers & Register Pressure

### 2.1 Fast On-Chip Registers
- Each Streaming Multiprocessor (SM) contains a fixed register file (e.g. 64K 32-bit registers on Ampere/Hopper).
- Registers are private to each thread and provide near-zero cycle latency access.
- Local variables declared inside a kernel (`int a; float b;`) reside in registers by default.

### 2.2 Register Spilling to Local Memory
- If a kernel uses more registers than available per thread (limited by block size and architecture max of 255 registers per thread), the compiler **spills** the excess variables into **Local Memory**.
- **Local Memory is stored in high-latency off-chip DRAM (HBM/GDDR)**, cached in L1/L2. Spilling causes dramatic performance degradation!
- You can check register usage and spilling during compilation using:
  ```bash
  nvcc --ptxas-options=-v -O3 kernel.cu
  ```
  Output shows:
  ```
  ptxas info    : Compiling entry function 'my_kernel' for 'sm_80'
  ptxas info    : Used 32 registers, 0 bytes smem, 0 bytes cmem[0], 0 bytes lmem
  ```
  `lmem` indicates spilled local memory bytes!

---

## 3. Vectorized Memory Access (128-bit Loads)

Modern NVIDIA GPUs achieve peak memory bus efficiency when threads load/store 128-bit data chunks in a single instruction:
- `float4` (4 x 32-bit floats = 16 bytes = 128 bits)
- `ulonglong2` / `uint4` (128 bits)
- `__nv_bfloat162` (2 x 16-bit bfloats in 32 bits, or vector of 4/8)

Using `float4` allows 1 thread to read 4 contiguous elements simultaneously, reducing instruction count by 4x and maximizing memory bus utilization:
```cpp
float4 data = *reinterpret_cast<const float4*>(&input[gid * 4]);
data.x = data.x * scale;
data.y = data.y * scale;
data.z = data.z * scale;
data.w = data.w * scale;
*reinterpret_cast<float4*>(&output[gid * 4]) = data;
```
