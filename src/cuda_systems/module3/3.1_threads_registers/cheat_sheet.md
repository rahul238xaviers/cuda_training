# Topic 3.1: Threads, Registers & Vectorized Loads — Cheat Sheet

A concise reference for thread coordinates, register budgeting, and 128-bit memory loads.

---

### 1. Global Thread Coordinate Formulas
- **1D Grid**:
  ```cpp
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  ```
- **2D Grid**:
  ```cpp
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row * width + col;
  ```
- **3D Tensor Grid [Batch, Heads, Dim]**:
  ```cpp
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  int offset = b * (num_heads * dim) + h * dim + d;
  ```

---

### 2. Register Budget & Spilling
- SM register capacity is fixed (e.g. 64K 32-bit registers).
- Each thread can use up to 255 registers. If usage exceeds the limit, variables spill into **Local Memory** (DRAM cached in L1/L2), reducing throughput by up to 10x.
- Check register usage: `nvcc --ptxas-options=-v -O3 kernel.cu`.

---

### 3. Vectorized 128-bit Memory Loads (`float4`)
- GPUs achieve maximum memory bus throughput when loading 16 bytes (128 bits) per thread in a single instruction:
  ```cpp
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < n) {
      float4 val = *reinterpret_cast<const float4*>(&in[idx]);
      val.x *= scale; val.y *= scale; val.z *= scale; val.w *= scale;
      *reinterpret_cast<float4*>(&out[idx]) = val;
  }
  ```

---

### 4. LLM Systems Context
- **Hidden Dimension Streaming**: Modern LLMs (hidden dim 4096 or 8192) divide dimension into 1024 or 2048 `float4` transactions per token.

---

### 5. Common Pitfalls
- Casting unaligned pointers to `float4*`: `float4` loads require 16-byte aligned base addresses. If unaligned, the GPU triggers an illegal address trap.
