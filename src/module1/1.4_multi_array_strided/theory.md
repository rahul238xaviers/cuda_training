# Chapter 1.4: Multi-Array Strided Streaming & Memory Bandwidth Bound Ops

In machine learning workloads, element-wise vector operations (such as residual addition `Residual + LayerOutput`, GeLU activations, and SAXPY `Y = alpha * X + Y`) stream multiple arrays simultaneously through the memory bus.

Understanding how multiple independent pointers interact with the cache hierarchy and memory controller is fundamental to optimizing bandwidth-bound CUDA kernels.

---

## 1. Multi-Stream Traversal & Pointer Pacing

When an operation consumes elements from two or more input buffers (`A` and `B`) and writes to an output buffer (`C`), the memory controller must service simultaneous memory access streams.

### The SAXPY Kernel Pattern (Single-Precision A*X + Y)
```cpp
void saxpy_strided(size_t n, float alpha, 
                   const float* __restrict__ x, size_t stride_x,
                   float* __restrict__ y, size_t stride_y) {
    for (size_t i = 0; i < n; ++i) {
        y[i * stride_y] = alpha * x[i * stride_x] + y[i * stride_y];
    }
}
```

### The `__restrict__` Keyword
In C++, the compiler must assume that pointers `x` and `y` might point to overlapping memory regions (pointer aliasing). 
* Without `__restrict__`, every time the compiler writes to `y[i]`, it is forced to assume `x[i+1]` might have changed, forbidding it from keeping values cached in registers.
* With `__restrict__`, you promise to the compiler and CUDA runtime that the pointers reference distinct physical memory allocations. This enables aggressive loop unrolling, register caching, and SIMD vectorization.

---

## 2. Arithmetic Intensity & The Memory Wall

Vector operations like SAXPY have extremely low **Arithmetic Intensity**:
* **Operations Performed**: 1 multiply + 1 add = 2 FLOPs per element.
* **Bytes Transferred**: Load `x` (4 bytes) + Load `y` (4 bytes) + Store `y` (4 bytes) = 12 bytes per element.
* **Intensity**: `2 FLOPs / 12 Bytes = 0.167 FLOPs / Byte`.

```diagram:pointer-addressing
{
  "title": "Textbook Schematic: Dual-Stream Vector Traversal",
  "subtitle": "Concurrent streaming from Input Buffer X and Input/Output Buffer Y over the memory bus.",
  "pointer": {
    "name": "stream_ptrs",
    "type": "float* (x, y)",
    "location": "Registers",
    "address": "0x7ffee2bc6000",
    "value": "x: 0x1000 | y: 0x4000",
    "size": "8 bytes each"
  },
  "target": {
    "location": "DRAM / VRAM Channels",
    "baseAddress": "0x1000 (x) | 0x4000 (y)",
    "typeName": "float[N] Independent Arrays",
    "cells": [
      { "name": "x[i]", "offset": "+0", "address": "0x1000", "bytes": "4B", "hex": "Load", "val": "1.5f" },
      { "name": "x[i+1]", "offset": "+4", "address": "0x1004", "bytes": "4B", "hex": "Load", "val": "2.5f" },
      { "name": "y[i]", "offset": "+0", "address": "0x4000", "bytes": "4B", "hex": "Load/Store", "val": "10.0f" },
      { "name": "y[i+1]", "offset": "+4", "address": "0x4004", "bytes": "4B", "hex": "Load/Store", "val": "20.0f" }
    ]
  }
}
```

Because modern GPUs can perform trillions of FLOPs per second but are bounded by 1 to 3 TB/s of memory bandwidth, operations with low arithmetic intensity are strictly **Memory Bandwidth Bound**. The ALUs are mostly idle waiting for DRAM requests to return.

---

## 3. Vectorized Memory Access (`float4` / 128-bit Loads)

When elements are stored contiguously (`stride = 1`), issuing 32-bit scalar loads (`LDR.32`) leaves the memory bus underutilized.

Both standard CPU architectures (via AVX-512) and NVIDIA GPUs (via `LDG.E.128`) can load **128 bits (16 bytes = 4 floats)** in a single instruction:

```cpp
// Vectorized 128-bit memory load in CUDA:
void saxpy_vectorized(int n, float alpha, const float* x, float* y) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int n4 = n / 4;

    for (int i = 0; i < n4; ++i) {
        float4 vx = x4[i]; // Exactly ONE 128-bit memory load instruction
        float4 vy = y4[i]; // Exactly ONE 128-bit memory load instruction
        
        vy.x = alpha * vx.x + vy.x;
        vy.y = alpha * vx.y + vy.y;
        vy.z = alpha * vx.z + vy.z;
        vy.w = alpha * vx.w + vy.w;
        
        y4[i] = vy;        // Exactly ONE 128-bit memory store instruction
    }
}
```

### Why Vectorized Loads Double Throughput
1. **Fewer Instructions**: Issuing one 128-bit instruction instead of four 32-bit instructions cuts instruction decode and warp scheduling overhead by 75%.
2. **Coalesced Bus Bursts**: Hardware memory buses fetch 32-byte or 64-byte chunks. A 128-bit load fills an entire memory transaction sector in one shot.

---

## 4. Summary & Best Practices

1. **Memory Bound Nature**: Multi-array element-wise ops are bounded by memory bandwidth, not compute. Minimizing memory transactions is the number one priority.
2. **Pointer Aliasing & `__restrict__`**: Always mark non-overlapping pointers with `__restrict__` so the compiler can aggressively cache values in registers.
3. **128-bit Vectorization**: Cast contiguous 32-bit buffers to 128-bit types (`float4`) to maximize memory controller bus utilization and reduce instruction issuance overhead.
