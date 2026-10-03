# Chapter 1.2: Strides, Pitch & Pointer Indirection

In deep learning frameworks and high-performance CUDA computing, multi-dimensional tensors (e.g. matrices of shape `[Batch, Heads, SeqLen, Dim]`) do not physically exist as multi-dimensional objects in silicon. Hardware memory is strictly a one-dimensional array of bytes.

To represent higher-dimensional tensors in flat DRAM/VRAM, systems software relies on **strides** and **pointer indirection**. Mastering these concepts is essential to writing high-throughput CUDA kernels and avoiding catastrophic memory bus bottlenecks.

---

## 1. Physical Memory vs. Multidimensional Tensors

DRAM and VRAM controllers only understand a single scalar address: a 64-bit integer representing a byte offset from the start of memory. 

When you define a 2D matrix of shape `[Rows, Cols]`, the compiler and hardware must map a 2D logical coordinate `(row, col)` to a 1D physical memory address.

### The Stride Representation
A **stride** defines the number of elements (or bytes) you must skip in physical memory to advance by exactly 1 unit along a specific logical dimension:

* **Row Stride**: The offset required to move to the next row: `row_stride = Cols`
* **Column Stride**: The offset required to move to the next column: `col_stride = 1`

### Flat Address Mapping Formula:
```text
Address(row, col) = BaseAddress + (row * Stride_Row + col * Stride_Col) * sizeof(T)
```

```diagram:pointer-addressing
{
  "title": "Textbook Schematic: Strided 2D Matrix Memory Layout",
  "subtitle": "Logical (row, col) indexed into flat physical memory with Stride_Row = 4 elements (16 bytes).",
  "pointer": {
    "name": "matrix_base",
    "type": "float*",
    "location": "Stack Frame / Register",
    "address": "0x7ffee2bc9000",
    "value": "0x2000",
    "size": "8 bytes (64-bit)"
  },
  "target": {
    "location": "RAM / VRAM (Global Memory)",
    "baseAddress": "0x2000",
    "typeName": "float[2][4] (8 Elements, 32 Bytes)",
    "cells": [
      { "name": "(0, 0)", "offset": "+0", "address": "0x2000", "bytes": "4B", "hex": "0x00", "val": "1.0f" },
      { "name": "(0, 1)", "offset": "+4", "address": "0x2004", "bytes": "4B", "hex": "0x04", "val": "2.0f" },
      { "name": "(1, 0)", "offset": "+16", "address": "0x2010", "bytes": "4B", "hex": "0x10", "val": "5.0f" },
      { "name": "(1, 1)", "offset": "+20", "address": "0x2014", "bytes": "4B", "hex": "0x14", "val": "6.0f" }
    ]
  }
}
```

---

## 2. Row-Major vs. Column-Major Layouts

The order in which dimensions are flattened into physical memory determines the memory access pattern of your algorithms:

### Row-Major Layout (C, C++, PyTorch default)
* Consecutive elements of a row are placed adjacent to each other in memory.
* Advancing `col` moves by `+1 * sizeof(T)`.
* Advancing `row` moves by `+Cols * sizeof(T)`.
* Optimal for iterating row-by-row: consecutive loop iterations load consecutive physical addresses, generating 100% cache line hits and coalesced GPU memory bursts.

### Column-Major Layout (Fortran, MATLAB, cuBLAS)
* Consecutive elements of a column are stored adjacently.
* Advancing `row` moves by `+1 * sizeof(T)`.
* Advancing `col` moves by `+Rows * sizeof(T)`.

```cpp
// Traversing row-major data contiguously (Cache-friendly):
for (int r = 0; r < Rows; ++r) {
    float* row_ptr = base_ptr + r * Cols;
    for (int c = 0; c < Cols; ++c) {
        // Consecutive iterations touch row_ptr[0], row_ptr[1], row_ptr[2]...
        float val = row_ptr[c];
        process(val);
    }
}
```

---

## 3. Pointer Indirection: The Two Mental Models

When dealing with dynamically-sized multi-dimensional data, engineers typically choose between two architectural approaches:

### Model A: Flat Contiguous Buffer with Stride Arithmetic (Champion Approach)
* Allocate one single contiguous block of `Rows * Cols * sizeof(T)` bytes using `malloc`, `new`, or `cudaMalloc`.
* Index elements via explicit strided arithmetic: `base[r * Cols + c]`.
* **Hardware Advantage**: 
  - Exactly 1 allocation call.
  - Zero pointer overhead.
  - Memory is guaranteed contiguous in physical address space.
  - Hardware prefetchers and GPU memory coalescing work at peak theoretical bandwidth.

### Model B: Array-of-Pointers Indirection (`float**`)
* Allocate an array of `Rows` pointer variables (`float*`), where each pointer holds the address of a separately allocated row buffer.
* Indexing syntax is `ptr[r][c]`.
* **Hardware Penalty**:
  - Requires `Rows + 1` separate memory allocations.
  - Wastes `Rows * 8` bytes of memory just to store 64-bit row pointers.
  - **Double Dereference**: Hardware must first load the address from `ptr[r]` (DRAM access 1), wait for it to arrive, and then load `ptr[r][c]` (DRAM access 2).
  - Row buffers are scattered across heap memory, completely breaking spatial locality and GPU coalescing.

```cpp
// Flat contiguous allocation (Optimal for CUDA and Systems):
float* d_matrix;
size_t total_elements = Rows * Cols;
cudaMalloc(&d_matrix, total_elements * sizeof(float));

// In kernel execution:
// threadIdx.x accesses contiguous elements: d_matrix[row * Cols + threadIdx.x]
```

---

## 5. Cache Lines, Bandwidth Waste & GPU Coalescing

In CPU and GPU memory hardware, memory controllers never transfer 1 byte or 1 float in isolation. Physical memory buses are wide, parallel data highways designed for burst transfers.

### The Cache Line / Transaction Granularity
* **x86 CPUs**: Transfer in chunks of **64 bytes** (16 single-precision floats).
* **Apple Silicon M-Series CPUs**: Transfer in chunks of **128 bytes** (32 single-precision floats).
* **NVIDIA GPUs**: Memory controllers issue requests in **32-byte or 128-byte sector transactions** (for 32 threads in a warp).

Whenever your program requests a single 4-byte float at address `A`, the hardware cache controller fetches the entire cache line enclosing that address.

```diagram:cache-align
{
  "title": "Cache Line Granularity: 128-Byte Bus Transfer (Apple Silicon / NVIDIA GPU)",
  "subtitle": "A single float read fetches 32 floats (128 bytes). Strided access wastes up to 96.875% of this transferred data."
}
```

### The Physics of Stride: Bandwidth Utilization

The stride at which your algorithm traverses memory directly dictates what percentage of fetched bus bandwidth is actually used:

#### Case 1: Contiguous Traversal (Unit Stride = 1)
```cpp
for (int i = 0; i < N; ++i) { sum += buffer[i]; }
```
1. `buffer[0]` triggers a cache miss. The bus fetches 128 bytes (32 floats).
2. The next 31 loop iterations (`buffer[1]` through `buffer[31]`) read directly from L1 cache at zero bus cost.
3. **Bandwidth Utilization**: `(32 * 4 bytes used) / 128 bytes fetched = 100%`.
4. **Hardware Stream Prefetcher**: Detects the contiguous sequential stream and proactively streams upcoming cache lines from DRAM before the CPU core even requests them.

#### Case 2: Strided Traversal (Stride = 32 floats = 128 bytes)
```cpp
for (int i = 0; i < N; i += 32) { sum += buffer[i]; }
```
1. `buffer[0]` triggers a cache miss. The bus fetches 128 bytes (32 floats).
2. The next iteration immediately reads `buffer[32]`. It **discards the remaining 31 floats** in the previous cache line!
3. `buffer[32]` resides on a completely separate 128-byte line, triggering another cache miss.
4. **Bandwidth Utilization**: `(1 * 4 bytes used) / 128 bytes fetched = 3.125%`.
5. **Bus Waste**: **96.875%** of the transferred memory data is completely wasted and thrown away.

```text
Strided Read (Stride = 32 floats = 128 bytes):

Fetch 1:  [f0=USED] [f1 discarded] [f2 discarded] ... [f31 discarded]  -> 4B used / 128B fetched
Fetch 2:  [f32=USED] [f33 discarded] ... [f63 discarded]               -> 4B used / 128B fetched
Fetch 3:  [f64=USED] [f65 discarded] ... [f95 discarded]               -> 4B used / 128B fetched

Bandwidth utilization: 4B / 128B = 3.125%
Bus waste: 96.875% -- bus is choked with unused data!
```

### Direct Bridge to CUDA: Warp Memory Coalescing
In CUDA architecture, a warp consists of **32 parallel threads** executing the exact same instruction simultaneously:
* **Coalesced Access (Golden Rule of CUDA)**: If Thread 0 reads `ptr[0]`, Thread 1 reads `ptr[1]`, ..., Thread 31 reads `ptr[31]`, their 32 requests fall cleanly into a single aligned 128-byte segment. The GPU hardware memory controller fulfills all 32 threads in **1 single memory transaction**!
* **Non-Coalesced / Strided Access**: If each thread reads with a stride of 32 (Thread 0 reads `ptr[0]`, Thread 1 reads `ptr[32]`, etc.), the 32 addresses land in 32 separate cache lines. The memory controller is forced to issue **32 separate memory transactions**, serializing the warp and throttling kernel throughput down to a fraction of the GPU's memory bandwidth.

---

## 6. Summary & Key Takeaways

1. **Physical Reality**: Hardware memory is strictly a 1-dimensional array of bytes. Tensors are an illusion created by stride arithmetic.
2. **Stride Equation**: `Address = Base + (row * row_stride + col * col_stride) * sizeof(T)`.
3. **Prefer Flat Buffers**: Never use double-pointer indirection (`T**`) in GPU or high-throughput systems. Always use flat contiguous allocations with strided pointer offsets.
4. **Memory Pitch**: Use padded row strides to ensure every row aligns cleanly with 64-byte or 128-byte hardware cache lines.
5. **Hardware Granularity**: Memory is moved in cache lines (64B or 128B). Non-unit strides discard up to 96.8% of memory bus bandwidth.
6. **CUDA Warp Coalescing**: Align thread memory access so 32 threads in a warp touch contiguous addresses, consolidating 32 memory requests into a single hardware transaction.
