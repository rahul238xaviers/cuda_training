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

## 4. Hardware Memory Pitch & Alignment

When matrix widths are not integer multiples of hardware cache lines or memory bus transaction sizes (e.g. 64 bytes or 128 bytes), accessing consecutive rows can cause unaligned straddles.

To prevent unaligned memory penalties, high-performance runtimes use **Padded 2D Allocation** (known in CUDA as `cudaMallocPitch`):

* **Logical Width (`Cols`)**: The number of valid elements in a row.
* **Pitch (Byte Stride)**: The allocated row width rounded up to the nearest multiple of the hardware alignment boundary (e.g. 64 or 128 bytes).
* The gap between `Cols * sizeof(T)` and `pitch` consists of unused padding bytes.

```text
Row 0: [Element 0] [Element 1] ... [Element Cols-1] [PADDING BYTES]
Row 1: [Element 0] [Element 1] ... [Element Cols-1] [PADDING BYTES]
|<----------------------- pitch bytes ------------------------->|
```

By guaranteeing that each row starts at a memory address divisible by 64 or 128, every row access begins aligned on a hardware cache line boundary, eliminating multi-transaction bus penalties.

---

## 5. Summary & Key Takeaways

1. **Physical Reality**: Hardware memory is strictly a 1-dimensional array of bytes. Tensors are an illusion created by stride arithmetic.
2. **Stride Equation**: `Address = Base + (row * row_stride + col * col_stride) * sizeof(T)`.
3. **Prefer Flat Buffers**: Never use double-pointer indirection (`T**`) in GPU or high-throughput systems. Always use flat contiguous allocations with strided pointer offsets.
4. **Memory Pitch**: Use padded row strides to ensure every row aligns cleanly with 64-byte or 128-byte hardware cache lines.
