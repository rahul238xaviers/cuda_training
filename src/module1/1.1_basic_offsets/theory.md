# Module Chapter 1.1: Basic Memory Offsets & Pointer Arithmetic

In high-performance computing, deep learning, and CUDA GPU programming, memory efficiency is the bottleneck. High-level abstractions like multidimensional tensors must eventually be translated into physical, flat memory allocations. To write correct and fast CUDA kernels, you must master the fundamental mechanics of pointers, memory addressing, scaling, alignment, and low-level offsets.

---

## 1. Anatomy of a Pointer & Memory Addressing

A pointer is a primitive variable whose value is a physical or virtual memory address. A pointer does not store the data itself; it stores a locator pointing to the byte where the data begins.

### Virtual Address Space & Addressing
In modern operating systems and GPUs, applications interact with a Virtual Address Space:
* Byte-Addressability: Modern hardware is byte-addressable. Every unique address (e.g. 0x7ffee2bc81a0) refers to exactly 1 byte (8 bits) of physical memory.
* Pointer Width: On 64-bit architectures (both standard CPUs and CUDA GPUs), a memory address is represented by a 64-bit unsigned integer (8 bytes, represented in C++ as uintptr_t).
* All pointer variables occupy exactly 8 bytes of storage on 64-bit systems, regardless of whether they point to a 1-byte char, a 4-byte float, or a 16-byte Float4 struct.

```diagram:pointer-addressing
{
  "title": "Textbook Schematic: 64-Bit Pointer Resolution & Byte Ladder",
  "subtitle": "Modeled after CS:APP Figure 2.1 & 3.2. 64-bit pointer resolving to contiguous 16-byte buffer in RAM/VRAM.",
  "pointer": {
    "name": "ptr",
    "type": "uint32_t*",
    "location": "Stack Frame / CPU Register",
    "address": "0x7ffee2bc81a0",
    "value": "0x1000",
    "size": "8 bytes (64-bit)"
  },
  "target": {
    "location": "RAM / VRAM (Global Memory Buffer)",
    "baseAddress": "0x1000",
    "typeName": "uint32_t[4] (16 Bytes)",
    "cells": [
      { "name": "ptr[0]", "offset": "+0", "address": "0x1000", "bytes": "4 bytes", "hex": "0x0000002A", "val": "42" },
      { "name": "ptr[1]", "offset": "+4", "address": "0x1004", "bytes": "4 bytes", "hex": "0x00000054", "val": "84" },
      { "name": "ptr[2]", "offset": "+8", "address": "0x1008", "bytes": "4 bytes", "hex": "0x0000007E", "val": "126" },
      { "name": "ptr[3]", "offset": "+12", "address": "0x100C", "bytes": "4 bytes", "hex": "0x000000A8", "val": "168" }
    ]
  }
}
```


---

## 2. Heap vs. Stack Memory Management

Physical memory is partitioned into distinct regions, the most critical being the Stack and the Heap.

### Stack Memory
* Managed automatically by the compiler.
* Follows a strict Last-In, First-Out (LIFO) model. Fast, pointer-bump cleanup.
* Small size (1 to 8 MB). Allocating large tensors on the stack causes a Stack Overflow.

### Heap Memory
* Managed dynamically by the programmer via malloc/free, new/delete, or cudaMalloc/cudaFree.
* Slower allocation (requires searching free-lists in the OS memory manager).
* Practically unbounded (limited only by physical RAM or GPU VRAM).
* Risk of Memory Leaks: In long-running training loops, failing to free heap buffers causes Out Of Memory (OOM) crashes.

---

## 3. Pointer Arithmetic & Scaling Laws

When you perform addition or subtraction on a typed pointer (T*), the compiler does not shift the underlying address by raw bytes. Instead, it scales the offset by sizeof(T).

### The Fundamental Scaling Formula:
```text
Address(ptr + N) = Address(ptr) + N * sizeof(T)
Address(ptr - N) = Address(ptr) - N * sizeof(T)
```

### Visualizing Address Jumps:
Given base address 0x1000:
* uint8_t* p: (p + 1) advances to 0x1001 (jump of 1 byte)
* float* p:   (p + 1) advances to 0x1004 (jump of 4 bytes)
* Float4* p:  (p + 1) advances to 0x1010 (jump of 16 bytes)

Rule: To step memory by raw bytes, the pointer MUST be cast to a 1-byte lens: uint8_t* or char*.

---

## 4. Hardware Memory Alignment & Cache-Line Granularity

Processors and GPUs do not read memory one byte at a time. They fetch memory across an ultra-wide physical memory bus in chunks called Cache Lines or Memory Transactions (typically 64, 128, or 256 bytes).

### The Warehouse Forklift Mental Model
Think of memory like a warehouse:
* The floor has yellow lines painted every 256 bytes (0, 256, 512, 768...).
* The hardware memory bus is a wide forklift that ONLY drops its prongs on the yellow lines.
* If a 16-byte tensor starts at address 250 (straddling the yellow line at 256):
  - The first 6 bytes sit in Cache Line 0 (bytes 0 to 255).
  - The remaining 10 bytes spill into Cache Line 1 (bytes 256 to 511).
  - A single load forces the hardware to perform TWO slow memory transactions!
  - On GPUs, unaligned 128-bit loads trigger an immediate hardware crash: CUDA_ERROR_ILLEGAL_ADDRESS.

### The Bitwise Alignment Formula:
To round any address up to the nearest power-of-2 alignment boundary without slow modulo division:
```text
aligned_addr = (addr + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1)
```

Why this works:
1. Adding (ALIGNMENT - 1) adds the maximum possible remainder. If addr is already aligned, it stays below the next boundary. If addr has even 1 extra byte, it crosses into the next boundary bracket.
2. The mask ~(ALIGNMENT - 1) has 0s in the bottom bits, acting as an eraser that wipes out the remainder and snaps cleanly onto the boundary.

### Padding Calculation:
```text
padding_bytes = aligned_addr - addr
```
Padding represents the empty slack bytes deliberately skipped so the next tensor begins on a hardware boundary.

---

## 5. 2D Pitched Memory Layouts (cudaMallocPitch)

In 2D matrices (images, weight tables), rows are stored consecutively in flat RAM.
If a row has an irregular byte width (e.g. 17 floats = 68 bytes), the next row will start at an unaligned address (68 is not divisible by 128 or 256).

### Pitch vs. Width:
To solve this, GPU allocators (cudaMallocPitch) pad every row:
* Width in Bytes: The actual useful payload (e.g. 32 floats = 128 bytes).
* Pitch: The physical stride in bytes from the start of one row to the start of the next (e.g. 256 bytes).
* Padding per Row: Pitch - Width (e.g. 256 - 128 = 128 bytes of untouched padding).

```text
Flat Memory Layout:
Row 0: [ 128 bytes of useful data ] [ 128 bytes of padding ] -> 256 bytes
Row 1: [ 128 bytes of useful data ] [ 128 bytes of padding ] -> 256 bytes
```

### Accessing Element (r, c) in Pitched Memory:
```cpp
// Step 1: Jump across rows using PITCH in byte space:
uint8_t* row_bytes = base_ptr + r * PITCH;

// Step 2: Cast to typed pointer and index the column:
float* row_floats = reinterpret_cast<float*>(row_bytes);
float value = row_floats[c];
```

---

## 6. Vectorized Memory Loads: 128-Bit Transfers (Float4)

Loading individual 32-bit floats one by one forces the CPU/GPU to issue 4 separate load instructions.
Modern hardware contains 128-bit wide vector load units (ARM NEON, AVX-512, and NVIDIA PTX LDG.E.128).

### The Float4 Struct:
```cpp
struct alignas(16) Float4 {
    float x, y, z, w; // 4 floats = 16 bytes = 128 bits
};
```
* alignas(16) forces the compiler to ensure every Float4 starts on an address divisible by 16.
* 1 single instruction moves 16 bytes into vector registers.
* Issues 4x fewer instructions and achieves near-peak DRAM bus saturation (40 to 100+ GB/s).

---

## 7. Custom Aligned Bump Allocator (Memory Pools)

High-performance inference engines (PyTorch c10::CUDAAllocator, TensorRT) cannot afford expensive OS malloc/cudaMalloc calls during real-time inference (each syscall takes microseconds).

### The Bump Allocator Pattern:
1. Allocate one massive contiguous pool (e.g. 1 GB) once at startup.
2. Maintain a single integer pointer: curr_bump = pool_start.
3. On every tensor request (size, align):
   - Round curr_bump UP to align: aligned_addr = (curr_bump + (align - 1)) & ~(align - 1)
   - Verify it fits: if (aligned_addr + size <= pool_end)
   - Hand out: return reinterpret_cast<void*>(aligned_addr)
   - Advance: curr_bump = aligned_addr + size
Allocation takes less than 1 nanosecond with zero syscalls!

---

## 8. Asynchronous Double-Buffering (Pointer Ping-Pong)

In streaming inference, GPUs execute computations orders of magnitude faster than the PCIe bus can supply data. If execution is sequential, the GPU sits idle 50% of the time waiting for PCIe transfers.

### The Double-Buffer Solution:
Allocate two buffers: Buffer A and Buffer B.
Maintain two pointers:
* compute_buf: pointing to the buffer the GPU is actively computing on.
* transfer_buf: pointing to the buffer the CPU/PCIe bus is actively filling with the next batch.

```text
Batch Timeline:
[ PCIe fills transfer_buf ] <--- Running in Parallel ---> [ GPU computes on compute_buf ]
                                        |
                          [ Sync Barrier: Both Finish ]
                                        |
                    [ Pointer Swap: std::swap(compute_buf, transfer_buf) ]
```
Physical data in RAM never moves. Only two 8-byte pointer variables swap their addresses in a single CPU cycle.

---

## 9. 2D Matrix Tiling & Cache Thrashing Prevention

In a naive 2D matrix transpose (dst[col, row] = src[row, col]), reading is sequential, but writing jumps across the entire matrix (each write jumps by DIM floats).
When DIM = 512, every write lands 2,048 bytes away in a different cache line. The CPU/GPU cache is overwhelmed, repeatedly evicting cache lines before they can be filled (Cache Thrashing), causing memory bandwidth to collapse by 5x to 10x.

### Tiling (Blocking) Solution:
Decompose the matrix into 16x16 tiles:
* A 16x16 tile occupies only 256 floats (1,024 bytes).
* 1,024 bytes fits comfortably inside the ultra-fast L1 cache.
* All 16 cache lines remain warm in L1 cache while the tile is transposed and written, eliminating cache evictions.

### Clear Frame-of-Reference Coordinate Mapping:
Avoid confusing abbreviations. Explicitly separate global grid coordinates from local tile coordinates:
```text
num_tiles_per_row = DIM / TILE_SIZE;
total_tiles = num_tiles_per_row * num_tiles_per_row;

For each tile_number from 0 to total_tiles - 1:
    global_tile_start_row = (tile_number / num_tiles_per_row) * TILE_SIZE;
    global_tile_start_col = (tile_number % num_tiles_per_row) * TILE_SIZE;

    For each local_row from 0 to TILE_SIZE - 1:
        For each local_col from 0 to TILE_SIZE - 1:
            current_row = global_tile_start_row + local_row;
            current_col = global_tile_start_col + local_col;

            // Transpose assignment:
            dst[current_col * DIM + current_row] = src[current_row * DIM + current_col];
```

---

## 10. Master Revision Cheat Sheet

A quick-reference guide to the essential patterns learned in Chapter 1.1:

| Technique | Problem Solved | Core C++ Syntax / Idiom |
| :--- | :--- | :--- |
| **Pointer Scaling** | Advancing typed pointers | `ptr + N` jumps by `N * sizeof(*ptr)` bytes |
| **Byte-Level Lens** | Inspecting raw RAM bytes | `const uint8_t* raw = reinterpret_cast<const uint8_t*>(data);` |
| **Two-Pointer Reversal** | In-place array inversion | `while (left < right) { std::swap(*left++, *right--); }` |
| **Circular Ring Buffer** | Fixed-size streaming window | `*head = val; head++; if (head == base + CAP) head = base;` |
| **Aligning Up (Bitwise)**| Fast cache-line snapping | `aligned = (addr + (ALIGN - 1)) & ~(ALIGN - 1);` |
| **Padding Calculation** | Tracking wasted alignment bytes | `padding = aligned_addr - original_addr;` |
| **Pitched Row Offset** | Navigating padded 2D memory | `row_ptr = reinterpret_cast<float*>(base + r * PITCH);` |
| **128-Bit Vectorization**| Saturating memory bus throughput | `dst_vec[i] = src_vec[i];` using `struct alignas(16) Float4` |
| **Bump Allocator** | Sub-nanosecond heap allocation | `addr = aligned_addr; curr_bump = aligned_addr + size;` |
| **Pointer Ping-Pong** | Zero-copy pipeline swapping | `std::swap(compute_buf, transfer_buf);` |
| **Tiled Matrix Transpose**| Preventing L1 cache thrashing | Outer loop over `tile_number`, inner loops over `TILE_SIZE` |
