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

```diagram:stride-layout
{
  "title": "Row-Major Stride: 2D Logical Grid Mapped to Flat Physical Memory",
  "subtitle": "float[2][4] in row-major order. Stride_Row = 4 elements (16 bytes). Stride_Col = 1 element (4 bytes).",
  "rows": 2,
  "cols": 4,
  "base": "0x2000",
  "typeName": "float[2][4] — 8 elements, 32 bytes"
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

## 4. Memory Pitch, Padded Strides & cudaMallocPitch

When storing 2D matrices in GPU memory, a subtle hardware trap occurs whenever the row width in bytes is not an exact multiple of the memory controller's transaction size (typically 128 bytes on NVIDIA Ampere/Hopper/Blackwell).

### The Address Drift Problem
Suppose you allocate a matrix with `Cols = 33` floats (each row is `33 * 4 = 132` bytes):
* **Row 0**: Starts at address `0x0000` (128-byte aligned). Threads 0..31 load elements 0..31 in 1 coalesced 128-byte transaction.
* **Row 1**: Starts at address `0x0084` (132 decimal). **This is NOT aligned to a 128-byte boundary!**
* **Result**: When threads in a warp access Row 1, the hardware memory controller detects an unaligned transaction and splits it into **two separate 128-byte memory transactions**.
* Every subsequent row drifts further out of alignment, degrading global memory throughput across the entire kernel.

### The Pitch Solution: Hardware Padding
To guarantee that the start of every row is strictly aligned to a 128-byte or 256-byte cache-line boundary, hardware engineers introduce **Row Pitch** (also called *padded stride* or *stride in bytes*):

```text
Row 0: [ 33 floats data (132 bytes) ] [ 31 floats padding (124 bytes) ] -> Pitch = 256 bytes (Aligned)
Row 1: [ 33 floats data (132 bytes) ] [ 31 floats padding (124 bytes) ] -> Pitch = 256 bytes (Aligned)
Row 2: [ 33 floats data (132 bytes) ] [ 31 floats padding (124 bytes) ] -> Pitch = 256 bytes (Aligned)
```

### CUDA Hardware API: cudaMallocPitch & cudaMemcpy2D
NVIDIA provides dedicated runtime APIs to automatically compute the optimal hardware pitch:

```cpp
float* d_matrix;
size_t pitch; // Returned in bytes by CUDA runtime (e.g. 256 or 512)
size_t width_bytes = Cols * sizeof(float);

// Allocate with hardware-aligned row pitch:
cudaMallocPitch(&d_matrix, &pitch, width_bytes, Rows);

// Copy host matrix to 2D pitched device allocation:
cudaMemcpy2D(
    d_matrix, pitch,             // Destination and device pitch (bytes)
    h_matrix, width_bytes,       // Source and host pitch (bytes)
    width_bytes, Rows,           // Width (bytes) and Height (rows)
    cudaMemcpyHostToDevice
);
```

### Addressing Pitched Memory Inside a CUDA Kernel
Because `pitch` is measured in **raw bytes** (not elements), you MUST perform byte arithmetic using `char*` before casting back to the element type:

```cpp
__global__ void pitchedMatrixKernel(float* d_matrix, size_t pitch, int Rows, int Cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < Rows && col < Cols) {
        // Step 1: Cast to char* to step by raw pitch bytes
        char* row_byte_ptr = ((char*)d_matrix) + (row * pitch);

        // Step 2: Cast to float* and access column
        float* row_ptr = (float*)row_byte_ptr;
        row_ptr[col] = row_ptr[col] * 2.0f;
    }
}
```

### Why We Use This Technique
1. **100% Coalesced Warps**: Every row starts on a 128-byte boundary, ensuring the warp's first transaction never straddles two cache lines.
2. **Eliminates Stride Penalty**: GPU memory controllers sustain near-peak DRAM bandwidth even on odd-sized matrices.
3. **Tensor Core Alignment**: Modern Tensor Core operations (e.g. WMMA and MMA instructions) strictly require base pointers and leading dimensions (strides) to be 16-byte or 128-byte aligned.

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

### The Tiling Solution: Eliminating Stride Penalties

When an operation requires strided traversal (such as transposing a matrix where row elements must be written into strided column positions), a naive loop touches hundreds or thousands of different cache lines across memory. Because the CPU L1 cache only holds a limited number of cache lines (e.g. 512 lines for a 64 KB cache), early lines are evicted before they can be reused, resulting in catastrophic cache thrashing.

**The Solution: 2D Cache-Blocking (Tiling)**.
Instead of processing an entire matrix at once, we decompose the matrix into small square sub-grids called **tiles** (e.g. `32 x 32` or `64 x 64`). By keeping all operations confined within a single tile, all source and destination memory lines remain hot in L1 cache with zero evictions.

---

### The Golden Tile Sizing Formula: In-Place vs. Out-of-Place

In systems architecture, how do engineers mathematically determine the optimal tile dimension `T` without guessing?

We use the **50% Working Set Rule**:
> **Rule of Thumb**: The active memory required by a tile should never exceed **50% of the cache capacity** (ideally target between **25% and 50%**).
> 
> *Hardware Rationale*: L1 caches are set-associative (typically 8-way or 16-way). Addresses separated by large powers of 2 often map to the exact same cache sets. Exceeding 50% capacity causes **Cache Set Conflict Misses**, where lines evict each other even if other cache sets are empty. Headroom is also required for stack variables, prefetch buffers, and loop registers.

#### Formula Variables:
* `C_L1`: Total L1 Data Cache capacity in bytes (e.g. 65,536 bytes for 64 KB).
* `S`: Size of one data element in bytes (4 bytes for single-precision `float`).
* `alpha`: Target cache safety factor (`0.25 <= alpha <= 0.50`).
* `T`: Tile width and height (`T x T` elements).

#### Case 1: Out-of-Place Transpose (2 Active Buffers: `src` and `dst`)
Both the source tile and the destination tile reside in L1 cache simultaneously:

```text
Memory Footprint = 2 * (T * T * S) <= alpha * C_L1

T <= sqrt( (alpha * C_L1) / (2 * S) )
```

Plugging in real hardware parameters (`C_L1 = 65,536 bytes`, `S = 4 bytes`, `alpha = 0.25`):
```text
T <= sqrt( (0.25 * 65,536) / (2 * 4) )
T <= sqrt( 16,384 / 8 )
T <= sqrt( 2,048 )
T <= 45.2 elements -> Rounded down to power-of-2: T = 32
```
**Golden Result for Out-of-Place**: **`T = 32`** (`32 x 32` floats = 4 KB per buffer, 8 KB total = 12.5% of L1).

#### Case 2: In-Place Transpose (1 Active Buffer: In-Place Swap)
Only a single matrix buffer exists in memory. Elements are swapped in-place using `std::swap(A[r * N + c], A[c * N + r])`:

```text
Memory Footprint = T * T * S <= alpha * C_L1

T <= sqrt( (alpha * C_L1) / S )
```

Plugging in real hardware parameters (`C_L1 = 65,536 bytes`, `S = 4 bytes`, `alpha = 0.25`):
```text
T <= sqrt( (0.25 * 65,536) / 4 )
T <= sqrt( 16,384 / 4 )
T <= sqrt( 4,096 )
T <= 64 elements
```
**Golden Result for In-Place**: **`T = 64`** (`64 x 64` floats = 16 KB total = 25% of L1).

---

## 6. Summary & Key Takeaways

1. **Physical Reality**: Hardware memory is strictly a 1-dimensional array of bytes. Tensors are an illusion created by stride arithmetic.
2. **Stride Equation**: `Address = Base + (row * row_stride + col * col_stride) * sizeof(T)`.
3. **Prefer Flat Buffers**: Never use double-pointer indirection (`T**`) in GPU or high-throughput systems. Always use flat contiguous allocations with strided pointer offsets.
4. **Memory Pitch**: Use padded row strides to ensure every row aligns cleanly with 64-byte or 128-byte hardware cache lines.
5. **Hardware Granularity**: Memory is moved in cache lines (64B or 128B). Non-unit strides discard up to 96.8% of memory bus bandwidth.
6. **CUDA Warp Coalescing**: Align thread memory access so 32 threads in a warp touch contiguous addresses, consolidating 32 memory requests into a single hardware transaction.
7. **The 50% Working Set Rule**: Size 2D processing tiles between 25% and 50% of L1 cache capacity (`T = 32` for out-of-place, `T = 64` for in-place) to eliminate cache thrashing and set-conflict misses.
