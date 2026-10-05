# Chapter 1.7: Subgrids, Tiles & Padding in GPU Memory

Tiling is the foundational technique that makes GPU computation fast. Without it, every kernel that reads from a 2D matrix is limited by HBM bandwidth. With it, the same data is reused from fast on-chip shared memory — delivering 15x lower latency per access and dramatically higher arithmetic intensity.

Padding is tiling's necessary companion: hardware memory buses and shared memory banks have alignment and width requirements that make it essential to pad data arrays to specific widths even when the logical data is smaller.

---

## 1. The Subgrid Concept: Dividing Work into Tiles

A tile is a rectangular sub-region of a 2D matrix that a single thread block processes. The matrix is logically divided into a grid of tiles, and each tile is assigned to exactly one block.

```text
Global matrix M[8][8] divided into 2×2 tiles of size 4×4:

Block (0,0) processes:    Block (1,0) processes:    Block (0,1) processes:    Block (1,1) processes:
M[0..3][0..3]             M[0..3][4..7]             M[4..7][0..3]             M[4..7][4..7]

Each block:
  - Declares __shared__ float tile[4][4]
  - Loads 4×4 = 16 elements from global memory into shared memory
  - Computes on the 16 elements using fast SRAM
  - Writes results back to global memory
```

### The Arithmetic Intensity Benefit

Without tiling: every FMA operation reads from HBM (~400 cycles latency).

With tiling and shared memory reuse:
```formula
HBM reads per element = Total HBM bytes / (Tile size × reuse count)
For GEMM with TILE=32:
  - HBM reads = M*K + K*N instead of M*N*K (K-fold reduction in reads)
  - Arithmetic intensity increases from 0.17 to ~11 FLOPs/Byte
```

---

## 2. Computing Tile Offsets: The Math

For a tile of size `TILE_H × TILE_W` in a matrix of size `Rows × Cols`:

```cpp
// Thread block (bx, by) handles the tile at:
int tile_row_start = blockIdx.y * TILE_H;   // Row offset of this tile's top-left corner
int tile_col_start = blockIdx.x * TILE_W;   // Col offset of this tile's top-left corner

// Thread (tx, ty) within the block handles the element at:
int global_row = tile_row_start + threadIdx.y;
int global_col = tile_col_start + threadIdx.x;
int flat_idx   = global_row * Cols + global_col;
```

Grid sizing: the grid must have enough blocks to cover the entire matrix:
```cpp
dim3 block(TILE_W, TILE_H);                           // Threads per block
dim3 grid((Cols + TILE_W - 1) / TILE_W,               // Blocks to cover all columns
          (Rows + TILE_H - 1) / TILE_H);              // Blocks to cover all rows
```

The `+TILE-1` in the grid calculation is the ceiling division. It ensures that even when `Cols` is not divisible by `TILE_W`, there is still a block covering the last partial column. The kernel then bounds-checks before writing.

---

## 3. Partial Tiles: Bounds Checking

When the matrix dimensions are not exact multiples of the tile size, the rightmost column of blocks and the bottom row of blocks process partial tiles. Threads within these boundary blocks that fall outside the matrix bounds must be masked:

```cpp
__global__ void tiled_scale_kernel(float* M, float scale, int Rows, int Cols) {
    __shared__ float tile[TILE_H][TILE_W];

    int global_row = blockIdx.y * TILE_H + threadIdx.y;
    int global_col = blockIdx.x * TILE_W + threadIdx.x;

    // LOAD: Guard against out-of-bounds access
    if (global_row < Rows && global_col < Cols) {
        tile[threadIdx.y][threadIdx.x] = M[global_row * Cols + global_col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;   // Pad with zero for boundary tiles
    }
    __syncthreads();

    // COMPUTE: Process the tile (even boundary tiles are filled with zeros, so no special case needed)
    float result = tile[threadIdx.y][threadIdx.x] * scale;

    // STORE: Guard against out-of-bounds write
    if (global_row < Rows && global_col < Cols) {
        M[global_row * Cols + global_col] = result;
    }
}
```

---

## 4. Padding: Aligning to Hardware Boundaries

### Why Padding Is Necessary

GPU memory controllers fetch data in 128-byte aligned chunks. If a row of data is, say, 128 bytes wide (32 floats), row 0 starts at byte 0, row 1 starts at byte 128 — perfectly aligned. But if the row is 140 bytes wide (35 floats), row 1 starts at byte 140, which straddles two 128-byte sectors. The memory controller must fetch both sectors to read row 1, wasting bandwidth.

**Padded row width** (pitch): Round the row width up to the nearest multiple of the alignment boundary (typically 128 bytes or 64 bytes).

```text
Without padding (35 floats = 140 bytes per row):
Row 0: [140 bytes of data] ← starts at byte 0, spans sectors [0-127] and [128-255]
Row 1: [140 bytes of data] ← starts at byte 140, spans sectors [128-255] and [256-383]
→ Every row access requires 2 sector fetches!

With padding (36 floats = 144 bytes, padded to 160 bytes = 40 floats = nearest 128-byte boundary... or more practically, padded to 160 = 5 * 32-byte cache lines):
Row 0: [35 useful floats][1 padding float]   = 36 floats = 144 bytes → padded to 160 bytes
Row 1: starts at byte 160 → aligned to 32-byte boundary → 1 sector per access
```

### CUDA's `cudaMallocPitch` API

```cpp
float* d_matrix;
size_t pitch;  // Pitch in bytes (padded row width)

// Automatically pads each row to a hardware-optimal alignment:
cudaMallocPitch(&d_matrix, &pitch, Cols * sizeof(float), Rows);
// pitch >= Cols * sizeof(float), rounded up to alignment boundary

// Accessing element [row][col] in a pitched allocation:
float* row_ptr = reinterpret_cast<float*>(
    reinterpret_cast<char*>(d_matrix) + row * pitch);
float val = row_ptr[col];   // or: row_ptr[col] = val;

// Equivalently (as a single flat index with pitch/sizeof(float) as row stride):
size_t row_stride = pitch / sizeof(float);
float val = d_matrix[row * row_stride + col];
```

### Manual Padding

If you allocate with `cudaMalloc` directly, pad the width yourself:
```cpp
// Round Cols up to the nearest multiple of 32 (128-byte = 32-float alignment):
size_t padded_cols = ((Cols + 31) / 32) * 32;
cudaMalloc(&d_matrix, Rows * padded_cols * sizeof(float));
// Now each row starts at a 128-byte aligned address.
```

---

## 5. Shared Memory Padding for Bank Conflict Avoidance

As covered in Chapter 3.3, shared memory is also padded — but for a different reason. Shared memory is divided into 32 banks. If each row of a 2D shared memory array is 32 words wide, all elements in the same column map to the same bank → 32-way conflict.

```cpp
// Column access conflict (Chapter 3.3 revisit):
__shared__ float tile[32][32];  // Row width = 32 = bank count → column conflict
float v = tile[threadIdx.x][0]; // All threads hit bank 0!

// Fix: pad to 33 columns
__shared__ float tile[32][33];  // Row width = 33, prime relative to 32 banks
float v = tile[threadIdx.x][0]; // Thread i hits bank (i*33)%32 → all different!
```

---

## 6. Subgrid Sliding Window: The GEMM Tiling Loop

In matrix multiplication `C = A × B`, the tiles of `C` require contributions from tiles across the K dimension. The key is the **K-dimension sliding tile window**:

```cpp
// Each thread block computes one TILE×TILE tile of C:
// Accumulates partial dot products as it slides across K:

__global__ void tiled_matmul(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    const int TILE = 32;
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // C row
    int col = blockIdx.x * TILE + threadIdx.x;  // C column
    float acc = 0.0f;

    // Slide the tile window across K dimension:
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;  // Column in A for this tile
        int b_row = t * TILE + threadIdx.y;  // Row in B for this tile

        // Collaborative load of tile from A (each thread loads one element):
        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                         ? A[row * K + a_col] : 0.0f;
        // Collaborative load of tile from B:
        s_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                         ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Inner product of tile row (from s_A) and tile column (from s_B):
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();   // Prevent overwriting tiles before all threads finish computing
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
```

### Why Two `__syncthreads()` Are Required

The first `__syncthreads()` after loading ensures all threads have finished writing to shared memory before any thread starts reading from it.

The second `__syncthreads()` after computing ensures all threads have finished reading from the tiles before any thread loads new data into the same shared memory locations (next loop iteration).

Removing either barrier causes data races — some threads will read stale or partially-overwritten tile data.

---

## 7. Summary: Tiling and Padding Mental Model

| Concept | Purpose | Mechanism |
| :--- | :--- | :--- |
| Tile | Divide work for block-level caching | Thread block handles TILE×TILE submatrix |
| Tile offset | Map block ID to matrix position | `global_pos = block_idx * TILE + thread_idx` |
| Partial tile | Handle non-divisible matrix sizes | Bounds-check every load/store |
| Global memory padding | Align row starts to 128-byte boundary | `pitch = round_up(cols * sizeof(T), 128)` |
| Shared memory padding | Prevent bank conflicts in column access | Add 1 dummy element per row: `[TILE][TILE+1]` |
| Sliding K-tile | Accumulate GEMM dot product in SRAM | Loop over K with __syncthreads() barriers |
