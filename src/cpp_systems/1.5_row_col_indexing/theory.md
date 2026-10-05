# Chapter 1.5: Row/Column-Major Indexing & Coordinate-to-Flat Mapping

Every CUDA kernel that processes a 2D tensor — an attention matrix, a weight matrix, a convolution feature map — must answer one question: given the thread's logical coordinates `(row, col)`, what is the flat memory address to read or write?

This mapping is the single most executed computation in all of GPU deep learning. Getting it wrong causes incorrect output. Getting it slow (strided access, non-coalesced reads) causes 10x-32x performance loss.

---

## 1. The Physical Reality: Flat Memory, Multi-Dimensional Logic

Physical GPU DRAM is one-dimensional: a sequence of bytes numbered 0, 1, 2, ... up to the device's total memory. There is no "row" or "column" at the hardware level.

A 2D matrix `M[Rows][Cols]` is stored as a flat array of `Rows * Cols` elements. The question is: **which order?**

---

## 2. Row-Major Layout (C / C++ / PyTorch default)

Elements of the same row are adjacent in memory. Moving along a row (incrementing `col`) moves by 1 element. Moving to the next row (incrementing `row`) moves by `Cols` elements.

```text
Matrix M[3][4] (3 rows, 4 columns):

Logical view:        Physical memory layout (row-major):
M[0][0] M[0][1] M[0][2] M[0][3]     Index:  0  1  2  3  4  5  6  7  8  9 10 11
M[1][0] M[1][1] M[1][2] M[1][3]     Value: 00 01 02 03 10 11 12 13 20 21 22 23
M[2][0] M[2][1] M[2][2] M[2][3]
                                     Address formula:
                                     flat_index = row * Cols + col
```

Verification:
- M[0][3]: `0 * 4 + 3 = 3` → index 3 ✓
- M[1][0]: `1 * 4 + 0 = 4` → index 4 ✓
- M[2][2]: `2 * 4 + 2 = 10` → index 10 ✓

### The CUDA Thread-to-Index Mapping (Row-Major)

```cpp
// Canonical 2D thread block indexing:
__global__ void kernel_2d(float* M, int Rows, int Cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Global column index

    if (row < Rows && col < Cols) {
        int flat = row * Cols + col;   // Row-major flat index
        M[flat] = some_computation(row, col);
    }
}
// Launch: dim3 block(32, 32); dim3 grid((Cols+31)/32, (Rows+31)/32);
```

**Why `threadIdx.x` drives `col`**: In CUDA, threads with adjacent `threadIdx.x` values (0, 1, 2, ...) are in the same warp. For memory coalescing, adjacent threads must access adjacent memory addresses. In row-major layout, adjacent `col` values produce adjacent memory addresses. Therefore, `threadIdx.x → col` is the correct mapping for coalesced global memory access.

---

## 3. Column-Major Layout (Fortran / MATLAB / cuBLAS)

Elements of the same column are adjacent in memory. Moving along a column (incrementing `row`) moves by 1 element. Moving to the next column (incrementing `col`) moves by `Rows` elements.

```text
Matrix M[3][4] in column-major:

Physical memory:    Index:  0  1  2  3  4  5  6  7  8  9 10 11
                    Value: 00 10 20 01 11 21 02 12 22 03 13 23

                    Address formula:
                    flat_index = col * Rows + row
```

cuBLAS and CUBLAS-XT expect column-major matrices. When interfacing PyTorch row-major tensors with cuBLAS, you must either transpose (expensive) or exploit the mathematical equivalence: `(A @ B)^T = B^T @ A^T` to avoid physically transposing.

---

## 4. The Decode Direction: Flat Index → (Row, Col)

Just as important as encoding `(row, col) → flat` is decoding: given a flat linear index, compute its row and column. This appears in:
- Parallel prefix sum where a 1D thread handles 2D output
- Kernel grid-stride loops over a 2D matrix
- Debug tools that print `(row, col)` for a given error location

```cpp
// Row-major decode:
int flat_index = ...;    // Given
int row = flat_index / Cols;    // Integer division → row
int col = flat_index % Cols;    // Modulo → column

// Example: flat=10, Cols=4
// row = 10 / 4 = 2   (integer division)
// col = 10 % 4 = 2   → M[2][2] ✓
```

This decode pattern appears in grid-stride loops:
```cpp
// Process a 2D matrix with a 1D grid-stride loop:
__global__ void grid_stride_2d(float* M, int Rows, int Cols) {
    int total = Rows * Cols;
    for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
             flat < total;
             flat += gridDim.x * blockDim.x) {
        int row = flat / Cols;   // Decode row
        int col = flat % Cols;   // Decode column
        M[flat] = process(row, col);
    }
}
```

**Note on performance**: Integer division and modulo by non-power-of-two values require `IDIV` instructions on the GPU, which are slow (20+ cycles). For power-of-two `Cols`, use bit tricks:
```cpp
// If Cols is a power of 2 (e.g. Cols = 64 = 2^6):
int row = flat >> 6;    // Equivalent to flat / 64 (1 cycle)
int col = flat & 63;    // Equivalent to flat % 64 (1 cycle)
```

---

## 5. N-Dimensional Generalization: Tensor Flat Indexing

PyTorch tensors can be 3D, 4D, or more. The same principle extends:

### 3D Tensor `T[Batch][Rows][Cols]`:

```cpp
// 3D row-major flat index:
int flat = batch * (Rows * Cols) + row * Cols + col;

// Equivalently:
// Stride for batch dimension: stride_batch = Rows * Cols
// Stride for row dimension:   stride_row   = Cols
// Stride for col dimension:   stride_col   = 1
int flat = batch * stride_batch + row * stride_row + col * stride_col;
```

### 4D Tensor `T[Batch][Heads][SeqLen][Dim]` (Common in Attention):

```cpp
// 4D row-major flat index (all inner dimensions contiguous):
int stride_batch = Heads * SeqLen * Dim;
int stride_head  = SeqLen * Dim;
int stride_seq   = Dim;
int stride_dim   = 1;

int flat = b * stride_batch + h * stride_head + s * stride_seq + d * stride_dim;
float val = tensor[flat];
```

In CUDA kernels, each dimension's stride is typically precomputed and passed as kernel arguments to avoid recomputation inside the kernel.

---

## 6. Column Access: The GPU Performance Trap

When a kernel needs to access all elements of a **column** in a row-major matrix (e.g., computing the mean of each column in a matrix normalization), adjacent threads access elements with stride `Cols`:

```cpp
// POOR: Column-wise access by adjacent threads (strided, uncoalesced)
__global__ void column_mean_naive(const float* M, float* col_means, int Rows, int Cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Each thread handles one column
    if (col < Cols) {
        float sum = 0.0f;
        for (int row = 0; row < Rows; ++row)
            sum += M[row * Cols + col];   // Stride Cols between accesses = uncoalesced!
        col_means[col] = sum / Rows;
    }
}
```

Adjacent threads (col=0, col=1, col=2...) access `M[0*Cols+0]`, `M[0*Cols+1]`, `M[0*Cols+2]` — these are adjacent addresses, so the first row load is coalesced. But on the second iteration: `M[1*Cols+0]`, `M[1*Cols+1]`... also adjacent — coalesced again! The access pattern is actually fine here.

The problem appears when one thread accesses the entire column (each thread sequentially strides through Rows). This is not a coalescing issue but a **cache line thrashing** issue for large matrices.

Better pattern for column statistics: transpose first (with shared memory) then reduce:
```cpp
// BETTER: Load tile into shared memory (transposing as we go), then reduce contiguously
// See Chapter 3.3 for the bank-conflict-free transpose + shared memory pattern.
```

---

## 7. Applied Pattern: Attention Score Indexing

Multi-head attention computes `scores[b][h][q][k]` = dot product of query `q` with key `k` for batch `b` and head `h`.

```cpp
__global__ void attention_scores_kernel(
    const float* Q,     // [B, H, S, D]
    const float* K,     // [B, H, S, D]
    float* scores,      // [B, H, S, S]
    int B, int H, int S, int D)
{
    // Grid: (B, H, S) — one thread computes one row of one attention head
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x;   // Query position

    // Stride computation for Q/K (4D tensors):
    int q_base = ((b * H + h) * S + q) * D;

    float scores_row[MAX_S];  // Local accumulator per thread

    for (int k = 0; k < S; ++k) {
        int k_base = ((b * H + h) * S + k) * D;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            dot += Q[q_base + d] * K[k_base + d];
        dot = block_reduce_sum(dot);   // Sum across D dimension
        if (threadIdx.x == 0)
            scores[((b * H + h) * S + q) * S + k] = dot / sqrtf((float)D);
    }
}
```

---

## 8. Summary: Row/Column Indexing Reference

| Operation | Formula | Notes |
| :--- | :--- | :--- |
| 2D row-major encode | `flat = row * Cols + col` | Standard C/C++/PyTorch |
| 2D col-major encode | `flat = col * Rows + row` | cuBLAS/Fortran |
| 2D row-major decode | `row = flat / Cols; col = flat % Cols` | Use bit ops for power-of-2 |
| 3D tensor access | `flat = b*R*C + r*C + c` | Product of outer strides |
| Thread mapping | `threadIdx.x → col` | Coalesced warp reads |
| Block mapping | `blockIdx.x → col tiles` | Standard 2D grid |
