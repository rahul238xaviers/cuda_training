# 6.2 Tensor Cores & The WMMA (Warp Matrix Multiply and Accumulate) API

## 1. Tensor Cores Architecture

Starting with the Volta (sm_70) and Turing (sm_75) architectures and continuing through Ampere (sm_80), Hopper (sm_90), and Blackwell, NVIDIA GPUs feature dedicated mixed-precision matrix arithmetic units called **Tensor Cores**.

Unlike standard FP32/FP64 CUDA cores which execute 1 scalar operation per thread per cycle, Tensor Cores execute an entire matrix multiply-accumulate across a 32-thread warp in specialized hardware pipelines:
$$D = A \times B + C$$

---

## 2. The CUDA WMMA API

The CUDA C++ WMMA API is defined in `<mma.h>` under the `nvcuda::wmma` namespace.

```cpp
#include <mma.h>
using namespace nvcuda;
```

### Key Primitive: Fragments
A **fragment** is an opaque C++ struct distributed across the 32 threads in a warp. Each thread holds a small, private subset of the matrix elements in its registers:

```cpp
// Matrix A: 16x16, half precision, row-major
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

// Matrix B: 16x16, half precision, col-major
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

// Accumulator C: 16x16, float precision
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

---

## 3. WMMA Pipeline Operations

All WMMA operations are **warp-synchronous** (indicated by the `_sync` suffix), meaning all 32 threads in the warp must execute them collectively with identical arguments:

### 1. Initialization:
```cpp
wmma::fill_fragment(c_frag, 0.0f);
```

### 2. Loading from Memory (Global or Shared Memory):
```cpp
// lda / ldb is the leading dimension (stride in elements between consecutive rows/cols)
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
```

### 3. Compute (MMA):
```cpp
// Computes c_frag = a_frag * b_frag + c_frag
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### 4. Writeback to Memory:
```cpp
// Store accumulator back to row-major or col-major memory
wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
```

---

## 4. Element Access and In-Register Scaling
Threads can access and modify individual elements in their local fragment slices using `.num_elements` and `.x[i]`:

```cpp
// Example: Scale accumulator fragment elements before storing (e.g. attention 1/sqrt(d_k))
for (int t = 0; t < c_frag.num_elements; ++t) {
    c_frag.x[t] *= scale;
}
```
