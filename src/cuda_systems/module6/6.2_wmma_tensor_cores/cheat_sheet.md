# Topic 6.2: Tensor Cores & WMMA API — Cheat Sheet

A concise reference for hardware Tensor Cores using Warp Matrix Multiply and Accumulate.

---

### 1. What are Tensor Cores?
- Specialized execution units inside SMs that perform mixed-precision matrix multiply-accumulate:
  ```text
  D = A * B + C
  ```
  in a single clock cycle across a warp of 32 threads.

---

### 2. The WMMA (Warp Matrix Multiply Accumulate) Workflow
```cpp
#include <mma.h>
using namespace nvcuda;

// 1. Declare matrix fragments (16x16x16 tile)
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 2. Initialize accumulator to zero
wmma::fill_fragment(c_frag, 0.0f);

// 3. Load tiles from Shared Memory or DRAM
wmma::load_matrix_sync(a_frag, sh_A, stride_a);
wmma::load_matrix_sync(b_frag, sh_B, stride_b);

// 4. Perform warp matrix multiply
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 5. Store result tile back to memory
wmma::store_matrix_sync(d_out, c_frag, stride_out, wmma::mem_row_major);
```

---

### 3. LLM Systems Context
- **Peak FLOP Utilization**: Tensor Cores provide up to 10x higher TFLOPs than standard CUDA cores (e.g. 312 TFLOPs BF16 on A100).
