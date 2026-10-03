# Topic 2.4: Move Semantics & Device Tensor Views — Cheat Sheet

A concise reference for zero-copy tensor slicing, non-owning views, and move mechanics.

---

### 1. Non-Owning Tensor Span / View
- Wrap device memory pointers with multi-dimensional dimensions and strides without copying memory:
  ```cpp
  template <typename T>
  struct DeviceTensor2D {
      T* data;
      int rows;
      int cols;
      int stride_row;
      int stride_col;

      __host__ __device__ T& operator()(int r, int c) const {
          return data[r * stride_row + c * stride_col];
      }
  };
  ```

---

### 2. Zero-Cost Transpose via Strides
- Standard Contiguous Row-Major: `stride_row = cols`, `stride_col = 1`.
- Transposed View (Zero Copies): `stride_row = 1`, `stride_col = cols`.
- Changing view strides takes O(1) time and zero DRAM bandwidth.

---

### 3. LLM Systems Context
- **KV-Cache Slicing**: Slicing the active token sequence from an allocated continuous buffer without memory copies.
- **Multi-Head Attention Projection**: Unflattening `[Batch, SeqLen, NumHeads * HeadDim]` into 4D view `[Batch, SeqLen, NumHeads, HeadDim]`.

---

### 4. Common Pitfalls
- Dangling view pointers: Viewing a buffer after its underlying `CudaDeviceBuffer` has been deallocated.
- Non-contiguous memory coalescing: When strides are non-unit, reading elements adjacent in index space can result in uncoalesced memory transactions.
