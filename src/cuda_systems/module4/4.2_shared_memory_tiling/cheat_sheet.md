# Topic 4.2: Shared Memory Matrix Tiling — Cheat Sheet

A concise reference for 2D matrix tiling, register micro-kernels, and DRAM traffic reduction.

---

### 1. Arithmetic Intensity & Tiling
- **Without Tiling**: Each element of matrix A and B is loaded from high-latency DRAM for every multiply-accumulate operation. Arithmetic intensity = 0.25 FLOP/byte (severely memory bound).
- **With 2D Shared SRAM Tiling**: Thread blocks load a sub-tile of size `TILE_K x TILE_N` into on-chip SRAM once, and reuse it dozens of times across threads. Arithmetic intensity increases by TILE_DIM times!

---

### 2. Tiled GEMM Flow (Step by Step)
```cpp
__shared__ float s_a[TILE_DIM][TILE_DIM];
__shared__ float s_b[TILE_DIM][TILE_DIM];

float acc = 0.0f;
for (int t = 0; t < num_tiles; ++t) {
    // 1. Coalesced collaborative load from DRAM into SRAM
    s_a[threadIdx.y][threadIdx.x] = A[...];
    s_b[threadIdx.y][threadIdx.x] = B[...];
    __syncthreads();

    // 2. Compute partial dot product from fast on-chip SRAM
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
    }
    __syncthreads();
}
C[row * N + col] = acc;
```

---

### 3. LLM Systems Context
- **Self-Attention & Linear Layers**: Tiling is the cornerstone of FlashAttention, cuBLAS, and CUTLASS matrix multiplication engines.
