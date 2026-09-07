# 4.2 Shared Memory Matrix Tiling & Register Micro-Kernels

## 1. The Memory Wall & Arithmetic Intensity

General Matrix Multiply (GEMM) computes:
$$C_{M \times N} = A_{M \times K} \cdot B_{K \times N}$$
- Total FLOPs: $2 \times M \times N \times K$
- Total Data elements: $M \times K + K \times N + M \times N$

In a naive GEMM where each thread computes 1 dot product:
- Each thread reads an entire row of $A$ and column of $B$ directly from Global Memory.
- Global DRAM reads: $2 \times M \times N \times K \times 4$ bytes.
- This saturates memory bandwidth immediately, running at $< 5\%$ of GPU compute capacity!

---

## 2. 2D Block Tiling with Shared Memory

To break the memory wall, we decompose matrices into **Tiles** that fit into on-chip Shared Memory:
1. Divide matrix $C$ into tiles of size $BM \times BN$ (e.g. $64 \times 64$).
2. Assign one Thread Block to compute each $BM \times BN$ tile of $C$.
3. Slide a window of size $BK$ along the $K$ dimension:
   - Load tile $A_{sub}$ ($BM \times BK$) into `__shared__ float s_A[BM][BK]`.
   - Load tile $B_{sub}$ ($BK \times BN$) into `__shared__ float s_B[BK][BN]`.
   - `__syncthreads()`.
   - All threads compute partial matrix products from shared memory.
   - `__syncthreads()`.
4. Accumulate partial dot products over all $K / BK$ steps before writing to DRAM once.

Global Memory reads are reduced by a factor of $BK$!

---

## 3. Register Micro-Tiling ($TM \times TN$)

Shared memory bandwidth can itself become a bottleneck if every thread reads from shared memory for every single scalar FMA.
To achieve peak performance:
- Each thread in the block computes a small $TM \times TN$ sub-matrix (e.g. $4 \times 4 = 16$ elements) entirely in **registers**!
- In each step of $BK$:
  - Load $TM$ elements from `s_A` into thread registers.
  - Load $TN$ elements from `s_B` into thread registers.
  - Perform $TM \times TN$ FMAs:
    ```cpp
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            reg_c[m][n] += reg_a[m] * reg_b[n];
        }
    }
    ```
- Compute-to-SRAM load ratio is $\frac{TM \times TN}{TM + TN} = \frac{16}{8} = 2.0$!
This is the core design principle behind CUTLASS and production LLM linear projection kernels.
