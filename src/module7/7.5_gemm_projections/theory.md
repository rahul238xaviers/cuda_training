# Module 7.5 — GEMM & Linear Projection Suite

## 1. Architectural Motivation: GEMM in LLM Transformers

In modern Large Language Models (LLaMA, Mistral, Gemma, GPT), over **70% of total training FLOPs** are spent inside General Matrix Multiplications (GEMM). Every transformer layer consists of linear projections:
1. **Attention Projections**:
   - Query projection: $Q = X \cdot W_q$ ($[B \cdot S, D] \times [D, H_q \cdot D_h]$)
   - Key projection: $K = X \cdot W_k$ ($[B \cdot S, D] \times [D, H_{kv} \cdot D_h]$)
   - Value projection: $V = X \cdot W_v$ ($[B \cdot S, D] \times [D, H_{kv} \cdot D_h]$)
   - Output projection: $O = \text{AttnOut} \cdot W_o$ ($[B \cdot S, H_q \cdot D_h] \times [H_q \cdot D_h, D]$)
2. **Feed-Forward Network (FFN / MLP)**:
   - Gate projection: $G = X \cdot W_{\text{gate}}$ ($[B \cdot S, D] \times [D, D_{\text{ffn}}]$)
   - Up projection: $U = X \cdot W_{\text{up}}$ ($[B \cdot S, D] \times [D, D_{\text{ffn}}]$)
   - Down projection: $Y = \text{SwiGLU}(G, U) \cdot W_{\text{down}}$ ($[B \cdot S, D_{\text{ffn}}] \times [D_{\text{ffn}}, D]$)
3. **Backward Gradient GEMMs**:
   - Activation gradient: $dX = dO \cdot W^T$ ($[M, N] \times [K, N]^T = [M, K]$)
   - Weight gradient: $dW = X^T \cdot dO$ ($[K, M]^T \times [K, N] = [M, N]$)

Understanding how to structure high-performance CUDA kernels for these 7 specific projection kernels is foundational for any high-performance LLM training system.

---

## 2. 2D Block Tiling with Shared Memory

A naive matrix multiplication:
$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} B_{k, j}$$
performs $2 \cdot M \cdot N \cdot K$ arithmetic operations for $3 \cdot M \cdot N$ memory reads/writes. In naive GEMM, each element is re-read from DRAM $O(N)$ or $O(M)$ times, crippling performance due to DRAM bandwidth saturation.

### The Tiling Principle ($BM \times BN \times BK$)
To achieve compute-bound performance:
1. Divide matrix $C [M \times N]$ into tiles of size $BM \times BN$.
2. Each thread block computes one $BM \times BN$ tile of $C$.
3. The block iterates over the $K$ dimension in chunks of size $BK$:
   - Load tile $A_{\text{tile}} [BM \times BK]$ into shared memory `sh_A`.
   - Load tile $B_{\text{tile}} [BK \times BN]$ into shared memory `sh_B`.
   - Synchronize (`__syncthreads()`).
   - Accumulate the partial dot products in thread registers.
   - Synchronize (`__syncthreads()`).

```
Matrix A [M x K]              Matrix B [K x N]
+-------------------+         +-----------+-------+
|                   |         |           | BK    |
|   sh_A [BM x BK]  |    x    |           | x     |  --> Thread Block [BM x BN]
|                   |         |           | BN    |
+-------------------+         +-----------+-------+
                                        |
                                        v
                              Matrix C [M x N]
                              +-----------+-------+
                              |           |       |
                              |           |BM x BN|
                              |           |       |
                              +-----------+-------+
```

### Bank-Conflict-Free Shared Memory Allocation
Shared memory in NVIDIA architectures (sm_70 through sm_90) has 32 banks of 4-byte width.
If `BK = 32`, indexing `sh_A[row][col]` or `sh_B[row][col]` with stride 32 causes all 32 threads in a warp to hit the same bank simultaneously (a 32-way bank conflict).
By adding a **padding column**:
```cuda
__shared__ float sh_A[BM][BK + 1];
__shared__ float sh_B[BK][BN + 1];
```
successive rows are offset by 1 bank, completely eliminating bank conflicts when traversing columns.

---

## 3. Transposed Access Patterns in Backpropagation

In PyTorch and custom C++ training runtimes, weights $W$ are typically stored as $[N \times K]$ (row-major: `out_features x in_features`).
During forward propagation:
$$Y [M, N] = X [M, K] \cdot W^T [K, N]$$
During backward input gradient:
$$dX [M, K] = dY [M, N] \cdot W [N, K]$$
And during backward weight gradient:
$$dW [N, K] = dY^T [N, M] \cdot X [M, K]$$

Instead of running an explicit out-of-place memory transpose kernel (which costs DRAM bandwidth and allocations), the GEMM kernel can load transposed matrices directly into shared memory:
- **Transposed B ($B^T$)**: Matrix $B$ is physically stored as $[N \times K]$. To treat it as $[K \times N]$, thread $(r, c)$ loads from `B[col * K + row]`.
- **Transposed A ($A^T$)**: Matrix $A$ is physically stored as $[K \times M]$. To treat it as $[M \times K]$, thread $(r, c)$ loads from `A[col * M + row]`.

Once loaded into shared memory `sh_B[k][n]` or `sh_A[m][k]`, the inner compute loop runs at full speed without worrying about global memory stride penalties!

---

## 4. Grouped-Query Attention (GQA) Projection Kernel

In Grouped-Query Attention (LLaMA 3, Mistral), the number of query heads $H_q$ is larger than the number of key/value heads $H_{kv}$:
$$\text{group\_size} = \frac{H_q}{H_{kv}}$$
For example, LLaMA 3 8B has $H_q = 32$ and $H_{kv} = 8$, so $\text{group\_size} = 4$. Every 4 query heads share 1 KV head.

In the GQA projection / attention GEMM kernel:
```cuda
uint q_head = blockIdx.z;
uint kv_head = q_head / group_size; // Map query head to shared KV head
```
This broadcasting mapping allows the kernel to read KV heads directly with shared cache reuse, avoiding redundant memory duplication of KV projections across heads.

---

## 5. Fused SwiGLU Projections (`gemm_ffn` and `fused_swiglu_gemm`)

In LLaMA FFN:
$$\text{SwiGLU}(G, U) = \text{Swish}(G) \odot U = \left( G \cdot \sigma(G) \right) \odot U = \left(\frac{G}{1 + e^{-G}}\right) \cdot U$$

### Dual FFN Projection Fusion (`gemm_ffn`)
Standard PyTorch executes:
1. $G = X \cdot W_{\text{gate}}$ (GEMM -> write $G$ to DRAM)
2. $U = X \cdot W_{\text{up}}$ (GEMM -> write $U$ to DRAM)
3. Elementwise SwiGLU($G, U$) (Read $G, U$ from DRAM -> write output to DRAM)

By fusing the gate and up projections into a single kernel:
- Matrix $A = X [M \times K]$ is loaded into shared memory once.
- Both $W_{\text{gate}}$ and $W_{\text{up}}$ tiles are loaded into shared memory.
- Two sets of accumulator registers compute $G_{tile}$ and $U_{tile}$ simultaneously.
- When writing out to memory, threads compute $\text{SwiGLU}(G_{reg}, U_{reg})$ in register and write only the activated result!
- **Memory traffic reduction**: Eliminates writing $G$ and $U$ to DRAM and reading them back (saving 2x tensor bandwidth).

### Down-Projection Fusion (`fused_swiglu_gemm`)
Alternatively, before multiplying with $W_{\text{down}}$, we can fuse the SwiGLU activation directly as the tile is loaded into shared memory for the down-projection GEMM:
```cuda
sh_A[row][k] = swiglu(gate_tile[row][k], up_tile[row][k]);
```
This eliminates the intermediate activated tensor allocation in global memory altogether!

---

## 6. Register Micro-Tiling ($TM \times TN$)

To reach 80%+ of peak FP32 TFLOPS without Tensor Cores:
Each thread must compute a small sub-tile of size $TM \times TN$ (e.g. $4 \times 4$ or $8 \times 8$).
1. A thread block has $BM = 64, BN = 64$.
2. With $TM = 4, TN = 4$, each thread computes 16 elements.
3. Total threads per block: $(64 / 4) \times (64 / 4) = 16 \times 16 = 256$ threads.
4. In each $BK$ step:
   - Thread loads $TM$ elements from `sh_A` into registers `reg_A[TM]`.
   - Thread loads $TN$ elements from `sh_B` into registers `reg_B[TN]`.
   - Computes $TM \times TN$ multiply-accumulate operations in registers:
     ```cuda
     for (int m = 0; m < TM; ++m) {
         for (int n = 0; n < TN; ++n) {
             accum[m][n] += reg_A[m] * reg_B[n];
         }
     }
     ```
This achieves a 16:1 ratio of arithmetic instructions to shared memory loads, preventing shared memory bandwidth bottlenecks!
