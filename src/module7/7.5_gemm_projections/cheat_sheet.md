# Topic 7.5: GEMM & Linear Projections Suite — Cheat Sheet

A concise reference for QKV projection, FFN SwiGLU fusion, and leading dimension math.

---

### 1. Fused QKV Projection
- Instead of launching 3 separate matrix multiplications for Query, Key, and Value:
  ```text
  QKV = X * W_qkv  where W_qkv has shape [D, 3 * D]
  ```
- Reads the activation tensor `X` from DRAM exactly once and produces Q, K, V simultaneously.

---

### 2. SwiGLU Fused FFN Projection
- Multiply input activation `X` with Gate and Up weights concurrently:
  - Both tiles loaded into shared memory.
  - Compute `G_tile` and `U_tile` in registers.
  - Apply `SwiGLU(G, U)` before writing to DRAM.

---

### 3. LLM Systems Context
- GEMM computations consume over 70% of total training and inference FLOPs in transformer models.
