# Topic 7.6: FlashAttention Suite (Forward & Backward) — Cheat Sheet

A concise reference for IO-aware tiled online softmax attention without NxN HBM materialization.

---

### 1. FlashAttention Core Philosophy
- Standard Attention writes intermediate `S x S` matrix to HBM ($O(S^2)$ memory).
- FlashAttention tiles Q, K, V into on-chip SRAM blocks and streams Online Softmax in registers ($O(S)$ memory).

---

### 2. FlashAttention-2 Forward Loop Inversion
- **Outer Loop**: Iterates over Query blocks `Q_i` in SRAM.
- **Inner Loop**: Iterates over Key/Value blocks `K_j, V_j` in SRAM.
- Each thread block maintains output accumulator `O_i` and running softmax statistics `m_i, l_i` completely in registers.

---

### 3. Online Softmax Rescaling Formula
```text
m_new = max(m_prev, max(S_tile))
l_new = l_prev * exp(m_prev - m_new) + sum(exp(S_tile - m_new))
O_new = O_prev * exp(m_prev - m_new) + P_tile * V_tile
```

---

### 4. Causal Masking Optimization
- **Full Compute Block**: If `col_max <= row_min`, entire block is valid, no masking needed.
- **Skip Block**: If `col_min > row_max`, entire block is in causal future, skip loop iteration entirely (saves 50% FLOPs!).
- **Diagonal Block**: Apply elementwise masking where `col_idx > row_idx`.

---

### 5. LLM Systems Context
- Enables training and inference with 32K, 128K, and 1M context windows on modern GPUs.
