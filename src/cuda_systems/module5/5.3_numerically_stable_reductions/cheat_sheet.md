# Topic 5.3: Numerically Stable Online Reductions (Softmax) — Cheat Sheet

A concise reference for Online Softmax, Log-Sum-Exp, and running max/sum state tracking.

---

### 1. The Numerical Instability Problem
- Standard Softmax: `P_i = exp(x_i) / sum(exp(x_j))`.
- If `x_i = 100.0f`, `exp(100.0f)` overflows to `+inf` (NaN propagation).
- **Safe Softmax**: Subtract maximum value `m = max(x)`:
  ```text
  P_i = exp(x_i - m) / sum_j(exp(x_j - m))
  ```

---

### 2. 3-Pass vs. 1-Pass Online Softmax
- **Classic 3-Pass Softmax**:
  1. Pass 1: Find `m = max(x)` across all elements.
  2. Pass 2: Compute `d = sum(exp(x_i - m))`.
  3. Pass 3: Normalize `out_i = exp(x_i - m) / d`.
  - Flaw: Requires 3 trips to memory!
- **Online Softmax (Milakov & Gimelshein / FlashAttention)**:
  - Fuses Pass 1 and Pass 2 into a single streaming loop using running correction:
  ```cpp
  float m_prev = m;
  m = fmaxf(m_prev, x_i);
  d = d * expf(m_prev - m) + expf(x_i - m);
  ```

---

### 3. LLM Systems Context
- **FlashAttention Core**: The mathematical foundation that allows streaming attention without storing the sequence-by-sequence attention matrix in DRAM.
