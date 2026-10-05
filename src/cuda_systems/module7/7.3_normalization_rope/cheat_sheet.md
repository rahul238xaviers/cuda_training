# Topic 7.3: RMSNorm & Rotary Position Embeddings (RoPE) — Cheat Sheet

A concise reference for Root Mean Square Normalization and Rotational Position Embedding.

---

### 1. RMSNorm (Root Mean Square Normalization)
```text
RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
```
- Eliminates the mean-centering step of LayerNorm, saving 1 pass over the hidden dimension.
- Each thread block normalizes 1 token row using warp shuffle reduction for `mean(x^2)` in registers.

---

### 2. Rotary Position Embedding (RoPE)
- Encodes relative token distance by rotating pairs of coordinates in the complex plane:
  ```text
  x0_rot = x0 * cos(theta) - x1 * sin(theta)
  x1_rot = x0 * sin(theta) + x1 * cos(theta)
  ```
- **Frequency Formula**: `theta = position / (10000.0 ^ (2 * i / dim))`.
- Applied directly to Query and Key tensors before computing attention dot-products.

---

### 3. LLM Systems Context
- Foundation of modern architectures (LLaMA 3, Mistral, Gemma 2, DeepSeek).
