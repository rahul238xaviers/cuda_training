# Topic 7.4: Cross-Entropy Loss & Fused AdamW Optimizer — Cheat Sheet

A concise reference for loss reductions and single-pass GPU optimizer updates.

---

### 1. Fused Cross-Entropy Loss
```text
Loss = -log(Softmax(logits)[target_token]) = -(logits[target] - log(sum(exp(logits))))
```
- Fusing Log-Sum-Exp computation with target negative log-likelihood avoids materializing the full `[Batch, VocabSize]` probability matrix in DRAM.

---

### 2. Fused AdamW Optimizer Step
- **Memory Bandwidth Bound**: Standard PyTorch requires 4 separate kernel launches for parameters, gradients, momentum (m), and variance (v).
- **Fused CUDA Kernel**: Loads `w, g, m, v` once in a single pass:
  ```cpp
  float m_new = beta1 * m + (1.0f - beta1) * g;
  float v_new = beta2 * v + (1.0f - beta2) * (g * g);
  float m_hat = m_new / (1.0f - beta1_power);
  float v_hat = v_new / (1.0f - beta2_power);
  float w_new = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
  ```

---

### 3. LLM Systems Context
- **Training Throughput**: Fused AdamW saves gigabytes of redundant DRAM traffic per step during multi-billion parameter model pretraining.
