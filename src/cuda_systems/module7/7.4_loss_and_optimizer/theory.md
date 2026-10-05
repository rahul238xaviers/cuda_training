# 7.4 Stage 4: Loss Functions & AdamW Optimizer

## 1. Fused Cross-Entropy Loss & Logits Gradient

In autoregressive language modeling, predicting the next token requires computing the cross-entropy between predicted unnormalized logits `z in tensor[V]` (`V ~= 32k` to `128k`) and target integer class `y in {0, ..., V-1}`.

### Mathematical Formulation:
1. **Softmax Probability**:
   
```text
p_i = frac{e^{z_i - max_k(z_k)}}{sum(j=0) e^{z_j - max_k(z_k)}}
```

2. **Cross-Entropy Loss**:
   
```text
mathcal{L} = -log(p_y) = - (z_y - max_k(z_k)) + log <=ft( sum(j=0) e^{z_j - max_k(z_k)} )
```

3. **Analytical Logits Gradient**:
   
```text
frac{partial mathcal{L}}{partial z_i} = p_i - mathbb{I}[i = y]
```

   where `mathbb{I}[i = y] = 1` if `i` is the target token, and `0` otherwise.

### The 3-Pass Block Reduction Architecture
Because `V gg 1024`, a single threadblock cooperatively reduces across vocabulary columns:
* **Pass 1 (Find Max)**: Parallel max reduction into shared memory to compute `M = max_i(z_i)`.
* **Pass 2 (Sum Exponentials)**: Parallel sum reduction of `sum e^{z_i - M}`.
* **Pass 3 (Loss Accumulation & Gradient)**:
  - Thread 0 computes `mathcal{L}_{token} = -log(p_y) / T` and calls `atomicAdd(loss_out, loss_val)`.
  - All threads write normalized gradients: `grad[i] = (p_i - mathbb{I}[i = y]) / T`.

---

## 2. Decoupled Weight Decay (AdamW)

AdamW (Loshchilov & Hutter, 2017) decouples `L_2` regularization / weight decay from gradient moment updates:

1. **First Moment (Momentum)**:
   
```text
m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
```

2. **Second Moment (RMSprop)**:
   
```text
v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
```

3. **Bias Corrections**:
   
```text
m_hat_t = (m_t) / (1 - beta_1^t),   v_hat_t = (v_t) / (1 - beta_2^t)
```

4. **Decoupled Weight Decay & Parameter Update**:
   
```text
theta_t = theta_{t-1} - eta * lambda theta_{t-1} - frac{eta m_hat_t}{sqrt(v)_t_hat + eps}
```

where `eta` is learning rate and `lambda` is weight decay.

### Fusion Advantage
Instead of launching 6 sequential CUDA kernels for `m, v, m_hat, v_hat, decay, theta`, fusing them into a single 1D thread grid performs all operations in registers with a single DRAM read and write per parameter element.
