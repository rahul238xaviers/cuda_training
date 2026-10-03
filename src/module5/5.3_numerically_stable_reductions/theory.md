# 5.3 Numerically Stable Reductions: Log-Sum-Exp & Online Softmax

## 1. Floating-Point Limits & Catastrophic Overflow

In standard IEEE 754 32-bit floating point (FP32):
- Largest representable finite number: `~= 3.4 x 10^{38}`.
- If `x > 88.7228`, computing `exp(x)` yields `+infty` (**Overflow**)!
- If `x < -87.3365`, computing `exp(x)` yields `0.0` (**Underflow**).

In LLMs, pre-softmax logits `z` routinely reach values of `100` or larger during training initialization. A naive implementation:
```cpp
float p = expf(z[target]) / sum_exp; // expf(100.0) = INF -> INF / INF = NaN!
```
This produces `NaN` gradients and crashes training instantly!

---

## 2. The Log-Sum-Exp (LSE) Trick

To guarantee numerical stability, we subtract the maximum logit `m = max_j(z_j)` before exponentiating:

```text
LSE(z) = m + log <=ft( sum(j=1) )
```

Since `z_j - m <= 0` for all `j`:
- Every exponent is bounded in `(0.0, 1.0]`.
- Overflows are mathematically impossible.
- Softmax cross-entropy loss simplifies to:
  
```text
mathcal{L} = -z_{target} + m + log <=ft( sum(j=1) )
```


---

## 3. The Online Softmax Algorithm

Standard Softmax requires 3 sequential passes over the vector:
1. Pass 1: Find `m = max_i(z_i)`
2. Pass 2: Compute `d = sum_i e^{z_i - m}`
3. Pass 3: Normalize `p_i = frac{e^{z_i - m}}{d}`

**The Online Softmax Algorithm (Milakov & Gimelshteyn, 2018)** combines these into a single pass by updating running max `m` and running denominator `d`:
When encountering a new chunk with local max `m_{curr}`:

```text
m_{new} = max(m_{old}, m_{curr})
```


```text
d_{new} = d_{old} * e^{m_{old} - m_{new}} + sum e^{x_i - m_{new}}
```

This online rescaling factor `e^{m_{old} - m_{new}}` is the mathematical engine powering **FlashAttention**!
