# 7.1 Stage 1: Elementwise Operations, Indexing & Reshaping

## 1. Overview & Connection to LLM Architecture

In a transformer architecture (LLaMA / Mistral / DeepSeek), elementwise and indexing kernels form the glue between linear projection matrix multiplications:
1. **Multi-Head Tensor Permutation (`reshape_3d`, `reshape_4d`)**:
   Converts between projection output shape `[Batch, Seq_Len, N_Heads, Head_Dim]` and attention score shape `[Batch, N_Heads, Seq_Len, Head_Dim]`.
2. **Residual Connections (`residual_add`)**:
   Implements skip connections $h \leftarrow h + \text{attn\_out}$ and $h \leftarrow h + \text{ffn\_out}$ in-place on GPU to prevent CPU-GPU roundtrips.
3. **Gated Feed-Forward Activation (`swiglu_forward`, `swiglu_backward`)**:
   Applies the SwiGLU activation $y = \text{SiLU}(g) \odot u$ and computes analytical backpropagation gradients $\nabla_g L$ and $\nabla_u L$.

---

## 2. Multi-Dimensional Index Decomposition

Given flat linear thread ID `gid` for a 4D tensor with shape $[B, H, S, D]$:
```cpp
uint32_t tmp = gid;
uint32_t d = tmp % D; tmp /= D;
uint32_t s = tmp % S; tmp /= S;
uint32_t h = tmp % H; tmp /= H;
uint32_t b = tmp;
```
To transpose from $[B, H, S, D]$ to $[B, S, H, D]$:
```cpp
uint32_t src_idx = (b * H * S + h * S + s) * D + d;
uint32_t dst_idx = (b * S * H + s * H + h) * D + d;
dst[dst_idx] = src[src_idx];
```

---

## 3. Mathematical Derivations: SwiGLU Forward & Backward

### Forward Pass:
$$\text{SiLU}(g) = g \cdot \sigma(g) = \frac{g}{1 + e^{-g}}$$
$$\text{SwiGLU}(g, u) = \text{SiLU}(g) \cdot u$$

### Backward Pass:
Given incoming gradient $\delta = \frac{\partial L}{\partial y}$:

1. **Gradient w.r.t. Up projection ($u$)**:
   $$\frac{\partial L}{\partial u} = \delta \cdot \frac{\partial y}{\partial u} = \delta \cdot \text{SiLU}(g)$$

2. **Gradient w.r.t. Gate projection ($g$)**:
   Using the derivative of $\text{SiLU}(g)$:
   $$\frac{d}{dg} \text{SiLU}(g) = \frac{d}{dg} \left( g \cdot \sigma(g) \right) = \sigma(g) + g \cdot \sigma'(g)$$
   Since $\sigma'(g) = \sigma(g)(1 - \sigma(g))$:
   $$\frac{d}{dg} \text{SiLU}(g) = \sigma(g) \left[ 1 + g (1 - \sigma(g)) \right]$$
   $$\frac{\partial L}{\partial g} = \delta \cdot u \cdot \sigma(g) \left[ 1 + g (1 - \sigma(g)) \right]$$
