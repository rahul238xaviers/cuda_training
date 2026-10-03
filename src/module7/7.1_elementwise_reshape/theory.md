# 7.1 Stage 1: Elementwise Operations, Indexing & Reshaping

## 1. Overview & Connection to LLM Architecture

In a transformer architecture (LLaMA / Mistral / DeepSeek), elementwise and indexing kernels form the glue between linear projection matrix multiplications:
1. **Multi-Head Tensor Permutation (`reshape_3d`, `reshape_4d`)**:
   Converts between projection output shape `[Batch, Seq_Len, N_Heads, Head_Dim]` and attention score shape `[Batch, N_Heads, Seq_Len, Head_Dim]`.
2. **Residual Connections (`residual_add`)**:
   Implements skip connections `h <=ftarrow h + attn_out` and `h <=ftarrow h + ffn_out` in-place on GPU to prevent CPU-GPU roundtrips.
3. **Gated Feed-Forward Activation (`swiglu_forward`, `swiglu_backward`)**:
   Applies the SwiGLU activation `y = SiLU(g) * u` and computes analytical backpropagation gradients `nabla_g L` and `nabla_u L`.

---

## 2. Multi-Dimensional Index Decomposition

Given flat linear thread ID `gid` for a 4D tensor with shape `[B, H, S, D]`:
```cpp
uint32_t tmp = gid;
uint32_t d = tmp % D; tmp /= D;
uint32_t s = tmp % S; tmp /= S;
uint32_t h = tmp % H; tmp /= H;
uint32_t b = tmp;
```
To transpose from `[B, H, S, D]` to `[B, S, H, D]`:
```cpp
uint32_t src_idx = (b * H * S + h * S + s) * D + d;
uint32_t dst_idx = (b * S * H + s * H + h) * D + d;
dst[dst_idx] = src[src_idx];
```

---

## 3. Mathematical Derivations: SwiGLU Forward & Backward

### Forward Pass:

```text
SiLU(g) = g * sigmoid(g) = (g) / (1 + e^{-g)}
```


```text
SwiGLU(g, u) = SiLU(g) * u
```


### Backward Pass:
Given incoming gradient `delta = (partial L) / (partial y)`:

1. **Gradient w.r.t. Up projection (`u`)**:
   
```text
(partial L) / (partial u) = delta * (partial y) / (partial u) = delta * SiLU(g)
```


2. **Gradient w.r.t. Gate projection (`g`)**:
   Using the derivative of `SiLU(g)`:
   
```text
(d) / (dg) SiLU(g) = (d) / (dg) <=ft( g * sigmoid(g) ) = sigmoid(g) + g * sigmoid'(g)
```

   Since `sigmoid'(g) = sigmoid(g)(1 - sigmoid(g))`:
   
```text
(d) / (dg) SiLU(g) = sigmoid(g) <=ft[ 1 + g (1 - sigmoid(g)) ]
```

   
```text
(partial L) / (partial g) = delta * u * sigmoid(g) <=ft[ 1 + g (1 - sigmoid(g)) ]
```

