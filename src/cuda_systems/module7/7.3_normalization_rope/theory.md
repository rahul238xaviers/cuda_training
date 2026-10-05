# 7.3 Stage 3: Normalization & Rotary Position Embeddings (RoPE)

## 1. RMSNorm (Root Mean Square Normalization)

In modern LLMs (LLaMA, Gemma, DeepSeek, Mistral), **RMSNorm** replaces standard LayerNorm by omitting the mean-centering step, saving `~= 30%` compute and memory bandwidth without impacting convergence.

### Forward Equations:

```text
RMS(x) = sqrt((1) / (d) sum(i=1)^d x_i^2 + eps)
```


```text
y_i = (x_i) / (RMS(x)) * gamma_i = x_hat_i * gamma_i
```

where `x_hat_i = (x_i) / (RMS(x))` is the normalized activation and `gamma_i` is the learnable affine scale weight.

### Backward Equations:
Given incoming gradient `delta_i = (partial L) / (partial y_i)`:
1. **Weight Gradient**:
   
```text
(partial L) / (partial gamma_i) = sum(rows) * x_hat_{r, i}
```

2. **Input Gradient**:
   
```text
Let  S = (1) / (d) sum(j=1)_j
```

   
```text
(partial L) / (partial x_i) = (1) / (RMS(x)) <=ft( delta_i * gamma_i - x_hat_i * S )
```


---

## 2. Rotary Position Embeddings (RoPE)

RoPE encodes relative token position by applying a 2D rotation to pairs of features in the attention Query (`Q`) and Key (`K`) representations.

### Forward Rotation:
For feature pair `(x_{2i}, x_{2i+1})` at token position `s`:

```text
begin{pmatrix} x_{2i}'  x_{2i+1}' end{pmatrix} = begin{pmatrix} cos theta_{s, i} & -sin theta_{s, i}  sin theta_{s, i} & cos theta_{s, i} end{pmatrix} begin{pmatrix} x_{2i}  x_{2i+1} end{pmatrix}
```


```text
x_{2i}' = x_{2i} cos theta - x_{2i+1} sin theta
```


```text
x_{2i+1}' = x_{2i} sin theta + x_{2i+1} cos theta
```


### Backward Rotation:
Since the rotation matrix is orthogonal (`R^{-1} = R_transposed`), backpropagation through RoPE multiplies by the transposed matrix:

```text
x_{2i}' = x_{2i} cos theta + x_{2i+1} sin theta
```


```text
x_{2i+1}' = x_{2i+1} cos theta - x_{2i} sin theta
```


---

## 3. Kernel Fusion (`fused_add_norm`)

In a standard transformer layer, residual addition (`h <=ftarrow h + attn_out`) is followed immediately by pre-layer normalization.
Materializing `h` back to DRAM before reading it again for RMSNorm wastes enormous memory bandwidth.

**Fused Add-Norm Execution**:
1. Load `h` and `attn_out` into registers.
2. In-place add `h_{new} = h + attn_out` and store `h_{new}` back to global memory (for backward).
3. Simultaneously accumulate `h_{new}^2` in warp registers.
4. Perform warp shuffle reduction to compute RMS.
5. Multiply `h_{new} * RMS * gamma` and write normalized output to global memory in a single kernel launch.
