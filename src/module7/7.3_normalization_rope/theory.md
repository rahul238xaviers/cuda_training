# 7.3 Stage 3: Normalization & Rotary Position Embeddings (RoPE)

## 1. RMSNorm (Root Mean Square Normalization)

In modern LLMs (LLaMA, Gemma, DeepSeek, Mistral), **RMSNorm** replaces standard LayerNorm by omitting the mean-centering step, saving $\approx 30\%$ compute and memory bandwidth without impacting convergence.

### Forward Equations:
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
$$y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i = \hat{x}_i \cdot \gamma_i$$
where $\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$ is the normalized activation and $\gamma_i$ is the learnable affine scale weight.

### Backward Equations:
Given incoming gradient $\delta_i = \frac{\partial L}{\partial y_i}$:
1. **Weight Gradient**:
   $$\frac{\partial L}{\partial \gamma_i} = \sum_{\text{rows}} \delta_{r, i} \cdot \hat{x}_{r, i}$$
2. **Input Gradient**:
   $$\text{Let } S = \frac{1}{d} \sum_{j=1}^d \delta_j \cdot \gamma_j \cdot \hat{x}_j$$
   $$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}(x)} \left( \delta_i \cdot \gamma_i - \hat{x}_i \cdot S \right)$$

---

## 2. Rotary Position Embeddings (RoPE)

RoPE encodes relative token position by applying a 2D rotation to pairs of features in the attention Query ($Q$) and Key ($K$) representations.

### Forward Rotation:
For feature pair $(x_{2i}, x_{2i+1})$ at token position $s$:
$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos \theta_{s, i} & -\sin \theta_{s, i} \\ \sin \theta_{s, i} & \cos \theta_{s, i} \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
$$x_{2i}' = x_{2i} \cos \theta - x_{2i+1} \sin \theta$$
$$x_{2i+1}' = x_{2i} \sin \theta + x_{2i+1} \cos \theta$$

### Backward Rotation:
Since the rotation matrix is orthogonal ($R^{-1} = R^T$), backpropagation through RoPE multiplies by the transposed matrix:
$$x_{2i}' = x_{2i} \cos \theta + x_{2i+1} \sin \theta$$
$$x_{2i+1}' = x_{2i+1} \cos \theta - x_{2i} \sin \theta$$

---

## 3. Kernel Fusion (`fused_add_norm`)

In a standard transformer layer, residual addition ($h \leftarrow h + \text{attn\_out}$) is followed immediately by pre-layer normalization.
Materializing $h$ back to DRAM before reading it again for RMSNorm wastes enormous memory bandwidth.

**Fused Add-Norm Execution**:
1. Load $h$ and $\text{attn\_out}$ into registers.
2. In-place add $h_{new} = h + \text{attn\_out}$ and store $h_{new}$ back to global memory (for backward).
3. Simultaneously accumulate $h_{new}^2$ in warp registers.
4. Perform warp shuffle reduction to compute RMS.
5. Multiply $h_{new} \cdot \text{RMS} \cdot \gamma$ and write normalized output to global memory in a single kernel launch.
