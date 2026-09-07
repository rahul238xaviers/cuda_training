# Module 7.6 — FlashAttention Suite (Forward & Backward)

## 1. Algorithmic Motivation: IO-Aware Attention

In standard multi-head attention:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
For sequence length $S$ and head dimension $d$:
1. $S_{\text{score}} = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{S \times S}$ (Requires $O(S^2)$ DRAM writes)
2. $P = \text{Softmax}(S_{\text{score}}) \in \mathbb{R}^{S \times S}$ (Requires $O(S^2)$ DRAM reads and writes)
3. $O = P V \in \mathbb{R}^{S \times d}$ (Requires $O(S^2)$ DRAM reads)

When $S = 4096$ or $S = 32768$, materializing $S \times S$ probability matrices exhausts GPU High Bandwidth Memory (HBM) and causes extreme memory latency bottlenecks.

**FlashAttention (Dao et al., 2022, 2023)** eliminates all intermediate $O(S^2)$ DRAM reads and writes by:
1. **Tiling** $Q, K, V$ into SRAM (Shared Memory) blocks of size $B_r \times d$ and $B_c \times d$.
2. Computing the attention dot-products, online softmax, and value multiplication completely inside on-chip SRAM and registers.
3. Writing only the final $O \in \mathbb{R}^{S \times d}$ back to HBM.

---

## 2. FlashAttention-2 Forward Mathematics

In FlashAttention-1, the outer loop iterated over $K, V$ blocks, requiring atomic adds to write output tiles.
**FlashAttention-2** inverts the loop hierarchy:
- **Outer Loop**: Iterates over query blocks $Q_i \in \mathbb{R}^{B_r \times d}$ ($i = 0, \dots, T_r - 1$).
- **Inner Loop**: Iterates over key/value blocks $K_j, V_j \in \mathbb{R}^{B_c \times d}$ ($j = 0, \dots, T_c - 1$).

Because each thread block owns a query tile $Q_i$, it maintains the unnormalized output accumulator $O_i$ and running softmax statistics completely in registers without any atomic operations!

### Online Softmax Formulation
Let $S_i^{(j)} = \frac{Q_i K_j^T}{\sqrt{d}} \in \mathbb{R}^{B_r \times B_c}$.
For each row $r \in [0, B_r - 1]$:
1. Local row maximum of tile $j$:
   $$\tilde{m}_i^{(j)} = \max_{c} S_{i, c}^{(j)}$$
2. New global row maximum:
   $$m_i^{(j)} = \max\left(m_i^{(j-1)}, \tilde{m}_i^{(j)}\right)$$
3. Local unnormalized exponentials:
   $$\tilde{P}_i^{(j)} = \exp\left(S_i^{(j)} - m_i^{(j)}\right)$$
4. Local sum of exponentials:
   $$\tilde{l}_i^{(j)} = \sum_c \tilde{P}_{i, c}^{(j)}$$
5. Update global normalization denominator:
   $$l_i^{(j)} = l_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \tilde{l}_i^{(j)}$$
6. Rescale previous output accumulator and add current tile product:
   $$O_i^{(j)} = O_i^{(j-1)} \cdot \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) + \tilde{P}_i^{(j)} V_j$$

After completing the inner loop over all $j$, normalize the output accumulator once:
$$O_i = \frac{O_i^{(\text{final})}}{l_i^{(\text{final})}}$$

---

## 3. Causal Masking Optimization

In autoregressive decoder models (e.g. LLaMA, GPT), token $q$ can only attend to tokens $k \le q$. Any attention score where $k > q$ is set to $-\infty$ ($P_{q, k} = 0$).

When tiling with $B_r$ and $B_c$:
1. **Full Compute Blocks**: If $j \cdot B_c + B_c - 1 \le i \cdot B_r$, all tokens in $K_j$ are $\le$ all tokens in $Q_i$. No masking needed!
2. **Skip Blocks**: If $j \cdot B_c > i \cdot B_r + B_r - 1$, all tokens in $K_j$ are strictly in the causal future ($k > q$). The block is skipped completely, saving 50% of inner loop FLOPs!
3. **Diagonal Masked Blocks**: When the block intersects the diagonal, elementwise masking is applied:
   $$\text{if } (j \cdot B_c + \text{lane}) > (i \cdot B_r + \text{row}), \quad S_{r, c} = -\infty$$

---

## 4. FlashAttention-2 Backward Pass Mathematics

During the backward pass, we are given $dO \in \mathbb{R}^{S \times d}$. We must compute gradients $dQ, dK, dV$ without materializing the $S \times S$ attention matrix!

### Step 1: Precompute Scalar $D_i$
From the softmax backward derivative:
$$D_i = \sum_{d=0}^{D-1} dO_{i, d} \cdot O_{i, d}$$
$D_i$ is a scalar per query row $i$.

### Step 2: Recompute Attention Probabilities $P_{ij}$ in SRAM
Using stored statistics $m_i$ and $l_i$:
$$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}$$
$$P_{ij} = \exp(S_{ij} - m_i) / l_i$$

### Step 3: Gradient Formulation
1. **Value Gradient**:
   $$dV_j = \sum_i P_{ij}^T \cdot dO_i$$
2. **Score Gradient**:
   $$dP_{ij} = dO_i \cdot V_j^T$$
   $$dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$$
3. **Query & Key Gradients**:
   $$dQ_i = \frac{1}{\sqrt{d}} \sum_j dS_{ij} \cdot K_j$$
   $$dK_j = \frac{1}{\sqrt{d}} \sum_i dS_{ij}^T \cdot Q_i$$

By recomputing $P_{ij}$ on-the-fly in SRAM blocks, FlashAttention backward achieves $O(S)$ peak memory footprint and outperforms standard PyTorch attention by 2x to 4x while requiring zero extra HBM memory!
