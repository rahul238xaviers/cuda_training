# Module 7.6 — FlashAttention Suite (Forward & Backward)

## 1. Algorithmic Motivation: IO-Aware Attention

In standard multi-head attention:

```text
Attention(Q, K, V) = Softmax<=ft((Q K_transposed) / (sqrt(d))) V
```

For sequence length `S` and head dimension `d`:
1. `S_{score} = (Q K_transposed) / (sqrt(d)) in tensor[S x S]` (Requires `O(S^2)` DRAM writes)
2. `P = Softmax(S_{score}) in tensor[S x S]` (Requires `O(S^2)` DRAM reads and writes)
3. `O = P V in tensor[S x d]` (Requires `O(S^2)` DRAM reads)

When `S = 4096` or `S = 32768`, materializing `S x S` probability matrices exhausts GPU High Bandwidth Memory (HBM) and causes extreme memory latency bottlenecks.

**FlashAttention (Dao et al., 2022, 2023)** eliminates all intermediate `O(S^2)` DRAM reads and writes by:
1. **Tiling** `Q, K, V` into SRAM (Shared Memory) blocks of size `B_r x d` and `B_c x d`.
2. Computing the attention dot-products, online softmax, and value multiplication completely inside on-chip SRAM and registers.
3. Writing only the final `O in tensor[S x d]` back to HBM.

---

## 2. FlashAttention-2 Forward Mathematics

In FlashAttention-1, the outer loop iterated over `K, V` blocks, requiring atomic adds to write output tiles.
**FlashAttention-2** inverts the loop hierarchy:
- **Outer Loop**: Iterates over query blocks `Q_i in tensor[B_r x d]` (`i = 0, ..., T_r - 1`).
- **Inner Loop**: Iterates over key/value blocks `K_j, V_j in tensor[B_c x d]` (`j = 0, ..., T_c - 1`).

Because each thread block owns a query tile `Q_i`, it maintains the unnormalized output accumulator `O_i` and running softmax statistics completely in registers without any atomic operations!

### Online Softmax Formulation
Let `S_i^{(j)} = (Q_i K_j_transposed) / (sqrt(d)) in tensor[B_r x B_c]`.
For each row `r in [0, B_r - 1]`:
1. Local row maximum of tile `j`:
   
```text
m_local_i^{(j)} = max(c) S_{i, c}^{(j)}
```

2. New global row maximum:
   
```text
m_i^{(j)} = max<=ft(m_i^{(j-1)}, m_local_i^{(j)})
```

3. Local unnormalized exponentials:
   
```text
P_local_i^{(j)} = exp<=ft(S_i^{(j)} - m_i^{(j)})
```

4. Local sum of exponentials:
   
```text
l_local_i^{(j)} = sum_c P_local_{i, c}^{(j)}
```

5. Update global normalization denominator:
   
```text
l_i^{(j)} = l_i^{(j-1)} * exp<=ft(m_i^{(j-1)} - m_i^{(j)}) + l_local_i^{(j)}
```

6. Rescale previous output accumulator and add current tile product:
   
```text
O_i^{(j)} = O_i^{(j-1)} * exp<=ft(m_i^{(j-1)} - m_i^{(j)}) + P_local_i^{(j)} V_j
```


After completing the inner loop over all `j`, normalize the output accumulator once:

```text
O_i = frac{O_i^{(final)}}{l_i^{(final)}}
```


---

## 3. Causal Masking Optimization

In autoregressive decoder models (e.g. LLaMA, GPT), token `q` can only attend to tokens `k <= q`. Any attention score where `k > q` is set to `-infty` (`P_{q, k} = 0`).

When tiling with `B_r` and `B_c`:
1. **Full Compute Blocks**: If `j * B_c + B_c - 1 <= i * B_r`, all tokens in `K_j` are `<=` all tokens in `Q_i`. No masking needed!
2. **Skip Blocks**: If `j * B_c > i * B_r + B_r - 1`, all tokens in `K_j` are strictly in the causal future (`k > q`). The block is skipped completely, saving 50% of inner loop FLOPs!
3. **Diagonal Masked Blocks**: When the block intersects the diagonal, elementwise masking is applied:
   
```text
if  (j * B_c + lane) > (i * B_r + row),   S_{r, c} = -infty
```


---

## 4. FlashAttention-2 Backward Pass Mathematics

During the backward pass, we are given `dO in tensor[S x d]`. We must compute gradients `dQ, dK, dV` without materializing the `S x S` attention matrix!

### Step 1: Precompute Scalar `D_i`
From the softmax backward derivative:

```text
D_i = sum(d=0) dO_{i, d} * O_{i, d}
```

`D_i` is a scalar per query row `i`.

### Step 2: Recompute Attention Probabilities `P_{ij}` in SRAM
Using stored statistics `m_i` and `l_i`:

```text
S_{ij} = (Q_i K_j_transposed) / (sqrt(d))
```


```text
P_{ij} = exp(S_{ij} - m_i) / l_i
```


### Step 3: Gradient Formulation
1. **Value Gradient**:
   
```text
dV_j = sum_i P_{ij}_transposed * dO_i
```

2. **Score Gradient**:
   
```text
dP_{ij} = dO_i * V_j_transposed
```

   
```text
dS_{ij} = P_{ij} * (dP_{ij} - D_i)
```

3. **Query & Key Gradients**:
   
```text
dQ_i = (1) / (sqrt(d)) sum_j dS_{ij} * K_j
```

   
```text
dK_j = (1) / (sqrt(d)) sum_i dS_{ij}_transposed * Q_i
```


By recomputing `P_{ij}` on-the-fly in SRAM blocks, FlashAttention backward achieves `O(S)` peak memory footprint and outperforms standard PyTorch attention by 2x to 4x while requiring zero extra HBM memory!
