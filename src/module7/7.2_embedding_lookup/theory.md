# 7.2 Stage 2: Token Embedding Lookup

## 1. Role in the LLM Pipeline

The **Token Embedding** layer is the first GPU operation in every forward pass of an LLM:
$$\text{Input: } \text{Tokens } \in \mathbb{N}^{B \times S} \longrightarrow \text{Hidden States } h_0 \in \mathbb{R}^{B \times S \times D}$$

In CPU-based implementations, embedding lookups require triple-nested loops over batch ($B$), sequence length ($S$), and hidden dimension ($D$), materializing millions of memory copies and causing high host overhead. Moving this lookup to GPU allows hidden activations to remain entirely within High Bandwidth Memory (HBM).

---

## 2. Memory Coalescing in Embedding Lookup

Given:
* Flat thread ID `gid`
* Total tokens $T = B \times S$
* Hidden dimension $D$

Thread coordinate decomposition:
```cpp
uint32_t token_pos = gid / D;
uint32_t d         = gid % D;
```

Because adjacent threads in a warp have adjacent `gid` values ($gid, gid+1, gid+2, \dots$):
* All 32 threads in the warp share the exact same `token_pos` (when $D \ge 32$)!
* The dimension index $d$ increases monotonically by 1 per lane: $d, d+1, d+2, \dots, d+31$.
* The global memory address:
  $$\text{addr} = \text{token\_id} \times D + d$$
  is strictly contiguous and aligned to 128 bytes, achieving **100% memory coalescing efficiency**!

---

## 3. Vectorized Memory Transfers (128-bit)

When $D$ is divisible by 4 (or 8 for BF16), each thread can load and store `float4` (16 bytes) or four `__nv_bfloat162` values:
```cpp
uint32_t vec_d = gid % (D / 4);
uint32_t token_pos = gid / (D / 4);
uint32_t token_id = token_ids[token_pos];
reinterpret_cast<float4*>(out)[gid] = 
    reinterpret_cast<const float4*>(table)[token_id * (D / 4) + vec_d];
```
This maximizes DRAM burst bandwidth utilization on NVIDIA SM memory controllers.
