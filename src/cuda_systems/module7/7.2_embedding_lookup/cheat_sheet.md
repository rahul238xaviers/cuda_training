# Topic 7.2: Token Embedding Lookup & High-Throughput Gather — Cheat Sheet

A concise reference for gathering dense hidden vectors from vocabulary token IDs.

---

### 1. Embedding Forward Mapping
- Given Token IDs array `tokens [B, S]` and Weight Table `W [VocabSize, HiddenDim]`:
- Output shape: `[B, S, HiddenDim]`.
- Row `i` of token `t = tokens[i]` is retrieved:
  ```cpp
  int token_idx = blockIdx.y; // Which token in the batch
  int token_id = tokens[token_idx];
  int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (dim_idx < hidden_dim) {
      const float4* src = reinterpret_cast<const float4*>(&weights[token_id * hidden_dim + dim_idx]);
      float4* dst = reinterpret_cast<float4*>(&out[token_idx * hidden_dim + dim_idx]);
      *dst = *src;
  }
  ```

---

### 2. LLM Systems Context
- **First Layer of LLMs**: Translating input prompt token IDs into high-dimensional latent space (e.g. 4096 dimensions).
- **Vocab Slicing**: Distributed tensor parallel embeddings across multiple GPUs.
