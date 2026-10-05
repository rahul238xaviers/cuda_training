# Chapter 5.3: Numerically Stable Reductions — Log-Sum-Exp & Online Softmax

Floating-point arithmetic is finite. A GPU kernel that computes `expf(1000.0f)` returns `+inf`. A kernel that subtracts two nearly-equal large numbers produces catastrophic cancellation. These are not edge cases — they happen constantly in LLM training where logits grow large during early training, and in long-sequence attention where scores accumulate across thousands of tokens.

This chapter covers the mathematical techniques that make computations numerically correct regardless of scale, and how they are implemented in GPU kernels efficiently.

---

## 1. The IEEE 754 Float Landscape: What Can Go Wrong

A 32-bit `float` encodes a number as: `(-1)^sign * 1.mantissa * 2^exponent`

```text
IEEE 754 FP32 key limits:
  Largest finite value:    3.4 * 10^38  (exponent = 127)
  Smallest positive value: 1.2 * 10^-38 (exponent = -126)
  Machine epsilon:         1.19 * 10^-7  (smallest value where 1.0f + eps != 1.0f)

exp() function limits:
  expf(x) overflows  to +inf  when x > 88.72  (ln(3.4e38) ≈ 88.72)
  expf(x) underflows to 0.0   when x < -87.34

BF16 limits (used for LLM weights):
  Largest finite:  3.4 * 10^38  (same exponent range as FP32!)
  Machine epsilon: 7.8 * 10^-3  (only 7 bits of mantissa vs 23 in FP32)
  exp() overflow:  same as FP32 (same exponent range)
```

The immediate practical problem: in LLM attention, we compute `softmax(QK^T / sqrt(d))`. With sequence length 4096 and d=128, attention logits can easily reach values of 20-100. In early training, unbounded logits can reach 500+.

---

## 2. The Softmax Overflow Problem

Naive softmax:

```cpp
// WRONG: Will produce inf / inf = NaN for large logits
__global__ void naive_softmax(const float* logits, float* probs, int N) {
    // Step 1: Compute sum of exp
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i)
        sum_exp += expf(logits[i]);   // expf(100.0f) = INF → sum_exp = INF

    // Step 2: Normalize
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N)
        probs[gid] = expf(logits[gid]) / sum_exp;   // INF / INF = NaN!
}
```

A single logit value of 89 or above causes the entire softmax output to become NaN — silently corrupting every downstream computation.

---

## 3. The Log-Sum-Exp (LSE) Trick: Mathematically Stable Softmax

The mathematical insight: softmax is invariant to subtracting a constant from all logits.

```text
Proof:
softmax_i(z) = exp(z_i) / sum_j exp(z_j)

Subtract constant m from all:
softmax_i(z - m) = exp(z_i - m) / sum_j exp(z_j - m)
                 = exp(z_i) * exp(-m) / [sum_j exp(z_j) * exp(-m)]
                 = exp(z_i) / sum_j exp(z_j)     ← exp(-m) cancels!
                 = softmax_i(z)    ✓
```

Choose `m = max_j(z_j)`. Then `z_i - m <= 0` for all `i`, so `exp(z_i - m) <= 1.0f` always. **Overflow is mathematically impossible.**

```cpp
// CORRECT: Numerically stable softmax
__global__ void stable_softmax_kernel(float* logits, float* probs, int N) {
    // Pass 1: Find maximum
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, logits[i]);
    max_val = block_reduce_max(max_val);  // Warp+smem reduction to find global max

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;  // All threads now have the global max

    // Pass 2: Compute exp(z - max) and sum  [exp values guaranteed in (0, 1]]
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum_exp += expf(logits[i] - max_val);  // Always finite!
    sum_exp = block_reduce_sum(sum_exp);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();
    sum_exp = s_sum;

    // Pass 3: Normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        probs[i] = expf(logits[i] - max_val) / sum_exp;
}
```

This requires 3 passes over the logits (max, sum, normalize). Each pass reads N floats from memory. For attention kernels where N = seq_len = 4096, this means reading 3 × 4096 × 4 = 48 KB from memory per row.

---

## 4. The Online Softmax Algorithm: 3 Passes → 1 Pass

**Milakov & Gimelshein (2018)** showed that the max and sum can be computed simultaneously in a single pass by maintaining running statistics and rescaling the denominator whenever the running max changes.

### The Core Identity

When processing a new chunk with local max `m_new > m_old`, the previously accumulated sum `d_old` is no longer relative to the correct maximum. We must rescale it:

```text
d_old was: sum of exp(x_i - m_old) for elements seen so far
d_new needs: sum of exp(x_i - m_new) for all elements

Since exp(x_i - m_new) = exp(x_i - m_old) * exp(m_old - m_new):

d_new = d_old * exp(m_old - m_new) + sum(exp(x_j - m_new)) for new chunk
```

This rescaling factor `exp(m_old - m_new)` is always in `(0, 1]` since `m_new >= m_old`, so it never causes overflow.

### Single-Pass Online Algorithm

```cpp
// Process elements one-by-one (or chunk-by-chunk) maintaining running stats:
struct RunningStats {
    float max_val;    // Running maximum seen so far
    float sum_exp;    // Running sum of exp(x_i - current_max)
};

// Merge two partial running stats (combine left-chunk and right-chunk):
__device__ RunningStats merge_stats(RunningStats a, RunningStats b) {
    if (b.max_val > a.max_val) {
        return { b.max_val, a.sum_exp * expf(a.max_val - b.max_val) + b.sum_exp };
    } else {
        return { a.max_val, a.sum_exp + b.sum_exp * expf(b.max_val - a.max_val) };
    }
}

// Warp-level reduction of RunningStats using shuffles:
__device__ RunningStats warp_reduce_stats(RunningStats s) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        RunningStats other;
        other.max_val = __shfl_down_sync(0xffffffff, s.max_val, offset);
        other.sum_exp = __shfl_down_sync(0xffffffff, s.sum_exp, offset);
        s = merge_stats(s, other);
    }
    return s;
}

// Online softmax kernel: single memory pass over logits
__global__ void online_softmax_kernel(const float* logits, float* probs, int N) {
    // Each thread accumulates its own RunningStats:
    RunningStats local = { -INFINITY, 0.0f };

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = logits[i];
        // Merge new element with running stats:
        if (x > local.max_val) {
            local.sum_exp = local.sum_exp * expf(local.max_val - x) + 1.0f;
            local.max_val = x;
        } else {
            local.sum_exp += expf(x - local.max_val);
        }
    }

    // Block reduction of RunningStats:
    // (Simplified here; full implementation uses smem for warp partials)
    RunningStats block_stats = local;  // (Full block reduction of RunningStats omitted for clarity)
    float global_max = block_stats.max_val;
    float global_sum = block_stats.sum_exp;

    // Final normalization pass:
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        probs[i] = expf(logits[i] - global_max) / global_sum;
}
```

### Why This Powers FlashAttention

FlashAttention's core innovation is exactly this: it processes attention scores in tiles that fit in shared memory, and uses the online softmax merge formula to combine the `(max, sum)` statistics from each tile without materializing the full N×N attention matrix in HBM.

```text
FlashAttention tile processing (simplified):

For each K/V tile of size BLOCK_SIZE:
  1. Load Q tile and K tile into shared memory
  2. Compute local attention scores: S = Q @ K^T / sqrt(d)
  3. Find local max m_curr = max(S)
  4. Merge with running stats: (m_new, d_new) = merge(m_old, d_old, m_curr, sum_exp(S - m_curr))
  5. Compute output contribution: O += exp(S - m_new) * V
  6. Rescale previous O by exp(m_old - m_new)

Final: O /= d_new

Result: Exact softmax attention, computed tile-by-tile in SRAM, never materializing N^2 matrix.
Memory: O(N) instead of O(N^2).
```

---

## 5. Catastrophic Cancellation: The Other Numerical Enemy

When two nearly-equal large numbers are subtracted, significant bits cancel:

```cpp
float a = 1000000.1f;
float b = 1000000.0f;
float diff = a - b;   // Mathematically 0.1, but in FP32:
// a = 1.00000010 * 2^20  (stored exactly)
// b = 1.00000000 * 2^20  (stored exactly)
// diff ≈ 0.1f              → only 1 significant bit remains!
// Relative error can be enormous despite both inputs being "accurate"
```

### Where This Appears in LLMs

- **Variance computation**: `var = mean(x^2) - mean(x)^2` — subtracts two large squares
- **Adam optimizer**: `m_hat / sqrt(v_hat + eps)` — when `v_hat` is tiny, `eps` dominates
- **Log probability**: `log(softmax(z))` — prone to cancellation for high-probability classes

### Numerically Stable Variance

```cpp
// WRONG: Two-pass with subtraction (catastrophic cancellation):
float mean = compute_mean(x, N);
float var  = 0.0f;
for (int i = 0; i < N; ++i)
    var += (x[i] - mean) * (x[i] - mean);  // x[i] ≈ mean → catastrophic cancellation

// CORRECT: Welford's online algorithm (incremental variance, no cancellation):
__device__ void welford_update(float x, float* mean, float* M2, int* count) {
    (*count)++;
    float delta  = x - *mean;
    *mean += delta / (*count);
    float delta2 = x - *mean;
    *M2  += delta * delta2;
    // Variance = *M2 / (*count - 1)  [sample variance]
    // Variance = *M2 / (*count)       [population variance]
}
// Welford never subtracts large numbers from each other — stays numerically stable.
```

---

## 6. The Log-Space Trick for Cross-Entropy Loss

Standard cross-entropy: `L = -log(softmax(z)[target])`

Naive computation `log(exp(z_t) / sum(exp(z_j)))` = two sources of instability.

The stable form derives directly from LSE:
```text
-log(softmax(z)[t]) = -z_t + log(sum_j exp(z_j))
                    = -z_t + m + log(sum_j exp(z_j - m))   where m = max(z)
```

```cpp
__device__ float cross_entropy_stable(const float* logits, int target, int num_classes) {
    // Step 1: Find max
    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; ++i)
        max_val = fmaxf(max_val, logits[i]);

    // Step 2: Compute log-sum-exp = m + log(sum(exp(x - m)))
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i)
        sum_exp += expf(logits[i] - max_val);
    float lse = max_val + logf(sum_exp);   // log-sum-exp

    // Step 3: Loss = -z_target + LSE (always finite!)
    return -logits[target] + lse;
}
```

This is exactly what PyTorch's `F.cross_entropy` computes internally (via `torch.nn.functional.log_softmax` + NLLLoss).

---

## 7. Summary: Stability Techniques Reference

| Problem | Naive Symptom | Stable Technique |
| :--- | :--- | :--- |
| Softmax of large logits | exp() overflows → NaN | Subtract max (LSE trick) |
| Softmax in single memory pass | 3 sequential HBM reads | Online Softmax (Milakov 2018) |
| Variance of large values | Catastrophic cancellation | Welford's online algorithm |
| Cross-entropy loss | log(0) → -inf, overflow | log-sum-exp identity |
| Sum of many floats | Accumulated rounding error | Kahan compensated summation |
| Attention on long sequences | N^2 HBM → memory wall | FlashAttention SRAM tiling + online softmax |
