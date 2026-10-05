#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>

// =========================================================================
// INTERMEDIATE WORKBOOK: Embeddings & Projections
//
// Module: 1.17 - ML Tensor Memory Primitives
// Level:  Intermediate
//
// Focus: Multi-head projection unpacking, KV-cache dynamic slot updates,
//        and ALiBi geometric bias stride patterns.
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Embeddings & Projections (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Unpacking Interleaved QKV GEMM Output
    //
    // Context: In fused attention projection, a single linear layer produces
    //          a packed tensor of shape [B, S, 3 * H * D].
    //          The last dimension contains Query, Key, and Value concatenated:
    //          [Q (H*D floats) | K (H*D floats) | V (H*D floats)].
    //
    // Task: Given packed tensor `qkv_interleaved` of shape [B=2, S=4, 3 * H * D]
    //       where H=4, D=16 (so 3 * 4 * 16 = 192 floats per token):
    //       Unpack the data into three separate tensors:
    //         - `Q_out` of shape [B, S, H * D] (64 floats per token)
    //         - `K_out` of shape [B, S, H * D]
    //         - `V_out` of shape [B, S, H * D]
    // -------------------------------------------------------------------------
    {
        const int B = 2, S = 4, H = 4, D = 16;
        const int HEAD_DIM_TOTAL = H * D; // 64
        const int PACKED_DIM = 3 * HEAD_DIM_TOTAL; // 192

        std::vector<float> qkv_interleaved(B * S * PACKED_DIM);
        for (size_t i = 0; i < qkv_interleaved.size(); ++i) {
            qkv_interleaved[i] = static_cast<float>(i + 1);
        }

        std::vector<float> Q_out(B * S * HEAD_DIM_TOTAL, 0.0f);
        std::vector<float> K_out(B * S * HEAD_DIM_TOTAL, 0.0f);
        std::vector<float> V_out(B * S * HEAD_DIM_TOTAL, 0.0f);

        // TODO: For every batch b and sequence step s, slice and copy the 64 floats
        // for Q, 64 floats for K, and 64 floats for V into their respective output buffers.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int b = 0; b < B && p1_passed; ++b) {
            for (int s = 0; s < S && p1_passed; ++s) {
                size_t base_packed = (size_t)(b * S + s) * PACKED_DIM;
                size_t base_out = (size_t)(b * S + s) * HEAD_DIM_TOTAL;

                for (int i = 0; i < HEAD_DIM_TOTAL; ++i) {
                    if (Q_out[base_out + i] != qkv_interleaved[base_packed + i]) p1_passed = false;
                    if (K_out[base_out + i] != qkv_interleaved[base_packed + HEAD_DIM_TOTAL + i]) p1_passed = false;
                    if (V_out[base_out + i] != qkv_interleaved[base_packed + 2 * HEAD_DIM_TOTAL + i]) p1_passed = false;
                }
            }
        }

        reportStatus("Problem 1: Unpacking Interleaved QKV GEMM Outputs", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: In-Memory KV-Cache Slot Update
    //
    // Context: In autoregressive LLM decoding, at each generation step we compute
    //          K and V vectors for ONLY the new single token, and write them into
    //          the pre-allocated KV-cache at slot `step`.
    //
    // Layout: `kv_cache` shape: [BATCH=2, NUM_HEADS=4, MAX_SEQ=32, HEAD_DIM=16].
    //
    // Task: Given newly generated Key vectors `new_keys` of shape [BATCH=2, NUM_HEADS=4, HEAD_DIM=16]:
    //       Insert `new_keys` into `kv_cache` at sequence index `target_step = 7`.
    //       Ensure all other steps in `kv_cache` remain untouched!
    // -------------------------------------------------------------------------
    {
        const int BATCH = 2, NUM_HEADS = 4, MAX_SEQ = 32, HEAD_DIM = 16;
        std::vector<float> kv_cache(BATCH * NUM_HEADS * MAX_SEQ * HEAD_DIM, 0.0f);

        // Fill cache with placeholder -1.0f
        std::fill(kv_cache.begin(), kv_cache.end(), -1.0f);

        std::vector<float> new_keys(BATCH * NUM_HEADS * HEAD_DIM);
        for (size_t i = 0; i < new_keys.size(); ++i) {
            new_keys[i] = static_cast<float>(i + 100.0f);
        }

        const int target_step = 7;

        // TODO: Write new_keys into kv_cache at target_step for all batches and heads.
        // Cache stride for [b, h, s, d] is:
        // offset = b * (NUM_HEADS * MAX_SEQ * HEAD_DIM) + h * (MAX_SEQ * HEAD_DIM) + s * HEAD_DIM + d
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int b = 0; b < BATCH && p2_passed; ++b) {
            for (int h = 0; h < NUM_HEADS && p2_passed; ++h) {
                for (int s = 0; s < MAX_SEQ && p2_passed; ++s) {
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        size_t off = (size_t)b * (NUM_HEADS * MAX_SEQ * HEAD_DIM) +
                                     (size_t)h * (MAX_SEQ * HEAD_DIM) +
                                     (size_t)s * HEAD_DIM + d;
                        if (s == target_step) {
                            size_t src_off = (size_t)b * (NUM_HEADS * HEAD_DIM) + (size_t)h * HEAD_DIM + d;
                            if (kv_cache[off] != new_keys[src_off]) p2_passed = false;
                        } else {
                            if (kv_cache[off] != -1.0f) p2_passed = false;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: In-Memory KV-Cache Slot Update", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Multi-Head ALiBi (Attention with Linear Biases) Matrix Construction
    //
    // Context: ALiBi replaces positional embeddings by adding a static bias matrix
    //          directly to the attention score logits before softmax.
    //          Each head h has a slope m_h = 2^(-8 * (h + 1) / NUM_HEADS).
    //          The bias for query token i and key token j (with j <= i) is:
    //            bias[h, i, j] = -m_h * (i - j)
    //          If j > i (future tokens), the bias is set to -1e9 (causal masking).
    //
    // Task: Given NUM_HEADS=4 and SEQ_LEN=8, populate `alibi_bias` tensor
    //       of shape [NUM_HEADS, SEQ_LEN, SEQ_LEN].
    // -------------------------------------------------------------------------
    {
        const int NUM_HEADS = 4;
        const int SEQ_LEN = 8;
        std::vector<float> alibi_bias(NUM_HEADS * SEQ_LEN * SEQ_LEN, 0.0f);

        // TODO: Populate alibi_bias[h, i, j] according to ALiBi geometric slope formula.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int h = 0; h < NUM_HEADS && p3_passed; ++h) {
            float m_h = std::pow(2.0f, -8.0f * (h + 1) / static_cast<float>(NUM_HEADS));
            for (int i = 0; i < SEQ_LEN && p3_passed; ++i) {
                for (int j = 0; j < SEQ_LEN; ++j) {
                    size_t idx = (size_t)h * (SEQ_LEN * SEQ_LEN) + (size_t)i * SEQ_LEN + j;
                    if (j > i) {
                        if (alibi_bias[idx] != -1e9f) p3_passed = false;
                    } else {
                        float expected = -m_h * static_cast<float>(i - j);
                        if (std::abs(alibi_bias[idx] - expected) > 1e-4f) p3_passed = false;
                    }
                }
            }
        }

        reportStatus("Problem 3: Multi-Head ALiBi Attention Bias Matrix", p3_passed);
        if (p3_passed) passed++;
        total++;
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
