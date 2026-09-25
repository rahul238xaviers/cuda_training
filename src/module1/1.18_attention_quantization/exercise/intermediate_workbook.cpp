#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Attention & Quantization
//
// Module: 1.18 - ML Tensor Memory Primitives
// Level:  Intermediate
//
// Focus: Multi-head sliding-window attention masks, block-wise INT8 quantization
//        (AWQ/GPTQ style), and fused SwiGLU activation layer computation.
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
    std::cout << "--- WORKBOOK: Attention & Quantization (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Batched Sliding-Window Attention Masking
    //
    // Context: In long-context models (Mistral, Gemma), computing dense attention
    //          across thousands of tokens is computationally prohibitive (O(S^2)).
    //          Sliding Window Attention restricts each token at query index q to only
    //          attend to key tokens k within a local window:
    //            Valid range: max(0, q - W + 1) <= k <= q
    //          Tokens outside this range must be masked to -10000.0f.
    //
    // Task: Given attention scores tensor `scores` of shape [BATCH=2, HEADS=2, S=8, S=8]
    //       and WINDOW_SIZE W=3:
    //       Apply sliding window masking in-place.
    // -------------------------------------------------------------------------
    {
        const int BATCH = 2, HEADS = 2, S = 8, W = 3;
        const size_t total_elements = BATCH * HEADS * S * S;
        std::vector<float> scores(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            scores[i] = static_cast<float>((i % 17) + 1.0f);
        }
        std::vector<float> original = scores;

        // TODO: For each batch, head, and query position q in [0, S), mask keys k:
        // Set scores to -10000.0f if k > q OR k < q - W + 1.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int b = 0; b < BATCH && p1_passed; ++b) {
            for (int h = 0; h < HEADS && p1_passed; ++h) {
                for (int q = 0; q < S && p1_passed; ++q) {
                    for (int k = 0; k < S; ++k) {
                        size_t idx = (size_t)b * (HEADS * S * S) + (size_t)h * (S * S) + (size_t)q * S + k;
                        float val = scores[idx];
                        bool in_window = (k <= q) && (k >= q - W + 1);
                        if (in_window) {
                            if (val != original[idx]) p1_passed = false;
                        } else {
                            if (val != -10000.0f) p1_passed = false;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: Batched Sliding-Window Attention Masking", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Block-Wise INT8 Quantization with Block Scales
    //
    // Context: Per-tensor quantization suffers when outlier channels skew the scale.
    //          Modern LLM quantization (AWQ, BitsAndBytes) groups weights into small
    //          blocks (e.g., BLOCK_SIZE=64) and computes an independent scale per block.
    //
    // Task: Given weight buffer `weights` of size N=256 and BLOCK_SIZE=64 (NUM_BLOCKS=4):
    //       1. For each block b:
    //          a. Find block_absmax = max(|weights[b*64 ... b*64+63]|)
    //          b. Compute `scales[b] = block_absmax / 127.0f`
    //          c. Quantize elements to `quantized_weights` (int8_t clamped to [-127, 127])
    //       2. Dequantize all elements into `dequantized_weights`
    // -------------------------------------------------------------------------
    {
        const int N = 256;
        const int BLOCK_SIZE = 64;
        const int NUM_BLOCKS = N / BLOCK_SIZE;

        std::vector<float> weights(N);
        // Fill block 0 with small values, block 1 with large outlier values
        for (int i = 0; i < N; ++i) {
            if (i < 64) weights[i] = (i % 10) * 0.1f;
            else if (i < 128) weights[i] = ((i % 10) - 5) * 20.0f; // Large scale block
            else weights[i] = ((i % 7) - 3) * 1.5f;
        }

        std::vector<float> scales(NUM_BLOCKS, 0.0f);
        std::vector<int8_t> quantized_weights(N, 0);
        std::vector<float> dequantized_weights(N, 0.0f);

        // TODO: Perform block-wise quantization and dequantization.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int b = 0; b < NUM_BLOCKS && p2_passed; ++b) {
            float max_val = 0.0f;
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                max_val = std::max(max_val, std::abs(weights[b * BLOCK_SIZE + i]));
            }
            float expected_scale = max_val / 127.0f;
            if (std::abs(scales[b] - expected_scale) > 1e-4f) p2_passed = false;

            for (int i = 0; i < BLOCK_SIZE; ++i) {
                int idx = b * BLOCK_SIZE + i;
                float err = std::abs(dequantized_weights[idx] - weights[idx]);
                if (err > expected_scale) p2_passed = false;
            }
        }

        reportStatus("Problem 2: Block-Wise INT8 Quantization (AWQ/GPTQ Style)", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: SwiGLU Fused Activation Layer
    //
    // Context: In Llama, PaLM, and Mistral, the feed-forward layer uses SwiGLU:
    //            SwiGLU(gate, up) = swish(gate) * up
    //            where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    //
    // Task: Given two input activation buffers `gate` and `up` of size N=128:
    //       Compute SwiGLU into `out` buffer in-place:
    //         out[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i]
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        std::vector<float> gate(N);
        std::vector<float> up(N);
        std::vector<float> out(N, 0.0f);

        for (int i = 0; i < N; ++i) {
            gate[i] = (i - 64) * 0.1f;
            up[i] = (i % 5 + 1) * 0.5f;
        }

        // TODO: Compute SwiGLU activation for each index i in [0, N).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            float g = gate[i];
            float swish_g = g / (1.0f + std::exp(-g));
            float expected = swish_g * up[i];
            if (std::abs(out[i] - expected) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: SwiGLU Fused Activation Layer", p3_passed);
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
