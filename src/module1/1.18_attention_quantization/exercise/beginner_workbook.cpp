#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Attention & Quantization
//
// Module: 1.18 - ML Tensor Memory Primitives
// Level:  Beginner
//
// Focus: In-place numerically-stable row-wise Softmax, symmetric INT8
//        quantization/dequantization with scale factors, and causal masking.
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
    std::cout << "--- WORKBOOK: Attention & Quantization (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Numerically-Stable Row-Wise Softmax
    //
    // Context: In Self-Attention, raw logits Q @ K^T must be normalized into
    //          probabilities using Softmax: P_ij = exp(S_ij - max_k(S_ik)) / sum_k exp(S_ik - max_k(S_ik)).
    //          Subtracting the row maximum prevents exponential overflow (NaN / Inf).
    //
    // Task: Given a batched logits buffer `logits` [ROWS=4, COLS=8]:
    //       Apply Softmax in-place along each row:
    //         1. Find row_max = max(logits[r, :])
    //         2. Replace each element with exp(logits[r, c] - row_max)
    //         3. Compute row_sum = sum(exp_vals)
    //         4. Divide each element by row_sum (so each row sums to 1.0)
    // -------------------------------------------------------------------------
    {
        const int ROWS = 4;
        const int COLS = 8;
        std::vector<float> logits = {
            10.0f, 12.0f, 8.0f,  15.0f, 11.0f, 9.0f,  14.0f, 13.0f,
            -5.0f, -2.0f, -1.0f, -8.0f, -3.0f, -4.0f, -2.0f, -6.0f,
            100.0f, 102.0f, 98.0f, 105.0f, 101.0f, 99.0f, 104.0f, 103.0f, // Large values (test overflow)
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f
        };

        // TODO: Apply in-place 3-pass numerically stable softmax per row.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < ROWS && p1_passed; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < COLS; ++c) {
                float val = logits[r * COLS + c];
                if (std::isnan(val) || std::isinf(val) || val < 0.0f || val > 1.0f) {
                    p1_passed = false;
                    break;
                }
                sum += val;
            }
            if (std::abs(sum - 1.0f) > 1e-4f) p1_passed = false;
        }

        reportStatus("Problem 1: Numerically-Stable Row-Wise Softmax", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Symmetric Per-Tensor INT8 Quantization & Dequantization
    //
    // Context: Quantization reduces memory bandwidth by storing FP32 weights/activations
    //          as signed 8-bit integers (`int8_t`).
    //          Scale: s = max(|x|) / 127.0f
    //          Quantize:   q = clamp(round(x / s), -127, 127)
    //          Dequantize: x_recon = q * s
    //
    // Task: Given float tensor `activations` of size N=16:
    //       1. Find the maximum absolute value `absmax`
    //       2. Compute `scale = absmax / 127.0f`
    //       3. Quantize each float into `quantized` (int8_t)
    //       4. Dequantize into `reconstructed` (float)
    // -------------------------------------------------------------------------
    {
        const int N = 16;
        std::vector<float> activations = {
            -12.4f, 5.8f, -0.3f, 44.2f, -31.0f, 0.0f, 18.9f, -44.2f,
            2.1f, -8.7f, 15.3f, -22.1f, 33.0f, -1.5f, 9.8f, -14.6f
        };

        float scale = 0.0f;
        std::vector<int8_t> quantized(N, 0);
        std::vector<float> reconstructed(N, 0.0f);

        // TODO: Compute scale, quantize to quantized array, and reconstruct to reconstructed array.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        float expected_scale = 44.2f / 127.0f;
        if (std::abs(scale - expected_scale) > 1e-4f) p2_passed = false;

        float max_error = 0.0f;
        for (int i = 0; i < N && p2_passed; ++i) {
            float err = std::abs(activations[i] - reconstructed[i]);
            if (err > max_error) max_error = err;
            if (err > expected_scale) { // Error should never exceed half the quantization step
                p2_passed = false;
            }
        }

        reportStatus("Problem 2: Symmetric Per-Tensor INT8 Quantization/Dequantization", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Lower-Triangular Causal Attention Masking
    //
    // Context: In autoregressive language modeling (GPT/Llama), a query token at
    //          step i cannot attend to future key tokens at step j > i.
    //          Before softmax, upper-triangular logits must be set to -10000.0f.
    //
    // Task: Given square attention logit matrix `scores` [S=8, S=8]:
    //       For every element at row i, col j:
    //         If j > i, set scores[i * S + j] = -10000.0f.
    //         Leave lower-triangular and diagonal elements unchanged.
    // -------------------------------------------------------------------------
    {
        const int S = 8;
        std::vector<float> scores(S * S);
        for (int i = 0; i < S * S; ++i) {
            scores[i] = static_cast<float>(i + 1);
        }
        std::vector<float> original = scores;

        // TODO: Apply causal mask in-place on scores matrix.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < S && p3_passed; ++i) {
            for (int j = 0; j < S; ++j) {
                float val = scores[i * S + j];
                if (j > i) {
                    if (val != -10000.0f) p3_passed = false;
                } else {
                    if (val != original[i * S + j]) p3_passed = false;
                }
            }
        }

        reportStatus("Problem 3: Lower-Triangular Causal Attention Masking", p3_passed);
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
