#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: ML Layer Offsets & Optimization
//
// Module: 1.19 - ML Tensor Memory Primitives
// Level:  Intermediate
//
// Focus: Group Normalization statistics, fused Linear + Bias + GELU layers,
//        and numerically stable Cross-Entropy Loss via Log-Sum-Exp.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.19_intermediate
//   ../../../output/1.19_intermediate
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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Group Normalization (GroupNorm) Mean & Variance per Group
    //
    // Context: In Vision Models and Diffusion (Stable Diffusion UNet), GroupNorm
    //          divides C channels into G groups. The normalization statistics
    //          (mean and variance) are computed over (C/G, H, W) for each batch item:
    //            mu_{n, g}    = (1 / M) * sum_{c in g, h, w} X[n, c, h, w]
    //            var_{n, g}   = (1 / M) * sum_{c in g, h, w} (X[n, c, h, w] - mu_{n, g})^2
    //            norm_{n, c, h, w} = (X[n, c, h, w] - mu_{n, g}) / sqrt(var_{n, g} + eps)
    //
    // Task: Given tensor `X` [N=2, C=8, H=4, W=4] and G=2 groups (4 channels per group).
    //       M = 4 * 4 * 4 = 64 elements per group.
    //       Normalize X in-place with eps = 1e-5f.
    // -------------------------------------------------------------------------
    {
        const int N = 2, C = 8, H = 4, W = 4, G = 2;
        const int CHANNELS_PER_GROUP = C / G; // 4
        const int M = CHANNELS_PER_GROUP * H * W; // 64
        const float EPS = 1e-5f;

        const size_t total_elements = N * C * H * W;
        std::vector<float> X(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            X[i] = static_cast<float>((i % 23) - 11) * 0.5f;
        }
        std::vector<float> original_X = X;

        // TODO: For each batch n and group g, compute mean and variance over the M elements,
        // and normalize all elements in that group in-place.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int n = 0; n < N && p1_passed; ++n) {
            for (int g = 0; g < G && p1_passed; ++g) {
                float sum = 0.0f;
                for (int c_rel = 0; c_rel < CHANNELS_PER_GROUP; ++c_rel) {
                    int c = g * CHANNELS_PER_GROUP + c_rel;
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            sum += original_X[(n * C + c) * (H * W) + h * W + w];
                        }
                    }
                }
                float mu = sum / M;
                float sum_sq_diff = 0.0f;
                for (int c_rel = 0; c_rel < CHANNELS_PER_GROUP; ++c_rel) {
                    int c = g * CHANNELS_PER_GROUP + c_rel;
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            float diff = original_X[(n * C + c) * (H * W) + h * W + w] - mu;
                            sum_sq_diff += diff * diff;
                        }
                    }
                }
                float var = sum_sq_diff / M;
                float inv_std = 1.0f / std::sqrt(var + EPS);

                for (int c_rel = 0; c_rel < CHANNELS_PER_GROUP; ++c_rel) {
                    int c = g * CHANNELS_PER_GROUP + c_rel;
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            size_t idx = (n * C + c) * (H * W) + h * W + w;
                            float expected = (original_X[idx] - mu) * inv_std;
                            if (std::abs(X[idx] - expected) > 1e-4f) {
                                p1_passed = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: Group Normalization (GroupNorm) In-Place", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Fused Linear + Bias + GELU Layer
    //
    // Context: In Transformer FFN (MLP) layers, linear projections are fused with
    //          GELU activation to avoid writing intermediate activations to memory:
    //            y = GELU(x @ W^T + b)
    //            GELU(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    //
    // Task: Given batch of tokens `X` [B=4, IN_DIM=16], weight `W` [OUT_DIM=32, IN_DIM=16],
    //       and bias `b` [OUT_DIM=32]:
    //       Compute output `Y` [B=4, OUT_DIM=32].
    // -------------------------------------------------------------------------
    {
        const int B = 4, IN_DIM = 16, OUT_DIM = 32;
        const float SQRT_2_OVER_PI = 0.79788456f;

        std::vector<float> X(B * IN_DIM);
        for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<float>((i % 7) - 3) * 0.2f;

        std::vector<float> W(OUT_DIM * IN_DIM);
        for (size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>((i % 11) - 5) * 0.1f;

        std::vector<float> bias(OUT_DIM);
        for (int i = 0; i < OUT_DIM; ++i) bias[i] = 0.1f * (i % 3);

        std::vector<float> Y(B * OUT_DIM, 0.0f);

        // TODO: For each batch b and output feature o:
        // Compute z = sum_i(X[b, i] * W[o, i]) + bias[o]
        // Compute Y[b, o] = 0.5 * z * (1 + tanh(SQRT_2_OVER_PI * (z + 0.044715 * z^3)))
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int b = 0; b < B && p2_passed; ++b) {
            for (int o = 0; o < OUT_DIM; ++o) {
                float z = bias[o];
                for (int i = 0; i < IN_DIM; ++i) {
                    z += X[b * IN_DIM + i] * W[o * IN_DIM + i];
                }
                float gelu_z = 0.5f * z * (1.0f + std::tanh(SQRT_2_OVER_PI * (z + 0.044715f * z * z * z)));
                if (std::abs(Y[b * OUT_DIM + o] - gelu_z) > 1e-4f) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: Fused Linear + Bias + GELU Layer", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Cross-Entropy Loss with Log-Sum-Exp Trick
    //
    // Context: Computing softmax probabilities before taking the log produces underflow:
    //          CrossEntropy(z, y) = -z[y] + log(sum_j exp(z_j))
    //          The Log-Sum-Exp trick avoids numerical overflow:
    //            m = max_j(z_j)
    //            log(sum_j exp(z_j)) = m + log(sum_j exp(z_j - m))
    //
    // Task: Given logits `logits` of shape [BATCH=4, VOCAB=16] and target labels `targets` [BATCH=4]:
    //       Compute per-batch loss into `losses` [BATCH=4].
    // -------------------------------------------------------------------------
    {
        const int BATCH = 4, VOCAB = 16;
        std::vector<float> logits = {
            12.0f, 15.0f, 11.0f, 8.0f, 9.0f, 14.0f, 10.0f, 13.0f, 10.0f, 11.0f, 12.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f,
            -2.0f, -5.0f, -1.0f, -8.0f, -3.0f, -4.0f, -2.0f, -6.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
            80.0f, 85.0f, 90.0f, 75.0f, 88.0f, 82.0f, 84.0f, 86.0f, 81.0f, 83.0f, 85.0f, 79.0f, 78.0f, 77.0f, 76.0f, 75.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
        };

        std::vector<int> targets = {1, 2, 2, 15};
        std::vector<float> losses(BATCH, 0.0f);

        // TODO: For each batch b, compute loss using log-sum-exp over VOCAB elements.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int b = 0; b < BATCH && p3_passed; ++b) {
            float max_z = -1e30f;
            for (int v = 0; v < VOCAB; ++v) max_z = std::max(max_z, logits[b * VOCAB + v]);
            float sum_exp = 0.0f;
            for (int v = 0; v < VOCAB; ++v) sum_exp += std::exp(logits[b * VOCAB + v] - max_z);
            float lse = max_z + std::log(sum_exp);
            float expected_loss = -logits[b * VOCAB + targets[b]] + lse;
            if (std::abs(losses[b] - expected_loss) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Cross-Entropy Loss with Log-Sum-Exp Trick", p3_passed);
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
