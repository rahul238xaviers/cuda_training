#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: ML Layer Offsets & Optimization
//
// Module: 1.19 - ML Tensor Memory Primitives
// Level:  Beginner
//
// Focus: Batched AdamW optimizer updates, 4D channel-broadcast bias addition,
//        and in-place residual skip connection memory traversal.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.19_beginner
//   ../../../output/1.19_beginner
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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Batched AdamW Weight & Momentum Update Step
    //
    // Context: In PyTorch and custom CUDA optimizer kernels, AdamW updates model weights:
    //            m_t = beta1 * m_{t-1} + (1 - beta1) * g
    //            v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    //            m_hat = m_t / (1 - beta1^t)
    //            v_hat = v_t / (1 - beta2^t)
    //            w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w_{t-1})
    //
    // Task: Perform in-place AdamW update on weight vector `W`, first-moment `M`,
    //       and second-moment `V` of size N=128 given gradient vector `G`.
    //       Constants: beta1=0.9f, beta2=0.999f, lr=1e-3f, wd=0.01f, eps=1e-8f, t=1.
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        const float beta1 = 0.9f, beta2 = 0.999f, lr = 1e-3f, wd = 0.01f, eps = 1e-8f;
        const int t = 1;
        const float corr1 = 1.0f - std::pow(beta1, t);
        const float corr2 = 1.0f - std::pow(beta2, t);

        std::vector<float> W(N), M(N, 0.0f), V(N, 0.0f), G(N);
        for (int i = 0; i < N; ++i) {
            W[i] = static_cast<float>(i * 0.05f - 3.2f);
            G[i] = static_cast<float>((i % 11) - 5) * 0.02f;
        }
        std::vector<float> original_W = W;

        // TODO: In a loop over i in [0, N), update M[i], V[i], and W[i] in-place.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int i = 0; i < N && p1_passed; ++i) {
            float exp_m = beta1 * 0.0f + (1.0f - beta1) * G[i];
            float exp_v = beta2 * 0.0f + (1.0f - beta2) * (G[i] * G[i]);
            float m_hat = exp_m / corr1;
            float v_hat = exp_v / corr2;
            float exp_w = original_W[i] - lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * original_W[i]);
            if (std::abs(W[i] - exp_w) > 1e-4f) p1_passed = false;
        }

        reportStatus("Problem 1: Batched AdamW Weight & Momentum Update", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Channel-Broadcast Bias Addition across 4D Activations
    //
    // Context: In Convolutional Layers, a 1D bias vector of size C must be broadcast
    //          and added to every spatial position (h, w) across all batch elements:
    //            X[n, c, h, w] += bias[c]
    //
    // Task: Given tensor `X` [N=2, C=4, H=4, W=4] and bias vector `bias` [C=4]:
    //       Add `bias[c]` in-place to all elements in channel c using strided pointers.
    // -------------------------------------------------------------------------
    {
        const int N = 2, C = 4, H = 4, W = 4;
        const size_t total_elements = N * C * H * W;
        std::vector<float> X(total_elements);
        for (size_t i = 0; i < total_elements; ++i) X[i] = static_cast<float>(i + 1);
        std::vector<float> original_X = X;

        std::vector<float> bias = {10.0f, -20.0f, 30.0f, -40.0f};

        // TODO: For each n in [0, N), c in [0, C), h in [0, H), w in [0, W):
        // Add bias[c] to X[n, c, h, w].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int n = 0; n < N && p2_passed; ++n) {
            for (int c = 0; c < C && p2_passed; ++c) {
                for (int h = 0; h < H && p2_passed; ++h) {
                    for (int w = 0; w < W; ++w) {
                        size_t idx = (size_t)n * (C * H * W) + (size_t)c * (H * W) + (size_t)h * W + w;
                        float expected = original_X[idx] + bias[c];
                        if (std::abs(X[idx] - expected) > 1e-4f) {
                            p2_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: 4D Channel-Broadcast Bias Addition", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: In-Place Residual Skip Connection Addition
    //
    // Context: In Transformers (pre-norm / post-norm architecture), after computing
    //          an attention or FFN sub-block, the residual is added:
    //            x = x + sublayer_output
    //
    // Task: Given activation tensor `X` and residual tensor `Residual` of size N=512:
    //       Add `Residual` into `X` in-place using pointer traversal.
    // -------------------------------------------------------------------------
    {
        const int N = 512;
        std::vector<float> X(N);
        std::vector<float> Residual(N);
        for (int i = 0; i < N; ++i) {
            X[i] = static_cast<float>(i * 0.1f);
            Residual[i] = static_cast<float>((i % 7) - 3) * 0.5f;
        }
        std::vector<float> original_X = X;

        // TODO: In-place add Residual into X.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(X[i] - (original_X[i] + Residual[i])) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: In-Place Residual Skip Connection Addition", p3_passed);
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
