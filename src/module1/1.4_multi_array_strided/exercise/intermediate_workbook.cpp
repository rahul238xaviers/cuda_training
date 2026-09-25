#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Multi-Array & Strided Ops
//
// Module: 1.4 - Multi-Array Linear Algebra & Memory Traversal
// Level:  Intermediate
//
// Focus: BLAS Level-2 GEMV (y = alpha*A*x + beta*y), fused Scale-Add-ReLU,
//        and 4-channel parallel zip-sum reductions.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.4_intermediate
//   ../../../output/1.4_intermediate
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
    std::cout << "--- WORKBOOK: Multi-Array & Strided Ops (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: BLAS Level-2 GEMV (y = alpha * A * x + beta * y)
    //
    // Context: Matrix-vector multiplication is the foundational operation in
    //          recurrent networks (RNN/LSTM) and LLM autoregressive token generation.
    //
    // Task: Given matrix `A` [M=32, K=64], vector `x` [K=64], and vector `y` [M=32]:
    //       Update y in-place with alpha=1.5f and beta=0.5f:
    //         y[m] = alpha * sum_{k=0..K-1}(A[m, k] * x[k]) + beta * y[m]
    // -------------------------------------------------------------------------
    {
        const int M = 32, K = 64;
        const float alpha = 1.5f, beta = 0.5f;

        std::vector<float> A(M * K);
        for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>((i % 13) - 6) * 0.1f;

        std::vector<float> x(K);
        for (int i = 0; i < K; ++i) x[i] = static_cast<float>((i % 7) + 1) * 0.2f;

        std::vector<float> y(M);
        for (int i = 0; i < M; ++i) y[i] = 1.0f;
        std::vector<float> original_y = y;

        // TODO: In-place GEMV update on y.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int m = 0; m < M && p1_passed; ++m) {
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                dot += A[m * K + k] * x[k];
            }
            float expected = alpha * dot + beta * original_y[m];
            if (std::abs(y[m] - expected) > 1e-4f) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: BLAS Level-2 GEMV (y = alpha*A*x + beta*y)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Fused Multi-Tensor Scale-Add-ReLU Memory Pass
    //
    // Context: In deep learning frameworks, fusing multiple elementwise operations
    //          into a single pass prevents round-tripping data through main DRAM.
    //
    // Task: Given three vectors `X`, `Y`, `Z` of size N=256:
    //       Compute `Out[i] = std::max(0.0f, 2.0f * X[i] - 1.5f * Y[i] + Z[i])`.
    // -------------------------------------------------------------------------
    {
        const int N = 256;
        std::vector<float> X(N), Y(N), Z(N);
        for (int i = 0; i < N; ++i) {
            X[i] = static_cast<float>((i % 17) - 8);
            Y[i] = static_cast<float>((i % 13) - 6);
            Z[i] = static_cast<float>((i % 7) - 3);
        }

        std::vector<float> Out(N, -1.0f);

        // TODO: Compute Out[i] using fused pointer arithmetic.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < N; ++i) {
            float val = 2.0f * X[i] - 1.5f * Y[i] + Z[i];
            float expected = std::max(0.0f, val);
            if (std::abs(Out[i] - expected) > 1e-4f) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: Fused Scale-Add-ReLU Memory Pass", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Strided Zip-Sum of 4 Parallel Channels
    //
    // Context: In multi-modal attention or multi-channel convolution, activations
    //          from separate channels are combined via dot-product accumulation:
    //            R[i] = A[i] * B[i] + C[i] * D[i]
    //
    // Task: Given 4 float arrays `A`, `B`, `C`, `D` of size N=128:
    //       Compute `R[i]` using SIMD-style parallel accumulation.
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        std::vector<float> A(N), B(N), C(N), D(N);
        for (int i = 0; i < N; ++i) {
            A[i] = static_cast<float>(i + 1);
            B[i] = static_cast<float>(i * 2 + 1);
            C[i] = static_cast<float>(i * 3 + 1);
            D[i] = static_cast<float>(i * 4 + 1);
        }

        std::vector<float> R(N, 0.0f);

        // TODO: Compute R[i] = A[i] * B[i] + C[i] * D[i].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            float expected = A[i] * B[i] + C[i] * D[i];
            if (std::abs(R[i] - expected) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: 4-Channel Parallel Zip-Sum Reduction", p3_passed);
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
