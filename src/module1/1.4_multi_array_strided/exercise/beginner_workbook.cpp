#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Multi-Array & Strided Ops
//
// Module: 1.4 - Multi-Array Linear Algebra & Memory Traversal
// Level:  Beginner
//
// Focus: BLAS Level-1 SAXPY throughput, strided Hadamard products, and
//        strided cosine similarity computation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.4_beginner
//   ../../../output/1.4_beginner
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " GB/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Multi-Array & Strided Ops (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: High-Performance SAXPY (Single-Precision A * X Plus Y)
    //
    // Context: BLAS-1 SAXPY (y = alpha * x + y) is the standard baseline benchmark
    //          for GPU memory streaming bandwidth.
    //
    // Task: Given vectors `X` and `Y` of size N=131,072 floats (512 KB each) and alpha=2.5f:
    //       Update Y in-place: Y[i] = alpha * X[i] + Y[i].
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int N = 131072;
        const float alpha = 2.5f;
        std::vector<float> X(N), Y(N);
        for (int i = 0; i < N; ++i) {
            X[i] = static_cast<float>(i * 0.01f);
            Y[i] = static_cast<float>((i % 100) * 0.5f);
        }
        std::vector<float> original_Y = Y;

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: In-place SAXPY: Y[i] = alpha * X[i] + Y[i] using pointer arithmetic.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 3.0 * N * sizeof(float); // Read X, Read Y, Write Y
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int i = 0; i < N; ++i) {
            float expected = alpha * X[i] + original_Y[i];
            if (std::abs(Y[i] - expected) > 1e-3f) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: BLAS Level-1 SAXPY Vector Stream", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Strided Hadamard Product (Element-Wise Multiplication)
    //
    // Context: In gating mechanisms (e.g. GRU / LSTM gates), two activations are
    //          multiplied elementwise with stride skips: Z[i] = X[i * stride_x] * Y[i * stride_y].
    //
    // Task: Given `X` of size 128 (stride_x = 2) and `Y` of size 192 (stride_y = 3):
    //       Compute 64 elementwise products into `Z` [64].
    // -------------------------------------------------------------------------
    {
        const int OUT_N = 64;
        const int STRIDE_X = 2;
        const int STRIDE_Y = 3;

        std::vector<float> X(OUT_N * STRIDE_X);
        std::vector<float> Y(OUT_N * STRIDE_Y);
        for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < Y.size(); ++i) Y[i] = static_cast<float>((i % 10) + 1);

        std::vector<float> Z(OUT_N, 0.0f);

        // TODO: Compute Z[i] = X[i * STRIDE_X] * Y[i * STRIDE_Y].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < OUT_N; ++i) {
            float expected = X[i * STRIDE_X] * Y[i * STRIDE_Y];
            if (std::abs(Z[i] - expected) > 1e-4f) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: Strided Hadamard Element-Wise Product", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Strided Cosine Similarity
    //
    // Context: In vector databases (RAG embeddings), cosine similarity measures
    //          the angle between embedding vectors:
    //            cos(A, B) = (A . B) / (||A||_2 * ||B||_2)
    //
    // Task: Given two vectors `A` and `B` of size N=128:
    //       Compute their dot product, both L2 norms, and cosine similarity.
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        std::vector<float> A(N), B(N);
        for (int i = 0; i < N; ++i) {
            A[i] = static_cast<float>((i % 7) - 3);
            B[i] = static_cast<float>((i % 5) - 2);
        }

        float cosine_sim = 0.0f;

        // TODO: Compute dot product and L2 norms to find cosine_sim.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < N; ++i) {
            dot += A[i] * B[i];
            norm_a += A[i] * A[i];
            norm_b += B[i] * B[i];
        }
        float expected_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));

        bool p3_passed = (std::abs(cosine_sim - expected_sim) < 1e-4f);
        reportStatus("Problem 3: Dual-Vector Cosine Similarity", p3_passed);
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
