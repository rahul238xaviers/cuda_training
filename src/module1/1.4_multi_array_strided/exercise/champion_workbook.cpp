#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>

// =========================================================================
// CHAMPION WORKBOOK: Multi-Array & Strided Ops
//
// Module: 1.4 - Multi-Array Linear Algebra & Memory Traversal
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Batched GEMV memory bandwidth benchmarking, on-the-fly transposed
//        matrix addition (C = A + B^T), and planar-to-interleaved RGB packing.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.4_champion
//   ../../../output/1.4_champion
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
    std::cout << "--- WORKBOOK: Multi-Array & Strided Ops (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Batched GEMV Bandwidth Benchmark (LLM Generation Simulation)
    //
    // Context: In LLM token decoding (batch size B > 1), each token vector in the batch
    //          is multiplied by the shared weight matrix W [M, K].
    //          Y[b, m] = sum_{k=0..K-1} W[m, k] * X[b, k]
    //
    // Task: Given shared weight matrix `W` [M=512, K=1024] (2 MB) and input batch `X` [B=4, K=1024]:
    //       Compute output batch `Y` [B=4, M=512].
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int B = 4, M = 512, K = 1024;
        std::vector<float> W(M * K);
        for (size_t i = 0; i < W.size(); ++i) W[i] = static_cast<float>((i % 19) - 9) * 0.05f;

        std::vector<float> X(B * K);
        for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<float>((i % 11) - 5) * 0.1f;

        std::vector<float> Y(B * M, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Compute Batched GEMV Y[b, m] = sum_k(W[m, k] * X[b, k]).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        // Memory moved: Read W once (M * K), Read X for all B (B * K), Write Y (B * M)
        double bytes_moved = ((size_t)M * K + (size_t)B * K + (size_t)B * M) * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int b = 0; b < B && p1_passed; ++b) {
            for (int m = 0; m < M; ++m) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected += W[m * K + k] * X[b * K + k];
                }
                if (std::abs(Y[b * M + m] - expected) > 1e-3f) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: Batched GEMV Memory Throughput Benchmark", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: On-The-Fly Strided Transposed Matrix Addition (C = A + B^T)
    //
    // Context: In attention matrix symmetrization or graph adjacency updates, we often
    //          need C = A + B^T without allocating a separate temporary matrix for B^T.
    //
    // Task: Given matrices `A` and `B` of shape [DIM=256, DIM=256] (total 65,536 floats each):
    //       Compute `C[i, j] = A[i, j] + B[j, i]`.
    // -------------------------------------------------------------------------
    {
        const int DIM = 256;
        const size_t total_elements = DIM * DIM;

        std::vector<float> A(total_elements);
        std::vector<float> B(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            A[i] = static_cast<float>(i * 0.1f);
            B[i] = static_cast<float>((i % 100) * 0.2f);
        }

        std::vector<float> C(total_elements, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Compute C[i, j] = A[i * DIM + j] + B[j * DIM + i].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 3.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p2_passed = true;
        for (int i = 0; i < DIM && p2_passed; ++i) {
            for (int j = 0; j < DIM; ++j) {
                float expected = A[i * DIM + j] + B[j * DIM + i];
                if (std::abs(C[i * DIM + j] - expected) > 1e-4f) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: On-The-Fly Transposed Addition (C = A + B^T)", p2_passed, throughput);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Planar to Interleaved RGB Image Conversion
    //
    // Context: Deep learning vision models use Planar format (RRR... GGG... BBB...),
    //          while display hardware and image codecs require Interleaved (RGBRGBRGB...).
    //
    // Task: Given planar image `planar_img` [C=3, H=32, W=32]:
    //       Pack into `interleaved_img` [H=32, W=32, C=3] using strided writes.
    // -------------------------------------------------------------------------
    {
        const int C = 3, H = 32, W = 32;
        const int SPATIAL = H * W; // 1024
        const size_t total_elements = C * SPATIAL;

        std::vector<float> planar_img(total_elements);
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < SPATIAL; ++i) {
                planar_img[c * SPATIAL + i] = static_cast<float>(c * 1000 + i);
            }
        }

        std::vector<float> interleaved_img(total_elements, -1.0f);

        // TODO: For each spatial pixel (h, w), pack R, G, B channels contiguously into interleaved_img.
        // Interleaved offset for (h, w, c): (h * W + w) * 3 + c
        // Planar offset for (c, h, w): c * (H * W) + h * W + w
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int h = 0; h < H && p3_passed; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    float val = interleaved_img[(h * W + w) * C + c];
                    float expected = planar_img[c * SPATIAL + (h * W + w)];
                    if (val != expected) {
                        p3_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 3: Planar to Interleaved RGB Image Packing", p3_passed);
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
