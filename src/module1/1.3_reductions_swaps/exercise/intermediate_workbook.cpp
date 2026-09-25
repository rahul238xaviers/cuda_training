#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Array Reductions & Swaps
//
// Module: 1.3 - Parallel Reduction Foundations & In-Place Memory Swapping
// Level:  Intermediate
//
// Focus: 256-element power-of-2 tree reduction, in-place two-pointer
//        threshold partitioning, and batched L2-norm calculations.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.3_intermediate
//   ../../../output/1.3_intermediate
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
    std::cout << "--- WORKBOOK: Array Reductions & Swaps (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 256-Element Power-of-2 Tree Reduction
    //
    // Context: In CUDA thread-block reductions (e.g. 256 threads per block), threads
    //          collaborate across log2(256) = 8 steps to sum 256 values:
    //          stride = 128, 64, 32, 16, 8, 4, 2, 1.
    //
    // Task: Given array `data` of size N=256:
    //       Perform in-place tree reduction to store total sum in data[0].
    // -------------------------------------------------------------------------
    {
        const int N = 256;
        std::vector<float> data(N);
        float expected_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i * 0.1f + 0.5f);
            expected_sum += data[i];
        }

        // TODO: In-place tree reduction over 8 halving steps.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (std::abs(data[0] - expected_sum) < 1e-2f);
        reportStatus("Problem 1: 256-Element Tree Reduction", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: In-Place Two-Pointer Threshold Partitioning (LLM Sampler Primitives)
    //
    // Context: In Top-P (nucleus) and threshold sampling, logits are partitioned
    //          around a threshold without allocating auxiliary memory.
    //
    // Task: Given array `logits` of size N=64 and threshold PIVOT=2.0f:
    //       Partition `logits` in-place so that all elements >= PIVOT appear
    //       at the front (indices 0..num_above - 1), and all elements < PIVOT
    //       appear at the back (indices num_above..N - 1).
    //       Record the count of elements >= PIVOT in `num_above`.
    // -------------------------------------------------------------------------
    {
        const int N = 64;
        const float PIVOT = 2.0f;
        std::vector<float> logits(N);
        int expected_above = 0;
        for (int i = 0; i < N; ++i) {
            logits[i] = static_cast<float>((i % 11) - 5) * 0.75f;
            if (logits[i] >= PIVOT) expected_above++;
        }

        int num_above = 0;

        // TODO: Two-pointer partition: left = 0, right = N - 1.
        // Swap elements when logits[left] < PIVOT and logits[right] >= PIVOT.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = (num_above == expected_above);
        for (int i = 0; i < num_above && p2_passed; ++i) {
            if (logits[i] < PIVOT) p2_passed = false;
        }
        for (int i = num_above; i < N && p2_passed; ++i) {
            if (logits[i] >= PIVOT) p2_passed = false;
        }

        reportStatus("Problem 2: In-Place Two-Pointer Threshold Partitioning", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Batched L2-Norm Vector Reduction
    //
    // Context: In Cosine Similarity, Weight Normalization, and Gradient Clipping,
    //          calculating the L2 norm of vectors is a core linear algebra reduction:
    //            norm[b] = sqrt( sum_{d=0}^{D-1} X[b, d]^2 )
    //
    // Task: Given batch tensor `X` [BATCH=8, DIM=64] (512 floats):
    //       Compute the L2 norm for each batch vector into `norms` [BATCH=8].
    // -------------------------------------------------------------------------
    {
        const int BATCH = 8, DIM = 64;
        std::vector<float> X(BATCH * DIM);
        for (size_t i = 0; i < X.size(); ++i) {
            X[i] = static_cast<float>((i % 17) - 8) * 0.25f;
        }

        std::vector<float> norms(BATCH, 0.0f);

        // TODO: For each batch b, compute sqrt(sum(X[b, d]^2)).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int b = 0; b < BATCH && p3_passed; ++b) {
            float sum_sq = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                float v = X[b * DIM + d];
                sum_sq += v * v;
            }
            float expected = std::sqrt(sum_sq);
            if (std::abs(norms[b] - expected) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Batched L2-Norm Vector Reduction", p3_passed);
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
