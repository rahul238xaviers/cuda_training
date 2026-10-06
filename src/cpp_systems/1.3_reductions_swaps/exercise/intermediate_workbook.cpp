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
    // PROBLEM 1: Non-Power-of-2 In-Place Tree Reduction (Odd-Length Tensor Folding)
    //
    // Context: Real deep learning sequence lengths (e.g. N=125 tokens) are rarely
    //          exact powers of 2. In parallel tree reduction, halving an odd length
    //          leaves un-paired tail elements that must be folded into the reduction.
    //
    // Given:
    //   - Array `data` of odd size N = 125 floats.
    //
    // Goal:
    //   - Perform in-place tree reduction folding halves while ensuring odd tail
    //     elements are folded into the reduction until the total sum resides in data[0].
    //
    // Reference:
    //   - See Section 1 (Technique 2B: Non-Power-of-2 & Odd-Length Boundary Folding) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 125;
        std::vector<float> data(N);
        float expected_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i * 0.1f + 0.5f);
            expected_sum += data[i];
        }

        // TODO: Reduce `data` in-place so data[0] contains the total sum of all N elements.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (std::abs(data[0] - expected_sum) < 1e-2f);
        reportStatus("Problem 1: Non-Power-of-2 Tree Reduction (N=125)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: In-Place Two-Pointer Threshold Partitioning (LLM Sampler Primitives)
    //
    // Context: In Top-P (nucleus) and threshold sampling, logits are partitioned
    //          around a threshold without allocating auxiliary memory.
    //
    // Given:
    //   - Array `logits` of size N = 64 floats.
    //   - Threshold value `PIVOT = 2.0f`.
    //
    // Goal:
    //   - Partition `logits` in-place so that all elements >= PIVOT appear at the
    //     front (indices 0..num_above - 1), and all elements < PIVOT appear at
    //     the back (indices num_above..N - 1).
    //   - Count elements >= PIVOT and store in `num_above`.
    //   - Zero auxiliary memory allocations allowed.
    //
    // Reference:
    //   - See Section 2B (In-Place Two-Pointer Partitioning) of theory.md.
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

        // TODO: Partition logits in-place around PIVOT and record the count in num_above.
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
    // Given:
    //   - Batch tensor `X` with dimensions [BATCH=8, DIM=64] (512 floats total).
    //   - Output vector `norms` of size BATCH=8 (initialized to 0.0f).
    //
    // Goal:
    //   - Compute the L2 norm for each batch vector b into `norms[b]`.
    //
    // Reference:
    //   - See Section 1 (Sequential Reduction Mechanics & Accumulation) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int BATCH = 8, DIM = 64;
        std::vector<float> X(BATCH * DIM);
        for (size_t i = 0; i < X.size(); ++i) {
            X[i] = static_cast<float>((i % 17) - 8) * 0.25f;
        }

        std::vector<float> norms(BATCH, 0.0f);

        // TODO: Compute the L2 norm for each batch vector b into norms[b].
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
