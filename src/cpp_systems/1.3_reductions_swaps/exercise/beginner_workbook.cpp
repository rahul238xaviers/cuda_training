#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Array Reductions & Swaps
//
// Module: 1.3 - Parallel Reduction Foundations & In-Place Memory Swapping
// Level:  Beginner
//
// Focus: In-place tree reduction, row-wise min/max reductions, and tensor
//        half-buffer pointer swaps.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.3_beginner
//   ../../../output/1.3_beginner
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
    std::cout << "--- WORKBOOK: Array Reductions & Swaps (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: In-Place Tree Reduction (CUDA Block Reduction Pattern)
    //
    // Context: In GPU shared memory reductions, threads perform pairwise additions
    //          with halving strides: s = N / 2, N / 4, ..., 1.
    //          After log2(N) steps, arr[0] contains the total sum of all elements.
    //
    // Given:
    //   - Array `arr` of size N = 64 floats.
    //
    // Goal:
    //   - Reduce `arr` in-place using halving strides until `arr[0]` contains
    //     the total sum of all 64 elements.
    //
    // Reference:
    //   - See Section 1 (Technique 2: Tree-Based In-Place Reduction) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 64;
        std::vector<float> arr(N);
        float expected_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            arr[i] = static_cast<float>(i + 1);
            expected_sum += arr[i];
        }

        // TODO: Perform in-place tree reduction so arr[0] holds the total sum.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (std::abs(arr[0] - expected_sum) < 1e-3f);
        reportStatus("Problem 1: In-Place Tree Reduction (CUDA Style)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Row-Wise Min and Max Vector Reduction
    //
    // Context: In bounding box calculations and activation clipping, finding the
    //          min and max value per row is a common reduction primitive.
    //
    // Given:
    //   - Matrix `mat` [ROWS=8, COLS=16].
    //   - Output vectors `min_vals` (size 8) and `max_vals` (size 8).
    //
    // Goal:
    //   - Compute `min_vals[r]` and `max_vals[r]` for each row using pointer traversal.
    //
    // Reference:
    //   - See Section 1 (Sequential Reduction Mechanics) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8, COLS = 16;
        std::vector<float> mat(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                mat[r * COLS + c] = static_cast<float>((r * 17 + c * 31) % 59 - 25);
            }
        }

        std::vector<float> min_vals(ROWS, 0.0f);
        std::vector<float> max_vals(ROWS, 0.0f);

        // TODO: Compute the min and max value for each row r into min_vals[r] and max_vals[r].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 0; r < ROWS && p2_passed; ++r) {
            float exp_min = 1e30f, exp_max = -1e30f;
            for (int c = 0; c < COLS; ++c) {
                float v = mat[r * COLS + c];
                exp_min = std::min(exp_min, v);
                exp_max = std::max(exp_max, v);
            }
            if (min_vals[r] != exp_min || max_vals[r] != exp_max) p2_passed = false;
        }

        reportStatus("Problem 2: Row-Wise Min/Max Reductions", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: In-Place Buffer Halves Swapping
    //
    // Context: In FFT (Fast Fourier Transform) frequency shift (`fftshift`), the
    //          first half of the spectrum is swapped with the second half in-place.
    //
    // Given:
    //   - Array `spectrum` of size 2*HALF_N (HALF_N=32, total 64 floats).
    //
    // Goal:
    //   - Swap `spectrum[i]` with `spectrum[i + HALF_N]` for all i in [0, HALF_N).
    //   - Zero auxiliary memory allocations allowed.
    //
    // Reference:
    //   - See Section 2 (In-Place Swapping: Pointer Swap vs. Memory Copy) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int HALF_N = 32;
        const int TOTAL = 2 * HALF_N;
        std::vector<float> spectrum(TOTAL);
        for (int i = 0; i < TOTAL; ++i) spectrum[i] = static_cast<float>(i * 10);
        std::vector<float> original = spectrum;

        // TODO: Swap elements of the first half with the second half in-place.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < HALF_N; ++i) {
            if (spectrum[i] != original[i + HALF_N] || spectrum[i + HALF_N] != original[i]) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: In-Place Buffer Halves Swapping", p3_passed);
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
