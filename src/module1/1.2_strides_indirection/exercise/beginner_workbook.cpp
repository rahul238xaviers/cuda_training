#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Strides & Indirection
//
// Module: 1.2 - Strided Indexing & Indirection Patterns
// Level:  Beginner
//
// Focus: Subsampled strided decimation, column vector extraction from row-major
//        matrices, and pointer array indirection tables.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.2_beginner
//   ../../../output/1.2_beginner
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
    std::cout << "--- WORKBOOK: Strides & Indirection (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Strided Signal Decimation (Downsampling)
    //
    // Context: In audio preprocessing for speech models (Whisper), continuous 1D
    //          audio signals are decimated by factor K using strided pointer jumps:
    //            output[i] = input[i * K]
    //
    // Task: Given input signal `audio` of size N=128 floats, downsample by FACTOR=4
    //       into `downsampled` (size = N / 4 = 32 floats).
    //       Traverse `audio` using pointer increments `ptr += FACTOR`.
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        const int FACTOR = 4;
        const int OUT_SIZE = N / FACTOR; // 32

        std::vector<float> audio(N);
        for (int i = 0; i < N; ++i) audio[i] = static_cast<float>(i * 0.25f);

        std::vector<float> downsampled(OUT_SIZE, -1.0f);

        const float* ptr = audio.data();
        float* dst = downsampled.data();

        // TODO: Copy OUT_SIZE elements from audio into downsampled by advancing ptr by FACTOR.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int i = 0; i < OUT_SIZE; ++i) {
            if (downsampled[i] != audio[i * FACTOR]) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: Strided Signal Decimation (Downsampling)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Column Vector Extraction from Row-Major Matrix
    //
    // Context: In row-major matrices, consecutive column elements are separated by
    //          stride equal to the matrix width W. Accessing a column is a classic
    //          strided memory read.
    //
    // Task: Given matrix `mat` [ROWS=16, COLS=8], extract column `target_col = 3`
    //       into vector `col_vec` of size 16.
    //       Advance through `mat` using stride = COLS.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 16;
        const int COLS = 8;
        std::vector<float> mat(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                mat[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        const int target_col = 3;
        std::vector<float> col_vec(ROWS, 0.0f);

        // TODO: Extract column target_col using pointer stepping with stride COLS.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 0; r < ROWS; ++r) {
            if (col_vec[r] != mat[r * COLS + target_col]) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: Column Extraction via Non-Unit Strides", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Indirection Table of Row Pointers
    //
    // Context: Numerical libraries (e.g. NR in C, older BLAS interfaces) use an
    //          indirection table `float**` where table[r] points to row r of a matrix.
    //
    // Task: Given flat buffer `matrix_data` [ROWS=4, COLS=4]:
    //       1. Initialize `float* row_ptrs[4]` so row_ptrs[r] points to row r.
    //       2. Use double indirection `row_ptrs[r][c]` to compute the trace
    //          (sum of main diagonal elements where r == c).
    // -------------------------------------------------------------------------
    {
        const int ROWS = 4, COLS = 4;
        std::vector<float> matrix_data = {
            1.0f,  2.0f,  3.0f,  4.0f,
            5.0f,  6.0f,  7.0f,  8.0f,
            9.0f,  10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };

        float* row_ptrs[ROWS] = {nullptr};
        float trace = 0.0f;

        // TODO: Populate row_ptrs and compute trace using row_ptrs[i][i].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        float expected_trace = 1.0f + 6.0f + 11.0f + 16.0f; // 34.0f
        bool p3_passed = (std::abs(trace - expected_trace) < 1e-4f);
        for (int r = 0; r < ROWS; ++r) {
            if (row_ptrs[r] != &matrix_data[r * COLS]) p3_passed = false;
        }

        reportStatus("Problem 3: Pointer Indirection Table & Matrix Trace", p3_passed);
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
