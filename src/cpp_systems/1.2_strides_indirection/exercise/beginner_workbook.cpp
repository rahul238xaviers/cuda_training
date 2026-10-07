#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

using namespace std;

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
    // Given:
    //   - Input signal `audio` of size N = 128 floats.
    //   - Downsampling factor FACTOR = 4.
    //   - Output buffer `downsampled` of size OUT_SIZE = 32 floats.
    //   - Base pointers `ptr = audio.data()` and `dst = downsampled.data()`.
    //
    // Goal:
    //   - Downsample `audio` into `downsampled` by stepping across elements with stride FACTOR.
    //
    // Reference:
    //   - See Section 1 (Physical Memory vs. Multidimensional Tensors: Stride Representation) of theory.md.
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

        // TODO: Downsample the audio signal into the output buffer using strided pointer traversal.
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
    // Given:
    //   - Matrix `mat` of shape [ROWS=16, COLS=8] in row-major order.
    //   - Target column index: `target_col = 3`.
    //   - Output vector `col_vec` of size 16.
    //
    // Goal:
    //   - Extract all elements of column `target_col` into `col_vec` using strided offsets.
    //
    // Reference:
    //   - See Section 2 (Row-Major vs. Column-Major Layouts) of theory.md.
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

        // TODO: Extract the column vector from the row-major matrix using strided pointer offsets.
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
    // Given:
    //   - Flat matrix buffer `matrix_data` [ROWS=4, COLS=4].
    //   - Array of row pointers `row_ptrs` of size ROWS = 4.
    //
    // Goal:
    //   - Set each `row_ptrs[r]` to point to the start of row r in `matrix_data`.
    //   - Compute the matrix trace (sum of diagonal elements) using double indirection.
    //
    // Reference:
    //   - See Section 3 (Pointer Indirection: The Two Mental Models - Model B) of theory.md.
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

        // TODO: Populate row pointers and calculate matrix trace using double indirection.
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
