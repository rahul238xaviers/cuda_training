#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Row/Col-Major Indexing
//
// Module: 1.5 - Multi-Dimensional Layout Foundations
// Level:  Beginner
//
// Focus: Row-major to Column-major conversion, flat-to-2D coordinate
//        unflattening, and diagonal extraction from non-square matrices.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.5_beginner
//   ../../../output/1.5_beginner
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
    std::cout << "--- WORKBOOK: Row/Col-Major Indexing (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Row-Major to Column-Major Conversion
    //
    // Context: C++ and PyTorch store matrices in Row-Major order (consecutive row
    //          elements are contiguous). However, NVIDIA cuBLAS, Fortran, and MATLAB
    //          require Column-Major order (consecutive column elements are contiguous).
    //            Row-major index: r * COLS + c
    //            Col-major index: c * ROWS + r
    //
    // Task: Given row-major matrix `row_mat` of shape [ROWS=8, COLS=16] (128 floats):
    //       Pack elements into `col_mat` in Column-Major order.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8, COLS = 16;
        std::vector<float> row_mat(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                row_mat[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> col_mat(ROWS * COLS, -1.0f);

        // TODO: Populate col_mat in column-major order.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < ROWS && p1_passed; ++r) {
            for (int c = 0; c < COLS; ++c) {
                float val = col_mat[c * ROWS + r];
                float expected = row_mat[r * COLS + c];
                if (val != expected) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Row-Major to Column-Major Conversion", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Coordinate Unflattening (Linear Index -> 2D Coordinates)
    //
    // Context: In GPU kernels launched with 1D thread grids (e.g. threadIdx.x),
    //          each thread must unflatten its 1D global ID into 2D coordinates (r, c).
    //            r = flat_id / COLS
    //            c = flat_id % COLS
    //
    // Task: Given an array of linear indices `flat_ids` for a matrix with COLS=12:
    //       Compute `out_rows` and `out_cols`.
    // -------------------------------------------------------------------------
    {
        const int COLS = 12;
        std::vector<int> flat_ids = {0, 11, 12, 13, 23, 24, 47, 50, 95};
        const size_t N = flat_ids.size();

        std::vector<int> out_rows(N, -1);
        std::vector<int> out_cols(N, -1);

        // TODO: Unflatten each flat_id in flat_ids into out_rows[i] and out_cols[i].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (size_t i = 0; i < N; ++i) {
            int exp_r = flat_ids[i] / COLS;
            int exp_c = flat_ids[i] % COLS;
            if (out_rows[i] != exp_r || out_cols[i] != exp_c) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: 2D Coordinate Unflattening (div/mod)", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Main Diagonal Extraction from Non-Square Row-Major Matrix
    //
    // Context: In numerical linear algebra (SVD, QR decomposition), extracting the
    //          main diagonal from non-square matrices requires strided pointer stepping
    //          where stride = COLS + 1.
    //
    // Task: Given non-square matrix `mat` [ROWS=6, COLS=10]:
    //       Extract the diagonal elements (where r == c, length = min(ROWS, COLS) = 6)
    //       into `diag` using pointer increments of (COLS + 1).
    // -------------------------------------------------------------------------
    {
        const int ROWS = 6, COLS = 10;
        const int DIAG_LEN = std::min(ROWS, COLS); // 6

        std::vector<float> mat(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                mat[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> diag(DIAG_LEN, -1.0f);

        // TODO: Extract diagonal elements using pointer stepping.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < DIAG_LEN; ++i) {
            if (diag[i] != mat[i * COLS + i]) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Non-Square Matrix Diagonal Extraction (Stride COLS+1)", p3_passed);
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
