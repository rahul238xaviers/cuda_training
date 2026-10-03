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
// CHAMPION WORKBOOK: Strides & Indirection
//
// Module: 1.2 - Strided Indexing & Indirection Patterns
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Compressed Sparse Row (CSR) SpMV indirection, generalized 3D
//        tensor stride permutation, and stride cache-thrashing profiling.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.2_champion
//   ../../../output/1.2_champion
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
    std::cout << "--- WORKBOOK: Strides & Indirection (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Compressed Sparse Row (CSR) Sparse Matrix-Vector Multiply (SpMV)
    //
    // Context: In Graph Neural Networks (PyG) and pruned LLMs, sparse matrices
    //          are stored in CSR format:
    //            - `row_ptr` [M+1]: index in `values` where each row begins
    //            - `col_indices` [NNZ]: column index of each non-zero element
    //            - `values` [NNZ]: the non-zero float values
    //          SpMV computes: y[r] = sum_{k=row_ptr[r]..row_ptr[r+1]-1} values[k] * x[col_indices[k]]
    //
    // Task: Implement CSR SpMV for M=4 rows, NNZ=6 non-zero values, and dense vector x [4].
    // -------------------------------------------------------------------------
    {
        const int M = 4;
        const int NNZ = 6;
        std::vector<int> row_ptr = {0, 2, 3, 5, 6};
        std::vector<int> col_indices = {0, 2, 1, 0, 3, 2};
        std::vector<float> values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> y(M, 0.0f);

        // TODO: Compute y[r] using CSR indirection traversal.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        // Row 0: 10*1 + 20*3 = 70
        // Row 1: 30*2 = 60
        // Row 2: 40*1 + 50*4 = 240
        // Row 3: 60*3 = 180
        std::vector<float> expected_y = {70.0f, 60.0f, 240.0f, 180.0f};

        bool p1_passed = true;
        for (int r = 0; r < M; ++r) {
            if (std::abs(y[r] - expected_y[r]) > 1e-4f) p1_passed = false;
        }

        reportStatus("Problem 1: Compressed Sparse Row (CSR) SpMV Indirection", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Generalized 3D Tensor Stride Permutation (D0, D1, D2 -> D0, D2, D1)
    //
    // Context: In GPU tensor frameworks, permuting dimensions requires calculating
    //          generalized strides:
    //            stride[dim] = prod_{d > dim} shape[d]
    //
    // Task: Given tensor `src` of shape [D0=4, D1=8, D2=16] (512 floats),
    //       permute to `dst` of shape [D0=4, D2=16, D1=8].
    //       Compute dynamic source and destination offsets for every element.
    // -------------------------------------------------------------------------
    {
        const int D0 = 4, D1 = 8, D2 = 16;
        const size_t total_elements = D0 * D1 * D2;
        std::vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src[i] = static_cast<float>(i + 1);

        std::vector<float> dst(total_elements, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Permute (d0, d1, d2) -> (d0, d2, d1).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p2_passed = true;
        for (int d0 = 0; d0 < D0 && p2_passed; ++d0) {
            for (int d1 = 0; d1 < D1 && p2_passed; ++d1) {
                for (int d2 = 0; d2 < D2; ++d2) {
                    size_t src_idx = (size_t)d0 * (D1 * D2) + (size_t)d1 * D2 + d2;
                    size_t dst_idx = (size_t)d0 * (D2 * D1) + (size_t)d2 * D1 + d1;
                    if (dst[dst_idx] != src[src_idx]) {
                        p2_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 2: Generalized 3D Tensor Stride Permutation", p2_passed, p2_passed ? throughput : -1.0);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Cache-Blocked 2D Matrix Transposition (Golden Tile Rule: TILE=32)
    //
    // Context: In GPU & CPU systems, naive matrix transpose causes strided writes
    //          that thrash cache lines across memory. Using the Golden Formula:
    //            T <= sqrt( (alpha * C_L1) / (2 * sizeof(float)) ) = 32
    //          We decompose a 1024x1024 matrix into 32x32 tiles that stay 100%
    //          resident inside L1 cache lines with zero eviction thrashing.
    //
    // Task: Transpose `src` of shape [1024, 1024] into `dst` of shape [1024, 1024]
    //       using 2D tiled traversal (TILE = 32).
    // -------------------------------------------------------------------------
    {
        const int Rows = 1024;
        const int Cols = 1024;
        const int TILE = 32;
        const size_t total_elements = (size_t)Rows * Cols;

        std::vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            src[i] = static_cast<float>(i + 1);
        }
        std::vector<float> dst(total_elements, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Implement 2D cache-blocked matrix transpose using TILE = 32:
        //   1. Outer loops step across tile coordinates: r0 and c0 with step TILE
        //   2. Inner loops step through elements within the tile: r and c
        //   3. Transpose element: dst[c * Rows + r] = src[r * Cols + c];
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int r = 0; r < Rows && p3_passed; ++r) {
            for (int c = 0; c < Cols; ++c) {
                if (dst[c * Rows + r] != src[r * Cols + c]) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: Cache-Blocked 2D Matrix Transpose (TILE=32)", p3_passed,
                     p3_passed ? throughput : -1.0);
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
