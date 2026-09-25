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
// CHAMPION WORKBOOK: Row/Col-Major Indexing
//
// Module: 1.5 - Multi-Dimensional Layout Foundations
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: In-place square matrix transposition benchmark, cuBLAS leading
//        dimension (lda) sub-matrix layout prep, and 4D tensor unravelling.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.5_champion
//   ../../../output/1.5_champion
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
    std::cout << "--- WORKBOOK: Row/Col-Major Indexing (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D In-Place Square Matrix Transposition Benchmark
    //
    // Context: Transposing a square matrix in-place without allocating a second matrix
    //          is an important memory optimization. It requires swapping A[i, j]
    //          with A[j, i] for all j > i.
    //
    // Task: Given square matrix `A` [DIM=512, DIM=512] (262,144 floats = 1 MB):
    //       Transpose A in-place.
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int DIM = 512;
        const size_t total_elements = DIM * DIM;
        std::vector<float> A(total_elements);
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                A[i * DIM + j] = static_cast<float>(i * 1000 + j);
            }
        }
        std::vector<float> original_A = A;

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: In-place transpose: for i = 0..DIM-1, for j = i+1..DIM-1:
        // swap A[i*DIM + j] with A[j*DIM + i].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        // Number of elements swapped = DIM*(DIM-1)/2, each swap moves 2 elements (read+write)
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int i = 0; i < DIM && p1_passed; ++i) {
            for (int j = 0; j < DIM; ++j) {
                if (A[i * DIM + j] != original_A[j * DIM + i]) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D In-Place Square Matrix Transpose", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: cuBLAS Leading Dimension (lda) Strided Submatrix Layout Prep
    //
    // Context: In cuBLAS, a sub-matrix of shape [M, N] inside a larger column-major
    //          matrix of height M_PARENT has leading dimension lda = M_PARENT.
    //          Column c of the submatrix starts at: base + col_start * lda + row_start.
    //
    // Task: Given parent column-major matrix `parent_col` of shape [M_PARENT=64, N_PARENT=64],
    //       extract the submatrix of shape [M=16, N=16] located at row_offset=8, col_offset=16
    //       into a compact row-major buffer `sub_row` [16, 16].
    // -------------------------------------------------------------------------
    {
        const int M_PARENT = 64, N_PARENT = 64;
        const int LDA = M_PARENT; // 64
        const int M = 16, N = 16;
        const int ROW_OFFSET = 8, COL_OFFSET = 16;

        std::vector<float> parent_col(M_PARENT * N_PARENT);
        for (int c = 0; c < N_PARENT; ++c) {
            for (int r = 0; r < M_PARENT; ++r) {
                parent_col[c * LDA + r] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> sub_row(M * N, -1.0f);

        // TODO: Extract submatrix into sub_row in row-major order: sub_row[r * N + c].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 0; r < M && p2_passed; ++r) {
            for (int c = 0; c < N; ++c) {
                int orig_r = ROW_OFFSET + r;
                int orig_c = COL_OFFSET + c;
                float expected = parent_col[orig_c * LDA + orig_r];
                if (sub_row[r * N + c] != expected) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: cuBLAS Leading Dimension Submatrix Prep", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: 4D Tensor Coordinate Unravelling (flat_idx -> b, c, h, w)
    //
    // Context: In GPU debuggers and CUDA error tracking, kernel crash reports
    //          give a linear buffer index. The developer must unravel this index
    //          into 4D tensor coordinates (b, c, h, w) to locate the defect.
    //
    // Task: Given tensor shape [B=4, C=8, H=16, W=16] (16,384 floats):
    //       For an array of 5 query indices `query_indices`:
    //       Unravel each into `out_b`, `out_c`, `out_h`, `out_w`.
    //       Strides: stride_b = C*H*W, stride_c = H*W, stride_h = W, stride_w = 1.
    // -------------------------------------------------------------------------
    {
        const int B = 4, C = 8, H = 16, W = 16;
        const int STRIDE_H = W;
        const int STRIDE_C = H * W;
        const int STRIDE_B = C * H * W;

        std::vector<int> query_indices = {0, 15, 255, 256, 12345};
        const size_t NUM_QUERIES = query_indices.size();

        std::vector<int> out_b(NUM_QUERIES, -1);
        std::vector<int> out_c(NUM_QUERIES, -1);
        std::vector<int> out_h(NUM_QUERIES, -1);
        std::vector<int> out_w(NUM_QUERIES, -1);

        // TODO: Unravel each index in query_indices.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (size_t i = 0; i < NUM_QUERIES && p3_passed; ++i) {
            int rem = query_indices[i];
            int exp_b = rem / STRIDE_B; rem %= STRIDE_B;
            int exp_c = rem / STRIDE_C; rem %= STRIDE_C;
            int exp_h = rem / STRIDE_H; rem %= STRIDE_H;
            int exp_w = rem;

            if (out_b[i] != exp_b || out_c[i] != exp_c || out_h[i] != exp_h || out_w[i] != exp_w) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: 4D Tensor Coordinate Unravelling (b, c, h, w)", p3_passed);
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
