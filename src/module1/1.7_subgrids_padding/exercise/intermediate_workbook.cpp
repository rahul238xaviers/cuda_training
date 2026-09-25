#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Subgrids, Padding & Diagonals
//
// Module: 1.7 - Multi-Dimensional Layout Foundations
// Level:  Intermediate
//
// Focus: 2D Reflection/Mirror padding, tiled GEMM subgrid base pointer
//        resolution, and matrix symmetrization.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.7_intermediate
//   ../../../output/1.7_intermediate
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
    std::cout << "--- WORKBOOK: Subgrids, Padding & Diagonals (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Reflection / Mirror Padding
    //
    // Context: Zero-padding introduces dark border artifacts in image restoration.
    //          Reflection padding mirrors the pixels across the boundary without repeating
    //          the edge pixel:
    //            Border mapping for index x with pad P:
    //              if x < 0: x = -x
    //              if x >= W: x = 2*(W - 1) - x
    //
    // Task: Given input image `img` [H=4, W=4] and PAD=2:
    //       Generate `mirror_padded` [PADDED_H=8, PADDED_W=8].
    // -------------------------------------------------------------------------
    {
        const int H = 4, W = 4, P = 2;
        const int PADDED_H = H + 2 * P; // 8
        const int PADDED_W = W + 2 * P; // 8

        std::vector<float> img(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                img[r * W + c] = static_cast<float>(r * 10 + c);
            }
        }

        std::vector<float> mirror_padded(PADDED_H * PADDED_W, -1.0f);

        // TODO: Populate mirror_padded using reflection boundary coordinates.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int pr = 0; pr < PADDED_H && p1_passed; ++pr) {
            for (int pc = 0; pc < PADDED_W; ++pc) {
                int r = pr - P;
                int c = pc - P;
                // Reflection logic
                if (r < 0) r = -r;
                else if (r >= H) r = 2 * (H - 1) - r;
                if (c < 0) c = -c;
                else if (c >= W) c = 2 * (W - 1) - c;

                float expected = img[r * W + c];
                if (mirror_padded[pr * PADDED_W + pc] != expected) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Reflection/Mirror Padding", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Tiled GEMM Subgrid Base Pointer Resolution
    //
    // Context: In high-performance matrix multiplication (CUTLASS / cuBLAS), large
    //          matrices are broken into thread-block tiles (e.g. 16x16).
    //          Each thread-block computes the base address of its input tiles.
    //
    // Task: Given parent matrix `A` [M=64, K=64] and TILE_DIM=16:
    //       For any tile coordinates (ti, tj) where ti in [0..3], tj in [0..3]:
    //       Resolve the exact pointer `tile_ptr` pointing to the top-left element of tile (ti, tj).
    //       Verify all 16 tile base pointers!
    // -------------------------------------------------------------------------
    {
        const int M = 64, K = 64, TILE_DIM = 16;
        const int TILES_M = M / TILE_DIM; // 4
        const int TILES_K = K / TILE_DIM; // 4

        std::vector<float> A(M * K);
        for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i + 1);

        bool p2_passed = true;
        for (int ti = 0; ti < TILES_M && p2_passed; ++ti) {
            for (int tj = 0; tj < TILES_K; ++tj) {
                const float* tile_ptr = nullptr;

                // TODO: Compute pointer to top-left of tile (ti, tj) in A.
                // --- YOUR CODE STARTS HERE ---

                // --- YOUR CODE ENDS HERE ---

                size_t expected_offset = (size_t)ti * TILE_DIM * K + tj * TILE_DIM;
                if (tile_ptr != &A[expected_offset]) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: Tiled GEMM Subgrid Base Pointer Resolution", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: In-Place Matrix Symmetrization (A = 0.5 * (A + A^T))
    //
    // Context: Covariance matrices and graph adjacency matrices must be strictly
    //          symmetric. Symmetrizing in-place requires setting:
    //            A[i, j] = A[j, i] = 0.5 * (A[i, j] + A[j, i])
    //
    // Task: Given matrix `A` of shape [N=16, N=16]:
    //       Symmetrize A in-place without allocating an extra matrix!
    // -------------------------------------------------------------------------
    {
        const int N = 16;
        std::vector<float> A(N * N);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                A[r * N + c] = static_cast<float>(r * 100 + c * 2);
            }
        }
        std::vector<float> original = A;

        // TODO: Symmetrize A in-place.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int r = 0; r < N && p3_passed; ++r) {
            for (int c = 0; c < N; ++c) {
                float expected = 0.5f * (original[r * N + c] + original[c * N + r]);
                if (std::abs(A[r * N + c] - expected) > 1e-4f || A[r * N + c] != A[c * N + r]) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: In-Place Matrix Symmetrization", p3_passed);
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
