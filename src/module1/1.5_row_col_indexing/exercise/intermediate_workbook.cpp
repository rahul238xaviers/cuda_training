#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Row/Col-Major Indexing
//
// Module: 1.5 - Multi-Dimensional Layout Foundations
// Level:  Intermediate
//
// Focus: cuBLAS column-major memory marshaling, 3D volumetric voxel grid
//        axial/coronal slicing, and pitched 2D array indexing.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.5_intermediate
//   ../../../output/1.5_intermediate
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
    std::cout << "--- WORKBOOK: Row/Col-Major Indexing (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: cuBLAS Column-Major Memory Marshaling
    //
    // Context: Because cuBLAS expects column-major matrices, passing a C++ row-major
    //          matrix of shape [M, N] directly treats it as a column-major matrix
    //          of shape [N, M] (its transpose). To pass matrix A [M=4, N=8] without
    //          runtime transposition, we marshal it into an explicit column-major buffer.
    //
    // Task: Given row-major buffer `A_row` [M=4, N=8]:
    //       Marshal into `A_col` [N columns, each containing M elements].
    //       Verify that A_col[c * M + r] == A_row[r * N + c].
    // -------------------------------------------------------------------------
    {
        const int M = 4, N = 8;
        std::vector<float> A_row(M * N);
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                A_row[r * N + c] = static_cast<float>(r * 10 + c + 1);
            }
        }

        std::vector<float> A_col(M * N, 0.0f);

        // TODO: Marshal A_row into A_col in column-major order.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < M && p1_passed; ++r) {
            for (int c = 0; c < N; ++c) {
                if (A_col[c * M + r] != A_row[r * N + c]) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: cuBLAS Column-Major Memory Marshaling", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 3D Volumetric Voxel Grid Axial & Coronal Slicing
    //
    // Context: In 3D computer vision and CT/MRI medical imaging, 3D grids
    //          are stored in Depth-Height-Width hierarchy:
    //            flat_idx = d * (H * W) + h * W + w
    //          An Axial slice is a 2D plane at fixed depth d: [H, W].
    //          A Coronal slice is a 2D plane at fixed height h: [D, W].
    //
    // Task: Given 3D grid `volume` of shape [D=4, H=8, W=8] (256 floats):
    //       1. Extract Axial slice at d=2 into `axial_slice` [H=8, W=8]
    //       2. Extract Coronal slice at h=3 into `coronal_slice` [D=4, W=8]
    // -------------------------------------------------------------------------
    {
        const int D = 4, H = 8, W = 8;
        std::vector<float> volume(D * H * W);
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    volume[d * (H * W) + h * W + w] = static_cast<float>(d * 1000 + h * 10 + w);
                }
            }
        }

        const int target_d = 2;
        const int target_h = 3;

        std::vector<float> axial_slice(H * W, -1.0f);
        std::vector<float> coronal_slice(D * W, -1.0f);

        // TODO: Extract axial_slice at target_d and coronal_slice at target_h.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int h = 0; h < H && p2_passed; ++h) {
            for (int w = 0; w < W; ++w) {
                float expected = volume[target_d * (H * W) + h * W + w];
                if (axial_slice[h * W + w] != expected) p2_passed = false;
            }
        }
        for (int d = 0; d < D && p2_passed; ++d) {
            for (int w = 0; w < W; ++w) {
                float expected = volume[d * (H * W) + target_h * W + w];
                if (coronal_slice[d * W + w] != expected) p2_passed = false;
            }
        }

        reportStatus("Problem 2: 3D Voxel Grid Axial & Coronal Slicing", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Pitched 2D Array Dynamic Indexing
    //
    // Context: In GPU memory allocations (`cudaMallocPitch`), rows are padded
    //          to PITCH_ELEMENTS (e.g. 16 floats) even when WIDTH=10 floats.
    //          Formula for element (r, c): index = r * PITCH_ELEMENTS + c.
    //
    // Task: Given pitched 2D buffer of ROWS=8, WIDTH=10, PITCH_ELEMENTS=16:
    //       Fill each valid element (r, c) with (r * 100 + c).
    //       Ensure padding elements (from column 10 to 15 in each row) remain untouched (-1.0f).
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8, WIDTH = 10, PITCH_ELEMENTS = 16;
        std::vector<float> pitched_grid(ROWS * PITCH_ELEMENTS, -1.0f);

        // TODO: Populate valid elements in pitched_grid.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int r = 0; r < ROWS && p3_passed; ++r) {
            for (int c = 0; c < WIDTH; ++c) {
                float expected = static_cast<float>(r * 100 + c);
                if (pitched_grid[r * PITCH_ELEMENTS + c] != expected) {
                    p3_passed = false;
                    break;
                }
            }
            // Check padding
            for (int p = WIDTH; p < PITCH_ELEMENTS; ++p) {
                if (pitched_grid[r * PITCH_ELEMENTS + p] != -1.0f) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: Pitched 2D Array Dynamic Indexing", p3_passed);
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
