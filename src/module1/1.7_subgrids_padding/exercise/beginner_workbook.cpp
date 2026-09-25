#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Subgrids, Padding & Diagonals
//
// Module: 1.7 - Multi-Dimensional Layout Foundations
// Level:  Beginner
//
// Focus: 2D Zero-Padding for convolution prep, ROI subgrid cropping,
//        and tridiagonal matrix band extraction.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.7_beginner
//   ../../../output/1.7_beginner
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
    std::cout << "--- WORKBOOK: Subgrids, Padding & Diagonals (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Zero-Padding for Convolution Prep
    //
    // Context: In Convolutional Neural Networks, spatial feature maps of size [H, W]
    //          are padded with P zeros along all 4 borders, producing an enlarged
    //          tensor of size [H + 2*P, W + 2*P].
    //
    // Task: Given input `in_feat` [H=6, W=6] and padding P=1:
    //       Embed `in_feat` into the center of `padded_feat` [PADDED_H=8, PADDED_W=8].
    //       Ensure all border pixels (row 0, row 7, col 0, col 7) are 0.0f.
    // -------------------------------------------------------------------------
    {
        const int H = 6, W = 6, P = 1;
        const int PADDED_H = H + 2 * P; // 8
        const int PADDED_W = W + 2 * P; // 8

        std::vector<float> in_feat(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                in_feat[r * W + c] = static_cast<float>(r * 10 + c + 1);
            }
        }

        std::vector<float> padded_feat(PADDED_H * PADDED_W, -999.0f);

        // TODO: Embed in_feat into the center of padded_feat and fill borders with 0.0f.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < PADDED_H && p1_passed; ++r) {
            for (int c = 0; c < PADDED_W; ++c) {
                float val = padded_feat[r * PADDED_W + c];
                bool is_border = (r < P || r >= H + P || c < P || c >= W + P);
                if (is_border) {
                    if (val != 0.0f) p1_passed = false;
                } else {
                    float expected = in_feat[(r - P) * W + (c - P)];
                    if (val != expected) p1_passed = false;
                }
            }
        }

        reportStatus("Problem 1: 2D Zero-Padding for Convolution Prep", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Subgrid Crop (ROI Extraction)
    //
    // Context: In object detection (Faster R-CNN / YOLO), bounding box regions
    //          of interest (ROIs) are cropped out of a parent feature map.
    //
    // Task: Given parent feature map `parent` [H=16, W=16]:
    //       Extract subgrid with top-left (r0=4, c0=6) and dimensions [CROP_H=4, CROP_W=4]
    //       into contiguous buffer `cropped` [16].
    // -------------------------------------------------------------------------
    {
        const int H = 16, W = 16;
        std::vector<float> parent(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                parent[r * W + c] = static_cast<float>(r * 100 + c);
            }
        }

        const int r0 = 4, c0 = 6;
        const int CROP_H = 4, CROP_W = 4;
        std::vector<float> cropped(CROP_H * CROP_W, -1.0f);

        // TODO: Extract 4x4 subgrid from parent into cropped.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 0; r < CROP_H && p2_passed; ++r) {
            for (int c = 0; c < CROP_W; ++c) {
                float expected = parent[(r0 + r) * W + (c0 + c)];
                if (cropped[r * CROP_W + c] != expected) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: 2D Subgrid Crop (ROI Extraction)", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Tridiagonal Matrix Band Extraction
    //
    // Context: Tridiagonal solvers (Thomas algorithm for PDEs / Splines) require
    //          extracting the main diagonal (d), upper sub-diagonal (u), and
    //          lower sub-diagonal (l).
    //
    // Task: Given square matrix `mat` [N=6, N=6]:
    //       Extract:
    //         - `diag_main` [N=6]: mat[i, i]
    //         - `diag_upper` [N-1=5]: mat[i, i + 1]
    //         - `diag_lower` [N-1=5]: mat[i + 1, i]
    // -------------------------------------------------------------------------
    {
        const int N = 6;
        std::vector<float> mat(N * N);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                mat[r * N + c] = static_cast<float>(r * 10 + c);
            }
        }

        std::vector<float> diag_main(N, -1.0f);
        std::vector<float> diag_upper(N - 1, -1.0f);
        std::vector<float> diag_lower(N - 1, -1.0f);

        // TODO: Extract the three diagonals.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            if (diag_main[i] != mat[i * N + i]) p3_passed = false;
        }
        for (int i = 0; i < N - 1; ++i) {
            if (diag_upper[i] != mat[i * N + (i + 1)]) p3_passed = false;
            if (diag_lower[i] != mat[(i + 1) * N + i]) p3_passed = false;
        }

        reportStatus("Problem 3: Tridiagonal Matrix Band Extraction", p3_passed);
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
