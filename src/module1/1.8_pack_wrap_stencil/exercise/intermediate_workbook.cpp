#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Pack, Wrap, Stencil & Crops
//
// Module: 1.8 - Multi-Dimensional Layout Foundations
// Level:  Intermediate
//
// Focus: 2D 9-point Gaussian blur stencil, 3D volumetric sub-box cropping,
//        and toroidal periodic boundary heat diffusion.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.8_intermediate
//   ../../../output/1.8_intermediate
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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D 9-Point Gaussian Blur Stencil
    //
    // Context: Image filtering applies a 3x3 Gaussian kernel:
    //            [ 1/16  2/16  1/16 ]
    //            [ 2/16  4/16  2/16 ]
    //            [ 1/16  2/16  1/16 ]
    //
    // Task: Given grid `img` [H=8, W=8]:
    //       Apply the 9-point Gaussian blur on all interior pixels (r in [1..6], c in [1..6]).
    //       Store results in `blurred`.
    // -------------------------------------------------------------------------
    {
        const int H = 8, W = 8;
        std::vector<float> img(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                img[r * W + c] = static_cast<float>(r * 10 + c);
            }
        }

        std::vector<float> blurred(H * W, 0.0f);

        // TODO: Apply 3x3 Gaussian blur on interior pixels.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 1; r < H - 1 && p1_passed; ++r) {
            for (int c = 1; c < W - 1; ++c) {
                float sum = (img[(r - 1) * W + (c - 1)] * 1.0f + img[(r - 1) * W + c] * 2.0f + img[(r - 1) * W + (c + 1)] * 1.0f +
                             img[r * W + (c - 1)] * 2.0f       + img[r * W + c] * 4.0f       + img[r * W + (c + 1)] * 2.0f +
                             img[(r + 1) * W + (c - 1)] * 1.0f + img[(r + 1) * W + c] * 2.0f + img[(r + 1) * W + (c + 1)] * 1.0f) / 16.0f;
                if (std::abs(blurred[r * W + c] - sum) > 1e-4f) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D 9-Point Gaussian Blur Stencil", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 3D Volumetric Sub-Box Cropping & Contiguous Packing
    //
    // Context: In 3D U-Net (medical image segmentation), sliding 3D sub-boxes
    //          are cropped out of a whole-body scan and packed contiguously.
    //
    // Task: Given 3D volume `scan` [D=8, H=8, W=8] (512 floats):
    //       Crop subvolume at (d0=2, h0=2, w0=2) with shape [CROP_D=4, CROP_H=4, CROP_W=4]
    //       into `cropped_volume` [64 floats].
    // -------------------------------------------------------------------------
    {
        const int D = 8, H = 8, W = 8;
        const int d0 = 2, h0 = 2, w0 = 2;
        const int CROP_D = 4, CROP_H = 4, CROP_W = 4;

        std::vector<float> scan(D * H * W);
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    scan[d * (H * W) + h * W + w] = static_cast<float>(d * 1000 + h * 10 + w);
                }
            }
        }

        std::vector<float> cropped_volume(CROP_D * CROP_H * CROP_W, -1.0f);

        // TODO: Extract 4x4x4 sub-volume into cropped_volume.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int d = 0; d < CROP_D && p2_passed; ++d) {
            for (int h = 0; h < CROP_H && p2_passed; ++h) {
                for (int w = 0; w < CROP_W; ++w) {
                    float expected = scan[(d0 + d) * (H * W) + (h0 + h) * W + (w0 + w)];
                    size_t out_idx = d * (CROP_H * CROP_W) + h * CROP_W + w;
                    if (cropped_volume[out_idx] != expected) {
                        p2_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 2: 3D Volumetric Sub-Box Cropping", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Toroidal Periodic Boundary Heat Diffusion Stencil
    //
    // Context: In physical grid simulations with periodic boundary conditions,
    //          the grid wraps like a torus: neighbor coordinate -1 wraps to DIM-1,
    //          and DIM wraps to 0. Every single pixel (including borders) is updated.
    //
    // Task: Given grid `temp` [N=8, N=8]:
    //       Compute one step of heat diffusion into `next_temp` [8, 8]:
    //         next_temp[r, c] = 0.5 * Center + 0.125 * (Up + Down + Left + Right)
    //         All boundary lookups must wrap toroidally: ((x % N) + N) % N.
    // -------------------------------------------------------------------------
    {
        const int N = 8;
        std::vector<float> temp(N * N);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                temp[r * N + c] = static_cast<float>((r * c) % 17 + 1);
            }
        }

        std::vector<float> next_temp(N * N, 0.0f);

        // TODO: Compute toroidal diffusion for all (r, c) in [0..N-1].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int r = 0; r < N && p3_passed; ++r) {
            for (int c = 0; c < N; ++c) {
                int up    = ((r - 1) % N + N) % N;
                int down  = ((r + 1) % N + N) % N;
                int left  = ((c - 1) % N + N) % N;
                int right = ((c + 1) % N + N) % N;

                float center_v = temp[r * N + c];
                float up_v     = temp[up * N + c];
                float down_v   = temp[down * N + c];
                float left_v   = temp[r * N + left];
                float right_v  = temp[r * N + right];

                float expected = 0.5f * center_v + 0.125f * (up_v + down_v + left_v + right_v);
                if (std::abs(next_temp[r * N + c] - expected) > 1e-4f) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: Toroidal Periodic Boundary Heat Diffusion", p3_passed);
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
