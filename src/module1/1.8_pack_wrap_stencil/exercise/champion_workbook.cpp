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
// CHAMPION WORKBOOK: Pack, Wrap, Stencil & Crops
//
// Module: 1.8 - Multi-Dimensional Layout Foundations
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: 2D Stencil kernel memory throughput benchmarking, fused Sobel edge
//        gradient magnitude, and 3D subvolume extraction.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.8_champion
//   ../../../output/1.8_champion
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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Stencil Memory Bandwidth Benchmark (Heat Equation Simulation)
    //
    // Context: In GPU stencil codes, each output element requires reading 5 input
    //          values. In CUDA, shared memory tiles cache halos to avoid repeated
    //          global memory reads.
    //
    // Task: Given grid `grid` [DIM=512, DIM=512] (262,144 floats = 1 MB):
    //       Apply 5-point star stencil on all interior pixels:
    //         out[r, c] = 0.25 * (Up + Down + Left + Right)
    //       Measure memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int DIM = 512;
        const size_t total_elements = DIM * DIM;
        std::vector<float> grid(total_elements);
        for (size_t i = 0; i < total_elements; ++i) grid[i] = static_cast<float>(i % 31 + 1);

        std::vector<float> out_grid(total_elements, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Apply 5-point stencil on interior pixels (r in [1..DIM-2], c in [1..DIM-2]).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int r = 1; r < DIM - 1 && p1_passed; ++r) {
            for (int c = 1; c < DIM - 1; ++c) {
                float expected = 0.25f * (grid[(r - 1) * DIM + c] + grid[(r + 1) * DIM + c] +
                                          grid[r * DIM + (c - 1)] + grid[r * DIM + (c + 1)]);
                if (std::abs(out_grid[r * DIM + c] - expected) > 1e-4f) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Stencil Memory Bandwidth Benchmark", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Fused Sobel Edge Detection Gradient Magnitude
    //
    // Context: In computer vision preprocessing, Sobel kernels compute horizontal
    //          and vertical gradient approximations simultaneously:
    //            Gx = (TR + 2*R + BR) - (TL + 2*L + BL)
    //            Gy = (BL + 2*B + BR) - (TL + 2*T + TR)
    //            Magnitude = sqrt(Gx^2 + Gy^2)
    //
    // Task: Given image `img` [H=16, W=16]:
    //       Compute gradient magnitude for all interior pixels into `magnitude` [16, 16].
    // -------------------------------------------------------------------------
    {
        const int H = 16, W = 16;
        std::vector<float> img(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                img[r * W + c] = static_cast<float>(r * r + c * c);
            }
        }

        std::vector<float> magnitude(H * W, 0.0f);

        // TODO: Compute Sobel magnitude for interior pixels.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 1; r < H - 1 && p2_passed; ++r) {
            for (int c = 1; c < W - 1; ++c) {
                float tl = img[(r - 1) * W + (c - 1)];
                float t  = img[(r - 1) * W + c];
                float tr = img[(r - 1) * W + (c + 1)];
                float l  = img[r * W + (c - 1)];
                float rt = img[r * W + (c + 1)];
                float bl = img[(r + 1) * W + (c - 1)];
                float b  = img[(r + 1) * W + c];
                float br = img[(r + 1) * W + (c + 1)];

                float gx = (tr + 2.0f * rt + br) - (tl + 2.0f * l + bl);
                float gy = (bl + 2.0f * b + br) - (tl + 2.0f * t + tr);
                float expected = std::sqrt(gx * gx + gy * gy);

                if (std::abs(magnitude[r * W + c] - expected) > 1e-4f) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: Fused Sobel Edge Gradient Magnitude", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: 3D Volumetric Crop Parent Address Mapping
    //
    // Context: In medical image annotation, 3D subvolume crops map local voxel
    //          coordinates (d, h, w) back to the global physical scan volume.
    //
    // Task: Given parent 3D volume [D_PARENT=16, H_PARENT=32, W_PARENT=32] (16,384 floats),
    //       and a subvolume crop of size [CROP_D=4, CROP_H=8, CROP_W=8]
    //       originating at global offset (d0=4, h0=10, w0=12):
    //       Extract the subvolume into contiguous buffer `crop` [4 * 8 * 8 = 256 floats].
    // -------------------------------------------------------------------------
    {
        const int D_PARENT = 16, H_PARENT = 32, W_PARENT = 32;
        const int d0 = 4, h0 = 10, w0 = 12;
        const int CROP_D = 4, CROP_H = 8, CROP_W = 8;
        const size_t total_parent = D_PARENT * H_PARENT * W_PARENT;

        std::vector<float> parent(total_parent);
        for (size_t i = 0; i < total_parent; ++i) parent[i] = static_cast<float>(i + 1);

        std::vector<float> crop(CROP_D * CROP_H * CROP_W, -1.0f);

        // TODO: Extract the 3D crop from parent.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int d = 0; d < CROP_D && p3_passed; ++d) {
            for (int h = 0; h < CROP_H && p3_passed; ++h) {
                for (int w = 0; w < CROP_W; ++w) {
                    size_t parent_idx = (size_t)(d0 + d) * (H_PARENT * W_PARENT) + (size_t)(h0 + h) * W_PARENT + (w0 + w);
                    size_t crop_idx = (size_t)d * (CROP_H * CROP_W) + (size_t)h * CROP_W + w;
                    if (crop[crop_idx] != parent[parent_idx]) {
                        p3_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 3: 3D Volumetric Subvolume Crop Extraction", p3_passed);
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
