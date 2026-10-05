#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: 3D/4D Permute & Flatten
//
// Module: 1.6 - Multi-Dimensional Layout Foundations
// Level:  Beginner
//
// Focus: 3D tensor flattening to contiguous 1D buffers, 2D spatial patch
//        extraction and flattening, and reversible 3D coordinates.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.6_beginner
//   ../../../output/1.6_beginner
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
    std::cout << "--- WORKBOOK: 3D/4D Permute & Flatten (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 3D Tensor Flattening to Contiguous Memory
    //
    // Context: In PyTorch, calling `tensor.flatten()` on a 3D tensor [D, H, W]
    //          linearizes elements in row-major order:
    //            idx = d * (H * W) + h * W + w
    //
    // Task: Given 3D tensor `volume` [D=4, H=8, W=8] (256 floats):
    //       Flatten into 1D buffer `flat_buffer` [256].
    // -------------------------------------------------------------------------
    {
        const int D = 4, H = 8, W = 8;
        const size_t total_elements = D * H * W;

        std::vector<float> volume(total_elements);
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    volume[d * (H * W) + h * W + w] = static_cast<float>(d * 100 + h * 10 + w);
                }
            }
        }

        std::vector<float> flat_buffer(total_elements, -1.0f);

        // TODO: Copy all elements from volume into flat_buffer in flattened order.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (size_t i = 0; i < total_elements; ++i) {
            if (flat_buffer[i] != volume[i]) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: 3D Tensor Flattening to 1D Buffer", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Spatial Sub-Patch Extraction and Flattening
    //
    // Context: In Vision Transformers (ViT), images are partitioned into non-overlapping
    //          patches (e.g. 4x4 pixels), and each patch is flattened into a 1D vector.
    //
    // Task: Given 2D image `img` [H=8, W=8] and PATCH_SIZE=4:
    //       There are (8/4) * (8/4) = 4 patches. Each patch contains 4*4 = 16 pixels.
    //       Extract each patch and flatten it into row p of `patch_matrix` [NUM_PATCHES=4, 16].
    // -------------------------------------------------------------------------
    {
        const int H = 8, W = 8, P = 4;
        const int PATCHES_H = H / P; // 2
        const int PATCHES_W = W / P; // 2
        const int NUM_PATCHES = PATCHES_H * PATCHES_W; // 4
        const int PATCH_DIM = P * P; // 16

        std::vector<float> img(H * W);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                img[h * W + w] = static_cast<float>(h * 10 + w);
            }
        }

        std::vector<float> patch_matrix(NUM_PATCHES * PATCH_DIM, -1.0f);

        // TODO: For each patch (ph, pw) in [0..1], extract its 4x4 pixels into patch_matrix[p, :].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int ph = 0; ph < PATCHES_H && p2_passed; ++ph) {
            for (int pw = 0; pw < PATCHES_W && p2_passed; ++pw) {
                int patch_idx = ph * PATCHES_W + pw;
                for (int pi = 0; pi < P; ++pi) {
                    for (int pj = 0; pj < P; ++pj) {
                        int img_h = ph * P + pi;
                        int img_w = pw * P + pj;
                        float expected = img[img_h * W + img_w];
                        float actual = patch_matrix[patch_idx * PATCH_DIM + (pi * P + pj)];
                        if (actual != expected) {
                            p2_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: 2D Spatial Sub-Patch Flattening", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Reversible 3D Coordinate Mapping
    //
    // Context: In GPU volumetric rendering and sparse octrees, coordinates (d, h, w)
    //          must map bidirectionally to a 1D index with zero loss of fidelity.
    //
    // Task: Given dimensions D=5, H=7, W=11 (total 385 voxels):
    //       Verify that for every voxel (d, h, w):
    //         flat = d * (H * W) + h * W + w
    //         rec_d = flat / (H * W), rec_h = (flat / W) % H, rec_w = flat % W
    //         (rec_d == d && rec_h == h && rec_w == w).
    // -------------------------------------------------------------------------
    {
        const int D = 5, H = 7, W = 11;
        bool all_reversible = true;

        // TODO: In nested loops over d, h, w:
        // Compute flat index, unflatten it back, and verify equality.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        reportStatus("Problem 3: Reversible 3D Coordinate Mapping", all_reversible);
        if (all_reversible) passed++;
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
