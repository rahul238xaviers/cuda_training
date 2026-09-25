#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: 3D/4D Permute & Flatten
//
// Module: 1.6 - Multi-Dimensional Layout Foundations
// Level:  Intermediate
//
// Focus: Vision Transformer (ViT) multi-channel patch extraction, NCHW to CHWN
//        systolic array layout permutation, and 4D Depth-last transposition.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.6_intermediate
//   ../../../output/1.6_intermediate
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
    std::cout << "--- WORKBOOK: 3D/4D Permute & Flatten (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Vision Transformer (ViT) Multi-Channel Patch Flattening
    //
    // Context: In ViT, an RGB image [C=3, H=16, W=16] is split into patches of size P=4.
    //          Number of patches = (H/P) * (W/P) = 4 * 4 = 16 patches.
    //          Each patch contains C * P * P = 3 * 4 * 4 = 48 floats.
    //          The output is a 2D matrix of shape [16, 48].
    //
    // Task: Transform `image` [C=3, H=16, W=16] into `patch_tokens` [16, 48].
    //       Inside each patch vector of 48 floats, channels and pixels are flattened:
    //       offset = c * (P * P) + pi * P + pj.
    // -------------------------------------------------------------------------
    {
        const int C = 3, H = 16, W = 16, P = 4;
        const int PATCHES_H = H / P; // 4
        const int PATCHES_W = W / P; // 4
        const int NUM_PATCHES = PATCHES_H * PATCHES_W; // 16
        const int PATCH_DIM = C * P * P; // 48

        std::vector<float> image(C * H * W);
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    image[c * (H * W) + h * W + w] = static_cast<float>(c * 1000 + h * 10 + w);
                }
            }
        }

        std::vector<float> patch_tokens(NUM_PATCHES * PATCH_DIM, -1.0f);

        // TODO: Extract 16 patches from image into patch_tokens.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int ph = 0; ph < PATCHES_H && p1_passed; ++ph) {
            for (int pw = 0; pw < PATCHES_W && p1_passed; ++pw) {
                int patch_idx = ph * PATCHES_W + pw;
                for (int c = 0; c < C && p1_passed; ++c) {
                    for (int pi = 0; pi < P; ++pi) {
                        for (int pj = 0; pj < P; ++pj) {
                            int im_h = ph * P + pi;
                            int im_w = pw * P + pj;
                            float expected = image[c * (H * W) + im_h * W + im_w];
                            int token_col = c * (P * P) + pi * P + pj;
                            if (patch_tokens[patch_idx * PATCH_DIM + token_col] != expected) {
                                p1_passed = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: ViT Multi-Channel Patch Flattening", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: NCHW to CHWN Layout Permutation (Systolic Array Weight-Stationary)
    //
    // Context: In custom TPU / NPU accelerators, tensors are organized in CHWN format
    //          to keep channel weights stationary while batch vectors stream through.
    //
    // Task: Given tensor `src_nchw` [N=2, C=4, H=8, W=8] (1,024 floats):
    //       Permute to `dst_chwn` [C=4, H=8, W=8, N=2].
    // -------------------------------------------------------------------------
    {
        const int N = 2, C = 4, H = 8, W = 8;
        const size_t total_elements = N * C * H * W;
        std::vector<float> src_nchw(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src_nchw[i] = static_cast<float>(i + 1);

        std::vector<float> dst_chwn(total_elements, -1.0f);

        // TODO: Permute NCHW to CHWN.
        // NCHW offset: n * (C * H * W) + c * (H * W) + h * W + w
        // CHWN offset: c * (H * W * N) + h * (W * N) + w * N + n
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int n = 0; n < N && p2_passed; ++n) {
            for (int c = 0; c < C && p2_passed; ++c) {
                for (int h = 0; h < H && p2_passed; ++h) {
                    for (int w = 0; w < W; ++w) {
                        size_t src_idx = (size_t)n * (C * H * W) + (size_t)c * (H * W) + (size_t)h * W + w;
                        size_t dst_idx = (size_t)c * (H * W * N) + (size_t)h * (W * N) + (size_t)w * N + n;
                        if (dst_chwn[dst_idx] != src_nchw[src_idx]) {
                            p2_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: NCHW to CHWN Layout Permutation", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Batched 4D Depthwise Permutation ([B, D, H, W] -> [B, H, W, D])
    //
    // Context: In 3D MRI processing, voxels are often transposed from Depth-first
    //          to Depth-last (channels-last 3D format) for 3D convolution kernels.
    //
    // Task: Given tensor `A` [B=2, D=4, H=4, W=4] (128 floats):
    //       Permute to `B` [B=2, H=4, W=4, D=4].
    // -------------------------------------------------------------------------
    {
        const int B = 2, D = 4, H = 4, W = 4;
        const size_t total_elements = B * D * H * W;
        std::vector<float> A(total_elements);
        for (size_t i = 0; i < total_elements; ++i) A[i] = static_cast<float>(i * 0.5f);

        std::vector<float> out_B(total_elements, -1.0f);

        // TODO: Permute [B, D, H, W] -> [B, H, W, D].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int b = 0; b < B && p3_passed; ++b) {
            for (int d = 0; d < D && p3_passed; ++d) {
                for (int h = 0; h < H && p3_passed; ++h) {
                    for (int w = 0; w < W; ++w) {
                        size_t src_idx = (size_t)b * (D * H * W) + (size_t)d * (H * W) + (size_t)h * W + w;
                        size_t dst_idx = (size_t)b * (H * W * D) + (size_t)h * (W * D) + (size_t)w * D + d;
                        if (out_B[dst_idx] != A[src_idx]) {
                            p3_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 3: Batched 4D Depthwise Permutation [B,D,H,W]->[B,H,W,D]", p3_passed);
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
