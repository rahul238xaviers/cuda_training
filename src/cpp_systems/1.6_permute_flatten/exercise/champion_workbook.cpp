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
// CHAMPION WORKBOOK: 3D/4D Permute & Flatten
//
// Module: 1.6 - Multi-Dimensional Layout Foundations
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Generalized 4D arbitrary permutation engine, fused permute+scale,
//        and Space-to-Depth (PixelShuffle inversion) memory transformation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.6_champion
//   ../../../output/1.6_champion
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
    std::cout << "--- WORKBOOK: 3D/4D Permute & Flatten (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Generalized 4D Arbitrary Dimension Permutation Engine
    //
    // Context: PyTorch `tensor.permute(dims)` and ONNX Transpose operations take
    //          an arbitrary permutation vector `perm` of length 4 and transpose memory.
    //
    // Task: Given source shape `shape` = {2, 8, 32, 64} (32,768 floats)
    //       and permutation vector `perm` = {0, 2, 1, 3}:
    //       Destination shape: {2, 32, 8, 64}.
    //       Compute dynamic strides for both shapes, and permute all elements.
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const std::vector<int> src_shape = {2, 8, 32, 64};
        const std::vector<int> perm = {0, 2, 1, 3};
        const std::vector<int> dst_shape = {
            src_shape[perm[0]], src_shape[perm[1]], src_shape[perm[2]], src_shape[perm[3]]
        };

        const size_t total_elements = src_shape[0] * src_shape[1] * src_shape[2] * src_shape[3];

        std::vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src[i] = static_cast<float>(i * 0.01f);

        std::vector<float> dst(total_elements, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Compute source and destination strides dynamically:
        // stride[d] = product of remaining dimensions to the right.
        // Permute all elements from src to dst.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        // Verify permutation
        for (int i0 = 0; i0 < src_shape[0] && p1_passed; ++i0) {
            for (int i1 = 0; i1 < src_shape[1] && p1_passed; ++i1) {
                for (int i2 = 0; i2 < src_shape[2] && p1_passed; ++i2) {
                    for (int i3 = 0; i3 < src_shape[3]; ++i3) {
                        int coords[4] = {i0, i1, i2, i3};
                        size_t src_idx = (size_t)i0 * (src_shape[1] * src_shape[2] * src_shape[3]) +
                                         (size_t)i1 * (src_shape[2] * src_shape[3]) +
                                         (size_t)i2 * src_shape[3] + i3;
                        size_t dst_idx = (size_t)coords[perm[0]] * (dst_shape[1] * dst_shape[2] * dst_shape[3]) +
                                         (size_t)coords[perm[1]] * (dst_shape[2] * dst_shape[3]) +
                                         (size_t)coords[perm[2]] * dst_shape[3] + coords[perm[3]];
                        if (dst[dst_idx] != src[src_idx]) {
                            p1_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: Generalized 4D Tensor Permutation Engine", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Fused Permute + Scale + Bias Memory Pass
    //
    // Context: In Attention QK normalization, transposing [B, S, H, D] -> [B, H, S, D]
    //          is often combined with scale factor 1/sqrt(D) and head bias:
    //            dst[b, h, s, d] = src[b, s, h, d] * scale + bias[d]
    //
    // Task: Given `src` [B=2, S=16, H=4, D=16] (2,048 floats), scale=0.25f, and `bias` [D=16]:
    //       Permute to `dst` [B=2, H=4, S=16, D=16] with fused scale and bias.
    // -------------------------------------------------------------------------
    {
        const int B = 2, S = 16, H = 4, D = 16;
        const size_t total_elements = B * S * H * D;
        const float scale = 0.25f;

        std::vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src[i] = static_cast<float>(i + 1);

        std::vector<float> bias(D);
        for (int d = 0; d < D; ++d) bias[d] = static_cast<float>(d * 0.1f);

        std::vector<float> dst(total_elements, -999.0f);

        // TODO: Permute and apply scale and bias in-place.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int b = 0; b < B && p2_passed; ++b) {
            for (int h = 0; h < H && p2_passed; ++h) {
                for (int s = 0; s < S && p2_passed; ++s) {
                    for (int d = 0; d < D; ++d) {
                        size_t src_idx = (size_t)b * (S * H * D) + (size_t)s * (H * D) + (size_t)h * D + d;
                        size_t dst_idx = (size_t)b * (H * S * D) + (size_t)h * (S * D) + (size_t)s * D + d;
                        float expected = src[src_idx] * scale + bias[d];
                        if (std::abs(dst[dst_idx] - expected) > 1e-4f) {
                            p2_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: Fused Permute + Scale + Bias Memory Pass", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Space-to-Depth (PixelShuffle Inversion) Memory Layout
    //
    // Context: Space-to-Depth takes an image of shape [C, H, W] and folds spatial
    //          blocks of size R x R into channels, producing:
    //            Out Shape: [C * R * R, H / R, W / R]
    //          For R=2, H=8, W=8, C=1:
    //            Output Shape: [4, 4, 4] (64 floats total).
    //
    // Task: Transform `space_img` [C=1, H=8, W=8] into `depth_tensor` [C_OUT=4, H_OUT=4, W_OUT=4].
    //       Channel index for sub-pixel (rh, rw): channel = rh * R + rw.
    // -------------------------------------------------------------------------
    {
        const int C_IN = 1, H = 8, W = 8, R = 2;
        const int C_OUT = C_IN * R * R; // 4
        const int H_OUT = H / R;        // 4
        const int W_OUT = W / R;        // 4

        std::vector<float> space_img(H * W);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                space_img[h * W + w] = static_cast<float>(h * 10 + w);
            }
        }

        std::vector<float> depth_tensor(C_OUT * H_OUT * W_OUT, -1.0f);

        // TODO: Fold 2x2 spatial pixels into 4 channels.
        // For each (oh, ow) in [0..3]:
        //   For each (rh, rw) in [0..1]:
        //     out_c = rh * R + rw
        //     depth_tensor[out_c, oh, ow] = space_img[oh * R + rh, ow * R + rw]
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int c = 0; c < C_OUT && p3_passed; ++c) {
            int rh = c / R;
            int rw = c % R;
            for (int oh = 0; oh < H_OUT && p3_passed; ++oh) {
                for (int ow = 0; ow < W_OUT; ++ow) {
                    float expected = space_img[(oh * R + rh) * W + (ow * R + rw)];
                    size_t out_idx = (size_t)c * (H_OUT * W_OUT) + (size_t)oh * W_OUT + ow;
                    if (depth_tensor[out_idx] != expected) {
                        p3_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 3: Space-to-Depth (PixelShuffle Inversion)", p3_passed);
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
