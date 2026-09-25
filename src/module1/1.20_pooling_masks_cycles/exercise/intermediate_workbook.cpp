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
// INTERMEDIATE WORKBOOK: Pooling, Masks & Cycles
//
// Module: 1.20 - ML Tensor Memory Primitives
// Level:  Intermediate
//
// Focus: Max Pooling with Argmax index tracking for backpropagation,
//        Global Average Pooling (GAP), and 32-bit integer bitmask unpacking.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.20_intermediate
//   ../../../output/1.20_intermediate
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
    std::cout << "--- WORKBOOK: Pooling, Masks & Cycles (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Max Pooling Forward Pass with Argmax Index Tracking
    //
    // Context: In PyTorch and CUDA backward kernels (`max_pool2d_backward`), gradients
    //          flow only to the single input location that produced the maximum value.
    //          The forward pass must store the flattened input index (argmax) for each pooled cell.
    //
    // Task: Given input tensor `X` [C=2, H=4, W=4], pool size POOL=2, stride STRIDE=2:
    //       1. Compute pooled values in `out_pooled` [C=2, OUT_H=2, OUT_W=2]
    //       2. Store the input index (c * H * W + in_h * W + in_w) of the winner in `argmax_indices`
    // -------------------------------------------------------------------------
    {
        const int C = 2, H = 4, W = 4;
        const int POOL = 2, STRIDE = 2;
        const int OUT_H = H / STRIDE; // 2
        const int OUT_W = W / STRIDE; // 2

        std::vector<float> X = {
            // Channel 0
            1.0f,  9.0f,  3.0f,  4.0f,
            2.0f,  5.0f,  7.0f,  8.0f,
            12.0f, 6.0f,  15.0f, 11.0f,
            0.0f,  8.0f,  14.0f, 13.0f,
            // Channel 1
            20.0f, 21.0f, 22.0f, 23.0f,
            19.0f, 18.0f, 17.0f, 25.0f,
            30.0f, 29.0f, 28.0f, 27.0f,
            35.0f, 31.0f, 32.0f, 33.0f
        };

        const size_t out_size = C * OUT_H * OUT_W;
        std::vector<float> out_pooled(out_size, -999.0f);
        std::vector<int> argmax_indices(out_size, -1);

        // TODO: Compute max pooling and record argmax index into argmax_indices.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int c = 0; c < C && p1_passed; ++c) {
            for (int oh = 0; oh < OUT_H && p1_passed; ++oh) {
                for (int ow = 0; ow < OUT_W; ++ow) {
                    float max_v = -1e30f;
                    int best_idx = -1;
                    for (int kh = 0; kh < POOL; ++kh) {
                        for (int kw = 0; kw < POOL; ++kw) {
                            int in_h = oh * STRIDE + kh;
                            int in_w = ow * STRIDE + kw;
                            int flat_in = c * (H * W) + in_h * W + in_w;
                            if (X[flat_in] > max_v) {
                                max_v = X[flat_in];
                                best_idx = flat_in;
                            }
                        }
                    }
                    size_t out_idx = c * (OUT_H * OUT_W) + oh * OUT_W + ow;
                    if (out_pooled[out_idx] != max_v || argmax_indices[out_idx] != best_idx) {
                        p1_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 1: Max Pooling Forward Pass with Argmax Tracking", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Global Average Pooling (GAP) Spatial Reduction
    //
    // Context: Modern CNN classifiers collapse all spatial pixels (H, W) per channel
    //          into a single representative scalar feature before linear classification:
    //            gap[b, c] = (1 / (H * W)) * sum_{h=0..H-1, w=0..W-1} X[b, c, h, w]
    //
    // Task: Given tensor `X` [BATCH=2, CHANNELS=8, H=8, W=8] (1,024 floats):
    //       Compute output `gap` of shape [BATCH=2, CHANNELS=8].
    // -------------------------------------------------------------------------
    {
        const int BATCH = 2, CHANNELS = 8, H = 8, W = 8;
        const int SPATIAL_SIZE = H * W; // 64
        const size_t total_elements = BATCH * CHANNELS * SPATIAL_SIZE;

        std::vector<float> X(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            X[i] = static_cast<float>((i % 19) + 1.0f);
        }

        std::vector<float> gap(BATCH * CHANNELS, 0.0f);

        // TODO: Compute Global Average Pooling across spatial dimensions for each (b, c).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int b = 0; b < BATCH && p2_passed; ++b) {
            for (int c = 0; c < CHANNELS && p2_passed; ++c) {
                float sum = 0.0f;
                size_t base = (size_t)(b * CHANNELS + c) * SPATIAL_SIZE;
                for (int i = 0; i < SPATIAL_SIZE; ++i) {
                    sum += X[base + i];
                }
                float expected = sum / SPATIAL_SIZE;
                if (std::abs(gap[b * CHANNELS + c] - expected) > 1e-4f) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: Global Average Pooling (GAP) Spatial Reduction", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Packed 32-Bit Integer Bitmask Unpacking to Float Tensor
    //
    // Context: In high-throughput LLM engines, token padding masks are packed into
    //          32-bit words (`uint32_t`) to minimize PCIe transfer latency.
    //          Each bit represents one token: bit 1 = valid (0.0f logit bias),
    //          bit 0 = padding (-10000.0f logit bias).
    //
    // Task: Given `packed_mask` containing two uint32_t words (representing SEQ_LEN=64 tokens):
    //       Unpack the 64 bits (from lowest bit 0 of word 0 to highest bit 31 of word 1)
    //       into `unpacked_float_mask` [SEQ_LEN=64].
    // -------------------------------------------------------------------------
    {
        const int SEQ_LEN = 64;
        // Word 0: 0xFFFFFFF0 -> lower 4 bits are 0 (padding), bits 4..31 are 1 (valid)
        // Word 1: 0x0000FFFF -> lower 16 bits are 1 (valid), upper 16 bits are 0 (padding)
        std::vector<uint32_t> packed_mask = {0xFFFFFFF0, 0x0000FFFF};

        std::vector<float> unpacked_float_mask(SEQ_LEN, -1.0f);

        // TODO: For each token i in [0, 64):
        // Determine word_idx = i / 32, bit_idx = i % 32.
        // If the bit is set, unpacked_float_mask[i] = 0.0f; else -10000.0f.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < SEQ_LEN && p3_passed; ++i) {
            int word_idx = i / 32;
            int bit_idx = i % 32;
            bool is_set = (packed_mask[word_idx] >> bit_idx) & 1;
            float expected = is_set ? 0.0f : -10000.0f;
            if (unpacked_float_mask[i] != expected) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Packed 32-Bit Bitmask Unpacking to Float Tensor", p3_passed);
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
