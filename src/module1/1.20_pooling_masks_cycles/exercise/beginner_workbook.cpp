#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Pooling, Masks & Cycles
//
// Module: 1.20 - ML Tensor Memory Primitives
// Level:  Beginner
//
// Focus: 2D Max Pooling, 2D Average Pooling, and Boolean Attention Mask
//        broadcasting to logit matrices.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.20_beginner
//   ../../../output/1.20_beginner
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
    std::cout << "--- WORKBOOK: Pooling, Masks & Cycles (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Max Pooling with Spatial Window Strides
    //
    // Context: In Convolutional Networks, Max Pooling downsamples feature maps
    //          by taking the maximum element within each non-overlapping window.
    //
    // Task: Given input feature map `in_img` of shape [H=8, W=8] (64 floats),
    //       pool size POOL=2, stride STRIDE=2:
    //       Compute output `out_img` of shape [OUT_H=4, OUT_W=4] where:
    //         out_img[oh, ow] = max_{kh=0..1, kw=0..1} in_img[oh*2 + kh, ow*2 + kw]
    // -------------------------------------------------------------------------
    {
        const int H = 8, W = 8;
        const int POOL = 2, STRIDE = 2;
        const int OUT_H = H / STRIDE; // 4
        const int OUT_W = W / STRIDE; // 4

        std::vector<float> in_img(H * W);
        for (int i = 0; i < H * W; ++i) in_img[i] = static_cast<float>((i * 13) % 47);

        std::vector<float> out_img(OUT_H * OUT_W, -999.0f);

        // TODO: For each (oh, ow), compute the max value over the 2x2 receptive window.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int oh = 0; oh < OUT_H && p1_passed; ++oh) {
            for (int ow = 0; ow < OUT_W; ++ow) {
                float expected_max = -1e30f;
                for (int kh = 0; kh < POOL; ++kh) {
                    for (int kw = 0; kw < POOL; ++kw) {
                        expected_max = std::max(expected_max, in_img[(oh * STRIDE + kh) * W + (ow * STRIDE + kw)]);
                    }
                }
                if (out_img[oh * OUT_W + ow] != expected_max) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Max Pooling (2x2 Window, Stride 2)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Average Pooling with Normalization
    //
    // Context: Average Pooling smooths feature activations and reduces spatial variance.
    //
    // Task: Given input tensor `X` [C=2, H=6, W=6], pool size POOL=3, stride STRIDE=3:
    //       Compute output `Y` [C=2, OUT_H=2, OUT_W=2] where each cell is the
    //       arithmetic mean of its 3x3 receptive field (sum divided by 9.0f).
    // -------------------------------------------------------------------------
    {
        const int C = 2, H = 6, W = 6;
        const int POOL = 3, STRIDE = 3;
        const int OUT_H = H / STRIDE; // 2
        const int OUT_W = W / STRIDE; // 2

        const size_t in_size = C * H * W;
        std::vector<float> X(in_size);
        for (size_t i = 0; i < in_size; ++i) X[i] = static_cast<float>(i + 1);

        std::vector<float> Y(C * OUT_H * OUT_W, 0.0f);

        // TODO: Compute average pooling for each channel c and output pixel (oh, ow).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int c = 0; c < C && p2_passed; ++c) {
            for (int oh = 0; oh < OUT_H && p2_passed; ++oh) {
                for (int ow = 0; ow < OUT_W; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < POOL; ++kh) {
                        for (int kw = 0; kw < POOL; ++kw) {
                            sum += X[(c * H + (oh * STRIDE + kh)) * W + (ow * STRIDE + kw)];
                        }
                    }
                    float expected = sum / (POOL * POOL);
                    if (std::abs(Y[(c * OUT_H + oh) * OUT_W + ow] - expected) > 1e-4f) {
                        p2_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 2: 2D Average Pooling (3x3 Window)", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Boolean Token Mask Broadcast to Attention Logits
    //
    // Context: In batched Transformer NLP, input sequences are padded to equal length.
    //          A 1D boolean mask [BATCH, SEQ_LEN] marks valid tokens (1) vs padding (0).
    //          In attention, any key token j with mask[b, j] == 0 must have its logit
    //          set to -10000.0f so its attention weight exp(logit) becomes 0.
    //
    // Task: Given `token_mask` [BATCH=2, SEQ_LEN=8] and `scores` [BATCH=2, SEQ_LEN=8, SEQ_LEN=8]:
    //       If token_mask[b, j] == 0 (key is a padding token), set scores[b, i, j] = -10000.0f.
    // -------------------------------------------------------------------------
    {
        const int BATCH = 2, SEQ_LEN = 8;
        // Batch 0 has 5 valid tokens (tokens 0..4), batch 1 has 6 valid tokens (0..5)
        std::vector<uint8_t> token_mask = {
            1, 1, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0
        };

        const size_t total_scores = BATCH * SEQ_LEN * SEQ_LEN;
        std::vector<float> scores(total_scores, 1.0f);

        // TODO: Apply padding mask in-place to scores tensor.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int b = 0; b < BATCH && p3_passed; ++b) {
            for (int i = 0; i < SEQ_LEN && p3_passed; ++i) {
                for (int j = 0; j < SEQ_LEN; ++j) {
                    size_t idx = (size_t)b * (SEQ_LEN * SEQ_LEN) + (size_t)i * SEQ_LEN + j;
                    if (token_mask[b * SEQ_LEN + j] == 0) {
                        if (scores[idx] != -10000.0f) p3_passed = false;
                    } else {
                        if (scores[idx] != 1.0f) p3_passed = false;
                    }
                }
            }
        }

        reportStatus("Problem 3: Boolean Token Mask Broadcast to Attention Scores", p3_passed);
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
