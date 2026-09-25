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
// CHAMPION WORKBOOK: ML Layer Offsets & Layout Transformations
//
// Module: 1.19 - ML Tensor Memory Primitives
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Real-world memory layouts for Convolution (im2col), RMSNorm,
//        NCHW <-> NHWC Tensor Core memory conversions, and MoE token dispatch.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.19_champion
//   ../../../output/1.19_champion
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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Real 2D Convolution `im2col` Memory Transformation
    //
    // Context: In deep learning frameworks (cuDNN, Caffe, Darknet), 2D convolution
    //          is implemented by unfolding overlapping receptive fields into columns
    //          (im2col), reducing convolution to a standard GEMM:
    //            Output = Weights [OutC, InC * K * K] @ Col_Matrix [InC * K * K, OutH * OutW]
    //
    // Geometry:
    //   Input Image:  C=3, H=8, W=8
    //   Kernel Size:  K=3, Stride: S=1, Padding: P=1
    //   Output Shape: OutH = 8, OutW = 8
    //   Col Matrix:   Rows = C * K * K = 27, Cols = OutH * OutW = 64
    //
    // Task: Transform `input_img` [C, H, W] into `col_matrix` [C * K * K, OutH * OutW].
    //       For any pixel located outside the input image bounds (due to padding P=1),
    //       fill the corresponding value in col_matrix with 0.0f.
    // -------------------------------------------------------------------------
    {
        const int C = 3, H = 8, W = 8;
        const int K = 3, STRIDE = 1, PAD = 1;
        const int OUT_H = (H + 2 * PAD - K) / STRIDE + 1; // 8
        const int OUT_W = (W + 2 * PAD - K) / STRIDE + 1; // 8
        const int COL_ROWS = C * K * K;                   // 27
        const int COL_COLS = OUT_H * OUT_W;               // 64

        std::vector<float> input_img(C * H * W);
        for (size_t i = 0; i < input_img.size(); ++i) {
            input_img[i] = static_cast<float>(i + 1);
        }

        std::vector<float> col_matrix(COL_ROWS * COL_COLS, -999.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Populate col_matrix.
        // For each output pixel (oh, ow) and kernel offset (c, kh, kw):
        //   in_h = oh * STRIDE - PAD + kh
        //   in_w = ow * STRIDE - PAD + kw
        //   value = (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) ? input_img[c, in_h, in_w] : 0.0f
        //   col_row = c * (K * K) + kh * K + kw
        //   col_col = oh * OUT_W + ow
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double throughput = (COL_ROWS * COL_COLS * sizeof(float) / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int c = 0; c < C && p1_passed; ++c) {
            for (int kh = 0; kh < K && p1_passed; ++kh) {
                for (int kw = 0; kw < K && p1_passed; ++kw) {
                    int col_row = c * (K * K) + kh * K + kw;
                    for (int oh = 0; oh < OUT_H && p1_passed; ++oh) {
                        for (int ow = 0; ow < OUT_W; ++ow) {
                            int col_col = oh * OUT_W + ow;
                            int in_h = oh * STRIDE - PAD + kh;
                            int in_w = ow * STRIDE - PAD + kw;
                            float expected = 0.0f;
                            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                expected = input_img[c * (H * W) + in_h * W + in_w];
                            }
                            if (col_matrix[col_row * COL_COLS + col_col] != expected) {
                                p1_passed = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: Real 2D Convolution `im2col` Memory Transformation", p1_passed, p1_passed ? throughput : -1.0);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Root Mean Square Normalization (RMSNorm)
    //
    // Context: Modern LLMs (Llama 3, Mistral, Gemma) replace LayerNorm with RMSNorm:
    //            RMS(x) = sqrt( (1 / D) * sum_{i=0}^{D-1} x_i^2 + eps )
    //            y_i    = (x_i / RMS(x)) * gamma_i
    //
    // Task: Given activations `X` of shape [BATCH=4, SEQ_LEN=8, HIDDEN_DIM=32]
    //       and learnable scale `gamma` of shape [HIDDEN_DIM=32]:
    //       Apply RMSNorm in-place to `X` with eps = 1e-6f.
    // -------------------------------------------------------------------------
    {
        const int BATCH = 4, SEQ_LEN = 8, HIDDEN_DIM = 32;
        const float EPS = 1e-6f;
        const size_t total_elements = BATCH * SEQ_LEN * HIDDEN_DIM;

        std::vector<float> X(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            X[i] = static_cast<float>((i % 29) - 14) * 0.25f;
        }
        std::vector<float> original_X = X;

        std::vector<float> gamma(HIDDEN_DIM);
        for (int d = 0; d < HIDDEN_DIM; ++d) {
            gamma[d] = 0.5f + (d % 4) * 0.25f;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Apply RMSNorm in-place on each row of length HIDDEN_DIM.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double throughput = (2.0 * total_elements * sizeof(float) / elapsed_sec) / 1e9;

        bool p2_passed = true;
        for (int b = 0; b < BATCH && p2_passed; ++b) {
            for (int s = 0; s < SEQ_LEN && p2_passed; ++s) {
                size_t base = (size_t)(b * SEQ_LEN + s) * HIDDEN_DIM;
                float sum_sq = 0.0f;
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    float v = original_X[base + d];
                    sum_sq += v * v;
                }
                float rms = std::sqrt(sum_sq / HIDDEN_DIM + EPS);
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    float expected = (original_X[base + d] / rms) * gamma[d];
                    if (std::abs(X[base + d] - expected) > 1e-4f) {
                        p2_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 2: Batched In-Place RMSNorm with Scale Vector", p2_passed, p2_passed ? throughput : -1.0);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: NCHW to NHWC Tensor Core Memory Layout Transposition
    //
    // Context: In PyTorch and CUDA cuDNN, default convolutional memory is NCHW.
    //          However, NVIDIA Tensor Cores require NHWC (channels-last) format
    //          where C is a multiple of 8 or 16 for 128-bit vectorized memory loads.
    //
    // Task: Transpose `src_nchw` [N=2, C=16, H=8, W=8] (8,192 floats)
    //       into `dst_nhwc` [N=2, H=8, W=8, C=16].
    //       Compute throughput and ensure 100% data fidelity.
    // -------------------------------------------------------------------------
    {
        const int N = 2, C = 16, H = 8, W = 8;
        const size_t total_elements = N * C * H * W;

        std::vector<float> src_nchw(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            src_nchw[i] = static_cast<float>(i * 0.05f);
        }

        std::vector<float> dst_nhwc(total_elements, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Transpose src_nchw to dst_nhwc.
        // NCHW offset: n * (C * H * W) + c * (H * W) + h * W + w
        // NHWC offset: n * (H * W * C) + h * (W * C) + w * C + c
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double throughput = (2.0 * total_elements * sizeof(float) / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int n = 0; n < N && p3_passed; ++n) {
            for (int h = 0; h < H && p3_passed; ++h) {
                for (int w = 0; w < W && p3_passed; ++w) {
                    for (int c = 0; c < C; ++c) {
                        size_t src_idx = (size_t)n * (C * H * W) + (size_t)c * (H * W) + (size_t)h * W + w;
                        size_t dst_idx = (size_t)n * (H * W * C) + (size_t)h * (W * C) + (size_t)w * C + c;
                        if (dst_nhwc[dst_idx] != src_nchw[src_idx]) {
                            p3_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 3: NCHW to NHWC Tensor Core Layout Transposition", p3_passed, p3_passed ? throughput : -1.0);
        if (p3_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 4: Mixture-of-Experts (MoE) Token Dispatch Offsets
    //
    // Context: In sparse MoE layers (Mixtral 8x7B), each token is assigned to
    //          its top-1 or top-2 experts. Before executing expert GEMMs, tokens
    //          must be grouped by expert. This requires:
    //            1. Counting tokens assigned to each expert (`expert_counts`)
    //            2. Computing prefix sums to find start offsets (`expert_offsets`)
    //            3. Writing token indices into contiguous expert bins (`dispatch_indices`)
    //
    // Task: Given `token_expert_assignment` for NUM_TOKENS=16 across NUM_EXPERTS=4:
    //       Compute `expert_offsets` and fill `dispatch_indices` so all tokens
    //       assigned to expert e appear contiguously in dispatch_indices[expert_offsets[e] ... expert_offsets[e+1]-1].
    // -------------------------------------------------------------------------
    {
        const int NUM_TOKENS = 16;
        const int NUM_EXPERTS = 4;
        std::vector<int> token_expert_assignment = {
            2, 0, 1, 3, 2, 2, 0, 1, 3, 0, 1, 2, 3, 2, 0, 1
        };

        std::vector<int> expert_offsets(NUM_EXPERTS + 1, 0);
        std::vector<int> dispatch_indices(NUM_TOKENS, -1);

        // TODO: Compute expert_offsets using histogram and prefix sum.
        // Then populate dispatch_indices with original token IDs.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p4_passed = true;
        // Verify expert bins
        for (int e = 0; e < NUM_EXPERTS && p4_passed; ++e) {
            int start_idx = expert_offsets[e];
            int end_idx = expert_offsets[e + 1];
            for (int i = start_idx; i < end_idx; ++i) {
                int tok_id = dispatch_indices[i];
                if (tok_id < 0 || tok_id >= NUM_TOKENS || token_expert_assignment[tok_id] != e) {
                    p4_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 4: Mixture-of-Experts (MoE) Token Dispatch Offsets", p4_passed);
        if (p4_passed) passed++;
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
