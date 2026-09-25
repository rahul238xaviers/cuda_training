#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>

// =========================================================================
// BEGINNER WORKBOOK: Embeddings & Projections
//
// Module: 1.17 - ML Tensor Memory Primitives
// Level:  Beginner
//
// Focus: Core embedding gather, RoPE frequency preparation, and linear
//        projection row indexing using real memory buffers and verification.
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
    std::cout << "--- WORKBOOK: Embeddings & Projections (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Sequential Token Embedding Gather
    //
    // Context: In a Transformer, the first operation is gathering embedding
    //          vectors for an input token sequence from a 2D weight table.
    //
    // Task: Given an embedding table `weight_table` [VOCAB_SIZE=32, HIDDEN_DIM=16]
    //       and a sequence of token IDs `tokens` of length SEQ_LEN=8:
    //       Copy each token's 16-element embedding vector into `out_embeddings`
    //       of shape [SEQ_LEN, HIDDEN_DIM].
    // -------------------------------------------------------------------------
    {
        const int VOCAB_SIZE = 32;
        const int HIDDEN_DIM = 16;
        const int SEQ_LEN = 8;

        std::vector<float> weight_table(VOCAB_SIZE * HIDDEN_DIM);
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            for (int d = 0; d < HIDDEN_DIM; ++d) {
                weight_table[v * HIDDEN_DIM + d] = static_cast<float>(v * 100 + d);
            }
        }

        std::vector<int> tokens = {4, 12, 0, 31, 7, 25, 3, 19};
        std::vector<float> out_embeddings(SEQ_LEN * HIDDEN_DIM, 0.0f);

        // TODO: For each token at index i (0 <= i < SEQ_LEN), locate its row
        // in weight_table and copy HIDDEN_DIM floats into the corresponding row
        // of out_embeddings.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int i = 0; i < SEQ_LEN && p1_passed; ++i) {
            int tok = tokens[i];
            for (int d = 0; d < HIDDEN_DIM; ++d) {
                float expected = static_cast<float>(tok * 100 + d);
                if (out_embeddings[i * HIDDEN_DIM + d] != expected) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: Sequential Token Embedding Gather", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Rotary Position Embedding (RoPE) Interleaved Rotation
    //
    // Context: Modern LLMs (Llama 3, Gemma, Mistral) apply RoPE to Q and K.
    //          Consecutive pairs of hidden dimensions [x_2i, x_2i+1] are rotated by
    //          angle theta_i:
    //            y_2i   = x_2i * cos(theta_i) - x_{2i+1} * sin(theta_i)
    //            y_{2i+1} = x_2i * sin(theta_i) + x_{2i+1} * cos(theta_i)
    //
    // Task: Given vector `vec` of size DIM=8 and rotation angles `theta` of size 4 (DIM/2):
    //       Apply RoPE rotation in-place on `vec`.
    // -------------------------------------------------------------------------
    {
        const int DIM = 8;
        std::vector<float> vec = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, 4.0f};
        std::vector<float> original = vec;
        std::vector<float> theta = {0.0f, 3.14159265f / 2.0f, 3.14159265f, 3.14159265f / 4.0f};

        // TODO: For each pair i (0 <= i < DIM / 2), rotate [vec[2*i], vec[2*i+1]]
        // using cos(theta[i]) and sin(theta[i]).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < DIM / 2; ++i) {
            float x0 = original[2 * i];
            float x1 = original[2 * i + 1];
            float c = std::cos(theta[i]);
            float s = std::sin(theta[i]);
            float exp_y0 = x0 * c - x1 * s;
            float exp_y1 = x0 * s + x1 * c;
            if (std::abs(vec[2 * i] - exp_y0) > 1e-4f || std::abs(vec[2 * i + 1] - exp_y1) > 1e-4f) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: Rotary Position Embedding (RoPE) Pair Rotation", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Linear Projection Weight Row Stride Traversal
    //
    // Context: A linear projection layer computes y = x @ W^T + b.
    //          Weight matrix W is stored in row-major shape [OUT_FEATURES, IN_FEATURES].
    //
    // Task: Given weight matrix `W` [OUT_FEATURES=4, IN_FEATURES=8] and input vector
    //       `x` [IN_FEATURES=8], compute output vector `y` [OUT_FEATURES=4] where:
    //       y[i] = dot_product(W_row_i, x) + bias[i]
    //       Use pointer offsets to navigate rows of W.
    // -------------------------------------------------------------------------
    {
        const int IN_FEATURES = 8;
        const int OUT_FEATURES = 4;

        std::vector<float> W(OUT_FEATURES * IN_FEATURES);
        for (int i = 0; i < OUT_FEATURES * IN_FEATURES; ++i) {
            W[i] = static_cast<float>(i + 1);
        }
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> bias = {0.5f, -0.5f, 1.0f, -1.0f};
        std::vector<float> y(OUT_FEATURES, 0.0f);

        // TODO: Compute y[i] for all i in [0, OUT_FEATURES) using row pointers into W.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < OUT_FEATURES; ++i) {
            float expected = bias[i];
            for (int j = 0; j < IN_FEATURES; ++j) {
                expected += W[i * IN_FEATURES + j] * x[j];
            }
            if (std::abs(y[i] - expected) > 1e-4f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Linear Projection Row Stride Dot-Product", p3_passed);
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
