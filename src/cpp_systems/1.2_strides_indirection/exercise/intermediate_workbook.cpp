#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// INTERMEDIATE WORKBOOK: Strides & Indirection
//
// Module: 1.2 - Strided Indexing & Indirection Patterns
// Level:  Intermediate
//
// Focus: Interleaved multi-channel de-muxing, jagged/ragged nested tensor
//        pointer arrays, and strided 2D matrix slicing.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.2_intermediate
//   ../../../output/1.2_intermediate
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
    std::cout << "--- WORKBOOK: Strides & Indirection (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Interleaved Multi-Channel Audio De-Muxing
    //
    // Context: Audio hardware captures audio in interleaved format: [L0, R0, L1, R1, L2, R2...].
    //          Before feeding into neural audio encoders (e.g. Whisper / EnCodec),
    //          channels must be separated into planar buffers: [L0, L1, L2...] and [R0, R1, R2...].
    //
    // Task: Given interleaved stereo buffer `interleaved` of N=64 frames (total 128 floats):
    //       De-mux into `left_channel` and `right_channel` (each 64 floats)
    //       using pointer stepping with stride = 2.
    // -------------------------------------------------------------------------
    {
        const int FRAMES = 64;
        std::vector<float> interleaved(FRAMES * 2);
        for (int i = 0; i < FRAMES; ++i) {
            interleaved[2 * i] = static_cast<float>(i * 10);     // Left
            interleaved[2 * i + 1] = static_cast<float>(i * 10 + 5); // Right
        }

        std::vector<float> left_channel(FRAMES, 0.0f);
        std::vector<float> right_channel(FRAMES, 0.0f);

        // TODO: De-mux interleaved into left_channel and right_channel.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int i = 0; i < FRAMES; ++i) {
            if (left_channel[i] != interleaved[2 * i] || right_channel[i] != interleaved[2 * i + 1]) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: Interleaved Multi-Channel Audio De-Muxing", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Ragged/Nested Tensor Indirection Pointers
    //
    // Context: In continuous LLM serving, sequences have variable lengths.
    //          Rather than padding, tokens are packed into a single 1D memory pool.
    //          An array of pointers `seq_ptrs` points to the start of each sequence.
    //
    // Task: Given packed token buffer `token_pool` (total 16 floats),
    //       and sequence lengths: lengths = {4, 7, 2, 3} (NUM_SEQS = 4):
    //       1. Set `seq_ptrs[s]` to point to the start of sequence s in token_pool.
    //       2. For each sequence, compute the mean value of its tokens and store in `seq_means[s]`.
    // -------------------------------------------------------------------------
    {
        const int NUM_SEQS = 4;
        std::vector<int> lengths = {4, 7, 2, 3};
        std::vector<float> token_pool = {
            1.0f, 2.0f, 3.0f, 4.0f,                   // Seq 0 (len 4, mean = 2.5)
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, // Seq 1 (len 7, mean = 40.0)
            100.0f, 200.0f,                           // Seq 2 (len 2, mean = 150.0)
            0.5f, 1.5f, 2.5f                          // Seq 3 (len 3, mean = 1.5)
        };

        const float* seq_ptrs[NUM_SEQS] = {nullptr};
        std::vector<float> seq_means(NUM_SEQS, 0.0f);

        // TODO: Populate seq_ptrs using running offsets, and compute seq_means.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        std::vector<float> expected_means = {2.5f, 40.0f, 150.0f, 1.5f};
        bool p2_passed = true;
        for (int s = 0; s < NUM_SEQS; ++s) {
            if (std::abs(seq_means[s] - expected_means[s]) > 1e-4f) p2_passed = false;
        }

        reportStatus("Problem 2: Ragged/Nested Tensor Indirection & Reductions", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Strided 2D Matrix Sub-Grid Slicing
    //
    // Context: Emulate PyTorch strided slicing: `sub_mat = A[start_r::step_r, start_c::step_c]`.
    //
    // Task: Given matrix `A` [ROWS=16, COLS=16]:
    //       Extract sub-matrix with start_r=1, step_r=3, start_c=2, step_c=4.
    //       Resulting sub-matrix shape: OUT_ROWS = (16 - 1 + 2) / 3 = 5,
    //                                  OUT_COLS = (16 - 2 + 3) / 4 = 4.
    //       Write extracted elements into contiguous buffer `sub_matrix` [5, 4].
    // -------------------------------------------------------------------------
    {
        const int ROWS = 16, COLS = 16;
        std::vector<float> A(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                A[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        const int start_r = 1, step_r = 3;
        const int start_c = 2, step_c = 4;
        const int OUT_ROWS = 5;
        const int OUT_COLS = 4;

        std::vector<float> sub_matrix(OUT_ROWS * OUT_COLS, -1.0f);

        // TODO: Extract elements from A into sub_matrix using strided loops.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int r_out = 0; r_out < OUT_ROWS && p3_passed; ++r_out) {
            int orig_r = start_r + r_out * step_r;
            for (int c_out = 0; c_out < OUT_COLS; ++c_out) {
                int orig_c = start_c + c_out * step_c;
                float expected = A[orig_r * COLS + orig_c];
                if (sub_matrix[r_out * OUT_COLS + c_out] != expected) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: Strided 2D Matrix Sub-Grid Slicing", p3_passed);
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
