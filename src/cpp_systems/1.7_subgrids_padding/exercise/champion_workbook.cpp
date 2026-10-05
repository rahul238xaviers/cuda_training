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
// CHAMPION WORKBOOK: Subgrids, Padding & Diagonals
//
// Module: 1.7 - Multi-Dimensional Layout Foundations
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: 2D Circular halo padding (toroidal wrapping), block-diagonal matrix
//        assembly, and replicated edge padding throughput benchmarking.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.7_champion
//   ../../../output/1.7_champion
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
    std::cout << "--- WORKBOOK: Subgrids, Padding & Diagonals (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Circular Halo Padding Kernel (Toroidal Boundary Wrapping)
    //
    // Context: In GPU stencil simulations and periodic convolutional filters, boundaries
    //          wrap toroidally: coordinates < 0 wrap to the far right/bottom, and
    //          coordinates >= H wrap to the top/left:
    //            wrap(x, N) = ((x % N) + N) % N
    //
    // Task: Given image `src` [H=16, W=16] and HALO=2:
    //       Generate `halo_padded` [PADDED_H=20, PADDED_W=20] using circular wrapping.
    // -------------------------------------------------------------------------
    {
        const int H = 16, W = 16, HALO = 2;
        const int PADDED_H = H + 2 * HALO; // 20
        const int PADDED_W = W + 2 * HALO; // 20

        std::vector<float> src(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                src[r * W + c] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> halo_padded(PADDED_H * PADDED_W, -1.0f);

        // TODO: Populate halo_padded using toroidal wrapping logic.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int pr = 0; pr < PADDED_H && p1_passed; ++pr) {
            for (int pc = 0; pc < PADDED_W; ++pc) {
                int r = ((pr - HALO) % H + H) % H;
                int c = ((pc - HALO) % W + W) % W;
                float expected = src[r * W + c];
                if (halo_padded[pr * PADDED_W + pc] != expected) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Circular Halo Padding (Toroidal Wrapping)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Block-Diagonal Matrix Assembly
    //
    // Context: In multi-task learning and independent multi-head transformations,
    //          transformations operate on disjoint sub-spaces, forming a block-diagonal matrix:
    //            diag_block[0]  0            0
    //            0              diag_block[1] 0
    //            0              0            diag_block[2]
    //
    // Task: Given K=4 square matrices `blocks` (each BLOCK_DIM=4, total 4x16 = 64 floats):
    //       Assemble into full matrix `big_mat` [N=16, N=16] (256 floats).
    //       Ensure all off-diagonal blocks are strictly 0.0f!
    // -------------------------------------------------------------------------
    {
        const int K = 4;
        const int BLOCK_DIM = 4;
        const int N = K * BLOCK_DIM; // 16

        std::vector<float> blocks(K * BLOCK_DIM * BLOCK_DIM);
        for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = static_cast<float>(i + 1);

        std::vector<float> big_mat(N * N, -1.0f);

        // TODO: Assemble big_mat as a block-diagonal matrix.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int r = 0; r < N && p2_passed; ++r) {
            for (int c = 0; c < N; ++c) {
                int block_r = r / BLOCK_DIM;
                int block_c = c / BLOCK_DIM;
                float expected = 0.0f;
                if (block_r == block_c) {
                    int k = block_r;
                    int sub_r = r % BLOCK_DIM;
                    int sub_c = c % BLOCK_DIM;
                    expected = blocks[k * (BLOCK_DIM * BLOCK_DIM) + sub_r * BLOCK_DIM + sub_c];
                }
                if (big_mat[r * N + c] != expected) {
                    p2_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 2: Block-Diagonal Matrix Assembly", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Replicated Edge Padding Memory Throughput Benchmark
    //
    // Context: In image processing, edge replication (`clamp_to_edge`) repeats the
    //          outermost row and column into the padding borders:
    //            r_clamped = std::clamp(r, 0, H - 1)
    //            c_clamped = std::clamp(c, 0, W - 1)
    //
    // Task: Given image `img` [C=3, H=64, W=64] (12,288 floats) and PAD=4:
    //       Generate `padded` [C=3, PADDED_H=72, PADDED_W=72] (15,552 floats).
    //       Measure throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int C = 3, H = 64, W = 64, PAD = 4;
        const int PADDED_H = H + 2 * PAD; // 72
        const int PADDED_W = W + 2 * PAD; // 72

        const size_t total_in = C * H * W;
        const size_t total_out = C * PADDED_H * PADDED_W;

        std::vector<float> img(total_in);
        for (size_t i = 0; i < total_in; ++i) img[i] = static_cast<float>(i * 0.01f);

        std::vector<float> padded(total_out, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Populate padded with replicated boundary pixels.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = (total_in + total_out) * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int c = 0; c < C && p3_passed; ++c) {
            for (int pr = 0; pr < PADDED_H && p3_passed; ++pr) {
                int r = std::clamp(pr - PAD, 0, H - 1);
                for (int pc = 0; pc < PADDED_W; ++pc) {
                    int col = std::clamp(pc - PAD, 0, W - 1);
                    float expected = img[c * (H * W) + r * W + col];
                    size_t out_idx = (size_t)c * (PADDED_H * PADDED_W) + (size_t)pr * PADDED_W + pc;
                    if (padded[out_idx] != expected) {
                        p3_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 3: Replicated Edge Padding Benchmark", p3_passed, throughput);
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
