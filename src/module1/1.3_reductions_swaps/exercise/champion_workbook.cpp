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
// CHAMPION WORKBOOK: Array Reductions & Swaps
//
// Module: 1.3 - Parallel Reduction Foundations & In-Place Memory Swapping
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Block-chunked multi-pass vector reduction benchmark, in-place Top-K
//        QuickSelect partitioning, and Ring-AllReduce memory shift simulation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.3_champion
//   ../../../output/1.3_champion
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
    std::cout << "--- WORKBOOK: Array Reductions & Swaps (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Block-Chunked Two-Pass Vector Reduction Benchmark
    //
    // Context: Large-scale reductions on GPUs cannot fit all elements in one block.
    //          Grid-level reduction uses a two-pass kernel:
    //            Pass 1: Each block of size B=256 reduces 256 elements into a partial sum.
    //            Pass 2: A second kernel reduces the array of partial sums.
    //
    // Task: Given array `data` of size N=262,144 floats (1 MB):
    //       1. Pass 1: For each chunk of BLOCK_SIZE=256, reduce to `partial_sums[block_id]`.
    //       2. Pass 2: Reduce `partial_sums` (NUM_BLOCKS=1024) to `final_total_sum`.
    //       Measure throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int N = 262144;
        const int BLOCK_SIZE = 256;
        const int NUM_BLOCKS = N / BLOCK_SIZE; // 1024

        std::vector<float> data(N);
        float expected_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            data[i] = 1.0f;
            expected_sum += data[i];
        }

        std::vector<float> partial_sums(NUM_BLOCKS, 0.0f);
        float final_total_sum = 0.0f;

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Pass 1: Compute partial sums for each block.
        // Pass 2: Sum partial_sums into final_total_sum.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = N * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = (std::abs(final_total_sum - expected_sum) < 1e-2f);
        reportStatus("Problem 1: Two-Pass Block-Chunked Vector Reduction", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: In-Place Top-K QuickSelect Partitioning (LLM Sampler)
    //
    // Context: In Top-K sampling for language models, finding the K largest logits
    //          does NOT require a full O(N log N) sort. QuickSelect partitions
    //          the top K elements into indices [0..K-1] in O(N) average time.
    //
    // Task: Given logits array `logits` of size N=512 and K=16:
    //       Partition `logits` in-place so that the 16 largest elements are located
    //       in `logits[0..15]` (order within the top 16 does not matter).
    //       Every element in logits[0..15] must be >= every element in logits[16..511].
    // -------------------------------------------------------------------------
    {
        const int N = 512;
        const int K = 16;
        std::vector<float> logits(N);
        for (int i = 0; i < N; ++i) {
            logits[i] = static_cast<float>((i * 37) % 1000) * 0.1f;
        }
        std::vector<float> sorted_reference = logits;
        std::sort(sorted_reference.rbegin(), sorted_reference.rend());
        float kth_largest_val = sorted_reference[K - 1];

        // TODO: Implement in-place partition so that all elements >= kth_largest_val
        // are placed in indices 0..K-1.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < K; ++i) {
            if (logits[i] < kth_largest_val) {
                p2_passed = false;
                break;
            }
        }
        for (int i = K; i < N; ++i) {
            if (logits[i] > kth_largest_val) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: In-Place Top-K QuickSelect Partitioning", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Distributed Ring-AllReduce In-Place Memory Shift Simulator
    //
    // Context: In distributed deep learning (NCCL / Megatron-LM), Ring-AllReduce
    //          splits data into P chunks across P GPUs arranged in a ring.
    //          Over (P - 1) steps of Reduce-Scatter, each rank sends one chunk to its
    //          neighbor and adds the received chunk to its local buffer.
    //
    // Task: Simulate 4 GPUs (RANKS=4), each holding 4 chunks of size CHUNK_SIZE=16 floats.
    //       `gpu_buffers` shape: [RANKS=4, CHUNKS=4, CHUNK_SIZE=16].
    //       Perform 3 steps of Reduce-Scatter so that at the end, each rank r
    //       holds the global sum of chunk (r + 1) % 4.
    // -------------------------------------------------------------------------
    {
        const int RANKS = 4;
        const int CHUNKS = 4;
        const int CHUNK_SIZE = 16;
        const size_t total_elements = RANKS * CHUNKS * CHUNK_SIZE;

        // Each GPU r has chunks initialized with (r + 1)
        std::vector<float> gpu_buffers(total_elements);
        for (int r = 0; r < RANKS; ++r) {
            for (int c = 0; c < CHUNKS; ++c) {
                for (int i = 0; i < CHUNK_SIZE; ++i) {
                    gpu_buffers[(r * CHUNKS + c) * CHUNK_SIZE + i] = static_cast<float>(r + 1);
                }
            }
        }

        // Expected global sum across all 4 ranks for any chunk is 1 + 2 + 3 + 4 = 10.0f
        const float expected_sum = 10.0f;

        // TODO: In 3 steps (step = 0..2):
        // Each rank r sends chunk c_send = (r - step + RANKS) % RANKS
        // to rank (r + 1) % RANKS, which accumulates into its local chunk c_send.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int r = 0; r < RANKS && p3_passed; ++r) {
            int reduced_chunk = (r + 1) % CHUNKS;
            for (int i = 0; i < CHUNK_SIZE; ++i) {
                float val = gpu_buffers[(r * CHUNKS + reduced_chunk) * CHUNK_SIZE + i];
                if (std::abs(val - expected_sum) > 1e-3f) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: Distributed Ring-AllReduce Memory Shift Simulation", p3_passed);
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
