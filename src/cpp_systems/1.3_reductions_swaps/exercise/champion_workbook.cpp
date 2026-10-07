#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>

using namespace std;

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

void reportStatus(const string& name, bool passed, double throughputGBs = -1.0) {
    cout << "  " << left << setw(60) << name;
    if (passed) {
        cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            cout << " (" << fixed << setprecision(2) << throughputGBs << " GB/s)";
        }
        cout << endl;
    } else {
        cout << "\033[1;31m[FAILED]\033[0m" << endl;
    }
}

int main() {
    cout << "=================================================================" << endl;
    cout << "--- WORKBOOK: Array Reductions & Swaps (Champion Masterclass) ---" << endl;
    cout << "=================================================================" << endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: FlashAttention Online Softmax Rescaling Primitive
    //
    // Context: In transformer attention, standard softmax takes 3 passes over memory:
    //          finding max, computing sum of exp(x - max), and normalizing.
    //          FlashAttention computes softmax in small chunks (tiles) in SRAM.
    //          When a chunk arrives with a larger local max, the running denominator
    //          must be rescaled on the fly using exp(running_max - new_max).
    //
    // Given:
    //   - Array `logits` of size N = 1024 floats.
    //   - Tile chunk size: CHUNK_SIZE = 64 (NUM_CHUNKS = 16).
    //   - Running accumulators: `running_max = -INFINITY`, `running_sum = 0.0f`.
    //
    // Goal:
    //   - Process `logits` chunk-by-chunk in a single pass.
    //   - For each chunk, find local chunk max and local sum of exp(x - chunk_max).
    //   - Rescale `running_sum` by exp(running_max - new_max) and add the new chunk.
    //   - Verify final `running_max` and `running_sum` match ground truth.
    //
    // Reference:
    //   - See Section 5 (Technique: Online Softmax Rescaling) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 1024;
        const int CHUNK_SIZE = 64;
        const int NUM_CHUNKS = N / CHUNK_SIZE;

        vector<float> logits(N);
        for (int i = 0; i < N; ++i) {
            logits[i] = sin(static_cast<float>(i) * 0.1f) * 15.0f + static_cast<float>((i % 13) - 6) * 1.5f;
        }

        // Ground-truth 3-pass safe softmax reference
        float gt_max = -INFINITY;
        for (float v : logits) {
            gt_max = max(gt_max, v);
        }
        double gt_sum = 0.0;
        for (float v : logits) {
            gt_sum += exp(static_cast<double>(v - gt_max));
        }

        float running_max = -INFINITY;
        float running_sum = 0.0f;

        auto start = chrono::high_resolution_clock::now();

        // TODO: Process each chunk of CHUNK_SIZE floats and update running_max and running_sum via online rescaling.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = chrono::high_resolution_clock::now();
        double elapsed_sec = chrono::duration<double>(end - start).count();
        double bytes_moved = N * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = (abs(running_max - gt_max) < 1e-4f) &&
                         (abs(running_sum - static_cast<float>(gt_sum)) / static_cast<float>(gt_sum) < 1e-4f);

        reportStatus("Problem 1: FlashAttention Online Softmax Rescaling", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Hierarchical Parallel Welford Merge (Chan's Formula)
    //
    // Context: In distributed or multi-warp LayerNorm / RMSNorm, independent blocks
    //          compute local (count, mean, M2) accumulators. Instead of sequential
    //          passes or naive variance sums that suffer catastrophic cancellation,
    //          blocks merge their accumulators pairwise using Chan's exact formula.
    //
    // Given:
    //   - Array `activations` of N = 1024 floats partitioned into NUM_BLOCKS = 4 blocks (size 256).
    //   - Structure: `struct WelfordAcc { int count; float mean; float M2; };`
    //   - Pre-computed `vector<WelfordAcc> block_accs(4)` containing each block's local statistics.
    //
    // Goal:
    //   - Merge the 4 block accumulators into a single global `WelfordAcc` using Chan's formula.
    //   - Derive `global_mean`, `global_variance = M2 / count`, and `global_std = sqrt(global_variance)`.
    //   - Match double-precision ground truth.
    //
    // Reference:
    //   - See Section 4 (Technique: Parallel Welford Merge - Chan's Formula) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 1024;
        const int BLOCK_SIZE = 256;
        const int NUM_BLOCKS = N / BLOCK_SIZE;

        vector<float> activations(N);
        double gt_sum = 0.0;
        for (int i = 0; i < N; ++i) {
            activations[i] = 10000.0f + static_cast<float>((i % 7) - 3) * 0.125f;
            gt_sum += activations[i];
        }

        double gt_mean = gt_sum / N;
        double gt_var = 0.0;
        for (int i = 0; i < N; ++i) {
            double diff = activations[i] - gt_mean;
            gt_var += diff * diff;
        }
        gt_var /= N;
        double gt_std = sqrt(gt_var);

        struct WelfordAcc {
            int count = 0;
            float mean = 0.0f;
            float M2 = 0.0f;
        };

        // Each block pre-computes its local Welford statistics
        vector<WelfordAcc> block_accs(NUM_BLOCKS);
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            int count = 0;
            float mean = 0.0f;
            float M2 = 0.0f;
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float x = activations[b * BLOCK_SIZE + i];
                count++;
                float delta1 = x - mean;
                mean += delta1 / count;
                float delta2 = x - mean;
                M2 += delta1 * delta2;
            }
            block_accs[b] = {count, mean, M2};
        }

        float global_mean = 0.0f;
        float global_variance = 0.0f;
        float global_std = 0.0f;

        auto start = chrono::high_resolution_clock::now();

        // TODO: Merge the 4 block_accs into a single global accumulator using Chan's merge formula.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = chrono::high_resolution_clock::now();
        double elapsed_sec = chrono::duration<double>(end - start).count();
        double bytes_moved = NUM_BLOCKS * sizeof(WelfordAcc);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p2_passed = (abs(global_mean - static_cast<float>(gt_mean)) < 1e-2f) &&
                         (abs(global_variance - static_cast<float>(gt_var)) < 1e-4f) &&
                         (abs(global_std - static_cast<float>(gt_std)) < 1e-4f);

        reportStatus("Problem 2: Hierarchical Parallel Welford Merge", p2_passed, throughput);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: In-Place Top-K QuickSelect Partitioning (LLM Sampler)
    //
    // Context: In Top-K sampling for language models, finding the K largest logits
    //          does NOT require a full O(N log N) sort. QuickSelect partitions
    //          the top K elements into indices [0..K-1] in O(N) average time.
    //
    // Given:
    //   - Array `logits` of size N = 512 floats.
    //   - Target top element count: K = 16.
    //   - Target boundary value: `kth_largest_val`.
    //
    // Goal:
    //   - Partition `logits` in-place so that the 16 largest elements are located
    //     in `logits[0..15]` (order within the top 16 does not matter).
    //   - Every element in logits[0..15] must be >= every element in logits[16..511].
    //   - Zero auxiliary memory allocations allowed.
    //
    // Reference:
    //   - See Section 2B (In-Place Two-Pointer Partitioning) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 512;
        const int K = 16;
        vector<float> logits(N);
        for (int i = 0; i < N; ++i) {
            logits[i] = static_cast<float>((i * 37) % 1000) * 0.1f;
        }
        vector<float> sorted_reference = logits;
        sort(sorted_reference.rbegin(), sorted_reference.rend());
        float kth_largest_val = sorted_reference[K - 1];

        // TODO: Partition logits in-place so that all elements in [0..K-1] are >= kth_largest_val.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < K; ++i) {
            if (logits[i] < kth_largest_val) {
                p3_passed = false;
                break;
            }
        }
        for (int i = K; i < N; ++i) {
            if (logits[i] > kth_largest_val) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: In-Place Top-K QuickSelect Partitioning", p3_passed);
        if (p3_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 4: Distributed Ring-AllReduce In-Place Memory Shift Simulator
    //
    // Context: In distributed deep learning (NCCL / Megatron-LM), Ring-AllReduce
    //          splits data into P chunks across P GPUs arranged in a ring.
    //          Over (P - 1) steps of Reduce-Scatter, each rank sends one chunk to its
    //          neighbor and adds the received chunk to its local buffer.
    //
    // Given:
    //   - 4 virtual GPUs (RANKS = 4), each holding 4 chunks of CHUNK_SIZE = 16 floats.
    //   - `gpu_buffers` shape: [RANKS=4, CHUNKS=4, CHUNK_SIZE=16].
    //
    // Goal:
    //   - Perform 3 steps of Reduce-Scatter in a logical ring so that each rank r
    //     holds the global sum of chunk (r + 1) % 4.
    //
    // Reference:
    //   - See Section 3 (Hierarchy Level 4: Distributed Ring-AllReduce) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int RANKS = 4;
        const int CHUNKS = 4;
        const int CHUNK_SIZE = 16;
        const size_t total_elements = RANKS * CHUNKS * CHUNK_SIZE;

        // Each GPU r has chunks initialized with (r + 1)
        vector<float> gpu_buffers(total_elements);
        for (int r = 0; r < RANKS; ++r) {
            for (int c = 0; c < CHUNKS; ++c) {
                for (int i = 0; i < CHUNK_SIZE; ++i) {
                    gpu_buffers[(r * CHUNKS + c) * CHUNK_SIZE + i] = static_cast<float>(r + 1);
                }
            }
        }

        // Expected global sum across all 4 ranks for any chunk is 1 + 2 + 3 + 4 = 10.0f
        const float expected_sum = 10.0f;

        // TODO: Simulate 3 steps of Ring-AllReduce Reduce-Scatter across the 4 ranks.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p4_passed = true;
        for (int r = 0; r < RANKS && p4_passed; ++r) {
            int reduced_chunk = (r + 1) % CHUNKS;
            for (int i = 0; i < CHUNK_SIZE; ++i) {
                float val = gpu_buffers[(r * CHUNKS + reduced_chunk) * CHUNK_SIZE + i];
                if (abs(val - expected_sum) > 1e-3f) {
                    p4_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 4: Distributed Ring-AllReduce Memory Shift Simulation", p4_passed);
        if (p4_passed) passed++;
        total++;
    }

    cout << "\n=================================================================" << endl;
    cout << "--- SCORECARD ---" << endl;
    cout << "=================================================================" << endl;
    cout << "  Passed: " << passed << " / " << total << " tests." << endl;
    if (passed == total) {
        cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << endl;
    } else {
        cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << endl;
    }
    cout << "=================================================================" << endl;

    return (passed == total) ? 0 : 1;
}
