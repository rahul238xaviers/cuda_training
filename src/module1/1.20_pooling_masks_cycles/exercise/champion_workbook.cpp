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
// CHAMPION WORKBOOK: Pooling, Masks & Cycles
//
// Module: 1.20 - ML Tensor Memory Primitives
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: KV-Cache Page Freelist Cycle Detection (Floyd's Algorithm), Memory
//        Bandwidth Coalescing Degradation Benchmark, and 2D Adaptive Pooling.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.20_champion
//   ../../../output/1.20_champion
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
    std::cout << "--- WORKBOOK: Pooling, Masks & Cycles (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Paged KV-Cache Freelist Cycle Detection & Remediation
    //
    // Context: In paged inference memory managers, physical page blocks are chained
    //          via linked indices. A corrupted next-pointer creates an infinite loop
    //          (cycle), locking up inference threads.
    //
    // Task: Given an array of `next_block` pointers for NUM_BLOCKS=16:
    //       Chain starts at block `head = 0`.
    //       1. Use Floyd's Cycle-Finding Algorithm (Tortoise and Hare) to detect if a cycle exists.
    //       2. If a cycle exists, locate the start of the cycle, break it by setting its
    //          preceding block's next pointer to -1 (terminating the list).
    //       3. Record `has_cycle = true` and `cycle_entry_block`.
    // -------------------------------------------------------------------------
    {
        const int NUM_BLOCKS = 16;
        // Chain: 0 -> 3 -> 7 -> 11 -> 4 -> 9 -> 11 (Loop starts at 11: 11 -> 4 -> 9 -> 11)
        std::vector<int> next_block = {
            3,  -1, -1, 7,  9,  -1, -1, 11,
            -1, 11, -1, 4,  -1, -1, -1, -1
        };

        bool has_cycle = false;
        int cycle_entry_block = -1;

        // TODO: Detect cycle using slow and fast pointers.
        // If detected, find entry block and break the cycle.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = has_cycle && (cycle_entry_block == 11);
        // Verify cycle is broken
        int curr = 0;
        int count = 0;
        while (curr != -1 && count < NUM_BLOCKS + 2) {
            curr = next_block[curr];
            count++;
        }
        if (count >= NUM_BLOCKS) p1_passed = false; // Still an infinite loop

        reportStatus("Problem 1: KV-Cache Freelist Cycle Detection & Repair", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Memory Bandwidth Degradation: Coalesced vs Strided Traversal
    //
    // Context: Memory subsystems read entire 64-byte / 128-byte cache lines.
    //          When a kernel reads with stride S > 1, unused bytes in the cache line
    //          are discarded, causing memory throughput to collapse.
    //
    // Task: Given a buffer of N=1,048,576 floats (4 MB):
    //       1. Compute sum sequentially (stride = 1).
    //       2. Compute sum with strided access (stride = 16, i.e., 64 bytes between elements).
    //       Measure memory throughput (GB/s) for both passes.
    // -------------------------------------------------------------------------
    {
        const int N = 1048576; // 4 MB
        std::vector<float> data(N);
        for (int i = 0; i < N; ++i) data[i] = 1.0f;

        float sum_coalesced = 0.0f;
        auto start_c = std::chrono::high_resolution_clock::now();

        // Pass 1: Sequential Coalesced
        for (int i = 0; i < N; ++i) sum_coalesced += data[i];

        auto end_c = std::chrono::high_resolution_clock::now();
        double elapsed_c = std::chrono::duration<double>(end_c - start_c).count();
        double throughput_c = (N * sizeof(float) / elapsed_c) / 1e9;

        float sum_strided = 0.0f;
        const int STRIDE = 16;
        auto start_s = std::chrono::high_resolution_clock::now();

        // TODO: Pass 2: Strided Traversal.
        // For s = 0 to STRIDE - 1:
        //   For i = s to N - 1 step STRIDE:
        //     sum_strided += data[i]
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end_s = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(end_s - start_s).count();
        double throughput_s = (N * sizeof(float) / elapsed_s) / 1e9;

        bool p2_passed = (std::abs(sum_coalesced - static_cast<float>(N)) < 1e-3f) &&
                         (std::abs(sum_strided - static_cast<float>(N)) < 1e-3f);

        reportStatus("Problem 2: Memory Bandwidth Degradation (Coalesced vs Strided)", p2_passed, throughput_c);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: 2D Adaptive Average Pooling (Arbitrary Target Dimensions)
    //
    // Context: In PyTorch, `nn.AdaptiveAvgPool2d((H_out, W_out))` pools any input
    //          spatial resolution to the requested target size.
    //          For each output coordinate (oh, ow), the pooling window bounds are:
    //            start_h = floor(oh * H_in / H_out), end_h = ceil((oh + 1) * H_in / H_out)
    //            start_w = floor(ow * W_in / W_out), end_w = ceil((ow + 1) * W_in / W_out)
    //            out[oh, ow] = (1 / ((end_h - start_h) * (end_w - start_w))) * sum(in[start_h..end_h, start_w..end_w])
    //
    // Task: Adaptively pool `in_feat` [H_in=10, W_in=10] to `out_feat` [H_out=4, W_out=4].
    // -------------------------------------------------------------------------
    {
        const int H_IN = 10, W_IN = 10;
        const int H_OUT = 4, W_OUT = 4;

        std::vector<float> in_feat(H_IN * W_IN);
        for (int i = 0; i < H_IN * W_IN; ++i) in_feat[i] = static_cast<float>(i + 1);

        std::vector<float> out_feat(H_OUT * W_OUT, 0.0f);

        // TODO: Compute Adaptive Average Pooling from [10, 10] to [4, 4].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int oh = 0; oh < H_OUT && p3_passed; ++oh) {
            int start_h = (oh * H_IN) / H_OUT;
            int end_h = ((oh + 1) * H_IN + H_OUT - 1) / H_OUT;
            for (int ow = 0; ow < W_OUT && p3_passed; ++ow) {
                int start_w = (ow * W_IN) / W_OUT;
                int end_w = ((ow + 1) * W_IN + W_OUT - 1) / W_OUT;

                float sum = 0.0f;
                int count = (end_h - start_h) * (end_w - start_w);
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        sum += in_feat[h * W_IN + w];
                    }
                }
                float expected = sum / count;
                if (std::abs(out_feat[oh * W_OUT + ow] - expected) > 1e-4f) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: 2D Adaptive Average Pooling (Arbitrary Resolution)", p3_passed);
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
