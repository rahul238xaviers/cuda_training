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
// CHAMPION WORKBOOK: Pointer Relations & Copying
//
// Module: 1.16 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Champion
//
// Focus: 128-bit vector aligned copy benchmark, buffer assignment memory hazard
//        graph resolver, and multi-block gather-scatter with conflict detection.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.16_champion
//   ../../../output/1.16_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughput = -1.0) {
    std::cout << "  " << std::left << std::setw(55) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughput > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughput << " GB/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Pointer Relations & Copying (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 128-Bit Aligned Vector Memory Copy Benchmark
    //
    // Context: In GPU memory engines and high-performance drivers, copying memory
    //          in 128-bit (16-byte) vector increments achieves dramatically higher
    //          bus utilization and bandwidth than scalar byte copying.
    //
    // Task: Implement `copyVector128(void* dst, const void* src, size_t bytes)`:
    //       - Copies memory in 16-byte chunks (e.g. using `struct alignas(16) Vec128 { uint64_t lo, hi; }`).
    //       - Handles any remaining tail bytes (< 16 bytes).
    //       - Compare against scalar byte copy across a 4 MB buffer and report GB/s.
    // -------------------------------------------------------------------------
    {
        struct alignas(16) Vec128 {
            uint64_t lo;
            uint64_t hi;
        };

        auto copyVector128 = [](void* dst, const void* src, size_t bytes) {
            // --- YOUR CODE STARTS HERE ---
            size_t num_vectors = bytes / sizeof(Vec128);
            size_t remainder = bytes % sizeof(Vec128);

            Vec128* d_vec = reinterpret_cast<Vec128*>(dst);
            const Vec128* s_vec = reinterpret_cast<const Vec128*>(src);

            for (size_t i = 0; i < num_vectors; ++i) {
                d_vec[i] = s_vec[i];
            }

            if (remainder > 0) {
                const uint8_t* s_tail = reinterpret_cast<const uint8_t*>(src) + num_vectors * sizeof(Vec128);
                uint8_t* d_tail = reinterpret_cast<uint8_t*>(dst) + num_vectors * sizeof(Vec128);
                for (size_t i = 0; i < remainder; ++i) {
                    d_tail[i] = s_tail[i];
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t NUM_BYTES = 4 * 1024 * 1024; // 4 MB
        std::vector<uint8_t> src(NUM_BYTES);
        for (size_t i = 0; i < NUM_BYTES; ++i) src[i] = static_cast<uint8_t>(i & 0xFF);
        std::vector<uint8_t> dst(NUM_BYTES, 0);

        // Benchmark vector copy
        auto start = std::chrono::high_resolution_clock::now();
        const int ITERS = 100;
        for (int it = 0; it < ITERS; ++it) {
            copyVector128(dst.data(), src.data(), NUM_BYTES);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double total_gb = (static_cast<double>(NUM_BYTES) * ITERS) / (1024.0 * 1024.0 * 1024.0);
        double gbps = total_gb / elapsed_sec;

        bool ok = (std::memcmp(dst.data(), src.data(), NUM_BYTES) == 0);

        reportStatus("Problem 1: 128-Bit Vector Aligned Memory Copy", ok, gbps);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Buffer Assignment Memory Hazard Graph Resolver
    //
    // Context: In compiler graph optimizations (XLA / torch.compile), intermediate
    //          tensors share a common scratch buffer to conserve memory. If two
    //          tensors overlap in physical memory AND their live execution time
    //          spans overlap, a data hazard (corruption) occurs.
    //
    // Task: Given struct `TensorInterval { int id; size_t mem_start; size_t mem_end; int t_start; int t_end; }`:
    //       Implement `hasMemoryConflict(const TensorInterval& a, const TensorInterval& b)`:
    //       - Time overlaps if: NOT (a.t_end <= b.t_start || b.t_end <= a.t_start).
    //       - Space overlaps if: NOT (a.mem_end <= b.mem_start || b.mem_end <= a.mem_start).
    //       - Conflict exists if BOTH time and space overlap.
    //       Implement `findConflicts(const std::vector<TensorInterval>& tensors)`:
    //       - Returns vector of conflicting ID pairs `std::pair<int, int>`.
    // -------------------------------------------------------------------------
    {
        struct TensorInterval {
            int id;
            size_t mem_start;
            size_t mem_end;
            int t_start;
            int t_end;
        };

        auto findConflicts = [](const std::vector<TensorInterval>& tensors) {
            std::vector<std::pair<int, int>> conflicts;
            // --- YOUR CODE STARTS HERE ---
            auto timeOverlaps = [](const TensorInterval& a, const TensorInterval& b) {
                return !(a.t_end <= b.t_start || b.t_end <= a.t_start);
            };
            auto memOverlaps = [](const TensorInterval& a, const TensorInterval& b) {
                return !(a.mem_end <= b.mem_start || b.mem_end <= a.mem_start);
            };

            for (size_t i = 0; i < tensors.size(); ++i) {
                for (size_t j = i + 1; j < tensors.size(); ++j) {
                    if (timeOverlaps(tensors[i], tensors[j]) && memOverlaps(tensors[i], tensors[j])) {
                        conflicts.emplace_back(tensors[i].id, tensors[j].id);
                    }
                }
            }
            // --- YOUR CODE ENDS HERE ---
            return conflicts;
        };

        std::vector<TensorInterval> intervals = {
            {1, 0, 1024, 0, 5},        // Live t=0..5, mem=0..1024
            {2, 0, 1024, 5, 10},       // Live t=5..10, mem=0..1024 (Reuses mem after tensor 1 dies -> SAFE)
            {3, 512, 1536, 2, 7},      // Live t=2..7, mem=512..1536 (Overlaps t with 1 and mem with 1 -> CONFLICT with 1)
                                       //                             (Overlaps t with 2 and mem with 2 -> CONFLICT with 2)
            {4, 2048, 4096, 0, 10}     // Disjoint memory -> SAFE
        };

        auto conflicts = findConflicts(intervals);
        bool ok = true;
        // Should detect conflict between (1, 3) and (2, 3)
        if (conflicts.size() != 2) ok = false;
        if (conflicts.size() == 2) {
            bool has_1_3 = (conflicts[0] == std::make_pair(1, 3) || conflicts[1] == std::make_pair(1, 3));
            bool has_2_3 = (conflicts[0] == std::make_pair(2, 3) || conflicts[1] == std::make_pair(2, 3));
            if (!has_1_3 || !has_2_3) ok = false;
        }

        reportStatus("Problem 2: Memory Hazard & Lifetime Conflict Resolver", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Multi-Block Strided Gather-Scatter Engine
    //
    // Context: In Mixture-of-Experts (MoE) architectures and ragged embedding
    //          pipelines, blocks of tokens are gathered from scattered source
    //          buffers into contiguous expert processing queues.
    //
    // Task: Implement `executeGatherScatter(const std::vector<TransferOp>& ops)`:
    //       Each `TransferOp`:
    //         `{ const float* src; float* dst; size_t count; }`
    //       - First validates that no two destination spans [dst, dst + count) overlap
    //         (write conflict prevention). If any two dst spans overlap, abort and return false.
    //       - Otherwise, executes all copies: `std::memcpy(dst, src, count * sizeof(float))`
    //         and returns true.
    // -------------------------------------------------------------------------
    {
        struct TransferOp {
            const float* src;
            float* dst;
            size_t count;
        };

        auto executeGatherScatter = [](const std::vector<TransferOp>& ops) -> bool {
            // --- YOUR CODE STARTS HERE ---
            auto spansOverlap = [](uintptr_t a_start, size_t a_len, uintptr_t b_start, size_t b_len) {
                uintptr_t a_end = a_start + a_len;
                uintptr_t b_end = b_start + b_len;
                return !(a_end <= b_start || b_end <= a_start);
            };

            // Check for destination collisions
            for (size_t i = 0; i < ops.size(); ++i) {
                uintptr_t dst_i = reinterpret_cast<uintptr_t>(ops[i].dst);
                size_t bytes_i  = ops[i].count * sizeof(float);
                for (size_t j = i + 1; j < ops.size(); ++j) {
                    uintptr_t dst_j = reinterpret_cast<uintptr_t>(ops[j].dst);
                    size_t bytes_j  = ops[j].count * sizeof(float);
                    if (spansOverlap(dst_i, bytes_i, dst_j, bytes_j)) {
                        return false; // Destination collision!
                    }
                }
            }

            // Execute transfers
            for (const auto& op : ops) {
                std::memcpy(op.dst, op.src, op.count * sizeof(float));
            }
            return true;
            // --- YOUR CODE ENDS HERE ---
        };

        std::vector<float> src_pool(1024);
        for (size_t i = 0; i < 1024; ++i) src_pool[i] = static_cast<float>(i + 1);

        std::vector<float> dst_pool(1024, 0.0f);

        // Valid non-colliding operations
        std::vector<TransferOp> valid_ops = {
            {src_pool.data() + 0, dst_pool.data() + 0, 128},
            {src_pool.data() + 200, dst_pool.data() + 128, 64},
            {src_pool.data() + 500, dst_pool.data() + 192, 256}
        };

        // Colliding operations (dst 100 overlaps dst 0..128)
        std::vector<TransferOp> colliding_ops = {
            {src_pool.data() + 0, dst_pool.data() + 0, 128},
            {src_pool.data() + 200, dst_pool.data() + 100, 64}
        };

        bool ok = true;
        if (!executeGatherScatter(valid_ops)) ok = false;
        if (executeGatherScatter(colliding_ops)) ok = false; // must reject

        // Verify data copied in valid run
        for (size_t i = 0; i < 128; ++i) {
            if (dst_pool[i] != src_pool[i]) ok = false;
        }
        for (size_t i = 0; i < 64; ++i) {
            if (dst_pool[128 + i] != src_pool[200 + i]) ok = false;
        }

        reportStatus("Problem 3: Multi-Block Gather-Scatter with Collision Guard", ok);
        if (ok) passed++;
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
