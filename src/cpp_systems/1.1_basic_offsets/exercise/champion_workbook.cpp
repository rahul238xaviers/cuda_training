#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <algorithm>

// =========================================================================
// CHAMPION WORKBOOK: Basic Offsets & Pointer Arithmetic
//
// Module: 1.1 - Memory Addressing & Pointer Foundations
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Aligned Arena/Bump Allocators, Asynchronous Double-Buffer Pointer
//        Ping-Ponging, and Cache-Thrashing vs Tiled Memory Benchmarking.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.1_champion
//   ../../../output/1.1_champion
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
    std::cout << "--- WORKBOOK: Basic Offsets & Pointer Arithmetic (Champion) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Custom Aligned Bump/Arena Allocator
    //
    // Context: GPU frameworks (e.g. PyTorch c10::CUDAAllocator, TensorRT) cannot afford
    //          expensive OS malloc/cudaMalloc calls during inference. They pre-allocate
    //          a large contiguous buffer and carve out memory with a sub-microsecond
    //          bump pointer aligned to 64- or 128-byte cache lines.
    //
    // Task: Implement an aligned bump allocator over a 64 KB pool.
    //       Given allocation requests with (size_in_bytes, alignment_in_bytes):
    //         Req 0: (size=120, align=64)
    //         Req 1: (size=350, align=128)
    //         Req 2: (size=48,  align=16)
    //         Req 3: (size=1024, align=64)
    //       For each request:
    //         1. Align current bump address upwards: `aligned_addr = (curr + (align - 1)) & ~(align - 1)`
    //         2. Check if (aligned_addr + size <= pool_end)
    //         3. Update current bump address = aligned_addr + size
    //         4. Track total padding bytes wasted
    // -------------------------------------------------------------------------
    {
        const size_t POOL_SIZE = 65536; // 64 KB
        std::vector<uint8_t> pool(POOL_SIZE);
        uintptr_t pool_start = reinterpret_cast<uintptr_t>(pool.data());
        uintptr_t pool_end = pool_start + POOL_SIZE;

        struct AllocReq {
            size_t size;
            size_t align;
        };

        std::vector<AllocReq> requests = {
            {120, 64},
            {350, 128},
            {48, 16},
            {1024, 64},
            {512, 128}
        };

        std::vector<void*> allocated_ptrs(requests.size(), nullptr);
        size_t total_padding_wasted = 0;

        uintptr_t curr_bump = pool_start;

        // TODO: Process each request, compute aligned address, store pointer in allocated_ptrs,
        // advance curr_bump, and accumulate padding into total_padding_wasted.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        uintptr_t verif_bump = pool_start;
        size_t verif_padding = 0;
        for (size_t i = 0; i < requests.size() && p1_passed; ++i) {
            size_t sz = requests[i].size;
            size_t al = requests[i].align;
            uintptr_t expected_addr = (verif_bump + (al - 1)) & ~(al - 1);
            verif_padding += (expected_addr - verif_bump);
            verif_bump = expected_addr + sz;

            if (reinterpret_cast<uintptr_t>(allocated_ptrs[i]) != expected_addr) {
                p1_passed = false;
                break;
            }
            if (expected_addr % al != 0) {
                p1_passed = false;
                break;
            }
        }
        if (total_padding_wasted != verif_padding) p1_passed = false;

        reportStatus("Problem 1: Custom Aligned Bump Allocator", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Double-Buffered Asynchronous Pipeline Pointer Ping-Pong
    //
    // Context: In streaming CUDA inference, while the GPU executes a kernel on
    //          buffer A (Compute Buffer), the host copies the next batch into buffer B
    //          (Transfer Buffer) via PCIe. At each pipeline sync point, pointers swap.
    //
    // Task: Simulate 8 pipeline iterations over two buffers of size N=64 floats:
    //       `buffer_A` and `buffer_B`.
    //       Maintain two pointers: `compute_buf` and `transfer_buf`.
    //       In each iteration i (0 <= i < 8):
    //         1. Fill transfer_buf with (i * 100 + element_index).
    //         2. In compute_buf (if i > 0), verify all elements equal ((i - 1) * 100 + element_index),
    //            and multiply each element by 2 in-place.
    //         3. Swap compute_buf and transfer_buf pointers using std::swap.
    // -------------------------------------------------------------------------
    {
        const int N = 64;
        std::vector<float> buffer_A(N, 0.0f);
        std::vector<float> buffer_B(N, 0.0f);

        float* compute_buf = buffer_A.data();
        float* transfer_buf = buffer_B.data();

        bool p2_passed = true;
        const int NUM_ITERATIONS = 8;

        // TODO: Implement the double-buffer iteration loop.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        // After 8 iterations, compute_buf points to the buffer filled in iteration 7
        // Verify buffer integrity
        reportStatus("Problem 2: Double-Buffered Pointer Ping-Pong Pipeline", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Tiled 16x16 Matrix Transpose vs Cache-Thrashing Benchmark
    //
    // Context: Naive 2D matrix transpose `dst[c, r] = src[r, c]` causes non-coalesced
    //          strided writes across cache lines, dropping memory bandwidth by 5x–10x.
    //          Tiling the transpose into 16x16 blocks keeps reads and writes inside
    //          L1/L2 cache lines before flushing.
    //
    // Task: Given matrix `src` of shape [512, 512] (262,144 floats = 1 MB):
    //       Implement a 16x16 tiled transpose into `dst_tiled`.
    //       Measure memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int DIM = 512;
        const size_t total_elements = DIM * DIM;
        std::vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src[i] = static_cast<float>(i * 0.01f);
        std::vector<float> dst_tiled(total_elements, -1.0f);

        const int TILE_SIZE = 16;

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Transpose src [512, 512] into dst_tiled [512, 512] using 16x16 tiles.
        // For each tile (ti, tj) where ti, tj advance by TILE_SIZE:
        //   For each row i inside tile:
        //     For each col j inside tile:
        //       dst_tiled[j * DIM + i] = src[i * DIM + j]
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int i = 0; i < DIM && p3_passed; ++i) {
            for (int j = 0; j < DIM; ++j) {
                if (dst_tiled[j * DIM + i] != src[i * DIM + j]) {
                    p3_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 3: 16x16 Tiled Matrix Transpose Benchmark", p3_passed, p3_passed ? throughput : -1.0);
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
