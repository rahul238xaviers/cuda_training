#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <algorithm>

using namespace std;

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
    cout << "--- WORKBOOK: Basic Offsets & Pointer Arithmetic (Champion) ---" << endl;
    cout << "=================================================================" << endl;

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
    // Given:
    //   - Memory pool of size POOL_SIZE = 65,536 bytes (64 KB).
    //   - List of allocation requests: `requests` specifying `{size, align}`.
    //   - Current bump pointer `curr_bump = pool_start`.
    //
    // Goal:
    //   - Carve out aligned memory for each request, storing results in `allocated_ptrs`.
    //   - Advance `curr_bump` past each allocated block.
    //   - Accumulate total alignment padding bytes wasted in `total_padding_wasted`.
    //
    // Reference:
    //   - See Section 7 (Custom Aligned Bump Allocator: Memory Pools) of theory.md.
    // -------------------------------------------------------------------------
    {
        const size_t POOL_SIZE = 65536; // 64 KB
        vector<uint8_t> pool(POOL_SIZE);
        uintptr_t pool_start = reinterpret_cast<uintptr_t>(pool.data());
        uintptr_t pool_end = pool_start + POOL_SIZE;

        struct AllocReq {
            size_t size;
            size_t align;
        };

        vector<AllocReq> requests = {
            {120, 64},
            {350, 128},
            {48, 16},
            {1024, 64},
            {512, 128}
        };

        vector<void*> allocated_ptrs(requests.size(), nullptr);
        size_t total_padding_wasted = 0;

        uintptr_t curr_bump = pool_start;

        // TODO: Allocate each request with power-of-2 alignment, advancing the bump pointer and tracking padding.
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
    // Given:
    //   - Two buffers `buffer_A` and `buffer_B` of size N = 64 floats.
    //   - Initial pointers: `compute_buf = buffer_A.data()`, `transfer_buf = buffer_B.data()`.
    //   - Pipeline iterations: NUM_ITERATIONS = 8.
    //
    // Goal:
    //   - In each iteration i:
    //     1. Fill `transfer_buf` with (i * 100 + element_index).
    //     2. If i > 0, verify and multiply each element of `compute_buf` by 2 in-place.
    //     3. Swap `compute_buf` and `transfer_buf` pointers (zero memory copies).
    //
    // Reference:
    //   - See Section 8 (Asynchronous Double-Buffering: Pointer Ping-Pong) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 64;
        vector<float> buffer_A(N, 0.0f);
        vector<float> buffer_B(N, 0.0f);

        float* compute_buf = buffer_A.data();
        float* transfer_buf = buffer_B.data();

        bool p2_passed = true;
        const int NUM_ITERATIONS = 8;

        // TODO: Execute the double-buffered pipeline swapping compute and transfer pointers each iteration.
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
    // Given:
    //   - Matrix `src` of shape [DIM=512, DIM=512] (262,144 floats = 1 MB).
    //   - Tile dimension TILE_SIZE = 16.
    //   - Output buffer `dst_tiled`.
    //
    // Goal:
    //   - Transpose `src` into `dst_tiled` using 16x16 cache-blocked tiles to eliminate cache thrashing.
    //   - Benchmark memory throughput in GB/s.
    //
    // Reference:
    //   - See Section 9 (2D Matrix Tiling & Cache Thrashing Prevention) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int DIM = 512;
        const size_t total_elements = DIM * DIM;
        vector<float> src(total_elements);
        for (size_t i = 0; i < total_elements; ++i) src[i] = static_cast<float>(i * 0.01f);
        vector<float> dst_tiled(total_elements, -1.0f);

        const int TILE_SIZE = 16;

        auto start = chrono::high_resolution_clock::now();

        // TODO: Transpose matrix using 16x16 cache-blocked tiles to prevent cache thrashing.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = chrono::high_resolution_clock::now();
        double elapsed_sec = chrono::duration<double>(end - start).count();
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
