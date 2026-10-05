#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <unordered_map>

// =========================================================================
// INTERMEDIATE WORKBOOK: Single & Array Allocations
//
// Module: 1.9 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Intermediate
//
// Focus: 64-byte aligned memory allocations, peak memory watermark profiler,
//        and contiguous vs fragmented 2D allocations.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.9_intermediate
//   ../../../output/1.9_intermediate
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
    std::cout << "--- WORKBOOK: Single & Array Allocations (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 64-Byte Cache-Line Aligned Heap Allocation
    //
    // Context: SIMD instructions (AVX-512) and GPU PCIe staging buffers require
    //          heap allocations to be aligned to 64-byte cache-line boundaries.
    //
    // Task: Allocate a buffer of N=1024 floats (4096 bytes) aligned to 64 bytes.
    //       Verify `(reinterpret_cast<uintptr_t>(aligned_ptr) % 64 == 0)`.
    //       Fill with test values, verify, and properly deallocate.
    // -------------------------------------------------------------------------
    {
        const size_t N = 1024;
        const size_t ALIGN = 64;
        float* aligned_ptr = nullptr;

        // TODO: Allocate 64-byte aligned buffer using posix_memalign or aligned_alloc.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (aligned_ptr != nullptr) &&
                         (reinterpret_cast<uintptr_t>(aligned_ptr) % ALIGN == 0);

        if (aligned_ptr) {
            for (size_t i = 0; i < N; ++i) aligned_ptr[i] = static_cast<float>(i * 0.5f);
            if (aligned_ptr[10] != 5.0f) p1_passed = false;
            std::free(aligned_ptr);
        }

        reportStatus("Problem 1: 64-Byte Cache-Line Aligned Allocation", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Device Memory Allocator Profiler (Peak Watermark Tracking)
    //
    // Context: In GPU frameworks, tracking memory consumption is critical for preventing
    //          Out-Of-Memory (OOM) errors. Allocators track current active bytes
    //          and record the maximum peak watermark reached.
    //
    // Task: Implement a tracking allocator class `MemoryTracker`:
    //         - `void* allocate(size_t bytes)`: allocates memory and updates current_bytes and peak_bytes
    //         - `void deallocate(void* ptr)`: frees memory and decrements current_bytes
    // -------------------------------------------------------------------------
    {
        class MemoryTracker {
        public:
            size_t current_bytes = 0;
            size_t peak_bytes = 0;
            std::unordered_map<void*, size_t> alloc_map;

            void* allocate(size_t bytes) {
                void* p = std::malloc(bytes);
                if (p) {
                    alloc_map[p] = bytes;
                    current_bytes += bytes;
                    if (current_bytes > peak_bytes) peak_bytes = current_bytes;
                }
                return p;
            }

            void deallocate(void* ptr) {
                if (ptr && alloc_map.count(ptr)) {
                    current_bytes -= alloc_map[ptr];
                    alloc_map.erase(ptr);
                    std::free(ptr);
                }
            }
        };

        MemoryTracker tracker;

        // Sequence of allocations
        void* p1 = tracker.allocate(1000); // cur: 1000, peak: 1000
        void* p2 = tracker.allocate(2500); // cur: 3500, peak: 3500
        tracker.deallocate(p1);            // cur: 2500, peak: 3500
        void* p3 = tracker.allocate(1500); // cur: 4000, peak: 4000
        tracker.deallocate(p2);            // cur: 1500, peak: 4000
        tracker.deallocate(p3);            // cur: 0,    peak: 4000

        bool p2_passed = (tracker.current_bytes == 0) && (tracker.peak_bytes == 4000);
        reportStatus("Problem 2: Device Allocator Peak Watermark Tracker", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Contiguous 2D Heap Matrix Allocation
    //
    // Context: In novice C++, 2D arrays are often allocated as an array of row pointers
    //          `float** rows = new float*[M]`, each pointing to a separate `new float[N]`.
    //          This causes cache thrashing and memory fragmentation.
    //          High-performance code allocates a single flat buffer: `float* data = new float[M*N]`.
    //
    // Task: Allocate a single contiguous heap buffer of shape [M=32, N=32] (1024 floats).
    //       Access and populate element (r, c) with (r * 100 + c) using flat indexing.
    //       Verify data and clean up with a single `delete[]` or `free()`.
    // -------------------------------------------------------------------------
    {
        const int M = 32, N = 32;
        float* contiguous_mat = nullptr;

        // TODO: Allocate a single flat buffer of M * N floats and populate it.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = (contiguous_mat != nullptr);
        if (contiguous_mat) {
            for (int r = 0; r < M && p3_passed; ++r) {
                for (int c = 0; c < N; ++c) {
                    if (contiguous_mat[r * N + c] != static_cast<float>(r * 100 + c)) {
                        p3_passed = false;
                        break;
                    }
                }
            }
            std::free(contiguous_mat);
        }

        reportStatus("Problem 3: Contiguous 2D Heap Matrix Allocation", p3_passed);
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
