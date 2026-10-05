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
// BEGINNER WORKBOOK: Null, Bounds & Alignment Checks
//
// Module: 1.13 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Beginner
//
// Focus: Boundary-clamped memory access, pointer span overlap detection,
//        and power-of-two memory alignment arithmetic.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.13_beginner
//   ../../../output/1.13_beginner
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
    std::cout << "--- WORKBOOK: Null, Bounds & Alignment Checks (Beginner) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Safe Boundary-Clamped Strided 2D Matrix Fetcher
    //
    // Context: In image processing, convolutions with replicate padding, and
    //          texture sampling, threads frequently query coordinates outside
    //          the physical tensor dimensions. Instead of crashing with illegal
    //          memory access, coordinates must be safely clamped to [0, Dim - 1].
    //
    // Task: Implement `safeClampedFetch(const float* mat, int H, int W, int r, int c)`:
    //       - Clamps row index `r` to [0, H - 1].
    //       - Clamps col index `c` to [0, W - 1].
    //       - Returns mat[clamped_r * W + clamped_c].
    //       - If mat is nullptr or H <= 0 or W <= 0, return 0.0f.
    // -------------------------------------------------------------------------
    {
        auto safeClampedFetch = [](const float* mat, int H, int W, int r, int c) -> float {
            // --- YOUR CODE STARTS HERE ---
            if (!mat || H <= 0 || W <= 0) return 0.0f;
            int cr = std::clamp(r, 0, H - 1);
            int cc = std::clamp(c, 0, W - 1);
            return mat[cr * W + cc];
            // --- YOUR CODE ENDS HERE ---
        };

        const int H = 64;
        const int W = 128;
        std::vector<float> mat(H * W);
        for (int i = 0; i < H * W; ++i) {
            mat[i] = static_cast<float>(i * 1.5f + 0.25f);
        }

        bool ok = true;
        // Test in-bounds
        if (safeClampedFetch(mat.data(), H, W, 10, 20) != mat[10 * W + 20]) ok = false;
        // Test negative clamp
        if (safeClampedFetch(mat.data(), H, W, -5, -10) != mat[0]) ok = false;
        // Test positive clamp
        if (safeClampedFetch(mat.data(), H, W, 100, 200) != mat[(H - 1) * W + (W - 1)]) ok = false;
        // Test edge clamp (row in-bounds, col out-of-bounds)
        if (safeClampedFetch(mat.data(), H, W, 30, -1) != mat[30 * W + 0]) ok = false;
        if (safeClampedFetch(mat.data(), H, W, 30, 150) != mat[30 * W + (W - 1)]) ok = false;
        // Test null pointer safety
        if (safeClampedFetch(nullptr, H, W, 0, 0) != 0.0f) ok = false;

        reportStatus("Problem 1: Safe Boundary-Clamped 2D Fetcher", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Memory Range Overlap and Disjointness Sanitizer
    //
    // Context: In GPU pipelines and CUDA memory copies (e.g. cudaMemcpyDeviceToDevice),
    //          launching kernels or copies where source and destination buffers
    //          partially overlap causes undefined behavior and race conditions.
    //
    // Task: Implement `classifyOverlap(const void* ptrA, size_t sizeA, const void* ptrB, size_t sizeB)`:
    //       Given spans [ptrA, ptrA + sizeA) and [ptrB, ptrB + sizeB):
    //       Return enum OverlapType:
    //         - DISJOINT: Neither range touches or overlaps.
    //         - EXACT_MATCH: Both ranges start at same address and have equal size.
    //         - SUBSET_A_IN_B: Range A is completely contained within Range B.
    //         - SUBSET_B_IN_A: Range B is completely contained within Range A.
    //         - PARTIAL_OVERLAP: The ranges intersect partially without full containment.
    // -------------------------------------------------------------------------
    {
        enum OverlapType {
            DISJOINT,
            EXACT_MATCH,
            SUBSET_A_IN_B,
            SUBSET_B_IN_A,
            PARTIAL_OVERLAP
        };

        auto classifyOverlap = [](const void* ptrA, size_t sizeA,
                                  const void* ptrB, size_t sizeB) -> OverlapType {
            // --- YOUR CODE STARTS HERE ---
            uintptr_t a_start = reinterpret_cast<uintptr_t>(ptrA);
            uintptr_t a_end   = a_start + sizeA;
            uintptr_t b_start = reinterpret_cast<uintptr_t>(ptrB);
            uintptr_t b_end   = b_start + sizeB;

            if (a_start == b_start && a_end == b_end) {
                return EXACT_MATCH;
            }
            if (a_end <= b_start || b_end <= a_start) {
                return DISJOINT;
            }
            if (a_start >= b_start && a_end <= b_end) {
                return SUBSET_A_IN_B;
            }
            if (b_start >= a_start && b_end <= a_end) {
                return SUBSET_B_IN_A;
            }
            return PARTIAL_OVERLAP;
            // --- YOUR CODE ENDS HERE ---
        };

        std::vector<uint8_t> buffer(1024, 0);
        uint8_t* base = buffer.data();

        bool ok = true;
        // Disjoint checks
        if (classifyOverlap(base, 100, base + 100, 100) != DISJOINT) ok = false; // touching boundary
        if (classifyOverlap(base, 100, base + 200, 50) != DISJOINT) ok = false;
        // Exact match
        if (classifyOverlap(base + 50, 128, base + 50, 128) != EXACT_MATCH) ok = false;
        // Subset A in B
        if (classifyOverlap(base + 60, 20, base + 50, 100) != SUBSET_A_IN_B) ok = false;
        // Subset B in A
        if (classifyOverlap(base + 50, 100, base + 60, 20) != SUBSET_B_IN_A) ok = false;
        // Partial overlap
        if (classifyOverlap(base + 50, 100, base + 100, 100) != PARTIAL_OVERLAP) ok = false;
        if (classifyOverlap(base + 100, 100, base + 50, 100) != PARTIAL_OVERLAP) ok = false;

        reportStatus("Problem 2: Buffer Span Overlap Sanitizer", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Multi-Byte Alignment Offset and Padding Calculator
    //
    // Context: Modern tensor cores and vectorized instructions require memory
    //          to be aligned to 16 bytes (float4), 32 bytes (AVX2), or 64 bytes
    //          (cache lines, cuBLAS). Given an arbitrary pointer and power-of-two
    //          alignment, we must calculate the padding bytes required to reach
    //          the next aligned boundary.
    //
    // Task: Implement:
    //       1. `isAligned(const void* ptr, size_t alignment)`:
    //          Returns true if ptr is aligned to `alignment` bytes.
    //       2. `bytesToNextAligned(const void* ptr, size_t alignment)`:
    //          Returns number of bytes to advance ptr to reach alignment boundary.
    //          (Returns 0 if already aligned).
    //       3. `alignForward(void* ptr, size_t alignment)`:
    //          Returns adjusted pointer advanced to next aligned boundary.
    // -------------------------------------------------------------------------
    {
        auto isAligned = [](const void* ptr, size_t alignment) -> bool {
            // --- YOUR CODE STARTS HERE ---
            return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
            // --- YOUR CODE ENDS HERE ---
        };

        auto bytesToNextAligned = [](const void* ptr, size_t alignment) -> size_t {
            // --- YOUR CODE STARTS HERE ---
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            size_t rem = addr & (alignment - 1);
            return rem == 0 ? 0 : (alignment - rem);
            // --- YOUR CODE ENDS HERE ---
        };

        auto alignForward = [&](void* ptr, size_t alignment) -> void* {
            // --- YOUR CODE STARTS HERE ---
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(ptr);
            return byte_ptr + bytesToNextAligned(ptr, alignment);
            // --- YOUR CODE ENDS HERE ---
        };

        std::vector<uint8_t> memory_pool(2048, 0);
        uint8_t* raw_ptr = memory_pool.data();

        bool ok = true;
        for (size_t align : {2, 4, 8, 16, 32, 64, 128}) {
            for (size_t offset = 0; offset < 256; ++offset) {
                uint8_t* test_ptr = raw_ptr + offset;
                uintptr_t addr = reinterpret_cast<uintptr_t>(test_ptr);
                bool expected_aligned = (addr % align == 0);

                if (isAligned(test_ptr, align) != expected_aligned) {
                    ok = false;
                    break;
                }

                size_t pad = bytesToNextAligned(test_ptr, align);
                void* aligned_p = alignForward(test_ptr, align);
                uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(aligned_p);

                if (aligned_addr % align != 0) {
                    ok = false;
                    break;
                }
                if (aligned_addr < addr || aligned_addr >= addr + align) {
                    ok = false;
                    break;
                }
                if (reinterpret_cast<uint8_t*>(aligned_p) != test_ptr + pad) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }

        reportStatus("Problem 3: Multi-Byte Alignment Offset Calculator", ok);
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
