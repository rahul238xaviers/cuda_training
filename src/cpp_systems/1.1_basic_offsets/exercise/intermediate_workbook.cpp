#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdint>

using namespace std;

// =========================================================================
// INTERMEDIATE WORKBOOK: Basic Offsets & Pointer Arithmetic
//
// Module: 1.1 - Memory Addressing & Pointer Foundations
// Level:  Intermediate
//
// Focus: Pitched 2D memory layouts (cudaMallocPitch simulation), 128-bit
//        vectorized stream copying, and cache-line alignment padding.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.1_intermediate
//   ../../../output/1.1_intermediate
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
    std::cout << "--- WORKBOOK: Basic Offsets & Pointer Arithmetic (Intermediate) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Pitched Memory Buffer Row Offset Resolution
    //
    // Context: In CUDA, `cudaMallocPitch(&devPtr, &pitch, widthInBytes, height)`
    //          allocates 2D arrays where each row is padded to a multiple of 256/512 bytes
    //          so that every row starts at a hardware-coalesced address.
    //
    // Given:
    //   - Raw byte buffer `pitched_buffer` of ROWS = 8, COLS = 32 floats (4 bytes each).
    //   - PITCH = 256 bytes (useful row width = 128 bytes, padding = 128 bytes per row).
    //   - Base pointer `base_ptr = pitched_buffer.data()`.
    //
    // Goal:
    //   - Calculate each row's start address in byte space using `PITCH`.
    //   - Populate element (r, c) with value `r * 100.0f + c`.
    //   - Ensure padding bytes (offsets 128..255 of each row) remain untouched.
    //
    // Reference:
    //   - See Section 5 (2D Pitched Memory Layouts: cudaMallocPitch) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8;
        const int COLS = 32;
        const size_t PITCH = 256; // 256 bytes per row
        const size_t total_allocated_bytes = ROWS * PITCH;

        std::vector<uint8_t> pitched_buffer(total_allocated_bytes, 0xFF); // Initialize with 0xFF

        uint8_t* const base_ptr = pitched_buffer.data();

        // TODO: Populate 2D pitched buffer elements using row pitch offsets without touching padding.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < ROWS && p1_passed; ++r) {
            const float* row = reinterpret_cast<const float*>(base_ptr + r * PITCH);
            for (int c = 0; c < COLS; ++c) {
                float expected = r * 100.0f + c;
                if (row[c] != expected) {
                    p1_passed = false;
                    break;
                }
            }
            // Check that padding bytes remained untouched (0xFF)
            const uint8_t* padding_start = base_ptr + r * PITCH + (COLS * sizeof(float));
            for (size_t p = 0; p < (PITCH - COLS * sizeof(float)); ++p) {
                if (padding_start[p] != 0xFF) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D Pitched Memory Buffer Offset Resolution", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 128-Bit Vectorized Stream Copy (float4 Emulation)
    //
    // Context: In GPU kernels, loading memory in 128-bit chunks (`float4` / `uint4`)
    //          saturates the DRAM bus by issuing full cache-line transaction segments.
    //
    // Given:
    //   - Source buffer `src` and destination buffer `dst` of N = 4096 floats (16,384 bytes).
    //   - 16-byte aligned vector struct `Float4` (holding 4 floats = 128 bits).
    //
    // Goal:
    //   - Transfer all elements from `src` into `dst` using 128-bit vector pointers.
    //   - Verify data integrity and benchmark bandwidth throughput in GB/s.
    //
    // Reference:
    //   - See Section 6 (Vectorized Memory Loads: 128-Bit Transfers) of theory.md.
    // -------------------------------------------------------------------------
    {
        struct alignas(16) Float4 {
            float x, y, z, w;
        };

        const int N = 4096;
        std::vector<float> src(N);
        for (int i = 0; i < N; ++i) src[i] = static_cast<float>(i * 0.125f);
        std::vector<float> dst(N, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Copy data from src to dst using 128-bit Float4 vector pointers.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * N * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p2_passed = true;
        for (int i = 0; i < N; ++i) {
            if (dst[i] != src[i]) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: 128-Bit Vectorized Stream Copy (float4)", p2_passed, p2_passed ? throughput : -1.0);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Pointer Cache-Line Alignment & Padding Bytes Calculation
    //
    // Context: CPU and GPU memory subsystems read data in 64-byte or 128-byte cache lines.
    //          Misaligned base pointers force every transaction to span two cache lines,
    //          halving effective memory bandwidth.
    //
    // Given:
    //   - List of arbitrary addresses `addresses` (as `uintptr_t`).
    //   - Target alignment boundary ALIGNMENT = 256 bytes.
    //
    // Goal:
    //   - For each address, compute `aligned_results[i]` rounded up to the nearest ALIGNMENT boundary.
    //   - Compute `padding_results[i]` representing the slack bytes skipped to align.
    //
    // Reference:
    //   - See Section 4 (Hardware Memory Alignment & Cache-Line Granularity) of theory.md.
    // -------------------------------------------------------------------------
    {
        std::vector<uintptr_t> addresses = {
            0x1000, // Already aligned (0x1000 % 256 == 0)
            0x1001, // 255 bytes padding needed
            0x103F, // 193 bytes padding needed
            0x1100, // Already aligned (0x1100 = 4352 % 256 == 0)
            0x2A1B4 // 0x2A1B4 = 172468 -> nearest 256 boundary = 172544 (0x2A200)
        };

        const size_t ALIGNMENT = 256;
        std::vector<uintptr_t> aligned_results(addresses.size(), 0);
        std::vector<size_t> padding_results(addresses.size(), 0);

        // TODO: Compute power-of-2 aligned addresses and padding bytes using bitwise operations.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (size_t i = 0; i < addresses.size(); ++i) {
            uintptr_t addr = addresses[i];
            uintptr_t expected_aligned = (addr % ALIGNMENT == 0) ? addr : (addr + (ALIGNMENT - (addr % ALIGNMENT)));
            size_t expected_padding = expected_aligned - addr;

            if (aligned_results[i] != expected_aligned || padding_results[i] != expected_padding) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Cache-Line Alignment & Padding Calculation", p3_passed);
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
