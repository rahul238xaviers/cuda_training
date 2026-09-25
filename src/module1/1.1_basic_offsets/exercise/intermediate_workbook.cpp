#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdint>

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
    //          To access element (r, c), you must step by `pitch` bytes:
    //            float* row_ptr = reinterpret_cast<float*>(raw_bytes + r * pitch);
    //            float val = row_ptr[c];
    //
    // Task: Given a raw byte buffer `pitched_buffer` of ROWS=8, COLS=32 (each float = 4 bytes).
    //       Width in bytes = 32 * 4 = 128 bytes, but PITCH = 256 bytes (128 bytes padding per row).
    //       Write value (r * 100 + c) into every element (r, c) using pitched pointer offsets.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8;
        const int COLS = 32;
        const size_t PITCH = 256; // 256 bytes per row
        const size_t total_allocated_bytes = ROWS * PITCH;

        std::vector<uint8_t> pitched_buffer(total_allocated_bytes, 0xFF); // Initialize with 0xFF

        uint8_t* const base_ptr = pitched_buffer.data();

        // TODO: For each row r in [0, ROWS) and col c in [0, COLS):
        // Compute pointer to row r using PITCH byte offset, and assign value (r * 100.0f + c).
        // Ensure padding bytes between col 32 and pitch (128..255) are NOT overwritten!
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
    // Task: Given source and destination buffers of N=4096 floats (16,384 bytes):
    //       Copy data from `src` to `dst` using 16-byte chunks (`struct alignas(16) Float4`)
    //       via pointer casting.
    //       Measure memory throughput in GB/s!
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

        // TODO: Copy src to dst using Float4 pointers.
        // const Float4* src_vec = reinterpret_cast<const Float4*>(src.data());
        // Float4* dst_vec = reinterpret_cast<Float4*>(dst.data());
        // Copy all (N / 4) Float4 elements.
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
    // Task: Given arbitrary memory addresses `addresses` (as uintptr_t):
    //       For each address, calculate:
    //         1. `aligned_address`: rounded UP to the nearest 64-byte boundary using bitwise ops.
    //         2. `padding_bytes`: the number of padding bytes between address and aligned_address.
    // -------------------------------------------------------------------------
    {
        std::vector<uintptr_t> addresses = {
            0x1000, // Already aligned (0x1000 % 64 == 0)
            0x1001, // 63 bytes padding needed
            0x103F, // 1 byte padding needed
            0x1040, // Already aligned
            0x2A1B4 // 0x2A1B4 = 172468 -> nearest 64 boundary = 172480 (0x2A1C0)
        };

        const size_t ALIGNMENT = 64;
        std::vector<uintptr_t> aligned_results(addresses.size(), 0);
        std::vector<size_t> padding_results(addresses.size(), 0);

        // TODO: For each address in addresses, compute aligned_address using bitwise formula
        // (addr + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1), and padding = aligned_address - addr.
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
