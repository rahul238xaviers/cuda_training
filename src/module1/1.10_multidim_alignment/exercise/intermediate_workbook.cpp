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
// INTERMEDIATE WORKBOOK: Multi-Dim & Alignment
//
// Module: 1.10 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Intermediate
//
// Focus: Warp-aligned 2D matrix padding (multiples of 32), 3D pitched memory
//        extents (cudaMalloc3D simulation), and cache-line aligned structures.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.10_intermediate
//   ../../../output/1.10_intermediate
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
    std::cout << "--- WORKBOOK: Multi-Dim & Alignment (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Warp-Aligned 2D Matrix Padding (Multiples of 32 Threads)
    //
    // Context: In CUDA, threads in a warp (32 threads) execute simultaneously.
    //          If a matrix width COLS is not a multiple of 32, a warp reading a row
    //          spans across two physical rows, splitting memory transactions in two.
    //          Padded width = (COLS + 31) & ~31.
    //
    // Task: Given unpadded matrix `unpadded` [ROWS=8, COLS=50]:
    //       Compute PITCH_COLS (next multiple of 32 = 64 floats).
    //       Embed unpadded into `padded` [ROWS=8, PITCH_COLS=64] with zero padding.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 8, COLS = 50;
        const int PITCH_COLS = (COLS + 31) & ~31; // 64

        std::vector<float> unpadded(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                unpadded[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> padded(ROWS * PITCH_COLS, -1.0f);

        // TODO: Copy unpadded into padded with row stride = PITCH_COLS.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < ROWS && p1_passed; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (padded[r * PITCH_COLS + c] != unpadded[r * COLS + c]) p1_passed = false;
            }
            for (int p = COLS; p < PITCH_COLS; ++p) {
                if (padded[r * PITCH_COLS + p] != 0.0f) p1_passed = false;
            }
        }

        reportStatus("Problem 1: Warp-Aligned 2D Matrix Padding (Multiple of 32)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 3D Pitched Memory Extents (cudaMalloc3D Simulation)
    //
    // Context: In `cudaMalloc3D`, 3D volumes require two levels of padding:
    //            1. `pitch`: bytes per row, aligned to 128 bytes (32 floats)
    //            2. `slice_pitch`: bytes per 2D slice = pitch * HEIGHT
    //          Formula for element (d, h, w):
    //            byte_offset = d * slice_pitch + h * pitch + w * sizeof(float)
    //
    // Task: Given volume dimensions WIDTH=20, HEIGHT=10, DEPTH=4 floats:
    //       1. Compute `pitch_floats` = round_up_to_32(WIDTH) = 32 floats (128 bytes)
    //       2. Compute `slice_floats` = pitch_floats * HEIGHT = 320 floats
    //       3. Allocate flat buffer `pitched_volume` of size DEPTH * slice_floats.
    //       4. Write (d*1000 + h*10 + w) into each valid element (d, h, w).
    // -------------------------------------------------------------------------
    {
        const int WIDTH = 20, HEIGHT = 10, DEPTH = 4;
        const int PITCH_FLOATS = (WIDTH + 31) & ~31; // 32
        const int SLICE_FLOATS = PITCH_FLOATS * HEIGHT; // 320
        const size_t total_elements = DEPTH * SLICE_FLOATS;

        std::vector<float> pitched_volume(total_elements, -1.0f);

        // TODO: Populate valid elements in pitched_volume.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int d = 0; d < DEPTH && p2_passed; ++d) {
            for (int h = 0; h < HEIGHT && p2_passed; ++h) {
                for (int w = 0; w < WIDTH; ++w) {
                    size_t idx = (size_t)d * SLICE_FLOATS + (size_t)h * PITCH_FLOATS + w;
                    float expected = static_cast<float>(d * 1000 + h * 10 + w);
                    if (pitched_volume[idx] != expected) {
                        p2_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 2: 3D Pitched Memory Extents (cudaMalloc3D Simulation)", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: alignas(64) Cache-Line Struct Alignment to Avoid False Sharing
    //
    // Context: In multi-threaded CPU and GPU staging buffers, two adjacent worker
    //          threads writing to distinct variables within the SAME 64-byte cache line
    //          cause "False Sharing", degrading multi-core performance by 10x.
    //          Using `alignas(64)` guarantees each worker's data sits on its own cache line.
    //
    // Task: Define a struct `WorkerSlot` aligned to 64 bytes containing:
    //         uint64_t task_id;
    //         float result;
    //       Verify that `alignof(WorkerSlot) == 64` and `sizeof(WorkerSlot) == 64`.
    // -------------------------------------------------------------------------
    {
        struct alignas(64) WorkerSlot {
            uint64_t task_id;
            float result;
        };

        WorkerSlot slots[4];
        uintptr_t addr0 = reinterpret_cast<uintptr_t>(&slots[0]);
        uintptr_t addr1 = reinterpret_cast<uintptr_t>(&slots[1]);
        ptrdiff_t distance = addr1 - addr0;

        bool p3_passed = (alignof(WorkerSlot) == 64) &&
                         (sizeof(WorkerSlot) == 64) &&
                         (distance == 64);

        reportStatus("Problem 3: alignas(64) Struct Alignment & False-Sharing Prevention", p3_passed);
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
