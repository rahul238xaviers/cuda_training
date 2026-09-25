#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Multi-Dim & Alignment
//
// Module: 1.10 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Beginner
//
// Focus: 2D row-stride alignment padding, AoS to SoA layout conversion,
//        and compiler struct padding inspection.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.10_beginner
//   ../../../output/1.10_beginner
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
    std::cout << "--- WORKBOOK: Multi-Dim & Alignment (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Row-Stride Alignment Padding (64-Byte Cache-Line Alignment)
    //
    // Context: In GPU memory, each row of a matrix must start at a 64-byte (16 float)
    //          boundary to avoid unaligned memory access penalties.
    //          When COLS = 13 (52 bytes), each row must be padded to PITCH = 16 floats (64 bytes).
    //
    // Task: Given unpadded matrix `unpadded` [ROWS=4, COLS=13] (52 floats):
    //       Embed into `padded` [ROWS=4, PITCH=16] (64 floats total).
    //       Ensure padding elements in columns 13..15 of each row are 0.0f.
    // -------------------------------------------------------------------------
    {
        const int ROWS = 4, COLS = 13, PITCH = 16;
        std::vector<float> unpadded(ROWS * COLS);
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                unpadded[r * COLS + c] = static_cast<float>(r * 100 + c);
            }
        }

        std::vector<float> padded(ROWS * PITCH, -1.0f);

        // TODO: Copy unpadded into padded, setting row stride = PITCH.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 0; r < ROWS && p1_passed; ++r) {
            for (int c = 0; c < COLS; ++c) {
                float expected = unpadded[r * COLS + c];
                if (padded[r * PITCH + c] != expected) p1_passed = false;
            }
            for (int p = COLS; p < PITCH; ++p) {
                if (padded[r * PITCH + p] != 0.0f) p1_passed = false;
            }
        }

        reportStatus("Problem 1: 2D Row-Stride Alignment Padding (64-Byte)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Array-of-Structures (AoS) to Structure-of-Arrays (SoA)
    //
    // Context: In CPU/GPU graphics and particle simulations, storing data as
    //          Array-of-Structures `struct Point3D { float x, y, z; }` causes strided
    //          uncoalesced access. Converting to Structure-of-Arrays (separate float* x,
    //          float* y, float* z arrays) allows 100% coalesced 128-bit vector loads.
    //
    // Task: Given array of N=64 Particle structs:
    //       Split into three contiguous float arrays `soa_x`, `soa_y`, `soa_z`.
    // -------------------------------------------------------------------------
    {
        struct Particle {
            float x, y, z;
        };

        const int N = 64;
        std::vector<Particle> aos(N);
        for (int i = 0; i < N; ++i) {
            aos[i] = { static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(i * 3) };
        }

        std::vector<float> soa_x(N, 0.0f);
        std::vector<float> soa_y(N, 0.0f);
        std::vector<float> soa_z(N, 0.0f);

        // TODO: De-interleave AoS into SoA arrays.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < N; ++i) {
            if (soa_x[i] != aos[i].x || soa_y[i] != aos[i].y || soa_z[i] != aos[i].z) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: AoS to SoA Layout Conversion for Coalesced Access", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Compiler Struct Padding & Offsetof Verification
    //
    // Context: In mixed-type C++ structs, the compiler inserts padding bytes
    //          to align 64-bit doubles or 32-bit floats.
    //          Understanding struct packing prevents memory bloat when transferring
    //          vertex or token metadata across PCIe to GPUs.
    //
    // Task: Given struct `NodeMetadata`:
    //         char flag;      // 1 byte -> followed by 3 bytes padding
    //         int id;         // 4 bytes
    //         double weight;  // 8 bytes
    //       Inspect actual memory addresses of its fields relative to struct base.
    //       Verify that sizeof(NodeMetadata) == 16 bytes.
    // -------------------------------------------------------------------------
    {
        struct NodeMetadata {
            char flag;
            int id;
            double weight;
        };

        NodeMetadata node = {'A', 42, 3.14159};
        uintptr_t base = reinterpret_cast<uintptr_t>(&node);
        uintptr_t off_flag = reinterpret_cast<uintptr_t>(&node.flag) - base;
        uintptr_t off_id = reinterpret_cast<uintptr_t>(&node.id) - base;
        uintptr_t off_weight = reinterpret_cast<uintptr_t>(&node.weight) - base;

        bool p3_passed = (sizeof(NodeMetadata) == 16) &&
                         (off_flag == 0) &&
                         (off_id == 4) &&
                         (off_weight == 8);

        reportStatus("Problem 3: Struct Memory Padding & Alignment Inspection", p3_passed);
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
