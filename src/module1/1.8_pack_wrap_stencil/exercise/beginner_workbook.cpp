#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

// =========================================================================
// BEGINNER WORKBOOK: Pack, Wrap, Stencil & Crops
//
// Module: 1.8 - Multi-Dimensional Layout Foundations
// Level:  Beginner
//
// Focus: 2D 5-point cross stencil (Laplacian), 1D 3-point convolution stencil,
//        and packed bit array operations.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.8_beginner
//   ../../../output/1.8_beginner
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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D 5-Point Cross Stencil (Laplacian Operator)
    //
    // Context: In physics simulations (heat equation / wave equation) and image edge
    //          detection, a 5-point star stencil evaluates neighbors:
    //            Laplacian(r, c) = 4 * Center - (Up + Down + Left + Right)
    //
    // Task: Given grid `grid` [H=8, W=8]:
    //       Compute `laplacian` for all interior pixels (r in [1..6], c in [1..6]).
    //       Leave boundary pixels as 0.0f.
    // -------------------------------------------------------------------------
    {
        const int H = 8, W = 8;
        std::vector<float> grid(H * W);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                grid[r * W + c] = static_cast<float>(r * 10 + c);
            }
        }

        std::vector<float> laplacian(H * W, 0.0f);

        // TODO: Apply 5-point stencil on interior pixels.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int r = 1; r < H - 1 && p1_passed; ++r) {
            for (int c = 1; c < W - 1; ++c) {
                float center = grid[r * W + c];
                float up     = grid[(r - 1) * W + c];
                float down   = grid[(r + 1) * W + c];
                float left   = grid[r * W + (c - 1)];
                float right  = grid[r * W + (c + 1)];
                float expected = 4.0f * center - (up + down + left + right);
                if (std::abs(laplacian[r * W + c] - expected) > 1e-4f) {
                    p1_passed = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: 2D 5-Point Cross Stencil (Laplacian)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 1D 3-Point Convolution Stencil with Zero Boundaries
    //
    // Context: 1D convolution stencils smooth time-series sensor data or audio:
    //            y[i] = kernel[0] * x[i-1] + kernel[1] * x[i] + kernel[2] * x[i+1]
    //          Treat values outside [0, N-1] as 0.0f.
    //
    // Task: Given `signal` of size N=32 and `kernel` = {0.25f, 0.5f, 0.25f}:
    //       Compute output `smooth_signal` [N=32].
    // -------------------------------------------------------------------------
    {
        const int N = 32;
        std::vector<float> signal(N);
        for (int i = 0; i < N; ++i) signal[i] = static_cast<float>(i % 5 + 1);

        float k0 = 0.25f, k1 = 0.5f, k2 = 0.25f;
        std::vector<float> smooth_signal(N, -1.0f);

        // TODO: Compute smooth_signal with 3-point stencil.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < N; ++i) {
            float left = (i > 0) ? signal[i - 1] : 0.0f;
            float mid  = signal[i];
            float right = (i < N - 1) ? signal[i + 1] : 0.0f;
            float expected = k0 * left + k1 * mid + k2 * right;
            if (std::abs(smooth_signal[i] - expected) > 1e-4f) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: 1D 3-Point Convolution Stencil", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Packed Bit Array (Bit-Level Packing & Unpacking)
    //
    // Context: In GPU boolean masks, packing 8 boolean flags per byte saves 87.5% memory.
    //
    // Task: Given array of 16 boolean flags `flags`:
    //       1. Pack into 2 bytes in `packed_bytes` (flag i goes to bit i % 8 of byte i / 8).
    //       2. Unpack `packed_bytes` back into `unpacked_flags` and verify identity.
    // -------------------------------------------------------------------------
    {
        const int N = 16;
        std::vector<bool> flags = {
            true, false, true, true, false, false, true, false,
            false, true, true, false, true, true, false, true
        };

        std::vector<uint8_t> packed_bytes(2, 0);
        std::vector<bool> unpacked_flags(N, false);

        // TODO: Pack flags into packed_bytes, then unpack into unpacked_flags.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            if (unpacked_flags[i] != flags[i]) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Packed Bit Array (8 Flags per Byte)", p3_passed);
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
