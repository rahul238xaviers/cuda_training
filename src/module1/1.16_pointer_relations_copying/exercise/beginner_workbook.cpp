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
// BEGINNER WORKBOOK: Pointer Relations & Copying
//
// Module: 1.16 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Beginner
//
// Focus: Safe overlapping memory copy (memmove semantics), typed vs byte
//        pointer distance arithmetic, and strided signal downsampling copy.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.16_beginner
//   ../../../output/1.16_beginner
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
    std::cout << "--- WORKBOOK: Pointer Relations & Copying (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Safe In-Place Overlapping Memory Shifter (`memmove` semantics)
    //
    // Context: In sequence models and ring buffers, shifting token tensors in-place
    //          causes source and destination ranges to overlap. A naive copy loop
    //          will overwrite elements before they are read.
    //
    // Task: Implement `safeMemMoveFloats(float* dst, const float* src, size_t count)`:
    //       - If dst == src or count == 0, return immediately.
    //       - If dst < src: copy forward from index 0 up to count - 1.
    //       - If dst > src: copy backward from index count - 1 down to 0.
    // -------------------------------------------------------------------------
    {
        auto safeMemMoveFloats = [](float* dst, const float* src, size_t count) {
            // --- YOUR CODE STARTS HERE ---
            if (dst == src || count == 0) return;
            if (dst < src) {
                for (size_t i = 0; i < count; ++i) {
                    dst[i] = src[i];
                }
            } else {
                for (size_t i = count; i > 0; --i) {
                    dst[i - 1] = src[i - 1];
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t N = 16;
        std::vector<float> buf_right(N);
        for (size_t i = 0; i < N; ++i) buf_right[i] = static_cast<float>(i + 1);

        // Shift right by 4 positions: elements 0..7 -> 4..11
        // dst = &buf[4], src = &buf[0], count = 8
        safeMemMoveFloats(buf_right.data() + 4, buf_right.data(), 8);

        bool ok = true;
        for (size_t i = 0; i < 8; ++i) {
            if (buf_right[4 + i] != static_cast<float>(i + 1)) ok = false;
        }

        // Shift left by 4 positions: elements 4..11 -> 0..7
        std::vector<float> buf_left = buf_right;
        safeMemMoveFloats(buf_left.data(), buf_left.data() + 4, 8);
        for (size_t i = 0; i < 8; ++i) {
            if (buf_left[i] != static_cast<float>(i + 1)) ok = false;
        }

        reportStatus("Problem 1: In-Place Overlapping Memory Copy (memmove)", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Typed vs Byte Pointer Distance Arithmetic
    //
    // Context: In low-level tensor allocators, computing offsets requires converting
    //          between typed pointer differences (in elements) and byte differences (in bytes).
    //
    // Task: Given struct `TensorDescriptor`:
    //       Implement:
    //         1. `elementDist(const TensorDescriptor* a, const TensorDescriptor* b)`: returns b - a.
    //         2. `byteDist(const TensorDescriptor* a, const TensorDescriptor* b)`:
    //            returns byte difference: reinterpret_cast<const char*>(b) - reinterpret_cast<const char*>(a).
    //       Verify that byteDist == elementDist * sizeof(TensorDescriptor).
    // -------------------------------------------------------------------------
    {
        struct TensorDescriptor {
            int id;
            float shape[4];
            double timestamp;
        };

        auto elementDist = [](const TensorDescriptor* a, const TensorDescriptor* b) -> int64_t {
            // --- YOUR CODE STARTS HERE ---
            return b - a;
            // --- YOUR CODE ENDS HERE ---
        };

        auto byteDist = [](const TensorDescriptor* a, const TensorDescriptor* b) -> int64_t {
            // --- YOUR CODE STARTS HERE ---
            return reinterpret_cast<const char*>(b) - reinterpret_cast<const char*>(a);
            // --- YOUR CODE ENDS HERE ---
        };

        std::vector<TensorDescriptor> descriptors(32);
        bool ok = true;

        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                int64_t e_diff = elementDist(&descriptors[i], &descriptors[j]);
                int64_t b_diff = byteDist(&descriptors[i], &descriptors[j]);

                if (e_diff != (j - i)) ok = false;
                if (b_diff != e_diff * static_cast<int64_t>(sizeof(TensorDescriptor))) ok = false;
            }
        }

        reportStatus("Problem 2: Typed vs Byte Pointer Distance", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Strided Signal Downsampling Copy
    //
    // Context: Audio, sensor, and sequence processing downsample signals by
    //          skipping elements with stride S.
    //
    // Task: Implement `stridedGatherCopy(const float* src, float* dst, size_t count, size_t stride)`:
    //       Copies `count` samples: dst[i] = src[i * stride].
    // -------------------------------------------------------------------------
    {
        auto stridedGatherCopy = [](const float* src, float* dst, size_t count, size_t stride) {
            // --- YOUR CODE STARTS HERE ---
            for (size_t i = 0; i < count; ++i) {
                dst[i] = src[i * stride];
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t NUM_SAMPLES = 100;
        const size_t STRIDE = 4;
        std::vector<float> input(NUM_SAMPLES * STRIDE);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i * 0.5f);

        std::vector<float> output(NUM_SAMPLES, 0.0f);
        stridedGatherCopy(input.data(), output.data(), NUM_SAMPLES, STRIDE);

        bool ok = true;
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            if (output[i] != input[i * STRIDE]) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 3: Strided Signal Downsampling Copy", ok);
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
