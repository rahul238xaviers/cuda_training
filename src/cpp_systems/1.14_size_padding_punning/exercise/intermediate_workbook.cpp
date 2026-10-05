#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <bit>

// =========================================================================
// INTERMEDIATE WORKBOOK: Size, Padding & Type Punning
//
// Module: 1.14 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Intermediate
//
// Focus: INT4 nibble packing with two's complement sign extension, FP16 half
//        precision conversion, and 64-byte cache-line false-sharing isolation.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.14_intermediate
//   ../../../output/1.14_intermediate
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
    std::cout << "--- WORKBOOK: Size, Padding & Type Punning (Intermediate) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: INT4 (4-bit signed integer) Nibble Packing and Unpacking Engine
    //
    // Context: In low-bit LLM weight quantization (GPTQ, AWQ, BitBLAS), model
    //          weights are quantized to signed 4-bit integers [-8, +7]. Two
    //          weights are packed into each byte. Proper sign extension when
    //          unpacking is required to restore negative values.
    //
    // Task: Implement `packINT4(const int8_t* in, size_t N, uint8_t* out_packed)`:
    //       - Stores in[2*i] in bits 0..3 (low nibble) and in[2*i+1] in bits 4..7 (high nibble).
    //       Implement `unpackINT4(const uint8_t* packed, size_t N, int8_t* out)`:
    //       - Extracts low and high nibbles.
    //       - CRITICAL: Sign-extends 4-bit two's complement: if bit 3 is 1, bits 4..7 must be 1.
    // -------------------------------------------------------------------------
    {
        auto packINT4 = [](const int8_t* in, size_t N, uint8_t* out_packed) {
            // --- YOUR CODE STARTS HERE ---
            for (size_t i = 0; i < N / 2; ++i) {
                uint8_t low = static_cast<uint8_t>(in[2 * i]) & 0x0F;
                uint8_t high = static_cast<uint8_t>(in[2 * i + 1]) & 0x0F;
                out_packed[i] = low | (high << 4);
            }
            // --- YOUR CODE ENDS HERE ---
        };

        auto unpackINT4 = [](const uint8_t* packed, size_t N, int8_t* out) {
            // --- YOUR CODE STARTS HERE ---
            auto sign_extend_4bit = [](uint8_t nibble) -> int8_t {
                // If bit 3 is set (negative in 4-bit signed), sign-extend to 8 bits
                if (nibble & 0x08) {
                    return static_cast<int8_t>(nibble | 0xF0);
                } else {
                    return static_cast<int8_t>(nibble & 0x0F);
                }
            };

            for (size_t i = 0; i < N / 2; ++i) {
                uint8_t byte = packed[i];
                out[2 * i]     = sign_extend_4bit(byte & 0x0F);
                out[2 * i + 1] = sign_extend_4bit((byte >> 4) & 0x0F);
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t N = 1024;
        std::vector<int8_t> original(N);
        // Fill with valid 4-bit signed integers [-8, 7]
        for (size_t i = 0; i < N; ++i) {
            original[i] = static_cast<int8_t>((static_cast<int>(i % 16)) - 8);
        }

        std::vector<uint8_t> packed(N / 2);
        packINT4(original.data(), N, packed.data());

        std::vector<int8_t> unpacked(N, 0);
        unpackINT4(packed.data(), N, unpacked.data());

        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (unpacked[i] != original[i]) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 1: INT4 Nibble Packing & Sign Extension", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: IEEE 754 FP16 (Half Precision) Bit Conversion
    //
    // Context: Standard FP16 (IEEE 754-2008) is the workhorse precision for
    //          GPU tensor operations. It has 1 sign bit, 5 exponent bits
    //          (bias 15), and 10 mantissa bits.
    //
    // Task: Implement `float32_to_fp16(float val)` returning `uint16_t`:
    //       - Extract FP32 sign (1 bit), exp (8 bits, bias 127), mant (23 bits).
    //       - Unbias FP32 exp: true_exp = exp - 127.
    //       - Rebias for FP16: fp16_exp = true_exp + 15.
    //       - Truncate mantissa from 23 bits down to 10 bits.
    //       - Combine: (sign << 15) | (fp16_exp << 10) | (fp16_mant).
    //       Implement `fp16_to_float32(uint16_t h)` returning `float`:
    //       - Inverse operation.
    //       (Assume normalized non-zero inputs for simplicity).
    // -------------------------------------------------------------------------
    {
        auto float32_to_fp16 = [](float val) -> uint16_t {
            // --- YOUR CODE STARTS HERE ---
            if (val == 0.0f) return 0;
            uint32_t f_bits = std::bit_cast<uint32_t>(val);
            uint32_t sign = (f_bits >> 31) & 0x1;
            int32_t exp = static_cast<int32_t>((f_bits >> 23) & 0xFF) - 127;
            uint32_t mant = f_bits & 0x7FFFFF;

            int32_t fp16_exp = exp + 15;
            if (fp16_exp <= 0) return static_cast<uint16_t>(sign << 15); // underflow to 0
            if (fp16_exp >= 31) return static_cast<uint16_t>((sign << 15) | (0x1F << 10)); // overflow to inf

            uint16_t fp16_mant = static_cast<uint16_t>(mant >> 13);
            return static_cast<uint16_t>((sign << 15) | (static_cast<uint16_t>(fp16_exp) << 10) | fp16_mant);
            // --- YOUR CODE ENDS HERE ---
        };

        auto fp16_to_float32 = [](uint16_t h) -> float {
            // --- YOUR CODE STARTS HERE ---
            if (h == 0) return 0.0f;
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t fp16_exp = (h >> 10) & 0x1F;
            uint32_t fp16_mant = h & 0x3FF;

            if (fp16_exp == 0) return 0.0f; // Subnormals treated as 0 for simplicity

            int32_t exp = static_cast<int32_t>(fp16_exp) - 15 + 127;
            uint32_t f_bits = (sign << 31) | (static_cast<uint32_t>(exp) << 23) | (fp16_mant << 13);
            return std::bit_cast<float>(f_bits);
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        std::vector<float> test_vals = {1.0f, -1.0f, 2.0f, 0.5f, 16.0f, -8.0f, 0.25f};

        for (float v : test_vals) {
            uint16_t h = float32_to_fp16(v);
            float restored = fp16_to_float32(h);
            if (restored != v) {
                ok = false;
                break;
            }
        }

        // Test 1.0f bit representation:
        // sign=0, fp16_exp = 0 + 15 = 15 (0b01111), mant = 0 -> 0x3C00
        if (float32_to_fp16(1.0f) != 0x3C00) ok = false;
        // Test 2.0f bit representation:
        // sign=0, fp16_exp = 1 + 15 = 16 (0b10000), mant = 0 -> 0x4000
        if (float32_to_fp16(2.0f) != 0x4000) ok = false;

        reportStatus("Problem 2: IEEE 754 FP16 Bit Construction", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Struct Cache-Line False Sharing & Alignment Verification
    //
    // Context: In high-throughput GPU host drivers and multi-threaded C++ runtimes,
    //          when multiple CPU threads frequently write to variables located
    //          on the same 64-byte cache line, performance degrades drastically
    //          due to cache line bouncing (false sharing).
    //
    // Task: Implement `checkCacheLineIsolation(const void* ptrA, const void* ptrB)`:
    //       - Returns true if ptrA and ptrB reside on DIFFERENT 64-byte cache lines:
    //         i.e., (reinterpret_cast<uintptr_t>(ptrA) / 64) != (reinterpret_cast<uintptr_t>(ptrB) / 64).
    //       Verify that `alignas(64)` successfully separates adjacent counter fields.
    // -------------------------------------------------------------------------
    {
        struct BadCounters {
            uint64_t worker0_count;
            uint64_t worker1_count;
        };

        struct AlignedCounters {
            alignas(64) uint64_t worker0_count;
            alignas(64) uint64_t worker1_count;
        };

        auto checkCacheLineIsolation = [](const void* ptrA, const void* ptrB) -> bool {
            // --- YOUR CODE STARTS HERE ---
            uintptr_t lineA = reinterpret_cast<uintptr_t>(ptrA) / 64;
            uintptr_t lineB = reinterpret_cast<uintptr_t>(ptrB) / 64;
            return lineA != lineB;
            // --- YOUR CODE ENDS HERE ---
        };

        BadCounters bad;
        AlignedCounters aligned;

        bool bad_isolated = checkCacheLineIsolation(&bad.worker0_count, &bad.worker1_count);
        bool aligned_isolated = checkCacheLineIsolation(&aligned.worker0_count, &aligned.worker1_count);

        bool ok = (!bad_isolated && aligned_isolated);
        if (sizeof(AlignedCounters) < 128) ok = false;

        reportStatus("Problem 3: Cache-Line False Sharing Isolation", ok);
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
