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
// CHAMPION WORKBOOK: Size, Padding & Type Punning
//
// Module: 1.14 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Champion
//
// Focus: Fast inverse square root via standard-compliant bit_cast, NVIDIA Hopper
//        FP8 E4M3 quantization, and unaligned 128-bit vector load emulation.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.14_champion
//   ../../../output/1.14_champion
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
    std::cout << "--- WORKBOOK: Size, Padding & Type Punning (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Modern Fast Inverse Square Root (1/sqrt(x)) via bit_cast
    //
    // Context: In physics simulations and graphics, normalizing vectors requires
    //          calculating 1 / sqrt(x). The classic Carmack fast inverse square root
    //          computes an initial guess using integer bit-punning, followed by
    //          Newton-Raphson refinement steps. In C++20, this is performed
    //          without undefined behavior using `std::bit_cast`.
    //
    // Task: Implement `fastInvSqrt(float number)`:
    //       - const float xhalf = 0.5f * number;
    //       - uint32_t i = std::bit_cast<uint32_t>(number);
    //       - i = 0x5f3759df - (i >> 1);
    //       - float y = std::bit_cast<float>(i);
    //       - Perform 2 Newton-Raphson steps:
    //           y = y * (1.5f - xhalf * y * y);
    //           y = y * (1.5f - xhalf * y * y);
    //       - Returns y.
    // -------------------------------------------------------------------------
    {
        auto fastInvSqrt = [](float number) -> float {
            // --- YOUR CODE STARTS HERE ---
            const float xhalf = 0.5f * number;
            uint32_t i = std::bit_cast<uint32_t>(number);
            i = 0x5f3759df - (i >> 1);
            float y = std::bit_cast<float>(i);
            y = y * (1.5f - xhalf * y * y); // 1st iteration
            y = y * (1.5f - xhalf * y * y); // 2nd iteration
            return y;
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        std::vector<float> test_values = {1.0f, 4.0f, 16.0f, 0.25f, 100.0f, 0.001f, 12345.0f};

        for (float v : test_values) {
            float approx = fastInvSqrt(v);
            float exact = 1.0f / std::sqrt(v);
            float rel_error = std::abs(approx - exact) / exact;
            // 2 iterations of Newton-Raphson yield relative error < 0.01%
            if (rel_error > 0.0001f) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 1: Fast Inverse Square Root via bit_cast", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: NVIDIA Hopper FP8 E4M3 Quantization & Dequantization
    //
    // Context: Modern tensor cores (e.g. Hopper H100) support FP8 E4M3 for GEMM
    //          activations and weights.
    //          E4M3 format:
    //            - Sign: 1 bit (bit 7)
    //            - Exponent: 4 bits (bits 3..6, bias 7)
    //            - Mantissa: 3 bits (bits 0..2)
    //            - Max value: 448.0f
    //
    // Task: Implement `quantize_fp8_e4m3(float x)`:
    //       - Clamps x to [-448.0f, 448.0f].
    //       - Extracts sign, computes unbiased exponent, rebiases with bias 7,
    //         truncates 23-bit mantissa to 3 bits.
    //       Implement `dequantize_fp8_e4m3(uint8_t b)`:
    //       - Restores float from sign, 4-bit exp, 3-bit mantissa.
    // -------------------------------------------------------------------------
    {
        auto quantize_fp8_e4m3 = [](float x) -> uint8_t {
            // --- YOUR CODE STARTS HERE ---
            if (x == 0.0f) return 0;
            float clamped = std::clamp(x, -448.0f, 448.0f);
            uint32_t f_bits = std::bit_cast<uint32_t>(clamped);
            uint8_t sign = (f_bits >> 31) & 0x1;
            int32_t exp = static_cast<int32_t>((f_bits >> 23) & 0xFF) - 127;
            uint32_t mant = f_bits & 0x7FFFFF;

            int32_t fp8_exp = exp + 7;
            if (fp8_exp <= 0) return static_cast<uint8_t>(sign << 7); // Underflow to signed zero
            if (fp8_exp > 15) fp8_exp = 15;

            uint8_t fp8_mant = static_cast<uint8_t>(mant >> 20); // 23 bits -> top 3 bits
            return static_cast<uint8_t>((sign << 7) | (fp8_exp << 3) | (fp8_mant & 0x7));
            // --- YOUR CODE ENDS HERE ---
        };

        auto dequantize_fp8_e4m3 = [](uint8_t b) -> float {
            // --- YOUR CODE STARTS HERE ---
            if ((b & 0x7F) == 0) return 0.0f;
            uint32_t sign = (b >> 7) & 0x1;
            uint32_t fp8_exp = (b >> 3) & 0xF;
            uint32_t fp8_mant = b & 0x7;

            int32_t exp = static_cast<int32_t>(fp8_exp) - 7 + 127;
            uint32_t f_bits = (sign << 31) | (static_cast<uint32_t>(exp) << 23) | (fp8_mant << 20);
            return std::bit_cast<float>(f_bits);
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        std::vector<float> test_values = {1.0f, -1.0f, 2.0f, 0.5f, 4.0f, 16.0f, -32.0f, 0.0f};

        for (float val : test_values) {
            uint8_t q = quantize_fp8_e4m3(val);
            float restored = dequantize_fp8_e4m3(q);
            if (restored != val) {
                ok = false;
                break;
            }
        }

        // Verify 1.0f encoding:
        // sign=0, exp = 0 + 7 = 7 (0b0111), mant = 0 -> (0 << 7) | (7 << 3) | 0 = 0x38 (56)
        if (quantize_fp8_e4m3(1.0f) != 0x38) ok = false;
        // Verify -1.0f encoding: (1 << 7) | 0x38 = 0xB8
        if (quantize_fp8_e4m3(-1.0f) != 0xB8) ok = false;

        reportStatus("Problem 2: NVIDIA Hopper FP8 E4M3 Quantizer", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Unaligned 128-Bit Vector Load Emulation via Memory Splicing
    //
    // Context: High-performance memory engines require 128-bit vector alignment
    //          (16 bytes) for vector instructions. When reading from unaligned
    //          offsets in arbitrary byte buffers, data must be safely loaded
    //          and spliced without causing hardware bus faults.
    //
    // Task: Implement `loadUnalignedFloat4(const void* unaligned_ptr)`:
    //       - Loads 16 contiguous bytes from `unaligned_ptr` into a 16-byte
    //         aligned struct `alignas(16) struct Float4 { float x, y, z, w; };`
    //         using standard-compliant `std::memcpy`.
    //       - Verify that reading 1000 consecutive unaligned offsets (offset % 16 != 0)
    //         returns exact float4 elements identical to direct indexing.
    // -------------------------------------------------------------------------
    {
        struct alignas(16) Float4 {
            float x, y, z, w;
        };

        auto loadUnalignedFloat4 = [](const void* unaligned_ptr) -> Float4 {
            // --- YOUR CODE STARTS HERE ---
            Float4 result;
            std::memcpy(&result, unaligned_ptr, sizeof(Float4));
            return result;
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t NUM_FLOATS = 1024;
        std::vector<float> source_data(NUM_FLOATS);
        for (size_t i = 0; i < NUM_FLOATS; ++i) {
            source_data[i] = static_cast<float>(i * 3.14159f);
        }

        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(source_data.data());
        bool ok = true;

        // Test at arbitrary byte offsets
        for (size_t byte_offset = 1; byte_offset < (NUM_FLOATS - 4) * sizeof(float); byte_offset += 7) {
            Float4 loaded = loadUnalignedFloat4(byte_ptr + byte_offset);

            // Compare bit-for-bit against expected memcpy
            Float4 expected;
            std::memcpy(&expected, byte_ptr + byte_offset, sizeof(Float4));

            if (std::memcmp(&loaded, &expected, sizeof(Float4)) != 0) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 3: Unaligned 128-Bit Vector Memory Splicing", ok);
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
