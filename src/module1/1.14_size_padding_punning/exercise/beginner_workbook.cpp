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
#include <cstddef>

// =========================================================================
// BEGINNER WORKBOOK: Size, Padding & Type Punning
//
// Module: 1.14 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Beginner
//
// Focus: IEEE-754 bit-level deconstruction via bit_cast, struct padding
//        optimization with offsetof, and FP32-to-BF16 truncation.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.14_beginner
//   ../../../output/1.14_beginner
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
    std::cout << "--- WORKBOOK: Size, Padding & Type Punning (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: IEEE 754 Single-Precision Float Bit Deconstruction
    //
    // Context: In low-precision quantization and custom GPU bitwise kernels,
    //          engineers must inspect and manipulate individual sign, exponent,
    //          and mantissa bitfields without invoking undefined behavior.
    //
    // Task: Implement `extractFP32(float val, uint32_t& sign, int32_t& biased_exp, uint32_t& mantissa)`:
    //       - Use `std::bit_cast<uint32_t>(val)` to inspect the 32 bits safely.
    //       - sign: bit 31 (0 or 1).
    //       - biased_exp: bits 23..30 (0 to 255).
    //       - mantissa: bits 0..22 (23-bit fractional payload).
    //       Implement `assembleFP32(uint32_t sign, uint32_t biased_exp, uint32_t mantissa)`:
    //       - Recombines the bitfields and returns the reconstituted `float`.
    // -------------------------------------------------------------------------
    {
        auto extractFP32 = [](float val, uint32_t& sign, uint32_t& biased_exp, uint32_t& mantissa) {
            // --- YOUR CODE STARTS HERE ---
            uint32_t bits = std::bit_cast<uint32_t>(val);
            sign = (bits >> 31) & 0x1;
            biased_exp = (bits >> 23) & 0xFF;
            mantissa = bits & 0x7FFFFF;
            // --- YOUR CODE ENDS HERE ---
        };

        auto assembleFP32 = [](uint32_t sign, uint32_t biased_exp, uint32_t mantissa) -> float {
            // --- YOUR CODE STARTS HERE ---
            uint32_t bits = ((sign & 0x1) << 31) |
                            ((biased_exp & 0xFF) << 23) |
                            (mantissa & 0x7FFFFF);
            return std::bit_cast<float>(bits);
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        std::vector<float> test_values = {1.0f, -1.0f, 2.5f, -0.15625f, 1024.0f, 0.00390625f};

        for (float val : test_values) {
            uint32_t s = 0, exp = 0, mant = 0;
            extractFP32(val, s, exp, mant);
            float reconstructed = assembleFP32(s, exp, mant);
            if (reconstructed != val) {
                ok = false;
                break;
            }
        }

        // Test 1.0f explicitly: sign=0, biased_exp=127, mantissa=0
        uint32_t s = 0, exp = 0, mant = 0;
        extractFP32(1.0f, s, exp, mant);
        if (s != 0 || exp != 127 || mant != 0) ok = false;

        // Test -2.0f explicitly: sign=1, biased_exp=128, mantissa=0
        extractFP32(-2.0f, s, exp, mant);
        if (s != 1 || exp != 128 || mant != 0) ok = false;

        reportStatus("Problem 1: IEEE-754 Bit Deconstruction & Assembly", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Struct Padding Optimization & Memory Footprint Analyzer
    //
    // Context: In GPU streaming pipelines, transmitting millions of metadata
    //          structs with poor member ordering wastes significant PCIe bandwidth
    //          and cache capacity due to alignment padding holes.
    //
    // Task:
    //   1. Compare `UnoptimizedBatch` vs `OptimizedBatch`.
    //   2. Compute padding bytes in each struct: `sizeof(T) - sum_of_member_sizes`.
    //   3. Implement `serializePacked(const UnoptimizedBatch* in, size_t n, uint8_t* out_bytes)`:
    //      Packs all members contiguously without any padding bytes:
    //      [flag (1B) | lr (8B) | id (2B) | weight (4B)] = 15 bytes per record.
    // -------------------------------------------------------------------------
    {
        struct UnoptimizedBatch {
            uint8_t flag;      // 1 byte + 7 padding
            double lr;         // 8 bytes
            uint16_t id;       // 2 bytes + 2 padding
            float weight;      // 4 bytes
        }; // Total: 24 bytes

        struct OptimizedBatch {
            double lr;         // 8 bytes
            float weight;      // 4 bytes
            uint16_t id;       // 2 bytes
            uint8_t flag;      // 1 byte + 1 padding
        }; // Total: 16 bytes

        auto serializePacked = [](const UnoptimizedBatch* in, size_t n, uint8_t* out) -> size_t {
            // --- YOUR CODE STARTS HERE ---
            size_t written = 0;
            for (size_t i = 0; i < n; ++i) {
                std::memcpy(out + written, &in[i].flag, sizeof(uint8_t));
                written += sizeof(uint8_t);
                std::memcpy(out + written, &in[i].lr, sizeof(double));
                written += sizeof(double);
                std::memcpy(out + written, &in[i].id, sizeof(uint16_t));
                written += sizeof(uint16_t);
                std::memcpy(out + written, &in[i].weight, sizeof(float));
                written += sizeof(float);
            }
            return written;
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        size_t unopt_size = sizeof(UnoptimizedBatch);
        size_t opt_size = sizeof(OptimizedBatch);
        size_t raw_payload_size = sizeof(uint8_t) + sizeof(double) + sizeof(uint16_t) + sizeof(float); // 15 bytes

        if (unopt_size != 24 || opt_size != 16 || raw_payload_size != 15) ok = false;

        const size_t N = 100;
        std::vector<UnoptimizedBatch> batch(N);
        for (size_t i = 0; i < N; ++i) {
            batch[i] = {static_cast<uint8_t>(i % 255), 0.001 * (i + 1),
                        static_cast<uint16_t>(i), static_cast<float>(i * 1.25f)};
        }

        std::vector<uint8_t> packed_stream(N * raw_payload_size);
        size_t bytes_written = serializePacked(batch.data(), N, packed_stream.data());

        if (bytes_written != N * 15) ok = false;

        // Verify deserialized contents
        for (size_t i = 0; i < N; ++i) {
            const uint8_t* record = packed_stream.data() + i * 15;
            uint8_t f; double lr; uint16_t id; float w;
            std::memcpy(&f, record, 1);
            std::memcpy(&lr, record + 1, 8);
            std::memcpy(&id, record + 9, 2);
            std::memcpy(&w, record + 11, 4);

            if (f != batch[i].flag || lr != batch[i].lr || id != batch[i].id || w != batch[i].weight) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 2: Struct Padding & Contiguous Serialization", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: FP32 to BF16 (Bfloat16) Bitcast & Truncation
    //
    // Context: Google Brain Floating Point (BF16) format is widely used in TPU/GPU
    //          training. It retains the 8-bit dynamic range of FP32 but drops the
    //          least significant 16 bits of the mantissa.
    //
    // Task: Implement `float32_to_bfloat16(float val)`:
    //       - Extracts the top 16 bits of FP32 (with optional nearest-even rounding).
    //         For truncation: (bit_cast<uint32_t>(val) >> 16).
    //       Implement `bfloat16_to_float32(uint16_t bf)`:
    //       - Shifts bf left by 16 bits and bitcasts back to `float`.
    // -------------------------------------------------------------------------
    {
        auto float32_to_bfloat16 = [](float val) -> uint16_t {
            // --- YOUR CODE STARTS HERE ---
            uint32_t bits = std::bit_cast<uint32_t>(val);
            // Nearest rounding with tie breaking
            uint32_t lsb = (bits >> 16) & 1;
            uint32_t rounding_bias = 0x7FFF + lsb;
            bits += rounding_bias;
            return static_cast<uint16_t>(bits >> 16);
            // --- YOUR CODE ENDS HERE ---
        };

        auto bfloat16_to_float32 = [](uint16_t bf) -> float {
            // --- YOUR CODE STARTS HERE ---
            uint32_t bits = static_cast<uint32_t>(bf) << 16;
            return std::bit_cast<float>(bits);
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        std::vector<float> values = {0.0f, 1.0f, -1.0f, 4.0f, 65536.0f, 0.125f};

        for (float val : values) {
            uint16_t bf = float32_to_bfloat16(val);
            float restored = bfloat16_to_float32(bf);
            if (restored != val) {
                ok = false;
                break;
            }
        }

        // Test an approximate value (0.1f cannot be represented exactly in binary)
        float approx = 0.1f;
        uint16_t bf_approx = float32_to_bfloat16(approx);
        float restored_approx = bfloat16_to_float32(bf_approx);
        // Relative error must be within 1% (BF16 has ~2-3 decimal digits of precision)
        if (std::abs(restored_approx - approx) / approx > 0.01f) ok = false;

        reportStatus("Problem 3: FP32 <-> BF16 Bitcast Conversion", ok);
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
