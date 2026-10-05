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
// INTERMEDIATE WORKBOOK: Pointer Relations & Copying
//
// Module: 1.16 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Intermediate
//
// Focus: Pitched 2D submatrix copy (cudaMemcpy2D emulation), sliding-window
//        KV cache in-place roll, and dual-channel planar-to-interleaved copy.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.16_intermediate
//   ../../../output/1.16_intermediate
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
    std::cout << "--- WORKBOOK: Pointer Relations & Copying (Intermediate) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Pitched 2D Subgrid Memory Copy (`cudaMemcpy2D` emulation)
    //
    // Context: In GPU programming, `cudaMemcpy2D` copies rectangular regions between
    //          matrices with different row pitch (stride in bytes) without copying
    //          unwanted row padding.
    //
    // Task: Implement `copy2DStrided(void* dst, size_t dst_pitch,
    //                                const void* src, size_t src_pitch,
    //                                size_t width_bytes, size_t height_rows)`:
    //       For row r in [0, height_rows):
    //         const uint8_t* s_row = reinterpret_cast<const uint8_t*>(src) + r * src_pitch;
    //         uint8_t* d_row       = reinterpret_cast<uint8_t*>(dst) + r * dst_pitch;
    //         std::memcpy(d_row, s_row, width_bytes);
    // -------------------------------------------------------------------------
    {
        auto copy2DStrided = [](void* dst, size_t dst_pitch,
                                const void* src, size_t src_pitch,
                                size_t width_bytes, size_t height_rows) {
            // --- YOUR CODE STARTS HERE ---
            for (size_t r = 0; r < height_rows; ++r) {
                const uint8_t* s_row = reinterpret_cast<const uint8_t*>(src) + (r * src_pitch);
                uint8_t* d_row       = reinterpret_cast<uint8_t*>(dst) + (r * dst_pitch);
                std::memcpy(d_row, s_row, width_bytes);
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t SRC_H = 32, SRC_W = 64;
        const size_t SRC_PITCH = SRC_W * sizeof(float) + 64; // with padding
        std::vector<uint8_t> src_buf(SRC_H * SRC_PITCH, 0);

        // Fill source data
        for (size_t r = 0; r < SRC_H; ++r) {
            float* row = reinterpret_cast<float*>(src_buf.data() + r * SRC_PITCH);
            for (size_t c = 0; c < SRC_W; ++c) {
                row[c] = static_cast<float>(r * 1000 + c);
            }
        }

        // Subgrid of 16 rows x 32 columns
        const size_t SUB_H = 16, SUB_W = 32;
        const size_t DST_PITCH = SUB_W * sizeof(float) + 32; // distinct pitch
        std::vector<uint8_t> dst_buf(SUB_H * DST_PITCH, 0);

        // Copy top-left 16x32 region
        copy2DStrided(dst_buf.data(), DST_PITCH, src_buf.data(), SRC_PITCH,
                      SUB_W * sizeof(float), SUB_H);

        bool ok = true;
        for (size_t r = 0; r < SUB_H; ++r) {
            const float* d_row = reinterpret_cast<const float*>(dst_buf.data() + r * DST_PITCH);
            for (size_t c = 0; c < SUB_W; ++c) {
                float expected = static_cast<float>(r * 1000 + c);
                if (d_row[c] != expected) {
                    ok = false;
                    break;
                }
            }
        }

        reportStatus("Problem 1: Pitched 2D Subgrid Memory Copy", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Sliding-Window KV Cache In-Place Roll
    //
    // Context: In transformer KV cache management (e.g. streaming LLMs), when
    //          the cache reaches capacity M tokens, the oldest K tokens are
    //          evicted by sliding tokens [K, M) left into positions [0, M - K) in-place.
    //
    // Task: Implement `rollKVCacheLeft(float* kv_buf, size_t M, size_t D, size_t K)`:
    //       - Moves (M - K) * D floats from `kv_buf + K * D` to `kv_buf`.
    //       - Zeroes out the vacated tail region of K * D floats.
    // -------------------------------------------------------------------------
    {
        auto rollKVCacheLeft = [](float* kv_buf, size_t M, size_t D, size_t K) {
            // --- YOUR CODE STARTS HERE ---
            if (K >= M) {
                std::memset(kv_buf, 0, M * D * sizeof(float));
                return;
            }
            size_t move_count = (M - K) * D;
            std::memmove(kv_buf, kv_buf + K * D, move_count * sizeof(float));
            std::memset(kv_buf + move_count, 0, K * D * sizeof(float));
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t M = 8;  // max tokens
        const size_t D = 4;  // hidden dim
        const size_t K = 2;  // evict oldest 2 tokens
        std::vector<float> cache(M * D);

        for (size_t t = 0; t < M; ++t) {
            for (size_t d = 0; d < D; ++d) {
                cache[t * D + d] = static_cast<float>((t + 1) * 10 + d);
            }
        }

        rollKVCacheLeft(cache.data(), M, D, K);

        bool ok = true;
        // Tokens originally at index 2..7 should now be at index 0..5
        for (size_t t = 0; t < M - K; ++t) {
            size_t orig_t = t + K;
            for (size_t d = 0; d < D; ++d) {
                float expected = static_cast<float>((orig_t + 1) * 10 + d);
                if (cache[t * D + d] != expected) ok = false;
            }
        }
        // Vacated tail tokens (index 6 and 7) must be 0.0f
        for (size_t t = M - K; t < M; ++t) {
            for (size_t d = 0; d < D; ++d) {
                if (cache[t * D + d] != 0.0f) ok = false;
            }
        }

        reportStatus("Problem 2: Sliding-Window KV Cache In-Place Roll", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Planar to Interleaved Dual-Channel Copy
    //
    // Context: In audio stereo processing and complex-valued neural networks
    //          (e.g., FFT real and imaginary components), data frequently arrives
    //          in planar format [Channel 0, Channel 1] and must be converted
    //          to interleaved format [L0, R0, L1, R1, ...].
    //
    // Task: Implement `interleaveStereo(const float* chA, const float* chB, float* out, size_t N)`:
    //       out[2 * i]     = chA[i];
    //       out[2 * i + 1] = chB[i];
    // -------------------------------------------------------------------------
    {
        auto interleaveStereo = [](const float* chA, const float* chB, float* out, size_t N) {
            // --- YOUR CODE STARTS HERE ---
            for (size_t i = 0; i < N; ++i) {
                out[2 * i]     = chA[i];
                out[2 * i + 1] = chB[i];
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t N = 512;
        std::vector<float> chA(N), chB(N);
        for (size_t i = 0; i < N; ++i) {
            chA[i] = static_cast<float>(i);
            chB[i] = static_cast<float>(i * 100);
        }

        std::vector<float> interleaved(2 * N, 0.0f);
        interleaveStereo(chA.data(), chB.data(), interleaved.data(), N);

        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (interleaved[2 * i] != chA[i] || interleaved[2 * i + 1] != chB[i]) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 3: Planar to Interleaved Channel Copy", ok);
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
