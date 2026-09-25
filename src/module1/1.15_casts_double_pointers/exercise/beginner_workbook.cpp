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
// BEGINNER WORKBOOK: Type Casts & Double Pointers
//
// Module: 1.15 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Beginner
//
// Focus: CUDA-style double pointer output allocation, raw byte stream
//        deserialization via reinterpret_cast, and ragged float** batch padding.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.15_beginner
//   ../../../output/1.15_beginner
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
    std::cout << "--- WORKBOOK: Type Casts & Double Pointers (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: CUDA-Style Double Pointer Allocation Simulator (`cudaMalloc` emulation)
    //
    // Context: In CUDA, functions like `cudaMalloc(void** devPtr, size_t size)`
    //          take a pointer-to-pointer because the caller passes a pointer
    //          variable by address for the runtime to overwrite.
    //
    // Task: Implement `hostMalloc(void** ptr, size_t bytes)`:
    //       - If `ptr == nullptr` or `bytes == 0`, return 1 (ERROR_INVALID_ARGUMENT).
    //       - Allocate 64-byte aligned memory: `void* mem = nullptr; posix_memalign(&mem, 64, bytes);`
    //       - If allocation fails, return 2 (ERROR_OUT_OF_MEMORY).
    //       - Assign `*ptr = mem;` and return 0 (SUCCESS).
    //       Implement `hostFree(void** ptr)`:
    //       - If `ptr == nullptr` or `*ptr == nullptr`, return;
    //       - Free `*ptr;` and set `*ptr = nullptr;`.
    // -------------------------------------------------------------------------
    {
        auto hostMalloc = [](void** ptr, size_t bytes) -> int {
            // --- YOUR CODE STARTS HERE ---
            if (!ptr || bytes == 0) return 1;
            void* mem = nullptr;
            int err = posix_memalign(&mem, 64, bytes);
            if (err != 0 || !mem) return 2;
            *ptr = mem;
            return 0;
            // --- YOUR CODE ENDS HERE ---
        };

        auto hostFree = [](void** ptr) {
            // --- YOUR CODE STARTS HERE ---
            if (!ptr || !*ptr) return;
            free(*ptr);
            *ptr = nullptr;
            // --- YOUR CODE ENDS HERE ---
        };

        bool ok = true;
        float* d_tensor = nullptr;

        // Valid allocation
        int res = hostMalloc(reinterpret_cast<void**>(&d_tensor), 1024 * sizeof(float));
        if (res != 0 || d_tensor == nullptr) ok = false;
        if (reinterpret_cast<uintptr_t>(d_tensor) % 64 != 0) ok = false;

        // Write and read
        for (int i = 0; i < 1024; ++i) d_tensor[i] = static_cast<float>(i * 2);
        for (int i = 0; i < 1024; ++i) {
            if (d_tensor[i] != static_cast<float>(i * 2)) {
                ok = false;
                break;
            }
        }

        hostFree(reinterpret_cast<void**>(&d_tensor));
        if (d_tensor != nullptr) ok = false;

        // Null pointer safety test
        if (hostMalloc(nullptr, 1024) != 1) ok = false;
        if (hostMalloc(reinterpret_cast<void**>(&d_tensor), 0) != 1) ok = false;

        reportStatus("Problem 1: CUDA-Style Double Pointer Allocation", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Raw Binary Blob Deserialization via reinterpret_cast
    //
    // Context: Loading serialized weights (Safetensors / GGUF) involves parsing
    //          a raw uint8_t byte stream where headers precede structured float/int arrays.
    //
    // Task: Implement `parseModelPayload(const uint8_t* blob,
    //                                   uint32_t& num_layers, uint32_t& hidden_dim,
    //                                   const float*& weights, const int32_t*& vocab_ids)`:
    //       Stream format:
    //         Offset 0: uint32_t num_layers
    //         Offset 4: uint32_t hidden_dim
    //         Offset 8: (hidden_dim * hidden_dim) floats (weights)
    //         Offset 8 + hidden_dim * hidden_dim * 4: 128 int32s (vocab_ids)
    // -------------------------------------------------------------------------
    {
        auto parseModelPayload = [](const uint8_t* blob,
                                    uint32_t& num_layers, uint32_t& hidden_dim,
                                    const float*& weights, const int32_t*& vocab_ids) {
            // --- YOUR CODE STARTS HERE ---
            num_layers = *reinterpret_cast<const uint32_t*>(blob);
            hidden_dim = *reinterpret_cast<const uint32_t*>(blob + 4);
            weights    = reinterpret_cast<const float*>(blob + 8);
            size_t weight_bytes = static_cast<size_t>(hidden_dim) * hidden_dim * sizeof(float);
            vocab_ids  = reinterpret_cast<const int32_t*>(blob + 8 + weight_bytes);
            // --- YOUR CODE ENDS HERE ---
        };

        const uint32_t L = 12;
        const uint32_t D = 16;
        size_t total_bytes = 8 + (D * D * sizeof(float)) + (128 * sizeof(int32_t));
        std::vector<uint8_t> blob(total_bytes, 0);

        *reinterpret_cast<uint32_t*>(blob.data()) = L;
        *reinterpret_cast<uint32_t*>(blob.data() + 4) = D;
        float* w_ptr = reinterpret_cast<float*>(blob.data() + 8);
        for (uint32_t i = 0; i < D * D; ++i) w_ptr[i] = static_cast<float>(i + 0.5f);
        int32_t* v_ptr = reinterpret_cast<int32_t*>(blob.data() + 8 + D * D * sizeof(float));
        for (int i = 0; i < 128; ++i) v_ptr[i] = i * 10;

        uint32_t out_L = 0, out_D = 0;
        const float* out_w = nullptr;
        const int32_t* out_v = nullptr;
        parseModelPayload(blob.data(), out_L, out_D, out_w, out_v);

        bool ok = true;
        if (out_L != L || out_D != D) ok = false;
        if (out_w != w_ptr || out_v != v_ptr) ok = false;
        if (out_w[10] != w_ptr[10] || out_v[50] != 500) ok = false;

        reportStatus("Problem 2: Binary Blob Deserialization via reinterpret_cast", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Ragged Pointer Array (float**) to Contiguous Dense Matrix
    //
    // Context: In transformer NLP models, batches arrive with variable sequence lengths
    //          stored as an array of float pointers: `float* batch[B]`. To feed into
    //          matrix multiplication (GEMM), we must pad each sequence to L_max and
    //          pack them into a single contiguous [B, L_max] tensor.
    //
    // Task: Implement `packRaggedBatch(const float** batch_ptrs, const int* seq_lens,
    //                                 int B, int L_max, float* dense_out)`:
    //       For batch b in [0, B):
    //         For pos l in [0, L_max):
    //           If l < seq_lens[b], dense_out[b * L_max + l] = batch_ptrs[b][l];
    //           Else dense_out[b * L_max + l] = 0.0f;
    // -------------------------------------------------------------------------
    {
        auto packRaggedBatch = [](const float** batch_ptrs, const int* seq_lens,
                                  int B, int L_max, float* dense_out) {
            // --- YOUR CODE STARTS HERE ---
            for (int b = 0; b < B; ++b) {
                int len = seq_lens[b];
                const float* seq = batch_ptrs[b];
                for (int l = 0; l < L_max; ++l) {
                    dense_out[b * L_max + l] = (l < len) ? seq[l] : 0.0f;
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const int B = 4;
        const int L_max = 8;
        std::vector<int> seq_lens = {3, 8, 5, 2};

        std::vector<float> seq0 = {1.0f, 2.0f, 3.0f};
        std::vector<float> seq1 = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
        std::vector<float> seq2 = {20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
        std::vector<float> seq3 = {30.0f, 31.0f};

        const float* batch_ptrs[B] = {seq0.data(), seq1.data(), seq2.data(), seq3.data()};
        std::vector<float> dense_out(B * L_max, -1.0f);

        packRaggedBatch(batch_ptrs, seq_lens.data(), B, L_max, dense_out.data());

        bool ok = true;
        for (int b = 0; b < B; ++b) {
            int len = seq_lens[b];
            for (int l = 0; l < L_max; ++l) {
                float val = dense_out[b * L_max + l];
                if (l < len) {
                    if (val != batch_ptrs[b][l]) ok = false;
                } else {
                    if (val != 0.0f) ok = false;
                }
            }
        }

        reportStatus("Problem 3: Ragged float** Array to Dense Padded Matrix", ok);
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
