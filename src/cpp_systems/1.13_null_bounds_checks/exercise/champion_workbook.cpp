#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

// =========================================================================
// CHAMPION WORKBOOK: Null, Bounds & Alignment Checks
//
// Module: 1.13 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Champion
//
// Focus: SIMD predicate bitmask boundary protection, memory canary corruption
//        detection (guard headers/footers), and kernel dispatch validator.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.13_champion
//   ../../../output/1.13_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

enum CorruptionStatus {
    OK = 0,
    UNDERFLOW_CORRUPT = 1,
    OVERFLOW_CORRUPT = 2,
    BOTH_CORRUPTED = 3,
    INVALID_PTR = 4
};

class CanaryAllocator {
    static constexpr uint64_t MAGIC_HEAD = 0xDEADBEEFCAFEBABEULL;
    static constexpr uint64_t MAGIC_TAIL = 0xBADC0FFEE0DDF00DULL;
    std::unordered_map<void*, size_t> active_allocs;

public:
    // --- YOUR CODE STARTS HERE ---
    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        size_t total_bytes = sizeof(uint64_t) + size + sizeof(uint64_t);
        uint8_t* raw = new uint8_t[total_bytes];

        // Write head canary
        *reinterpret_cast<uint64_t*>(raw) = MAGIC_HEAD;

        uint8_t* payload = raw + sizeof(uint64_t);

        // Write tail canary
        *reinterpret_cast<uint64_t*>(payload + size) = MAGIC_TAIL;

        active_allocs[payload] = size;
        return payload;
    }

    CorruptionStatus check(void* ptr) {
        auto it = active_allocs.find(ptr);
        if (it == active_allocs.end()) return INVALID_PTR;

        size_t size = it->second;
        uint8_t* payload = reinterpret_cast<uint8_t*>(ptr);
        uint8_t* head_ptr = payload - sizeof(uint64_t);
        uint8_t* tail_ptr = payload + size;

        uint64_t actual_head = *reinterpret_cast<uint64_t*>(head_ptr);
        uint64_t actual_tail = *reinterpret_cast<uint64_t*>(tail_ptr);

        bool head_ok = (actual_head == MAGIC_HEAD);
        bool tail_ok = (actual_tail == MAGIC_TAIL);

        if (head_ok && tail_ok) return OK;
        if (!head_ok && !tail_ok) return BOTH_CORRUPTED;
        if (!head_ok) return UNDERFLOW_CORRUPT;
        return OVERFLOW_CORRUPT;
    }

    void deallocate(void* ptr) {
        auto it = active_allocs.find(ptr);
        if (it == active_allocs.end()) return;
        uint8_t* raw = reinterpret_cast<uint8_t*>(ptr) - sizeof(uint64_t);
        active_allocs.erase(it);
        delete[] raw;
    }
    // --- YOUR CODE ENDS HERE ---
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Null, Bounds & Alignment Checks (Champion) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: SIMD Predicate Bitmask Generator for Vectorized Boundary Handling
    //
    // Context: In vectorized CUDA kernels (and SIMD architectures), threads
    //          process vectors of width V = 8 elements. When tensor length N
    //          is not an exact multiple of V, tail elements must be masked out
    //          to prevent out-of-bounds loads and invalid accumulations.
    //
    // Task: Implement `generateVectorPredicates(size_t N, size_t V)`:
    //       - Computes number of tiles: num_tiles = (N + V - 1) / V.
    //       - Returns `std::vector<uint8_t>` where each byte represents the
    //         predicate bitmask for that tile:
    //         Bit `i` (0 <= i < V) is 1 if (tile * V + i < N), else 0.
    //       - Implement `maskedVectorSum(const float* data, size_t N, size_t V,
    //                                   const std::vector<uint8_t>& masks)`:
    //         Simulates vectorized addition: for each tile and each element i in [0, V),
    //         accumulates data[tile * V + i] ONLY if bit i is set in the tile's mask.
    // -------------------------------------------------------------------------
    {
        auto generateVectorPredicates = [](size_t N, size_t V) -> std::vector<uint8_t> {
            // --- YOUR CODE STARTS HERE ---
            size_t num_tiles = (N + V - 1) / V;
            std::vector<uint8_t> masks(num_tiles, 0);
            for (size_t t = 0; t < num_tiles; ++t) {
                uint8_t mask = 0;
                for (size_t i = 0; i < V; ++i) {
                    size_t global_idx = t * V + i;
                    if (global_idx < N) {
                        mask |= (1 << i);
                    }
                }
                masks[t] = mask;
            }
            return masks;
            // --- YOUR CODE ENDS HERE ---
        };

        auto maskedVectorSum = [](const float* data, size_t N, size_t V,
                                  const std::vector<uint8_t>& masks) -> float {
            // --- YOUR CODE STARTS HERE ---
            float total = 0.0f;
            size_t num_tiles = (N + V - 1) / V;
            for (size_t t = 0; t < num_tiles; ++t) {
                uint8_t mask = masks[t];
                for (size_t i = 0; i < V; ++i) {
                    if ((mask & (1 << i)) != 0) {
                        total += data[t * V + i];
                    }
                }
            }
            return total;
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t N = 37;
        const size_t V = 8;
        std::vector<float> data(N);
        float expected_sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i + 1);
            expected_sum += data[i];
        }

        std::vector<uint8_t> masks = generateVectorPredicates(N, V);
        bool ok = true;
        // Total tiles: (37 + 7) / 8 = 5 tiles
        if (masks.size() != 5) ok = false;
        // Tiles 0, 1, 2, 3 should be 0xFF (all 8 bits set)
        for (int t = 0; t < 4; ++t) {
            if (masks[t] != 0xFF) ok = false;
        }
        // Tile 4 has 5 active elements (37 - 32 = 5): bits 0..4 set -> 0x1F (0b00011111)
        if (masks[4] != 0x1F) ok = false;

        float computed_sum = maskedVectorSum(data.data(), N, V, masks);
        if (std::abs(computed_sum - expected_sum) > 1e-4f) ok = false;

        reportStatus("Problem 1: SIMD Predicate Bitmask Boundary Generator", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Memory Canary Corruption Detector (Guard Headers/Footers)
    //
    // Context: In GPU execution runtimes, off-by-one errors in custom kernels
    //          silently overwrite adjacent tensors. Canary wrappers place
    //          known 64-bit bit patterns directly adjacent to buffers to
    //          detect underflows and overflows instantly.
    //
    // Task: Implement `CanaryAllocator`:
    //       Constants:
    //         `MAGIC_CANARY_HEAD = 0xDEADBEEFCAFEBABEULL;`
    //         `MAGIC_CANARY_TAIL = 0xBADC0FFEE0DDF00DULL;`
    //       Methods:
    //         `void* allocate(size_t size)`:
    //           Allocates buffer of size: sizeof(uint64_t) + size + sizeof(uint64_t).
    //           Writes MAGIC_CANARY_HEAD at the beginning.
    //           Writes MAGIC_CANARY_TAIL immediately after the payload.
    //           Returns pointer to the user payload.
    //         `CorruptionStatus check(void* ptr)`:
    //           Checks both canaries.
    //           Returns OK, UNDERFLOW_CORRUPT, OVERFLOW_CORRUPT, or BOTH_CORRUPTED.
    //         `void free(void* ptr)`: Frees the entire block.
    // -------------------------------------------------------------------------
    {
        CanaryAllocator allocator;
        bool ok = true;

        // Test 1: Clean allocation
        float* p1 = reinterpret_cast<float*>(allocator.allocate(16 * sizeof(float)));
        for (int i = 0; i < 16; ++i) p1[i] = static_cast<float>(i);
        if (allocator.check(p1) != OK) ok = false;

        // Test 2: Underflow corruption (writing 1 byte before payload)
        uint8_t* p1_bytes = reinterpret_cast<uint8_t*>(p1);
        p1_bytes[-1] ^= 0xFF; // corrupt head
        if (allocator.check(p1) != UNDERFLOW_CORRUPT) ok = false;
        p1_bytes[-1] ^= 0xFF; // restore

        // Test 3: Overflow corruption (writing past end of payload)
        p1_bytes[16 * sizeof(float) + 2] ^= 0xAA; // corrupt tail
        if (allocator.check(p1) != OVERFLOW_CORRUPT) ok = false;
        p1_bytes[16 * sizeof(float) + 2] ^= 0xAA; // restore

        if (allocator.check(p1) != OK) ok = false;
        allocator.deallocate(p1);

        reportStatus("Problem 2: Memory Canary Corruption Detector", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Fault-Tolerant Batched Kernel Dispatch Validator
    //
    // Context: In high-concurrency inference runtimes (e.g. TensorRT, vLLM),
    //          dispatches specify inputs and outputs. An invalid pointer, an
    //          unaligned buffer, out-of-bounds addresses, or buffer aliasing
    //          will crash the GPU driver or cause data hazards.
    //
    // Task: Implement `validateKernelDispatches(...)`:
    //       Inspects a list of `DispatchJob`:
    //         `{ int id; const void* in_ptr; size_t in_bytes; size_t in_align;
    //            void* out_ptr; size_t out_bytes; size_t out_align; }`
    //       Against designated pool `[pool_start, pool_start + pool_size)`:
    //       Checks:
    //         1. Neither pointer is null and sizes > 0.
    //         2. Both pointers are within [pool_start, pool_start + pool_size - bytes].
    //         3. Both pointers satisfy their required alignment (addr % align == 0).
    //         4. Within each dispatch: [in_ptr, in_ptr + in_bytes) does NOT overlap
    //            with [out_ptr, out_ptr + out_bytes).
    //         5. Across all dispatches: no two distinct output buffers overlap.
    //       Returns index of first invalid dispatch, or -1 if all are valid.
    // -------------------------------------------------------------------------
    {
        struct DispatchJob {
            int id;
            const void* in_ptr;
            size_t in_bytes;
            size_t in_align;
            void* out_ptr;
            size_t out_bytes;
            size_t out_align;
        };

        auto validateKernelDispatches = [](const std::vector<DispatchJob>& jobs,
                                           const void* pool_start, size_t pool_size) -> int {
            // --- YOUR CODE STARTS HERE ---
            uintptr_t p_start = reinterpret_cast<uintptr_t>(pool_start);
            uintptr_t p_end = p_start + pool_size;

            auto isValidSpan = [&](uintptr_t addr, size_t bytes, size_t align) -> bool {
                if (addr == 0 || bytes == 0) return false;
                if ((addr & (align - 1)) != 0) return false;
                if (addr < p_start || addr + bytes > p_end) return false;
                return true;
            };

            auto spansOverlap = [](uintptr_t a_start, size_t a_len,
                                   uintptr_t b_start, size_t b_len) -> bool {
                uintptr_t a_end = a_start + a_len;
                uintptr_t b_end = b_start + b_len;
                return !(a_end <= b_start || b_end <= a_start);
            };

            for (size_t i = 0; i < jobs.size(); ++i) {
                const auto& job = jobs[i];
                uintptr_t in_addr = reinterpret_cast<uintptr_t>(job.in_ptr);
                uintptr_t out_addr = reinterpret_cast<uintptr_t>(job.out_ptr);

                if (!isValidSpan(in_addr, job.in_bytes, job.in_align)) return static_cast<int>(i);
                if (!isValidSpan(out_addr, job.out_bytes, job.out_align)) return static_cast<int>(i);

                // Check self-aliasing between in and out
                if (spansOverlap(in_addr, job.in_bytes, out_addr, job.out_bytes)) {
                    return static_cast<int>(i);
                }

                // Check collision with prior output buffers
                for (size_t j = 0; j < i; ++j) {
                    uintptr_t prev_out = reinterpret_cast<uintptr_t>(jobs[j].out_ptr);
                    if (spansOverlap(out_addr, job.out_bytes, prev_out, jobs[j].out_bytes)) {
                        return static_cast<int>(i);
                    }
                }
            }

            return -1; // All valid
            // --- YOUR CODE ENDS HERE ---
        };

        std::vector<uint8_t> memory_pool(65536, 0);
        uint8_t* pool_base = memory_pool.data();
        // Ensure pool_base is aligned to 64 bytes
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(pool_base);
        size_t pad = (64 - (raw_addr % 64)) % 64;
        uint8_t* aligned_pool = pool_base + pad;
        size_t effective_size = 65536 - pad;

        // Valid batch
        std::vector<DispatchJob> valid_batch = {
            {1, aligned_pool, 512, 64, aligned_pool + 512, 512, 64},
            {2, aligned_pool + 1024, 256, 32, aligned_pool + 1536, 256, 32},
            {3, aligned_pool + 2048, 128, 16, aligned_pool + 3072, 128, 16}
        };

        // Self-aliasing batch (in and out overlap)
        std::vector<DispatchJob> self_alias_batch = {
            {1, aligned_pool, 512, 64, aligned_pool + 256, 512, 64}
        };

        // Cross-dispatch output collision batch (dispatches 0 and 1 write to same output)
        std::vector<DispatchJob> collision_batch = {
            {1, aligned_pool, 512, 64, aligned_pool + 1024, 512, 64},
            {2, aligned_pool + 512, 512, 64, aligned_pool + 1280, 512, 64} // overlaps 1024..1536
        };

        // Misaligned pointer batch
        std::vector<DispatchJob> misaligned_batch = {
            {1, aligned_pool + 3, 512, 64, aligned_pool + 1024, 512, 64}
        };

        bool ok = true;
        if (validateKernelDispatches(valid_batch, aligned_pool, effective_size) != -1) ok = false;
        if (validateKernelDispatches(self_alias_batch, aligned_pool, effective_size) != 0) ok = false;
        if (validateKernelDispatches(collision_batch, aligned_pool, effective_size) != 1) ok = false;
        if (validateKernelDispatches(misaligned_batch, aligned_pool, effective_size) != 0) ok = false;

        reportStatus("Problem 3: Batched Kernel Dispatch Validator", ok);
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
