#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <new>

// =========================================================================
// INTERMEDIATE WORKBOOK: Arenas & Placements
//
// Module: 1.11 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Intermediate
//
// Focus: Arena save/restore markers (rewinding scratchpad memory),
//        heterogeneous multi-type aligned allocations, and placement new arrays.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.11_intermediate
//   ../../../output/1.11_intermediate
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
    std::cout << "--- WORKBOOK: Arenas & Placements (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Arena Allocator with Save & Restore Markers (Scratchpad Rewinding)
    //
    // Context: In transformer forward passes, temporary activations (e.g. Q @ K^T scores)
    //          are needed only during the attention calculation. After computing the output,
    //          the arena is rewound to a previous snapshot marker to reuse memory for FFN.
    //
    // Task: Implement `RewindableArena` over 4096 bytes:
    //         - `void* allocate(size_t bytes)`: bump allocation
    //         - `size_t save()`: returns current byte offset
    //         - `void rewind(size_t marker)`: resets byte offset back to marker
    // -------------------------------------------------------------------------
    {
        class RewindableArena {
        public:
            std::vector<uint8_t> buffer;
            size_t offset = 0;

            RewindableArena(size_t cap) : buffer(cap) {}

            void* allocate(size_t bytes) {
                if (offset + bytes > buffer.size()) return nullptr;
                void* p = &buffer[offset];
                offset += bytes;
                return p;
            }

            size_t save() const { return offset; }
            void rewind(size_t marker) {
                if (marker <= offset) offset = marker;
            }
        };

        RewindableArena arena(4096);
        void* persistent_weights = arena.allocate(1024); // Persistent weights [0..1024)
        size_t checkpoint = arena.save();                // Marker at 1024

        void* temp_attention_scores = arena.allocate(2048); // Temp memory [1024..3072)
        arena.rewind(checkpoint);                           // Rewind back to 1024!

        void* temp_ffn_activations = arena.allocate(1500);  // Reuses memory starting at 1024!

        bool p1_passed = (temp_attention_scores == temp_ffn_activations) &&
                         (arena.offset == 1024 + 1500);

        reportStatus("Problem 1: Arena Save & Restore Markers (Scratchpad Rewind)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Heterogeneous Typed Aligned Allocations from a Single Arena
    //
    // Context: A single GPU memory arena must service requests of different types
    //          with different alignment constraints:
    //            uint8_t  (align 1)
    //            int32_t  (align 4)
    //            double   (align 8)
    //            float4   (align 16)
    //
    // Task: Implement an aligned allocator function:
    //       `void* allocate_aligned(size_t size, size_t alignment)`
    //       Allocate one uint8_t, one int32_t, one double, and one float4.
    //       Verify that every returned address satisfies `(addr % alignment == 0)`.
    // -------------------------------------------------------------------------
    {
        std::vector<uint8_t> memory_pool(1024);
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(memory_pool.data());
        size_t current_bump = 0;

        auto allocate_aligned = [&](size_t size, size_t align) -> void* {
            uintptr_t curr_addr = base_addr + current_bump;
            uintptr_t aligned_addr = (curr_addr + (align - 1)) & ~(align - 1);
            size_t new_bump = (aligned_addr - base_addr) + size;
            if (new_bump > 1024) return nullptr;
            current_bump = new_bump;
            return reinterpret_cast<void*>(aligned_addr);
        };

        void* p_u8 = allocate_aligned(1, 1);
        void* p_i32 = allocate_aligned(4, 4);
        void* p_f64 = allocate_aligned(8, 8);
        void* p_vec = allocate_aligned(16, 16);

        bool p2_passed = (p_u8 != nullptr) && (p_i32 != nullptr) &&
                         (p_f64 != nullptr) && (p_vec != nullptr) &&
                         (reinterpret_cast<uintptr_t>(p_i32) % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(p_f64) % 8 == 0) &&
                         (reinterpret_cast<uintptr_t>(p_vec) % 16 == 0);

        reportStatus("Problem 2: Heterogeneous Typed Aligned Arena Allocations", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Placement new Array Construction & Explicit Destruction Loop
    //
    // Context: In low-level engines, an array of C++ objects is constructed in a
    //          contiguous memory slab using placement new, and destroyed manually.
    //
    // Task: Given struct `ManagedToken`:
    //       Construct 5 ManagedToken objects in `storage` using placement new:
    //         `new (&storage[i]) ManagedToken(i, i * 100);`
    //       Verify all objects, then destroy all 5 objects in reverse order (4 down to 0).
    // -------------------------------------------------------------------------
    {
        struct ManagedToken {
            int token_id;
            int position;
            bool active = false;

            ManagedToken(int id, int pos) : token_id(id), position(pos), active(true) {}
            ~ManagedToken() { active = false; }
        };

        const int COUNT = 5;
        alignas(alignof(ManagedToken)) uint8_t storage[COUNT * sizeof(ManagedToken)];
        ManagedToken* tokens = reinterpret_cast<ManagedToken*>(storage);

        // TODO: Construct 5 ManagedTokens using placement new.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < COUNT; ++i) {
            if (tokens[i].token_id != i || tokens[i].position != i * 100 || !tokens[i].active) {
                p3_passed = false;
            }
        }

        // Explicit destruction loop
        for (int i = COUNT - 1; i >= 0; --i) {
            tokens[i].~ManagedToken();
            if (tokens[i].active) p3_passed = false;
        }

        reportStatus("Problem 3: Placement new Array Construction & Destruction", p3_passed);
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
