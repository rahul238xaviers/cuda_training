#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <new>

// =========================================================================
// BEGINNER WORKBOOK: Arenas & Placements
//
// Module: 1.11 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Beginner
//
// Focus: Placement new construction in pre-allocated buffers, linear arena
//        allocation, and arena capacity boundary validation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.11_beginner
//   ../../../output/1.11_beginner
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
    std::cout << "--- WORKBOOK: Arenas & Placements (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Placement `new` Object Construction in Raw Buffer
    //
    // Context: In GPU shared memory or pre-allocated scratchpads, C++ objects
    //          are initialized without heap allocation using placement new:
    //            `new (raw_ptr) MyClass(args...)`
    //          Because `delete` cannot be used on raw buffers, the destructor
    //          must be called explicitly: `obj->~MyClass()`.
    //
    // Task: Given struct `TensorDescriptor`, construct it in `raw_storage` buffer
    //       using placement new with dims=(8, 16).
    //       Verify its members, and invoke its destructor explicitly!
    // -------------------------------------------------------------------------
    {
        struct TensorDescriptor {
            int rows;
            int cols;
            bool destructor_called = false;

            TensorDescriptor(int r, int c) : rows(r), cols(c) {}
            ~TensorDescriptor() {
                destructor_called = true;
            }
        };

        alignas(alignof(TensorDescriptor)) uint8_t raw_storage[sizeof(TensorDescriptor)];

        TensorDescriptor* desc_ptr = nullptr;

        // TODO: Construct TensorDescriptor in raw_storage with rows=8, cols=16.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (desc_ptr != nullptr) &&
                         (desc_ptr->rows == 8) &&
                         (desc_ptr->cols == 16);

        if (desc_ptr) {
            desc_ptr->~TensorDescriptor();
            if (!desc_ptr->destructor_called) p1_passed = false;
        }

        reportStatus("Problem 1: Placement new in Pre-Allocated Storage", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Linear Arena Sub-Buffer Allocation
    //
    // Context: A Linear Arena (Bump Pointer) allocates sequential chunks of memory
    //          by advancing an offset.
    //
    // Task: Implement a simple linear arena over a 1024-byte buffer:
    //       Allocate three buffers:
    //         - `buf1`: 64 bytes
    //         - `buf2`: 128 bytes
    //         - `buf3`: 256 bytes
    //       Verify that allocations are contiguous and non-overlapping.
    // -------------------------------------------------------------------------
    {
        const size_t ARENA_SIZE = 1024;
        uint8_t arena_pool[ARENA_SIZE];
        size_t current_offset = 0;

        auto arena_alloc = [&](size_t bytes) -> void* {
            if (current_offset + bytes > ARENA_SIZE) return nullptr;
            void* p = &arena_pool[current_offset];
            current_offset += bytes;
            return p;
        };

        void* b1 = arena_alloc(64);
        void* b2 = arena_alloc(128);
        void* b3 = arena_alloc(256);

        bool p2_passed = (b1 != nullptr) && (b2 != nullptr) && (b3 != nullptr) &&
                         (static_cast<uint8_t*>(b2) == static_cast<uint8_t*>(b1) + 64) &&
                         (static_cast<uint8_t*>(b3) == static_cast<uint8_t*>(b2) + 128) &&
                         (current_offset == 448);

        reportStatus("Problem 2: Linear Arena Contiguous Sub-Allocations", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Arena Capacity Boundary & Overflow Rejection
    //
    // Context: Safe allocators must prevent memory corruption by rejecting requests
    //          that exceed the remaining capacity.
    //
    // Task: Given an arena with CAPACITY=500 bytes and currently used 450 bytes:
    //       Attempt to allocate 100 bytes (should return nullptr).
    //       Attempt to allocate 50 bytes (should succeed and saturate arena to 500).
    //       Attempt to allocate 1 byte (should return nullptr).
    // -------------------------------------------------------------------------
    {
        const size_t CAPACITY = 500;
        uint8_t memory[CAPACITY];
        size_t used = 450;

        auto safe_alloc = [&](size_t bytes) -> void* {
            if (used + bytes > CAPACITY) return nullptr;
            void* p = &memory[used];
            used += bytes;
            return p;
        };

        void* req1 = safe_alloc(100); // Fail -> nullptr
        void* req2 = safe_alloc(50);  // Succeed -> fills to 500
        void* req3 = safe_alloc(1);   // Fail -> nullptr

        bool p3_passed = (req1 == nullptr) && (req2 != nullptr) && (req3 == nullptr) && (used == 500);
        reportStatus("Problem 3: Arena Capacity Boundary & Overflow Rejection", p3_passed);
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
