#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdlib>

// =========================================================================
// BEGINNER WORKBOOK: Single & Array Allocations
//
// Module: 1.9 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Beginner
//
// Focus: Heap tensor allocations with memset, RAII memory wrappers, and
//        dynamic array growth with capacity doubling.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.9_beginner
//   ../../../output/1.9_beginner
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
    std::cout << "--- WORKBOOK: Single & Array Allocations (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Dynamic Heap Tensor Allocation & Zero-Initialization
    //
    // Context: Unlike stack arrays, GPU and host tensors are dynamically sized
    //          at runtime based on batch size and sequence length.
    //          Uninitialized heap memory contains garbage bytes; initializing
    //          with std::memset or calloc is essential for safety.
    //
    // Task: Allocate a heap buffer of N=256 floats using std::malloc.
    //       Zero-initialize all bytes using std::memset.
    //       Fill elements with val[i] = i * 2.0f.
    //       Verify all values, then free the memory with std::free.
    // -------------------------------------------------------------------------
    {
        const size_t N = 256;
        float* heap_tensor = nullptr;

        // TODO: Allocate heap_tensor, memset to 0, populate with i * 2.0f.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = (heap_tensor != nullptr);
        if (heap_tensor) {
            for (size_t i = 0; i < N && p1_passed; ++i) {
                if (heap_tensor[i] != static_cast<float>(i * 2.0f)) {
                    p1_passed = false;
                }
            }
            std::free(heap_tensor);
        }

        reportStatus("Problem 1: Dynamic Heap Allocation & Zero-Initialization", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: RAII Buffer Wrapper (Scoped Automatic Cleanup)
    //
    // Context: In high-performance C++ and CUDA, manual free calls cause leaks
    //          whenever exceptions occur. An RAII struct ensures memory is automatically
    //          freed when the variable falls out of scope.
    //
    // Task: Implement a simple RAII class `ScopedFloatBuffer` that allocates `size`
    //       floats in its constructor and calls `std::free` in its destructor.
    //       Support `data()` and `size()` accessors.
    // -------------------------------------------------------------------------
    {
        struct ScopedFloatBuffer {
            float* ptr = nullptr;
            size_t count = 0;

            ScopedFloatBuffer(size_t n) : count(n) {
                ptr = static_cast<float*>(std::malloc(n * sizeof(float)));
            }

            ~ScopedFloatBuffer() {
                if (ptr) {
                    std::free(ptr);
                    ptr = nullptr;
                }
            }

            float* data() { return ptr; }
            size_t size() const { return count; }

            // Prevent copying to avoid double-free
            ScopedFloatBuffer(const ScopedFloatBuffer&) = delete;
            ScopedFloatBuffer& operator=(const ScopedFloatBuffer&) = delete;
        };

        bool p2_passed = false;
        {
            ScopedFloatBuffer buf(128);
            if (buf.data() != nullptr && buf.size() == 128) {
                for (size_t i = 0; i < buf.size(); ++i) {
                    buf.data()[i] = static_cast<float>(i * 1.5f);
                }
                p2_passed = (buf.data()[10] == 15.0f);
            }
        } // buf is destroyed and freed here automatically

        reportStatus("Problem 2: RAII Scoped Buffer Memory Management", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Dynamic Array Resizing with Capacity Doubling
    //
    // Context: In streaming token generation, when the number of generated tokens
    //          exceeds buffer capacity, the buffer must reallocate with capacity
    //          doubling (amortized O(1) appending).
    //
    // Task: Implement dynamic appending for 20 elements into an initial capacity of 4:
    //       Whenever `size == capacity`, double capacity, allocate a new buffer,
    //       copy old elements over, free the old buffer, and insert the new element.
    // -------------------------------------------------------------------------
    {
        size_t capacity = 4;
        size_t size = 0;
        int* dynamic_arr = static_cast<int*>(std::malloc(capacity * sizeof(int)));

        const int TOTAL_ELEMENTS = 20;

        // TODO: In a loop from i = 0 to TOTAL_ELEMENTS - 1:
        // Append i to dynamic_arr. If size == capacity, double capacity and reallocate.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = (size == TOTAL_ELEMENTS) && (capacity >= TOTAL_ELEMENTS);
        for (int i = 0; i < TOTAL_ELEMENTS && p3_passed; ++i) {
            if (dynamic_arr[i] != i) p3_passed = false;
        }

        std::free(dynamic_arr);

        reportStatus("Problem 3: Dynamic Array Growth with Capacity Doubling", p3_passed);
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
