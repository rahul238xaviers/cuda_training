#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <memory>

// =========================================================================
// BEGINNER WORKBOOK: Lifetimes, Ownership & Resize
//
// Module: 1.12 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Beginner
//
// Focus: Unique ownership transfer via move semantics, RAII custom deleters,
//        and safe buffer resizing.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.12_beginner
//   ../../../output/1.12_beginner
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
    std::cout << "--- WORKBOOK: Lifetimes, Ownership & Resize (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Unique Ownership Transfer via Move Semantics
    //
    // Context: In GPU systems, moving a tensor between execution pipelines
    //          must transfer buffer ownership without copying data or causing double-free.
    //
    // Task: Implement a class `DeviceBuffer` that holds `float* data` and `size_t size`:
    //       Implement a move constructor:
    //         `DeviceBuffer(DeviceBuffer&& other)`
    //       It takes ownership of other's data pointer and sets other.data to nullptr.
    // -------------------------------------------------------------------------
    {
        struct DeviceBuffer {
            float* data = nullptr;
            size_t size = 0;

            DeviceBuffer(size_t n) : size(n) {
                data = new float[n];
                for (size_t i = 0; i < n; ++i) data[i] = static_cast<float>(i * 10);
            }

            ~DeviceBuffer() {
                delete[] data;
            }

            // TODO: Move constructor
            DeviceBuffer(DeviceBuffer&& other) noexcept {
                data = other.data;
                size = other.size;
                other.data = nullptr;
                other.size = 0;
            }

            // Prevent copying
            DeviceBuffer(const DeviceBuffer&) = delete;
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        };

        DeviceBuffer bufA(64);
        float* original_ptr = bufA.data;

        // Move construct bufB from bufA
        DeviceBuffer bufB = std::move(bufA);

        bool p1_passed = (bufA.data == nullptr) && (bufA.size == 0) &&
                         (bufB.data == original_ptr) && (bufB.size == 64) &&
                         (bufB.data[5] == 50.0f);

        reportStatus("Problem 1: Unique Ownership Transfer via Move Semantics", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: RAII Unique Pointer with Custom Deleter
    //
    // Context: In CUDA, host-pinned memory allocated with `cudaMallocHost` must be
    //          freed with `cudaFreeHost`. `std::unique_ptr` with a custom deleter
    //          ensures the correct deallocation function is invoked.
    //
    // Task: Implement a custom deleter functor `HostPinnedDeleter` that records
    //       when `free_called` is set to true. Wrap a malloc-allocated float buffer
    //       in `std::unique_ptr<float, HostPinnedDeleter>`.
    // -------------------------------------------------------------------------
    {
        static bool custom_free_invoked = false;
        custom_free_invoked = false;

        struct HostPinnedDeleter {
            void operator()(float* ptr) const {
                if (ptr) {
                    custom_free_invoked = true;
                    std::free(ptr);
                }
            }
        };

        {
            float* raw = static_cast<float*>(std::malloc(32 * sizeof(float)));
            std::unique_ptr<float, HostPinnedDeleter> uptr(raw);
            raw[0] = 42.0f;
        } // uptr destroyed here -> must invoke HostPinnedDeleter!

        bool p2_passed = custom_free_invoked;
        reportStatus("Problem 2: RAII Unique Pointer with Custom Deleter", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Safe Buffer Resizing with Existing Data Preservation
    //
    // Context: In dynamic token generation, resizing an activation buffer requires
    //          allocating new capacity and copying valid old elements.
    //
    // Task: Given buffer with 4 elements {10.0f, 20.0f, 30.0f, 40.0f}:
    //       Resize to new capacity of 8 elements.
    //       Ensure old 4 elements are preserved, and new 4 elements are initialized to 0.0f.
    // -------------------------------------------------------------------------
    {
        const size_t old_cap = 4;
        const size_t new_cap = 8;
        float* buf = new float[old_cap]{10.0f, 20.0f, 30.0f, 40.0f};

        // TODO: Allocate new_buf of new_cap, copy old elements, zero-fill rest, delete old buf.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = (buf != nullptr);
        if (buf) {
            if (buf[0] != 10.0f || buf[1] != 20.0f || buf[2] != 30.0f || buf[3] != 40.0f) p3_passed = false;
            for (size_t i = old_cap; i < new_cap; ++i) {
                if (buf[i] != 0.0f) p3_passed = false;
            }
            delete[] buf;
        }

        reportStatus("Problem 3: Safe Buffer Resizing with Data Preservation", p3_passed);
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
