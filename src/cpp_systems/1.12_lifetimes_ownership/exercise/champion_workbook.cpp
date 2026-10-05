#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <atomic>
#include <functional>

// =========================================================================
// CHAMPION WORKBOOK: Lifetimes, Ownership & Resize
//
// Module: 1.12 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: Intrusive reference-counted tensor storage (c10::intrusive_ptr),
//        Zero-copy move vs deep copy benchmark, and alien buffer adoption.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.12_champion
//   ../../../output/1.12_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " Mmoves/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Lifetimes, Ownership & Resize (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Intrusive Reference-Counted Tensor Storage (c10 Style)
    //
    // Context: `std::shared_ptr` allocates a separate heap "control block" for the
    //          refcount. PyTorch uses intrusive reference counting (`c10::intrusive_ptr`)
    //          where the atomic refcount is embedded directly inside the storage struct,
    //          saving an allocation and improving cache locality.
    //
    // Task: Implement `IntrusiveStorage` with atomic `ref_count`:
    //       Support `retain()` and `release()` where release deletes the struct
    //       when ref_count reaches 0.
    // -------------------------------------------------------------------------
    {
        static int total_freed = 0;
        total_freed = 0;

        struct IntrusiveStorage {
            std::atomic<int> ref_count{1};
            float* data;
            size_t size;

            IntrusiveStorage(size_t n) : size(n) {
                data = new float[n];
            }

            ~IntrusiveStorage() {
                delete[] data;
                total_freed++;
            }

            void retain() {
                ref_count.fetch_add(1, std::memory_order_relaxed);
            }

            void release() {
                if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    delete this;
                }
            }
        };

        IntrusiveStorage* storage = new IntrusiveStorage(256);
        storage->retain(); // View A
        storage->retain(); // View B

        storage->release(); // View B dropped
        bool p1_passed = (total_freed == 0);

        storage->release(); // View A dropped
        if (total_freed != 0) p1_passed = false;

        storage->release(); // Root dropped -> deleted!
        if (total_freed != 1) p1_passed = false;

        reportStatus("Problem 1: Intrusive Ref-Counted Storage (c10 Style)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Zero-Copy Memory Move vs Deep Copy Benchmark
    //
    // Context: Demonstrates that move semantics transfer multi-megabyte tensors
    //          in nanoseconds by simply exchanging 8-byte pointer addresses,
    //          independent of tensor size.
    //
    // Task: Execute 1,000,000 move operations transferring ownership of a 10 MB buffer.
    //       Measure throughput in Million moves/second (Mmoves/s).
    // -------------------------------------------------------------------------
    {
        struct LargeTensor {
            float* data;
            size_t size;

            LargeTensor(size_t n) : size(n), data(new float[n]) {}
            ~LargeTensor() { delete[] data; }

            LargeTensor(LargeTensor&& other) noexcept : data(other.data), size(other.size) {
                other.data = nullptr;
                other.size = 0;
            }

            LargeTensor& operator=(LargeTensor&& other) noexcept {
                if (this != &other) {
                    delete[] data;
                    data = other.data;
                    size = other.size;
                    other.data = nullptr;
                    other.size = 0;
                }
                return *this;
            }

            LargeTensor(const LargeTensor&) = delete;
            LargeTensor& operator=(const LargeTensor&) = delete;
        };

        const int NUM_MOVES = 1000000;
        LargeTensor holder(2621440); // 10 MB tensor

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_MOVES; ++i) {
            LargeTensor temp = std::move(holder);
            holder = std::move(temp);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double mmoves_per_sec = (NUM_MOVES / elapsed_sec) / 1e6;

        bool p2_passed = (holder.data != nullptr) && (mmoves_per_sec > 10.0);
        reportStatus("Problem 2: Zero-Copy Tensor Move Benchmark", p2_passed, mmoves_per_sec);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Alien Buffer Adoption with Custom Lifecycle Callbacks
    //
    // Context: When integrating external C/CUDA libraries (e.g. cuFFT, cuDNN), memory
    //          allocated by an external runtime must be adopted into a managed C++
    //          object with an arbitrary callback deleter (e.g. `cudaFree` or custom pool return).
    //
    // Task: Implement `AdoptedBuffer` that takes a raw pointer and a `std::function<void(void*)>`
    //       deleter. Verify that when AdoptedBuffer is destroyed, the custom callback runs.
    // -------------------------------------------------------------------------
    {
        bool alien_free_invoked = false;

        struct AdoptedBuffer {
            void* raw_ptr;
            std::function<void(void*)> custom_deleter;

            AdoptedBuffer(void* ptr, std::function<void(void*)> deleter)
                : raw_ptr(ptr), custom_deleter(deleter) {}

            ~AdoptedBuffer() {
                if (raw_ptr && custom_deleter) {
                    custom_deleter(raw_ptr);
                    raw_ptr = nullptr;
                }
            }
        };

        {
            int* external_memory = new int[10];
            AdoptedBuffer managed(external_memory, [&](void* p) {
                alien_free_invoked = true;
                delete[] static_cast<int*>(p);
            });
        } // managed destroyed here -> callback must run

        bool p3_passed = alien_free_invoked;
        reportStatus("Problem 3: Alien Buffer Adoption with Custom Callback", p3_passed);
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
