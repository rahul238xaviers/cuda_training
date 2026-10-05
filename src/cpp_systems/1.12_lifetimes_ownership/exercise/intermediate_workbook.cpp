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
// INTERMEDIATE WORKBOOK: Lifetimes, Ownership & Resize
//
// Module: 1.12 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Intermediate
//
// Focus: Reference-counted shared tensor storage, clone() vs view() semantics,
//        and disowned raw pointer extraction (.release()).
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.12_intermediate
//   ../../../output/1.12_intermediate
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
    std::cout << "--- WORKBOOK: Lifetimes, Ownership & Resize (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Reference-Counted Shared Tensor Storage (PyTorch Style)
    //
    // Context: In PyTorch (`c10::StorageImpl`), multiple `at::Tensor` instances can
    //          point to the same underlying storage. The storage maintains an atomic
    //          reference count and is freed only when the last reference is destroyed.
    //
    // Task: Implement `SharedStorage` class holding `float* data`, `size_t size`, and `int ref_count`:
    //         - `void retain()`: increments ref_count
    //         - `void release()`: decrements ref_count; if 0, deletes data and deletes this
    // -------------------------------------------------------------------------
    {
        static int total_destructions = 0;
        total_destructions = 0;

        struct SharedStorage {
            float* data;
            size_t size;
            int ref_count;

            SharedStorage(size_t n) : size(n), ref_count(1) {
                data = new float[n];
            }

            ~SharedStorage() {
                delete[] data;
                total_destructions++;
            }

            void retain() { ref_count++; }
            void release() {
                ref_count--;
                if (ref_count == 0) delete this;
            }
        };

        SharedStorage* storage = new SharedStorage(128); // ref = 1
        storage->retain(); // View 1 created -> ref = 2
        storage->retain(); // View 2 created -> ref = 3

        storage->release(); // View 2 destroyed -> ref = 2
        bool p1_passed = (total_destructions == 0);

        storage->release(); // View 1 destroyed -> ref = 1
        if (total_destructions != 0) p1_passed = false;

        storage->release(); // Root tensor destroyed -> ref = 0 -> deleted!
        if (total_destructions != 1) p1_passed = false;

        reportStatus("Problem 1: Reference-Counted Shared Tensor Storage", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Deep Copy (.clone()) vs Shallow Copy (.view()) Semantics
    //
    // Context: In tensor frameworks, `.view()` creates a new tensor object that
    //          shares the same memory buffer (aliasing). In contrast, `.clone()`
    //          allocates an independent buffer and copies elements.
    //
    // Task: Given tensor A with values {1.0f, 2.0f, 3.0f, 4.0f}:
    //       1. Create view V (shares memory with A)
    //       2. Create clone C (independent allocation with copy)
    //       3. Modify V[0] = 99.0f. Verify A[0] is changed, but C[0] is NOT changed!
    // -------------------------------------------------------------------------
    {
        struct Tensor {
            std::shared_ptr<std::vector<float>> storage;

            Tensor(size_t n) : storage(std::make_shared<std::vector<float>>(n)) {}

            Tensor view() {
                Tensor v(0);
                v.storage = this->storage; // Shares buffer
                return v;
            }

            Tensor clone() {
                Tensor c(0);
                c.storage = std::make_shared<std::vector<float>>(*this->storage); // Deep copy
                return c;
            }

            float& operator[](size_t i) { return (*storage)[i]; }
        };

        Tensor A(4);
        A[0] = 1.0f; A[1] = 2.0f; A[2] = 3.0f; A[3] = 4.0f;

        Tensor V = A.view();
        Tensor C = A.clone();

        // Mutate through view
        V[0] = 99.0f;

        bool p2_passed = (A[0] == 99.0f) && (C[0] == 1.0f);
        reportStatus("Problem 2: Deep Copy (.clone()) vs Shallow Copy (.view())", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Move-Only Vector Wrapper with Disowned Extraction (.release())
    //
    // Context: When passing memory allocated in modern C++ to raw C APIs (like
    //          legacy cuBLAS or MPI), the container must disown the memory pointer
    //          without calling its destructor.
    //
    // Task: Implement `release()` method on a move-only buffer:
    //       It returns the internal raw pointer and sets its member to nullptr.
    // -------------------------------------------------------------------------
    {
        struct MoveOnlyBuffer {
            float* ptr = nullptr;
            size_t size = 0;

            MoveOnlyBuffer(size_t n) : size(n) {
                ptr = new float[n];
            }

            ~MoveOnlyBuffer() {
                delete[] ptr;
            }

            float* release() {
                float* p = ptr;
                ptr = nullptr;
                size = 0;
                return p;
            }

            MoveOnlyBuffer(const MoveOnlyBuffer&) = delete;
            MoveOnlyBuffer& operator=(const MoveOnlyBuffer&) = delete;
        };

        MoveOnlyBuffer buf(64);
        float* extracted = buf.release();

        bool p3_passed = (buf.ptr == nullptr) && (buf.size == 0) && (extracted != nullptr);
        delete[] extracted; // Clean up extracted pointer

        reportStatus("Problem 3: Disowned Raw Pointer Extraction (.release())", p3_passed);
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
