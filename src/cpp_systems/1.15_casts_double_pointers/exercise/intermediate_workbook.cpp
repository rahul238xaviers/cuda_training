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
// INTERMEDIATE WORKBOOK: Type Casts & Double Pointers
//
// Module: 1.15 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Intermediate
//
// Focus: Handle-based memory relocation (double-pointer table), row-index
//        float** GEMV dispatch, and type-erased void* runtime tensor scaling.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.15_intermediate
//   ../../../output/1.15_intermediate
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// Structs and enums for Problem 3
enum class DType { FP32, INT32, INT8 };

class RelocationManager {
    static constexpr size_t MAX_SLOTS = 32;
    float* table[MAX_SLOTS] = {nullptr};
    bool in_use[MAX_SLOTS] = {false};

public:
    // --- YOUR CODE STARTS HERE ---
    float** createHandle(float* initial_ptr) {
        for (size_t i = 0; i < MAX_SLOTS; ++i) {
            if (!in_use[i]) {
                in_use[i] = true;
                table[i] = initial_ptr;
                return &table[i];
            }
        }
        return nullptr;
    }

    void relocate(float** handle, float* new_ptr, size_t num_elements) {
        if (!handle || !*handle || !new_ptr) return;
        std::memcpy(new_ptr, *handle, num_elements * sizeof(float));
        delete[] *handle;
        *handle = new_ptr;
    }

    void releaseHandle(float** handle) {
        if (!handle) return;
        for (size_t i = 0; i < MAX_SLOTS; ++i) {
            if (&table[i] == handle) {
                delete[] table[i];
                table[i] = nullptr;
                in_use[i] = false;
                return;
            }
        }
    }
    // --- YOUR CODE ENDS HERE ---
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Type Casts & Double Pointers (Intermediate) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Handle-Based Memory Relocation (Double Pointer Indirection)
    //
    // Context: In GPU tensor memory managers (e.g. PyTorch Caching Allocator),
    //          fragmentation is solved by compacting live buffers. If client
    //          code holds direct pointers (float*), moving memory breaks those
    //          pointers. Using handles (float** pointing to an internal table)
    //          allows relocating the physical buffer by updating only the table.
    //
    // Task: Implement `RelocationManager`:
    //       - Stores an internal array of pointers: `float* table[MAX_SLOTS]`.
    //       - `float** createHandle(float* initial_ptr)`:
    //         Finds a free slot in table, sets `table[slot] = initial_ptr`,
    //         and returns address `&table[slot]`.
    //       - `void relocate(float** handle, float* new_ptr, size_t num_elements)`:
    //         Copies `num_elements` from `*handle` to `new_ptr`, frees `*handle`,
    //         and updates `*handle = new_ptr`.
    //       - `void releaseHandle(float** handle)`: frees memory and marks slot free.
    // -------------------------------------------------------------------------
    {
        RelocationManager mgr;
        const size_t N = 64;
        float* blockA = new float[N];
        for (size_t i = 0; i < N; ++i) blockA[i] = static_cast<float>(i * 3);

        float** handleA = mgr.createHandle(blockA);
        bool ok = true;
        if (!handleA || *handleA != blockA) ok = false;
        if ((**handleA) != 0.0f || (*handleA)[10] != 30.0f) ok = false;

        // Relocate blockA to a new location
        float* blockB = new float[N];
        mgr.relocate(handleA, blockB, N);

        // Verify that client's handle variable (handleA) was NOT reassigned,
        // but dereferencing it yields the relocated memory
        if (*handleA != blockB) ok = false;
        if ((**handleA) != 0.0f || (*handleA)[10] != 30.0f) ok = false;

        mgr.releaseHandle(handleA);

        reportStatus("Problem 1: Double-Pointer Handle Memory Relocation", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Pointer-to-Pointer Matrix Row Indexing & GEMV
    //
    // Context: In classical numerical libraries and CUDA C bindings, 2D arrays
    //          are sometimes accessed via a table of row pointers `float** rows`.
    //
    // Task: Implement `buildRowIndex(float* flat_data, int R, int C, float** row_table)`:
    //       - Sets row_table[r] = &flat_data[r * C].
    //       Implement `gemvRowIndex(const float** rows, const float* x, float* y, int R, int C)`:
    //       - Computes y[r] = sum_{c=0}^{C-1} rows[r][c] * x[c].
    // -------------------------------------------------------------------------
    {
        auto buildRowIndex = [](float* flat_data, int R, int C, float** row_table) {
            // --- YOUR CODE STARTS HERE ---
            for (int r = 0; r < R; ++r) {
                row_table[r] = flat_data + (static_cast<size_t>(r) * C);
            }
            // --- YOUR CODE ENDS HERE ---
        };

        auto gemvRowIndex = [](const float** rows, const float* x, float* y, int R, int C) {
            // --- YOUR CODE STARTS HERE ---
            for (int r = 0; r < R; ++r) {
                float acc = 0.0f;
                const float* row = rows[r];
                for (int c = 0; c < C; ++c) {
                    acc += row[c] * x[c];
                }
                y[r] = acc;
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const int R = 16, C = 32;
        std::vector<float> flat(R * C);
        for (int i = 0; i < R * C; ++i) flat[i] = static_cast<float>(i % 7);

        std::vector<float*> row_ptrs(R);
        buildRowIndex(flat.data(), R, C, row_ptrs.data());

        std::vector<float> x(C, 1.0f);
        std::vector<float> y(R, 0.0f);

        gemvRowIndex(const_cast<const float**>(row_ptrs.data()), x.data(), y.data(), R, C);

        bool ok = true;
        for (int r = 0; r < R; ++r) {
            float expected = 0.0f;
            for (int c = 0; c < C; ++c) expected += flat[r * C + c];
            if (std::abs(y[r] - expected) > 1e-4f) {
                ok = false;
                break;
            }
        }

        reportStatus("Problem 2: Row-Index float** Table GEMV", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Type-Erased Generic Tensor Buffer (`void*`) with Runtime DType Dispatch
    //
    // Context: Modern tensor runtimes (PyTorch, ONNX Runtime) store tensor storage
    //          as a generic `void* raw_data` accompanied by an enumeration `DType`.
    //          Low-level operators cast `void*` to the appropriate concrete pointer.
    //
    // Task: Implement `scaleGenericTensor(void* data, size_t count, DType dtype, float scale)`:
    //       - If DType::FP32:
    //           cast to `float*`, for each element: val *= scale.
    //       - If DType::INT32:
    //           cast to `int32_t*`, for each element: val = round(val * scale).
    //       - If DType::INT8:
    //           cast to `int8_t*`, for each element: val = clamp(round(val * scale), -128, 127).
    // -------------------------------------------------------------------------
    {
        auto scaleGenericTensor = [](void* data, size_t count, DType dtype, float scale) {
            // --- YOUR CODE STARTS HERE ---
            if (!data || count == 0) return;
            switch (dtype) {
                case DType::FP32: {
                    float* ptr = reinterpret_cast<float*>(data);
                    for (size_t i = 0; i < count; ++i) ptr[i] *= scale;
                    break;
                }
                case DType::INT32: {
                    int32_t* ptr = reinterpret_cast<int32_t*>(data);
                    for (size_t i = 0; i < count; ++i) {
                        ptr[i] = static_cast<int32_t>(std::round(ptr[i] * scale));
                    }
                    break;
                }
                case DType::INT8: {
                    int8_t* ptr = reinterpret_cast<int8_t*>(data);
                    for (size_t i = 0; i < count; ++i) {
                        float v = std::round(ptr[i] * scale);
                        ptr[i] = static_cast<int8_t>(std::clamp(v, -128.0f, 127.0f));
                    }
                    break;
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t COUNT = 16;
        std::vector<float> fp32_buf(COUNT, 2.5f);
        std::vector<int32_t> int32_buf(COUNT, 10);
        std::vector<int8_t> int8_buf(COUNT, 50);

        scaleGenericTensor(fp32_buf.data(), COUNT, DType::FP32, 2.0f);
        scaleGenericTensor(int32_buf.data(), COUNT, DType::INT32, 3.0f);
        scaleGenericTensor(int8_buf.data(), COUNT, DType::INT8, 4.0f); // 50 * 4 = 200 -> clamps to 127

        bool ok = true;
        for (size_t i = 0; i < COUNT; ++i) {
            if (fp32_buf[i] != 5.0f) ok = false;
            if (int32_buf[i] != 30) ok = false;
            if (int8_buf[i] != 127) ok = false;
        }

        reportStatus("Problem 3: Type-Erased void* Runtime DType Dispatch", ok);
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
