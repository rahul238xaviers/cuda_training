// ==============================================================================
// Module 2.2: Templates & Type Traits — Beginner Workbook
// ==============================================================================
// In this workbook, you will learn foundational generic C++ programming and
// type trait idioms essential for writing multi-precision CUDA kernels:
// 1. Function Template for Host Vector Elementwise Addition
// 2. Compile-Time Dispatch with `if constexpr`
// 3. SFINAE Function Overload with `std::enable_if_t`
// 4. Generic Templated CUDA Kernel (float & int instantiations)
// 5. Custom Precision Traits Struct for Widened Accumulation
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ==============================================================================
// Exercise 1: Function Template for Host Vector Elementwise Addition
// Implement a generic host function template: c[i] = a[i] + b[i]
// ==============================================================================
template <typename T>
void generic_host_add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c) {
    // TODO:
    // For each element i, compute c[i] = a[i] + b[i];
}

// ==============================================================================
// Exercise 2: Compile-Time Dispatch with `if constexpr`
// If T is floating point: return val * 0.5f.
// If T is integral: return val * 2.
// ==============================================================================
template <typename T>
T scale_by_type(T val) {
    // TODO:
    // Use if constexpr (std::is_floating_point_v<T>) ...
    return val;
}

// ==============================================================================
// Exercise 3: SFINAE Function Overload with std::enable_if_t
// Define a function template compute_reciprocal<T>(T val) enabled ONLY when
// std::is_floating_point_v<T> is true. Returns 1.0 / val.
// ==============================================================================
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
T compute_reciprocal(T val) {
    // TODO:
    // Return 1.0 / val;
    return val;
}

// ==============================================================================
// Exercise 4: Generic Templated CUDA Kernel
// Write a generic GPU kernel c[idx] = a[idx] + b[idx].
// Launch it for float and int types.
// ==============================================================================
template <typename T>
__global__ void generic_add_kernel(const T* a, const T* b, T* c, int N) {
    // TODO:
    // Calculate global index and perform elementwise add if in bounds.
}

template <typename T>
void launch_generic_add(const T* d_a, const T* d_b, T* d_c, int N) {
    // TODO:
    // Launch generic_add_kernel<T><<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
}

// ==============================================================================
// Exercise 5: Custom Precision Traits Struct for Widened Accumulation
// In ML, 16-bit or 32-bit types are accumulated into wider types to avoid overflow.
// Define a traits struct:
//   AccumulatorTraits<int>::Type -> long long
//   AccumulatorTraits<float>::Type -> double
// ==============================================================================
template <typename T>
struct AccumulatorTraits {
    using Type = T;
};

template <>
struct AccumulatorTraits<int> {
    using Type = long long;
};

template <>
struct AccumulatorTraits<float> {
    using Type = double;
};

template <typename T>
typename AccumulatorTraits<T>::Type accumulate_vector(const std::vector<T>& vec) {
    using AccT = typename AccumulatorTraits<T>::Type;
    AccT total = 0;
    // TODO:
    // Accumulate all elements into total and return.
    return total;
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Generic Host Add
    // --------------------------------------------------------------------------
    {
        std::vector<float> a = {1.5f, 2.5f, 3.5f};
        std::vector<float> b = {0.5f, 1.5f, 2.5f};
        std::vector<float> c(3, 0.0f);
        generic_host_add(a, b, c);

        if (std::fabs(c[0] - 2.0f) < 1e-4f &&
            std::fabs(c[1] - 4.0f) < 1e-4f &&
            std::fabs(c[2] - 6.0f) < 1e-4f) {
            std::cout << "[Test 1: Generic Host Add] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Generic Host Add] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: `if constexpr` Dispatch
    // --------------------------------------------------------------------------
    {
        float f_res = scale_by_type(10.0f); // 10.0 * 0.5 = 5.0
        int i_res = scale_by_type(10);       // 10 * 2 = 20

        if (std::fabs(f_res - 5.0f) < 1e-4f && i_res == 20) {
            std::cout << "[Test 2: if constexpr Dispatch] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: if constexpr Dispatch] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: SFINAE Reciprocal
    // --------------------------------------------------------------------------
    {
        float res = compute_reciprocal(4.0f);
        if (std::fabs(res - 0.25f) < 1e-4f) {
            std::cout << "[Test 3: SFINAE Reciprocal] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: SFINAE Reciprocal] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: Generic CUDA Kernel
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_fa(N, 2.0f), h_fb(N, 3.0f), h_fc(N, 0.0f);
        float *d_fa, *d_fb, *d_fc;
        CHECK_CUDA(cudaMalloc(&d_fa, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_fb, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_fc, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_fa, h_fa.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_fb, h_fb.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_fc, 0, N * sizeof(float)));

        launch_generic_add(d_fa, d_fb, d_fc, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_fc.data(), d_fc, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_fc[i] - 5.0f) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 4: Generic CUDA Kernel] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Generic CUDA Kernel] FAILED" << std::endl;
        }

        cudaFree(d_fa); cudaFree(d_fb); cudaFree(d_fc);
    }

    // --------------------------------------------------------------------------
    // Test 5: Widened Accumulator Traits
    // --------------------------------------------------------------------------
    {
        std::vector<int> big_ints = {1000000000, 1500000000}; // sum = 2.5 billion (overflows 32-bit signed int!)
        long long sum = accumulate_vector(big_ints);

        if (sum == 2500000000LL) {
            std::cout << "[Test 5: Widened Accumulator Traits] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Widened Accumulator Traits] FAILED" << std::endl;
        }
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
