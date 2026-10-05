// ==============================================================================
// Module 2.1: Lambdas & Function Objects — Beginner Workbook
// ==============================================================================
// In this workbook, you will learn how modern C++ lambdas bridge host logic
// and GPU execution:
// 1. Host Lambda for Affine Transform: y = alpha * x + beta
// 2. Mutable Lambdas with State Capture
// 3. Templated Higher-Order Vector Transform Function
// 4. Basic CUDA Device Lambda (__device__ with --extended-lambda)
// 5. Generic Higher-Order GPU Transform Kernel with Captured Parameter
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
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
// Exercise 1: Host Lambda for Affine Transform
// Implement a host C++ lambda that captures float alpha and float beta by value,
// and computes: y = alpha * x + beta.
// ==============================================================================
void run_affine_transform(const std::vector<float>& in, std::vector<float>& out,
                          float alpha, float beta) {
    // TODO:
    // 1. Define auto affine_op = [alpha, beta](float x) -> float { ... };
    // 2. Loop through each element i and assign out[i] = affine_op(in[i]);
}

// ==============================================================================
// Exercise 2: Mutable Lambdas with State Capture
// Value captures are immutable (const) by default inside operator().
// Create a lambda using the 'mutable' keyword that captures an integer counter = 0,
// and on each call returns (val + counter++).
// ==============================================================================
void run_counter_offset(const std::vector<float>& in, std::vector<float>& out) {
    // TODO:
    // 1. Define auto add_increasing = [count = 0](float x) mutable -> float {
    //        return x + (count++);
    //    };
    // 2. Apply to out[i] for all i.
}

// ==============================================================================
// Exercise 3: Templated Higher-Order Host Transform
// Implement a generic template function that applies any callable operator:
// template <typename Op>
// void host_transform(const std::vector<float>& in, std::vector<float>& out, Op op)
// ==============================================================================
template <typename Op>
void host_transform(const std::vector<float>& in, std::vector<float>& out, Op op) {
    // TODO:
    // For each element in `in`, apply `op` and store in `out`.
}

// ==============================================================================
// Exercise 4: Basic CUDA Device Lambda
// With nvcc --extended-lambda, define a __device__ lambda that computes:
// ReLU(x) = fmaxf(0.0f, x).
// Pass it into simple_relu_kernel.
// ==============================================================================
template <typename Op>
__global__ void simple_device_op_kernel(const float* in, float* out, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(in[idx]);
    }
}

void launch_device_relu(const float* d_in, float* d_out, int N) {
    // TODO:
    // 1. Define device lambda: auto relu = [] __device__ (float x) { return fmaxf(0.0f, x); };
    // 2. Launch simple_device_op_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N, relu);
}

// ==============================================================================
// Exercise 5: Generic Higher-Order GPU Transform with Captured Parameters
// Launch a GPU kernel with a device lambda that captures float slope by value:
// LeakyReLU(x) = (x >= 0.0f) ? x : (x * slope)
// ==============================================================================
void launch_leaky_relu_captured(const float* d_in, float* d_out, int N, float slope) {
    // TODO:
    // 1. Define device lambda capturing slope by value:
    //    auto leaky = [slope] __device__ (float x) {
    //        return (x >= 0.0f) ? x : (x * slope);
    //    };
    // 2. Launch simple_device_op_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N, leaky);
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Host Affine Lambda
    // --------------------------------------------------------------------------
    {
        int N = 16;
        std::vector<float> in(N), out(N, 0.0f);
        for (int i = 0; i < N; ++i) in[i] = static_cast<float>(i);
        float alpha = 2.5f, beta = -1.0f;

        run_affine_transform(in, out, alpha, beta);

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = alpha * in[i] + beta;
            if (std::fabs(out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 1: Host Affine Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Host Affine Lambda] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: Mutable Lambda with State Capture
    // --------------------------------------------------------------------------
    {
        int N = 8;
        std::vector<float> in(N, 10.0f), out(N, 0.0f);
        run_counter_offset(in, out);

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = 10.0f + i;
            if (std::fabs(out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 2: Mutable Lambda with State] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Mutable Lambda with State] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: Templated Higher-Order Host Transform
    // --------------------------------------------------------------------------
    {
        int N = 16;
        std::vector<float> in(N), out(N, 0.0f);
        for (int i = 0; i < N; ++i) in[i] = static_cast<float>(i - 8);

        auto square_op = [](float x) { return x * x; };
        host_transform(in, out, square_op);

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = in[i] * in[i];
            if (std::fabs(out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 3: Templated Host Transform] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Templated Host Transform] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: Device Lambda (ReLU)
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_in(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i - 16);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_device_relu(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 4: Device Lambda (ReLU)] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Device Lambda (ReLU)] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 5: Captured Parameters Device Lambda (LeakyReLU)
    // --------------------------------------------------------------------------
    {
        int N = 32;
        float slope = 0.01f;
        std::vector<float> h_in(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i - 16);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_leaky_relu_captured(d_in, d_out, N, slope);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = (h_in[i] >= 0.0f) ? h_in[i] : (h_in[i] * slope);
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 5: Captured Device Lambda (LeakyReLU)] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Captured Device Lambda (LeakyReLU)] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
