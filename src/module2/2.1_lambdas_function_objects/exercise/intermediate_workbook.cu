// ==============================================================================
// Module 2.1: Lambdas & Function Objects — Intermediate Workbook
// ==============================================================================
// In this workbook, you will implement production-grade activation and elementwise
// fusion patterns using C++ device closures and higher-order GPU kernels:
// 1. Host Binary Lambda with Captured Scalar Scaling
// 2. GPU SiLU / Swish Activation via Device Lambda: x / (1 + exp(-x))
// 3. GPU GELU (Gaussian Error Linear Unit) via Device Lambda
// 4. In-Place Templated Transform Kernel
// 5. Fused Residual Addition + SiLU Activation Kernel via Binary Device Lambda
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
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
// Exercise 1: Host Binary Lambda with Captured Scaling
// Implement a host lambda taking two inputs: a and b, computing:
// result = scale * (a + b) + offset
// ==============================================================================
void run_scaled_add(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& out, float scale, float offset) {
    // TODO:
    // 1. Define auto op = [scale, offset](float x, float y) { ... };
    // 2. Assign out[i] = op(a[i], b[i]);
}

// ==============================================================================
// Exercise 2: GPU SiLU / Swish Activation via Device Lambda
// SiLU(x) = x / (1.0f + expf(-x))
// Launch higher-order unary kernel with a __device__ lambda.
// ==============================================================================
template <typename Op>
__global__ void unary_op_kernel(const float* in, float* out, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(in[idx]);
    }
}

void launch_gpu_silu(const float* d_in, float* d_out, int N) {
    // TODO:
    // 1. Define device lambda: auto silu = [] __device__ (float x) { return x / (1.0f + expf(-x)); };
    // 2. Launch unary_op_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N, silu);
}

// ==============================================================================
// Exercise 3: GPU GELU Activation via Device Lambda
// GELU(x) = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)))
// Launch higher-order unary kernel with a __device__ lambda.
// ==============================================================================
void launch_gpu_gelu(const float* d_in, float* d_out, int N) {
    // TODO:
    // 1. Define device lambda for GELU approximation.
    // 2. Launch unary_op_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N, gelu);
}

// ==============================================================================
// Exercise 4: In-Place Templated Transform Kernel
// Modifies buffer in-place: data[idx] = op(data[idx]).
// Apply a device lambda that clamps values to range [min_val, max_val].
// ==============================================================================
template <typename Op>
__global__ void in_place_transform_kernel(float* data, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = op(data[idx]);
    }
}

void launch_gpu_clamp(float* d_data, int N, float min_val, float max_val) {
    // TODO:
    // 1. Define device lambda capturing min_val and max_val by value:
    //    auto clamp_op = [min_val, max_val] __device__ (float x) {
    //        return fminf(fmaxf(x, min_val), max_val);
    //    };
    // 2. Launch in_place_transform_kernel<<<(N + 255) / 256, 256>>>(d_data, N, clamp_op);
}

// ==============================================================================
// Exercise 5: Fused Residual Addition + SiLU Activation
// In transformer layers, we frequently compute: out[i] = SiLU(x[i] + residual[i]).
// Implement a binary templated kernel and pass a device lambda performing this fusion.
// ==============================================================================
template <typename BinaryOp>
__global__ void binary_op_kernel(const float* a, const float* b, float* out, int N, BinaryOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(a[idx], b[idx]);
    }
}

void launch_fused_residual_silu(const float* d_x, const float* d_res, float* d_out, int N) {
    // TODO:
    // 1. Define binary device lambda:
    //    auto fused_op = [] __device__ (float x, float res) {
    //        float s = x + res;
    //        return s / (1.0f + expf(-s));
    //    };
    // 2. Launch binary_op_kernel<<<(N + 255) / 256, 256>>>(d_x, d_res, d_out, N, fused_op);
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Host Scaled Add Lambda
    // --------------------------------------------------------------------------
    {
        int N = 16;
        std::vector<float> a(N), b(N), out(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            a[i] = 1.0f * i;
            b[i] = 2.0f * i;
        }
        float scale = 0.5f, offset = 3.0f;
        run_scaled_add(a, b, out, scale, offset);

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = scale * (a[i] + b[i]) + offset;
            if (std::fabs(out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 1: Host Scaled Add Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Host Scaled Add Lambda] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: GPU SiLU via Device Lambda
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_in(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_in[i] = 0.2f * (i - 16);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_gpu_silu(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = h_in[i] / (1.0f + std::exp(-h_in[i]));
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 2: GPU SiLU Device Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: GPU SiLU Device Lambda] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 3: GPU GELU via Device Lambda
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_in(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_in[i] = 0.15f * (i - 16);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_gpu_gelu(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float x = h_in[i];
            float exp = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 3: GPU GELU Device Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: GPU GELU Device Lambda] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 4: In-Place Clamp Transform
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_data(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(i - 16);
        float min_val = -5.0f, max_val = 5.0f;

        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        launch_gpu_clamp(d_data, N, min_val, max_val);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float exp = std::min(std::max(h_data[i], min_val), max_val);
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 4: In-Place GPU Clamp] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: In-Place GPU Clamp] FAILED" << std::endl;
        }

        cudaFree(d_data);
    }

    // --------------------------------------------------------------------------
    // Test 5: Fused Residual + SiLU
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_x(N), h_res(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            h_x[i]   = 0.1f * (i - 10);
            h_res[i] = 0.05f * i;
        }

        float *d_x, *d_res, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_res, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_res, h_res.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_fused_residual_silu(d_x, d_res, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float sum = h_x[i] + h_res[i];
            float exp = sum / (1.0f + std::exp(-sum));
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 5: Fused Residual + SiLU] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Fused Residual + SiLU] FAILED" << std::endl;
        }

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
