// ==============================================================================
// Module 2.1: Lambdas & Function Objects — Champion Workbook
// ==============================================================================
// In this champion workbook, you will master advanced GPU device closures used
// in production LLM training engines (PyTorch ATen / Cutlass style):
// 1. Dual-Input SwiGLU Activation via Ternary/Binary Device Lambda
// 2. High-Performance Function Composition (compose(f, g) -> f(g(x)))
// 3. Higher-Order Stride-Aware 2D Tensor Transform Kernel
// 4. Vectorized 128-bit float4 Device Lambda for Max Memory Bandwidth
// 5. Branchless HardSwish Activation via Device Lambda
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
// Exercise 1: Dual-Input SwiGLU Activation via Device Lambda
// SwiGLU(gate, up) = (gate / (1 + exp(-gate))) * up
// Launch a higher-order binary kernel passing a device lambda.
// ==============================================================================
template <typename BinaryOp>
__global__ void binary_transform_kernel(const float* a, const float* b, float* out,
                                        int N, BinaryOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(a[idx], b[idx]);
    }
}

void launch_swiglu_lambda(const float* d_gate, const float* d_up, float* d_out, int N) {
    // TODO:
    // 1. Define device lambda:
    //    auto swiglu = [] __device__ (float g, float u) {
    //        return (g / (1.0f + expf(-g))) * u;
    //    };
    // 2. Launch binary_transform_kernel<<<(N + 255) / 256, 256>>>(d_gate, d_up, d_out, N, swiglu);
}

// ==============================================================================
// Exercise 2: Function Composition: compose(f, g)
// Implement a host generic composition function `compose(f, g)` that returns
// a new lambda representing h(x) = f(g(x)).
// ==============================================================================
template <typename F, typename G>
auto compose(F f, G g) {
    // TODO:
    // Return a lambda [=](float x) { return f(g(x)); };
    return [=](float x) { return 0.0f; };
}

// ==============================================================================
// Exercise 3: Higher-Order Stride-Aware 2D Tensor Transform
// Given a 2D tensor [rows, cols] with physical row stride in elements.
// Element at (r, c) is located at data[r * stride + c].
// Apply an arbitrary device lambda `op` to each valid element.
// ==============================================================================
template <typename Op>
__global__ void strided_2d_kernel(float* data, int rows, int cols, int stride, Op op) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO:
    // If r < rows && c < cols:
    // data[r * stride + c] = op(data[r * stride + c]);
}

void launch_strided_2d_scale_bias(float* d_data, int rows, int cols, int stride,
                                  float scale, float bias) {
    // TODO:
    // 1. Define device lambda: [scale, bias] __device__ (float x) { return scale * x + bias; };
    // 2. Launch strided_2d_kernel with dim3 block(16, 16) and dim3 grid((cols + 15)/16, (rows + 15)/16).
}

// ==============================================================================
// Exercise 4: Vectorized 128-bit float4 Device Lambda
// Vectorized memory access is essential for memory-bound kernels.
// Implement a kernel operating on float4 elements:
// N_vec = N / 4.
// Pass a device lambda taking float4 and returning float4.
// ==============================================================================
template <typename VecOp>
__global__ void vectorized_transform_kernel(const float4* in, float4* out, int N_vec, VecOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_vec) {
        out[idx] = op(in[idx]);
    }
}

void launch_vectorized_abs(const float4* d_in, float4* d_out, int N_vec) {
    // TODO:
    // 1. Define device lambda taking float4:
    //    auto vec_abs = [] __device__ (float4 v) -> float4 {
    //        return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
    //    };
    // 2. Launch vectorized_transform_kernel<<<(N_vec + 255) / 256, 256>>>(d_in, d_out, N_vec, vec_abs);
}

// ==============================================================================
// Exercise 5: Branchless HardSwish Activation via Device Lambda
// HardSwish(x) = x * clamp(x + 3, 0, 6) / 6
// Must be branchless using fminf and fmaxf intrinsics.
// ==============================================================================
template <typename Op>
__global__ void unary_device_kernel(const float* in, float* out, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = op(in[idx]);
    }
}

void launch_hardswish(const float* d_in, float* d_out, int N) {
    // TODO:
    // 1. Define device lambda:
    //    auto hardswish = [] __device__ (float x) {
    //        return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
    //    };
    // 2. Launch unary_device_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N, hardswish);
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: SwiGLU Activation Lambda
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_gate(N), h_up(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            h_gate[i] = 0.1f * (i - 16);
            h_up[i]   = 0.2f * (i - 8);
        }

        float *d_gate, *d_up, *d_out;
        CHECK_CUDA(cudaMalloc(&d_gate, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_up, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_up, h_up.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_swiglu_lambda(d_gate, d_up, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float g = h_gate[i];
            float u = h_up[i];
            float exp = (g / (1.0f + std::exp(-g))) * u;
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 1: SwiGLU Activation Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: SwiGLU Activation Lambda] FAILED" << std::endl;
        }

        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 2: Function Composition: compose(f, g)
    // --------------------------------------------------------------------------
    {
        auto add_two = [](float x) { return x + 2.0f; };
        auto square  = [](float x) { return x * x; };
        auto composed = compose(square, add_two); // (x + 2)^2

        bool ok = true;
        for (int i = 0; i < 10; ++i) {
            float x = static_cast<float>(i);
            float exp = (x + 2.0f) * (x + 2.0f);
            if (std::fabs(composed(x) - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 2: Function Composition] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Function Composition] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: Strided 2D Tensor Transform
    // --------------------------------------------------------------------------
    {
        int rows = 8, cols = 8, stride = 12;
        std::vector<float> h_data(rows * stride, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                h_data[r * stride + c] = static_cast<float>(r * cols + c);
            }
        }
        float scale = 2.0f, bias = -5.0f;

        float* d_data;
        CHECK_CUDA(cudaMalloc(&d_data, rows * stride * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), rows * stride * sizeof(float), cudaMemcpyHostToDevice));

        launch_strided_2d_scale_bias(d_data, rows, cols, stride, scale, bias);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, rows * stride * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float initial = static_cast<float>(r * cols + c);
                float exp = scale * initial + bias;
                if (std::fabs(h_data[r * stride + c] - exp) > 1e-4f) ok = false;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Strided 2D Transform] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Strided 2D Transform] FAILED" << std::endl;
        }

        cudaFree(d_data);
    }

    // --------------------------------------------------------------------------
    // Test 4: Vectorized float4 Lambda
    // --------------------------------------------------------------------------
    {
        int N_vec = 16;
        std::vector<float4> h_in(N_vec), h_out(N_vec);
        for (int i = 0; i < N_vec; ++i) {
            h_in[i] = make_float4(-static_cast<float>(i * 4 + 0),
                                   static_cast<float>(i * 4 + 1),
                                  -static_cast<float>(i * 4 + 2),
                                   static_cast<float>(i * 4 + 3));
        }

        float4 *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N_vec * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_out, N_vec * sizeof(float4)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N_vec * sizeof(float4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N_vec * sizeof(float4)));

        launch_vectorized_abs(d_in, d_out, N_vec);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N_vec * sizeof(float4), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N_vec; ++i) {
            if (std::fabs(h_out[i].x - std::fabs(h_in[i].x)) > 1e-4f ||
                std::fabs(h_out[i].y - std::fabs(h_in[i].y)) > 1e-4f ||
                std::fabs(h_out[i].z - std::fabs(h_in[i].z)) > 1e-4f ||
                std::fabs(h_out[i].w - std::fabs(h_in[i].w)) > 1e-4f) {
                ok = false;
            }
        }
        if (ok) {
            std::cout << "[Test 4: Vectorized float4 Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Vectorized float4 Lambda] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 5: Branchless HardSwish Lambda
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_in(N), h_out(N, 0.0f);
        for (int i = 0; i < N; ++i) h_in[i] = 0.3f * (i - 16);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        launch_hardswish(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float x = h_in[i];
            float clamped = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
            float exp = x * clamped / 6.0f;
            if (std::fabs(h_out[i] - exp) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 5: Branchless HardSwish Lambda] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Branchless HardSwish Lambda] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
