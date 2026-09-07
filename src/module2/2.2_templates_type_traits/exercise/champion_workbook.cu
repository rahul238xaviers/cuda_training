// ==============================================================================
// Module 2.2: Templates & Type Traits — Champion Workbook
// ==============================================================================
// In this champion workbook, you will master production-grade template metaprogramming
// used in deep learning systems (e.g. PyTorch ATen, FlashAttention, and CUTLASS):
// 1. Unified Multi-Precision Traits (float, half, __nv_bfloat16)
// 2. Generic Templated RMSNorm Kernel across Floating-Point Types
// 3. Variadic Template Kernel Launcher with Perfect Forwarding
// 4. Compile-Time TileConfig Struct with Static Assertions
// 5. Type-Dispatched Vectorized Memory Transfer Kernel
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <utility>
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
// Exercise 1: Unified Multi-Precision Traits
// Complete PrecisionTraits<T> for float, half, and __nv_bfloat16.
// Must define:
//   using AccumT = float;
//   static constexpr bool is_16bit = ...;
//   __host__ __device__ static float to_float(T x);
//   __host__ __device__ static T from_float(float x);
// ==============================================================================
template <typename T>
struct PrecisionTraits;

template <>
struct PrecisionTraits<float> {
    using AccumT = float;
    static constexpr bool is_16bit = false;
    __host__ __device__ static float to_float(float x) { return x; }
    __host__ __device__ static float from_float(float x) { return x; }
};

template <>
struct PrecisionTraits<half> {
    using AccumT = float;
    static constexpr bool is_16bit = true;
    // TODO: Implement to_float and from_float for half
    __host__ __device__ static float to_float(half x) { return 0.0f; }
    __host__ __device__ static half from_float(float x) { return __float2half(0.0f); }
};

template <>
struct PrecisionTraits<__nv_bfloat16> {
    using AccumT = float;
    static constexpr bool is_16bit = true;
    // TODO: Implement to_float and from_float for __nv_bfloat16
    __host__ __device__ static float to_float(__nv_bfloat16 x) { return 0.0f; }
    __host__ __device__ static __nv_bfloat16 from_float(float x) { return __float2bfloat16(0.0f); }
};

// ==============================================================================
// Exercise 2: Generic Templated RMSNorm Kernel
// For single token x [D], weight [D], out [D]:
// 1. sum_sq = sum_{d=0}^{D-1} to_float(x[d])^2
// 2. inv_rms = rsqrtf(sum_sq / D + eps)
// 3. out[d] = from_float(to_float(x[d]) * inv_rms * to_float(weight[d]))
// Use PrecisionTraits<T>.
// ==============================================================================
template <typename T>
__global__ void generic_rmsnorm_kernel(const T* x, const T* weight, T* out, int D, float eps) {
    // Single block handles 1 row of size D. D <= 256.
    __shared__ float s_sq[256];
    int tid = threadIdx.x;

    // TODO:
    // 1. Load x[tid] converted to float, compute square, store in s_sq[tid]
    // 2. Perform block reduction over s_sq
    // 3. __shared__ float inv_rms; if (tid == 0) compute inv_rms = rsqrtf(s_sq[0] / D + eps);
    // 4. __syncthreads();
    // 5. out[tid] = from_float(to_float(x[tid]) * inv_rms * to_float(weight[tid]));
}

// ==============================================================================
// Exercise 3: Variadic Template Kernel Launcher with Perfect Forwarding
// Implement launch_1d_kernel that takes KernelFunc, N, block_size, and args...
// Computes grid size = (N + block_size - 1) / block_size and launches kernel.
// ==============================================================================
template <typename KernelFunc, typename... Args>
void launch_1d_kernel(KernelFunc k, int N, int block_size, Args&&... args) {
    // TODO:
    // int grid_size = (N + block_size - 1) / block_size;
    // k<<<grid_size, block_size>>>(std::forward<Args>(args)...);
}

// Helper kernel for Exercise 3
__global__ void scale_vector_kernel(float* data, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= factor;
}

// ==============================================================================
// Exercise 4: Compile-Time TileConfig Struct with Static Assertions
// Modern GEMM libraries configure tile dimensions as compile-time constants.
// TileConfig<BM, BN, BK> must verify:
// static_assert(BM % 16 == 0, "BM must be multiple of 16");
// static_assert(BN % 16 == 0, "BN must be multiple of 16");
// static_assert(BK % 8 == 0,  "BK must be multiple of 8");
// Total elements per tile = BM * BN.
// ==============================================================================
template <int BM, int BN, int BK>
struct TileConfig {
    static_assert(BM % 16 == 0, "BM must be multiple of 16");
    static_assert(BN % 16 == 0, "BN must be multiple of 16");
    static_assert(BK % 8 == 0,  "BK must be multiple of 8");

    // TODO:
    // Return total output elements BM * BN
    static constexpr int total_elements() {
        return 0;
    }
};

// ==============================================================================
// Exercise 5: Type-Dispatched Vectorized Memory Transfer
// If sizeof(T) == 4 (float): interpret as float4 (load 4 elements at a time).
// If sizeof(T) == 2 (half/bf16): interpret as uint32_t (load 2 elements at a time).
// ==============================================================================
template <typename T>
__global__ void type_dispatched_copy_kernel(const T* in, T* out, int N) {
    // TODO:
    // Dispatched copy implementation
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: PrecisionTraits
    // --------------------------------------------------------------------------
    {
        half h = PrecisionTraits<half>::from_float(2.25f);
        float h_f = PrecisionTraits<half>::to_float(h);

        __nv_bfloat16 bf = PrecisionTraits<__nv_bfloat16>::from_float(4.5f);
        float bf_f = PrecisionTraits<__nv_bfloat16>::to_float(bf);

        if (std::fabs(h_f - 2.25f) < 1e-3f && std::fabs(bf_f - 4.5f) < 1e-2f &&
            PrecisionTraits<half>::is_16bit && PrecisionTraits<__nv_bfloat16>::is_16bit) {
            std::cout << "[Test 1: PrecisionTraits] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: PrecisionTraits] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: Generic RMSNorm Kernel (Float & Half)
    // --------------------------------------------------------------------------
    {
        int D = 32;
        float eps = 1e-5f;
        std::vector<float> h_x(D), h_w(D), h_out(D, 0.0f);
        for (int i = 0; i < D; ++i) {
            h_x[i] = 0.5f * (i + 1);
            h_w[i] = 1.0f;
        }

        float sum_sq = 0.0f;
        for (int i = 0; i < D; ++i) sum_sq += h_x[i] * h_x[i];
        float inv_rms = 1.0f / std::sqrt(sum_sq / D + eps);
        std::vector<float> h_ref(D);
        for (int i = 0; i < D; ++i) h_ref[i] = h_x[i] * inv_rms * h_w[i];

        float *d_x, *d_w, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w, D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, D * sizeof(float)));

        generic_rmsnorm_kernel<float><<<1, D>>>(d_x, d_w, d_out, D, eps);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, D * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < D; ++i) {
            if (std::fabs(h_out[i] - h_ref[i]) > 1e-3f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 2: Generic RMSNorm Kernel] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Generic RMSNorm Kernel] FAILED" << std::endl;
        }

        cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 3: Variadic Template Launcher
    // --------------------------------------------------------------------------
    {
        int N = 64;
        std::vector<float> h_data(N, 10.0f), h_out(N, 0.0f);
        float* d_data;
        CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        launch_1d_kernel(scale_vector_kernel, N, 32, d_data, 2.5f, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_out[i] - 25.0f) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 3: Variadic Template Launcher] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Variadic Template Launcher] FAILED" << std::endl;
        }

        cudaFree(d_data);
    }

    // --------------------------------------------------------------------------
    // Test 4: TileConfig Static Assertions & Elements
    // --------------------------------------------------------------------------
    {
        constexpr int elems = TileConfig<64, 64, 16>::total_elements();
        if (elems == 4096) {
            std::cout << "[Test 4: TileConfig Traits] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: TileConfig Traits] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 5: Type-Dispatched Copy
    // --------------------------------------------------------------------------
    {
        int N = 32;
        std::vector<float> h_in(N, 7.5f), h_out(N, 0.0f);
        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

        type_dispatched_copy_kernel<float><<<(N + 255) / 256, 256>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_out[i] - 7.5f) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 5: Type-Dispatched Copy] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Type-Dispatched Copy] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
